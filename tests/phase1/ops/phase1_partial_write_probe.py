# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive a real native warm archive, fail before publication and revalidate."""

import argparse
import asyncio
import errno
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
from kitepond.integrations.vllm.cold_prefill_http import TOKEN_ENV, is_control_token
from kitepond.models.glm5next.frontier import digest
from kitepond.ops.glm_cold_probe import _API_KEY_ENV
from phase1_fault_probe import Pair, require
from phase1_fsl2_partial_hook import SCHEMA, archive_counts, device_identity, emit, read
from phase1_mixed_lab import MODEL_LABEL, _nvme_module


async def wait_for_clean(root):
    async with asyncio.timeout(15):
        while any(archive_counts(root).values()):
            await asyncio.sleep(0.2)
    return archive_counts(root)


def validate_failure(boot, proof, states):
    require(
        proof["schema"] == SCHEMA
        and proof["boot"] == boot
        and proof["rank"] == 0
        and proof["errno"] == errno.ENOSPC
        and proof["independent_witness"]["pid"] != proof["pid"]
        and proof["independent_witness"]["writer_pid"] == proof["pid"]
        and proof["independent_witness"]["observed_counts"] == proof["observed_counts"]
        and proof["first"]["name"] != proof["second_temporary"]["name"]
        and proof["first"]["bytes"] > 0
        and proof["second_temporary"]["bytes"] > 0
        and proof["observed_counts"]
        == {
            "payloads": 1,
            "temporary": 1,
            "rank_manifests": 0,
            "aggregates": 0,
            "catalog": 0,
        }
        and "captured" in states
        and "demoting" in states
        and states[-1] == "failed"
        and "committed" not in states,
        "Expected real nonempty native partial output before failed publication",
    )


def output_delta(before, after):
    a = before["metrics"]["components"]["decode"]
    b = after["metrics"]["components"]["decode"]
    key = "generated_tokens_total"
    x, y = a["metrics"].get(key), b["metrics"].get(key)
    require(
        a["epoch"]
        and a["epoch"] == b["epoch"]
        and a["scope"] == b["scope"]
        and type(x) is int
        and type(y) is int
        and y >= x,
        "Native D output counter has a changed or unobserved scope",
    )
    return y - x


async def run(args):
    from kitepond.integrations.vllm.glm_cold_catalog import GlmColdCatalog
    from kitepond.integrations.vllm.glm_cold_runtime import ColdDirectoryScope

    arm = read(args.arm)
    pair_config = read(args.pair)
    root = Path(arm["root"])
    boot = pair_config["authority_boot_id"]
    require(
        arm["schema"] == SCHEMA
        and arm["boot"] == boot
        and root == args.runtime / "kitepond-nvme"
        and device_identity(root) == arm["root_identity"]
        and not any(archive_counts(root).values())
        and not (args.arm.parent / "first.json").exists()
        and not (args.arm.parent / "injected.json").exists(),
        "Expected unused fault arm on this exact private archive root and boot",
    )
    auth = os.environ.get(TOKEN_ENV)
    driver = _nvme_module(args.driver, args.driver_sha256)
    private = os.environ.get(driver.TOKEN_ENV)
    require(
        is_control_token(auth) and is_control_token(private),
        "Missing local control auth",
    )
    key = driver.private_key(Path(os.environ["KITEPOND_GLM_NVME_OPERATOR_KEY_FILE"]))
    catalog_key = driver.private_key(
        Path(os.environ["KITEPOND_GLM_NVME_CATALOG_KEY_FILE"])
    )
    require(
        args.output.parent.resolve() == args.runtime,
        "Keep partial store evidence in this private runtime",
    )
    args.output.mkdir(mode=0o700)
    fields = SimpleNamespace(
        engine=pair_config["authority_engine_id"],
        boot=boot,
        model=MODEL_LABEL,
        prefill_engine=pair_config["mooncake_api"]["p_engine_id"],
        model_sha256=pair_config["checkpoint_identity"]["model_sha256"],
        config_sha256=args.config_sha256,
        prompt=args.prompt,
        suffix=args.suffix,
        artifact=args.output / "repaired-oracle.json",
        session=f"partial-{boot[:12]}",
        generation=1,
        seed=677,
        output_tokens=8,
        timeout=110,
    )
    catalog = GlmColdCatalog(
        root,
        engine_id=fields.engine,
        boot_id=boot,
        model_fingerprint=fields.model_sha256,
        config_fingerprint=fields.config_sha256,
        signing_key=catalog_key,
    )
    failed_scope = ColdDirectoryScope(fields.session, fields.generation)
    async with httpx.AsyncClient(trust_env=False, timeout=None) as client:
        pair = Pair(args, client, auth, os.environ.get(_API_KEY_ENV))
        call = driver.command(client, fields, key, private)
        before = await pair.drain()
        emit(args.output / "before.json", before)
        status = []
        ids = set()

        async def observe(operation, request_id, **kwargs):
            reply = await call(operation, request_id, **kwargs)
            ids.add(request_id)
            if (
                not status
                or status[-1]["operation"] != operation
                or status[-1]["reply"] != reply
            ):
                status.append(
                    {"operation": operation, "request_id": request_id, "reply": reply}
                )
            return reply

        helper = Path(__file__).with_name("phase1_fsl2_partial_hook.py")
        watcher = await asyncio.create_subprocess_exec(
            sys.executable, "-B", str(helper), "--observe", str(args.arm)
        )
        try:
            try:
                await driver.archive(client, fields, observe)
            except RuntimeError:
                pass
            else:
                raise ValueError("Faulted native archive unexpectedly committed")
            require(
                await asyncio.wait_for(watcher.wait(), 5) == 0,
                "Independent partial-write witness did not complete",
            )
        finally:
            if watcher.returncode is None:
                watcher.terminate()
                await watcher.wait()
        emit(args.output / "native-write-state.json", status)
        proof = read(args.arm.parent / "injected.json")
        [request_id] = ids
        validate_failure(boot, proof, [row["reply"]["state"] for row in status])
        failure_done = await pair.drain()
        emit(args.output / "after-failed-store.json", failure_done)
        require(output_delta(before, failure_done) == 16, "D lost native warm output")
        removed = await wait_for_clean(root)
        require(
            catalog.reopen(failed_scope) is None
            and await call("forget", request_id) == {"state": "forgotten"},
            "Failed archive remains reachable or registered",
        )
        emit(
            args.output / "rollback.json",
            {
                "after_failed_fsl2_cleanup": removed,
                "failed_catalog_reopened": False,
                "failed_generation": fields.generation,
                "partial_write": proof,
            },
        )

        fields.generation += 1
        repaired = await driver.archive(client, fields, call)
        serving = await driver.restore(client, fields, call, same_boot=True)
        finished = await pair.drain()
        emit(args.output / "after-positive-recovery.json", finished)
        valid = catalog.reopen(ColdDirectoryScope(fields.session, fields.generation))
        expected = read(fields.artifact)["expected_token_ids"]
        require(
            repaired["status"] == "committed"
            and valid is not None
            and catalog.verify(valid)
            and serving["accepted_native"] is True
            and serving["matches_warm_token_ids"] is True
            and serving["completion_sha256"] == digest(expected)
            and serving["completion_tokens"] == 8
            and output_delta(failure_done, finished) == 24
            and catalog.reopen(failed_scope) is None,
            "New generation did not recover exact native serving",
        )
        for request_id in (
            repaired["source_request_id"],
            serving["restore_request_id"],
        ):
            require(
                await call("forget", request_id) == {"state": "forgotten"},
                "Repaired native operation did not retire",
            )
        outcome = {
            "schema": SCHEMA,
            "case": "nvme_partial_write",
            "scope": "native_overlay_lab",
            "boot": boot,
            "failed_generation": 1,
            "retired_fsl2_parts": removed,
            "new_generation": fields.generation,
            "catalog_sha256": valid.aggregate_sha256,
            "recovery": serving,
            "release_qualification": "not_evaluated",
        }
        emit(args.output / "result.json", outcome)
        return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest",
        "pair",
        "runtime",
        "output",
        "driver",
        "prompt",
        "suffix",
        "arm",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("driver-sha256", "config-sha256"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    args.runtime = args.runtime.resolve(strict=True)
    args.timeout = 110
    os.umask(0o077)
    require(
        is_control_token(args.config_sha256), "Expected sealed model config fingerprint"
    )
    result = asyncio.run(run(args))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
