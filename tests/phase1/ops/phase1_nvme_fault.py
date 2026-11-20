# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare authenticated NVMe clones and verify native missing-part recovery."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import signal
import stat
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import monotonic_ns
from types import SimpleNamespace

import httpx
from kitepond.models.glm5next.frontier import digest
from kitepond.ops.capture import (
    EngineSource,
    _capture_engine,
    _process_start,
)
from kitepond.ops.cli import _read_manifest
from kitepond.ops.glm_cold_probe import _control
from kitepond.ops.wheel import sha256_file
from phase1_mixed_lab import _nvme_module

SCHEMA = "kitepond.glm-native-nvme-fault.v1"
INVENTORY_SCHEMA = "kitepond.glm-native-nvme-inventory.v1"
PORT = 18162
MODEL = "nvidia/GLM-5.3-Flash-NVFP4"
KINDS = ("corrupt", "unreadable")
COUNTER = "generated_tokens_total"


def require(valid, reason):
    if not valid:
        raise ValueError(reason)


def load(path):
    return json.loads(path.read_bytes())


def emit(path, record):
    with path.open("x", encoding="utf-8") as f:
        json.dump(record, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def identity(path):
    value = path.stat()
    return {"device": value.st_dev, "inode": value.st_ino}


def below(root, path):
    rel = Path(path)
    full = root / rel
    require(
        not rel.is_absolute() and ".." not in rel.parts and full.resolve() == full,
        "Archive path must stay beneath the designated private root",
    )
    return full


def exact_file(root, relative, expected):
    path = below(root, relative)
    require(
        path.is_file() and sha256_file(path) == expected,
        "Authenticated archive member changed",
    )
    return path


def copy_verified(root, dest, relative, expected, *, size=None):
    source = exact_file(root, relative, expected)
    target = below(dest, relative)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, 1024 * 1024)
    require(
        not target.samefile(source)
        and target.stat().st_nlink == 1
        and (size is None or target.stat().st_size == size)
        and sha256_file(target) == expected,
        "Clone must own independent exact archive bytes",
    )


def make_catalog(root, key, oracle, boot, model_sha, config_sha):
    from kitepond.integrations.vllm.glm_cold_catalog import GlmColdCatalog

    return GlmColdCatalog(
        root,
        engine_id=oracle["engine"],
        boot_id=boot,
        model_fingerprint=model_sha,
        config_fingerprint=config_sha,
        signing_key=key,
    )


def authenticated(root, key, oracle, boot, model_sha, config_sha):
    catalog = make_catalog(root, key, oracle, boot, model_sha, config_sha)
    return reopen_catalog(catalog, oracle, boot)


def reopen_catalog(catalog, oracle, boot):
    from kitepond.integrations.vllm.glm_cold_runtime import ColdDirectoryScope

    actual = catalog.reopen(
        ColdDirectoryScope(oracle["namespace"], oracle["generation"])
    )
    require(
        actual is not None
        and catalog.verify(actual)
        and actual.source_boot_id == oracle["source_boot"]
        and actual.target_boot_id == boot
        and actual.accepted_prefix_sha256 == oracle["prefix_sha256"],
        "Archive does not have four authenticated rank manifests",
    )
    return actual


def output_count(oracle, expected):
    ids = oracle["expected_token_ids"]
    require(
        type(expected) is int
        and 1 <= expected <= 128
        and type(ids) is list
        and len(ids) == expected
        and all(type(token) is int and token >= 0 for token in ids),
        "Exact restore token count differs from the sealed oracle",
    )
    return expected


def inventory(args):
    """Select one authenticated archive by its committed transcript."""
    from kitepond.backends.fsl2 import digest as structure_digest
    from kitepond.integrations.vllm.glm_nvme_protocol import private_key

    runtime = args.runtime.absolute()
    require(
        runtime.resolve(strict=True) == runtime
        and runtime.name.startswith("pair-run."),
        "Inventory requires one exact completed pair-run directory",
    )
    archive = below(runtime, "kitepond-nvme")
    oracle_path = below(runtime, args.oracle_relative)
    key_path = below(runtime, "private-catalog-key/catalog.key")
    owner_path = below(runtime, "OWNERSHIP.json")
    for path, mode in ((archive, 0o700), (oracle_path, 0o600), (key_path, 0o600)):
        st = path.stat()
        require(
            stat.S_IMODE(st.st_mode) == mode and st.st_uid == os.geteuid(),
            "Inventory requires operator-owned private source files",
        )
    require(archive.is_dir() and oracle_path.is_file(), "Missing archive or oracle")
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    plan_sha = sha256(plan_bytes).hexdigest()
    owner, oracle = load(owner_path), load(oracle_path)
    require(
        owner["plan_sha256"] == plan_sha
        and owner["models_ready"] is True
        and owner["gpu_count"] == 8
        and owner["four_rank_same_physical_backing"] is True
        and len(owner["roles"]["decode"]) == 4
        and oracle["status"] == "committed"
        and owner["engine_boot_id"]
        == oracle["source_boot"]
        == plan["authority_boot_id"]
        and oracle["engine"] == plan["authority_engine_id"],
        "Oracle and owner receipt must name the exact original pair and boot",
    )
    argv = plan["roles"]["decode"]["argv"]
    require(
        oracle["model"] == argv[argv.index("--served-model-name") + 1],
        "Oracle model identifier differs from the original decode role",
    )
    prompt, boundary = oracle["prompt_token_ids"], oracle["boundary"]
    count = output_count(oracle, len(oracle["expected_token_ids"]))
    require(
        type(prompt) is list
        and all(type(token) is int and token >= 0 for token in prompt)
        and type(boundary) is int
        and 0 < boundary < len(prompt)
        and digest(prompt) == oracle["prompt_sha256"]
        and digest(prompt[:boundary]) == oracle["prefix_sha256"],
        "Oracle prompt or cached prefix digest differs",
    )
    catalog = make_catalog(
        archive,
        private_key(key_path),
        oracle,
        oracle["source_boot"],
        plan["checkpoint_identity"]["model_sha256"],
        args.config_sha256,
    )
    receipt = reopen_catalog(catalog, oracle, oracle["source_boot"])
    state = receipt.checkpoint
    require(
        state.layout.version == 2
        and state.layout.target_valid_tokens
        == state.layout.draft_valid_tokens
        == boundary
        and state.layout.request_id == oracle["source_request_id"]
        and oracle["source_terminal_epochs"] == [state.layout.epoch],
        "Archive and oracle do not share the same committed native v2 boundary",
    )
    entry = catalog._target(oracle["namespace"], oracle["generation"])
    aggregate = (
        archive / "aggregates" / (structure_digest(asdict(state.identity)) + ".json")
    )
    return {
        "schema": INVENTORY_SCHEMA,
        "runtime": str(runtime),
        "oracle_relative": str(args.oracle_relative),
        "source_boot": oracle["source_boot"],
        "engine_id": oracle["engine"],
        "source_plan_sha256": plan_sha,
        "source_ownership_sha256": sha256_file(owner_path),
        "expected_output_count": count,
        "layout_version": state.layout.version,
        "namespace": state.namespace,
        "generation": state.generation,
        "aggregate": {
            "path": str(aggregate.relative_to(archive)),
            "sha256": sha256_file(below(archive, aggregate.relative_to(archive))),
        },
        "catalog_entry": {
            "path": str(entry.relative_to(archive)),
            "sha256": sha256_file(below(archive, entry.relative_to(archive))),
        },
    }


def payload_path(rank, part):
    from kitepond.backends.fsl2 import ColdStorageAdapter, _object_key_to_filename

    return (
        Path(f"rank-{rank}")
        / "payloads"
        / _object_key_to_filename(ColdStorageAdapter._key(part["key"]))
    )


def clone(args):
    from kitepond.integrations.vllm.glm_nvme_protocol import private_key

    proof = load(args.persistence)
    source = args.archive.resolve(strict=True)
    oracle = load(args.oracle)
    key = private_key(args.key)
    count = output_count(oracle, proof.get("expected_output_count", 8))
    require(
        identity(source)
        == {"device": proof["root_device"], "inode": proof["root_inode"]}
        and sha256_file(args.oracle) == proof["oracle_sha256"]
        and sha256_file(args.key) == proof["catalog_key_sha256"]
        and oracle["engine"] == proof["engine_id"]
        and oracle["source_boot"] == proof["source_boot"]
        and oracle["status"] == "committed",
        "Archive, key and oracle must come from the same committed deployment",
    )
    aggregate = proof["aggregate"]
    entry = proof["catalog_entry"]
    index = load(exact_file(source, entry["path"], entry["sha256"]))
    model_sha = index["identity"]["model_fingerprint"]
    config_sha = index["layout"]["config_fingerprint"]
    verified = authenticated(
        source, key, oracle, oracle["source_boot"], model_sha, config_sha
    )
    require(
        verified.aggregate_sha256 == index["aggregate_sha256"], "Cold aggregate changed"
    )
    work = args.output.resolve()
    require(
        not work.is_relative_to(source)
        and not work.is_relative_to(Path(proof["source_runtime"])),
        "Clone output must live outside the source run",
    )
    work.mkdir(mode=0o700, exist_ok=False)
    roots = {}
    for kind in KINDS:
        arm = work / kind
        dest = arm / "archive"
        dest.mkdir(mode=0o700, parents=True)
        arm.chmod(0o700)
        for meta in (aggregate, entry):
            copy_verified(source, dest, meta["path"], meta["sha256"])
        parts = []
        for rank, rank_file in enumerate(index["rank_files"]):
            rpath = Path(f"rank-{rank}") / "manifests" / rank_file
            document = load(below(source, rpath))
            copy_verified(source, dest, rpath, sha256_file(below(source, rpath)))
            for position, part in enumerate(document["parts"]):
                path = payload_path(rank, part)
                size = part["spec"]["nbytes"]
                copy_verified(source, dest, path, part["sha256"], size=size)
                parts.append(
                    {
                        "rank": rank,
                        "part": position,
                        "relative_path": str(path),
                        "payload_sha256": part["sha256"],
                        "payload_bytes": size,
                    }
                )
        target = parts[0]
        checked = authenticated(
            dest, key, oracle, oracle["source_boot"], model_sha, config_sha
        )
        require(
            checked.aggregate_sha256 == verified.aggregate_sha256,
            "Cloned aggregate differs",
        )
        copy_verified(dest, arm, target["relative_path"], target["payload_sha256"])
        # This backup is outside the mount visible to the FSL2 engine.
        manifest = {
            "schema": SCHEMA,
            "case": kind,
            "original_root": identity(source),
            "clone_root": identity(dest),
            "catalog_key_sha256": proof["catalog_key_sha256"],
            "oracle_sha256": proof["oracle_sha256"],
            "source_persistence_sha256": sha256_file(args.persistence),
            "archive_receipt": {"aggregate": aggregate, "catalog_entry": entry},
            "aggregate_sha256": verified.aggregate_sha256,
            "model_sha256": model_sha,
            "config_sha256": config_sha,
            "source_boot": oracle["source_boot"],
            "engine_id": oracle["engine"],
            "namespace": oracle["namespace"],
            "generation": oracle["generation"],
            "expected_output_count": count,
            "copied_parts": len(parts),
            "copied_payload_bytes": sum(part["payload_bytes"] for part in parts),
            "target": target
            | {"mode": stat.S_IMODE((dest / target["relative_path"]).stat().st_mode)},
        }
        emit(arm / "LAB.json", manifest)
        roots[kind] = identity(dest)
    require(
        roots["corrupt"] != roots["unreadable"], "Fault arms share one mount identity"
    )
    return {"schema": SCHEMA, "prepared": list(KINDS), "root_identities": roots}


@contextlib.contextmanager
def mutate(path, target, kind):
    expected = target["payload_sha256"]
    mode = target["mode"]
    with os.fdopen(os.open(path, os.O_RDWR | os.O_NOFOLLOW), "r+b", buffering=0) as f:
        before = os.fstat(f.fileno())
        require(
            before.st_uid == os.geteuid()
            and before.st_nlink == 1
            and before.st_size == target["payload_bytes"]
            and stat.S_IMODE(before.st_mode) == mode
            and sha256_file(path) == expected,
            "Refuse to modify anything except the exact cloned private part",
        )
        first = f.read(1)
        try:
            if kind == "corrupt":
                f.seek(0)
                f.write(bytes([first[0] ^ 1]))
                os.fsync(f.fileno())
                changed = sha256_file(path)
                require(changed != expected, "Corruption did not change the payload")
                yield {
                    "sha256_before": expected,
                    "sha256_mutated": changed,
                    "offset": 0,
                }
            else:
                os.fchmod(f.fileno(), 0)
                try:
                    with path.open("rb") as fresh:
                        fresh.read(1)
                except PermissionError as error:
                    require(error.errno == 13, "Permission probe did not see EACCES")
                else:
                    raise ValueError("The service UID can still read the cloned part")
                yield {"sha256_before": expected, "mutated_mode": 0, "local_errno": 13}
        finally:
            if kind == "corrupt":
                f.seek(0)
                f.write(first)
            else:
                os.fchmod(f.fileno(), mode)
            os.fsync(f.fileno())
        require(
            sha256_file(path) == expected and stat.S_IMODE(path.stat().st_mode) == mode,
            "Fault clone did not recover its exact payload and permissions",
        )


def _uid_no_dac(pid):
    proc = Path(f"/proc/{pid}/status").read_text()
    entries = dict(row.split(":", 1) for row in proc.splitlines() if ":" in row)
    # CAP_DAC_OVERRIDE=1, CAP_DAC_READ_SEARCH=2.
    return (
        os.geteuid() > 0
        and int(entries["Uid"].split()[1]) == os.geteuid()
        and int(entries["CapEff"].strip(), 16) & 0b110 == 0
    )


def runtime_source(args, lab):
    plan = load(args.plan)
    record = _read_manifest(args.release)
    wheel = record["wheel"]
    require(
        Path(wheel["filename"]).name == wheel["filename"]
        and sha256_file(args.release.with_name(wheel["filename"])) == wheel["sha256"],
        "Candidate wheel differs from the sealed release",
    )
    decode = plan["roles"]["decode"]["argv"]

    def flag(name):
        return decode[decode.index(name) + 1]

    extra = json.loads(flag("--additional-config"))
    boot = plan["authority_boot_id"]
    root = args.archive.resolve(strict=True)
    require(
        lab["schema"] == SCHEMA
        and lab["case"] in KINDS
        and root == args.runtime.resolve(strict=True) / "kitepond-nvme"
        and stat.S_IMODE(root.stat().st_mode) == 0o700
        and identity(root) == lab["clone_root"]
        and identity(root) != lab["original_root"]
        and sha256_file(args.oracle) == lab["oracle_sha256"]
        and record["base"]["image"].endswith(plan["base_image"])
        and plan["authority_engine_id"] == lab["engine_id"]
        and boot != lab["source_boot"]
        and plan["checkpoint_identity"]["model_sha256"] == lab["model_sha256"]
        and flag("--port") == str(PORT)
        and flag("--host") == "127.0.0.1"
        and flag("--served-model-name") == MODEL
        and "--no-enable-prefix-caching" in decode
        and extra["kitepond_glm_cold"]["root"] == str(root)
        and extra["kitepond_glm_cold"]["boot_id"] == boot,
        "Health and recovery must run only on this cloned, private D boot",
    )
    from VERIFY_SHARED_OWNERSHIP import cold_decode_by_rank, process_tree

    pid = int((args.runtime / "decode/PID").read_text())
    workers, authority = cold_decode_by_rank(pid, process_tree())
    worker_pids = {str(rank): row[0] for rank, row in workers.items()}
    core_pids = {row[0] for row in authority.values()}
    require(
        set(worker_pids) == {"0", "1", "2", "3"}
        and len(set(worker_pids.values())) == 4
        and len(core_pids) == 1
        and all(_uid_no_dac(proc) for proc in {pid, *worker_pids.values(), *core_pids}),
        "Native D core and each rank must run under this unprivileged UID",
    )
    source = EngineSource(
        PORT, pid, _process_start(pid), f"{boot}/decode", MODEL, ("0",)
    )
    owners = {
        "api_pid": pid,
        "rank_pids": worker_pids,
        "core_pid": min(core_pids),
    }
    return root, source, boot, owners


def native_idle(sample, state):
    return (
        sample["metrics"].get("healthy") is True
        and sample["metrics"].get("running") == sample["metrics"].get("waiting") == 0
        and state["stage"] == "decode"
        and all(
            type(value) is int and value == 0
            for value in (
                state.get("active_leases"),
                state.get("ready_leases"),
                state.get("scheduled_imports"),
                *(
                    state["native_pool"].get(k)
                    for k in (
                        "active_grants",
                        "awaiting_settlement",
                        "grant_block_references",
                    )
                ),
            )
        )
    )


async def drain(client, token, source, host_boot):
    async with asyncio.timeout(30):
        while True:
            sample = await asyncio.to_thread(
                _capture_engine, "decode", source, host_boot
            )
            status = await _control(client, PORT, "status", {}, token)
            if native_idle(sample, status):
                return {"decode": sample, "native": status}
            await asyncio.sleep(0.2)


def metered_delta(start, end):
    before, after = start["decode"], end["decode"]
    a, b = before["metrics"].get(COUNTER), after["metrics"].get(COUNTER)
    require(
        before.get("epoch") is not None
        and before["epoch"] == after["epoch"]
        and before.get("scope") == after.get("scope")
        and type(a) is int
        and type(b) is int
        and b >= a,
        "Native D generation scope changed or its counter is unavailable",
    )
    return b - a


async def probe(args):
    lab = load(args.lab)
    root, source, boot, owners = runtime_source(args, lab)
    selected = lab["target"]
    path = exact_file(root, selected["relative_path"], selected["payload_sha256"])
    driver = _nvme_module(args.driver, args.driver_sha256)
    private = os.environ.get(driver.TOKEN_ENV)
    cold = os.environ.get("KITEPOND_GLM_COLD_HTTP_TOKEN")
    require(
        driver.is_control_token(private) and driver.is_control_token(cold),
        "The isolated native D control credentials are unavailable",
    )
    key = driver.private_key(Path(os.environ["KITEPOND_GLM_NVME_OPERATOR_KEY_FILE"]))
    catalog_key = Path(os.environ["KITEPOND_GLM_NVME_CATALOG_KEY_FILE"])
    require(
        sha256_file(catalog_key) == lab["catalog_key_sha256"],
        "Catalog authority changed",
    )
    oracle = load(args.oracle)
    count = output_count(oracle, lab.get("expected_output_count", 8))
    checked = authenticated(
        root,
        driver.private_key(catalog_key),
        oracle,
        boot,
        lab["model_sha256"],
        lab["config_sha256"],
    )
    require(
        checked.aggregate_sha256 == lab["aggregate_sha256"],
        "Authenticated aggregate changed",
    )
    receipt = lab["archive_receipt"]["catalog_entry"]
    entry = load(exact_file(root, receipt["path"], receipt["sha256"]))
    rank = selected["rank"]
    rank_file = Path(f"rank-{rank}") / "manifests" / entry["rank_files"][rank]
    part = load(below(root, rank_file))["parts"][selected["part"]]
    spec = checked.checkpoint.plans[rank].parts[selected["part"]]
    require(
        str(payload_path(rank, part)) == selected["relative_path"]
        and selected["payload_sha256"] == checked.checksums(rank)[spec]
        and selected["payload_bytes"] == spec.nbytes,
        "The selected part is outside the authenticated native rank",
    )
    host_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    request = SimpleNamespace(
        boot=boot,
        engine=lab["engine_id"],
        artifact=args.oracle,
        model=MODEL,
        timeout=args.timeout,
    )
    require(
        args.output.parent.resolve() == args.runtime.resolve(),
        "Keep evidence in the lab",
    )
    args.output.mkdir(mode=0o700)
    async with httpx.AsyncClient(trust_env=False, timeout=None) as client:
        call = driver.command(client, request, key, private)
        start = await drain(client, cold, source, host_boot)
        emit(args.output / "before.json", start)
        status = []

        async def observe(operation, request_id, **fields):
            reply = await call(operation, request_id, **fields)
            if operation in ("status", "cancel"):
                status.append(
                    {
                        "operation": operation,
                        "request_id": request_id,
                        "state": reply.get("state"),
                        "reason": reply.get("reason"),
                        "observed_ns": monotonic_ns(),
                    }
                )
            return reply

        try:
            with mutate(path, selected, lab["case"]) as evidence:
                emit(
                    args.output / "injection.json",
                    {
                        "schema": SCHEMA,
                        "case": lab["case"],
                        "target": selected,
                        "d_processes_same_uid_without_dac": owners,
                        **evidence,
                    },
                )
                try:
                    await driver.restore(client, request, observe)
                except driver.RestoreProbeFailure as error:
                    failure = error.receipt
                else:
                    raise ValueError("Faulted native restore generated output")
                emit(
                    args.output / "miss.json",
                    {"fault": failure, "native_status": status},
                )
                require(
                    failure["phase"] == "native_cold_readback"
                    and any(
                        row["state"] == "miss"
                        and row["reason"] == "GLM_COLD_READBACK_MISS"
                        and row["request_id"] == failure["restore_request_id"]
                        for row in status
                    )
                    and all(row["state"] != "admitted" for row in status),
                    "Restore did not report a native payload MISS before D admission",
                )
                drained = await drain(client, cold, source, host_boot)
                emit(args.output / "after-miss.json", drained)
                require(
                    metered_delta(start, drained) == 0, "Fault target generated tokens"
                )
                require(
                    await call("forget", failure["restore_request_id"])
                    == {"state": "forgotten"},
                    "Native owner retained the failed operation",
                )
        finally:
            emit(
                args.output / "payload-recovered.json",
                {
                    "sha256": sha256_file(path),
                    "mode": stat.S_IMODE(path.stat().st_mode),
                },
            )

        clean = await driver.restore(client, request, call)
        final = await drain(client, cold, source, host_boot)
        emit(args.output / "after-revalidation.json", final)
        require(
            await call("forget", clean["restore_request_id"]) == {"state": "forgotten"}
            and clean["accepted_native"] is True
            and clean["matches_warm_token_ids"] is True
            and clean["completion_sha256"] == digest(oracle["expected_token_ids"])
            and metered_delta(drained, final) == clean["completion_tokens"] == count,
            "Repaired archive failed exact positive native recovery",
        )
        result = {
            "schema": SCHEMA,
            "case": lab["case"],
            "scope": "native_clone_lab",
            "lab_sha256": sha256_file(args.lab),
            "source_boot": lab["source_boot"],
            "decode_boot": boot,
            "failure": failure,
            "revalidation": clean,
            "fault_target_generated_tokens": 0,
            "native_grants_idle": True,
            "release_qualification": "not_evaluated",
        }
        emit(args.output / "result.json", result)
        return result


async def interruptible(args):
    """A launcher SIGTERM unwinds the exact file mutation before exiting."""
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for name in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(name, task.cancel)
    return await probe(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    inspect = sub.add_parser("inventory")
    stage = sub.add_parser("clone")
    active = sub.add_parser("probe")
    for name in ("runtime", "oracle-relative", "plan"):
        inspect.add_argument(f"--{name}", type=Path, required=True)
    inspect.add_argument("--config-sha256", required=True)
    for child in (stage, active):
        for name in ("archive", "oracle", "output"):
            child.add_argument(f"--{name}", type=Path, required=True)
    for name in ("persistence", "key"):
        stage.add_argument(f"--{name}", type=Path, required=True)
    for name in ("lab", "runtime", "plan", "release", "driver"):
        active.add_argument(f"--{name}", type=Path, required=True)
    active.add_argument("--driver-sha256", required=True)
    active.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        if args.mode == "inventory":
            result = inventory(args)
        elif args.mode == "clone":
            result = clone(args)
        else:
            require(1 <= args.timeout <= 180, "Unexpected fault runtime envelope")
            result = asyncio.run(interruptible(args))
    except asyncio.CancelledError:
        parser.exit(
            2, "Fault probe interrupted. Inspect the private payload recovery record.\n"
        )
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        httpx.HTTPError,
    ) as error:
        parser.exit(
            2,
            "Private NVMe fault case did not finish "
            f"({type(error).__name__}): {error}\n",
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
