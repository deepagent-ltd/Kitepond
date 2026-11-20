# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Operate the private lost_write_ack fixture on an already isolated pair."""

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

import httpx
from kitepond.integrations.vllm.cold_prefill_http import TOKEN_ENV, is_control_token
from kitepond.ops.glm_cold_probe import _API_KEY_ENV
from kitepond.ops.wheel import sha256_file
from lost_write_ack_injector import (
    ABORT,
    DROP,
    FENCED,
    HOLD,
    SETTLE,
    AckTarget,
    arm_fault,
)
from phase1_fault_probe import (
    COMPLETION,
    RANKS,
    Pair,
    _emit,
    identities,
    raw_request,
    read,
    recover,
    require,
    stage,
)

LAB_DIR_ENV = "KITEPOND_LAB_LOST_WRITE_ACK_DIRECTORY"
SCHEMA = "kitepond.glm-native-lab-lost-write-ack.v1"


def validate_evidence(directory, target):
    dropped, held, aborted, settled, fenced = [
        read(directory / name) for name in (DROP, HOLD, ABORT, SETTLE, FENCED)
    ]
    attempt = dropped["attempt"]
    identity = target.attempt
    require(
        (
            attempt["request_id"],
            attempt["transfer_id"],
            attempt["epoch"],
            attempt["grant_sha256"],
            attempt["rank"],
        )
        == (
            identity.request_id,
            identity.transfer_id,
            identity.epoch,
            identity.lease_sha256,
            target.rank,
        )
        and dropped["p_boot"] == target.p_boot
        and dropped["d_boot"] == target.d_boot
        and all(row["attempt"] == attempt for row in (held, aborted, settled)),
        "The three native wire responses and D slot must name one exact request",
    )
    require(
        dropped["native_write_completed"] is True
        and dropped["source_released"] is False
        and dropped["window_sequence"] == 0
        and dropped["nbytes"] > 0
        and held["slot"] == dropped["slot"]
        and held["generation"] == dropped["generation"]
        and held["leased"] is True
        and held["native_result_published"] is False
        and len(held["payload_sha256"]) == 64,
        "No actual pre-fence D lease observation for the discarded sync-write ACK",
    )
    require(
        dropped["control_sequence"]
        < aborted["control_sequence"]
        < settled["control_sequence"]
        and dropped["monotonic_ns"]
        <= held["monotonic_ns"]
        < aborted["monotonic_ns"]
        < settled["monotonic_ns"]
        <= fenced["monotonic_ns"]
        and aborted["closed"] is True
        and settled["outcome"] == "abort_settled",
        "The actual aborted source and target fences did not follow the missing ACK",
    )
    receipt = fenced["native_receipt"]
    require(
        (fenced["request_id"], fenced["transfer_id"], fenced["epoch"])
        == (identity.request_id, identity.transfer_id, identity.epoch)
        and receipt["rank"] == target.rank
        and receipt["grant_sha256"] == identity.lease_sha256
        and receipt["destination_sha256"] == attempt["destination_sha256"]
        and receipt["outcome"] == "abort_settled"
        and receipt["pending_reads"] == receipt["pending_writes"] == 0
        and fenced["leased_scratch_slots"] == fenced["quarantined_scratch_slots"] == 0,
        "D did not publish its actual native abort fence and clean scratch state",
    )
    return {
        name: sha256_file(directory / name)
        for name in (DROP, HOLD, ABORT, SETTLE, FENCED)
    }


async def run(args):
    auth = os.environ.get(TOKEN_ENV)
    require(is_control_token(auth), "Missing private connector token")
    args.runtime = args.runtime.resolve(strict=True)
    require(
        args.output.parent.resolve() == args.runtime,
        "Fault output must stay within the current isolated boot",
    )
    directory = Path(os.environ[LAB_DIR_ENV])
    require(
        directory.parent.resolve() == args.runtime, "Fault arm belongs to this boot"
    )
    tokens = read(args.prompt)
    require(
        isinstance(tokens, list)
        and 2 <= len(tokens) <= 8192
        and all(type(token) is int and token >= 0 for token in tokens),
        "Expected a bounded frozen raw-token fault prompt",
    )
    args.output.mkdir(mode=0o700)
    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
        pair = Pair(args, client, auth, os.environ.get(_API_KEY_ENV))
        for role in ("prefill", "decode"):
            argv = pair.roles[role]["argv"]
            worker = argv[argv.index("--worker-cls") + 1]
            label = "Prefill" if role == "prefill" else "Decode"
            require(
                worker == f"phase1_ack_fault_worker.LostWriteAck{label}Worker",
                "Pair did not explicitly select the diagnostic worker",
            )
        before = await pair.drain()
        _emit(args.output / "before.json", before)
        api_id, transfer = identities()
        target = None

        async def before_confirm(offer, _batch):
            nonlocal target
            target = AckTarget(
                offer.attempt, args.rank, pair.boots["prefill"], pair.boots["decode"]
            )
            arm_fault(directory, target)

        async with asyncio.timeout(args.timeout):
            from time import time_ns

            _, _, _, body = await stage(
                pair,
                tokens,
                api_id,
                transfer,
                time_ns(),
                raw_request(tokens),
                COMPLETION,
                before_confirm=before_confirm,
            )
            if target is None:
                raise ValueError("Native P did not provide the exact READY grant")
            try:
                response = await pair.post(
                    "decode", COMPLETION, body | {"max_tokens": 8}
                )
            except httpx.HTTPStatusError as error:
                terminal = {"http_status": error.response.status_code}
            else:
                choice = response["choices"][0]
                require(
                    choice["finish_reason"] in {"abort", "error"}
                    and response["usage"]["completion_tokens"] == 0,
                    "Targeted native lost ACK unexpectedly produced model output",
                )
                terminal = {
                    "http_status": 200,
                    "finish_reason": choice["finish_reason"],
                }

        _emit(args.output / "after-abort-before-followup.json", await pair.drain())
        async with asyncio.timeout(4):
            while not (directory / FENCED).is_file():
                await asyncio.sleep(0.05)
        evidence = validate_evidence(directory, target)
        recovery = await recover(pair, tokens)
        require(recovery["closed_ranks"] == RANKS, "Fresh native request did not close")
        after = await pair.drain()
        require(
            evidence == validate_evidence(directory, target),
            "Fault fired more than once",
        )
        _emit(args.output / "after.json", after)
        result = {
            "schema": SCHEMA,
            "case": "lost_write_ack",
            "artifact": pair.artifact,
            "plan_sha256": pair.plan_sha,
            "target": asdict(target),
            "injection_observed": True,
            "native_terminal": terminal,
            "expected_disposition": True,
            "model_recovered": True,
            "lease_released": True,
            "wire_and_target_evidence_sha256": evidence,
            "follow_up": recovery,
        }
        _emit(args.output / "result.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "pair", "prompt", "runtime", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(4), default=2)
    parser.add_argument("--timeout", type=int, default=110)
    args = parser.parse_args()
    try:
        require(10 <= args.timeout <= 120, "Fault deadline must be bounded")
        result = asyncio.run(run(args))
    except (ValueError, RuntimeError, OSError, TimeoutError, httpx.HTTPError) as error:
        parser.exit(
            2, f"Private lost ACK probe unfinished ({type(error).__name__}): {error}\n"
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
