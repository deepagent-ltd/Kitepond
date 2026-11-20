# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inject one isolated native P/D process exit, or verify a fresh-epoch recovery."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import signal
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from time import monotonic_ns

import httpx
from kitepond.integrations.vllm.cold_prefill_http import TOKEN_ENV, check_ready_batch
from kitepond.models.glm5next.frontier import digest
from kitepond.ops.glm_cold_probe import _API_KEY_ENV, ProbeFailure, run_probe
from kitepond.ops.wheel import sha256_file
from phase1_fault_probe import Pair
from phase1_role_exit_gate import (
    POSITION,
    LinuxProcess,
    emit,
    load,
    private_directory,
)

CASES = {
    "prefill_process_exit": ("prefill", "decode"),
    "decode_process_exit": ("decode", "prefill"),
}
RANKS = tuple(range(4))


def group_members(leader: LinuxProcess) -> tuple[LinuxProcess, ...]:
    return tuple(
        sorted(
            (
                item
                for path in Path("/proc").iterdir()
                if path.name.isdecimal()
                and (item := LinuxProcess.read(int(path.name))) is not None
                and item.group == leader.pid
                and item.session == leader.pid
                and item.state not in {"Z", "X"}
            ),
            key=lambda item: item.pid,
        )
    )


def candidate_group(runtime: Path, role: str, expected_gpu_ranks: Iterable[int]):
    pid = int((runtime / role / "PID").read_text())
    leader = LinuxProcess.read(pid)
    if (
        leader is None
        or not leader.live()
        or (leader.group, leader.session) != (pid, pid)
        or pid == os.getpid()
        or pid == os.getsid(0)
    ):
        raise ValueError("Sealed role PID is not a live isolated session leader")
    peers = group_members(leader)
    if pid not in {row.pid for row in peers} or not set(expected_gpu_ranks) <= {
        row.pid for row in peers
    }:
        raise ValueError("GPU ranks do not belong to their leased role group")
    current = Path("/proc/self/cgroup").read_text()
    # Runtime is owned by a dedicated container. No unrelated group may be sent a signal.
    for member in peers:
        process = Path(f"/proc/{member.pid}")
        if (
            process.stat().st_uid != os.geteuid()
            or (process / "cgroup").read_text() != current
            or not member.live()
        ):
            raise ValueError("Sealed role group contains a foreign or replaced process")
    return leader, peers


def process_identity(process):
    return process.pid, process.group, process.session, process.start_ticks


def kill_identity_group(leader, members):
    """Pin every PID, then signal this registered role only."""
    with ExitStack() as stack:
        descriptors = {}
        for process in members:
            descriptor = os.pidfd_open(process.pid)
            stack.callback(os.close, descriptor)
            descriptors[process.pid] = descriptor
        if tuple(map(process_identity, group_members(leader))) != tuple(
            map(process_identity, members)
        ) or not all(m.live() for m in members):
            raise ValueError("Role process group changed before the exit signal")
        started = monotonic_ns()
        for process in sorted(members, key=lambda row: row.pid != leader.pid):
            try:
                signal.pidfd_send_signal(descriptors[process.pid], signal.SIGKILL)
            except ProcessLookupError:
                if process.live():
                    raise
        return started


async def await_exact_exit(leader, members):
    async with asyncio.timeout(8):
        while any(member.live() for member in members) or group_members(leader):
            await asyncio.sleep(0.025)
    return monotonic_ns()


def worker_pids(ownership, role):
    rows = ownership["roles"][role]
    if len(rows) != 4 or len({row[0] for row in rows}) != 4:
        raise ValueError("Sealed role needs its four genuine GPU worker owners")
    return tuple(row[0] for row in rows)


async def await_fences(directory, target, groups):
    paths = tuple(
        (directory / f"source-fenced-{rank}.json", directory / f"rank-{rank}.json")
        for rank in RANKS
    )
    while monotonic_ns() < target["deadline_ns"]:
        if all(source.is_file() and receiver.is_file() for source, receiver in paths):
            break
        await asyncio.sleep(0.015)
    else:
        raise TimeoutError(
            "Four real P/D fenced windows did not reach the exact fault boundary"
        )
    observed = []
    for rank, (source_file, receive_file) in enumerate(paths):
        source, receiver = load(source_file), load(receive_file)
        for event, role in ((source, "prefill"), (receiver, "decode")):
            identity = (
                "case_id",
                "request_id",
                "transfer_id",
                "epoch",
                "grant_sha256",
                "source_boot_id",
                "decode_boot_id",
            )
            if event["rank"] != rank or any(event[k] != target[k] for k in identity):
                raise ValueError("Fenced window belongs to another rank or attempt")
            matches = tuple(
                process
                for process in groups[role][1]
                if process.pid == event["worker_pid"]
                and process.start_ticks == event["worker_start_ticks"]
                and process.live()
            )
            if len(matches) != 1:
                raise ValueError("Window receipt does not come from the real role")
        if (
            source["writer_released"] is not False
            or source["dma_uncertain"] is not False
            or not (
                target["armed_ns"]
                <= source["observed_monotonic_ns"]
                <= receiver["observed_monotonic_ns"]
                < target["deadline_ns"]
            )
            or source["manifest_sha256"] != receiver["manifest_sha256"]
            or source["written_nbytes"] != receiver["received_nbytes"]
            or receiver["position"] != POSITION
            or receiver["window_sequence"] != 0
            or not all(
                receiver[field] is True
                for field in (
                    "remote_write_fenced",
                    "local_cuda_fenced",
                    "source_commit_pending",
                )
            )
        ):
            raise ValueError(
                "Real native transfer changed its fenced step before the exit"
            )
        observed.append({"rank": rank, "source": source, "receiver": receiver})
    return observed


async def survivor_status(pair, role, process):
    if not process.live():
        raise ValueError("Untargeted route exited before observing its old epoch")
    async with asyncio.timeout(7):
        health = await pair.client.get(
            f"http://127.0.0.1:{pair.config.decode_port if role == 'decode' else pair.config.prefill_port}/health"
        )
        health.raise_for_status()
        return await pair.control(role, "status", {})


async def old_request_disposition(task, cap):
    try:
        payload = await asyncio.wait_for(asyncio.shield(task), timeout=cap)
    except ProbeFailure as error:
        return {"disposition": "native_call_failed", "probe": error.report()}
    except TimeoutError:
        # A client-observation timeout is not evidence that the old leases were released.
        return {"disposition": "incomplete_at_observation_limit"}
    raise RuntimeError(
        f"Interrupted old operation completed unexpectedly: {payload['status']}"
    )


async def inject(pair, args, prompt):
    target_role, survivor = CASES[args.case]
    initial = await pair.drain()
    emit(args.output / "before.json", initial)
    gate = private_directory(args.runtime / "role-exit-gate")
    ownership = load(args.runtime / "OWNERSHIP.json")
    ranks = {role: worker_pids(ownership, role) for role in CASES[args.case]}
    groups = {role: candidate_group(args.runtime, role, ranks[role]) for role in ranks}
    if {row.pid for row in groups[target_role][1]} & {
        row.pid for row in groups[survivor][1]
    }:
        raise ValueError("Cannot safely isolate the two process role trees")
    if groups[target_role][0].pid == groups[survivor][0].pid:
        raise ValueError("P and D share a process group")
    plan = load(args.pair)
    restarts = plan.get("role_exit_test")
    if restarts != {"case": args.case, "action": "inject"}:
        raise ValueError("The sealed pair does not authorize this exit arm")
    armed = asyncio.Event()
    target = old_ready = None

    async def arm(native_prompt, offer):
        nonlocal target, old_ready
        if (
            offer.prefill_boot_id != pair.boots["prefill"]
            or offer.grant.boot_id != pair.boots["decode"]
        ):
            raise ValueError("Target request belongs to another sealed pair")
        stored = await pair.control("prefill", "read_ready", asdict(offer.attempt))
        if (
            stored.get("status") != "ready"
            or check_ready_batch(stored["receipt_batch"]) != offer.attempt
        ):
            raise ValueError("Old pair lost the exact signed READY before arming")
        old_ready = stored["receipt_batch"]
        now = monotonic_ns()
        target = {
            "case_id": secrets.token_hex(16),
            "recovery_case": args.case,
            "request_id": native_prompt.request_id,
            "transfer_id": native_prompt.transfer_id,
            "epoch": native_prompt.epoch,
            "grant_sha256": offer.grant.sha256,
            "source_boot_id": pair.boots["prefill"],
            "decode_boot_id": pair.boots["decode"],
            "armed_ns": now,
            "deadline_ns": now + 15_000_000_000,
        }
        emit(gate / "target.json", target)
        armed.set()

    task = asyncio.create_task(
        run_probe(
            pair.client, pair.config, prompt, pair.auth, pair.api_key, before_decode=arm
        )
    )
    try:
        async with asyncio.timeout(pair.config.timeout_seconds):
            waiting = asyncio.create_task(armed.wait())
            try:
                await asyncio.wait((task, waiting), return_when=asyncio.FIRST_COMPLETED)
                if task.done():
                    await task
                    raise RuntimeError(
                        "Ordinary native request finished before the arm"
                    )
                if not armed.is_set():
                    raise RuntimeError("Native P READY did not arm the exit attempt")
            finally:
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
        assert target is not None and old_ready is not None
        fences = await await_fences(gate, target, groups)
        for role in groups:
            exact_expected = {
                row["source" if role == "prefill" else "receiver"]["worker_pid"]
                for row in fences
            }
            if exact_expected != set(ranks[role]):
                raise ValueError("Fenced workers disagree with GPU ownership")
        # All four receivers are still held at the native gate. No source RELEASE may have run.
        if any((gate / f"source-released-{rank}.json").exists() for rank in RANKS):
            raise ValueError("A source rank already ceased being in flight")
        if monotonic_ns() >= target["deadline_ns"]:
            raise TimeoutError("Native phase expired before the role exit injection")
        fired = kill_identity_group(*groups[target_role])
        exited = await await_exact_exit(*groups[target_role])
        if not groups[survivor][0].live():
            raise ValueError("Role exit also killed the surviving native peer")
        emit(gate / "release.json", {"case_id": target["case_id"]})
        injection = {
            "case": args.case,
            "case_id": target["case_id"],
            "role": target_role,
            "phase": POSITION,
            "epochs": pair.boots,
            "native_request_id": target["request_id"],
            "native_transfer_id": target["transfer_id"],
            "native_grant_sha256": target["grant_sha256"],
            "signal": "SIGKILL",
            "old_target_role_terminated": True,
            "signals_sent_monotonic_ns": fired,
            "target_ended_monotonic_ns": exited,
            "target_role_pid_identity": asdict(groups[target_role][0]),
            "role_groups": {
                role: [asdict(row) for row in members]
                for role, (_, members) in groups.items()
            },
            "injection_context": teardown_context(),
            "ownership_sha256": sha256_file(args.runtime / "OWNERSHIP.json"),
            "untargeted_role_pid_identity": asdict(groups[survivor][0]),
            "untargeted_role_survived_injection": True,
            "fenced_ranks": fences,
            "old_signed_ready": old_ready,
        }
        emit(args.output / "injection.json", injection)
        disposition = await old_request_disposition(task, 25)
        status = await survivor_status(pair, survivor, groups[survivor][0])
        old_rank_releases = [
            rank for rank in RANKS if (gate / f"source-released-{rank}.json").exists()
        ]
        return {
            "schema": "kitepond.glm-role-exit-old.v1",
            "case": args.case,
            "case_id": target["case_id"],
            "status": "injection_observed",
            "artifact": pair.artifact,
            "engine_ids": {
                role: plan["mooncake_api"][
                    "p_engine_id" if role == "prefill" else "d_engine_id"
                ]
                for role in ranks
            },
            "before_role_epochs": pair.boots,
            "old_plan_sha256": pair.plan_sha,
            "injection_sha256": sha256_file(args.output / "injection.json"),
            "target_rank_devices": sorted(
                plan["roles"][target_role]["host_gpu_indices"]
            ),
            "old_request": disposition,
            "old_survivor_native_status": status,
            "old_source_ranks_with_observed_release": old_rank_releases,
            "same_epoch_reap_proven": False,
            "relaunch_mode_required": "paired_verified_teardown",
        }
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def teardown_context():
    pid_namespace = Path("/proc/self/ns/pid").stat()
    return {
        "pid_namespace": [pid_namespace.st_dev, pid_namespace.st_ino],
        "kernel_boot_sha256": sha256_file(Path("/proc/sys/kernel/random/boot_id")),
        "cgroup_sha256": sha256_file(Path("/proc/self/cgroup")),
    }


def finalize_teardown(runtime: Path, arm: Path):
    """Witness both sealed groups disappear from inside the old namespace."""
    private_directory(runtime)
    private_directory(arm)
    old, injection = load(arm / "result.json"), load(arm / "injection.json")
    recorded_context = injection["injection_context"]
    if (
        arm.parent != runtime
        or old["status"] != "injection_observed"
        or old["case_id"] != injection["case_id"]
        or old["injection_sha256"] != sha256_file(arm / "injection.json")
        or recorded_context != teardown_context()
        or injection["ownership_sha256"] != sha256_file(runtime / "OWNERSHIP.json")
        or set(injection["role_groups"]) != set(CASES[old["case"]])
    ):
        raise ValueError("Teardown must be checked in the matching old operation")
    roles = {}
    for role, rows in injection["role_groups"].items():
        members = tuple(LinuxProcess(**row) for row in rows)
        pid = int((runtime / role / "PID").read_text())
        leaders = [
            row
            for row in members
            if (row.pid, row.group, row.session) == (pid, pid, pid)
        ]
        if len(leaders) != 1 or any(row.live() for row in members):
            raise ValueError("The old pair has a live or unverified role member")
        if group_members(leaders[0]):
            raise ValueError("The old role session still has members")
        roles[role] = len(members)
    receipt = {
        "schema": "kitepond.glm-role-exit-teardown.v1",
        "case": old["case"],
        "case_id": old["case_id"],
        "epochs": old["before_role_epochs"],
        "artifact": old["artifact"],
        "injection_sha256": old["injection_sha256"],
        "old_result_sha256": sha256_file(arm / "result.json"),
        "context": recorded_context,
        "closed_session_members": roles,
        "observed_monotonic_ns": monotonic_ns(),
    }
    emit(arm / "teardown.json", receipt)
    return receipt


async def verify(pair, args, prompt):
    old, injection = (
        load(args.prior / "result.json"),
        load(args.prior / "injection.json"),
    )
    teardown = load(args.prior / "teardown.json")
    target_role, _ = CASES[args.case]
    plan = load(args.pair)
    state = plan.get("role_exit_test")
    prior_epochs = old["before_role_epochs"]
    if (
        state != {"case": args.case, "action": "verify"}
        or old["schema"] != "kitepond.glm-role-exit-old.v1"
        or old["status"] != "injection_observed"
        or old["case"] != args.case
        or injection["case"] != args.case
        or old["injection_sha256"] != sha256_file(args.prior / "injection.json")
        or old["case_id"] != injection["case_id"]
        or injection["role"] != target_role
        or injection["old_target_role_terminated"] is not True
        or teardown["schema"] != "kitepond.glm-role-exit-teardown.v1"
        or (teardown["case"], teardown["case_id"], teardown["epochs"])
        != (old["case"], old["case_id"], prior_epochs)
        or teardown["artifact"] != old["artifact"]
        or teardown["injection_sha256"] != old["injection_sha256"]
        or teardown["old_result_sha256"] != sha256_file(args.prior / "result.json")
        or teardown["context"] != injection["injection_context"]
        or teardown["closed_session_members"]
        != {role: len(rows) for role, rows in injection["role_groups"].items()}
        or teardown["observed_monotonic_ns"] < injection["target_ended_monotonic_ns"]
        or old["artifact"] != pair.artifact
        or pair.plan_sha == old["old_plan_sha256"]
        or any(pair.boots[role] == prior_epochs[role] for role in CASES[args.case])
        or any(
            plan["mooncake_api"][prefix + "_engine_id"] != old["engine_ids"][role]
            for role, prefix in (("prefill", "p"), ("decode", "d"))
        )
        or sorted(plan["roles"][target_role]["host_gpu_indices"])
        != old["target_rank_devices"]
    ):
        raise ValueError(
            "The new pair cannot be bound to the exact exit artifact and deployment"
        )
    initial = await pair.drain()
    emit(args.output / "fresh-before.json", initial)
    before = await pair.control("decode", "status", {})
    try:
        await pair.control("decode", "confirm_ready", injection["old_signed_ready"])
    except httpx.HTTPStatusError as error:
        if (
            error.response.status_code != 502
            or error.response.json().get("error") != "connector_control_rejected"
        ):
            raise
    else:
        raise ValueError("Fresh D owner accepted the terminated old attempt")
    if before != await pair.control("decode", "status", {}):
        raise ValueError("Old READY changed the fresh D owner")
    followup = prompt.copy()
    followup[0] = followup[0] - 1 if followup[0] else 1
    result = await run_probe(
        pair.client, pair.config, followup, pair.auth, pair.api_key
    )
    output_ids = result.pop("output_token_ids")
    after = await pair.drain()
    emit(args.output / "after.json", after)
    if (
        result["verified_ready_ranks"] != 4
        or result["request_id"] == injection["native_request_id"]
        or result["transfer_id"] == injection["native_transfer_id"]
        or not output_ids
        or after["native"]["decode"]["native_pool"]
        != initial["native"]["decode"]["native_pool"]
    ):
        raise ValueError("Fresh genuine request or lease reclamation did not recover")
    return {
        "schema": "kitepond.glm-role-exit-recovery.v1",
        "status": "paired_recovery_verified",
        "case": args.case,
        "case_id": old["case_id"],
        "artifact": pair.artifact,
        "before_role_epochs": prior_epochs,
        "after_role_epochs": pair.boots,
        "fresh_plan_sha256": pair.plan_sha,
        "injection_sha256": old["injection_sha256"],
        "teardown_sha256": sha256_file(args.prior / "teardown.json"),
        "proof": {
            "injection_observed": True,
            "expected_disposition": True,
            "model_recovered": True,
            "old_ready_rejected_without_mutation": True,
            "lease_released": True,
        },
        "release_scope": "verified_old_pair_teardown_then_fresh_epoch_recovery",
        "same_epoch_reap_proven": False,
        "raw_probe": result,
        "recovery_output_sha256": digest(output_ids),
    }


async def run(args):
    runtime = private_directory(args.runtime)
    if runtime != args.output.parent and runtime not in args.output.parents:
        raise ValueError(
            "Role exit receipt directory must belong to the sealed runtime"
        )
    args.output.mkdir(mode=0o700)
    prompt = json.loads(args.prompt.read_text())
    if type(prompt) is not list or not 1024 <= len(prompt) <= 4096:
        raise ValueError(
            "Choose one bounded raw prompt for an actual multi-window move"
        )
    token = os.environ[TOKEN_ENV]
    async with httpx.AsyncClient(trust_env=False, timeout=None) as client:
        pair = Pair(args, client, token, os.environ.get(_API_KEY_ENV))
        try:
            result = await (
                inject(pair, args, prompt)
                if args.action == "inject"
                else verify(pair, args, prompt)
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
            TimeoutError,
            httpx.HTTPError,
        ) as error:
            emit(
                args.output / "failure.json",
                {
                    "status": "failed",
                    "case": args.case,
                    "exception": type(error).__name__,
                },
            )
            raise
    emit(args.output / "result.json", result)
    print(
        json.dumps(
            {key: result[key] for key in ("case", "case_id", "status")}, sort_keys=True
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inject", "verify", "teardown"))
    parser.add_argument("--case", required=True, choices=tuple(CASES))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pair", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--timeout", type=float, default=85)
    args = parser.parse_args()
    if (args.prior is not None) != (args.action == "verify"):
        parser.error("Fresh verification needs the immutable old arm evidence")
    if args.action == "teardown":
        finalize_teardown(args.runtime, args.output)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
