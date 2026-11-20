# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the installed native cold probe continuously on an isolated 4P4D pair."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import secrets
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
from kitepond.integrations.vllm.cold_prefill_http import TOKEN_ENV, is_control_token
from kitepond.models.glm5next.frontier import digest as prompt_digest
from kitepond.ops.capture import capture_snapshot
from kitepond.ops.cli import _read_manifest
from kitepond.ops.glm_cold_probe import ProbeConfig, ProbeFailure, _control, run_probe
from kitepond.ops.report import operational_report
from kitepond.ops.wheel import sha256_file

SCHEMA = "kitepond.phase1-soak-plan.v1"
_TRACE = "request-results.jsonl"
_CONCURRENCIES = {1, 32, 64}
_PREFIX_WIDTH = 8
_ROLE_NAMES = ("prefill", "decode")


@dataclass(frozen=True)
class Population:
    name: str
    sessions: int
    prompt_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Workload:
    seed: int
    token_min: int
    token_max: int
    populations: tuple[Population, ...]

    def __post_init__(self):
        if (
            type(self.seed) is not int
            or type(self.token_min) is not int
            or type(self.token_max) is not int
            or not 0 <= self.token_min < self.token_max < 2**31
            or not self.populations
            or len({item.name for item in self.populations}) != len(self.populations)
            or any(
                not isinstance(item.name, str)
                or not item.name.strip()
                or type(item.sessions) is not int
                or not 1 <= item.sessions <= 1024
                or type(item.prompt_tokens) is not int
                or not _PREFIX_WIDTH <= item.prompt_tokens <= 8192
                or type(item.output_tokens) is not int
                or not 1 <= item.output_tokens <= 128
                for item in self.populations
            )
        ):
            raise ValueError("Invalid raw-token workload population")


class PromptStream:
    """Interleave planned sessions with a unique prefix within this run."""

    def __init__(self, workload: Workload, nonce: str):
        self._workload = workload
        self._sessions = [
            (item, index)
            for item in workload.populations
            for index in range(item.sessions)
        ]
        self._ordinal = 0
        self._base = workload.token_max - workload.token_min + 1
        self._space = self._base**_PREFIX_WIDTH
        self._seed = int.from_bytes(
            hashlib.sha256(f"{workload.seed}:{nonce}".encode()).digest()
        )
        rng = random.Random(self._seed)
        rng.shuffle(self._sessions)
        self._step = rng.randrange(1, self._space)
        while math.gcd(self._step, self._base) != 1:
            self._step = rng.randrange(1, self._space)

    def take(self) -> tuple[Population, dict, list[int]]:
        ordinal = self._ordinal
        if ordinal >= self._space:
            raise ValueError("Raw token prefix address space exhausted")
        self._ordinal += 1
        item, session = self._sessions[ordinal % len(self._sessions)]
        prefix_value = (self._seed + self._step * ordinal) % self._space
        tokens = []
        for _ in range(_PREFIX_WIDTH):
            prefix_value, digit = divmod(prefix_value, self._base)
            tokens.append(self._workload.token_min + digit)
        rng = random.Random(self._seed + ordinal)
        tokens.extend(
            rng.randint(self._workload.token_min, self._workload.token_max)
            for _ in range(item.prompt_tokens - _PREFIX_WIDTH)
        )
        return (
            item,
            {
                "ordinal": ordinal,
                "population": item.name,
                "session": session,
                "turn": ordinal // len(self._sessions),
                "planned_input_tokens": item.prompt_tokens,
                "planned_output_tokens": item.output_tokens,
                "prompt_sha256": prompt_digest(tokens),
            },
            tokens,
        )


class RequestStats:
    """Record only small counters; detailed results are flushed per request."""

    def __init__(self):
        self.started = 0
        self.completed = 0
        self.failed = 0
        self.peak_inflight = 0
        self.prefill_api_tokens = 0
        self.output_tokens = 0
        self.populations: dict[str, int] = {}
        self.failures: dict[str, int] = {}

    def launch(self):
        self.started += 1
        inflight = self.started - self.completed - self.failed
        self.peak_inflight = max(self.peak_inflight, inflight)

    def record(self, population: str, result: dict):
        if result["status"] == "completed":
            self.completed += 1
            self.prefill_api_tokens += result["prefill_api_prompt_tokens"]
            self.output_tokens += result["output_tokens"]
            self.populations[population] = self.populations.get(population, 0) + 1
        else:
            self.failed += 1
            key = f"{result['phase']} / {result['reason']}"
            self.failures[key] = self.failures.get(key, 0) + 1


def _emit(path: Path, value: dict | list) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _line(stream, value: dict) -> None:
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


async def fill_workload(
    stream,
    prompts: PromptStream,
    *,
    probe,
    concurrency: int,
    seconds: float,
    max_failures: int,
    stop: asyncio.Event,
    stats: RequestStats,
    max_attempts: int | None = None,
) -> dict:
    """Use independent slots so completion immediately admits the next prompt."""
    start = time.monotonic_ns()
    deadline = start + round(seconds * 1e9)
    last_launch = start

    async def slot():
        nonlocal last_launch
        while (
            not stop.is_set()
            and time.monotonic_ns() < deadline
            and (max_attempts is None or stats.started < max_attempts)
        ):
            population, row, tokens = prompts.take()
            stats.launch()
            last_start = time.monotonic_ns()
            last_launch = last_start
            try:
                reply = await probe(population, tokens)
            except ProbeFailure as exc:
                reply = exc.report()
            stats.record(population.name, reply)
            _line(
                stream,
                {
                    **row,
                    "request": reply,
                    "started_ns": last_start,
                    "ended_ns": time.monotonic_ns(),
                },
            )
            if stats.failed > max_failures:
                stop.set()

    # Time is recorded per slot before the first yield to the HTTP client.
    # All writes stay on the event loop thread.
    async with asyncio.TaskGroup() as group:
        for _ in range(concurrency):
            group.create_task(slot())

    return {
        "requested_active_seconds": seconds,
        "start_monotonic_ns": start,
        "last_launch_monotonic_ns": last_launch,
        "requests_drained_monotonic_ns": time.monotonic_ns(),
        "stopped_by_guard": stop.is_set(),
        "reached_active_deadline": time.monotonic_ns() >= deadline,
    }


def _role_epoch(snapshot: dict, role: str) -> str | None:
    return snapshot["components"].get(role, {}).get("epoch")


def _role_idle(snapshot: dict) -> bool:
    return all(
        (metrics := snapshot["components"].get(role, {}).get("metrics", {})).get(
            "healthy"
        )
        is True
        and metrics.get("running") == 0
        and metrics.get("waiting") == 0
        for role in _ROLE_NAMES
    )


def _check_capacity(
    prefill: dict, decode: dict, planned_concurrency: int, peak_prompt_tokens: int
) -> dict:
    limits = [row.get("max_pending") for row in (prefill, decode)]
    native_pool = decode["native_pool"]
    block_tokens = native_pool.get("block_tokens")
    free = native_pool.get("available_pages")
    if (
        any(type(value) is not int or value < planned_concurrency for value in limits)
        or type(block_tokens) is not int
        or block_tokens <= 0
        or type(free) is not int
        or free < 0
    ):
        raise ValueError("Current native admission or D pool capacity is unavailable")
    required = planned_concurrency * (
        (peak_prompt_tokens + block_tokens - 1) // block_tokens
    )
    if free < required:
        raise ValueError(
            "Current D pool cannot hold the planned raw prompt reservations"
        )
    return {
        "planned_concurrency": planned_concurrency,
        "prefill_pending_limit": limits[0],
        "decode_pending_limit": limits[1],
        "decode_free_pages": free,
        "decode_pages_required_for_prompt_reservations": required,
        "decode_block_tokens": block_tokens,
    }


def _no_native_holds(prefill: dict, decode: dict) -> bool:
    pool = decode["native_pool"]
    return all(
        type(value) is int and value == 0
        for value in (
            prefill.get("active_leases"),
            prefill.get("ready_leases"),
            decode.get("active_leases"),
            decode.get("ready_leases"),
            decode.get("scheduled_imports"),
            pool.get("active_grants"),
            pool.get("awaiting_settlement"),
            pool.get("grant_block_references"),
        )
    )


def _read_plan(path: Path):
    raw = json.loads(path.read_bytes())
    if raw["schema"] != SCHEMA:
        raise ValueError("Unsupported soak plan")
    concurrency = raw["concurrency"]
    seconds, interval, drain = (
        raw["active_seconds"],
        raw["sample_seconds"],
        raw["drain_timeout_seconds"],
    )
    if (
        type(concurrency) is not int
        or concurrency not in _CONCURRENCIES
        or any(
            type(value) is not int or value <= 0 for value in (seconds, interval, drain)
        )
        or interval >= seconds
        or type(raw["max_failures"]) is not int
        or raw["max_failures"] < 0
    ):
        raise ValueError(
            "Invalid soak duration, sampling period, failure cap or concurrency"
        )
    workload = Workload(
        seed=raw["seed"],
        token_min=raw["token_min"],
        token_max=raw["token_max"],
        populations=tuple(Population(**item) for item in raw["populations"]),
    )
    return raw, workload


async def _roles_status(client, config: ProbeConfig, token):
    async with asyncio.timeout(10):
        prefill, decode = await asyncio.gather(
            _control(client, config.prefill_port, "status", {}, token),
            _control(client, config.decode_port, "status", {}, token),
        )
    if prefill.get("stage") != "prefill" or decode.get("stage") != "decode":
        raise ValueError("Native status does not match the configured role")
    return prefill, decode


def _read_resources(limits: dict) -> dict:
    memory = Path("/proc/meminfo").read_text(encoding="ascii")
    mem_available = next(
        int(row.split()[1]) * 1024
        for row in memory.splitlines()
        if row.startswith("MemAvailable:")
    )
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout
    visible = {}
    for row in output.splitlines():
        name, used, total = (item.strip() for item in row.split(","))
        if name in limits["gpu_uuids"]:
            used_bytes, total_bytes = int(used) * 1024**2, int(total) * 1024**2
            if not 0 <= used_bytes <= total_bytes or not total_bytes:
                raise ValueError("Invalid device memory sample")
            visible[name] = {"used_bytes": used_bytes, "total_bytes": total_bytes}
    if visible.keys() != set(limits["gpu_uuids"]):
        raise ValueError("Reserved GPUs do not match the observed inventory")

    memory_events = {}
    for name, path in limits.get("memory_event_paths", {}).items():
        try:
            with Path(path).open(encoding="ascii") as handle:
                inode = os.fstat(handle.fileno()).st_ino
                counters = dict(row.split() for row in handle)
                memory_events[name] = {
                    "inode": inode,
                    "oom": int(counters["oom"]),
                    "oom_kill": int(counters["oom_kill"]),
                }
        except (OSError, KeyError, ValueError):
            memory_events[name] = None
    return {
        "source": "proc_meminfo_and_nvidia_smi",
        "host_available_bytes": mem_available,
        "device_memory": visible,
        "cgroup_memory_events": memory_events,
    }


def _read_resource_limits(value: dict) -> dict:
    uuids = value["gpu_uuids"]
    paths = value.get("memory_event_paths", {})
    if (
        not isinstance(uuids, list)
        or len(uuids) != 8
        or any(
            not isinstance(item, str) or not item.startswith("GPU-") for item in uuids
        )
        or len(set(uuids)) != 8
        or type(value["min_host_available_bytes"]) is not int
        or value["min_host_available_bytes"] < 0
        or type(value["max_device_used_fraction"]) not in (int, float)
        or not math.isfinite(value["max_device_used_fraction"])
        or not 0 < value["max_device_used_fraction"] <= 1
        or type(value["max_waiting_per_role"]) is not int
        or value["max_waiting_per_role"] < 0
        or not isinstance(paths, dict)
        or any(not Path(path).is_absolute() for path in paths.values())
    ):
        raise ValueError("Invalid observed resource budget in soak plan")
    return value


def _guards(snapshot: dict, current: dict, baseline: dict, limits: dict) -> list[str]:
    errors = []
    for role in _ROLE_NAMES:
        observed = snapshot["components"][role]["metrics"]
        if observed.get("healthy") is not True:
            errors.append(f"{role}_health_unavailable")
        waiting = observed.get("waiting")
        if waiting is None or waiting > limits["max_waiting_per_role"]:
            errors.append(f"{role}_wait_limit")
    if current["host_available_bytes"] < limits["min_host_available_bytes"]:
        errors.append("host_available_below_plan")
    if any(
        gpu["used_bytes"] / gpu["total_bytes"] > limits["max_device_used_fraction"]
        for gpu in current["device_memory"].values()
    ):
        errors.append("device_usage_above_plan")
    for name, now in current["cgroup_memory_events"].items():
        before = baseline["cgroup_memory_events"].get(name)
        if now is None or before is None or before["inode"] != now["inode"]:
            errors.append("cgroup_oom_accounting_unavailable")
            continue
        rises = [now[key] - before[key] for key in ("oom", "oom_kill")]
        if any(rise < 0 for rise in rises):
            errors.append("cgroup_oom_counter_reset")
        elif any(rises):
            errors.append("cgroup_oom_observed")
    return errors


def _zero_final_refs(sample: dict) -> bool:
    ownership = sample["components"].get("settlement", {})
    refs = ownership.get("metrics", {})
    return bool(ownership.get("epoch") and ownership.get("scope")) and all(
        refs.get(key) == 0 for key in ("held_references", "pending_retirement")
    )


def _counter_value(report: dict, component: str, metric: str):
    reading = report["components"][component]["metrics"][metric]["delta"]
    return reading.get("value") if reading["status"] == "observed" else None


async def _run_live(
    args,
    raw,
    workload: Workload,
    config: ProbeConfig,
    digest: str,
    *,
    workload_probe=None,
    max_attempts=None,
    peak_prompt_tokens=None,
    scope="observed_local_cold_only",
):
    token = os.environ.get(TOKEN_ENV)
    if not is_control_token(token):
        raise ValueError("Private local control token is required")
    api_key = os.environ.get("KITEPOND_GLM_PROBE_VLLM_API_KEY")
    concurrency = raw["concurrency"]
    limits = _read_resource_limits(raw["resource_limits"])
    nonce = secrets.token_hex(16)
    source_sha256 = sha256_file(args.sources)
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    _emit(
        args.output / "run-plan.json",
        {
            "plan": raw,
            "manifest_sha256": digest,
            "sources_file_sha256": source_sha256,
            "driver_sha256": sha256_file(Path(__file__)),
            "prompt_nonce": nonce,
            "run_scope": scope,
        },
    )

    async def capture():
        if sha256_file(args.sources) != source_sha256:
            raise ValueError("Telemetry source configuration changed during soak")
        return await asyncio.to_thread(capture_snapshot, args.sources, digest)

    async def resource_sample():
        return await asyncio.to_thread(_read_resources, limits)

    async with httpx.AsyncClient(
        timeout=config.timeout_seconds,
        trust_env=False,
        limits=httpx.Limits(max_connections=concurrency + 4),
    ) as client:

        async def request(item: Population, tokens: list[int]):
            if workload_probe is not None:
                return await workload_probe(client, item, tokens, token, api_key)
            return await run_probe(
                client,
                replace(config, output_tokens=item.output_tokens),
                tokens,
                token,
                api_key,
            )

        first_observation, first_resources = await asyncio.gather(
            capture(), resource_sample()
        )
        p_state, d_state = await _roles_status(client, config, token)
        _emit(args.output / "before.json", first_observation)
        _emit(args.output / "before-resources.json", first_resources)
        _emit(
            args.output / "before-native-status.json",
            {"prefill": p_state, "decode": d_state},
        )
        if (
            not _role_idle(first_observation)
            or not _no_native_holds(p_state, d_state)
            or p_state.get("registered_ranks") != 4
        ):
            raise ValueError("The current isolated P/D pair is not idle and enrolled")
        capacity = _check_capacity(
            p_state,
            d_state,
            concurrency,
            peak_prompt_tokens
            if peak_prompt_tokens is not None
            else max(item.prompt_tokens for item in workload.populations),
        )
        errors = _guards(first_observation, first_resources, first_resources, limits)
        if errors:
            raise ValueError(
                "The isolated pair starts outside the planned resource budget"
            )
        _emit(args.output / "admission.json", capacity)

        stop = asyncio.Event()
        finished = asyncio.Event()
        stats = RequestStats()
        initial_epochs = {
            role: _role_epoch(first_observation, role) for role in _ROLE_NAMES
        }
        guard_results = set()
        prior = first_observation

        async def monitor(stream):
            nonlocal prior
            try:
                while not finished.is_set():
                    with suppress(TimeoutError):
                        await asyncio.wait_for(finished.wait(), raw["sample_seconds"])
                    observation, resources = await asyncio.gather(
                        capture(), resource_sample()
                    )
                    if any(
                        _role_epoch(observation, role) != initial_epochs[role]
                        for role in _ROLE_NAMES
                    ):
                        guard_results.add("engine_restart")
                    guard_results.update(
                        _guards(observation, resources, first_resources, limits)
                    )
                    _line(
                        stream,
                        {
                            "snapshot": observation,
                            "resources": resources,
                            "interval_report": operational_report(
                                observation, prior, digest
                            ),
                            "client_counts": {
                                "started": stats.started,
                                "completed": stats.completed,
                                "failed": stats.failed,
                            },
                        },
                    )
                    prior = observation
                    if guard_results:
                        stop.set()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                guard_results.add(f"capture_failed:{type(exc).__name__}")
                stop.set()

        with (
            (args.output / _TRACE).open("x", encoding="utf-8") as records,
            (args.output / "trend.jsonl").open("x", encoding="utf-8") as trend,
        ):
            monitoring = asyncio.create_task(monitor(trend))
            try:
                active = await fill_workload(
                    records,
                    PromptStream(workload, nonce),
                    probe=request,
                    concurrency=concurrency,
                    seconds=raw["active_seconds"],
                    max_failures=raw["max_failures"],
                    stop=stop,
                    stats=stats,
                    max_attempts=max_attempts,
                )
            finally:
                finished.set()
                await monitoring

        drain_started = time.monotonic_ns()
        settled = False
        try:
            async with asyncio.timeout(raw["drain_timeout_seconds"]):
                while True:
                    p_state, d_state = await _roles_status(client, config, token)
                    final = await capture()
                    settled = _role_idle(final) and _no_native_holds(p_state, d_state)
                    if settled:
                        break
                    await asyncio.sleep(0.5)
        except (TimeoutError, httpx.HTTPError):
            final = await capture()
            settled = False
        _emit(args.output / "after.json", final)
        try:
            final_resources = await resource_sample()
            _emit(args.output / "after-resources.json", final_resources)
            guard_results.update(
                _guards(final, final_resources, first_resources, limits)
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            guard_results.add(f"capture_failed:{type(exc).__name__}")
        _emit(
            args.output / "after-native-status.json",
            {"prefill": p_state, "decode": d_state},
        )
    report = operational_report(final, first_observation, digest)
    _emit(args.output / "report.json", report)
    final_refs = _zero_final_refs(final)
    engine_input = _counter_value(report, "prefill", "logical_input_tokens_total")
    engine_output = _counter_value(report, "decode", "generated_tokens_total")
    prefix_reuse = _counter_value(report, "prefill", "reused_input_tokens_total")
    summary = {
        **active,
        "scope": scope,
        "started": stats.started,
        "completed": stats.completed,
        "failed": stats.failed,
        "failure_types": stats.failures,
        "peak_client_inflight": stats.peak_inflight,
        "completed_by_population": stats.populations,
        "native_import_and_lease_drained": settled,
        "owner_reference_counts_zero": final_refs,
        "drain_wait_seconds": round((time.monotonic_ns() - drain_started) / 1e9, 3),
        "resource_guard_errors": sorted(guard_results),
        "host_cgroup_oom_source": (
            "cgroup memory.events" if limits.get("memory_event_paths") else None
        ),
        "gpu_oom_source": None,
        "first_token_source": None,
        "first_token_status": "unavailable_in_nonstreaming_probe",
        "timing_sources": {
            "client_request": "whole private probe and response wall time",
            "model_phases": "native capture/report counters",
        },
        "prefill_api_prompt_tokens": stats.prefill_api_tokens,
        "prefill_engine_logical_tokens": engine_input,
        "output_api_tokens": stats.output_tokens,
        "decode_engine_output_tokens": engine_output,
        "native_output_counts_match": engine_output is not None
        and engine_output == stats.output_tokens,
        "prefill_prefix_reuse_tokens": prefix_reuse,
        "prefill_prefix_validation": (
            "unavailable"
            if prefix_reuse is None
            else "observed_zero"
            if prefix_reuse == 0
            else "observed_reuse"
        ),
        "release_qualification": (
            "not_evaluated_cold_only_workload"
            if workload_probe is None
            else "not_evaluated_native_lab_only"
        ),
    }
    _emit(args.output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "sources", "plan", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        _read_manifest(args.manifest)
        raw, workload = _read_plan(args.plan)
        config = ProbeConfig(**raw["probe"])
        if getattr(config, "capture_token_ids", False):
            raise ValueError(
                "Continuous workload retains usage, not generated token IDs"
            )
        sources = json.loads(args.sources.read_bytes())
        if (
            sources["roles"]["prefill"]["port"] != config.prefill_port
            or sources["roles"]["decode"]["port"] != config.decode_port
        ):
            raise ValueError(
                "Probe HTTP ports do not match the attributed telemetry roles"
            )
        summary = asyncio.run(
            _run_live(args, raw, workload, config, sha256_file(args.manifest))
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        httpx.HTTPError,
        subprocess.SubprocessError,
    ) as exc:
        parser.exit(1, f"soak driver could not finish ({type(exc).__name__})\n")
    print(json.dumps(summary, sort_keys=True))
    return (
        0
        if (
            summary["reached_active_deadline"]
            and not summary["stopped_by_guard"]
            and not summary["resource_guard_errors"]
            and summary["peak_client_inflight"] == raw["concurrency"]
            and summary["completed"] > 0
            and summary["native_import_and_lease_drained"]
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
