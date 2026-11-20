# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run cold C32/C64 offered load on an isolated native 4P4D launch."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import secrets
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import httpx
from kitepond.integrations.vllm.cold_prefill_http import TOKEN_ENV, is_control_token
from kitepond.ops.glm_cold_probe import ProbeConfig, ProbeFailure, run_probe
from kitepond.ops.wheel import sha256_file
from phase1_soak_driver import (
    Population,
    PromptStream,
    RequestStats,
    Workload,
    _emit,
    _line,
    _no_native_holds,
    _roles_status,
    fill_workload,
)
from prometheus_client.parser import text_string_to_metric_families

ROLES = ("prefill", "decode")
SLOTS = {"prefill": 4, "decode": 32}
ADMISSION_LIMIT = 32
CHOICES = {"short": (900, 96), "4k": (3965, 128)}
HOT = {"device_buffer_size": 8704, "eager_host_mirror": True}
D_OPTIONS = {"--max-num-seqs": "32", "--gpu-memory-utilization": "0.88"}
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
PROMPT = "vllm:prompt_tokens_total"
GENERATION = "vllm:generation_tokens_total"
PREFIX = "vllm:prefix_cache_hits_total"
EXTERNAL = "vllm:external_prefix_cache_hits_total"
PREEMPT = "vllm:num_preemptions_total"
SUCCESS = "vllm:request_success_total"
COUNTERS = (PROMPT, GENERATION, PREFIX, EXTERNAL, PREEMPT, SUCCESS)


def _require(valid, message):
    if not valid:
        raise ValueError(message)


def _value(argv, flag):
    _require(argv.count(flag) == 1, f"Expected one {flag}")
    return argv[argv.index(flag) + 1]


def _set(argv, flag, value):
    _value(argv, flag)
    argv[argv.index(flag) + 1] = value


def _encoding(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _role(pair, role, slots):
    entry = pair["roles"][role]
    argv = entry["argv"]
    _require(
        entry["host_gpu_indices"]
        == ([4, 5, 6, 7] if role == "prefill" else [0, 1, 2, 3])
        and all(
            _value(argv, name) == value
            for name, value in {
                "--host": "127.0.0.1",
                "--tensor-parallel-size": "4",
                "--decode-context-parallel-size": "4",
                "--pipeline-parallel-size": "1",
                "--data-parallel-size": "1",
                "--cp-kv-cache-interleave-size": "4",
                "--block-size": "256",
                "--kv-cache-dtype": "fp8_ds_mla",
                "--max-model-len": "8192",
                "--max-num-batched-tokens": "4096",
                "--max-num-seqs": str(slots),
            }.items()
        )
        and all(
            flag in argv
            for flag in (
                "--enable-expert-parallel",
                "--enforce-eager",
                "--no-enable-prefix-caching",
                "--no-async-scheduling",
                "--enable-chunked-prefill",
            )
        )
        and json.loads(_value(argv, "--speculative-config"))
        == {"method": "mtp", "num_speculative_tokens": 3},
        f"Unexpected native {role} capacity topology",
    )
    extra = json.loads(_value(argv, "--additional-config"))
    _require(
        extra["kitepond"]["stage"] == role
        and extra["kitepond"]["boot_id"] == pair["authority_boot_id"]
        and extra["kitepond_glm_cold_prefill"]["arm_timeout_seconds"] == 60
        and not set(extra).intersection(
            {
                "kitepond_glm_cold",
                "kitepond_glm_nvme_control",
                "kitepond_glm_shared_mla_preflight",
            }
        ),
        f"Unexpected native {role} cold-only configuration",
    )
    return argv, extra


def render(pair):
    """Derive only an isolated load-test plan, with no executable rewrite."""
    _require(pair["mode"] == "native-warm-return", "Use the sealed native warm pair")
    for role in ROLES:
        argv, extra = _role(pair, role, 4)
        _require(
            extra["kitepond_glm_cold_prefill"]["max_pending"] == 4,
            "Start from the C4 control baseline",
        )
        extra["kitepond_glm_cold_prefill"]["max_pending"] = ADMISSION_LIMIT
        _set(argv, "--additional-config", _encoding(extra))
        if role == "decode":
            for flag, value in D_OPTIONS.items():
                _set(argv, flag, value)
            _require("--attention-config" not in argv, "Expected default D hot rows")
            argv.extend(("--attention-config", _encoding({"hisparse_config": HOT})))
    return pair


def render_launcher(source):
    """Keep the isolated pair startup/ownership gate and replace only its trial."""
    begin = source.index("def run_requests():")
    end = source.index("\n\nchildren = {}", begin)
    trial = """def run_requests():
    env = dict(os.environ)
    env.update(role_env)
    env.update({"CUDA_VISIBLE_DEVICES": "", "VLLM_TARGET_DEVICE": "cpu",
                "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    launch = plan_path.parent / "launch"
    cap_logs = [state / f"{role}-startup.log" for role in ("prefill", "decode")]
    for role, target in zip(("prefill", "decode"), cap_logs, strict=True):
        shutil.copyfile(state / role / "server.log", target)
    capacity = state / "STARTUP_CAPACITY.json"
    with capacity.open("xb") as evidence:
        subprocess.run(
            [sys.executable, "-B", str(launch / "GLM_PD_CAPACITY_20261119.py"),
             *(str(item) for item in cap_logs)],
            env=env, stdout=evidence, timeout=20, check=True,
        )
    for offered in (32, 64):
        with (state / f"C{offered}.stdout").open("xb") as output, (
            state / f"C{offered}.stderr"
        ).open("xb") as diagnostics:
            result = subprocess.run(
                [sys.executable, "-B", str(launch / "phase1_offered_driver.py"), "run",
                 "--pair", str(plan_path), "--capacity", str(capacity),
                 "--prefill-startup", str(cap_logs[0]),
                 "--decode-startup", str(cap_logs[1]),
                 "--offered", str(offered), "--shape", "short",
                 "--output", str(state / f"C{offered}")],
                env=env, stdout=output, stderr=diagnostics, timeout=430,
            )
        if result.returncode:
            raise RuntimeError(f"C{offered} offered workload or cleanup failed")
    print("Native C32/C64 offered workloads completed and drained", flush=True)
"""
    updated = source[:begin] + trial + source[end:]
    compile(updated, "RUN_PAIR_INSIDE.py", "exec")
    return updated


def _config(pair):
    role_configs = {}
    for role in ROLES:
        argv, extra = _role(pair, role, SLOTS[role])
        _require(
            extra["kitepond_glm_cold_prefill"]["max_pending"] == ADMISSION_LIMIT,
            "Unexpected sealed native grant limit",
        )
        role_configs[role] = (argv, extra)
    p_argv, _ = role_configs["prefill"]
    d_argv, _ = role_configs["decode"]
    _require(
        json.loads(_value(d_argv, "--attention-config")) == {"hisparse_config": HOT}
        and _value(d_argv, "--gpu-memory-utilization")
        == D_OPTIONS["--gpu-memory-utilization"]
        and _value(p_argv, "--served-model-name")
        == _value(d_argv, "--served-model-name"),
        "Unexpected candidate memory profile",
    )
    return ProbeConfig(
        prefill_port=int(_value(p_argv, "--port")),
        decode_port=int(_value(d_argv, "--port")),
        prefill_bootstrap=(
            "http://127.0.0.1:"
            + str(pair["roles"]["prefill"]["mooncake_bootstrap_port"])
        ),
        prefill_engine_id=json.loads(_value(p_argv, "--kv-transfer-config"))[
            "engine_id"
        ],
        model=_value(p_argv, "--served-model-name"),
        timeout_seconds=55,
        seed=20261119,
    )


def capacity_gate(capacity, startup_logs, statuses, active_tokens):
    _require(
        capacity["schema"] == "kitepond.glm_pd_capacity_startup.v1",
        "Use the native startup capacity calculator",
    )
    for role, log in zip(ROLES, startup_logs, strict=True):
        row = capacity[role]
        _require(
            sha256_file(log) == row["startup_log_sha256"]
            and row["configured_scheduler_seqs"] == SLOTS[role]
            and row["gpu_headroom_percent"] == 5
            and row["logical_host_page_tokens"] == 1024
            and row["gpu_minimum_aligned_hot_rows_per_group"] == 8704,
            f"{role} startup does not prove the sealed capacity",
        )
    before_p, before_d = statuses
    pool = before_d["native_pool"]
    d = capacity["decode"]
    device_count = d["gpu_usable_device_blocks_per_rank"]
    (rule,) = (
        entry
        for entry in d["workloads_derived_not_measured"]
        if entry["clients"] == ADMISSION_LIMIT and entry["active_tokens"] == 4096
    )
    host_pages = pool["available_pages"]
    required = ADMISSION_LIMIT * math.ceil(active_tokens / 1024) + 1
    _require(
        before_p.get("registered_ranks") == 4
        and _no_native_holds(*statuses)
        and all(state["max_pending"] == ADMISSION_LIMIT for state in statuses)
        and pool["block_tokens"] == 1024
        and host_pages == d["host_usable_blocks_per_rank"] == 531
        and host_pages >= required
        and d["gpu_default_hot_rows_per_group"] == 8704
        and device_count >= max(19327, rule["gpu_usable_blocks_needed_with_reserve"]),
        "C32 admission lacks measured D pages, shared RAM or clean owner state",
    )
    return {
        "D_allocatable_device_pages_per_rank": device_count,
        "D_device_pages_required_for_C32_4Ki": max(
            19327, rule["gpu_usable_blocks_needed_with_reserve"]
        ),
        "shared_RAM_pages_before": host_pages,
        "shared_RAM_pages_required_for_native_limit": required,
        "native_admission_limit": ADMISSION_LIMIT,
    }


class NativeQueue:
    """Only a bounded pre-reserve queue can retry an explicit capacity Wait."""

    def __init__(self, limit, queue_seconds, probe, stop=None):
        self._permits = asyncio.BoundedSemaphore(limit)
        self._queue_seconds = queue_seconds
        self._probe = probe
        self._stop = stop
        self.running = self.waiting = 0
        self.high_running = self.high_waiting = 0

    async def run(self, item, tokens):
        started = time.monotonic()
        deadline = started + self._queue_seconds
        attempts = 0
        waiting = 0.0
        while True:
            blocked = self._permits.locked()
            self.waiting += blocked
            self.high_waiting = max(self.high_waiting, self.waiting)
            queued_at = time.monotonic()
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                async with asyncio.timeout(remaining):
                    await self._permits.acquire()
            except TimeoutError:
                raise ProbeFailure(
                    "client queue", "", "", "pre-reserve queue deadline expired"
                ) from None
            finally:
                self.waiting -= blocked
                waiting += time.monotonic() - queued_at

            self.running += 1
            self.high_running = max(self.high_running, self.running)
            attempted_at = time.monotonic()
            try:
                if self._stop is not None and self._stop.is_set():
                    raise ProbeFailure(
                        "client queue", "", "", "run stopped before native reserve"
                    )
                result = await self._probe(item, tokens)
            except ProbeFailure as error:
                if error.phase != "D reserve" or error.reason != "D capacity wait":
                    raise
                attempts += 1
                waiting += time.monotonic() - attempted_at
                if deadline <= time.monotonic():
                    raise
            else:
                return {
                    **result,
                    "pre_reserve_wait_seconds": round(waiting, 3),
                    "reserve_wait_retries": attempts,
                }
            finally:
                self.running -= 1
                self._permits.release()
            backoff_started = time.monotonic()
            await asyncio.sleep(0.1)
            waiting += time.monotonic() - backoff_started


def measure(text, model):
    expected = {name: 0 for name in (*COUNTERS, RUNNING, WAITING)}
    seen = set()
    matches = set()
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            name = sample.name
            if name not in expected or sample.labels.get("model_name") != model:
                continue
            labels = {"model_name", "engine"}
            if name == SUCCESS:
                labels.add("finished_reason")
            row = (name, tuple(sorted(sample.labels.items())))
            n = sample.value
            _require(
                row not in seen
                and sample.labels.keys() == labels
                and sample.labels["engine"] == "0"
                and math.isfinite(n)
                and n >= 0
                and n.is_integer(),
                "Invalid or ambiguous Prometheus series for the pinned engine",
            )
            seen.add(row)
            matches.add(name)
            expected[name] += int(n)
    _require(matches == expected.keys(), "Required engine counters are unavailable")
    return expected


def reconcile(before, after, completed, prompt, output):
    expected = {
        "prefill": {
            PROMPT: completed * (prompt - 1),
            GENERATION: completed,
            EXTERNAL: 0,
        },
        "decode": {
            PROMPT: completed * prompt,
            GENERATION: completed * output,
            EXTERNAL: completed * (prompt - 1),
        },
    }
    deltas = {}
    for role in ROLES:
        wanted = {**expected[role], PREFIX: 0, PREEMPT: 0, SUCCESS: completed}
        measured = {name: after[role][name] - before[role][name] for name in wanted}
        deltas[role] = {
            "observed": measured,
            "expected": wanted,
            "passed": measured == wanted,
        }
    return deltas


async def run(args, pair):
    token = os.environ.get(TOKEN_ENV)
    _require(is_control_token(token), "Provide the private local control token")
    config = _config(pair)
    prompt, output = CHOICES[args.shape]
    workload = Workload(
        config.seed, 271, 10999, (Population(args.shape, args.offered, prompt, output),)
    )
    nonce = secrets.token_hex(16)
    output_path = args.output
    output_path.mkdir(mode=0o700)
    _emit(
        output_path / "run.json",
        {
            "pair_sha256": sha256_file(args.pair),
            "capacity_sha256": sha256_file(args.capacity),
            "shape": args.shape,
            "offered_clients": args.offered,
            "admission_limit": ADMISSION_LIMIT,
            "requests": args.offered * args.rounds,
            "prompt": prompt,
            "output": output,
            "seed": config.seed,
            "prompt_nonce": nonce,
        },
    )
    peaks = {role: {RUNNING: 0, WAITING: 0} for role in ROLES}
    finished = asyncio.Event()
    stop = asyncio.Event()
    errors = []
    stats = RequestStats()

    async with httpx.AsyncClient(
        trust_env=False,
        timeout=config.timeout_seconds,
        limits=httpx.Limits(max_connections=ADMISSION_LIMIT + 8),
    ) as client:

        async def scrape():
            responses = await asyncio.gather(
                *(
                    client.get(f"http://127.0.0.1:{port}/metrics", timeout=5)
                    for port in (config.prefill_port, config.decode_port)
                )
            )
            for response in responses:
                response.raise_for_status()
            return {
                role: measure(response.text, config.model)
                for role, response in zip(ROLES, responses, strict=True)
            }

        async def state():
            return await _roles_status(client, config, token)

        async def native(item, tokens):
            return await run_probe(
                client,
                replace(config, output_tokens=item.output_tokens),
                tokens,
                token,
                os.environ.get("KITEPOND_GLM_PROBE_VLLM_API_KEY"),
            )

        queue = NativeQueue(ADMISSION_LIMIT, args.queue_seconds, native, stop)
        before, before_state = await asyncio.gather(scrape(), state())
        _require(
            all(before[role][key] == 0 for role in ROLES for key in (RUNNING, WAITING)),
            "The isolated pair is not idle",
        )
        capacity = capacity_gate(
            json.loads(args.capacity.read_bytes()),
            (args.prefill_startup, args.decode_startup),
            before_state,
            prompt + output + 3,
        )
        _emit(
            output_path / "before.json",
            {"metrics": before, "status": before_state, "capacity": capacity},
        )

        async def monitor(samples):
            try:
                while not finished.is_set():
                    sample, statuses = await asyncio.gather(scrape(), state())
                    for role in ROLES:
                        for key in (RUNNING, WAITING):
                            peaks[role][key] = max(peaks[role][key], sample[role][key])
                    if any(
                        sample[role][RUNNING] > SLOTS[role]
                        or sample[role][WAITING] > ADMISSION_LIMIT
                        or sample[role][PREEMPT] > before[role][PREEMPT]
                        for role in ROLES
                    ):
                        errors.append("engine_running_waiting_or_preempt_guard")
                        stop.set()
                    _line(
                        samples,
                        {
                            "monotonic_ns": time.monotonic_ns(),
                            "metrics": sample,
                            "native_status": statuses,
                            "waiting_before_reserve": queue.waiting,
                            "holding_native_permit": queue.running,
                        },
                    )
                    with suppress(TimeoutError):
                        await asyncio.wait_for(finished.wait(), args.sample_seconds)
            except (ValueError, KeyError, httpx.HTTPError) as error:
                errors.append(type(error).__name__)
                stop.set()

        with (
            (output_path / "samples.jsonl").open("x") as samples,
            (output_path / "requests.jsonl").open("x") as requests,
        ):
            monitor_task = asyncio.create_task(monitor(samples))
            try:
                active = await fill_workload(
                    requests,
                    PromptStream(workload, nonce),
                    probe=queue.run,
                    concurrency=args.offered,
                    seconds=args.launch_seconds,
                    max_failures=0,
                    stop=stop,
                    stats=stats,
                    max_attempts=args.offered * args.rounds,
                )
            finally:
                finished.set()
                await monitor_task

        drained = False
        after, final_state = before, before_state
        try:
            async with asyncio.timeout(60):
                while True:
                    after, final_state = await asyncio.gather(scrape(), state())
                    idle = all(
                        after[role][key] == 0
                        for role in ROLES
                        for key in (RUNNING, WAITING)
                    )
                    drained = (
                        idle
                        and _no_native_holds(*final_state)
                        and final_state[1]["native_pool"]["available_pages"]
                        == capacity["shared_RAM_pages_before"]
                        and all(
                            after[role][SUCCESS] - before[role][SUCCESS]
                            >= stats.completed
                            for role in ROLES
                        )
                    )
                    if drained:
                        break
                    await asyncio.sleep(0.5)
        except (TimeoutError, httpx.HTTPError, ValueError, KeyError) as error:
            errors.append(type(error).__name__)
        _emit(output_path / "after.json", {"metrics": after, "status": final_state})
    totals = reconcile(before, after, stats.completed, prompt, output)
    passed = (
        not errors
        and not stats.failed
        and drained
        and stats.completed == args.offered * args.rounds
        and stats.peak_inflight == args.offered
        and stats.prefill_api_tokens == stats.completed * (prompt - 1)
        and stats.output_tokens == stats.completed * output
        and all(row["passed"] for row in totals.values())
        and all(0 < peaks[role][RUNNING] <= SLOTS[role] for role in ROLES)
    )
    elapsed = (
        active["requests_drained_monotonic_ns"] - active["start_monotonic_ns"]
    ) / 1e9
    result = {
        "passed": passed,
        "offered_peak": stats.peak_inflight,
        "native_permits_peak": queue.high_running,
        "client_queue_depth_peak": queue.high_waiting,
        "engine_gauge_sample_interval_seconds": args.sample_seconds,
        "engine_observed_peaks": peaks,
        "completed": stats.completed,
        "failed": stats.failed,
        "failure_types": stats.failures,
        "monitor_or_drain_errors": errors,
        "metrics_deltas": totals,
        "owners_and_shared_RAM_drained": drained,
        "whole_request_wall_seconds": elapsed,
        "whole_request_output_tokens_per_second": stats.output_tokens / elapsed,
        "steady_decode_tps": None,
        "native_32_running_observed": peaks["decode"][RUNNING] == ADMISSION_LIMIT,
    }
    _emit(output_path / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    render_cli = sub.add_parser("render")
    live_cli = sub.add_parser("run")
    render_cli.add_argument("--launch-source", type=Path)
    render_cli.add_argument("--launch-output", type=Path)
    for command in (render_cli, live_cli):
        command.add_argument("--pair", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    for name in ("capacity", "prefill-startup", "decode-startup"):
        live_cli.add_argument(f"--{name}", type=Path, required=True)
    live_cli.add_argument("--offered", type=int, choices=(32, 64), required=True)
    live_cli.add_argument("--shape", choices=CHOICES, default="short")
    live_cli.add_argument("--rounds", type=int, choices=(2, 3, 4), default=3)
    live_cli.add_argument(
        "--queue-seconds", type=int, choices=range(10, 121), default=60
    )
    live_cli.add_argument(
        "--launch-seconds", type=int, choices=range(60, 601), default=240
    )
    live_cli.add_argument(
        "--sample-seconds", type=float, choices=(0.2, 0.5, 1), default=0.2
    )
    args = parser.parse_args()
    try:
        pair = json.loads(args.pair.read_bytes())
        if args.command == "render":
            rendered = render(pair)
            _config(rendered)
            _require(
                bool(args.launch_source) == bool(args.launch_output),
                "Provide both launcher paths or neither",
            )
            launch = (
                render_launcher(args.launch_source.read_text())
                if args.launch_source
                else None
            )
            _emit(args.output, rendered)
            if launch is not None:
                with args.launch_output.open("x") as target:
                    target.write(launch)
            return 0
        result = asyncio.run(run(args, pair))
        print(json.dumps(result))
        return 0 if result["passed"] else 1
    except (OSError, KeyError, TypeError, ValueError, httpx.HTTPError) as error:
        parser.exit(1, f"Offered-capacity check failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
