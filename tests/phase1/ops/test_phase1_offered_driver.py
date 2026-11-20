# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import io
import json
from collections import Counter

import pytest
from kitepond.ops.glm_cold_probe import ProbeFailure
from phase1_offered_driver import (
    COUNTERS,
    EXTERNAL,
    GENERATION,
    PREEMPT,
    PREFIX,
    PROMPT,
    ROLES,
    RUNNING,
    SUCCESS,
    WAITING,
    NativeQueue,
    capacity_gate,
    measure,
    reconcile,
)
from phase1_soak_driver import (
    Population,
    PromptStream,
    RequestStats,
    Workload,
    fill_workload,
)


def test_offered64_replenishes_each_completion_under_bounded_admission():
    async def check():
        stats = RequestStats()
        history = []
        active = set()
        attempts: Counter = Counter()
        record = io.StringIO()
        population = Population("short", 64, 12, 96)

        async def native(_item, tokens):
            key = tuple(tokens[:8])
            attempts[key] += 1
            if not history:
                history.append(("wait", key))
                raise ProbeFailure(
                    "D reserve", "request", "transfer", "D capacity wait"
                )
            active.add(key)
            assert len(active) <= 4
            history.append(("begin", key))
            await asyncio.sleep(0.015 if len(history) < 4 else 0.002)
            active.remove(key)
            history.append(("end", key))
            return {
                "status": "completed",
                "prefill_api_prompt_tokens": 11,
                "output_tokens": 96,
            }

        queue = NativeQueue(4, 10, native)
        await fill_workload(
            record,
            PromptStream(Workload(19, 271, 10999, (population,)), "test"),
            probe=queue.run,
            concurrency=64,
            seconds=10,
            max_failures=0,
            stop=asyncio.Event(),
            stats=stats,
            max_attempts=128,
        )
        rows = list(map(json.loads, record.getvalue().splitlines()))
        assert stats.peak_inflight == 64
        assert stats.completed == 128
        assert stats.output_tokens == 128 * 96
        assert queue.high_running == 4
        assert 60 <= queue.high_waiting <= 64
        assert queue.running == queue.waiting == 0
        assert len(attempts) == 128 and sum(attempts.values()) == 129
        assert sum(row["request"]["reserve_wait_retries"] for row in rows) == 1
        assert sum(action == "begin" for action, _ in history[:30]) > 4
        assert len({row["prompt_sha256"] for row in rows}) == 128
        by_offer = sorted(rows, key=lambda row: row["ordinal"])
        assert by_offer[64]["started_ns"] < max(
            row["ended_ns"] for row in by_offer[:64]
        )

    asyncio.run(check())


def test_native_failure_after_offer_is_never_retried():
    async def check():
        calls = 0

        async def native(_item, _tokens):
            nonlocal calls
            calls += 1
            raise ProbeFailure(
                "D completion", "request", "transfer", "deadline expired"
            )

        queue = NativeQueue(1, 5, native)
        with pytest.raises(ProbeFailure, match="D completion: deadline expired"):
            await queue.run(None, [])
        assert calls == 1
        assert queue.waiting == queue.running == 0

    asyncio.run(check())


def test_capacity_gate_uses_live_ram_and_usable_device_pages(tmp_path):
    import hashlib

    logs = tuple(tmp_path / f"{role}.log" for role in ROLES)
    for log in logs:
        log.write_text(log.name)
    cap = {
        "schema": "kitepond.glm_pd_capacity_startup.v1",
        "prefill": {},
        "decode": {},
    }
    for role, log, slots in zip(ROLES, logs, (4, 32), strict=True):
        cap[role].update(
            startup_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
            configured_scheduler_seqs=slots,
            gpu_headroom_percent=5,
            logical_host_page_tokens=1024,
            gpu_default_hot_rows_per_group=8704,
            gpu_minimum_aligned_hot_rows_per_group=8704,
        )
    d = cap["decode"]
    d.update(
        host_usable_blocks_per_rank=531,
        workloads_derived_not_measured=[
            dict(
                clients=32,
                active_tokens=4096,
                gpu_usable_blocks_needed_with_reserve=19327,
            )
        ],
        gpu_usable_device_blocks_per_rank=19327,
    )
    base = dict(stage="prefill", max_pending=32, active_leases=0, ready_leases=0)
    status = (
        {**base, "registered_ranks": 4},
        {
            **base,
            "stage": "decode",
            "scheduled_imports": 0,
            "native_pool": dict(
                block_tokens=1024,
                available_pages=531,
                active_grants=0,
                awaiting_settlement=0,
                grant_block_references=0,
            ),
        },
    )
    assert (
        capacity_gate(cap, logs, status, 4096)[
            "shared_RAM_pages_required_for_native_limit"
        ]
        == 129
    )
    d["gpu_usable_device_blocks_per_rank"] = 18832
    with pytest.raises(ValueError, match="lacks measured D pages"):
        capacity_gate(cap, logs, status, 4096)
    d["gpu_usable_device_blocks_per_rank"] = 19327
    status[1]["native_pool"]["available_pages"] = 128
    with pytest.raises(ValueError, match="shared RAM"):
        capacity_gate(cap, logs, status, 4096)


def test_prometheus_selection_and_cold_wire_token_equations():
    deltas = {
        "prefill": {
            PROMPT: 1798,
            GENERATION: 2,
            PREFIX: 0,
            EXTERNAL: 0,
            PREEMPT: 0,
            SUCCESS: 2,
        },
        "decode": {
            PROMPT: 1800,
            GENERATION: 192,
            PREFIX: 0,
            EXTERNAL: 1798,
            PREEMPT: 0,
            SUCCESS: 2,
        },
    }
    before = {role: dict.fromkeys((*COUNTERS, RUNNING, WAITING), 4) for role in ROLES}

    def encode(row):
        lines = []
        for name, count in row.items():
            labels = 'model_name="glm",engine="0"'
            if name == SUCCESS:
                labels += ',finished_reason="length"'
            lines.append(f"{name}{{{labels}}} {count}")
        return "\n".join(lines) + "\n"

    after = {
        role: measure(
            encode(
                {
                    name: count + deltas[role].get(name, 0)
                    for name, count in before[role].items()
                }
            ),
            "glm",
        )
        for role in ROLES
    }
    assert all(row["passed"] for row in reconcile(before, after, 2, 900, 96).values())
    after["decode"][EXTERNAL] -= 1
    assert not reconcile(before, after, 2, 900, 96)["decode"]["passed"]
    with pytest.raises(ValueError, match="unavailable"):
        measure(encode({PROMPT: 1}), "glm")
