# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import io
import json
import unittest

from kitepond.ops.glm_cold_probe import ProbeFailure
from phase1_soak_driver import (
    Population,
    PromptStream,
    RequestStats,
    Workload,
    _check_capacity,
    _guards,
    _no_native_holds,
    fill_workload,
)

_POPULATIONS = (
    Population("interactive", 2, 19, 4),
    Population("extension", 2, 256, 8),
)


def _workload():
    return Workload(19, 100, 400, _POPULATIONS)


def _reply(population):
    return {
        "status": "completed",
        "prefill_api_prompt_tokens": population.prompt_tokens,
        "output_tokens": population.output_tokens,
    }


class SoakPlanTests(unittest.TestCase):
    def test_spreads_sessions_and_preserves_deterministic_unique_prefixes(self):
        prompts = PromptStream(_workload(), "fixed-success-fixture")
        duplicate = PromptStream(_workload(), "fixed-success-fixture")
        alternate = PromptStream(_workload(), "separate-attempt")
        prefixes = set()
        labels = set()
        for _ in range(16):
            item, row, raw = prompts.take()
            self.assertEqual(duplicate.take(), (item, row, raw))
            self.assertNotEqual(alternate.take()[2][:8], raw[:8])
            self.assertEqual(len(raw), item.prompt_tokens)
            self.assertNotIn(tuple(raw[:8]), prefixes)
            self.assertTrue(all(100 <= token <= 400 for token in raw))
            prefixes.add(tuple(raw[:8]))
            labels.add((item.name, row["session"], row["turn"]))
        self.assertEqual(len(labels), 16)
        self.assertEqual({name for name, _, _ in labels}, {"interactive", "extension"})

    def test_rejects_unproven_admission_limits_and_insufficient_pages(self):
        prefill = {"max_pending": 4}
        decode = {
            "max_pending": 64,
            "native_pool": {"block_tokens": 1024, "available_pages": 128},
        }
        self.assertEqual(
            _check_capacity(prefill, decode, 1, 2049)[
                "decode_pages_required_for_prompt_reservations"
            ],
            3,
        )
        with self.assertRaisesRegex(ValueError, "admission"):
            _check_capacity(prefill, decode, 32, 2049)
        prefill["max_pending"] = 64
        with self.assertRaisesRegex(ValueError, "pool"):
            _check_capacity(prefill, decode, 64, 2049)
        decode["native_pool"]["available_pages"] = 192
        self.assertEqual(
            _check_capacity(prefill, decode, 64, 2049)[
                "decode_pages_required_for_prompt_reservations"
            ],
            192,
        )

    def test_requires_actual_native_grant_retirement(self):
        prefill = {"active_leases": 0, "ready_leases": 0}
        decode = {
            "active_leases": 0,
            "ready_leases": 0,
            "scheduled_imports": 0,
            "native_pool": {
                "active_grants": 0,
                "awaiting_settlement": 0,
                "grant_block_references": 0,
            },
        }
        self.assertTrue(_no_native_holds(prefill, decode))
        decode["native_pool"]["awaiting_settlement"] = 1
        self.assertFalse(_no_native_holds(prefill, decode))
        decode["native_pool"]["awaiting_settlement"] = None
        self.assertFalse(_no_native_holds(prefill, decode))

    def test_measured_oom_positive_is_distinct_from_missing_or_reset(self):
        snapshot = {
            "components": {
                role: {"metrics": {"healthy": True, "waiting": 0}}
                for role in ("prefill", "decode")
            }
        }
        limits = {
            "max_waiting_per_role": 2,
            "min_host_available_bytes": 10,
            "max_device_used_fraction": 0.95,
        }

        def reading(inode, value):
            return {
                "host_available_bytes": 100,
                "device_memory": {"GPU-fixture": {"used_bytes": 8, "total_bytes": 10}},
                "cgroup_memory_events": {
                    "pair": {"inode": inode, "oom": value, "oom_kill": 0}
                },
            }

        baseline = reading(4, 2)
        self.assertEqual(_guards(snapshot, baseline, baseline, limits), [])
        self.assertEqual(
            _guards(snapshot, reading(4, 3), baseline, limits), ["cgroup_oom_observed"]
        )
        self.assertEqual(
            _guards(snapshot, reading(4, 1), baseline, limits),
            ["cgroup_oom_counter_reset"],
        )
        self.assertEqual(
            _guards(snapshot, reading(5, 3), baseline, limits),
            ["cgroup_oom_accounting_unavailable"],
        )


class WorkloadLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_replaces_same_slot_without_waiting_for_a_batch(self):
        starts = []
        first_started = asyncio.Event()
        third_started = asyncio.Event()
        await_short = asyncio.Event()
        stop = asyncio.Event()
        stats = RequestStats()
        record = io.StringIO()

        async def probe(item, tokens):
            number = len(starts)
            starts.append(tokens)
            if number == 0:
                first_started.set()
                await third_started.wait()
            elif number == 1:
                await first_started.wait()
                await_short.set()
            else:
                self.assertTrue(await_short.is_set())
                third_started.set()
                stop.set()
            return _reply(item)

        result = await asyncio.wait_for(
            fill_workload(
                record,
                PromptStream(_workload(), "fixture"),
                probe=probe,
                concurrency=2,
                seconds=10,
                max_failures=0,
                stop=stop,
                stats=stats,
            ),
            1,
        )
        records = [json.loads(row) for row in record.getvalue().splitlines()]
        self.assertEqual(stats.peak_inflight, 2)
        self.assertEqual(stats.started, 3)
        self.assertEqual(stats.completed, 3)
        self.assertEqual(stats.failed, 0)
        first = next(row for row in records if row["ordinal"] == 0)
        third = next(row for row in records if row["ordinal"] == 2)
        self.assertLess(third["started_ns"], first["ended_ns"])
        self.assertFalse(result["reached_active_deadline"])
        self.assertTrue(result["stopped_by_guard"])

    async def test_records_native_failure_without_automatic_replay(self):
        stats = RequestStats()
        record = io.StringIO()
        stop = asyncio.Event()

        async def probe(item, tokens):
            raise ProbeFailure(
                "D reserve", "fixture-id", "fixture-xfer", "D capacity wait"
            )

        await fill_workload(
            record,
            PromptStream(_workload(), "fixture"),
            probe=probe,
            concurrency=1,
            seconds=10,
            max_failures=0,
            stop=stop,
            stats=stats,
        )
        self.assertEqual(stats.started, 1)
        self.assertEqual(stats.completed, 0)
        self.assertEqual(stats.failures, {"D reserve / D capacity wait": 1})
        self.assertEqual(
            json.loads(record.getvalue())["request"]["request_id"], "fixture-id"
        )
