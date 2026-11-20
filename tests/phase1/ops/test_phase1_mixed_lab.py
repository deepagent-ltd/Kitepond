# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage of evidence identity and phase accounting, no simulated lab PASS."""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from kitepond.ops.capture import capture_snapshot
from kitepond.ops.glm_cold_probe import ProbeConfig, ProbeFailure
from kitepond.ops.wheel import sha256_file
from phase1_mixed_lab import MixedWorkload, _capture_sources, _checked_warm, _eligible
from phase1_soak_driver import (
    Population,
    PromptStream,
    RequestStats,
    Workload,
    fill_workload,
)

MODEL = "nvidia/GLM-5.3-Flash-NVFP4"
POPULATION = Population("code", 4, 1025, 8)
SOURCES = {"local_compute": 0, "local_cache_hit": 0, "external_kv_transfer": 0}


def cold_turn(prompt):
    return {
        "status": "completed",
        "prompt_tokens": prompt,
        "prefill_api_prompt_tokens": prompt,
        "ready_tokens_per_rank": prompt - 1,
        "verified_ready_ranks": 4,
        "output_tokens": 8,
        "probe_wall_seconds": 9.8,
    }


def warm_r14_shape():
    return {
        "first_turn": cold_turn(1025),
        "second_turn": cold_turn(1290),
        "reused_prefix_tokens": 1032,
        "prefill_second_turn_source_tokens": {
            "local_compute": 257,
            "local_cache_hit": 0,
            "external_kv_transfer": 1032,
        },
        "settled_reverse_ranks": [0, 1, 2, 3],
        "prefill_released": True,
        "decode_released": True,
    }


def scenario():
    run = MixedWorkload.__new__(MixedWorkload)
    run.phase, run.expected_prefill = 0, 0
    run.counts = {"cold": 0, "warm": 0, "nvme": 0}
    run.plan = {"suffix_tokens": 257}
    run.decoder = object()
    run.config = ProbeConfig(
        18163, 18162, "http://127.0.0.1:18164", "test-native-p", MODEL
    )
    return run


class MixedLabFixtures(unittest.IsolatedAsyncioTestCase):
    def test_sources_bind_the_real_capture_contract_before_http(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            plan, release = runtime / "pair.json", runtime / "release.json"
            pair = {
                "authority_engine_id": "d-lab",
                "authority_boot_id": "d-boot",
                "mooncake_api": {"p_boot_id": "p-boot", "d_boot_id": "d-boot"},
            }
            plan.write_text(json.dumps(pair))
            release.write_text("{}")
            (runtime / "OWNERSHIP.json").write_text(
                json.dumps(
                    {
                        "plan_sha256": sha256_file(plan),
                        "engine_boot_id": pair["authority_boot_id"],
                        "gpu_count": 8,
                        "models_ready": True,
                    }
                )
            )
            for role, pid in (("prefill", 111), ("decode", 112)):
                (runtime / role).mkdir()
                (runtime / role / "PID").write_text(str(pid))
            args = SimpleNamespace(runtime=runtime, pair=plan, manifest=release)
            with patch("phase1_mixed_lab._process_start", return_value=23):
                settings = _capture_sources(args, pair)
            self.assertEqual(
                settings["roles"]["prefill"]["engine_epoch"], "p-boot/prefill"
            )
            self.assertEqual(
                settings["roles"]["decode"]["engine_epoch"], "d-boot/decode"
            )
            self.assertEqual(settings["deployment"], "d-lab")
            sources = runtime / "sources.json"
            sources.write_text(json.dumps(settings))
            with (
                patch(
                    "kitepond.ops.capture._capture_engine",
                    side_effect=RuntimeError("HTTP stage"),
                ),
                self.assertRaisesRegex(RuntimeError, "HTTP stage"),
            ):
                capture_snapshot(sources, sha256_file(release))

    async def test_c1_exits_after_exactly_one_actual_probe_result(self):
        now = asyncio.Event()
        stats = RequestStats()
        records = io.StringIO()
        stream = PromptStream(Workload(4, 100, 500, (POPULATION,)), "fixture")
        cold = AsyncMock(return_value=cold_turn(1025))
        result = await fill_workload(
            records,
            stream,
            probe=cold,
            concurrency=1,
            seconds=60,
            max_failures=0,
            stop=now,
            stats=stats,
            max_attempts=1,
        )
        cold.assert_awaited_once()
        self.assertEqual(len(records.getvalue().splitlines()), 1)
        self.assertTrue(not result["reached_active_deadline"] and stats.started == 1)

    def test_native_warm_requires_all_fences_and_matches_input_boundary(self):
        sample = warm_r14_shape()
        turns, shared = _checked_warm(sample)
        self.assertEqual((turns[1]["ready_tokens_per_rank"], shared), (1289, 1032))
        with self.assertRaisesRegex(ValueError, "fence"):
            _checked_warm(sample | {"settled_reverse_ranks": [0, 1, 2, 2]})
        with self.assertRaisesRegex(ValueError, "fence"):
            _checked_warm(sample | {"decode_released": False})
        with self.assertRaisesRegex(ValueError, "fence"):
            _checked_warm(sample | {"reused_prefix_tokens": 1024})
        with self.assertRaisesRegex(ValueError, "READY"):
            _checked_warm(
                sample | {"first_turn": cold_turn(0) | {"verified_ready_ranks": 3}}
            )

    async def test_cold_phase_rejects_cache_hits_and_prometheus_reset(self):
        for last in (
            SOURCES | {"local_compute": 1023, "local_cache_hit": 1},
            SOURCES | {"local_compute": 1},
        ):
            run = scenario()
            with (
                patch(
                    "phase1_mixed_lab.run_probe",
                    AsyncMock(return_value=cold_turn(1025)),
                ),
                patch(
                    "phase1_mixed_lab._prefill_counters",
                    AsyncMock(side_effect=[SOURCES | {"local_compute": 400}, last]),
                ),
                self.assertRaises(ProbeFailure) as error,
            ):
                await run.request(None, POPULATION, [1] * 1025, "auth", None)
            self.assertEqual(error.exception.phase, "lab_cold")
            self.assertEqual(run.counts, {"cold": 0, "warm": 0, "nvme": 0})

    async def test_exact_exclusive_phase_counts_keep_disk_decode_separate(self):
        run = scenario()
        warm = warm_r14_shape()
        warm_turns, shared = _checked_warm(warm)
        run._nvme_cycle = AsyncMock(return_value=(warm_turns, shared, 8, "aggregate"))
        final_p = SOURCES | {
            "local_compute": 1024 + 2 * (1024 + 257),
            "external_kv_transfer": 2064,
        }
        after_cold = SOURCES | {"local_compute": 1024}
        after_warm = SOURCES | {"local_compute": 2305, "external_kv_transfer": 1032}
        with (
            patch(
                "phase1_mixed_lab.run_probe", AsyncMock(return_value=cold_turn(1025))
            ),
            patch("phase1_mixed_lab.run_warm_probe", AsyncMock(return_value=warm)),
            patch(
                "phase1_mixed_lab._prefill_counters",
                AsyncMock(
                    side_effect=[
                        SOURCES,
                        after_cold,
                        after_cold,
                        after_warm,
                        after_warm,
                        final_p,
                    ]
                ),
            ),
        ):
            replies = [
                await run.request(None, POPULATION, [1] * 1025, "auth", None)
                for _ in range(3)
            ]
        self.assertEqual([row["phase"] for row in replies], ["cold", "warm", "nvme"])
        self.assertEqual([row["output_tokens"] for row in replies], [8, 16, 24])
        self.assertEqual(run.expected_prefill, 5650)
        self.assertEqual(replies[-1]["native_nvme_restore_output_tokens"], 8)
        self.assertEqual(run.counts, {"cold": 1, "warm": 1, "nvme": 1})

    async def test_failed_restore_keeps_private_oracle_and_structured_receipt(self):
        class FailedRestore(RuntimeError):
            receipt = {"phase": "exact_token_check", "error": "ValueError"}

        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            run = scenario()
            run.args = SimpleNamespace(output=work)
            run.phase = 9
            run.storage_root = work
            run.nvme_device = work.stat().st_dev
            run.nvme_generation = 0
            run.nvme_namespace = "private-test"
            run.plan["min_nvme_free_bytes"] = 1
            run.decoder = SimpleNamespace(
                engine_id="test",
                boot_id="boot",
                model_fingerprint="1" * 64,
                config_fingerprint="2" * 64,
            )
            run.nvme_key, run.nvme_token, run.catalog_key = b"key", "auth", b"expected"

            async def archive(client, fields, call):
                fields.artifact.write_text('{"synthetic":true}')
                return {"prefix_sha256": "digest"}

            entry = SimpleNamespace(
                source_boot_id="boot",
                target_boot_id="boot",
                accepted_prefix_sha256="digest",
            )
            catalog = SimpleNamespace(
                reopen=lambda scope: entry, verify=lambda value: True
            )
            run.nvme = SimpleNamespace(
                command=lambda *args: None,
                archive=archive,
                restore=AsyncMock(side_effect=FailedRestore),
                RestoreProbeFailure=FailedRestore,
            )
            with (
                patch(
                    "kitepond.integrations.vllm.glm_cold_catalog.GlmColdCatalog",
                    return_value=catalog,
                ),
                self.assertRaises(FailedRestore),
            ):
                await run._nvme_cycle(None, POPULATION, [1] * 1025, [2, 3])
            evidence = work / "nvme-9"
            self.assertEqual(
                json.loads((evidence / "restore-failure.json").read_bytes()),
                FailedRestore.receipt,
            )
            self.assertTrue((evidence / "private-oracle.json").is_file())
            self.assertTrue((evidence / "prompt.json").is_file())
            self.assertTrue((evidence / "suffix.json").is_file())
            self.assertEqual(evidence.stat().st_mode & 0o777, 0o700)

    def test_gate_needs_full_time_each_real_path_tokens_and_drain(self):
        run = SimpleNamespace(
            expected_prefill=5650, counts={"cold": 1, "warm": 1, "nvme": 1}
        )
        sample = {
            "completed": 3,
            "failed": 0,
            "stopped_by_guard": False,
            "resource_guard_errors": [],
            "peak_client_inflight": 1,
            "native_import_and_lease_drained": True,
            "native_output_counts_match": True,
            "prefill_engine_logical_tokens": 5650,
            "reached_active_deadline": True,
        }
        phases = tuple(run.counts)
        self.assertTrue(_eligible(sample, run, phases, complete_window=True))
        for bad in (
            {"reached_active_deadline": False},
            {"prefill_engine_logical_tokens": None},
            {"native_import_and_lease_drained": False},
            {"native_output_counts_match": False},
            {"resource_guard_errors": ["engine_restart"]},
        ):
            self.assertFalse(_eligible(sample | bad, run, phases, complete_window=True))
        run.counts["nvme"] = 0
        self.assertFalse(_eligible(sample, run, phases, complete_window=True))


if __name__ == "__main__":
    unittest.main()
