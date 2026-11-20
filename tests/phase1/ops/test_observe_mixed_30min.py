# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A cutoff receipt preserves separate evidence for accounting and drain."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from observe_mixed_30min import (
    CUTOFF_SCHEMA,
    METRICS,
    PHASES,
    SECOND,
    build,
    file_digest,
)

START = 9000 * SECOND


def store(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_rows(path, data):
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in data), encoding="utf-8"
    )


def turn(ready):
    return {
        "verified_ready_ranks": 4,
        "ready_tokens_per_rank": ready,
        "prefill_api_prompt_tokens": ready + 1,
        "output_tokens": 2,
    }


def one_row(i):
    phase = PHASES[i % len(PHASES)]
    t = [turn(10)]
    external = restore = 0
    chk = None
    if phase != "cold":
        external = 6
        t.append(turn(7))
    if phase == "nvme":
        restore, chk = 2, "a" * 64
    logical = sum(row["ready_tokens_per_rank"] for row in t)
    return {
        "ordinal": i,
        "started_ns": START + i * 40 * SECOND,
        "ended_ns": START + (i * 40 + 39) * SECOND,
        "request": {
            "phase": phase,
            "status": "completed",
            "settled_turns": t,
            "prefill_api_prompt_tokens": logical + len(t),
            "output_tokens": 2 * len(t) + restore,
            "native_nvme_restore_output_tokens": restore,
            "checkpoint_aggregate_sha256": chk,
            "prefill_source_delta": {
                "local_compute": logical - external,
                "local_cache_hit": 0,
                "external_kv_transfer": external,
            },
        },
    }


def running(count):
    return {"started": count + 1, "completed": count, "failed": 0}


def status():
    result = {}
    for role in METRICS:
        result[role] = {"stage": role, "active_leases": 0, "ready_leases": 0}
    result["prefill"]["registered_ranks"] = 4
    result["decode"]["scheduled_imports"] = 0
    result["decode"]["native_pool"] = {
        "active_grants": 0,
        "awaiting_settlement": 0,
        "grant_block_references": 0,
    }
    return result


class ObservedWindow(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.args = SimpleNamespace(
            mixed_dir=root,
            pair=root / "pair.json",
            manifest=root / "release.json",
            sources=root / "mixed.sources.json",
            cutoff=None,
        )
        store(self.args.pair, {"authority_engine_id": "D"})
        store(self.args.manifest, {"image": "candidate"})
        store(self.args.sources, {"same": "roles"})
        self.resources = {
            "host_available_bytes": 100,
            "device_memory": {
                f"GPU-{i}": {"used_bytes": 15, "total_bytes": 20} for i in range(8)
            },
            "cgroup_memory_events": {
                "candidate": {"inode": 123, "oom": 0, "oom_kill": 0}
            },
        }
        store(
            root / "run-plan.json",
            {
                "run_scope": "observed_local_native_lab_mixed",
                "manifest_sha256": file_digest(self.args.manifest),
                "sources_file_sha256": file_digest(self.args.sources),
                "plan": {
                    "active_seconds": 7200,
                    "concurrency": 1,
                    "sample_seconds": 15,
                    "max_failures": 0,
                    "resource_limits": {
                        "gpu_uuids": list(self.resources["device_memory"]),
                        "min_host_available_bytes": 50,
                        "max_device_used_fraction": 0.9,
                        "max_waiting_per_role": 4,
                        "memory_event_paths": {
                            "candidate": "/sys/fs/cgroup/memory.events"
                        },
                    },
                },
            },
        )
        self.ledger = [one_row(i) for i in range(45)]
        store(root / "before.json", self.capture(-1))
        store(root / "before-native-status.json", status())
        store(root / "before-resources.json", self.resources)
        write_rows(root / "request-results.jsonl", self.ledger)

    def capture(self, seconds, add_output=0):
        ended = [
            row["request"]
            for row in self.ledger
            if row["ended_ns"] <= START + seconds * SECOND
        ]
        counters = {
            "prefill": 55
            + sum(
                turn["ready_tokens_per_rank"]
                for row in ended
                for turn in row["settled_turns"]
            ),
            "decode": 14 + sum(row["output_tokens"] for row in ended) + add_output,
        }
        return {
            "schema": "kitepond.ops-snapshot.v1",
            "release_manifest_sha256": file_digest(self.args.manifest),
            "source": "observed",
            "deployment": "D",
            "sources_sha256": "sealed-sources",
            "collector": {
                "epoch": "host-boot",
                "monotonic_ns": START + seconds * SECOND,
            },
            "components": {
                role: {
                    "epoch": f"pid-{role}",
                    "scope": f"source-{role}",
                    "source_status": "observed",
                    "metrics": {
                        metric: counters[role],
                        "healthy": True,
                        "waiting": 0,
                        "running": 0,
                    },
                    "counter_series": {metric: {"0": counters[role]}},
                }
                for role, metric in METRICS.items()
            },
        }

    def trends(self, last=1800, *, guard_failure=False):
        rows = []
        for t in range(15, last + 1, 15):
            obs = self.capture(t)
            n = sum(row["ended_ns"] <= START + t * SECOND for row in self.ledger)
            res = self.resources
            if guard_failure and t == last:
                res = copy.deepcopy(res)
                res["cgroup_memory_events"]["candidate"]["oom"] = 1
            rows.append(
                {"snapshot": obs, "resources": res, "client_counts": running(n)}
            )
        write_rows(self.args.mixed_dir / "trend.jsonl", rows)

    def cutoff(self, *, tail_token=0, resources=True, change_epoch=False):
        end = self.capture(1805, add_output=tail_token)
        if change_epoch:
            end["components"]["decode"]["epoch"] = "restarted"
        self.args.cutoff = self.args.mixed_dir / "operator.json"
        drain = {
            "observed": True,
            "settled": True,
            "snapshot": end,
            "native": status(),
        }
        if resources:
            drain.update(resources=self.resources, guard_errors=[])
        store(
            self.args.cutoff,
            {
                "schema": CUTOFF_SCHEMA,
                "kind": "authorized_30_minute_limit",
                "manifest_sha256": file_digest(self.args.manifest),
                "pair_sha256": file_digest(self.args.pair),
                "run_plan_sha256": file_digest(self.args.mixed_dir / "run-plan.json"),
                "start_monotonic_ns": START,
                "target_monotonic_ns": START + 1800 * SECOND,
                "stop_monotonic_ns": START + 1800 * SECOND,
                "driver_exited": True,
                "finished_client_rows": len(self.ledger),
                "last_completion_monotonic_ns": self.ledger[-1]["ended_ns"],
                "drain": drain,
            },
        )

    def test_real_counter_point_at_30_minutes_has_no_fabricated_drain(self):
        self.trends()
        value = build(self.args)
        self.assertEqual(value["status"], "observed_complete")
        self.assertEqual(value["observed_window"]["seconds"], 1800)
        self.assertEqual(
            value["observed_window"]["completed_by_phase"], dict.fromkeys(PHASES, 15)
        )
        self.assertIsNone(value["cutoff_drain"]["native_import_and_lease_drained"])
        self.assertEqual(value["original_plan"]["active_seconds"], 7200)

    def test_fresh_cutoff_supplies_missing_common_point_and_narrow_drain(self):
        self.trends(1785)
        self.cutoff()
        value = build(self.args)
        self.assertEqual(value["status"], "observed_complete")
        self.assertEqual(
            value["observed_window"]["counter_snapshot_source"], "cutoff_snapshot"
        )
        self.assertTrue(value["cutoff_drain"]["native_import_and_lease_drained"])
        self.assertIsNone(value["cutoff_drain"]["owner_reference_counts_zero"])

    def test_unaccounted_native_token_and_missing_guards_never_qualify(self):
        self.trends(1785)
        for tail, resources in ((1, True), (0, False)):
            with self.subTest(tail=tail, resources=resources):
                self.cutoff(tail_token=tail, resources=resources)
                value = build(self.args)
                self.assertEqual(value["status"], "unreconciled")
                self.assertTrue(
                    value["cutoff_drain"]["native_import_and_lease_drained"]
                )
                self.assertEqual(value["operator_client_window"]["seconds"], 1800)

    def test_same_epoch_is_required_through_the_final_cutoff(self):
        self.trends(1785)
        self.cutoff(change_epoch=True)
        with self.assertRaisesRegex(ValueError, "epoch changed"):
            build(self.args)

    def test_observed_oom_is_preserved(self):
        self.trends(guard_failure=True)
        self.cutoff()
        value = build(self.args)
        self.assertEqual(value["status"], "unreconciled")
        self.assertEqual(value["resource_guard"]["errors"], ["cgroup_oom_observed"])


if __name__ == "__main__":
    unittest.main()
