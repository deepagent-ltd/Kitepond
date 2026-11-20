# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify archived metric interpretation across missing data and boot boundaries."""

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from kitepond.ops.cli import main
from kitepond.ops.report import operational_report

ARTIFACT = "a" * 64


def snapshot() -> dict:
    # Synthetic values that do not bind historical payloads to a new artifact.
    return {
        "schema": "kitepond.ops-snapshot.v1",
        "release_manifest_sha256": ARTIFACT,
        "source": "synthetic",
        "deployment": "fixture-deployment",
        "collector": {"epoch": "monitor-1", "monotonic_ns": 100},
        "components": {
            "prefill": {
                "epoch": "prefill-1",
                "metrics": {
                    "healthy": True,
                    "running": 1,
                    "waiting": 0,
                    "logical_input_tokens_total": 15,
                    "computed_input_tokens_total": 12,
                    "reused_input_tokens_total": 3,
                    "native_compute_seconds_total": 0.125,
                },
            },
            "decode": {
                "epoch": "decode-1",
                "metrics": {
                    "healthy": True,
                    "running": 0,
                    "waiting": 0,
                    "generated_tokens_total": 30,
                    "output_interval_seconds_total": 0.5,
                },
            },
            "handoff": {"epoch": "relay-1", "metrics": {"ready_events_total": 2}},
            "settlement": {
                "epoch": "owner-1",
                "scope": "fixture-namespace",
                "metrics": {
                    "held_references": 0,
                    "pending_retirement": 0,
                    "retired_objects_total": 7,
                    "closed_checkpoints_total": 10,
                },
            },
            "storage": {
                "epoch": "storage-1",
                "scope": "fixture-namespace",
                "metrics": {
                    "ram_resident_bytes": 0,
                    "nvme_payload_bytes": 256,
                    "nvme_written_bytes_total": 256,
                    "nvme_read_bytes_total": 128,
                },
            },
        },
    }


def delta(report, role, metric):
    return report["components"][role]["metrics"][metric]["delta"]


class OperationalReportTests(unittest.TestCase):
    def test_zero_and_unhealthy_are_observed_while_absent_metrics_stay_missing(self):
        current = snapshot()
        current["components"]["decode"]["metrics"]["healthy"] = False
        del current["components"]["storage"]["scope"]

        report = operational_report(current, None, ARTIFACT)

        decode = report["components"]["decode"]["metrics"]
        self.assertEqual(decode["waiting"], {"status": "observed", "value": 0})
        self.assertEqual(decode["healthy"], {"status": "observed", "value": False})
        self.assertEqual(
            report["components"]["handoff"]["metrics"]["payload_transfer_bytes_total"],
            {"status": "missing", "delta": {"status": "baseline_missing"}},
        )
        gates = report["gate_evidence"]
        self.assertEqual(gates["role_health"], "unhealthy")
        self.assertEqual(gates["storage"], "scope_missing")
        self.assertEqual(gates["checkpoint_closure"], "observed")
        self.assertEqual(gates["release_qualification"], "not_evaluated")
        self.assertEqual(report["source"], "synthetic")

    def test_same_epoch_emits_native_phase_and_scoped_deltas(self):
        previous = snapshot()
        current = copy.deepcopy(previous)
        current["collector"]["monotonic_ns"] = 600
        current["components"]["prefill"]["metrics"]["native_compute_seconds_total"] = (
            1.125
        )
        current["components"]["decode"]["metrics"].update(
            generated_tokens_total=46, output_interval_seconds_total=1.0
        )
        current["components"]["settlement"]["metrics"]["closed_checkpoints_total"] = 11

        report = operational_report(current, previous, ARTIFACT)

        self.assertEqual(report["interval"], {"status": "observed", "elapsed_ns": 500})
        self.assertEqual(
            delta(report, "prefill", "native_compute_seconds_total"),
            {"status": "observed", "value": 1.0},
        )
        self.assertEqual(
            delta(report, "decode", "generated_tokens_total"),
            {"status": "observed", "value": 16},
        )
        self.assertEqual(
            delta(report, "decode", "output_interval_seconds_total"),
            {"status": "observed", "value": 0.5},
        )
        self.assertEqual(
            delta(report, "settlement", "closed_checkpoints_total"),
            {"status": "observed", "value": 1},
        )
        self.assertEqual(
            delta(report, "storage", "nvme_written_bytes_total"),
            {"status": "observed", "value": 0},
        )

    def test_separates_component_restart_scope_change_and_counter_reset(self):
        previous = snapshot()
        current = copy.deepcopy(previous)
        current["collector"]["monotonic_ns"] += 1
        current["components"]["decode"]["epoch"] = "decode-2"
        current["components"]["storage"]["scope"] = "another-namespace"
        current["components"]["settlement"]["metrics"]["closed_checkpoints_total"] = 2

        report = operational_report(current, previous, ARTIFACT)

        self.assertEqual(
            delta(report, "decode", "generated_tokens_total"),
            {"status": "component_restarted"},
        )
        self.assertEqual(
            delta(report, "storage", "nvme_written_bytes_total"),
            {"status": "scope_changed"},
        )
        self.assertEqual(
            delta(report, "settlement", "closed_checkpoints_total"),
            {"status": "counter_reset"},
        )

    def test_refuses_interval_when_artifact_source_clock_or_deployment_changes(self):
        previous = snapshot()
        for field, value, reason in (
            ("release_manifest_sha256", "b" * 64, "artifact_mismatch"),
            ("deployment", "another-deployment", "deployment_changed"),
            ("source", "observed", "source_changed"),
            (
                "collector",
                {"epoch": "monitor-2", "monotonic_ns": 0},
                "collector_restarted",
            ),
            (
                "collector",
                {"epoch": "monitor-1", "monotonic_ns": 1},
                "non_increasing_time",
            ),
        ):
            with self.subTest(reason=reason):
                current = copy.deepcopy(previous)
                current["collector"]["monotonic_ns"] += 1
                current[field] = value
                report = operational_report(current, previous, ARTIFACT)
                self.assertEqual(report["interval"], {"status": reason})
                self.assertEqual(
                    delta(report, "prefill", "computed_input_tokens_total"),
                    {"status": reason},
                )

    def test_rejects_invalid_value_instead_of_silently_coercing(self):
        current = snapshot()
        current["components"]["decode"]["metrics"]["running"] = True
        with self.assertRaisesRegex(ValueError, "running"):
            operational_report(current, None, ARTIFACT)

        for value in (float("nan"), 10**400):
            with self.subTest(value=value):
                current = snapshot()
                current["components"]["decode"]["metrics"][
                    "output_interval_seconds_total"
                ] = value
                with self.assertRaisesRegex(
                    ValueError, "output_interval_seconds_total"
                ):
                    operational_report(current, None, ARTIFACT)

    def test_cli_binds_exact_manifest_bytes_without_writing_or_using_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            manifest_path = directory / "release.json"
            manifest_path.write_bytes(b'{"wheel":"fixture-identity-only"}\n')
            current_path = directory / "current.json"
            current = snapshot()
            current["release_manifest_sha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            current_path.write_text(json.dumps(current), encoding="utf-8")
            args = [
                "report",
                "--manifest",
                str(manifest_path),
                "--current",
                str(current_path),
            ]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(args), 0)
            receipt = json.loads(output.getvalue())
            self.assertTrue(receipt["artifact_matches"])
            self.assertEqual(
                receipt["snapshot_sha256"]["current"],
                hashlib.sha256(current_path.read_bytes()).hexdigest(),
            )
            manifest_path.write_bytes(b'{"wheel":"other-fixture"}\n')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(args), 1)
            self.assertFalse(json.loads(output.getvalue())["artifact_matches"])
            self.assertEqual(set(directory.iterdir()), {manifest_path, current_path})
