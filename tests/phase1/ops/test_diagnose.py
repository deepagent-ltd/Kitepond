# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from kitepond.ops.cli import main
from kitepond.ops.report import SNAPSHOT_SCHEMA


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.manifest = self.write(
            "release.json",
            {
                "schema": "kitepond.release-manifest.v1",
                "source_commit": "a" * 40,
                "base": {
                    "image": "vllm/vllm-openai@sha256:" + "b" * 64,
                    "vllm": "0.33.0",
                    "python": "3.12",
                },
                "wheel": {"version": "0.1.0.dev0", "sha256": "c" * 64},
            },
        )
        self.digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def snapshot(self, name, *, source="observed", observed=None, stamp=10):
        return self.write(
            name,
            {
                "schema": SNAPSHOT_SCHEMA,
                "release_manifest_sha256": self.digest,
                "source": source,
                "deployment": "/private/tenant-identifier",
                "collector": {
                    "epoch": "/private/collector-path",
                    "monotonic_ns": stamp,
                },
                "components": observed or {},
            },
        )

    def doctor(self, **hooks):
        return self.write(
            "doctor.json",
            {
                "manifest_sha256": self.digest,
                "compatible_metadata": True,
                "observed": {"vllm_origin": "/private/site-packages/vllm/__init__.py"},
                "matches": {
                    key: True for key in ("kitepond", "python", "vllm", "vllm_origin")
                },
                "engine_source_capabilities": hooks or None,
            },
        )

    def accepted(self, *, digest=None, receipt="d" * 64):
        manifest = json.loads(self.manifest.read_text())
        return self.write(
            "accept.json",
            {
                "schema": "kitepond.release-acceptance-report.v1",
                "identity": {
                    "release_manifest_sha256": digest or self.digest,
                    "wheel_sha256": manifest["wheel"]["sha256"],
                    "base_image": manifest["base"]["image"],
                    "source_commit": manifest["source_commit"],
                },
                "cases": {
                    "cold_restart": {
                        "status": "PASS",
                        "attempts": [
                            {"status": "FAIL", "reason": "/private/previous-run"},
                            {
                                "status": "PASS",
                                "run": "private-task",
                                "result_sha256": receipt,
                            },
                        ],
                    }
                },
            },
        )

    def diagnose(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["diagnose", "--manifest", str(self.manifest), *map(str, args)])
        self.assertEqual(code, 0)
        self.assertNotIn("/private/", output.getvalue())
        self.assertNotIn("private-task", output.getvalue())
        return json.loads(output.getvalue())

    def test_optional_data_stays_missing(self):
        result = self.diagnose()
        self.assertEqual(
            result["stock_integration"]["startup_source_verdict"], "missing"
        )
        self.assertEqual(
            result["core"]["settlement"]["held_references"], {"status": "missing"}
        )
        self.assertEqual(result["deployment"]["rollback_receipt"]["status"], "NOT_RUN")

    def test_redacts_identifiers_keeps_zero_age_scope_and_counter_deltas(self):
        def records(hot, byte_count):
            return {
                "settlement": {
                    "epoch": "/private/owner-generation",
                    "scope": "/private/scope",
                    "metrics": {"held_references": 0, "oldest_reference_age_ns": 0},
                },
                "decode_residency": {
                    "epoch": "decode-engine-1",
                    "scope": "decode-rank-group",
                    "metrics": {"hisparse_cache_hits_total": hot * 10},
                },
                "residency": {
                    "epoch": "r1",
                    "scope": "rank-0",
                    "metrics": {
                        "hot_page_hits_total": hot,
                        "host_recalled_bytes_total": byte_count,
                    },
                },
                "storage": {
                    "metrics": {"nvme_read_bytes_total": 0},
                },
            }

        earlier = self.snapshot("before.json", observed=records(2, 4096))
        current = self.snapshot("after.json", observed=records(5, 8192), stamp=20)
        result = self.diagnose("--current", current, "--previous", earlier)
        self.assertEqual(
            result["core"]["settlement"]["oldest_reference_age_ns"],
            {"status": "observed", "value": 0},
        )
        view = result["storage_and_transport"]
        self.assertEqual(
            view["residency"]["host_recalled_bytes_total"]["delta"]["value"], 4096
        )
        self.assertEqual(
            view["residency"]["host_recalled_pages_total"], {"status": "missing"}
        )
        self.assertEqual(
            view["storage"]["nvme_read_bytes_total"],
            {"status": "unattributed", "value": 0},
        )
        self.assertEqual(
            view["decode_residency"]["hisparse_cache_hits_total"]["delta"],
            {"status": "observed", "value": 30},
        )
        self.assertEqual(
            view["decode_residency"]["hisparse_host_to_device_bytes_total"],
            {"status": "missing"},
        )
        self.assertEqual(view["cold_restart_receipt"]["status"], "NOT_RUN")
        self.assertIsNotNone(result["run_scope"]["deployment_fingerprint"])

    def test_stock_probe_only_and_matching_cold_receipt(self):
        result = self.diagnose(
            "--doctor",
            self.doctor(connector_module_path=True, full_cache_mode="unavailable"),
            "--acceptance",
            self.accepted(),
        )
        self.assertEqual(
            result["stock_integration"]["startup_source_verdict"],
            "registration_probe_only",
        )
        self.assertEqual(
            result["storage_and_transport"]["cold_restart_receipt"],
            {"status": "PASS", "result_sha256": "d" * 64},
        )

    def test_synthetic_and_restarted_counts_do_not_become_deltas(self):
        before = {
            "residency": {
                "epoch": "r1",
                "scope": "pool",
                "metrics": {"hot_page_hits_total": 4},
            }
        }
        after = {
            "residency": {
                "epoch": "r2",
                "scope": "pool",
                "metrics": {"hot_page_hits_total": 8},
            }
        }
        previous = self.snapshot("previous.json", observed=before)
        current = self.snapshot("current.json", observed=after, stamp=20)
        result = self.diagnose("--current", current, "--previous", previous)
        hot = result["storage_and_transport"]["residency"]["hot_page_hits_total"]
        self.assertEqual(hot["delta"], {"status": "component_restarted"})
        synthetic = self.snapshot("fixture.json", source="synthetic", observed=after)
        result = self.diagnose("--current", synthetic)
        self.assertEqual(
            result["storage_and_transport"]["residency"]["hot_page_hits_total"],
            {"status": "synthetic", "value": 8},
        )

    def test_rejects_wrong_release_or_unbound_cold_result(self):
        for path in (self.accepted(digest="0" * 64),):
            with (
                contextlib.redirect_stderr(io.StringIO()) as error,
                self.assertRaises(SystemExit),
            ):
                self.diagnose("--acceptance", path)
            self.assertIn("another release", error.getvalue())
        with (
            contextlib.redirect_stderr(io.StringIO()) as error,
            self.assertRaises(SystemExit),
        ):
            self.diagnose("--acceptance", self.accepted(receipt=None))
        self.assertIn("original result digest", error.getvalue())
