# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise live telemetry with CPU-only loopback and timestamped adapter receipts."""

import contextlib
import hashlib
import http.server
import io
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from kitepond.ops.capture import (
    RECEIPT_SCHEMA,
    SOURCES_SCHEMA,
    EngineSource,
    _process_start,
    _role_export,
    capture_snapshot,
)
from kitepond.ops.cli import main
from kitepond.ops.report import operational_report

ARTIFACT = "a" * 64


def _exposition(rows: list[tuple]) -> bytes:
    return "\n".join(
        f'vllm:{name}{{model_name="{model}",engine="{engine}"{tag}}} {value}'
        for name, model, engine, tag, value in rows
    ).encode()


def _engine(pid: int, port: int, *, engine_epoch: str = "engine-1") -> dict:
    return {
        "port": port,
        "pid": pid,
        "process_start_ticks": _process_start(pid),
        "engine_epoch": engine_epoch,
        "model_name": "glm-fixture",
        "engine_ids": ["0", "1"],
    }


class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.requests.append(self.path)
        if self.path == "/health":
            self.send_response(self.server.health)
            self.end_headers()
        elif self.path == "/metrics":
            self.send_response(self.server.metrics_status)
            if self.server.metrics_status == 302:
                self.send_header("Location", "http://127.0.0.1:65533/unexpected")
            self.end_headers()
            self.wfile.write(self.server.metrics)
        else:
            self.send_error(404)

    def log_message(self, *_args):
        return


@contextlib.contextmanager
def _serve(metrics: bytes, health: int = 200):
    with http.server.HTTPServer(("127.0.0.1", 0), _LoopbackHandler) as server:
        server.metrics = metrics
        server.health = health
        server.metrics_status = 200
        server.requests = []
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.02}
        )
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=2)


def _rows(output: tuple[int, int] = (3, 12)) -> list[tuple]:
    return [
        ("num_requests_running", "glm-fixture", "0", "", 0),
        ("num_requests_running", "glm-fixture", "1", "", 0),
        ("num_requests_waiting", "glm-fixture", "0", "", 0),
        ("num_requests_waiting", "glm-fixture", "1", "", 1),
        ("num_requests_waiting", "elsewhere", "1", "", 1000),
        ("generation_tokens_total", "glm-fixture", "0", "", output[0]),
        ("generation_tokens_total", "glm-fixture", "1", "", output[1]),
        ("request_decode_time_seconds_sum", "glm-fixture", "0", "", 0.0),
        ("request_decode_time_seconds_sum", "glm-fixture", "1", "", 0.125),
    ]


class TelemetryTests(unittest.TestCase):
    def write_settings(self, root, prefill, decode, **fields):
        settings = {
            "schema": SOURCES_SCHEMA,
            "release_manifest_sha256": ARTIFACT,
            "deployment": "local-fixture",
            "roles": {
                "prefill": _engine(os.getpid(), prefill),
                "decode": _engine(os.getpid(), decode),
            },
        }
        settings.update(fields)
        path = root / "sources.json"
        path.write_text(json.dumps(settings), encoding="utf-8")
        return path, settings

    def test_loopback_attaches_role_and_detects_a_partial_counter_reset(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            _serve(_exposition(_rows())) as p,
            _serve(_exposition(_rows()), health=503) as d,
        ):
            path, _ = self.write_settings(Path(directory), p.server_port, d.server_port)
            first = capture_snapshot(path, ARTIFACT)
            self.assertEqual(first["components"]["prefill"]["metrics"]["running"], 0)
            self.assertEqual(first["components"]["decode"]["metrics"]["waiting"], 1)
            self.assertFalse(first["components"]["decode"]["metrics"]["healthy"])
            d.metrics = _exposition(_rows((1, 30)))

            after = capture_snapshot(path, ARTIFACT)
            result = operational_report(after, first, ARTIFACT)
            decode = result["components"]["decode"]["metrics"]
            self.assertEqual(decode["generated_tokens_total"]["value"], 31)
            self.assertEqual(
                decode["generated_tokens_total"]["delta"], {"status": "counter_reset"}
            )
            self.assertEqual(
                decode["output_interval_seconds_total"]["delta"],
                {"status": "observed", "value": 0},
            )
            self.assertEqual(p.requests, ["/health", "/metrics"] * 2)
            self.assertEqual(d.requests, ["/health", "/metrics"] * 2)
            self.assertEqual(
                result["gate_evidence"]["release_qualification"], "not_evaluated"
            )

    def test_process_generation_and_export_errors_preserve_unavailability(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            _serve(_exposition(_rows())) as p,
            _serve(_exposition(_rows())) as d,
        ):
            path, settings = self.write_settings(
                Path(directory), p.server_port, d.server_port
            )
            first = capture_snapshot(path, ARTIFACT)
            d.metrics_status = 302
            settings["roles"]["prefill"]["process_start_ticks"] += 1
            settings["roles"]["decode"]["engine_epoch"] = "engine-2"
            path.write_text(json.dumps(settings), encoding="utf-8")
            after = capture_snapshot(path, ARTIFACT)
            report = operational_report(after, first, ARTIFACT)
            self.assertEqual(
                after["components"]["prefill"]["source_status"], "process_unavailable"
            )
            self.assertEqual(after["components"]["prefill"]["metrics"], {})
            self.assertEqual(
                after["components"]["decode"]["metrics"], {"healthy": True}
            )
            self.assertEqual(
                report["components"]["decode"]["interval_status"], "component_restarted"
            )
            self.assertEqual(d.requests, ["/health", "/metrics"] * 2)

    def test_incomplete_and_ambiguous_prompt_metrics_never_turn_into_hits(self):
        engine = EngineSource.parse(_engine(os.getpid(), 8))
        rows = [
            ("prompt_tokens_total", "glm-fixture", "0", "", 40),
            ("prompt_tokens_total", "glm-fixture", "1", "", 80),
            (
                "prompt_tokens_by_source_total",
                "glm-fixture",
                "0",
                ',source="local_compute"',
                20,
            ),
            (
                "prompt_tokens_by_source_total",
                "glm-fixture",
                "1",
                ',source="local_compute"',
                40,
            ),
            (
                "prompt_tokens_by_source_total",
                "glm-fixture",
                "0",
                ',source="local_cache_hit"',
                20,
            ),
            (
                "prompt_tokens_by_source_total",
                "glm-fixture",
                "1",
                ',source="local_cache_hit"',
                20,
            ),
            (
                "prompt_tokens_by_source_total",
                "glm-fixture",
                "0",
                ',source="external_kv_transfer"',
                0,
            ),
            (
                "prompt_tokens_by_source_total",
                "glm-fixture",
                "1",
                ',source="external_kv_transfer"',
                20,
            ),
        ]
        readings, counters = _role_export(_exposition(rows).decode(), "prefill", engine)
        self.assertEqual(
            readings,
            {"logical_input_tokens_total": 120, "computed_input_tokens_total": 60},
        )
        self.assertEqual(counters["logical_input_tokens_total"], {"0": 40, "1": 80})
        duplicate, _ = _role_export(
            _exposition(rows + [rows[0]]).decode(), "prefill", engine
        )
        self.assertNotIn("logical_input_tokens_total", duplicate)
        rows[-1] = (*rows[-1][:-1], 0)
        cache, _ = _role_export(_exposition(rows).decode(), "prefill", engine)
        self.assertEqual(cache["reused_input_tokens_total"], 40)
        self.assertNotIn("native_compute_seconds_total", cache)

    def test_receipt_needs_current_boot_and_scope_and_can_observe_zero(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            _serve(b"") as p,
            _serve(b"") as d,
        ):
            root = Path(directory)
            closure, disk = root / "closure.json", root / "disk.json"
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            path, _ = self.write_settings(
                root,
                p.server_port,
                d.server_port,
                adapters={
                    "settlement": {
                        "path": str(closure),
                        "scope": "cache-owner",
                        "epoch": "own-1",
                        "max_age_seconds": 60,
                    },
                    "storage": {
                        "path": str(disk),
                        "scope": "cache-owner",
                        "epoch": "disk-1",
                        "max_age_seconds": 60,
                    },
                },
            )
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "release_manifest_sha256": ARTIFACT,
                "deployment": "local-fixture",
                "scope": "cache-owner",
                "boot_id": boot,
                "monotonic_ns": time.monotonic_ns(),
            }
            closure.write_text(
                json.dumps(
                    dict(
                        receipt,
                        component="settlement",
                        epoch="own-1",
                        metrics={"closed_checkpoints_total": 0},
                    )
                )
            )
            disk.write_text(
                json.dumps(
                    dict(
                        receipt,
                        component="storage",
                        epoch="disk-1",
                        boot_id="stale",
                        metrics={"ram_resident_bytes": 0, "nvme_payload_bytes": 2048},
                    )
                )
            )
            first = capture_snapshot(path, ARTIFACT)
            self.assertEqual(
                first["components"]["settlement"]["metrics"],
                {"closed_checkpoints_total": 0},
            )
            self.assertEqual(
                first["components"]["storage"]["source_status"], "receipt_unavailable"
            )
            self.assertEqual(first["components"]["storage"]["metrics"], {})
            disk.write_text(
                json.dumps(
                    dict(
                        receipt,
                        component="storage",
                        epoch="disk-1",
                        metrics={"ram_resident_bytes": 0, "nvme_payload_bytes": 2048},
                    )
                )
            )
            after = capture_snapshot(path, ARTIFACT)
            self.assertEqual(
                after["components"]["storage"]["metrics"],
                {"ram_resident_bytes": 0, "nvme_payload_bytes": 2048},
            )

    def test_hisparse_uses_only_decode_complete_series_and_process_epoch(self):
        hits = "hisparse_cache_hits_total"
        misses = "hisparse_cache_misses_total"
        recall = "hisparse_host_to_device_bytes_total"

        def scrape(hit0: int, hit1: int, *, omit_recall=False):
            return _exposition(
                _rows()
                + [
                    (hits, "glm-fixture", "0", "", hit0),
                    (hits, "glm-fixture", "1", "", hit1),
                    (hits, "another-model", "1", "", 100),
                    (misses, "glm-fixture", "0", "", 0),
                    (misses, "glm-fixture", "1", "", 1),
                    (recall, "glm-fixture", "0", "", 0),
                    (recall, "glm-fixture", "9" if omit_recall else "1", "", 656),
                ]
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            _serve(scrape(10, 20)) as p,
            _serve(_exposition(_rows())) as d,
        ):
            path, settings = self.write_settings(
                Path(directory), p.server_port, d.server_port
            )
            missing = capture_snapshot(path, ARTIFACT)["components"]
            self.assertEqual(missing["decode_residency"]["metrics"], {})
            self.assertNotIn(hits, missing["prefill"]["metrics"])
            self.assertNotIn("residency", missing)

            d.metrics = scrape(0, 0, omit_recall=True)
            first = capture_snapshot(path, ARTIFACT)
            measured = first["components"]["decode_residency"]
            decode = first["components"]["decode"]
            self.assertEqual(measured["epoch"], decode["epoch"])
            self.assertEqual(measured["scope"], decode["scope"])
            self.assertNotIn(hits, decode["metrics"])
            self.assertEqual(measured["metrics"], {hits: 0, misses: 1})
            self.assertEqual(measured["source_status"], "partial")

            d.metrics = scrape(3, 4)
            second = capture_snapshot(path, ARTIFACT)
            report = operational_report(second, first, ARTIFACT)
            residency = report["components"]["decode_residency"]["metrics"]
            self.assertEqual(
                residency[hits]["delta"], {"status": "observed", "value": 7}
            )
            self.assertEqual(residency[recall]["value"], 656)
            self.assertEqual(residency[recall]["delta"], {"status": "baseline_missing"})
            self.assertEqual(report["gate_evidence"]["decode_residency"], "observed")

            d.metrics = scrape(1, 30)
            third = capture_snapshot(path, ARTIFACT)
            report = operational_report(third, second, ARTIFACT)
            residency = report["components"]["decode_residency"]["metrics"]
            self.assertEqual(residency[hits]["value"], 31)
            self.assertEqual(residency[hits]["delta"], {"status": "counter_reset"})

            settings["roles"]["decode"]["engine_epoch"] = "engine-2"
            path.write_text(json.dumps(settings))
            fourth = capture_snapshot(path, ARTIFACT)
            report = operational_report(fourth, third, ARTIFACT)
            residency = report["components"]["decode_residency"]["metrics"]
            self.assertEqual(
                residency[misses]["delta"], {"status": "component_restarted"}
            )

    def test_cli_validates_exact_release_before_contacting_local_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "release.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "kitepond.release-manifest.v1",
                        "base": {
                            "image": f"vllm@sha256:{ARTIFACT}",
                            "python": "3.12",
                            "vllm": "fixture",
                        },
                        "source_commit": "b" * 40,
                        "wheel": {"sha256": ARTIFACT, "version": "0.1"},
                    }
                )
            )
            artifact = hashlib.sha256(manifest.read_bytes()).hexdigest()
            sources, settings = self.write_settings(root, 60001, 60002)
            args = ["capture", "--manifest", str(manifest), "--sources", str(sources)]
            with (
                patch("kitepond.ops.capture._request") as request,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as failure,
            ):
                main(args)
            self.assertEqual(failure.exception.code, 1)
            request.assert_not_called()
            settings["release_manifest_sha256"] = artifact
            sources.write_text(json.dumps(settings))
            output = io.StringIO()
            with (
                patch("kitepond.ops.capture._request", return_value=(503, b"")),
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(main(args), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["release_manifest_sha256"], artifact)
            self.assertEqual(
                result["components"]["decode"]["metrics"], {"healthy": False}
            )
