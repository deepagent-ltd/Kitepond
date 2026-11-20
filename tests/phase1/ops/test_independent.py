# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from kitepond.ops.cli import main
from kitepond.ops.independent import EXPECTED, INDEX_SCHEMA, PROFILE, REQUIRED_CHECKS
from kitepond.ops.wheel import sha256_file


class IndependentEvidenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.image = "sha256:" + "1" * 64
        self.model = "fixture-glm"
        plan = {
            "schema": "kitepond-phase1-independent-glm-lane.v1",
            "image": self.image,
            "source": {"commit": "1" * 40, "math_commit": "2" * 40},
            "native_sha256": {"fixture.so": "3" * 64},
            "model": {
                "served_model_id": self.model,
                "metadata_sha256": {
                    filename: "1" * 64
                    for filename in (
                        "config.json",
                        "model.safetensors.index.json",
                        "tokenizer.json",
                    )
                },
                "indexed_shard_bytes": {str(shard): 10 for shard in range(33)},
            },
            "server": [
                "--host",
                "127.0.0.1",
                "--port",
                "18160",
                "--tensor-parallel-size",
                "1",
                "--data-parallel-size",
                "8",
                "--pipeline-parallel-size",
                "1",
                "--enable-expert-parallel",
                "--no-enable-prefix-caching",
            ],
            "benchmark": {
                name: [
                    "--num-prompts",
                    str(values[0]),
                    "--max-concurrency",
                    str(values[0]),
                ]
                for name, values in EXPECTED.items()
            },
        }
        self.index = {
            "schema": INDEX_SCHEMA,
            "measurement_profile": PROFILE,
            "engine_receipt": self.write("engine.json", plan),
            "negative_package_probe": self.write_text(
                "import.log",
                '{"abi": {}, "optional_dependencies_imported": {"kitepond": false}}\n'
                "CLI_DRY_CHECK_PASS\n",
            ),
            "cases": {},
        }

    def write(self, name, obj):
        return self.write_text(name, json.dumps(obj))

    def write_text(self, name, value):
        path = self.root / name
        path.write_text(value)
        return {"file": name, "sha256": sha256_file(path)}

    def attempt(self, name, *, cache_hits=0):
        count, input_tokens, output_tokens = EXPECTED[name]
        summary = {
            "case": name,
            "checks": dict.fromkeys(REQUIRED_CHECKS, True),
            "completed": count,
            "failed": 0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_seconds": 10,
            "input_tok_s": input_tokens / 10,
            "output_tok_s": output_tokens / 10,
            "metrics_delta": {
                "vllm:prompt_tokens_total": input_tokens,
                "vllm:generation_tokens_total": output_tokens,
                "vllm:prefix_cache_hits_total": cache_hits,
                "vllm:external_prefix_cache_hits_total": 0,
                "vllm:num_preemptions_total": 0,
            },
            "ms": {
                metric: {"mean": 1, "p90": 2, "p99": 3}
                for metric in ("ttft", "tpot", "itl", "e2el")
            },
        }
        return {
            "engine_receipt_sha256": self.index["engine_receipt"]["sha256"],
            "result": self.write(f"{name}.json", summary),
            "container": self.write_text(
                f"{name}-container.txt", f"{'a' * 64} {self.image} 123 none true {{}}\n"
            ),
            "processes": self.write_text(
                f"{name}-processes.txt",
                "123 123 python -m vllm.entrypoints.cli.main serve /models/fixture "
                "--host 127.0.0.1 --port 18160 --data-parallel-size 8 "
                "--no-enable-prefix-caching\n",
            ),
            "devices": self.write_text(
                f"{name}-devices.txt",
                "\n".join(
                    f"{rank}, GPU-fixture-{rank}, NVIDIA RTX 6000D, 85651 MiB"
                    for rank in range(8)
                ),
            ),
            "native_closed": self.write_text(
                f"{name}-closed.log",
                "Native closed delta by rank: "
                + ", ".join(f"{rank}={count // 8}" for rank in range(8)),
            ),
            "model": self.write(
                f"{name}-model.json",
                {"data": [{"id": self.model, "max_model_len": 1048576}]},
            ),
            "client_image": self.write_text(
                f"{name}-client.txt", "sha256:" + "2" * 64 + "\n"
            ),
        }

    def evaluate(self):
        path = self.write("index.json", self.index)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["accept-glm", "--index", str(self.root / path["file"])])
        return status, json.loads(output.getvalue())

    def test_independent_command_does_not_need_release_wheel(self):
        self.index["cases"] = {case: self.attempt(case) for case in EXPECTED}
        status, report = self.evaluate()
        self.assertEqual(status, 0)
        self.assertTrue(report["complete"])
        self.assertEqual(report["cases"]["decode"]["status"], "PASS")

    def test_missing_and_cached_case_do_not_qualify(self):
        self.index["cases"]["prefill"] = self.attempt("prefill", cache_hits=42)
        status, report = self.evaluate()
        self.assertEqual(status, 2)
        self.assertEqual(report["cases"]["prefill"]["status"], "FAIL")
        self.assertEqual(report["cases"]["decode"]["status"], "NOT_RUN")

    def test_rejects_other_engine_receipt(self):
        case = self.attempt("decode")
        case["engine_receipt_sha256"] = "9" * 64
        self.index["cases"]["decode"] = case
        status, report = self.evaluate()
        self.assertEqual(status, 2)
        self.assertIn("different engine receipt", report["cases"]["decode"]["reason"])

    def test_detects_changed_observed_summary(self):
        case = self.attempt("prefill")
        self.index["cases"]["prefill"] = case
        self.write(case["result"]["file"], {"tampered": True})
        _, report = self.evaluate()
        self.assertIn("Digest mismatch", report["cases"]["prefill"]["reason"])
