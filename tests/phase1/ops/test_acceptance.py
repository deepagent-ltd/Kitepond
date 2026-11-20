# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from kitepond.ops.acceptance import ATTEMPT_SCHEMA, CASES, INDEX_SCHEMA
from kitepond.ops.cli import main
from kitepond.ops.report import SNAPSHOT_SCHEMA
from kitepond.ops.wheel import sha256_file
from test_release import write_fixture


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.wheel = self.root / "kitepond-0.1.0.dev0-py3-none-any.whl"
        self.manifest = self.root / "release.json"
        write_fixture(self.wheel)
        _, manifest = self.invoke(
            "seal",
            "--wheel",
            str(self.wheel),
            "--source-commit",
            "a" * 40,
            "--base-image",
            f"vllm/vllm-openai@sha256:{'b' * 64}",
            "--base-vllm-version",
            "0.33.0",
            "--base-python-version",
            "3.12",
        )
        self.manifest.write_text(json.dumps(manifest))
        self.release_digest = sha256_file(self.manifest)
        self.plan = {
            "manifest_sha256": self.release_digest,
            "source_commit": manifest["source_commit"],
            "base": manifest["base"],
            "wheel_sha256": manifest["wheel"]["sha256"],
            "model_files": {
                "config.json": "c" * 64,
                "model.safetensors.index.json": "d" * 64,
                "tokenizer.json": "e" * 64,
                "tokenizer_config.json": "f" * 64,
            },
            "roles": {"prefill": [4, 5, 6, 7], "decode": [0, 1, 2, 3]},
        }
        config = self.write("config.json", {"revision": 2})
        self.identity = {
            "release_manifest_sha256": self.release_digest,
            "wheel_sha256": manifest["wheel"]["sha256"],
            "source_commit": manifest["source_commit"],
            "base_image": manifest["base"]["image"],
            "model_files": self.plan["model_files"],
            "configuration_sha256": config["sha256"],
        }
        self.index = {
            "schema": INDEX_SCHEMA,
            "release": {
                "manifest": self.ref(self.manifest.name),
                "wheel": self.ref(self.wheel.name),
                "plan": self.write("plan.json", self.plan),
                "configuration": config,
            },
            "cases": {},
        }
        self.driver = self.write("driver.json", {"name": "CPU-only structural fixture"})

    def ref(self, name):
        return {"file": name, "sha256": sha256_file(self.root / name)}

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))
        return self.ref(name)

    def invoke(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(list(args))
        return status, json.loads(output.getvalue())

    def evaluate(self):
        ref = self.write("index.json", self.index)
        return self.invoke(
            "accept",
            "--index",
            str(self.root / ref["file"]),
            "--manifest",
            str(self.manifest),
            "--wheel",
            str(self.wheel),
        )

    def register(self, name, receipt):
        self.index["cases"].setdefault(receipt["case"], []).append(
            self.write(f"{name}.json", receipt)
        )

    def snapshot(
        self, name, count, epochs, *, source="observed", deployment="deployment"
    ):
        roles = {}
        for role, epoch in epochs.items():
            metrics = {"healthy": True, "running": 0, "waiting": 0}
            if role == "prefill":
                metrics.update(logical_input_tokens_total=count)
            else:
                metrics.update(generated_tokens_total=count)
            roles[role] = {"epoch": epoch, "metrics": metrics}
        return self.write(
            f"{name}.json",
            {
                "schema": SNAPSHOT_SCHEMA,
                "release_manifest_sha256": self.release_digest,
                "collector": {"epoch": "collector", "monotonic_ns": count + 1},
                "deployment": deployment,
                "source": source,
                "components": {
                    **roles,
                    "settlement": {
                        "epoch": "authority",
                        "scope": "run-namespace",
                        "metrics": {
                            "held_references": 0,
                            "pending_retirement": 0,
                            "closed_checkpoints_total": count,
                        },
                    },
                    "storage": {
                        "epoch": "store",
                        "scope": "cold-namespace",
                        "metrics": {"nvme_read_bytes_total": count * 1000},
                    },
                },
            },
        )

    def stock_doctor(self):
        return {
            "compatible_metadata": True,
            "manifest_sha256": self.release_digest,
            "engine_files": {
                "version": "0.33.0",
                "origin": "/vllm/__init__.py",
                "vllm_file_count": 123,
                "vllm_files_sha256": "d" * 64,
            },
            "matches": {
                key: True for key in ("kitepond", "python", "vllm", "vllm_origin")
            },
            "engine_source_capabilities": {
                "scope": "source_signatures",
                "distribution_version": "0.33.0",
            },
        }

    def stock_result(self):
        doctor = self.stock_doctor()
        return {
            "vllm_preserved": True,
            "gpu_qualified": False,
            "engine_imported": False,
            "stock": doctor["engine_files"],
            "checks": {
                "verify": {
                    "wheel_verified": True,
                    "manifest_sha256": self.release_digest,
                },
                "doctor": doctor,
            },
        }

    def stock_runtime(self, run, epochs):
        return {
            "stock": self.write("independent-stock.json", self.stock_result()),
            **{
                role: {
                    "role": role,
                    "epoch": epoch,
                    "base_image": self.identity["base_image"],
                    "doctor": self.write(
                        f"{run}-{role}-doctor.json", self.stock_doctor()
                    ),
                }
                for role, epoch in epochs.items()
            },
        }

    def attempt(
        self,
        case,
        run,
        *,
        mode="gpu",
        new=None,
        source="observed",
        deployment="deployment",
    ):
        original = {"prefill": "p-v1", "decode": "d-v1"}
        new = new or original
        return {
            "schema": ATTEMPT_SCHEMA,
            "case": case,
            "run": run,
            "mode": mode,
            "identity": self.identity.copy(),
            "status": "PASS",
            "driver": self.driver,
            "result": self.write(f"{run}-result.json", {"run": run}),
            "stock_engine_provenance": {
                "before": self.stock_runtime(f"{run}-before", original),
                "after": self.stock_runtime(f"{run}-after", new),
            },
            "gpu_devices": {
                "prefill": [f"GPU-{i}" for i in range(4)],
                "decode": [f"GPU-{i}" for i in range(4, 8)],
            },
            "snapshots": {
                "before": self.snapshot(f"{run}-before", 1, original, source=source),
                "after": self.snapshot(
                    f"{run}-after", 2, new, source=source, deployment=deployment
                ),
            },
            "before_role_epochs": original,
            "after_role_epochs": new,
            "proof": {
                "peak_concurrency": 32,
                "successful_requests": 32,
                "native_token_counts_match": True,
                "outputs_valid": True,
                "closed_ranks": {"prefill": [0, 1, 2, 3], "decode": [0, 1, 2, 3]},
            },
        }

    def test_missing_and_fixture_cannot_qualify_gpu(self):
        receipt = self.attempt("c32", "mock", mode="fixture")
        self.register("mock", receipt)
        status, report = self.evaluate()
        self.assertEqual(status, 2)
        self.assertSetEqual(set(report["cases"]), set(CASES))
        self.assertEqual(report["cases"]["c32"]["status"], "NOT_RUN")
        self.assertEqual(report["cases"]["cold_restart"]["status"], "NOT_RUN")
        self.assertFalse(report["complete"])

    def test_existing_stock_smoke_shape_uses_sealed_manifest(self):
        receipt = self.attempt("stock_install", "stock", mode="cpu")
        receipt["identity"] = {
            key: self.identity[key]
            for key in (
                "release_manifest_sha256",
                "wheel_sha256",
                "source_commit",
                "base_image",
            )
        }
        receipt["result"] = self.write("stock-smoke.json", self.stock_result())
        self.register("stock", receipt)
        _, report = self.evaluate()
        self.assertEqual(report["cases"]["stock_install"]["status"], "PASS")
        self.assertFalse(report["complete"])

    def test_phase_evidence_fails_wrong_identity_or_synthetic_snapshot(self):
        real = self.attempt("c32", "first")
        self.register("first", real)
        wrong = self.attempt("c32", "wrong")
        wrong["identity"]["configuration_sha256"] = "1" * 64
        self.register("wrong", wrong)
        mock = self.attempt("c32", "mock", source="synthetic")
        self.register("mock", mock)
        _, report = self.evaluate()
        rows = report["cases"]["c32"]["attempts"]
        self.assertEqual([row["status"] for row in rows], ["PASS", "FAIL", "FAIL"])
        self.assertIn("configuration identity", rows[1]["reason"])
        self.assertIn("matching observed", rows[2]["reason"])

    def test_gpu_evidence_requires_same_manifest_and_original_engine(self):
        for name in ("ok", "boolean-only", "overlay", "old-boot", "old-wheel"):
            receipt = self.attempt("c32", name)
            if name == "boolean-only":
                receipt.pop("stock_engine_provenance")
                receipt["stock_vllm_files_preserved"] = True
            if name == "old-boot":
                receipt["stock_engine_provenance"]["after"]["decode"]["epoch"] = "d-old"
            if name in {"overlay", "old-wheel"}:
                doctor = self.stock_doctor()
                if name == "overlay":
                    doctor["engine_files"]["vllm_files_sha256"] = "f" * 64
                else:
                    doctor["manifest_sha256"] = "c" * 64
                receipt["stock_engine_provenance"]["before"]["prefill"]["doctor"] = (
                    self.write(f"{name}-mismatch.json", doctor)
                )
            self.register(name, receipt)

        _, report = self.evaluate()
        rows = report["cases"]["c32"]["attempts"]
        self.assertEqual([row["status"] for row in rows], ["PASS"] + ["FAIL"] * 4)
        self.assertIn("Stock engine evidence", rows[1]["reason"])
        self.assertIn("file identity differs", rows[2]["reason"])
        self.assertIn("different role, boot or image", rows[3]["reason"])
        self.assertIn("intended manifest", rows[4]["reason"])

    def test_fault_arms_cannot_reuse_one_injection_or_result(self):
        for case in ("cancel", "lost_write_ack"):
            receipt = self.attempt(case, case)
            receipt["injection"] = self.write(
                "injection.json", {"native_event": "same_event"}
            )
            receipt["proof"] = {
                "injection_observed": True,
                "expected_disposition": True,
                "model_recovered": True,
                "lease_released": True,
            }
            self.register(case, receipt)
        _, report = self.evaluate()
        fault = report["cases"]
        self.assertEqual(fault["cancel"]["status"], "PASS")
        self.assertEqual(fault["lost_write_ack"]["status"], "FAIL")
        self.assertIn(
            "independent injection", fault["lost_write_ack"]["attempts"][0]["reason"]
        )

    def test_late_ack_requires_old_signature_and_fresh_decode_epoch(self):
        for name, new in (
            ("changed", {"prefill": "p-v2", "decode": "d-v2"}),
            ("wrong-source", {"prefill": "p-v2", "decode": "d-v2"}),
            ("same", {"prefill": "p-v1", "decode": "d-v1"}),
        ):
            receipt = self.attempt("late_epoch_ack", name, new=new)
            receipt["injection"] = self.write(f"{name}-injection.json", {"event": name})
            receipt["proof"] = {
                "injection_observed": True,
                "expected_disposition": True,
                "model_recovered": True,
                "lease_released": True,
                "replay": {
                    "source_decode_epoch": "d-v0" if name == "wrong-source" else "d-v1",
                    "target_decode_epoch": new["decode"],
                    "signed_ack": self.write(f"{name}-ack.json", {"signed": name}),
                    "old_ack_previously_accepted": True,
                    "fresh_lease_unchanged": True,
                },
            }
            self.register(name, receipt)
        _, report = self.evaluate()
        rows = report["cases"]["late_epoch_ack"]["attempts"]
        self.assertEqual([row["status"] for row in rows], ["PASS", "FAIL", "FAIL"])
        self.assertIn("old and fresh decode epochs", rows[1]["reason"])
        self.assertIn("targeted role epoch", rows[2]["reason"])

    def test_cold_read_after_fresh_pd_epochs(self):
        new = {"prefill": "p-v2", "decode": "d-v2"}
        receipt = self.attempt("cold_restart", "cold", new=new)
        receipt["snapshots"]["after_restart"] = self.snapshot("fresh", 0, new)
        receipt["proof"] = {
            "empty_cold_root_before_store": True,
            "persistent_commit_complete": True,
            "ram_owner_absent_after_restart": True,
            "restored_ranks": {"prefill": [0, 1, 2, 3], "decode": [0, 1, 2, 3]},
            "stored_payload_sha256": "1" * 64,
            "restored_payload_sha256": "1" * 64,
        }
        self.register("cold", receipt)
        _, report = self.evaluate()
        self.assertEqual(report["cases"]["cold_restart"]["status"], "PASS")
        receipt["snapshots"]["after_restart"] = self.snapshot(
            "false-fresh", 0, receipt["before_role_epochs"]
        )
        receipt["run"] = "cold-replay"
        self.register("cold-replay", receipt)
        _, report = self.evaluate()
        row = report["cases"]["cold_restart"]
        self.assertEqual(row["status"], "FAIL")
        self.assertIn("fresh P/D epochs", row["attempts"][-1]["reason"])

    def test_rollback_must_load_separate_verified_identity(self):
        receipt = self.attempt(
            "rollback",
            "rollback",
            new={"prefill": "p-v3", "decode": "d-v3"},
            deployment="prior",
        )
        receipt["proof"] = {
            "ordered_drain_and_stop": True,
            "prior_wheel_verified": True,
            "warm_and_cold_results_verified": True,
            "old_generation_recovered": True,
            "recovery_seconds": 42,
        }
        self.register("unverified-baseline", receipt)
        _, report = self.evaluate()
        self.assertEqual(report["cases"]["rollback"]["status"], "FAIL")
        self.assertIn("baseline", report["cases"]["rollback"]["attempts"][0]["reason"])

        rollback_config = self.write("rollback-config.json", {"revision": 1})
        self.index["rollback_baseline"] = {
            **self.index["release"],
            "configuration": rollback_config,
        }
        receipt["run"] = "rollback-confirmed"
        receipt["rollback_identity"] = {
            **self.identity,
            "configuration_sha256": rollback_config["sha256"],
        }
        self.register("confirmed-baseline", receipt)
        _, report = self.evaluate()
        row = report["cases"]["rollback"]
        self.assertEqual(
            [attempt["status"] for attempt in row["attempts"]], ["FAIL", "PASS"]
        )
        self.assertEqual(row["status"], "PASS")

    def test_evidence_digest_mismatch_does_not_get_a_pass(self):
        candidate = self.attempt("c32", "candidate")
        self.register("candidate", candidate)
        self.write(candidate["result"]["file"], {"tampered": True})
        _, report = self.evaluate()
        self.assertIn(
            "Digest mismatch", report["cases"]["c32"]["attempts"][0]["reason"]
        )
