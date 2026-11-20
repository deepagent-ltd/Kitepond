import contextlib
import io
import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path

from kitepond import dp8


class DP8PlanTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.model_dir = Path(workspace.name) / "local GLM NVFP4"
        self.model_dir.mkdir()
        self.config = {
            "architectures": ["Glm5NextForConditionalGeneration"],
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next_text",
                "index_topk": 2048,
                "num_hidden_layers": 45,
                "max_position_embeddings": 1048576,
            },
        }
        self.write_json("config.json", self.config)
        self.write_json(
            "hf_quant_config.json", {"quantization": {"quant_algo": "NVFP4"}}
        )
        self.write_json(
            "model.safetensors.index.json",
            {"weight_map": {"layer": "model-01.safetensors", "head": "model-02.safetensors"}},
        )
        for name in ("model-01.safetensors", "model-02.safetensors"):
            (self.model_dir / name).write_bytes(b"metadata fixture")
        self.args = [
            "--model", str(self.model_dir),
            "--host-pool-gib-per-rank", "8",
            "--host-pool-budget-gib", "64",
        ]

    def write_json(self, name, value):
        (self.model_dir / name).write_text(json.dumps(value), encoding="utf-8")

    def plan(self, *options):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(dp8.main([*self.args, *options]), 0)
        return output.getvalue()

    def assert_rejected(self, match, *options):
        output = io.StringIO()
        with contextlib.redirect_stderr(output), self.assertRaises(SystemExit) as exc:
            dp8.main([*self.args, *options])
        self.assertEqual(exc.exception.code, 2)
        self.assertIn(match, output.getvalue())

    def test_native_argv_fixes_topology_and_sets_per_dp_budget(self):
        plan = json.loads(self.plan())
        self.assertEqual(
            plan["topology"], {"tp": 1, "dp": 8, "ep": 8, "pp": 1, "pcp": 1, "dcp": 1}
        )
        self.assertEqual(
            plan["host_pool"],
            {"per_dp_rank_gib": 8, "requested_total_gib": 64, "budget_total_gib": 64},
        )
        self.assertEqual(plan["checkpoint_shards"], 2)
        self.assertFalse(plan["vllm_cli_checked"])
        argv = plan["argv"]
        self.assertEqual(argv[:3], ["vllm", "serve", str(self.model_dir)])
        for flag, value in {
            "--tensor-parallel-size": "1",
            "--data-parallel-size": "8",
            "--data-parallel-size-local": "8",
            "--data-parallel-backend": "mp",
            "--pipeline-parallel-size": "1",
            "--prefill-context-parallel-size": "1",
            "--decode-context-parallel-size": "1",
            "--max-num-seqs": "16",
            "--max-num-batched-tokens": "1024",
            "--kv-cache-dtype": "fp8",
        }.items():
            self.assertEqual(argv[argv.index(flag) + 1], value)
        for flag in (
            "--enable-expert-parallel",
            "--scheduler-reserve-full-isl",
            "--no-enable-prefix-caching",
        ):
            self.assertIn(flag, argv)
        self.assertEqual(
            json.loads(argv[argv.index("--kv-transfer-config") + 1]),
            {
                "kv_connector": "HiSparseConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {"host_pool_gib": 8},
            },
        )
        self.assertNotIn("--speculative-config", argv)
        self.assertNotIn("--hf-overrides", argv)

    def test_external_host_policy_is_opt_in_and_matches_launch_budgets(self):
        self.write_json("policy.json", {"rank_bytes": 8 << 30, "total_bytes": 64 << 30})
        path = str(self.model_dir / "policy.json")
        plan = json.loads(self.plan("--external-host-policy", path))
        self.assertEqual(plan["environment"]["KITEPOND_HISPARSE_HOST_POOL_POLICY"], path)
        argv = plan["argv"]
        connector = json.loads(argv[argv.index("--kv-transfer-config") + 1])
        self.assertEqual(
            connector["kv_connector_extra_config"]["host_pool_allocator"],
            "kitepond.hisparse_host.acquire",
        )
        self.assert_rejected(
            "requires the vLLM fork", "--stock-topk-override", "--external-host-policy", path
        )
        self.assert_rejected(
            "budgets must match", "--host-pool-budget-gib", "80", "--external-host-policy", path
        )

    def test_capacity_arms_share_the_frozen_sm120_settings(self):
        self.write_json("policy.json", {"rank_bytes": 8 << 30, "total_bytes": 64 << 30})
        policy = str(self.model_dir / "policy.json")
        control = io.StringIO()
        with contextlib.redirect_stdout(control):
            self.assertEqual(
                dp8.main([*self.args[:2], "--capacity-comparison", "stock-resident"]), 0
            )
        stock = json.loads(control.getvalue())
        candidate = json.loads(self.plan(
            "--capacity-comparison", "indexer2048", "--external-host-policy", policy
        ))
        self.assertEqual(stock["capacity_comparison"], "stock-resident")
        self.assertEqual(candidate["capacity_comparison"], "indexer2048")
        self.assertEqual(stock["host_pool"]["requested_total_gib"], 0)
        self.assertNotIn("KITEPOND_HISPARSE_HOST_POOL_POLICY", stock["environment"])
        self.assertEqual(candidate["host_pool"]["requested_total_gib"], 64)
        self.assertEqual(candidate["environment"]["KITEPOND_HISPARSE_HOST_POOL_POLICY"], policy)

        candidate_argv = candidate["argv"].copy()
        for flag, value in {
            "--kv-transfer-config": {
                "kv_connector": "HiSparseConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "host_pool_gib": 8, "host_pool_allocator": "kitepond.hisparse_host.acquire",
                },
            },
            "--attention-config": {"hisparse_config": {"indexer_block_size": 2048}},
        }.items():
            index = candidate_argv.index(flag)
            self.assertEqual(json.loads(candidate_argv[index + 1]), value)
            del candidate_argv[index:index + 2]
        self.assertEqual(stock["argv"], candidate_argv)
        for flag, value in {
            "--max-model-len": "131072", "--max-num-batched-tokens": "2048",
            "--max-num-seqs": "16", "--gpu-memory-utilization": "0.92",
            "--attention-backend": "FLASHINFER_MLA_SPARSE_SM120",
            "--all2all-backend": "allgather_reducescatter", "--mamba-cache-mode": "none",
        }.items():
            self.assertEqual(candidate_argv[candidate_argv.index(flag) + 1], value)
        self.assertIn("--language-model-only", candidate_argv)
        self.assertIn("--no-enable-prefix-caching", candidate_argv)
        self.assertEqual(
            json.loads(candidate_argv[candidate_argv.index("--speculative-config") + 1]),
            {"method": "mtp", "num_speculative_tokens": 3},
        )
        self.assertEqual(
            json.loads(candidate_argv[candidate_argv.index("--compilation-config") + 1]),
            {
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 96, 128,
                    192, 256, 384, 512, 768, 1024, 1536, 2048,
                ],
            },
        )

    def test_indexer_capacity_arm_requires_the_fork_policy(self):
        self.assert_rejected(
            "requires an external host policy", "--capacity-comparison", "indexer2048"
        )

    def test_cli_requires_explicit_room_for_all_eight_host_pools(self):
        self.assert_rejected("exceeds the 63 GiB budget", "--host-pool-budget-gib", "63")
        self.assert_rejected("must be positive GiB", "--host-pool-gib-per-rank", "0")

    def test_cli_rejects_partial_duplicate_and_noninteger_device_indices(self):
        for devices in ("0,1,2,3", "0,1,2,3,4,5,5,6", "0,1,2,3,4,5,6,-1"):
            with self.subTest(devices=devices):
                self.assert_rejected("exactly eight distinct", "--devices", devices)
        self.assert_rejected("comma-separated GPU indices", "--devices", "GPU-fake")

    def test_cli_rejects_parallelism_override(self):
        self.assert_rejected("unrecognized arguments", "--tensor-parallel-size", "2")

    def test_stock_topk_override_and_mtp_are_opt_in(self):
        plan = json.loads(
            self.plan("--stock-topk-override", "--mtp", "3", "--prefix-caching")
        )
        argv = plan["argv"]
        self.assertEqual(json.loads(argv[argv.index("--hf-overrides") + 1]), {"index_topk": 2048})
        self.assertEqual(
            json.loads(argv[argv.index("--speculative-config") + 1]),
            {"method": "mtp", "num_speculative_tokens": 3},
        )
        self.assertIn("--enable-prefix-caching", argv)

    def test_serving_limits_use_checkpoint_and_verification_width(self):
        self.assert_rejected("in 1..1048576", "--max-model-len", "1048577")
        self.assert_rejected("MTP verification width", "--max-num-batched-tokens", "63", "--mtp", "3")
        self.assert_rejected("in (0, 1)", "--gpu-memory-utilization", "nan")

    def test_shell_round_trips_args_and_local_env(self):
        json_plan = json.loads(self.plan("--served-model-name", "glm test"))
        shell = shlex.split(self.plan("--served-model-name", "glm test", "--format", "shell"))
        self.assertEqual(
            shell,
            ["env"]
            + [f"{key}={value}" for key, value in json_plan["environment"].items()]
            + json_plan["argv"],
        )

    def test_wrong_quantization_and_conflicting_topk_are_rejected(self):
        self.write_json("hf_quant_config.json", {"quantization": {"quant_algo": "FP8"}})
        self.assert_rejected("Expected a ModelOpt NVFP4 checkpoint")
        self.write_json("hf_quant_config.json", {"quantization": {"quant_algo": "NVFP4"}})
        self.config["index_topk"] = 4096
        self.write_json("config.json", self.config)
        self.assert_rejected("index_topk disagree")

    def test_missing_and_outside_shards_are_rejected(self):
        (self.model_dir / "model-02.safetensors").unlink()
        self.assert_rejected("Missing or empty")
        self.write_json(
            "model.safetensors.index.json",
            {"weight_map": {"head": "../elsewhere.safetensors"}},
        )
        self.assert_rejected("local .safetensors")

    @unittest.skipUnless(
        os.environ.get("KITEPOND_TEST_VLLM_CLI") == "1",
        "requires an installed vLLM v0.34 engine package",
    )
    def test_native_parser_accepts_profile_without_engine_start(self):
        plan = json.loads(self.plan("--check-vllm-cli", "--stock-topk-override"))
        self.assertTrue(plan["vllm_cli_checked"])


if __name__ == "__main__":
    unittest.main()
