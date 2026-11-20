import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from kitepond import dp8, glm_native
from kitepond.checkpoints import (
    FLASH_ARCHITECTURE,
    FULL_ARCHITECTURE,
    CheckpointKind,
    inspect_checkpoint,
)


class CheckpointInspectionTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.path = Path(workspace.name) / "directory-name-does-not-identify-model"
        self.path.mkdir()
        self.write_json(
            "model.safetensors.index.json",
            {
                "metadata": {"total_size": 7},
                "weight_map": {
                    "embedding": "model-1.safetensors",
                    "layer": "model-2.safetensors",
                    "head": "model-2.safetensors",
                },
            },
        )
        (self.path / "model-1.safetensors").write_bytes(b"first")
        (self.path / "model-2.safetensors").write_bytes(b"second")

    def write_json(self, name, data):
        (self.path / name).write_text(json.dumps(data), encoding="utf-8")

    def write_flash_config(self, quantization):
        self.write_json(
            "config.json",
            {
                "architectures": [FLASH_ARCHITECTURE],
                "model_type": "glm5_next",
                "quantization_config": quantization,
                "text_config": {
                    "model_type": "glm5_next_text",
                    "num_hidden_layers": 45,
                    "index_topk": 2048,
                    "max_position_embeddings": 1048576,
                    "layer_types": (
                        (["linear_attention"] * 3 + ["deepseek_sparse_attention"]) * 11
                        + ["linear_attention"]
                    ),
                },
            },
        )

    def write_full_config(self, sidecar_layers=None):
        quantized_layers = {
            "model.layers.0.mlp.gate_proj": {"quant_algo": "FP8"},
            "model.layers.3.mlp.experts.*": {"quant_algo": "NVFP4"},
        }
        self.write_json(
            "config.json",
            {
                "architectures": [FULL_ARCHITECTURE],
                "model_type": "glm_moe_dsa",
                "num_hidden_layers": 78,
                "index_topk": 2048,
                "max_position_embeddings": 1048576,
                "layer_types": ["deepseek_sparse_attention"] * 78,
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": quantized_layers,
                },
            },
        )
        self.write_json(
            "hf_quant_config.json",
            {
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": quantized_layers if sidecar_layers is None else sidecar_layers,
                },
            },
        )

    def test_native_flash_fp8_uses_embedded_metadata_and_nested_geometry(self):
        self.write_flash_config(
            {
                "quant_method": "fp8", "fmt": "e4m3",
                "activation_scheme": "dynamic", "weight_block_size": [128, 128],
            },
        )
        checkpoint = inspect_checkpoint(self.path)
        self.assertEqual(checkpoint.kind, CheckpointKind.FLASH_FP8)
        self.assertEqual(checkpoint.architecture, FLASH_ARCHITECTURE)
        self.assertEqual(checkpoint.context_limit, 1048576)
        self.assertEqual(checkpoint.index_topk, 2048)
        self.assertEqual(checkpoint.num_hidden_layers, 45)
        self.assertEqual(checkpoint.layer_types.count("deepseek_sparse_attention"), 11)
        self.assertEqual(checkpoint.shard_count, 2)
        self.assertEqual(checkpoint.shard_file_bytes, 11)
        self.assertEqual(checkpoint.declared_tensor_bytes, 7)
        self.assertEqual(
            dp8.Checkpoint.inspect(self.path),
            dp8.Checkpoint(self.path, 1048576, 2, 2048),
        )

    def test_upstream_device_profiles_keep_their_dtype_and_native_allocator(self):
        self.write_flash_config({
            "quant_method": "fp8", "fmt": "e4m3",
            "activation_scheme": "dynamic", "weight_block_size": [128, 128],
        })
        for profile, dtype in (
            ("h100-bf16", "bfloat16"), ("h200-bf16", "bfloat16"),
            ("b200-bf16", "bfloat16"), ("b200-fp8", "fp8"),
        ):
            with self.subTest(profile=profile), redirect_stdout(StringIO()) as output:
                glm_native.main([
                    "--profile", profile, "--model", str(self.path), "--host-pool-gib", "32",
                ])
                plan = json.loads(output.getvalue())
                argv = plan["argv"]
                self.assertEqual(plan["qualification"], "upstream_reference_only")
                self.assertEqual(plan["host_allocator"], "vllm_native")
                self.assertEqual(argv[argv.index("--kv-cache-dtype") + 1], dtype)
                self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], "4")
                self.assertEqual(argv[argv.index("--decode-context-parallel-size") + 1], "1")
                connector = json.loads(argv[argv.index("--kv-transfer-config") + 1])
                self.assertEqual(connector["kv_connector_extra_config"], {"host_pool_gib": 32})
                if profile == "h100-bf16":
                    self.assertEqual(argv[argv.index("--attention-backend") + 1], "FLASHMLA_SPARSE")

    def test_upstream_flash_recipe_rejects_full_modelopt_checkpoint(self):
        self.write_full_config()
        with redirect_stderr(StringIO()) as error, self.assertRaises(SystemExit):
            glm_native.main([
                "--profile", "h200-bf16", "--model", str(self.path), "--host-pool-gib", "32",
            ])
        self.assertIn("require GLM-5.3-Flash FP8 weights", error.getvalue())

    def test_native_flash_fp8_rejects_incompatible_block_quantization(self):
        self.write_flash_config(
            {
                "quant_method": "fp8", "fmt": "e4m3",
                "activation_scheme": "dynamic", "weight_block_size": [64, 128],
            },
        )
        with self.assertRaisesRegex(ValueError, "block-quantized E4M3"):
            inspect_checkpoint(self.path)

    def test_flash_nvfp4_keeps_the_existing_dp8_launch_checkpoint(self):
        self.write_flash_config({"quant_method": "modelopt", "quant_algo": "NVFP4"})
        self.write_json("hf_quant_config.json", {"quantization": {"quant_algo": "NVFP4"}})
        examined = inspect_checkpoint(self.path)
        self.assertEqual(examined.kind, CheckpointKind.FLASH_NVFP4)
        self.assertEqual(
            dp8.Checkpoint.inspect(self.path),
            dp8.Checkpoint(self.path, 1048576, 2, 2048),
        )

    def test_full_mixed_modelopt_has_root_geometry_and_is_not_a_flash_launch(self):
        self.write_full_config()
        examined = inspect_checkpoint(self.path)
        self.assertEqual(examined.kind, CheckpointKind.MODEL_OPT_MIXED)
        self.assertEqual(examined.architecture, FULL_ARCHITECTURE)
        self.assertEqual(examined.num_hidden_layers, 78)
        self.assertEqual(examined.layer_types, ("deepseek_sparse_attention",) * 78)
        with self.assertRaisesRegex(ValueError, "DP8 launch profile requires"):
            dp8.Checkpoint.inspect(self.path)

    def test_full_mixed_modelopt_rejects_disagreeing_sidecar(self):
        self.write_full_config(sidecar_layers={"model.layers.0.mlp.gate_proj": {"quant_algo": "NVFP4"}})
        with self.assertRaisesRegex(ValueError, "layer tables disagree"):
            inspect_checkpoint(self.path)

    def test_shard_validation_catches_inconsistent_declared_tensor_size(self):
        self.write_full_config()
        self.write_json(
            "model.safetensors.index.json",
            {"metadata": {"total_size": 1024}, "weight_map": {"layer": "model-1.safetensors"}},
        )
        with self.assertRaisesRegex(ValueError, "Invalid total_size"):
            inspect_checkpoint(self.path)
