"""Inspect local GLM checkpoint metadata without loading weights or an engine."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

FLASH_ARCHITECTURE = "Glm5NextForConditionalGeneration"
FULL_ARCHITECTURE = "GlmMoeDsaForCausalLM"
INDEX_TOPK = 2048


class CheckpointKind(StrEnum):
    FLASH_NVFP4 = "glm53-flash-nvfp4"
    FLASH_FP8 = "glm53-flash-fp8"
    MODEL_OPT_MIXED = "glm53-modelopt-mixed"


@dataclass(frozen=True)
class CheckpointMetadata:
    path: Path
    kind: CheckpointKind
    architecture: str
    context_limit: int
    index_topk: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    shard_count: int
    shard_file_bytes: int
    declared_tensor_bytes: int | None


def model_json(model_dir: Path, filename: str) -> dict:
    path = model_dir / filename
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return result


def _sidecar_quantization(path: Path) -> dict:
    value = model_json(path, "hf_quant_config.json").get("quantization")
    return value if isinstance(value, dict) else {}


def _flash_kind(path: Path, quantization: dict) -> CheckpointKind:
    if quantization.get("quant_method") == "fp8":
        if (
            quantization.get("fmt") != "e4m3"
            or quantization.get("activation_scheme") != "dynamic"
            or quantization.get("weight_block_size") != [128, 128]
        ):
            raise ValueError("Expected native block-quantized E4M3 Flash FP8 metadata")
        if (path / "hf_quant_config.json").exists():
            raise ValueError("Native Flash FP8 metadata conflicts with a ModelOpt sidecar")
        return CheckpointKind.FLASH_FP8

    sidecar = _sidecar_quantization(path)
    if (
        sidecar.get("quant_algo") != "NVFP4"
        or quantization.get("quant_method", "modelopt") != "modelopt"
        or quantization.get("quant_algo", "NVFP4") != "NVFP4"
    ):
        raise ValueError("Expected a ModelOpt NVFP4 checkpoint")
    return CheckpointKind.FLASH_NVFP4


def _full_kind(path: Path, quantization: dict) -> CheckpointKind:
    sidecar = _sidecar_quantization(path)
    if (
        quantization.get("quant_method") != "modelopt"
        or quantization.get("quant_algo") != "MIXED_PRECISION"
        or sidecar.get("quant_algo") != "MIXED_PRECISION"
    ):
        raise ValueError("Expected a mixed-precision ModelOpt GLM-5.3 checkpoint")
    layers = quantization.get("quantized_layers")
    if not isinstance(layers, dict) or layers != sidecar.get("quantized_layers"):
        raise ValueError("Mixed-precision checkpoint and sidecar layer tables disagree")
    if any(
        not isinstance(layer, dict) or layer.get("quant_algo") not in ("NVFP4", "FP8")
        for layer in layers.values()
    ) or {layer["quant_algo"] for layer in layers.values()} != {"NVFP4", "FP8"}:
        raise ValueError("Mixed-precision checkpoint must describe both FP8 and NVFP4 layers")
    return CheckpointKind.MODEL_OPT_MIXED


def _inspect_shards(path: Path) -> tuple[int, int, int | None]:
    index = model_json(path, "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Checkpoint requires a non-empty safetensors weight_map")
    filenames = set()
    for filename in weight_map.values():
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".safetensors")
        ):
            raise ValueError("Weight index must refer to local .safetensors files")
        filenames.add(filename)

    total_bytes = 0
    for filename in sorted(filenames):
        shard = path / filename
        try:
            info = shard.stat()
        except OSError as exc:
            raise ValueError(f"Missing or empty model shard: {shard}") from exc
        if not stat.S_ISREG(info.st_mode) or not info.st_size:
            raise ValueError(f"Missing or empty model shard: {shard}")
        total_bytes += info.st_size
    metadata = index.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Safetensors index metadata must be a JSON object")
    tensor_bytes = metadata.get("total_size")
    if tensor_bytes is not None and (
        type(tensor_bytes) is not int or not 0 < tensor_bytes <= total_bytes
    ):
        raise ValueError("Invalid total_size in safetensors index metadata")
    return len(filenames), total_bytes, tensor_bytes


def inspect_checkpoint(model_dir: Path) -> CheckpointMetadata:
    """Describe a recognized checkpoint without asserting any launch support."""
    path = model_dir.expanduser().resolve()
    config = model_json(path, "config.json")
    architectures = config.get("architectures")
    model_type = config.get("model_type")
    quantization = config.get("quantization_config", {})
    if not isinstance(quantization, dict):
        raise ValueError("quantization_config must be a JSON object")

    if architectures == [FLASH_ARCHITECTURE] and model_type == "glm5_next":
        architecture = FLASH_ARCHITECTURE
        text = config.get("text_config")
        if not isinstance(text, dict) or text.get("model_type") != "glm5_next_text":
            raise ValueError("Expected a GLM-5.3-Flash sparse MLA checkpoint")
        expected_layers = 45
        read_kind = _flash_kind
    elif architectures == [FULL_ARCHITECTURE] and model_type == "glm_moe_dsa":
        architecture = FULL_ARCHITECTURE
        text = config
        expected_layers = 78
        read_kind = _full_kind
    else:
        raise ValueError("Expected a GLM-5.3-Flash or full GLM-5.3 sparse MLA checkpoint")

    if text.get("num_hidden_layers") != expected_layers or text.get("index_topk") != INDEX_TOPK:
        raise ValueError("Unsupported GLM layer count or sparse MLA index_topk")
    if config.get("index_topk", INDEX_TOPK) != INDEX_TOPK:
        raise ValueError("Checkpoint root and text-config index_topk disagree")
    context_limit = text.get("max_position_embeddings")
    if type(context_limit) is not int or context_limit <= 0:
        raise ValueError("Checkpoint must declare a positive max_position_embeddings")
    layer_types = text.get("layer_types")
    if layer_types is not None and (
        not isinstance(layer_types, list)
        or len(layer_types) != expected_layers
        or not all(isinstance(layer, str) for layer in layer_types)
    ):
        raise ValueError("Checkpoint layer_types must describe every layer")

    kind = read_kind(path, quantization)
    shard_count, shard_bytes, tensor_bytes = _inspect_shards(path)
    return CheckpointMetadata(
        path, kind, architecture, context_limit, INDEX_TOPK, expected_layers,
        tuple(layer_types or ()), shard_count, shard_bytes, tensor_bytes,
    )
