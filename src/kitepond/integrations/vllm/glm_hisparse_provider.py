# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM layout provider for an engine with the explicit HiSparse DCP hooks."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import torch
from kitepond.models.glm5next.hisparse_geometry import (
    DCP_INTERLEAVE,
    DCP_RANKS,
    MANAGER_BLOCK_TOKENS,
    GlmHiSparseGeometry,
)
from kitepond.models.glm5next.hisparse_mirror import localize_row_mirrors
from kitepond.models.glm5next.hisparse_resolver import resolve_hisparse_rank_rows

from .glm_sparse_backend import GlmHiSparseDCPAttentionImpl, GlmSparseAttentionBinding
from .glm_sparse_registration import GLM_HISPARSE_BACKEND, GLM_HISPARSE_BACKEND_PATH

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.platforms.interface import DeviceCapability

_STOCK_ROW_BYTES = 656
_GLM_ARCHITECTURES = {"Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"}
_MODEL_GEOMETRY = {
    "model_type": "glm5_next_text",
    "num_attention_heads": 64,
    "num_hidden_layers": 45,
    "num_nextn_predict_layers": 1,
    "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 0,
    "kv_lora_rank": 512,
    "index_topk": 2048,
    "index_kpool": 4,
    "index_kpool_compress": True,
}


class GlmHiSparseDCPPlanner:
    def plan_groups(self, config, geometry, layer_specs):
        from vllm.v1.hisparse.layout import plan_native_hisparse_groups

        actual = GlmHiSparseGeometry.from_native_specs(
            layer_specs,
            dcp_size=geometry.decode_context_parallel_size,
            interleave_size=geometry.token_interleave,
        )
        if (
            actual.mla.row_bytes != geometry.row_bytes
            or actual.logical_page_tokens != geometry.logical_page_tokens
        ):
            raise ValueError("GLM HiSparse cache groups disagree with launch geometry")
        return plan_native_hisparse_groups(
            config, layer_specs, resident_dcp_sharded=True
        )

    def plan_host_layout(self, config, geometry, groups, total_host_budget_bytes):
        from vllm.v1.hisparse.layout import plan_native_hisparse_host_layout

        ranks = geometry.decode_context_parallel_size
        return plan_native_hisparse_host_layout(
            config,
            groups,
            total_host_budget_bytes // ranks,
            context_parallel_size=ranks,
        )


class GlmHiSparseDCPWorker:
    def __init__(self):
        self._bindings: dict[int, GlmSparseAttentionBinding] = {}

    def bind_physical_views(self, geometry, *, rank, source, resident, hot):
        from vllm.v1.hisparse.dcp_capability import HiSparseDCPPhysicalViews

        if not 0 <= rank < geometry.decode_context_parallel_size:
            raise ValueError("GLM HiSparse cache was bound to an invalid rank")
        if source.ndim == 4 and source.shape[1] == 1:
            source = source.squeeze(1)
        shape = (geometry.physical_page_rows, geometry.row_bytes)
        for tensor, expected_device in (
            (source, "cpu"),
            (resident, "cuda"),
            (hot, "cuda"),
        ):
            if (
                tensor.ndim != 3
                or tensor.shape[1:] != shape
                or tensor.dtype != torch.uint8
                or tensor.device.type != expected_device
                or tensor.stride()[1:] != (geometry.row_bytes, 1)
            ):
                raise ValueError("Native GLM HiSparse backing has incompatible views")
        if not source.is_contiguous():
            raise ValueError("GLM HiSparse host pages must be layer-compact")
        return HiSparseDCPPhysicalViews(source=source, resident=resident, hot=hot)

    def localize_row_mirrors(self, geometry, *, rank, mirrors):
        return localize_row_mirrors(
            mirrors,
            block_size=geometry.physical_page_rows,
            dcp_world_size=geometry.decode_context_parallel_size,
            dcp_rank=rank,
        )

    def bind_attention(self, geometry, *, rank, layer, cache_handle) -> None:
        impl = layer.impl
        if (
            not isinstance(impl, GlmHiSparseDCPAttentionImpl)
            or impl.dcp_rank != rank
            or impl.index_group.cache(impl.index_group_index) is not cache_handle
            or cache_handle.runtime.row_width != geometry.row_bytes
        ):
            raise ValueError("GLM HiSparse binding refers to the wrong layer")
        native_group = impl.index_group
        group_key = id(native_group)
        binding = self._bindings.get(group_key)
        if binding is None:
            if impl.index_group_index != 0:
                raise ValueError("GLM HiSparse needs the index leader first")
            binding = impl.kitepond_hisparse_binding
            if binding is None:
                binding = GlmSparseAttentionBinding.create(
                    native_group, resolve_hisparse_rank_rows
                )
            self._bindings[group_key] = binding
        impl.bind_kitepond_hisparse(binding)


def describe_glm_hisparse_dcp(config: VllmConfig, device: DeviceCapability | None):
    """Return the narrow native compatibility description without launching workers."""
    from vllm.v1.hisparse.dcp_capability import (
        HiSparseDCPExecutionCapability,
        HiSparseDCPGeometry,
    )

    model = config.model_config
    parallel = config.parallel_config
    if (
        model is None
        or model.architecture not in _GLM_ARCHITECTURES
        or device is None
        or (device.major, device.minor) != (12, 0)
        or (
            parallel.tensor_parallel_size,
            parallel.pipeline_parallel_size,
            parallel.data_parallel_size,
            parallel.prefill_context_parallel_size,
            parallel.decode_context_parallel_size,
            parallel.cp_kv_cache_interleave_size,
        )
        != (4, 1, 1, 1, DCP_RANKS, DCP_INTERLEAVE)
        or config.cache_config.block_size != MANAGER_BLOCK_TOKENS
        or config.cache_config.cache_dtype != "fp8_ds_mla"
    ):
        return None
    text = model.hf_text_config
    if any(getattr(text, key, None) != value for key, value in _MODEL_GEOMETRY.items()):
        return None

    layout_identity = _MODEL_GEOMETRY | {
        "architecture": model.architecture,
        "layer_types": list(text.layer_types),
        "indexer_types": list(text.indexer_types),
        "row_format": "fp8_ds_mla",
        "row_bytes": _STOCK_ROW_BYTES,
    }
    layout_sha = hashlib.sha256(
        json.dumps(layout_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    geometry = HiSparseDCPGeometry(
        architecture=model.architecture,
        device=device,
        tensor_parallel_size=parallel.tensor_parallel_size,
        decode_context_parallel_size=DCP_RANKS,
        prefill_context_parallel_size=parallel.prefill_context_parallel_size,
        pipeline_parallel_size=parallel.pipeline_parallel_size,
        token_interleave=DCP_INTERLEAVE,
        physical_page_rows=MANAGER_BLOCK_TOKENS,
        row_bytes=_STOCK_ROW_BYTES,
        configured_cache_dtype="fp8_ds_mla",
        row_format="fp8_ds_mla",
        model_layout_sha256=layout_sha,
    )
    return HiSparseDCPExecutionCapability(
        geometry=geometry,
        planner=GlmHiSparseDCPPlanner(),
        worker=GlmHiSparseDCPWorker(),
        backend_name=GLM_HISPARSE_BACKEND,
        backend_class_path=GLM_HISPARSE_BACKEND_PATH,
    )
