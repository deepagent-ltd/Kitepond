# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM HiSparse attention adapter for the native vLLM DCP reduction."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from kitepond.models.glm5next.hisparse_dcp import localize_hisparse_dcp_topk
from kitepond.models.glm5next.hisparse_prefill import (
    RankLocalPrefillPlan,
    plan_rank_local_prefill,
    resolve_staged_rank_rows,
    stage_rank_local_prefill_rows,
)
from kitepond.models.glm5next.hisparse_resolver import ResolvedHiSparseRows
from kitepond.models.glm5next.sparse_attention import run_sm120_rank_attention

from vllm.distributed.parallel_state import get_dcp_group
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    FlashInferMLASparseMetadataBuilder,
    FlashInferMLASparseSM120Backend,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl,
)
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup

from .glm_sparse_registration import GLM_HISPARSE_BACKEND

# 2,048 routed logical keys plus up to three incomplete-pool tail tokens,
# rounded to the GLM model's 128-wide sparse attention tile.
_GLM_TOPK_BUFFER_WIDTH = 2176


@dataclass
class GlmHiSparseDCPMetadata(FlashInferMLASparseMetadata):
    rank_local_prefill: RankLocalPrefillPlan | None = None


class GlmHiSparseDCPMetadataBuilder(FlashInferMLASparseMetadataBuilder):
    metadata_cls = GlmHiSparseDCPMetadata
    supports_rank_local_hisparse_prefill = True

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        result = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if self.dcp_world_size == 1 or result.num_prefills == 0:
            return result
        if self.dcp_world_size != 4 or result.prefill is not None:
            raise ValueError("GLM HiSparse requires native sparse MQA and DCP4")
        result.rank_local_prefill = plan_rank_local_prefill(
            common_attn_metadata,
            num_decodes=result.num_decodes,
            num_prefills=result.num_prefills,
            dcp_rank=get_dcp_group().rank_in_group,
        )
        return result


@dataclass
class GlmSparseAttentionBinding:
    """Process-local binding made by the engine after native layer setup."""

    group: HiSparseMLAIndexGroup
    resolve: Callable[..., ResolvedHiSparseRows]
    local_topk: torch.Tensor
    local_counts: torch.Tensor

    @classmethod
    def create(cls, group, resolve):
        if not isinstance(group, HiSparseMLAIndexGroup) or not callable(resolve):
            raise TypeError(
                "GLM attention requires a native HiSparse group and resolver"
            )
        topk = group.logical_topk_indices
        if (
            topk.ndim != 2
            or topk.shape[1] != _GLM_TOPK_BUFFER_WIDTH
            or topk.dtype != torch.int32
        ):
            raise ValueError("GLM HiSparse expects padded top-k width 2176")
        return cls(
            group,
            resolve,
            torch.empty_like(topk),
            topk.new_empty((topk.shape[0],)),
        )


class GlmHiSparseDCPAttentionImpl(FlashInferMLASparseSM120Impl):
    can_return_lse_for_decode = True
    supports_dcp = True
    supports_mtp_with_cp_non_trivial_interleave_size = True
    lse_base_on_e = False
    _kitepond_binding: GlmSparseAttentionBinding | None = None

    @property
    def kitepond_hisparse_binding(self) -> GlmSparseAttentionBinding | None:
        return self._kitepond_binding

    def bind_kitepond_hisparse(self, binding: GlmSparseAttentionBinding) -> None:
        if binding.group is not self.index_group or self.dcp_world_size != 4:
            raise ValueError("GLM HiSparse binding does not match this DCP layer")
        if self._kitepond_binding is not None and self._kitepond_binding is not binding:
            raise RuntimeError("GLM HiSparse attention binding is immutable")
        self._kitepond_binding = binding

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        if self.dcp_world_size == 1:
            return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)

        if (
            self.dcp_world_size != 4
            or self.pcp_world_size != 1
            or attn_metadata.cp_kv_cache_interleave_size != 4
            or self.qk_nope_head_dim != 256
            or self.qk_rope_head_dim != 0
            or self.kv_lora_rank != 512
            or self.kv_scale_format != "arbitrary_fp32"
        ):
            raise ValueError("GLM HiSparse DCP4 attention geometry is unsupported")
        binding = self._kitepond_binding
        if binding is None:
            raise RuntimeError("The native GLM HiSparse resolver is not bound")
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        num_tokens = q.shape[0]
        topk = binding.group.logical_topk_indices[:num_tokens]
        if not num_tokens or topk.shape[0] != num_tokens:
            raise ValueError("GLM HiSparse query rows exceed the native top-k envelope")

        layer_index = self.index_group_index
        local = binding.local_topk[:num_tokens]
        if layer_index == 0:
            localize_hisparse_dcp_topk(
                topk,
                dcp_rank=self.dcp_rank,
                out=local,
                valid_counts_out=binding.local_counts[:num_tokens],
            )
            binding.group.set_logical_topk_ready(layer_index)
        plan = getattr(attn_metadata, "rank_local_prefill", None)
        if plan is None:
            if getattr(attn_metadata, "num_prefills", 0):
                raise RuntimeError("GLM HiSparse is missing the native prefill plan")
            resolved = binding.resolve(binding.group, layer_index, local, attn_metadata)
            return self._rank_attention(q, resolved)

        decode_tokens = attn_metadata.num_decode_tokens
        if (plan.query_row_start, plan.query_row_stop) != (decode_tokens, num_tokens):
            raise ValueError("GLM HiSparse prefill/query boundaries disagree")
        decoded = None
        if decode_tokens:
            resolved = binding.resolve(
                binding.group, layer_index, local[:decode_tokens], attn_metadata
            )
            decoded = self._rank_attention(q[:decode_tokens], resolved)

        staged = stage_rank_local_prefill_rows(
            binding.group, layer_index, kv_c_and_k_pe_cache, attn_metadata, plan
        )
        resolved_prefill = resolve_staged_rank_rows(staged, local[decode_tokens:])
        prefill_out, prefill_lse = self._rank_attention(
            q[decode_tokens:], resolved_prefill
        )
        if decoded is None:
            return prefill_out, prefill_lse
        return (
            torch.cat((decoded[0], prefill_out)),
            torch.cat((decoded[1], prefill_lse)),
        )

    def _rank_attention(self, q, resolved: ResolvedHiSparseRows):
        if resolved.kv_cache.shape[-1] != 656:
            raise ValueError("GLM HiSparse stock attention requires 656-byte rows")
        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)
        return run_sm120_rank_attention(
            q,
            resolved.kv_cache,
            resolved.physical_topk,
            resolved.valid_counts,
            self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            scale=self.scale,
        )


class GlmHiSparseDCPBackend(FlashInferMLASparseSM120Backend):
    """Opt-in registry alias using the native 656-byte SM120 format."""

    registry_alias = GLM_HISPARSE_BACKEND

    @staticmethod
    def supports_hisparse_dcp_layout(row_bytes, dcp_size, interleave) -> bool:
        return (row_bytes, dcp_size, interleave) == (656, 4, 4)

    @staticmethod
    def get_builder_cls() -> type[GlmHiSparseDCPMetadataBuilder]:
        return GlmHiSparseDCPMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[GlmHiSparseDCPAttentionImpl]:
        return GlmHiSparseDCPAttentionImpl
