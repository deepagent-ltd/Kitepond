# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM DCP rank-local attention using the public SM120 FlashInfer kernel."""

import torch

from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_with_kv_cache_mla
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseImpl,
)


def run_sm120_rank_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    physical_topk: torch.Tensor,
    valid_counts: torch.Tensor,
    workspace: torch.Tensor,
    *,
    qk_nope_head_dim: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one rank's physical rows; the engine merges the base2 LSE."""
    if (
        query.ndim != 3
        or query.shape[1:] != (64, 512)
        or kv_cache.ndim != 3
        or kv_cache.shape[1] != 64
        or kv_cache.shape[2] not in {528, 656}
        or kv_cache.dtype != torch.uint8
        or physical_topk.ndim != 2
        or physical_topk.shape[0] != query.shape[0]
        or physical_topk.dtype != torch.int32
        or valid_counts.shape != (query.shape[0],)
        or valid_counts.dtype != torch.int32
        or any(
            item.device != query.device
            for item in (kv_cache, physical_topk, valid_counts, workspace)
        )
    ):
        raise ValueError("GLM rank attention does not match its native physical layout")
    count, heads, _ = query.shape
    topk = physical_topk.shape[1]
    output = query.new_empty((count, heads, 512))
    result = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
        query=query.unsqueeze(1),
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        block_tables=physical_topk.unsqueeze(1),
        seq_lens=valid_counts,
        max_seq_len=topk,
        out=output.unsqueeze(1),
        bmm1_scale=scale,
        bmm2_scale=1.0,
        sparse_mla_top_k=topk,
        kv_scale_format="arbitrary_fp32",
        backend="sparse",
        return_lse=True,
        return_lse_base="base2",
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("GLM DCP attention requires an actual FlashInfer LSE")
    out, raw_lse = result
    out = out.squeeze(1)
    lse = FlashInferMLASparseImpl._normalize_lse(raw_lse, count, heads)
    empty = valid_counts == 0
    out.masked_fill_(empty[:, None, None], 0.0)
    lse.masked_fill_(empty[:, None], float("-inf"))
    return out, lse
