# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM HiSparse DCP4 token coordinates; no cache-address conversion here."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _localize_topk_kernel(
    source,
    destination,
    counts,
    source_stride_0: tl.constexpr,
    source_stride_1: tl.constexpr,
    destination_stride_0: tl.constexpr,
    destination_stride_1: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
    RANK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, TILE)
    token = tl.load(
        source + row * source_stride_0 + column * source_stride_1,
        mask=column < WIDTH,
        other=-1,
    )
    # GLM kpool IDs have already been expanded into global token IDs by
    # the indexer. Token ownership remains interleave=4, not pool interleave=1.
    valid = (column < WIDTH) & (token >= 0) & ((token // 4) % 4 == RANK)
    local = (token // 16) * 4 + token % 4
    position = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32))
    # These writes are disjoint: compact valid prefix, then the invalid tail.
    # Stable compaction preserves selected duplicates and selection order.
    tl.store(
        destination + row * destination_stride_0 + position * destination_stride_1,
        local,
        mask=valid,
    )
    tl.store(
        destination + row * destination_stride_0 + column * destination_stride_1,
        -1,
        mask=(column < WIDTH) & (column >= count),
    )
    tl.store(counts + row, count)


def localize_hisparse_dcp_topk(
    global_topk: torch.Tensor,
    *,
    dcp_rank: int,
    dcp_world_size: int = 4,
    cp_kv_cache_interleave_size: int = 4,
    out: torch.Tensor | None = None,
    valid_counts_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return compact request-local positions for this GLM DCP rank.

    Output contains logical positions in this rank's host/resident history,
    not ordinary GPU physical slots. HiSparse performs the physical lookup.
    Optional persistent outputs avoid allocation in target/draft FULL graphs.
    The CPU path is a semantic test path; CUDA execution uses the Triton kernel.
    """
    if dcp_world_size != 4 or cp_kv_cache_interleave_size != 4:
        raise ValueError("GLM HiSparse requires DCP4 and token interleave4")
    if type(dcp_rank) is not int or not 0 <= dcp_rank < 4:
        raise ValueError("Invalid GLM HiSparse DCP rank")
    if global_topk.dtype != torch.int32 or global_topk.ndim != 2:
        raise ValueError("GLM HiSparse top-k must be a 2-D int32 tensor")
    rows, width = global_topk.shape
    if not 0 < width <= 4096:
        raise ValueError("GLM HiSparse top-k width exceeds its supported envelope")
    if global_topk.device.type not in {"cpu", "cuda"}:
        raise ValueError("GLM HiSparse supports CPU semantic tests and CUDA only")
    if out is None:
        out = torch.empty_like(global_topk)
    if (
        out.shape != global_topk.shape
        or out.dtype != torch.int32
        or out.device != global_topk.device
    ):
        raise ValueError("GLM HiSparse local output shape/dtype/device mismatch")
    if (
        global_topk.numel()
        and out.untyped_storage().data_ptr() == global_topk.untyped_storage().data_ptr()
    ):
        raise ValueError("GLM HiSparse must preserve the shared global top-k buffer")
    if valid_counts_out is None:
        valid_counts_out = torch.empty(
            rows, dtype=torch.int32, device=global_topk.device
        )
    if (
        valid_counts_out.shape != (rows,)
        or valid_counts_out.dtype != torch.int32
        or valid_counts_out.device != global_topk.device
        or not valid_counts_out.is_contiguous()
    ):
        raise ValueError("GLM HiSparse valid-count output mismatch")
    if rows == 0:
        return out, valid_counts_out
    if global_topk.device.type == "cpu":
        # Keep the real address-domain API testable without a CUDA context.
        out.fill_(-1)
        for row in range(rows):
            tokens = global_topk[row]
            owned = tokens[(tokens >= 0) & ((tokens // 4) % 4 == dcp_rank)]
            count = owned.numel()
            out[row, :count] = (owned // 16) * 4 + owned % 4
            valid_counts_out[row] = count
    else:
        _localize_topk_kernel[(rows,)](
            global_topk,
            out,
            valid_counts_out,
            global_topk.stride(0),
            global_topk.stride(1),
            out.stride(0),
            out.stride(1),
            WIDTH=width,
            TILE=triton.next_power_of_2(width),
            RANK=dcp_rank,
            num_warps=8 if width > 1024 else 4,
        )
    return out, valid_counts_out
