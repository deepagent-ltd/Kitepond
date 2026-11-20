# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve GLM DCP rank-local decode rows through native HiSparse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from kitepond.models.glm5next.hisparse_geometry import (
    DCP_INTERLEAVE,
    MANAGER_BLOCK_TOKENS,
    MLA_KERNEL_BLOCK_ROWS,
    MLA_ROW_WIDTHS,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.sparse_mla_attention import (
        SparseMLACommonMetadata,
    )


@dataclass(frozen=True)
class ResolvedHiSparseRows:
    kv_cache: torch.Tensor
    physical_topk: torch.Tensor
    valid_counts: torch.Tensor


@triton.jit
def _store_compacted_rows(
    destination,
    counts,
    row,
    column,
    position,
    valid,
    destination_stride_0: tl.constexpr,
    destination_stride_1: tl.constexpr,
    WIDTH: tl.constexpr,
):
    compact_column = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32))
    tl.store(
        destination
        + row * destination_stride_0
        + compact_column * destination_stride_1,
        position,
        mask=valid,
    )
    tl.store(
        destination + row * destination_stride_0 + column * destination_stride_1,
        -1,
        mask=(column < WIDTH) & (column >= count),
    )
    tl.store(counts + row, count)


@triton.jit
def _compact_physical_rows(
    source,
    destination,
    counts,
    source_stride_0: tl.constexpr,
    source_stride_1: tl.constexpr,
    destination_stride_0: tl.constexpr,
    destination_stride_1: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
    CACHE_ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, TILE)
    position = tl.load(
        source + row * source_stride_0 + column * source_stride_1,
        mask=column < WIDTH,
        other=-1,
    )
    valid = (column < WIDTH) & (position >= 0) & (position < CACHE_ROWS)
    _store_compacted_rows(
        destination,
        counts,
        row,
        column,
        position,
        valid,
        destination_stride_0,
        destination_stride_1,
        WIDTH,
    )


@triton.jit
def _compact_staged_physical_rows(
    source,
    logical,
    query_lens,
    destination,
    counts,
    source_stride_0: tl.constexpr,
    source_stride_1: tl.constexpr,
    logical_stride_0: tl.constexpr,
    logical_stride_1: tl.constexpr,
    query_stride: tl.constexpr,
    destination_stride_0: tl.constexpr,
    destination_stride_1: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
    FIRST_CACHE_ROW: tl.constexpr,
    CACHE_ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, TILE)
    position = tl.load(
        source + row * source_stride_0 + column * source_stride_1,
        mask=column < WIDTH,
        other=-1,
    )
    token = tl.load(
        logical + row * logical_stride_0 + column * logical_stride_1,
        mask=column < WIDTH,
        other=-1,
    )
    visible = tl.load(query_lens + row * query_stride)
    valid = (
        (column < WIDTH)
        & (position >= FIRST_CACHE_ROW)
        & (position < CACHE_ROWS)
        & (token >= 0)
        & (token < visible)
    )
    _store_compacted_rows(
        destination,
        counts,
        row,
        column,
        position,
        valid,
        destination_stride_0,
        destination_stride_1,
        WIDTH,
    )


def _result_buffers(
    rank_local_topk: torch.Tensor,
    physical_topk_out: torch.Tensor | None,
    valid_counts_out: torch.Tensor | None,
    protected_inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, _ = rank_local_topk.shape
    out = (
        torch.empty(
            rank_local_topk.shape, dtype=torch.int32, device=rank_local_topk.device
        )
        if physical_topk_out is None
        else physical_topk_out
    )
    counts = (
        torch.empty(rows, dtype=torch.int32, device=rank_local_topk.device)
        if valid_counts_out is None
        else valid_counts_out
    )
    if (
        out.shape != rank_local_topk.shape
        or counts.shape != (rows,)
        or not out.is_contiguous()
        or not counts.is_contiguous()
        or any(
            t.dtype != torch.int32 or t.device != rank_local_topk.device
            for t in (out, counts)
        )
    ):
        raise ValueError(
            "HiSparse result buffers must be contiguous matching int32 tensors"
        )
    if rows:
        protected = {
            t.untyped_storage().data_ptr() for t in (*protected_inputs, rank_local_topk)
        }
        out_ptr = out.untyped_storage().data_ptr()
        counts_ptr = counts.untyped_storage().data_ptr()
        if out_ptr in protected or counts_ptr in protected or counts_ptr == out_ptr:
            raise ValueError("HiSparse result buffers may not alias native inputs")
    return out, counts


def _compact_native_rows(
    resolved: torch.Tensor,
    out: torch.Tensor,
    counts: torch.Tensor,
    cache_rows: int,
    *,
    staged_limits: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    if (
        not isinstance(resolved, torch.Tensor)
        or resolved.shape != out.shape
        or resolved.dtype != torch.int32
        or resolved.device != out.device
        or resolved.untyped_storage().data_ptr()
        in (out.untyped_storage().data_ptr(), counts.untyped_storage().data_ptr())
    ):
        raise RuntimeError(
            "Native HiSparse resolver returned invalid or aliased physical rows"
        )
    rows, width = out.shape
    if resolved.device.type == "cpu":
        out.fill_(-1)
        for i, row in enumerate(resolved):
            valid = (row >= 0) & (row < cache_rows)
            if staged_limits is not None:
                logical, query_lens = staged_limits
                tokens = logical[i]
                valid &= (
                    (row >= MANAGER_BLOCK_TOKENS)
                    & (tokens >= 0)
                    & (tokens < query_lens[i])
                )
            selected = row[valid]
            out[i, : selected.numel()] = selected
            counts[i] = selected.numel()
        return
    tile = triton.next_power_of_2(width)
    num_warps = 8 if width > 1024 else 4
    if staged_limits is None:
        _compact_physical_rows[(rows,)](
            resolved,
            out,
            counts,
            *resolved.stride(),
            *out.stride(),
            WIDTH=width,
            TILE=tile,
            CACHE_ROWS=cache_rows,
            num_warps=num_warps,
        )
    else:
        logical, query_lens = staged_limits
        _compact_staged_physical_rows[(rows,)](
            resolved,
            logical,
            query_lens,
            out,
            counts,
            *resolved.stride(),
            *logical.stride(),
            query_lens.stride(0),
            *out.stride(),
            WIDTH=width,
            TILE=tile,
            FIRST_CACHE_ROW=MANAGER_BLOCK_TOKENS,
            CACHE_ROWS=cache_rows,
            num_warps=num_warps,
        )


def resolve_hisparse_rank_rows(
    index_group: HiSparseMLAIndexGroup,
    layer_index: int,
    rank_local_topk: torch.Tensor,
    attn_metadata: SparseMLACommonMetadata,
    *,
    query_row_start: int = 0,
    physical_topk_out: torch.Tensor | None = None,
    valid_counts_out: torch.Tensor | None = None,
) -> ResolvedHiSparseRows:
    """Resolve native sources for a decode segment, including resident rows.

    The caller prepares the index-sharing group and signals its logical top-k
    event after rank localization. A mixed batch can pass its decode prefix.
    FULL graph padding remains governed by the device query boundary.
    """
    if not isinstance(index_group, HiSparseMLAIndexGroup):
        raise TypeError("Native HiSparse index group is required")
    if type(layer_index) is not int or not 0 <= layer_index < len(index_group.caches):
        raise ValueError("HiSparse layer index is out of range")
    if type(query_row_start) is not int or query_row_start < 0:
        raise ValueError("HiSparse query row start must be a nonnegative integer")
    if (
        rank_local_topk.ndim != 2
        or rank_local_topk.dtype != torch.int32
        or rank_local_topk.stride(-1) != 1
        or rank_local_topk.device.type not in {"cpu", "cuda"}
        or not 0 < rank_local_topk.shape[1] <= 4096
    ):
        raise ValueError("HiSparse rank-local top-k must be a supported int32 matrix")

    cache = index_group.cache(layer_index)
    runtime = cache.runtime
    rows, _ = rank_local_topk.shape
    query_row_stop = query_row_start + rows
    if rows > runtime.max_swap_rows:
        raise NotImplementedError("Query segment exceeds native decode swap capacity")
    if not 0 <= attn_metadata.num_decode_tokens <= attn_metadata.num_actual_tokens:
        raise ValueError("Native HiSparse decode/prefill split is invalid")
    has_prefill = attn_metadata.num_decode_tokens < attn_metadata.num_actual_tokens
    if has_prefill and query_row_stop > attn_metadata.num_decode_tokens:
        raise NotImplementedError("GLM DCP4 prefill needs rank-local staging")

    request_ids = attn_metadata.req_id_per_token[query_row_start:query_row_stop]
    query_start_loc = attn_metadata.query_start_loc
    source_table = cache.source_block_table
    if (
        attn_metadata.cp_kv_cache_interleave_size != DCP_INTERLEAVE
        or attn_metadata.block_size != MANAGER_BLOCK_TOKENS
        or runtime.hot.block_size != MANAGER_BLOCK_TOKENS
        or (cache.view is not None and cache.view.block_size != MANAGER_BLOCK_TOKENS)
        or request_ids.shape != (rows,)
        or query_start_loc.ndim != 1
        or query_start_loc.numel() == 0
        or source_table is None
        or source_table.ndim != 2
        or any(
            t.dtype != torch.int32 or t.device != rank_local_topk.device
            for t in (request_ids, query_start_loc, source_table)
        )
    ):
        raise ValueError("GLM DCP4 metadata does not match the native source geometry")

    physical_cache = index_group.physical_kv_cache(layer_index)
    if (
        physical_cache.ndim != 3
        or physical_cache.shape[0] == 0
        or physical_cache.shape[1] != MANAGER_BLOCK_TOKENS
        or physical_cache.shape[2] not in MLA_ROW_WIDTHS
        or physical_cache.shape[2] != runtime.row_width
        or not physical_cache.is_contiguous()
        or physical_cache.dtype != torch.uint8
        or physical_cache.device != rank_local_topk.device
    ):
        raise ValueError("HiSparse layer has no matching native MLA physical cache")
    physical_rows, _ = flat_kv_row_view(physical_cache, runtime.hot.block_size)
    if physical_rows.shape[0] % MLA_KERNEL_BLOCK_ROWS:
        raise ValueError("HiSparse physical cache is not aligned to SM120 kernel tiles")
    kernel_cache = physical_rows.view(-1, MLA_KERNEL_BLOCK_ROWS, physical_rows.shape[1])

    out, counts = _result_buffers(
        rank_local_topk,
        physical_topk_out,
        valid_counts_out,
        (request_ids, source_table, query_start_loc, physical_cache),
    )
    if not rows:
        return ResolvedHiSparseRows(kernel_cache, out, counts)

    num_valid_rows = query_start_loc[-1:]
    if query_row_start:
        num_valid_rows = num_valid_rows - query_row_start
    # The stock group-level converter normally diverts an all-resident mixed
    # batch to a prefill-oriented mapping. The decode slice must use the
    # selected handle's native swap path and its authoritative source table.
    resolved = cache.swap_in(
        request_ids,
        block_table=source_table,
        logical_topk_indices=rank_local_topk,
        block_size=attn_metadata.block_size,
        num_valid_rows=num_valid_rows,
        return_valid_counts=False,
    )
    # FlashInfer's length applies to a valid prefix after native lookup.
    _compact_native_rows(resolved, out, counts, physical_rows.shape[0])
    return ResolvedHiSparseRows(kernel_cache, out, counts)
