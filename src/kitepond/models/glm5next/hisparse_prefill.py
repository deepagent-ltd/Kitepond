# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-local producer prefill staging for native GLM HiSparse."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import gcd
from typing import TYPE_CHECKING

import torch

from kitepond.models.glm5next.hisparse_geometry import (
    DCP_INTERLEAVE,
    DCP_RANKS,
    MANAGER_BLOCK_TOKENS,
    MLA_KERNEL_BLOCK_ROWS,
    MLA_ROW_WIDTHS,
)
from kitepond.models.glm5next.hisparse_resolver import (
    ResolvedHiSparseRows,
    _compact_native_rows,
    _result_buffers,
)
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.hisparse.runtime import (
    HiSparsePrefillStagingPlan,
    build_hisparse_prefill_staging_plan,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.sparse_mla_attention import (
        SparseMLACommonMetadata,
    )
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.hisparse.runtime import HiSparseCacheHandle

_INT_DTYPES = (torch.int32, torch.int64)
_CYCLE = DCP_RANKS * DCP_INTERLEAVE


@dataclass(frozen=True)
class RankLocalPrefillPlan:
    native: HiSparsePrefillStagingPlan
    dcp_rank: int
    num_decodes: int
    num_prefills: int
    query_row_start: int
    query_row_stop: int
    query_starts: torch.Tensor
    global_seq_lens: torch.Tensor
    local_seq_lens: torch.Tensor


@dataclass(frozen=True)
class StagedPrefillRows:
    kv_cache: torch.Tensor
    block_table: torch.Tensor
    request_ids: torch.Tensor
    local_seq_lens: torch.Tensor
    local_query_lens: torch.Tensor


@dataclass(frozen=True)
class _NativePrefill:
    host_staging_plan: HiSparsePrefillStagingPlan


@dataclass(frozen=True)
class _NativeStageMetadata:
    prefill: _NativePrefill
    num_decodes: int
    num_decode_tokens: int
    req_id_per_token: torch.Tensor


def _rank_rows(length: torch.Tensor, rank: int) -> torch.Tensor:
    cycles = length // _CYCLE
    remainder = length % _CYCLE - rank * DCP_INTERLEAVE
    return cycles * DCP_INTERLEAVE + remainder.clamp(0, DCP_INTERLEAVE)


def plan_rank_local_prefill(
    common: CommonAttentionMetadata,
    *,
    num_decodes: int,
    num_prefills: int,
    dcp_rank: int,
) -> RankLocalPrefillPlan:
    """Build the stock staging map using physical rank-local sequence lengths."""
    if type(dcp_rank) is not int or not 0 <= dcp_rank < DCP_RANKS:
        raise ValueError("P staging requires one valid DCP4 rank")
    if (
        type(num_decodes) is not int
        or type(num_prefills) is not int
        or num_decodes < 0
        or num_prefills <= 0
        or num_decodes + num_prefills > common.num_reqs
    ):
        raise ValueError("Invalid native decode/prefill request split")

    end = num_decodes + num_prefills
    q_cpu = common.query_start_loc_cpu
    q_gpu = common.query_start_loc
    lengths = common.dcp_local_seq_lens
    bounds = common.dcp_local_seq_lens_cpu_upper_bound
    global_lengths = common.seq_lens
    table = common.block_table_tensor
    if (
        lengths is None
        or q_cpu.device.type != "cpu"
        or q_cpu.ndim != 1
        or q_cpu.dtype not in _INT_DTYPES
        or q_cpu.numel() <= end
        or table.dtype != torch.int32
        or table.ndim != 2
        or table.shape[0] < end
        or table.shape[1] == 0
        or any(
            t.ndim != 1 or t.dtype not in _INT_DTYPES or t.device != table.device
            for t in (lengths, global_lengths, q_gpu)
        )
        or min(lengths.numel(), global_lengths.numel()) < end
        or q_gpu.numel() <= end
    ):
        raise ValueError("Missing native rank-local prefill metadata")

    q_cpu = q_cpu[num_decodes : end + 1]
    query_row_start = int(q_cpu[0])
    query_row_stop = int(q_cpu[-1])
    if (
        query_row_start < 0
        or query_row_stop > common.num_actual_tokens
        or bool((q_cpu.diff() <= 0).any())
    ):
        raise ValueError("Invalid native prefill query boundaries")

    if bounds is None:
        global_upper = common.seq_lens_cpu_upper_bound
        if (
            global_upper is None
            or global_upper.ndim != 1
            or global_upper.device.type != "cpu"
            or global_upper.dtype not in _INT_DTYPES
            or global_upper.numel() < end
            or bool((global_upper[num_decodes:end] < 0).any())
        ):
            raise ValueError("Native P staging has no usable CPU length bound")
        bounds = (
            (global_upper.to(torch.int64) + _CYCLE - 1) // _CYCLE
        ) * DCP_INTERLEAVE
    if (
        bounds.ndim != 1
        or bounds.device.type != "cpu"
        or bounds.dtype not in _INT_DTYPES
        or bounds.numel() < end
    ):
        raise ValueError("Invalid native rank-local CPU length bound")
    bounds = bounds[num_decodes:end].to(torch.int64)
    blocks = (bounds + MANAGER_BLOCK_TOKENS - 1) // MANAGER_BLOCK_TOKENS
    if bool((bounds < 0).any()) or int(blocks.max()) > table.shape[1]:
        raise ValueError("Prefill rank bounds exceed the native source block table")

    local_lens = lengths[num_decodes:end]
    global_lens = global_lengths[num_decodes:end]
    # Verify native device lengths against the selected rank without a CPU sync.
    torch._assert_async(
        (
            (local_lens == _rank_rows(global_lens, dcp_rank))
            & (global_lens >= 0)
            & (local_lens <= bounds.to(device=table.device, non_blocking=True))
        ).all(),
        "Native P staging length does not belong to this DCP rank",
    )
    native = build_hisparse_prefill_staging_plan(
        table[num_decodes:end], local_lens, MANAGER_BLOCK_TOKENS, int(blocks.sum())
    )
    required_blocks = (local_lens + MANAGER_BLOCK_TOKENS - 1) // MANAGER_BLOCK_TOKENS
    page_order = torch.arange(table.shape[1], device=table.device)
    torch._assert_async(
        ((native.block_table > 0) | (page_order >= required_blocks[:, None])).all(),
        "A live rank-local P history page has no native source block",
    )
    return RankLocalPrefillPlan(
        native=native,
        dcp_rank=dcp_rank,
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        query_row_start=query_row_start,
        query_row_stop=query_row_stop,
        query_starts=q_gpu[num_decodes : end + 1],
        global_seq_lens=global_lens,
        local_seq_lens=local_lens,
    )


def _validate_current_writer_slots(
    cache: HiSparseCacheHandle,
    state_indices: torch.Tensor,
    plan: RankLocalPrefillPlan,
    staging: HiSparsePrefillStagingPlan,
    request_ids: torch.Tensor,
) -> torch.Tensor:
    # Plan source indices are per resident group. Use a layer-local copy of the
    # plan because different groups can bind distinct persistent tables.
    resident = cache.view
    slots = cache.slot_mapping
    resident_table = cache.block_table
    if resident is None or slots is None or resident_table is None:
        raise ValueError("P staging needs the native layer's resident writer bindings")
    start, stop = plan.query_row_start, plan.query_row_stop
    if (
        resident.block_size != MANAGER_BLOCK_TOKENS
        or resident_table.ndim != 2
        or resident_table.dtype != torch.int32
        or state_indices.dtype not in _INT_DTYPES
        or resident_table.shape[1] == 0
        or slots.ndim != 1
        or slots.dtype not in _INT_DTYPES
        or slots.numel() < stop
        or any(
            t.device != request_ids.device
            for t in (resident.cache, resident_table, slots, state_indices)
        )
    ):
        raise ValueError("Native P resident writer geometry does not match staging")
    prefill_state = state_indices[
        plan.num_decodes : plan.num_decodes + plan.num_prefills
    ]
    torch._assert_async(
        ((prefill_state >= 0) & (prefill_state < resident_table.shape[0])).all(),
        "Native P staging requires valid resident states for all prefill requests",
    )
    staging.ensure_gpu_sources(resident_table, prefill_state, resident.block_size)
    gpu_rows = staging.gpu_row_ids
    if gpu_rows is None:
        raise ValueError("Native P staging has no resident source map")

    query_positions = torch.arange(start, stop, dtype=torch.int64, device=slots.device)
    starts = plan.query_starts.to(torch.int64)
    expected_ids = torch.searchsorted(
        starts[1:].contiguous(), query_positions, right=True
    )
    torch._assert_async(
        ((expected_ids == request_ids) & (expected_ids < plan.num_prefills)).all(),
        "Prefill query ordering differs from the native request split",
    )
    reqs = expected_ids.clamp(0, plan.num_prefills - 1)
    global_prefix = plan.global_seq_lens.to(torch.int64) - starts.diff()
    torch._assert_async(
        (global_prefix >= 0).all(),
        "Native P query count exceeds its completed sequence",
    )
    positions = global_prefix[reqs] + query_positions - starts[reqs]
    on_rank = ((positions // DCP_INTERLEAVE) % DCP_RANKS) == plan.dcp_rank
    local_positions = positions // _CYCLE * DCP_INTERLEAVE + positions % DCP_INTERLEAVE
    columns = local_positions // MANAGER_BLOCK_TOKENS
    offsets = local_positions % MANAGER_BLOCK_TOKENS
    table = staging.block_table
    blocks = table[reqs, columns.clamp(0, table.shape[1] - 1)].to(torch.int64)
    staged_index = blocks * MANAGER_BLOCK_TOKENS + offsets
    slots = slots[start:stop].to(torch.int64)
    sources = gpu_rows.flatten().index_select(0, staged_index)
    torch._assert_async(
        torch.where(
            on_rank,
            (columns >= 0)
            & (columns < table.shape[1])
            & (local_positions < plan.local_seq_lens[reqs])
            & (blocks > 0)
            & (slots >= 0)
            & (sources == slots),
            slots < 0,
        ).all(),
        "Native P stage does not select this rank's pending writer slots",
    )

    # One staged page can be shared by many prefixes. Its prefix ownership
    # ends at the largest prefix boundary among the requests using that page.
    page_offsets = (
        torch.arange(table.shape[1], device=table.device, dtype=torch.int64)
        * MANAGER_BLOCK_TOKENS
    )
    prefix_counts = _rank_rows(global_prefix, plan.dcp_rank)[:, None] - page_offsets
    prefix_counts = prefix_counts.clamp(0, MANAGER_BLOCK_TOKENS)
    max_prefix = torch.zeros(
        staging.row_ids.numel() // MANAGER_BLOCK_TOKENS,
        dtype=torch.int64,
        device=table.device,
    )
    max_prefix.scatter_reduce_(
        0, table.flatten().to(torch.int64), prefix_counts.flatten(), reduce="amax"
    )
    torch._assert_async(
        ((~on_rank) | (offsets >= max_prefix[blocks])).all(),
        "Native P writer aliases a retained prefix row",
    )
    current_claims = torch.zeros(
        gpu_rows.numel(), device=slots.device, dtype=torch.int32
    )
    current_claims.scatter_add_(0, staged_index, on_rank.to(torch.int32))
    torch._assert_async(
        (current_claims <= 1).all(), "Multiple native P writers claim one staged row"
    )

    # Every Q is still globally causal, even when its own KV belongs to another
    # rank. Clamp at the final live per-request length as a second boundary.
    return torch.minimum(
        _rank_rows(positions + 1, plan.dcp_rank), plan.local_seq_lens[reqs]
    ).to(torch.int32)


def stage_rank_local_prefill_rows(
    index_group: HiSparseMLAIndexGroup,
    layer_index: int,
    source_cache: torch.Tensor,
    attn_metadata: SparseMLACommonMetadata,
    plan: RankLocalPrefillPlan,
) -> StagedPrefillRows:
    """Gather the prefill suffix after the worker and layer's normal write fences.

    Call inside the layer's attention path after the native current KV write.
    Worker start-step ordering must already have fenced previous host writes.
    Current matching writer rows use resident storage, which lets this gather
    overlap the current step's asynchronous resident-to-host mirror.
    """
    if not isinstance(index_group, HiSparseMLAIndexGroup):
        raise TypeError("Native HiSparse index group is required")
    if (
        type(layer_index) is not int
        or not 0 <= layer_index < len(index_group.caches)
        or attn_metadata.cp_kv_cache_interleave_size != DCP_INTERLEAVE
        or attn_metadata.block_size != MANAGER_BLOCK_TOKENS
        or attn_metadata.num_decodes != plan.num_decodes
        or attn_metadata.num_prefills != plan.num_prefills
        or attn_metadata.num_decode_tokens != plan.query_row_start
        or attn_metadata.num_actual_tokens < plan.query_row_stop
    ):
        raise ValueError("Native layer metadata and rank-local P plan differ")

    cache = index_group.cache(layer_index)
    host = cache.runtime.host_cache
    state_indices = cache.runtime.request_state_indices
    req_ids = attn_metadata.req_id_per_token[: plan.query_row_stop]
    if (
        req_ids.shape != (plan.query_row_stop,)
        or req_ids.dtype != torch.int32
        or req_ids.device != plan.native.block_table.device
        or state_indices is None
        or state_indices.ndim != 1
        or state_indices.numel() < plan.num_decodes + plan.num_prefills
        or source_cache.device.type != "cpu"
        or source_cache.dtype != torch.uint8
        or source_cache.ndim != 3
        or not source_cache.is_contiguous()
        or source_cache.shape[1] != MANAGER_BLOCK_TOKENS
        or source_cache.shape[2] not in MLA_ROW_WIDTHS
        or source_cache.shape[2] != cache.runtime.row_width
        or host.data_ptr() != source_cache.data_ptr()
        or host.numel() != source_cache.numel()
    ):
        raise ValueError("Invalid native P source backing or request state")

    staging = replace(plan.native)
    local_query_lens = _validate_current_writer_slots(
        cache,
        state_indices,
        plan,
        staging,
        req_ids[plan.query_row_start :] - plan.num_decodes,
    )
    # Stock sparse backends can omit dense-prefill metadata. The native
    # gather consumes these four fields, without requiring a dense backend.
    native_metadata = _NativeStageMetadata(
        _NativePrefill(staging), plan.num_decodes, plan.query_row_start, req_ids
    )
    kv_cache, table, prefill_ids = index_group.stage_prefill_rows(
        layer_index, source_cache, native_metadata
    )
    return StagedPrefillRows(
        kv_cache, table, prefill_ids, plan.local_seq_lens, local_query_lens
    )


def resolve_staged_rank_rows(
    staged: StagedPrefillRows,
    rank_local_topk: torch.Tensor,
    *,
    physical_topk_out: torch.Tensor | None = None,
    valid_counts_out: torch.Tensor | None = None,
) -> ResolvedHiSparseRows:
    """Resolve the prefill suffix against the selected rank's native staged pages."""
    if (
        rank_local_topk.ndim != 2
        or rank_local_topk.dtype != torch.int32
        or rank_local_topk.device.type not in {"cpu", "cuda"}
        or not 0 < rank_local_topk.shape[1] <= 4096
    ):
        raise ValueError("Staged HiSparse top-k must be a supported int32 matrix")
    rows, width = rank_local_topk.shape
    page_cache = staged.kv_cache
    page_table = staged.block_table
    req_ids = staged.request_ids
    live_lengths = staged.local_seq_lens
    query_lens = staged.local_query_lens
    if (
        page_cache.ndim != 3
        or page_cache.shape[0] == 0
        or page_cache.shape[1] != MANAGER_BLOCK_TOKENS
        or page_cache.shape[2] not in MLA_ROW_WIDTHS
        or page_cache.dtype != torch.uint8
        or not page_cache.is_contiguous()
        or page_table.ndim != 2
        or min(page_table.shape) == 0
        or page_table.dtype != torch.int32
        or req_ids.shape != (rows,)
        or req_ids.dtype != torch.int32
        or query_lens.shape != (rows,)
        or query_lens.dtype != torch.int32
        or live_lengths.shape != (page_table.shape[0],)
        or live_lengths.dtype not in _INT_DTYPES
        or any(
            t.device != rank_local_topk.device
            for t in (page_cache, page_table, req_ids, query_lens, live_lengths)
        )
    ):
        raise ValueError("Native staged pages and prefill top-k metadata do not align")

    physical_rows, _ = flat_kv_row_view(page_cache, MANAGER_BLOCK_TOKENS)
    kernel_cache = physical_rows.view(-1, MLA_KERNEL_BLOCK_ROWS, physical_rows.shape[1])
    out, counts = _result_buffers(
        rank_local_topk,
        physical_topk_out,
        valid_counts_out,
        (page_cache, page_table, req_ids, query_lens, live_lengths),
    )
    if not rows:
        return ResolvedHiSparseRows(kernel_cache, out, counts)

    # Native conversion assumes all query request IDs are in the staged table.
    max_req_id = page_table.shape[0] - 1
    torch._assert_async(
        ((req_ids >= 0) & (req_ids <= max_req_id)).all(),
        "Staged query request ID exceeds its native remapped block table",
    )
    torch._assert_async(
        ((page_table >= 0) & (page_table < page_cache.shape[0])).all(),
        "Staged physical page ID is outside the native backing",
    )
    live = live_lengths.index_select(0, req_ids.clamp(0, max_req_id))
    effective_lens = torch.minimum(query_lens, live).to(torch.int32)
    mapped = triton_convert_req_index_to_global_index(
        req_ids,
        page_table,
        rank_local_topk,
        BLOCK_SIZE=MANAGER_BLOCK_TOKENS,
        NUM_TOPK_TOKENS=width,
        BLOCK_N=gcd(128, width),
        return_valid_counts=False,
    )
    # Preserve native order and duplicates. The per-query bounds exclude future
    # rows before the valid count is reported, even inside a staged partial page.
    _compact_native_rows(
        mapped,
        out,
        counts,
        physical_rows.shape[0],
        staged_limits=(rank_local_topk, effective_lens),
    )
    return ResolvedHiSparseRows(kernel_cache, out, counts)
