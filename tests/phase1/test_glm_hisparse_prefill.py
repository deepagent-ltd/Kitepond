# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
import torch
from kitepond.models.glm5next.hisparse_prefill import (
    StagedPrefillRows,
    plan_rank_local_prefill,
    resolve_staged_rank_rows,
    stage_rank_local_prefill_rows,
)

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla import sparse_utils
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.hisparse.runtime import HiSparseCacheHandle, HiSparseRuntime


def metadata(lengths, q_starts, dcp_lengths, source, *, decodes, bounds=None):
    qsl = torch.tensor(q_starts, dtype=torch.int32)
    lens = torch.tensor(lengths, dtype=torch.int32)
    count = len(lengths)
    tokens = int(qsl[-1])
    table = torch.tensor(source, dtype=torch.int32)
    slots = torch.full((tokens,), -1, dtype=torch.int64)
    common = CommonAttentionMetadata(
        num_reqs=count,
        max_query_len=int(qsl.diff().max()),
        max_seq_len=max(lengths),
        num_actual_tokens=tokens,
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=lens,
        seq_lens_cpu_upper_bound=lens,
        dcp_local_seq_lens=torch.tensor(dcp_lengths, dtype=torch.int32),
        dcp_local_seq_lens_cpu_upper_bound=(
            torch.tensor(bounds, dtype=torch.int32) if bounds is not None else None
        ),
        block_table_tensor=table,
        slot_mapping=slots,
    )
    req_ids = torch.repeat_interleave(
        torch.arange(count, dtype=torch.int32), qsl.diff()
    )
    attn = FlashInferMLASparseMetadata(
        num_reqs=count,
        max_query_len=common.max_query_len,
        max_seq_len=common.max_seq_len,
        num_actual_tokens=tokens,
        num_decode_tokens=int(qsl[decodes]),
        num_decodes=decodes,
        num_prefills=count - decodes,
        query_start_loc=qsl,
        slot_mapping=slots,
        req_id_per_token=req_ids,
        block_table=table,
        block_size=256,
        cp_kv_cache_interleave_size=4,
    )
    return common, attn


@pytest.fixture
def native_layer(monkeypatch):
    copies = []

    def gather_dma(source, destination, src_ids, dst_ids, required, *_):
        """CPU stand-in only for the stock C++ CUDA host gather."""
        mask = required.bool().flatten()
        host_rows = src_ids.flatten()[mask].to(torch.int64)
        device_rows = dst_ids.flatten()[mask].to(torch.int64)
        copies.append((host_rows.tolist(), device_rows.tolist()))
        destination.view(-1, destination.shape[-1])[device_rows] = source[host_rows]

    monkeypatch.setattr(
        torch.ops._C_cache_ops, "hisparse_gather_plan", gather_dma, raising=False
    )

    def bind(attn, resident_table, slots):
        width = 656
        source = torch.zeros((5, 256, width), dtype=torch.uint8)
        backing = torch.zeros(5 * 256 * width, dtype=torch.int8)
        runtime = object.__new__(HiSparseRuntime)
        runtime._host_cache = source.view(-1, width)
        runtime.row_width = width
        runtime.kv_dtype = torch.uint8
        runtime.request_state_indices = torch.arange(attn.num_reqs, dtype=torch.int32)
        cache = HiSparseCacheHandle(runtime)
        cache.bind_cache(
            backing,
            byte_offset=0,
            block_stride=256 * width,
            num_blocks=5,
            block_size=256,
            block_table=torch.tensor(resident_table, dtype=torch.int32),
            slot_mapping=torch.tensor(slots, dtype=torch.int64),
        )
        group = HiSparseMLAIndexGroup(
            logical_topk_indices=torch.empty(0, dtype=torch.int32),
            physical_topk_indices=torch.empty(0, dtype=torch.int32),
            valid_topk_counts=torch.empty(0, dtype=torch.int32),
            side_stream=None,
            logical_topk_ready=None,
            physical_topk_ready=None,
            has_indexer=True,
            num_layers=1,
            caches=[cache],
        )
        return group, cache.view.cache, source, copies

    return bind


def test_full_prefill_uses_pending_resident_writes_before_current_mirror(native_layer):
    common, attn = metadata([17], [0, 17], [5], [[1]], decodes=0)
    plan = plan_rank_local_prefill(common, num_decodes=0, num_prefills=1, dcp_rank=0)
    assert attn.prefill is None
    writer_slots = [3 * 256 + i for i in range(4)] + [-1] * 12 + [3 * 256 + 4]
    group, resident, host, copies = native_layer(attn, [[3]], writer_slots)
    host[1, :5, 0] = 91
    expected = torch.arange(211, 216, dtype=torch.uint8)
    resident[3, :5, 0] = expected
    staged = stage_rank_local_prefill_rows(group, 0, host, attn, plan)
    assert staged.local_seq_lens.tolist() == [5]
    assert staged.request_ids.tolist() == [0] * 17
    stage_block = staged.block_table[0, 0].item()
    assert torch.equal(staged.kv_cache[stage_block, :5, 0], expected)
    assert host[1, :5, 0].tolist() == [91] * 5
    copied_sources, copied_destinations = copies[0]
    assert all(row not in copied_sources for row in range(256, 261))
    assert all(
        row not in copied_destinations
        for row in range(stage_block * 256, stage_block * 256 + 5)
    )
    assert plan.native.gpu_row_ids is None


def test_mixed_suffix_reads_fenced_history_and_new_rank_writer(native_layer):
    common, attn = metadata(
        [10, 1025],
        [0, 1, 3],
        [4, 257],
        [[3, 0, 0], [1, 2, 0]],
        decodes=1,
        bounds=[4, 257],
    )
    attn.req_id_per_token = torch.cat(
        (attn.req_id_per_token, torch.tensor([0], dtype=torch.int32))
    )
    plan = plan_rank_local_prefill(common, num_decodes=1, num_prefills=1, dcp_rank=0)
    group, resident, host, copies = native_layer(
        attn, [[3, 0, 0], [0, 4, 0]], [-1, -1, 4 * 256]
    )
    host[1, 0, 0] = 67
    host[2, 0, 0] = 99
    resident[4, 0, 0] = 215
    staged = stage_rank_local_prefill_rows(group, 0, host, attn, plan)
    assert staged.request_ids.tolist() == [0, 0]
    assert staged.local_seq_lens.tolist() == [257]
    prefix, current, unused = staged.block_table[0].tolist()
    assert prefix > 0 and current > 0 and unused == 0
    assert staged.kv_cache[prefix, 0, 0].item() == 67
    assert staged.kv_cache[current, 0, 0].item() == 215
    assert 256 in copies[0][0] and 512 not in copies[0][0]
    assert host[2, 0, 0].item() == 99


def test_rank_without_current_writer_still_reads_its_prefix(native_layer):
    common, attn = metadata(
        [5],
        [0, 1],
        [4],
        [[1]],
        decodes=0,
    )
    plan = plan_rank_local_prefill(common, num_decodes=0, num_prefills=1, dcp_rank=0)
    group, _, host, _ = native_layer(attn, [[0]], [-1])
    host[1, :4, 0] = 71
    staged = stage_rank_local_prefill_rows(group, 0, host, attn, plan)
    assert staged.kv_cache[staged.block_table[0, 0], :4, 0].tolist() == [71] * 4


def test_bad_rank_lengths_missing_history_or_bound_fails():
    common, _ = metadata([1025], [0, 1], [256], [[1, 2]], decodes=0)
    with pytest.raises(RuntimeError, match="does not belong"):
        plan_rank_local_prefill(common, num_decodes=0, num_prefills=1, dcp_rank=0)
    common.dcp_local_seq_lens = torch.tensor([257], dtype=torch.int32)
    common.dcp_local_seq_lens_cpu_upper_bound = torch.tensor([256], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="does not belong"):
        plan_rank_local_prefill(common, num_decodes=0, num_prefills=1, dcp_rank=0)
    common.dcp_local_seq_lens_cpu_upper_bound = torch.tensor([257], dtype=torch.int32)
    common.block_table_tensor[0, 1] = 0
    with pytest.raises(RuntimeError, match="history page has no native source"):
        plan_rank_local_prefill(common, num_decodes=0, num_prefills=1, dcp_rank=0)


def test_stale_writer_binding_and_foreign_host_are_rejected(native_layer):
    common, attn = metadata([1], [0, 1], [1], [[1]], decodes=0)
    plan = plan_rank_local_prefill(common, num_decodes=0, num_prefills=1, dcp_rank=0)
    group, _, host, copies = native_layer(attn, [[3]], [2 * 256])
    with pytest.raises(ValueError, match="source backing"):
        stage_rank_local_prefill_rows(group, 0, host.clone(), attn, plan)
    with pytest.raises(RuntimeError, match="pending writer slots"):
        stage_rank_local_prefill_rows(group, 0, host, attn, plan)
    group.cache(0).runtime.request_state_indices.fill_(-1)
    with pytest.raises(RuntimeError, match="valid resident states"):
        stage_rank_local_prefill_rows(group, 0, host, attn, plan)
    assert copies == []


@pytest.mark.parametrize("alias", ["retained prefix", "second writer"])
def test_writable_alias_of_shared_staging_page_is_rejected(native_layer, alias):
    prefix = alias == "retained prefix"
    common, attn = metadata(
        [1, 5 if prefix else 1],
        [0, 1, 2],
        [1, 4 if prefix else 1],
        [[1], [1]],
        decodes=0,
    )
    plan = plan_rank_local_prefill(common, num_decodes=0, num_prefills=2, dcp_rank=0)
    group, _, host, copies = native_layer(
        attn, [[3], [3]], [3 * 256, -1 if prefix else 3 * 256]
    )
    with pytest.raises(
        RuntimeError, match="aliases a retained prefix|Multiple native P writers"
    ):
        stage_rank_local_prefill_rows(group, 0, host, attn, plan)
    assert copies == []


@pytest.fixture
def native_sparse_conversion(monkeypatch):
    launches = []

    def launch(req, table, topk, out, counts, *_args, **config):
        # CPU stand-in only for the kernel-launch boundary.
        assert req.is_contiguous() and table.is_contiguous() and topk.is_contiguous()
        assert config["BLOCK_SIZE"] == config["BLOCK_STRIDE_ROWS"] == 256
        assert config["DCP_SIZE"] == config["DCP_INTERLEAVE"] == 1
        assert not config["COUNT_VALID"] and counts is None
        assert not config["HAS_PREFILL_WORKSPACE"]
        launches.append((req.tolist(), table.tolist(), topk.clone()))
        columns = topk // 256
        valid = (topk >= 0) & (columns < table.shape[1])
        physical_blocks = table[req[:, None], columns.clamp(0, table.shape[1] - 1)]
        out.copy_(torch.where(valid, physical_blocks * 256 + topk % 256, -1))

    monkeypatch.setattr(
        sparse_utils, "_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX_KERNEL", launch
    )
    return launches


def test_full_prefill_resolves_only_rows_visible_at_each_query(
    native_layer, native_sparse_conversion
):
    common, attn = metadata([17], [0, 17], [5], [[1]], decodes=0)
    plan = plan_rank_local_prefill(common, num_decodes=0, num_prefills=1, dcp_rank=0)
    slots = [3 * 256 + i for i in range(4)] + [-1] * 12 + [3 * 256 + 4]
    group, resident, host, _ = native_layer(attn, [[3]], slots)
    resident[3, :5, 0] = torch.arange(211, 216, dtype=torch.uint8)
    staged = stage_rank_local_prefill_rows(group, 0, host, attn, plan)
    assert staged.local_query_lens.tolist() == [1, 2, 3] + [4] * 13 + [5]
    rank_topk = torch.tensor([[4, 2, 0, 5, -1, 0, 3, 256]], dtype=torch.int32)
    rank_topk = rank_topk.repeat(17, 1)
    out = torch.empty_like(rank_topk)
    counts = torch.empty(17, dtype=torch.int32)
    resolved = resolve_staged_rank_rows(
        staged, rank_topk, physical_topk_out=out, valid_counts_out=counts
    )
    assert resolved.physical_topk is out and resolved.valid_counts is counts
    assert out[0].tolist() == [256, 256, -1, -1, -1, -1, -1, -1]
    assert out[-1].tolist() == [260, 258, 256, 256, 259, -1, -1, -1]
    assert counts.tolist() == [2, 2, 3] + [4] * 13 + [5]
    assert resolved.kv_cache.data_ptr() == staged.kv_cache.data_ptr()
    assert resolved.kv_cache.shape == (8, 64, 656)
    assert resolved.kv_cache.view(-1, 656)[out[-1, :5], 0].tolist() == [
        215,
        213,
        211,
        211,
        214,
    ]
    assert torch.equal(native_sparse_conversion[0][2], rank_topk)


def test_mixed_suffix_resolves_new_rank_row_at_exact_query_frontier(
    native_layer, native_sparse_conversion
):
    common, attn = metadata(
        [10, 1025],
        [0, 1, 3],
        [4, 257],
        [[3, 0, 0], [1, 2, 0]],
        decodes=1,
        bounds=[4, 257],
    )
    plan = plan_rank_local_prefill(common, num_decodes=1, num_prefills=1, dcp_rank=0)
    group, resident, host, _ = native_layer(
        attn, [[3, 0, 0], [0, 4, 0]], [-1, -1, 4 * 256]
    )
    host[1, 255, 0] = 81
    host[1, 0, 0] = 70
    resident[4, 0, 0] = 231
    staged = stage_rank_local_prefill_rows(group, 0, host, attn, plan)
    assert staged.local_query_lens.tolist() == [256, 257]
    topk = torch.tensor([[256, 255, 257, 0, 256, -1, 512, 768]], dtype=torch.int32)
    got = resolve_staged_rank_rows(staged, topk.repeat(2, 1))
    assert got.valid_counts.tolist() == [2, 4]
    assert got.physical_topk.tolist() == [
        [511, 256, -1, -1, -1, -1, -1, -1],
        [512, 511, 256, 512, -1, -1, -1, -1],
    ]
    by_row = got.kv_cache.view(-1, 656)
    assert by_row[got.physical_topk[0, :2], 0].tolist() == [81, 70]
    assert by_row[got.physical_topk[1, :4], 0].tolist() == [231, 81, 70, 231]
    assert native_sparse_conversion[0][0] == [0, 0]


def test_native_remapped_nulls_physical_bounds_and_request_selection(
    native_sparse_conversion,
):
    page_cache = torch.zeros((4, 256, 656), dtype=torch.uint8)
    stage = StagedPrefillRows(
        page_cache,
        torch.tensor([[2, 0, 1], [3, 2, 0]], dtype=torch.int32),
        torch.tensor([1, 0, 1], dtype=torch.int32),
        torch.tensor([768, 600], dtype=torch.int32),
        torch.tensor([600, 700, 0], dtype=torch.int32),
    )
    topk = torch.tensor(
        [[257, 0, 512, -1, 513, 257, 767, 768]], dtype=torch.int32
    ).repeat(3, 1)
    got = resolve_staged_rank_rows(stage, topk)
    assert got.physical_topk.tolist() == [
        [513, 768, 513, -1, -1, -1, -1, -1],
        [512, 256, 257, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
    ]
    assert got.valid_counts.tolist() == [3, 3, 0]
    assert native_sparse_conversion[0][0] == [1, 0, 1]
    assert got.kv_cache.data_ptr() == page_cache.data_ptr()


def test_staged_resolution_rejects_wrong_metadata_or_buffer_alias(
    native_sparse_conversion,
):
    topk = torch.zeros((2, 8), dtype=torch.int32)
    stage = StagedPrefillRows(
        torch.zeros((2, 256, 656), dtype=torch.uint8),
        torch.tensor([[1]], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([1, 2], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="may not alias"):
        resolve_staged_rank_rows(stage, topk, physical_topk_out=topk)
    with pytest.raises(ValueError, match="do not align"):
        resolve_staged_rank_rows(stage, topk[:1])
    with pytest.raises(RuntimeError, match="request ID exceeds"):
        resolve_staged_rank_rows(
            replace(stage, request_ids=torch.tensor([0, -1], dtype=torch.int32)),
            topk,
        )
    with pytest.raises(RuntimeError, match="outside the native backing"):
        resolve_staged_rank_rows(
            replace(stage, block_table=torch.tensor([[9]], dtype=torch.int32)), topk
        )
    assert native_sparse_conversion == []
