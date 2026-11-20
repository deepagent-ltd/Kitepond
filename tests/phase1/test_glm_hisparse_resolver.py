# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from kitepond.models.glm5next.hisparse_resolver import resolve_hisparse_rank_rows

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.hisparse import runtime
from vllm.v1.hisparse.runtime import (
    HiSparseCacheHandle,
    HiSparseRuntime,
    build_hisparse_prefill_staging_plan,
)


class CpuStream:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def wait_stream(self, stream):
        assert stream is self

    def wait_event(self, event):
        assert event.recorded


class CpuEvent:
    def record(self, stream):
        self.recorded = True


class CpuResolverRuntime(HiSparseRuntime):
    """Exercise stock orchestration with CPU references for its two CUDA calls."""

    def __init__(self, group, states, width, layer, raw):
        self.index_group = group
        self.request_state_indices = states
        self.is_group_leader = layer == 0
        self.row_width = width
        self.kv_dtype = torch.uint8
        self.eager_host_mirror = True
        self._swap_staged = False
        self._layer_ready_event = CpuEvent()
        self._swap_step = 0
        self.resolutions = []
        self.copied = []
        self.bind_hot_cache(
            raw,
            byte_offset=layer * 256 * width,
            block_stride=512 * width,
            num_blocks=8,
            block_size=256,
            block_table=torch.tensor([[6], [7]], dtype=torch.int32),
        )
        self.host_rows = torch.zeros(3 * 256, width, dtype=torch.uint8)
        self.host_rows[257, 0] = 81 + layer
        self.host_rows[513, 0] = 91 + layer
        self.hot.cache[3, 5, 0] = 31 + layer
        self.hot.cache[4, 7, 0] = 41 + layer

    def _resolve_residency(
        self,
        *,
        resident,
        req_id_per_token,
        block_table,
        topk_indices,
        block_size,
        return_valid_counts,
        shared_rows,
        attention_indices_out,
        valid_counts_out,
    ):
        assert resident.view is not None and resident.view.block_size == block_size
        assert not return_valid_counts
        assert attention_indices_out is None and valid_counts_out is None
        self.resolutions.append(
            (req_id_per_token.tolist(), block_table, topk_indices.clone())
        )
        result = self.index_group.shared_topk.physical_topk_indices[shared_rows]
        result.fill_(-1)
        moves = {}
        for row, request in enumerate(req_id_per_token.tolist()):
            if request < 0:
                continue
            state = self.request_state_indices[request].item()
            for i, logical in enumerate(topk_indices[row].tolist()):
                if logical < 0:
                    continue
                page, offset = divmod(logical, block_size)
                if page >= block_table.shape[1]:
                    continue
                device_block = resident.block_table[state, page].item()
                host_block = block_table[request, page].item()
                if device_block > 0:
                    physical = device_block * self.hot.attention_block_stride + offset
                elif host_block > 0:
                    hot_block = self.hot_block_table[state, 0].item()
                    slot = host_block * block_size + offset
                    hot_offset = moves.setdefault((hot_block, slot), len(moves))
                    physical = hot_block * self.hot.attention_block_stride + hot_offset
                else:
                    continue
                result[row, i] = physical
        self.index_group.moves = moves

    def _swap_rows(self, shared_rows):
        for (block, host_row), hot_row in self.index_group.moves.items():
            self.hot.cache[block, hot_row].copy_(self.host_rows[host_row])
            self.copied.append((block, host_row))


@pytest.fixture
def index_group(monkeypatch):
    stream = CpuStream()
    monkeypatch.setattr(runtime, "current_stream", lambda: stream)
    group = SimpleNamespace(
        copy_stream=stream,
        logical_topk_ready=None,
        shared_topk=SimpleNamespace(
            physical_topk_indices=torch.empty((8, 6), dtype=torch.int32)
        ),
        request_ids=torch.empty(8, dtype=torch.int32),
        row_indices=torch.arange(8, dtype=torch.int32),
        followers=[],
    )
    states = torch.tensor([1, 0], dtype=torch.int32)
    raw = torch.zeros(8 * 512 * 656, dtype=torch.int8)
    leader = CpuResolverRuntime(group, states, 656, 0, raw)
    follower = CpuResolverRuntime(group, states, 656, 1, raw)
    group.leader = leader
    group.followers.append(follower)
    caches = [HiSparseCacheHandle(leader), HiSparseCacheHandle(follower)]
    for layer, cache in enumerate(caches):
        cache.index_group_caches = caches
        cache.source_block_table = torch.tensor(
            [[1, 0, 2], [0, 1, 0]], dtype=torch.int32
        )
        resident_table = torch.tensor([[4, 0, 0], [3, 0, 0]], dtype=torch.int32)
        cache.bind_cache(
            raw,
            byte_offset=layer * 256 * 656,
            block_stride=512 * 656,
            num_blocks=8,
            block_size=256,
            block_table=resident_table,
            slot_mapping=torch.empty(8, dtype=torch.int64),
        )
    return HiSparseMLAIndexGroup(
        logical_topk_indices=torch.empty(8, 6, dtype=torch.int32),
        physical_topk_indices=torch.empty(8, 6, dtype=torch.int32),
        valid_topk_counts=torch.empty(8, dtype=torch.int32),
        side_stream=stream,
        logical_topk_ready=CpuEvent(),
        physical_topk_ready=CpuEvent(),
        has_indexer=True,
        num_layers=2,
        caches=caches,
    )


def metadata(*, mixed=False, padded=True):
    if mixed:
        req_ids, query_starts = [0, 1, 1, 1], [0, 1, 4]
    else:
        req_ids, query_starts = ([0, 0, 1, 0, 0] if padded else [0, 0, 1]), [0, 2, 3]
    return FlashInferMLASparseMetadata(
        num_reqs=2,
        max_query_len=3 if mixed else 2,
        max_seq_len=1024,
        num_actual_tokens=4 if mixed else 3,
        num_decode_tokens=1 if mixed else 3,
        num_decodes=1 if mixed else 2,
        num_prefills=1 if mixed else 0,
        query_start_loc=torch.tensor(query_starts, dtype=torch.int32),
        slot_mapping=torch.zeros(len(req_ids), dtype=torch.int64),
        req_id_per_token=torch.tensor(req_ids, dtype=torch.int32),
        block_table=torch.full((2, 3), 101, dtype=torch.int32),
        block_size=256,
        cp_kv_cache_interleave_size=4,
    )


def test_native_handle_swap_and_physical_view_per_layer(index_group):
    attn = metadata()
    index_group.prepare_for_batch(0, attn)
    local = torch.tensor(
        [
            [5, 260, 513, 5, -1, -1],
            [513, 257, 5, 999, -1, -1],
            [7, 257, 900, -1, -1, -1],
            [5, 1, 5, 513, 7, 5],
            [5, 1, 5, 513, 7, 5],
        ],
        dtype=torch.int32,
    )
    original = local.clone()
    resident_zero = 3 * 512 + 5
    resident_one = 4 * 512 + 7
    warm_zero = 7 * 512
    warm_one = 6 * 512 + 1
    expect = [
        [resident_zero, warm_zero, resident_zero, -1, -1, -1],
        [warm_zero, resident_zero, -1, -1, -1, -1],
        [resident_one, warm_one, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ]
    for layer in range(2):
        out = torch.empty_like(local)
        counts = torch.empty(5, dtype=torch.int32)
        got = resolve_hisparse_rank_rows(
            index_group,
            layer,
            local,
            attn,
            physical_topk_out=out,
            valid_counts_out=counts,
        )
        assert got.valid_counts is counts
        assert got.physical_topk is out
        assert got.physical_topk.tolist() == expect
        assert got.valid_counts.tolist() == [3, 2, 2, 0, 0]
        assert got.kv_cache.shape[1:] == (64, 656)
        assert got.kv_cache.view(-1, 656)[
            [resident_zero, warm_zero, resident_one, warm_one], 0
        ].tolist() == [
            31 + layer,
            91 + layer,
            41 + layer,
            81 + layer,
        ]
        runtime_layer = index_group.cache(layer).runtime
        assert got.kv_cache.data_ptr() == runtime_layer.hot.attention_cache.data_ptr()
        assert runtime_layer.copied == [(7, 513), (6, 257)]
    assert torch.equal(local, original)
    leader, follower = [cache.runtime for cache in index_group.caches]
    [(resolved_req_ids, source_table, consumed_local)] = leader.resolutions
    assert resolved_req_ids == [0, 0, 1, -1, -1]
    assert source_table is index_group.cache(0).source_block_table
    assert torch.equal(consumed_local, original)
    assert follower.resolutions == []


def test_resident_mixed_batch_only_resolves_decode_prefix(index_group):
    attn = metadata(mixed=True)
    index_group.prepare_for_batch(0, attn)
    local = torch.tensor([[5, -1, -1, -1, -1, -1]], dtype=torch.int32)
    got = resolve_hisparse_rank_rows(index_group, 0, local, attn)
    assert got.valid_counts.tolist() == [1]
    assert got.physical_topk[0, 0].item() == 3 * 512 + 5
    with pytest.raises(NotImplementedError, match="prefill"):
        resolve_hisparse_rank_rows(index_group, 0, local, attn, query_row_start=1)
    with pytest.raises(NotImplementedError, match="prefill"):
        resolve_hisparse_rank_rows(index_group, 0, local.repeat(2, 1), attn)
    assert len(index_group.cache(0).runtime.resolutions) == 1


def test_decode_slice_respects_dynamic_padding_boundary(index_group):
    attn = metadata()
    index_group.prepare_for_batch(0, attn)
    local = torch.tensor([[7, -1, -1, -1, -1, -1]], dtype=torch.int32).repeat(3, 1)
    got = resolve_hisparse_rank_rows(index_group, 0, local, attn, query_row_start=2)
    assert got.valid_counts.tolist() == [1, 0, 0]
    assert index_group.cache(0).runtime.resolutions[0][0] == [1, -1, -1]


def test_invalid_buffers_and_metadata_fail_before_native_resolution(index_group):
    attn = metadata()
    local = torch.full((3, 6), -1, dtype=torch.int32)
    with pytest.raises(ValueError, match="int32 matrix"):
        resolve_hisparse_rank_rows(index_group, 0, local.repeat(1, 2)[:, ::2], attn)
    with pytest.raises(ValueError, match="may not alias"):
        resolve_hisparse_rank_rows(index_group, 0, local, attn, physical_topk_out=local)
    with pytest.raises(NotImplementedError, match="capacity"):
        resolve_hisparse_rank_rows(index_group, 0, local.repeat(3, 1), attn)
    with pytest.raises(ValueError, match="source geometry"):
        resolve_hisparse_rank_rows(index_group, 0, local, attn, query_row_start=5)
    attn.cp_kv_cache_interleave_size = 1
    with pytest.raises(ValueError, match="source geometry"):
        resolve_hisparse_rank_rows(index_group, 0, local, attn)
    assert index_group.cache(0).runtime.resolutions == []


def test_native_hot_view_without_dense_physical_coordinates_is_rejected(index_group):
    cache = index_group.cache(0)
    runtime_layer = cache.runtime
    runtime_layer.bind_hot_cache(
        runtime_layer.hot_backing,
        byte_offset=0,
        block_stride=320 * 656,
        num_blocks=8,
        block_size=256,
        block_table=runtime_layer.hot_block_table,
    )
    assert not index_group.physical_kv_cache(0).is_contiguous()
    with pytest.raises(ValueError, match="native MLA physical cache"):
        resolve_hisparse_rank_rows(
            index_group, 0, torch.full((3, 6), -1, dtype=torch.int32), metadata()
        )
    assert runtime_layer.resolutions == []


def test_native_prefill_staging_requires_rank_local_sequence_boundaries():
    pages = torch.tensor([[2, 9, 17, 5, 3]], dtype=torch.int32)
    global_length = torch.tensor([1025], dtype=torch.int32)
    stock_plan = build_hisparse_prefill_staging_plan(pages, global_length, 256, 5)
    assert torch.count_nonzero(stock_plan.block_table).item() == 5
    for local_length, capacity in ((torch.tensor([257]), 2), (torch.tensor([256]), 1)):
        rank_plan = build_hisparse_prefill_staging_plan(
            pages, local_length, 256, capacity
        )
        assert torch.count_nonzero(rank_plan.block_table).item() == capacity
        assert torch.count_nonzero(rank_plan.block_table[:, capacity:]).item() == 0
