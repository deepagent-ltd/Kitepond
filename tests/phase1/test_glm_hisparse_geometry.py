# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from mmap import PAGESIZE

import pytest
import torch
from kitepond.models.glm5next.hisparse_geometry import (
    GlmHiSparseGeometry,
    HostShardPlan,
    RankPageAddress,
    SparseAccessDelta,
)
from kitepond.models.glm5next.layout import CacheRole, dcp_row_mapping

from vllm.v1.kv_cache_interface import (
    KVCacheTensor,
    MLAAttentionSpec,
    SparseCacheRole,
    create_kv_cache_views,
)
from vllm.v1.kv_cache_layout import KVCacheLayout


@pytest.fixture
def specs():
    mla = MLAAttentionSpec(
        block_size=256,
        dtype=torch.uint8,
        num_kv_heads=1,
        head_size=576,
        state_content_bytes=656,
        cache_dtype_str="fp8_ds_mla",
    )
    indexer = MLAAttentionSpec(
        block_size=256,
        dtype=torch.uint8,
        num_kv_heads=1,
        head_size=132,
        tokens_per_state=4,
        cache_role=SparseCacheRole.INDEXER,
    )
    return {"mla.a": mla, "mla.b": mla, "indexer": indexer}


@pytest.fixture
def geometry(specs):
    return GlmHiSparseGeometry.from_native_specs(specs, dcp_size=4, interleave_size=4)


def test_native_group_page_geometry_and_host_budget(geometry, specs):
    assert geometry.logical_page_tokens == 1024
    bytes_per_rank_page = 256 * 656 * 2
    assert geometry.mla.page_bytes == bytes_per_rank_page
    assert geometry.mla.page_rows == 256
    assert geometry.mla.kernel_page_rows == 64
    assert geometry.indexer.page_bytes == 64 * 132
    assert geometry.indexer.page_rows == geometry.indexer.kernel_page_rows == 64
    assert geometry.indexer.page_bytes == specs["indexer"].page_size_bytes
    shards = geometry.plan_host_shards(4 * 5 * bytes_per_rank_page + 9, 3072)
    assert shards == HostShardPlan(
        5 * bytes_per_rank_page + 2,
        5,
        5 * bytes_per_rank_page,
        20 * bytes_per_rank_page,
        3072,
    )
    assert shards.shard_bytes % PAGESIZE == 0
    with pytest.raises(ValueError, match="one maximum context"):
        geometry.plan_host_shards(4 * 5 * bytes_per_rank_page + 9, 3073)


def test_compatible_528_is_explicit_per_geometry(specs):
    specs.update(
        {
            name: replace(spec, state_content_bytes=528)
            for name, spec in specs.items()
            if name.startswith("mla.")
        }
    )
    compact = GlmHiSparseGeometry.from_native_specs(
        specs, dcp_size=4, interleave_size=4
    )
    assert compact.mla.page_bytes == 2 * 256 * 528
    assert (
        compact.access_delta(
            "swap", host_recalls=5, mla_layers=("mla.a",)
        ).host_read_bytes
        == 5 * 528
    )


@pytest.mark.parametrize(
    ("name", "overrides"),
    [
        ("indexer", {"block_size": 3072}),
        ("indexer", {"tokens_per_state": 1}),
        ("indexer", {"dcp_sharded": False}),
        ("mla.a", {"state_content_bytes": 528}),
        ("mla.a", {"num_head_slots": 2}),
        ("mla.a", {"page_size_padded": 256 * 656 + 64}),
    ],
)
def test_reject_geometry_mismatches(specs, name, overrides):
    specs[name] = replace(specs[name], **overrides)
    with pytest.raises(ValueError, match="row width|geometry"):
        GlmHiSparseGeometry.from_native_specs(specs, dcp_size=4, interleave_size=4)


@pytest.mark.parametrize("role", [CacheRole.MLA, CacheRole.INDEXER])
def test_all_rank_owners_across_scheduler_page_boundaries(geometry, role):
    token_step = 1 if role is CacheRole.MLA else 4
    first = token_step - 1
    global_tokens = range(first, 2048 + 16, token_step)
    blocks = [3, 17, 2]
    owners, local_rows = dcp_row_mapping(
        torch.arange(len(global_tokens), dtype=torch.int64), cache_role=role
    )
    plane = geometry.mla if role is CacheRole.MLA else geometry.indexer
    for i, token in enumerate(global_tokens):
        for rank in range(4):
            got = geometry.address(role, token, blocks, rank=rank)
            if rank != owners[i].item():
                assert got is None
                continue
            manager_page = token // geometry.logical_page_tokens
            row = local_rows[i].item() % plane.page_rows
            flat = blocks[manager_page] * plane.page_rows + row
            assert got == RankPageAddress(
                manager_page,
                blocks[manager_page],
                row,
                flat // plane.kernel_page_rows,
                flat % plane.kernel_page_rows,
            )
    if role is CacheRole.INDEXER:
        for token in (0, 1, 2, 1024, 1025, 1026):
            assert all(
                geometry.address(role, token, blocks, rank=r) is None for r in range(4)
            )


def test_native_physical_views_alias_expected_rows(geometry, specs):
    mla, index = specs["mla.a"], specs["indexer"]
    for plane, spec, role in (
        (geometry.mla, mla, CacheRole.MLA),
        (geometry.indexer, index, CacheRole.INDEXER),
    ):
        page_bytes = spec.page_size_bytes
        storage = torch.zeros(3 * page_bytes, dtype=torch.int8)
        placement = KVCacheTensor(
            storage.numel(),
            ["probe"],
            storage.numel(),
            page_bytes,
        )
        (native_pages,) = create_kv_cache_views(
            storage, spec, 3, KVCacheLayout.LBNHC, placement
        )
        (native_kernel_pages,) = create_kv_cache_views(
            storage,
            spec,
            3,
            KVCacheLayout.LBNHC,
            placement,
            kernel_block_size=64 if role is CacheRole.MLA else 256,
        )
        address = geometry.address(role, 1027, [1, 2], rank=0)
        assert address is not None
        native_pages[address.physical_block, 0, address.physical_row, 0] = 91
        assert native_kernel_pages[address.kernel_block, 0, address.kernel_row, 0] == 91
        assert native_kernel_pages.shape == (
            3 * plane.page_rows // plane.kernel_page_rows,
            1,
            plane.kernel_page_rows,
            plane.row_bytes,
        )
        assert native_kernel_pages.untyped_storage().data_ptr() == storage.data_ptr()


def test_block_outermost_mla_uses_the_actual_row_stride(geometry, specs):
    from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

    spec = specs["mla.a"]
    storage = torch.zeros(3 * geometry.mla.page_bytes, dtype=torch.int8)
    physical = KVCacheTensor(
        storage.numel(),
        ["mla.a"],
        spec.page_size_bytes,
        geometry.mla.page_bytes,
    )
    (device_pages,) = create_kv_cache_views(
        storage,
        spec,
        3,
        KVCacheLayout.BLHNC,
        physical,
    )
    flat_rows, stride = flat_kv_row_view(device_pages.squeeze(1), spec.block_size)
    assert stride == 512
    address = geometry.address(
        CacheRole.MLA,
        1027,
        [1, 2],
        rank=0,
        mla_block_stride_rows=stride,
    )
    assert address == RankPageAddress(1, 2, 3, 16, 3)
    device_pages[2, 0, 3, 0] = 89
    assert (
        flat_rows.view(-1, 64, 656)[address.kernel_block, address.kernel_row, 0] == 89
    )


def test_invalid_addresses_and_branch_deltas(geometry):
    with pytest.raises(ValueError, match="no live native block"):
        geometry.address(CacheRole.MLA, 1024, [9, 0], rank=0)
    with pytest.raises(ValueError, match="no native block"):
        geometry.address(CacheRole.MLA, 1024, [9], rank=0)
    with pytest.raises(ValueError, match="stride"):
        geometry.address(CacheRole.MLA, 0, [9], rank=0, mla_block_stride_rows=257)
    assert geometry.access_delta("direct", direct_rows=11) == SparseAccessDelta(
        direct_rows=11
    )
    assert geometry.access_delta(
        "swap",
        hot_hits=2,
        host_recalls=3,
        mla_layers=("mla.a", "mla.b"),
    ) == SparseAccessDelta(hot_hits=2, host_recalls=3, host_read_bytes=3 * 2 * 656)
    assert geometry.access_delta(
        "swap",
        hot_hits=3,
        mla_layers=("mla.a",),
    ) == SparseAccessDelta(hot_hits=3)
    for kwargs in (
        {"branch": "direct", "host_recalls": 3},
        {"branch": "swap", "direct_rows": 11, "mla_layers": ("mla.a",)},
        {"branch": "swap", "host_recalls": 1, "mla_layers": ("indexer",)},
        {"branch": "swap", "host_recalls": 1, "mla_layers": ("mla.a", "mla.a")},
    ):
        with pytest.raises(ValueError, match="observed branch"):
            geometry.access_delta(**kwargs)
