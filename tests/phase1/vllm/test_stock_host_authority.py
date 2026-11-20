# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor

import pytest
from kitepond.core.contracts import Wait
from kitepond.integrations.vllm.host_authority import (
    HostRankSettlement,
    NativeHostAuthority,
    NativeHostGrant,
)

from vllm.v1.core.block_pool import BlockPool


def _authority():
    pool = BlockPool(num_gpu_blocks=6, enable_caching=True, hash_block_size=1024)
    authority = NativeHostAuthority(pool, engine_id="D", boot_id="2", block_tokens=1024)
    return pool, authority


def test_reservation_shares_immutable_prefix_and_uses_native_pool_for_cow_tail():
    pool, authority = _authority()
    source = tuple(pool.get_new_blocks(2))
    grant = authority.reserve(
        "r", "t", 1, prefix_tokens=1300, reservation_tokens=3500, source=source
    )
    assert isinstance(grant, NativeHostGrant)
    assert pool.get_num_free_blocks() == 0
    assert grant.readonly_pages == (source[0].block_id,)
    assert grant.copy_tail == (source[1].block_id, grant.writable_pages[0])
    assert pool.is_block_writable(pool.blocks[grant.writable_pages[0]])
    authority.dispatch(grant)
    for rank in range(4):
        assert authority.settle(
            grant, HostRankSettlement(grant.sha256, rank, "ready", 0, 0)
        ) == (rank == 3)
    adopted = authority.adopt(grant)
    assert adopted[0] is source[0]
    assert source[1].ref_cnt == 1
    pool.free_blocks(source)
    assert pool.get_num_free_blocks() == 1
    pool.free_blocks(adopted)
    assert pool.get_num_free_blocks() == 5


def test_full_pool_queues_and_cancelled_refs_wait_for_actual_read_and_write_fences():
    pool, authority = _authority()
    grant = authority.reserve(
        "r", "t", 1, prefix_tokens=0, reservation_tokens=5 * 1024, source=()
    )
    assert isinstance(grant, NativeHostGrant)
    assert authority.reserve(
        "other", "q", 1, prefix_tokens=0, reservation_tokens=1, source=()
    ) == Wait("RAM_ADMISSION_WAIT")
    authority.dispatch(grant)
    assert not authority.cancel(grant)
    with pytest.raises(ValueError, match="ready quorum"):
        authority.adopt(grant)
    with pytest.raises(ValueError, match="settle both"):
        HostRankSettlement(grant.sha256, 0, "abort_settled", 1, 0)
    for rank in range(3):
        assert not authority.settle(
            grant, HostRankSettlement(grant.sha256, rank, "abort_settled", 0, 0)
        )
    assert pool.get_num_free_blocks() == 0
    assert authority.settle(
        grant, HostRankSettlement(grant.sha256, 3, "abort_settled", 0, 0)
    )
    assert pool.get_num_free_blocks() == 5


def test_foreign_thread_cannot_reserve_a_native_page():
    pool, authority = _authority()
    with ThreadPoolExecutor(max_workers=1) as executor:
        later = executor.submit(
            authority.reserve,
            "r",
            "t",
            1,
            prefix_tokens=0,
            reservation_tokens=1024,
            source=(),
        )
        with pytest.raises(RuntimeError, match="D scheduler thread"):
            later.result(timeout=5)
    assert pool.get_num_free_blocks() == 5


def test_failed_native_adoption_preserves_the_exact_authority_lease():
    pool, authority = _authority()
    grant = authority.reserve(
        "r", "t", 1, prefix_tokens=0, reservation_tokens=1024, source=()
    )
    assert isinstance(grant, NativeHostGrant)
    authority.dispatch(grant)
    for rank in range(4):
        authority.settle(grant, HostRankSettlement(grant.sha256, rank, "ready", 0, 0))

    def fail_attach(request_id, refs):
        assert request_id == "r" and refs[0].block_id == grant.writable_pages[0]
        raise RuntimeError("Native request was not attachable")

    with pytest.raises(RuntimeError, match="not attachable"):
        authority.adopt(grant, attach=fail_attach)
    assert pool.get_num_free_blocks() == 4
    refs = authority.adopt(grant)
    assert refs[0].block_id == grant.writable_pages[0]
    pool.free_blocks(refs)
    assert pool.get_num_free_blocks() == 5
