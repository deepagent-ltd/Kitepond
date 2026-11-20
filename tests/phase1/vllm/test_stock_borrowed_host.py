# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
from kitepond.integrations.vllm.borrowed_host import BorrowedHostMetadata
from kitepond.integrations.vllm.host_authority import (
    NativeHostAuthority,
    NativeHostGrant,
)

from vllm.v1.core.block_pool import BlockPool


def test_p_has_only_lease_metadata_d_retains_physical_page_ownership():
    d = BlockPool(num_gpu_blocks=6, enable_caching=True, hash_block_size=1024)
    owner = NativeHostAuthority(d, engine_id="D", boot_id="boot", block_tokens=1024)
    p = BorrowedHostMetadata(
        engine_id="D", boot_id="boot", capacity=6, hash_block_size=1024
    )
    source = tuple(d.get_new_blocks(1))
    grant = owner.reserve(
        "request",
        "transfer",
        0,
        prefix_tokens=1024,
        reservation_tokens=2048,
        source=source,
    )
    assert isinstance(grant, NativeHostGrant)
    with pytest.raises(ValueError, match="foreign"):
        p.install(replace(grant, boot_id="stale"))
    readonly, writable = p.install(grant)
    assert readonly.block_id == source[0].block_id
    assert not p.is_block_writable(readonly)
    assert p.is_block_writable(writable)
    assert p.get_num_free_blocks() == 0
    with pytest.raises(RuntimeError, match="only install D-owned"):
        p.get_new_blocks(1)
    with pytest.raises(ValueError, match="still holds"):
        p.discard(grant)
    before = d.get_num_free_blocks()
    p.free_blocks((readonly, writable))
    p.discard(grant)
    assert d.get_num_free_blocks() == before
    assert owner.cancel(grant)
    d.free_blocks(source)
    assert d.get_num_free_blocks() == 5


def test_discard_cannot_release_the_last_reader_of_a_shared_native_page():
    d = BlockPool(num_gpu_blocks=6, enable_caching=True, hash_block_size=1024)
    owner = NativeHostAuthority(d, engine_id="D", boot_id="boot", block_tokens=1024)
    p = BorrowedHostMetadata(
        engine_id="D", boot_id="boot", capacity=6, hash_block_size=1024
    )
    source = tuple(d.get_new_blocks(1))
    first, second = (
        owner.reserve(
            request,
            request,
            0,
            prefix_tokens=1024,
            reservation_tokens=1024,
            source=source,
        )
        for request in ("first", "second")
    )
    assert isinstance(first, NativeHostGrant) and isinstance(second, NativeHostGrant)
    first_views = p.install(first)
    second_views = p.install(second)
    assert first_views[0] is second_views[0]
    with pytest.raises(ValueError, match="shared host page references"):
        p.discard(first)
    p.free_blocks(first_views)
    p.discard(first)
    assert owner.cancel(first)
    with pytest.raises(ValueError, match="shared host page references"):
        p.discard(second)
    p.free_blocks(second_views)
    p.discard(second)
    assert owner.cancel(second)
    d.free_blocks(source)
    assert d.get_num_free_blocks() == 5
