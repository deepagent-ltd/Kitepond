# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from kitepond.integrations.vllm.checkpoint import (
    CheckpointReadReceipt,
    StockProducerCheckpoints,
)
from kitepond.models.glm5next.checkpoint import ConsumerCheckpointContract

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.output import KVConnectorBlockState


@pytest.fixture
def models():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=1024)
    blocks = tuple(pool.get_new_blocks(2))
    contract = ConsumerCheckpointContract("a" * 64, "b" * 64)
    scope = StockProducerCheckpoints(pool, contract, frozenset({0, 1}))
    offered = [(i, block.block_id, 1024) for i, block in enumerate(blocks)]
    state = KVConnectorBlockState({"mamba"}, lambda _: (), {"mamba": offered})
    return pool, blocks, scope, state


def _pin(scope, state):
    return scope.pin_offer(
        state, request_id="mamba", transfer_id="exact-kda", epoch=1, valid_tokens=2500
    )


def test_native_offered_blocks_stay_pinned_until_four_rank_reads_settle(models):
    pool, blocks, scope, state = models
    lease = _pin(scope, state)
    assert lease is not None
    assert _pin(scope, state) is lease
    assert lease.checkpoint.checkpoint_tokens == 1024
    assert [item.ref_cnt for item in blocks] == [2, 2]
    scope.start_reads(lease)
    assert not scope.cancel(lease)

    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 5
    receipt = CheckpointReadReceipt(lease.sha256, 0, "acked", 0)
    assert not scope.settle(lease, receipt)
    assert not scope.settle(lease, receipt)
    assert not scope.settle(lease, CheckpointReadReceipt(lease.sha256, 1, "acked", 0))
    assert not scope.settle(
        lease, CheckpointReadReceipt(lease.sha256, 2, "abort_settled", 0)
    )
    assert scope.settle(lease, CheckpointReadReceipt(lease.sha256, 3, "acked", 0))
    assert pool.get_num_free_blocks() == 7


def test_invalid_group_or_no_exact_native_offer_do_not_pin(models):
    pool, blocks, scope, state = models
    state.boundary_state_offloads["mamba"].pop()
    with pytest.raises(ValueError, match="incomplete cache groups"):
        _pin(scope, state)
    state.boundary_state_offloads["mamba"].clear()
    assert _pin(scope, state) is None
    assert [item.ref_cnt for item in blocks] == [1, 1]
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 7


def test_cancel_before_source_dispatch_releases_only_the_plugin_pin(models):
    pool, blocks, scope, state = models
    lease = _pin(scope, state)
    assert lease is not None
    assert scope.cancel(lease)
    assert [item.ref_cnt for item in blocks] == [1, 1]
    with pytest.raises(ValueError, match="Stale"):
        scope.start_reads(lease)
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 7


def test_old_receipt_cannot_settle_a_reissued_native_offer(models):
    pool, blocks, scope, state = models
    old = _pin(scope, state)
    assert old is not None and scope.cancel(old)
    lease = _pin(scope, state)
    assert lease is not None and lease.sha256 != old.sha256
    scope.start_reads(lease)
    with pytest.raises(ValueError, match="active lease"):
        scope.settle(lease, CheckpointReadReceipt(old.sha256, 0, "acked", 0))
    assert [item.ref_cnt for item in blocks] == [2, 2]
