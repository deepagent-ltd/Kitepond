# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retain native Mamba boundary pages offered to the stock connector."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from kitepond.models.glm5next.checkpoint import (
    ConsumerCheckpointContract,
    ProducerCheckpoint,
)

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.sched.output import KVConnectorBlockState


@dataclass(frozen=True)
class CheckpointReadReceipt:
    lease_sha256: str
    local_rank: int
    status: Literal["acked", "abort_settled"]
    pending_reads: int

    def __post_init__(self) -> None:
        if (
            type(self.local_rank) is not int
            or not 0 <= self.local_rank < 4
            or self.status not in {"acked", "abort_settled"}
            or type(self.pending_reads) is not int
            or self.pending_reads != 0
        ):
            raise ValueError("KDA read still owns its native source")


@dataclass(frozen=True)
class CheckpointLease:
    checkpoint: ProducerCheckpoint
    nonce: str = field(default_factory=lambda: secrets.token_hex(16))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass
class _Pinned:
    lease: CheckpointLease
    blocks: tuple[KVCacheBlock, ...]
    reads: dict[int, CheckpointReadReceipt] = field(default_factory=dict)
    started: bool = False


class StockProducerCheckpoints:
    """Scheduler-thread access to existing engine pages, with no storage pool."""

    def __init__(
        self,
        pool: BlockPool,
        contract: ConsumerCheckpointContract,
        mamba_groups: frozenset[int],
    ) -> None:
        if not mamba_groups or any(type(i) is not int or i < 0 for i in mamba_groups):
            raise ValueError("Native Mamba cache group identities required")
        self.pool = pool
        self.contract = contract
        self.mamba_groups = mamba_groups
        self._pinned: dict[str, _Pinned] = {}

    def pin_offer(
        self,
        state: KVConnectorBlockState,
        *,
        request_id: str,
        transfer_id: str,
        epoch: int,
        valid_tokens: int,
    ) -> CheckpointLease | None:
        """Pin only exact source addresses offered during the scheduler step."""
        boundary = self.contract.checkpoint(valid_tokens)
        if not boundary or request_id not in state.req_ids:
            return None
        exact = tuple(
            sorted(
                (group, block)
                for group, block, tokens in state.boundary_state_offloads.get(
                    request_id, ()
                )
                if tokens == boundary
            )
        )
        if not exact:
            return None
        if {group for group, _ in exact} != self.mamba_groups:
            raise ValueError("Native KDA checkpoint has incomplete cache groups")
        checkpoint = ProducerCheckpoint(
            request_id,
            transfer_id,
            epoch,
            valid_tokens,
            boundary,
            self.contract.fingerprint,
            exact,
            "producer_H_COW_destination",
        )
        checkpoint.validate(self.contract)
        prior = self._pinned.get(request_id)
        if prior is not None:
            if prior.lease.checkpoint != checkpoint:
                raise ValueError("Native KDA checkpoint offer changed while pinned")
            return prior.lease

        indices = [block_id for _, block_id in exact]
        if len(set(indices)) != len(indices) or max(indices) >= len(self.pool.blocks):
            raise ValueError("Native KDA checkpoint addresses are invalid")
        blocks = tuple(self.pool.blocks[index] for index in indices)
        if any(b.is_null or b.ref_cnt <= 0 for b in blocks):
            raise ValueError("Native KDA checkpoint source is no longer live")
        lease = CheckpointLease(checkpoint)
        self.pool.touch(blocks)
        self._pinned[request_id] = _Pinned(lease, blocks)
        return lease

    def _check(self, lease: CheckpointLease) -> _Pinned:
        state = self._pinned.get(lease.checkpoint.request_id)
        if state is None or state.lease is not lease:
            raise ValueError("Stale native KDA checkpoint lease")
        return state

    def start_reads(self, lease: CheckpointLease) -> None:
        self._check(lease).started = True

    def cancel(self, lease: CheckpointLease) -> bool:
        state = self._check(lease)
        if state.started:
            return False
        self._release(state)
        return True

    def settle(self, lease: CheckpointLease, receipt: CheckpointReadReceipt) -> bool:
        state = self._check(lease)
        if not state.started or receipt.lease_sha256 != lease.sha256:
            raise ValueError("Native KDA receipt does not match the active lease")
        if state.reads.setdefault(receipt.local_rank, receipt) != receipt:
            raise ValueError("Conflicting native KDA rank receipts")
        if len(state.reads) != 4:
            return False
        self._release(state)
        return True

    def _release(self, state: _Pinned) -> None:
        self.pool.free_blocks(state.blocks)
        self._pinned.pop(state.lease.checkpoint.request_id)
