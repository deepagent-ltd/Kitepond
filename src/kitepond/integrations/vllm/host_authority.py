# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D-owned HiSparse reservations backed by the engine's real CPU BlockPool."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from kitepond.core.contracts import Wait

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock


@dataclass(frozen=True)
class NativeHostGrant:
    engine_id: str
    boot_id: str
    request_id: str
    transfer_id: str
    epoch: int
    prefix_tokens: int
    reservation_tokens: int
    readonly_pages: tuple[int, ...]
    writable_pages: tuple[int, ...]
    copy_tail: tuple[int, int] | None
    nonce: str = field(default_factory=lambda: secrets.token_hex(16))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class HostRankSettlement:
    lease_sha256: str
    rank: int
    result: Literal["ready", "abort_settled"]
    pending_reads: int
    pending_writes: int

    def __post_init__(self) -> None:
        if (
            type(self.rank) is not int
            or not 0 <= self.rank < 4
            or self.result not in {"ready", "abort_settled"}
            or type(self.pending_reads) is not int
            or self.pending_reads
            or type(self.pending_writes) is not int
            or self.pending_writes
        ):
            raise ValueError("The actual HiSparse rank must settle both I/O directions")


@dataclass(frozen=True)
class HostAuthorityView:
    block_tokens: int
    available_pages: int
    active_grants: int
    awaiting_settlement: int
    grant_block_references: int


@dataclass
class _Reservation:
    grant: NativeHostGrant
    source: tuple[KVCacheBlock, ...]
    issued: tuple[KVCacheBlock, ...]
    dispatched: bool = False
    cancelled: bool = False
    ranks: dict[int, HostRankSettlement] = field(default_factory=dict)


class NativeHostAuthority:
    """Only the D scheduler calls this controller, before or after engine steps."""

    def __init__(
        self, pool: BlockPool, *, engine_id: str, boot_id: str, block_tokens: int
    ) -> None:
        if (
            not engine_id
            or not boot_id
            or type(block_tokens) is not int
            or block_tokens < 1
        ):
            raise ValueError("Host authority needs the actual pool and boot identity")
        self.pool = pool
        self.engine_id = engine_id
        self.boot_id = boot_id
        self.block_tokens = block_tokens
        self._live: dict[str, _Reservation] = {}
        self._scheduler_thread = threading.get_ident()

    def _require_owner(self) -> None:
        if threading.get_ident() != self._scheduler_thread:
            raise RuntimeError("Native HiSparse refs belong to the D scheduler thread")

    def observe(self) -> HostAuthorityView:
        """Read the current native pool and controller's outstanding grants."""
        live = tuple(self._live.values())
        return HostAuthorityView(
            block_tokens=self.block_tokens,
            available_pages=self.pool.get_num_free_blocks(),
            active_grants=len(live),
            awaiting_settlement=sum(
                state.dispatched and len(state.ranks) < 4 for state in live
            ),
            grant_block_references=sum(
                len(state.source) + len(state.issued) for state in live
            ),
        )

    def reserve(
        self,
        request_id: str,
        transfer_id: str,
        epoch: int,
        *,
        prefix_tokens: int,
        reservation_tokens: int,
        source: tuple[KVCacheBlock, ...],
    ) -> NativeHostGrant | Wait:
        self._require_owner()
        # Use source blocks from the native durable-prefix resolver. No page IDs
        # from incoming request data are ever turned into source references.
        if (
            not request_id
            or request_id in self._live
            or not transfer_id
            or type(epoch) is not int
            or epoch < 0
            or type(prefix_tokens) is not int
            or type(reservation_tokens) is not int
            or not 0 <= prefix_tokens <= reservation_tokens
            or reservation_tokens <= 0
        ):
            raise ValueError("Invalid native host reservation identity or boundary")

        shared, partial = divmod(prefix_tokens, self.block_tokens)
        if len(source) != shared + bool(partial) or any(
            block.pool is not self.pool or block.is_null or block.ref_cnt <= 0
            for block in source
        ):
            raise ValueError(
                "Native prefix source is not the resolved live host page span"
            )

        required = (reservation_tokens + self.block_tokens - 1) // self.block_tokens
        needed = required - shared
        if self.pool.get_num_free_blocks() < needed:
            return Wait("RAM_ADMISSION_WAIT")

        issued = tuple(self.pool.get_new_blocks(needed))
        self.pool.touch(source)
        copy = (source[-1].block_id, issued[0].block_id) if partial else None
        grant = NativeHostGrant(
            self.engine_id,
            self.boot_id,
            request_id,
            transfer_id,
            epoch,
            prefix_tokens,
            reservation_tokens,
            tuple(block.block_id for block in source[:shared]),
            tuple(block.block_id for block in issued),
            copy,
        )
        self._live[request_id] = _Reservation(grant, source, issued)
        return grant

    def _require(self, grant: NativeHostGrant) -> _Reservation:
        self._require_owner()
        state = self._live.get(grant.request_id)
        if state is None or state.grant is not grant:
            raise ValueError("Host reservation is stale or foreign")
        return state

    def retains(self, grant: NativeHostGrant) -> bool:
        """Return whether this exact local grant is still held by the owner."""
        self._require_owner()
        state = self._live.get(grant.request_id)
        return state is not None and state.grant is grant

    def ready_for_adoption(self, grant: NativeHostGrant) -> bool:
        """Check the exact live grant and its uncancelled READY quorum."""
        self._require_owner()
        state = self._live.get(grant.request_id)
        return (
            state is not None
            and state.grant is grant
            and len(state.ranks) == 4
            and not state.cancelled
        )

    def dispatch(self, grant: NativeHostGrant) -> None:
        self._require(grant).dispatched = True

    def cancel(self, grant: NativeHostGrant) -> bool:
        state = self._require(grant)
        state.cancelled = True
        if state.dispatched and len(state.ranks) != 4:
            return False
        self._release(state)
        return True

    def settle(self, grant: NativeHostGrant, receipt: HostRankSettlement) -> bool:
        state = self._require(grant)
        if not state.dispatched or receipt.lease_sha256 != grant.sha256:
            raise ValueError("Host I/O receipt does not match the active grant")
        if state.ranks.setdefault(receipt.rank, receipt) != receipt:
            raise ValueError("Conflicting HiSparse rank settlement")
        if receipt.result == "abort_settled":
            state.cancelled = True
        done = len(state.ranks) == 4
        if done and state.cancelled:
            self._release(state)
        return done

    def adopt(
        self,
        grant: NativeHostGrant,
        *,
        attach: Callable[[str, tuple[KVCacheBlock, ...]], None] | None = None,
    ) -> tuple[KVCacheBlock, ...]:
        """Transfer native refs only after successful scheduler attachment."""
        state = self._require(grant)
        if len(state.ranks) != 4 or state.cancelled:
            raise ValueError("Host authority grant has no complete ready quorum")
        readonly = state.source[: len(grant.readonly_pages)]
        adopted = (*readonly, *state.issued)
        if attach is not None:
            attach(grant.request_id, adopted)
        if grant.copy_tail is not None:
            self.pool.free_blocks(state.source[-1:])
        self._live.pop(grant.request_id)
        return adopted

    def _release(self, state: _Reservation) -> None:
        self.pool.free_blocks((*state.source, *state.issued))
        self._live.pop(state.grant.request_id)
