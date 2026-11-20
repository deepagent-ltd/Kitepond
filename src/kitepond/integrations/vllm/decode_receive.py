# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Carry native D write fences over the vLLM worker metadata channel."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from kitepond.core.readiness import _digest
from kitepond.models.glm5next.shared_mla import valid_rank_digests

from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnectorMetadata,
    HiSparseConnectorWorkerMetadata,
)

if TYPE_CHECKING:
    from .hisparse_attachment import DecodeImportBinding


@dataclass(frozen=True)
class DecodeImportRankReceipt:
    """A D worker's post-fence receipt for exactly one import destination."""

    grant_sha256: str
    destination_sha256: str
    rank: int
    outcome: Literal["loaded", "abort_settled"]
    pending_reads: int
    pending_writes: int

    def __post_init__(self) -> None:
        if (
            not _digest(self.grant_sha256)
            or not _digest(self.destination_sha256)
            or type(self.rank) is not int
            or not 0 <= self.rank < 4
            or self.outcome not in {"loaded", "abort_settled"}
            or type(self.pending_reads) is not int
            or type(self.pending_writes) is not int
            or self.pending_reads != 0
            or self.pending_writes != 0
        ):
            raise ValueError("D receipt requires a settled import on the actual rank")


def destination_digest(imported_tokens, block_ids, mla_rank_digests=None) -> str:
    fields = (imported_tokens, block_ids)
    if mla_rank_digests is not None:
        fields = (*fields, mla_rank_digests)
    wire = json.dumps(fields, separators=(",", ":"))
    return hashlib.sha256(wire.encode()).hexdigest()


@dataclass(frozen=True)
class DecodeReceiveTicket:
    """Request and native allocation sent from D scheduler to local ranks."""

    request_id: str
    transfer_id: str
    engine_id: str
    boot_id: str
    epoch: int
    grant_sha256: str
    imported_tokens: int
    block_ids: tuple[tuple[int, ...], ...]
    prefix_tokens: int
    copy_tail: tuple[int, int] | None
    detached_cold: bool = False
    mla_rank_digests: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not str or not value
                for value in (
                    self.request_id,
                    self.transfer_id,
                    self.engine_id,
                    self.boot_id,
                )
            )
            or type(self.detached_cold) is not bool
            or (
                self.mla_rank_digests is not None
                and (
                    not valid_rank_digests(self.mla_rank_digests)
                    or self.detached_cold
                    or self.prefix_tokens
                    or self.copy_tail is not None
                )
            )
            or (
                self.detached_cold
                and (self.prefix_tokens or self.copy_tail is not None)
            )
            or not _digest(self.grant_sha256)
            or type(self.epoch) is not int
            or not 0 <= self.epoch < 2**63
            or type(self.prefix_tokens) is not int
            or type(self.imported_tokens) is not int
            or not 0 <= self.prefix_tokens <= self.imported_tokens <= 1048576
            or not self.imported_tokens
            or type(self.block_ids) is not tuple
            or not self.block_ids
            or any(
                type(group) is not tuple
                or any(type(block) is not int or block < 0 for block in group)
                for group in self.block_ids
            )
            or (
                self.copy_tail is not None
                and (
                    type(self.copy_tail) is not tuple
                    or len(self.copy_tail) != 2
                    or self.copy_tail[0] == self.copy_tail[1]
                    or any(
                        type(block) is not int or block <= 0 for block in self.copy_tail
                    )
                )
            )
        ):
            raise ValueError("Invalid D native receive ticket")

    @property
    def destination_sha256(self) -> str:
        return destination_digest(
            self.imported_tokens, self.block_ids, self.mla_rank_digests
        )

    @classmethod
    def from_binding(cls, binding: DecodeImportBinding) -> DecodeReceiveTicket:
        grant = binding.grant
        return cls(
            binding.request.request_id,
            grant.transfer_id,
            grant.engine_id,
            grant.boot_id,
            grant.epoch,
            grant.sha256,
            binding.imported_tokens,
            binding.block_ids,
            grant.prefix_tokens,
            grant.copy_tail,
            binding.detached_cold,
            binding.mla_rank_digests,
        )


@dataclass
class DecodeReceiveWork(HiSparseConnectorMetadata):
    tickets: tuple[DecodeReceiveTicket, ...]


@dataclass
class DecodeReceiveProofs(HiSparseConnectorWorkerMetadata):
    receipts: tuple[DecodeImportRankReceipt, ...] = ()

    def aggregate(self, other):
        base = super().aggregate(other)
        incoming = other.receipts if isinstance(other, DecodeReceiveProofs) else ()
        return DecodeReceiveProofs(
            base.enqueued_transfer_counts,
            base.completed_transfer_counts,
            base.completed_host_copy_dst_ids,
            self.receipts + incoming,
        )


class DecodeReceiveMailbox:
    """Bind receipts to this worker rank and its currently assigned tickets."""

    def __init__(self, rank: int, boot_id: str) -> None:
        if type(rank) is not int or not 0 <= rank < 4 or not boot_id:
            raise ValueError("D proof mailbox requires its native rank and boot")
        self.rank = rank
        self._boot_id = boot_id
        self._tickets: dict[str, DecodeReceiveTicket] = {}
        self._proofs: dict[str, DecodeImportRankReceipt] = {}
        self._unreported: list[DecodeImportRankReceipt] = []
        self._importing: set[str] = set()
        self._lock = threading.Lock()

    def bind(self, tickets: tuple[DecodeReceiveTicket, ...]) -> None:
        active = {ticket.grant_sha256: ticket for ticket in tickets}
        if len(active) != len(tickets) or any(
            ticket.boot_id != self._boot_id for ticket in active.values()
        ):
            raise ValueError("D rank received mismatched scheduler tickets")
        with self._lock:
            if any(
                active[key] != old
                for key, old in self._tickets.items()
                if key in active
            ):
                raise ValueError("D rank's active destination changed")
            if self._importing - active.keys():
                raise ValueError("D rank cannot retire unsettled local writes")
            self._tickets = active
            self._proofs = {
                key: proof for key, proof in self._proofs.items() if key in active
            }

    def ticket_for(
        self, request_id: str, transfer_id: str, epoch: int
    ) -> DecodeReceiveTicket:
        with self._lock:
            tickets = [
                ticket
                for ticket in self._tickets.values()
                if (ticket.request_id, ticket.transfer_id, ticket.epoch)
                == (request_id, transfer_id, epoch)
            ]
            if len(tickets) != 1:
                raise ValueError("D rank has no exact live scheduler ticket")
            return tickets[0]

    def _start_import(self, ticket: DecodeReceiveTicket) -> None:
        key = ticket.grant_sha256
        with self._lock:
            if (
                self._tickets.get(key) != ticket
                or key in self._proofs
                or key in self._importing
            ):
                raise ValueError("D rank has no available exact scheduler ticket")
            self._importing.add(key)

    def settled_proof(
        self, ticket: DecodeReceiveTicket
    ) -> DecodeImportRankReceipt | None:
        with self._lock:
            if self._tickets.get(ticket.grant_sha256) != ticket:
                raise ValueError("D proof query has no exact live import ticket")
            return self._proofs.get(ticket.grant_sha256)

    def _post_fenced(
        self,
        ticket: DecodeReceiveTicket,
        outcome: Literal["loaded", "abort_settled"],
    ) -> None:
        receipt = DecodeImportRankReceipt(
            ticket.grant_sha256, ticket.destination_sha256, self.rank, outcome, 0, 0
        )
        with self._lock:
            if (
                self._tickets.get(ticket.grant_sha256) != ticket
                or ticket.grant_sha256 not in self._importing
            ):
                raise ValueError("D proof has no exact live native writer")
            self._proofs[ticket.grant_sha256] = receipt
            self._importing.remove(ticket.grant_sha256)
            self._unreported.append(receipt)

    def drain(
        self, base: HiSparseConnectorWorkerMetadata | None
    ) -> DecodeReceiveProofs:
        with self._lock:
            pending = tuple(self._unreported)
            self._unreported.clear()
        base = base or HiSparseConnectorWorkerMetadata({}, {})
        return DecodeReceiveProofs(
            base.enqueued_transfer_counts,
            base.completed_transfer_counts,
            base.completed_host_copy_dst_ids,
            pending,
        )
