# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Signed rank-local detached NVMe restore leases on the native D worker."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hmac import compare_digest
from threading import RLock
from typing import TYPE_CHECKING, Literal

from kitepond.backends.fsl2 import digest
from kitepond.core.contracts import Wait
from kitepond.models.glm5next.import_recipe import ByteSpan, ColdBinding

from .decode_receive import DecodeReceiveMailbox, DecodeReceiveTicket
from .glm_cold_catalog import VerifiedColdCheckpoint
from .glm_cold_detached_rank import DetachedGlmColdRank
from .glm_terminal_manifest import TerminalRankSigner, _is_digest
from .host_authority import HostRankSettlement, NativeHostGrant
from .native_glm_cold_provider import NativeGlmRankMemory

if TYPE_CHECKING:
    import torch

    from .glm_native_views import GlmNativeStateViews


@dataclass(frozen=True)
class ColdRestoreControl:
    verb: Literal["prepare", "verified", "release"]
    sequence: int
    grant: NativeHostGrant
    verified: VerifiedColdCheckpoint

    def signed_message(self) -> dict:
        return {
            "schema": "kitepond.glm-cold-restore-control.v1",
            "verb": self.verb,
            "sequence": self.sequence,
            "grant": self.grant.sha256,
            "identity": asdict(self.verified.checkpoint.identity),
            "catalog_mac": self.verified.mac,
            "aggregate": self.verified.aggregate_sha256,
            "hashes": digest(self.verified.hashes_by_rank),
            "plans": digest(
                tuple(plan.document() for plan in self.verified.checkpoint.plans)
            ),
            "source_boot": self.verified.source_boot_id,
            "target_boot": self.verified.target_boot_id,
        }


@dataclass(frozen=True)
class ColdRestoreRankReply:
    sequence: int
    rank: int
    region_sha256: str
    command_sha256: str
    state: Literal["prepared", "ready", "released", "held"]
    bindings: tuple[ColdBinding, ...] = ()
    scratch: ByteSpan | None = None
    host_ready: HostRankSettlement | None = None
    signature: str = ""

    def signed_message(self) -> dict:
        return {
            "schema": "kitepond.glm-cold-restore-reply.v1",
            "sequence": self.sequence,
            "rank": self.rank,
            "region": self.region_sha256,
            "command": self.command_sha256,
            "state": self.state,
            "ranges": digest(tuple(asdict(binding) for binding in self.bindings)),
            "scratch": None if self.scratch is None else asdict(self.scratch),
            "host_ready": None if self.host_ready is None else asdict(self.host_ready),
        }


@dataclass
class _Attempt:
    command: ColdRestoreControl
    restore: DetachedGlmColdRank
    state: str = "prepared"


class GlmWorkerColdRestore:
    """Keep local byte leases through store closure and native import fencing."""

    def __init__(
        self,
        memory: NativeGlmRankMemory,
        views: GlmNativeStateViews,
        signer: TerminalRankSigner,
    ) -> None:
        if (
            type(signer) is not TerminalRankSigner
            or signer.rank != memory.rank
            or memory.rank != views.rank
            or memory.region.descriptor != views.descriptor
        ):
            raise ValueError("Cold restore worker does not own its enrolled rank")
        self.memory, self.views, self.signer = memory, views, signer
        self._lock = RLock()
        self._latest_sequence = 0
        self._pending: dict[str, _Attempt] = {}

    def _require_enrolled(self) -> None:
        if (
            self.memory.region.mapping.closed
            or self.memory.region.descriptor != self.views.descriptor
            or self.memory.region_sha256 != self.views.descriptor.fingerprint()
        ):
            raise ValueError("Cold restore rank lost its original mapped pool")

    def _reply(self, command, state, *, bindings=(), scratch=None, host_ready=None):
        reply = ColdRestoreRankReply(
            command.sequence,
            self.memory.rank,
            self.memory.region_sha256,
            digest(command.signed_message()),
            state,
            bindings,
            scratch,
            host_ready,
        )
        return replace(reply, signature=self.signer._sign(reply.signed_message()))

    def handle(self, command, signatures):
        authority = self.memory.region.descriptor
        if (
            type(command) is not ColdRestoreControl
            or type(command.grant) is not NativeHostGrant
            or type(command.verified) is not VerifiedColdCheckpoint
            or command.verb not in {"prepare", "verified", "release"}
            or type(command.sequence) is not int
            or not 0 < command.sequence < 2**63
            or (command.grant.engine_id, command.grant.boot_id)
            != (authority.authority_engine_id, authority.authority_boot_id)
            or command.verified.target_boot_id != authority.authority_boot_id
            or type(signatures) is not tuple
            or len(signatures) != 4
            or not _is_digest(signatures[self.signer.rank])
            or not compare_digest(
                signatures[self.signer.rank],
                self.signer._sign(command.signed_message()),
            )
        ):
            raise ValueError("Detached restore request has no native D signature")
        with self._lock:
            self._require_enrolled()
            handle = command.grant.sha256
            attempt = self._pending.get(handle)
            if (
                attempt is not None
                and replace(command, verb="prepare") != attempt.command
            ):
                raise ValueError("Detached restore changed the active rank claims")
            if command.verb == "release":
                return self._release(command, attempt)
            if command.verb == "prepare":
                return self._prepare(command, attempt)
            if attempt is None or attempt.state not in ("prepared", "ready"):
                return self._reply(command, "held")
            ready = (
                attempt.restore.verify_shared_readback()
                if attempt.state == "prepared"
                else attempt.restore.host_ready()
            )
            attempt.state = "ready"
            return self._reply(command, "ready", host_ready=ready)

    def _prepare(self, command, attempt):
        if attempt is None:
            if command.sequence <= self._latest_sequence:
                return self._reply(command, "released")
            self._latest_sequence = command.sequence
            restore = DetachedGlmColdRank(
                self.memory,
                self.views,
                command.grant,
                command.verified,
                require_native=self._require_enrolled,
            )
            authority = restore.prepare()
            if isinstance(authority, Wait):
                return authority
            attempt = _Attempt(command, restore)
            self._pending[command.grant.sha256] = attempt
        if attempt.state != "prepared":
            return self._reply(command, "held")
        restore = attempt.restore
        return self._reply(
            command,
            "prepared",
            bindings=restore._data.bindings,
            scratch=restore._stage.span,
        )

    def _release(self, command, attempt):
        if attempt is not None:
            if attempt.state == "prepared":
                attempt.restore.settle_cold("MISS")
            elif attempt.state == "ready":
                attempt.restore.abandon_before_import()
            elif attempt.state != "fenced":
                return self._reply(command, "held")
            self._pending.pop(command.grant.sha256)
        self._latest_sequence = max(self._latest_sequence, command.sequence)
        return self._reply(command, "released")

    def has_claim(self, ticket: DecodeReceiveTicket) -> bool:
        with self._lock:
            return ticket.grant_sha256 in self._pending

    def import_ticket(
        self,
        mailbox: DecodeReceiveMailbox,
        ticket: DecodeReceiveTicket,
        slab: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None,
        writes_before_import: torch.cuda.Event | None,
    ) -> None:
        with self._lock:
            attempt = self._pending.get(ticket.grant_sha256)
            if attempt is None or attempt.state != "ready":
                raise ValueError("Native D ticket has no verified detached readback")
            attempt.state = "importing"
        try:
            attempt.restore.import_ready(
                mailbox,
                ticket,
                slab,
                stream=stream,
                writes_before_import=writes_before_import,
            )
        except BaseException:
            with self._lock:
                attempt.state = "uncertain"
            proof = mailbox.settled_proof(ticket)
            if proof is not None and proof.outcome == "abort_settled":
                attempt.restore.release_after_abort(mailbox, ticket)
                with self._lock:
                    attempt.state = "fenced"
            raise
        else:
            with self._lock:
                attempt.state = "fenced"

    def fence_settled(self, ticket: DecodeReceiveTicket) -> bool:
        with self._lock:
            attempt = self._pending.get(ticket.grant_sha256)
            return attempt is not None and attempt.state == "fenced"

    def retire_fenced(self, ticket: DecodeReceiveTicket) -> None:
        with self._lock:
            attempt = self._pending.get(ticket.grant_sha256)
            if attempt is not None and attempt.state == "fenced":
                self._pending.pop(ticket.grant_sha256)

    def abort_unstarted(
        self, mailbox: DecodeReceiveMailbox, ticket: DecodeReceiveTicket
    ) -> None:
        with self._lock:
            attempt = self._pending.get(ticket.grant_sha256)
            if attempt is None or attempt.state != "ready":
                raise ValueError(
                    "Detached restore already started or lost its rank lease"
                )
            mailbox._start_import(ticket)
            mailbox._post_fenced(ticket, "abort_settled")
            attempt.restore.abandon_before_import()
            attempt.state = "fenced"
