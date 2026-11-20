# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use FSL2 on borrowed regions with signed prepare and drain receipts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hmac import compare_digest

from kitepond.backends.fsl2 import BundleIdentity, digest
from kitepond.core.contracts import Wait
from kitepond.models.glm5next.cold_authority import NativeRegionColdAuthority
from kitepond.models.glm5next.cold_lifecycle import GlmColdRamGrant

from .glm_cold_rank_control import RPC_NAME
from .glm_cold_remote_store import NativeGlmColdStore
from .glm_cold_restore_control import ColdRestoreControl, ColdRestoreRankReply
from .host_authority import HostRankSettlement


@dataclass
class _Attempt:
    ready: ColdRestoreControl
    opened: bool = False
    fsl2_verified: bool = False
    uncertain: bool = False
    receipts: tuple[HostRankSettlement, ...] = ()


class NativeGlmColdRestore(NativeGlmColdStore):
    """Retain native host pages on uncertainty until the full release quorum."""

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._sequence = 0
        self._restores: dict[BundleIdentity, _Attempt] = {}

    def register_grant(self, grant, verified) -> None:
        identity = verified.checkpoint.identity
        if identity in self._restores or grant.boot_id != self._boot:
            raise ValueError("Cold restore grant has a foreign or duplicate identity")
        self._sequence += 1
        self._restores[identity] = _Attempt(
            ColdRestoreControl("prepare", self._sequence, grant, verified)
        )

    def _lookup(self, grant) -> _Attempt:
        attempts = [
            item for item in self._restores.values() if item.ready.grant is grant
        ]
        if len(attempts) != 1:
            raise ValueError("Native cold restore has no exact host grant")
        return attempts[0]

    def _dispatch(self, command) -> tuple[ColdRestoreRankReply, ...] | Wait:
        message = command.signed_message()
        request_hash = digest(message)
        try:
            ranks = tuple(
                self._worker_control(
                    RPC_NAME,
                    timeout=5.0,
                    args=(
                        command,
                        tuple(signer._sign(message) for signer in self._signers),
                    ),
                )
            )
            if len(ranks) != 4:
                raise ValueError("Detached cold control lost a native rank")
            for rank, (reply, signer) in enumerate(
                zip(ranks, self._signers, strict=True)
            ):
                if isinstance(reply, Wait):
                    continue
                if (
                    type(reply) is not ColdRestoreRankReply
                    or reply.sequence != command.sequence
                    or reply.rank != rank
                    or reply.command_sha256 != request_hash
                    or reply.region_sha256 != self.region_fingerprints[rank]
                    or type(reply.signature) is not str
                    or not compare_digest(
                        reply.signature, signer._sign(reply.signed_message())
                    )
                ):
                    raise ValueError(
                        "Detached restore reply failed rank authentication"
                    )
        except Exception:
            return Wait("NATIVE_COLD_RANK_CONTROL_UNCERTAIN")
        return next((reply for reply in ranks if isinstance(reply, Wait)), ranks)

    def _close(self, attempt: _Attempt) -> Wait | None:
        if not attempt.opened:
            return None
        replies = self._dispatch(replace(attempt.ready, verb="release"))
        if isinstance(replies, Wait) or any(
            reply.state != "released" for reply in replies
        ):
            attempt.uncertain = True
            return Wait("NATIVE_COLD_RANK_RELEASE_UNPROVEN")
        attempt.opened = False
        attempt.uncertain = False
        attempt.receipts = ()
        return None

    def release_grant(self, grant) -> Wait | None:
        attempt = self._lookup(grant)
        held = self._close(attempt)
        if held is None:
            del self._restores[attempt.ready.verified.checkpoint.identity]
        return held

    def detach_to_worker(self, grant) -> None:
        attempt = self._lookup(grant)
        if not attempt.fsl2_verified or attempt.uncertain or len(attempt.receipts) != 4:
            raise ValueError("Native D cannot adopt an unverified detached restore")
        del self._restores[attempt.ready.verified.checkpoint.identity]

    def confirm_rank_ownership(self, grant) -> tuple[HostRankSettlement, ...]:
        attempt = self._lookup(grant)
        if attempt.fsl2_verified and not attempt.receipts:
            self._verify(attempt)
        return attempt.receipts

    def _verify(self, attempt) -> None:
        replies = self._dispatch(replace(attempt.ready, verb="verified"))
        if isinstance(replies, Wait) or any(
            reply.state != "ready"
            or type(reply.host_ready) is not HostRankSettlement
            or reply.host_ready.rank != reply.rank
            or reply.host_ready.lease_sha256 != attempt.ready.grant.sha256
            or reply.host_ready.result != "ready"
            for reply in replies
        ):
            attempt.uncertain = True
            return
        attempt.uncertain = False
        attempt.receipts = tuple(reply.host_ready for reply in replies)

    def reserve_restore(self, checkpoint, *, max_inflight_bytes, verified_catalog=None):
        attempt = self._restores.get(checkpoint.identity)
        if attempt is None or verified_catalog is None:
            return Wait("NATIVE_COLD_DETACHED_ADMISSION_REQUIRED")
        command = attempt.ready
        if (
            command.verified != verified_catalog
            or command.verified.checkpoint != checkpoint
        ):
            raise ValueError("Native cold restore changed its authorized catalog")
        sizes = tuple(
            max(part.nbytes for part in plan.parts) for plan in checkpoint.plans
        )
        if sum(sizes) + min(max(sizes), 65536) > max_inflight_bytes:
            return Wait("NATIVE_COLD_STAGING_BUDGET_WAIT")
        if attempt.opened:
            if (blocked := self._close(attempt)) is not None:
                return blocked
            self._sequence += 1
            command = attempt.ready = replace(command, sequence=self._sequence)
        attempt.opened = True
        replies = self._dispatch(command)
        if isinstance(replies, Wait) or any(
            reply.state != "prepared" for reply in replies
        ):
            released = self._close(attempt)
            if released is not None:
                return released
            self._sequence += 1
            attempt.ready = replace(command, sequence=self._sequence)
            return (
                replies
                if isinstance(replies, Wait)
                else Wait("NATIVE_COLD_RANK_PREPARE_WAIT")
            )

        def live():
            if not attempt.opened:
                raise ValueError("Detached FSL2 lost its native rank leases")

        try:
            authorities = []
            for rank, reply in enumerate(replies):
                if reply.scratch is None or reply.scratch.nbytes != sizes[rank]:
                    raise ValueError("Detached FSL2 stage changed its native budget")
                authorities.append(
                    NativeRegionColdAuthority(
                        region=self.regions.regions[rank],
                        identity=checkpoint.identity,
                        plan=checkpoint.plans[rank],
                        bindings=reply.bindings,
                        stage=reply.scratch,
                        verify_reference=live,
                        mode="RESTORE",
                        checksums=verified_catalog.checksums(rank),
                    )
                )
        except Exception:
            if (blocked := self._close(attempt)) is not None:
                return blocked
            raise

        def settle(state):
            for authority in authorities:
                authority.require_settled_restore(verified=state == "VERIFIED")
            if state == "VERIFIED":
                attempt.fsl2_verified = True
                self._verify(attempt)
            else:
                self._close(attempt)

        return GlmColdRamGrant(
            checkpoint, tuple(authorities), self.regions.layout_sha256, settle
        )
