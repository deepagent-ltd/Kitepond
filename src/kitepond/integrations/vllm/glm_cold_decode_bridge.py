# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold prefix admission while the native D engine remains idle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kitepond.core.contracts import Wait
from kitepond.models.glm5next.frontier import digest

from vllm.v1.request import Request, RequestStatus

from .glm_cold_catalog import VerifiedColdCheckpoint
from .host_authority import HostRankSettlement, NativeHostGrant
from .native_glm_cold_directory import NativeGlmColdDirectory

if TYPE_CHECKING:
    from .glm_cold_prefill_scheduler import GlmColdDecodeAdmissionMixin
    from .glm_cold_runtime import GlmColdRuntime


@dataclass(frozen=True, eq=False)
class ColdDecodePreadmission:
    request: Request
    verified: VerifiedColdCheckpoint
    grant: NativeHostGrant
    job_id: int


class GlmColdDecodeBridge:
    """Submit an idle FSL2 job, then stage the same host grant on native D."""

    def __init__(
        self,
        scheduler: GlmColdDecodeAdmissionMixin,
        runtime: GlmColdRuntime,
        *,
        authorize_prefix: Callable[[Request, VerifiedColdCheckpoint], str],
        register_grant: Callable[[NativeHostGrant, VerifiedColdCheckpoint], None],
        release_grant: Callable[[NativeHostGrant], None | Wait],
        confirm_rank_ownership: Callable[
            [NativeHostGrant], tuple[HostRankSettlement, ...]
        ],
        transfer_to_worker: Callable[[NativeHostGrant], None] | None = None,
    ) -> None:
        owner = getattr(scheduler, "_cold_decode_owner", None)
        if (
            owner is None
            or owner.stage != "decode"
            or owner.authority is not scheduler._cold_decode_authority
            or type(runtime.directory) is not NativeGlmColdDirectory
            or (transfer_to_worker is not None and not callable(transfer_to_worker))
            or (runtime.directory.engine_id, runtime.directory.boot_id)
            != (owner.authority.engine_id, owner.authority.boot_id)
            or any(
                not callable(op)
                for op in (
                    authorize_prefix,
                    register_grant,
                    release_grant,
                    confirm_rank_ownership,
                )
            )
        ):
            raise ValueError("Cold restore requires the real native D owner")
        self.scheduler, self.runtime, self.owner = scheduler, runtime, owner
        self.authorize_prefix = authorize_prefix
        self.register_grant, self.release_grant = register_grant, release_grant
        self.confirm_rank_ownership = confirm_rank_ownership
        self.transfer_to_worker = transfer_to_worker
        self._pending: dict[str, ColdDecodePreadmission] = {}

    def queue(
        self,
        request: Request,
        verified: VerifiedColdCheckpoint,
        *,
        transfer_id: str,
        epoch: int,
    ) -> ColdDecodePreadmission | Wait:
        self.owner._require_scheduler_owner()
        if type(request) is not Request or type(verified) is not VerifiedColdCheckpoint:
            raise ValueError(
                "Cold restore needs an authenticated native request and cache"
            )
        boundary = verified.checkpoint.layout.target_valid_tokens
        if (
            request.request_id in self.scheduler.requests
            or request.request_id in self._pending
            or request.request_id in self.scheduler._cold_decode_offered
            or request.status != RequestStatus.WAITING
            or request.num_computed_tokens
            or request.resumable
            or request.skip_reading_prefix_cache
            or type(request.prompt_token_ids) is not list
            or not 0 < boundary < request.num_prompt_tokens
        ):
            raise ValueError("Cold restore must precede native decode admission")
        if self.authorize_prefix(request, verified) != digest(
            request.prompt_token_ids[:boundary]
        ):
            raise ValueError("Cold restore prefix differs from its authorized tokens")
        checkpoint = self.runtime.directory.register_verified_cold(request, verified)
        if isinstance(checkpoint, Wait):
            return checkpoint
        if any(
            item.verified.checkpoint == checkpoint for item in self._pending.values()
        ):
            return Wait("GLM_COLD_GENERATION_IN_USE")
        grant = self.owner.authority.reserve(
            request.request_id,
            transfer_id,
            epoch,
            prefix_tokens=0,
            reservation_tokens=request.num_prompt_tokens,
            source=(),
        )
        if isinstance(grant, Wait):
            return grant
        try:
            self.register_grant(grant, verified)
        except BaseException:
            self.owner.authority.cancel(grant)
            raise
        try:
            job = self.runtime.scheduler.submit("RESTORE", checkpoint)
        except BaseException:
            self.release_grant(grant)
            self.owner.authority.cancel(grant)
            raise
        if isinstance(job, Wait):
            self.release_grant(grant)
            self.owner.authority.cancel(grant)
            return job
        state = ColdDecodePreadmission(request, verified, grant, job)
        self._pending[request.request_id] = state
        return state

    def stage(self, state: ColdDecodePreadmission) -> bool | Wait:
        """The final native ticket still gates executable decode after this call."""
        self.owner._require_scheduler_owner()
        request, grant = state.request, state.grant
        if self._pending.get(request.request_id) is not state:
            raise ValueError("Cold restore admission is stale or foreign")
        result = self.runtime.scheduler.view(state.job_id)
        if result.state in {"MISS", "CANCELLED", "FAILED"}:
            if isinstance(released := self.cancel(state), Wait):
                return released
            return Wait("GLM_COLD_READBACK_" + result.state)
        if result.state != "VERIFIED":
            return Wait(result.wait_reason or "GLM_COLD_READBACK_WAIT")
        if (
            request.request_id in self.scheduler.requests
            or request.request_id in self.scheduler._cold_decode_offered
            or request.status != RequestStatus.WAITING
            or len(request.prompt_token_ids) != grant.reservation_tokens
            or self.authorize_prefix(request, state.verified)
            != digest(
                request.prompt_token_ids[
                    : state.verified.checkpoint.layout.target_valid_tokens
                ]
            )
        ):
            raise ValueError("Cold restore request changed before native attachment")
        receipts = self.confirm_rank_ownership(grant)
        if (
            type(receipts) is not tuple
            or any(
                type(receipt) is not HostRankSettlement
                or receipt.lease_sha256 != grant.sha256
                or receipt.result != "ready"
                for receipt in receipts
            )
            or tuple(receipt.rank for receipt in receipts) != (0, 1, 2, 3)
        ):
            return Wait("GLM_COLD_RANK_READBACK_WAIT")
        self.owner.authority.dispatch(grant)
        for receipt in receipts:
            self.owner.authority.settle(grant, receipt)
        self.owner.stage_host_grant(
            request,
            grant,
            imported_tokens=state.verified.checkpoint.layout.target_valid_tokens,
            detached_cold=self.transfer_to_worker is not None,
        )
        self.scheduler.add_request(request)
        if self.transfer_to_worker is not None:
            self.transfer_to_worker(grant)
        self.runtime.scheduler.reap(state.job_id)
        del self._pending[request.request_id]
        return True

    def cancel(self, state: ColdDecodePreadmission) -> bool | Wait:
        self.owner._require_scheduler_owner()
        grant = state.grant
        if self._pending.get(grant.request_id) is not state:
            raise ValueError("Cold restore admission is stale or foreign")
        view = self.runtime.scheduler.cancel(state.job_id)
        if view.state == "PENDING":
            return Wait("GLM_COLD_READBACK_CANCEL_WAIT")
        if isinstance(released := self.release_grant(grant), Wait):
            return released
        if not self.owner.authority.cancel(grant):
            raise RuntimeError("Cold admission still holds unsettled native host I/O")
        self.runtime.scheduler.reap(state.job_id)
        del self._pending[grant.request_id]
        return True
