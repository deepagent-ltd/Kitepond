# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admit a verified D terminal prefix and route real P worker read fences."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from threading import get_ident

from kitepond.models.glm5next.frontier import digest as digest_tokens

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorWorkerMetadata,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from .glm_terminal_manifest import GlmTerminalManifest
from .glm_terminal_return import GlmTerminalReturn, PrefillTerminalReservation
from .hisparse_attachment import HiSparseConnector
from .host_authority import NativeHostGrant
from .prefill_return_receive import PrefillReturnFence, PrefillReturnProofs
from .terminal_checkpoint import TerminalPinLease

VerifyReturnOffer = Callable[
    [TerminalPinLease, GlmTerminalManifest, NativeHostGrant], None
]


@dataclass
class _WarmReturn:
    request: Request
    lease: TerminalPinLease
    manifest: GlmTerminalManifest
    grant: NativeHostGrant
    transfer: GlmTerminalReturn | None = None
    fences: dict[int, PrefillReturnFence] = field(default_factory=dict)
    routed: bool = False


class GlmPrefillReturnScheduler(Scheduler):
    """Keep a warm P request waiting until its four native ranks verify readback."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        children = (
            self.connector.sub_connectors
            if isinstance(self.connector, MultiConnector)
            else [self.connector]
        )
        owners = [child for child in children if isinstance(child, HiSparseConnector)]
        if (
            len(owners) != 1
            or children[0] is not owners[0]
            or owners[0].stage != "prefill"
            or owners[0].prefill_return_boot is None
            or self.scheduler_config.async_scheduling is not False
            or self.recompute_kv_load_failures
        ):
            raise ValueError("Native P terminal scheduler needs its synchronous P role")
        self._prefill_return_owner = owners[0]
        self._prefill_return_thread = get_ident()
        self._prefill_return_verify: VerifyReturnOffer | None = None
        self._warm_returns: dict[str, _WarmReturn] = {}

    def _require_return_owner(self) -> None:
        if self._prefill_return_thread != get_ident():
            raise RuntimeError(
                "P terminal return belongs to the native scheduler owner"
            )

    def bind_terminal_return_offer_verifier(self, verify: VerifyReturnOffer) -> None:
        self._require_return_owner()
        if self._prefill_return_verify is not None or not callable(verify):
            raise ValueError(
                "Native P terminal offer verifier is invalid or already bound"
            )
        self._prefill_return_verify = verify

    def stage_terminal_return(
        self,
        request: Request,
        lease: TerminalPinLease,
        manifest: GlmTerminalManifest,
        grant: NativeHostGrant,
        *,
        before_admission: bool = False,
    ) -> None:
        """Bind an authenticated D offer to this current waiting native P request."""
        self._require_return_owner()
        verify = self._prefill_return_verify
        req_id = request.request_id
        if (
            verify is None
            or (
                self.requests.get(req_id) is not (None if before_admission else request)
            )
            or request.status is not RequestStatus.WAITING
            or type(lease) is not TerminalPinLease
            or type(manifest) is not GlmTerminalManifest
            or type(grant) is not NativeHostGrant
            or req_id in self._warm_returns
        ):
            raise ValueError("Native P terminal return needs a verified waiting offer")
        boundary = lease.requested_common_tokens
        if (
            not 0 < boundary < request.num_prompt_tokens
            or manifest.lease_sha256 != lease.sha256
            or grant.request_id != req_id
            or (grant.engine_id, grant.boot_id, grant.epoch)
            != (
                lease.identity.source_engine_id,
                lease.identity.source_boot_id,
                lease.identity.epoch,
            )
            or digest_tokens(list(request.all_token_ids[:boundary]))
            != lease.prefix_sha256
            or request.all_token_ids[boundary] != lease.next_token_id
        ):
            raise ValueError("Native P terminal offer changed the D frontier or grant")
        verify(lease, manifest, grant)
        self._prefill_return_owner.stage_prefill_terminal_return(
            request, grant, imported_tokens=boundary
        )
        self._warm_returns[req_id] = _WarmReturn(request, lease, manifest, grant)

    def terminal_return_destination(
        self, request: Request
    ) -> PrefillTerminalReservation:
        self._require_return_owner()
        warm = self._warm_returns.get(request.request_id)
        binding = self._prefill_return_owner.pending_prefill_terminal_return(request)
        if (
            warm is None
            or warm.request is not request
            or binding is None
            or binding.grant is not warm.grant
            or binding.imported_tokens != warm.lease.requested_common_tokens
        ):
            raise ValueError(
                "Native P terminal return does not own an allocated destination"
            )
        boot = self._prefill_return_owner.prefill_return_boot
        assert boot is not None
        return PrefillTerminalReservation.capture(
            self, request, warm.lease, warm.grant, boot_id=boot
        )

    def bind_terminal_return(self, transfer: GlmTerminalReturn) -> None:
        """Accept the D-permitted transfer on the same verified control channel."""
        self._require_return_owner()
        if type(transfer) is not GlmTerminalReturn:
            raise TypeError("Native P terminal return requires a permitted transfer")
        req_id = transfer.destination.request_id
        warm = self._warm_returns.get(req_id)
        if warm is not None and warm.transfer == transfer:
            return
        if (
            warm is None
            or warm.transfer is not None
            or transfer.lease != warm.lease
            or transfer.manifest != warm.manifest
            or transfer.permit.lease_sha256 != warm.lease.sha256
            or transfer.permit.manifest_sha256 != warm.manifest.sha256
            or transfer.destination != self.terminal_return_destination(warm.request)
        ):
            raise ValueError("Native P terminal transfer changed the admitted offer")
        warm.transfer = transfer

    def _record_return_fences(self, output: KVConnectorOutput) -> None:
        metadata = output.kv_connector_worker_meta
        if isinstance(self.connector, MultiConnector):
            if not isinstance(metadata, MultiKVConnectorWorkerMetadata):
                return
            metadata = metadata.metadata[0]
        if not isinstance(metadata, PrefillReturnProofs):
            return
        for fence in metadata.fences:
            if type(fence) is not PrefillReturnFence:
                raise ValueError("Native P terminal worker produced an invalid fence")
            warm = self._warm_returns.get(fence.attempt.request_id)
            if warm is None or warm.transfer is None or warm.routed:
                continue
            if fence.receipt.lease_sha256 != warm.transfer.permit.sha256:
                continue
            rank = fence.receipt.local_rank
            if (
                fence.attempt != warm.transfer.attempt(rank)
                or warm.fences.setdefault(rank, fence) != fence
            ):
                raise ValueError("Native P terminal fence changed its admitted rank")

    def update_from_output(self, scheduler_output, model_runner_output):
        if not self._warm_returns:
            return super().update_from_output(scheduler_output, model_runner_output)
        owner = self._prefill_return_owner
        original = model_runner_output.kv_connector_output
        selected = original or KVConnectorOutput()
        self._record_return_fences(selected)
        protected = {
            req_id for req_id, warm in self._warm_returns.items() if not warm.routed
        }
        accepted = {}
        for req_id in protected:
            warm = self._warm_returns[req_id]
            binding = owner.pending_prefill_terminal_return(warm.request)
            if binding is None:
                continue
            if selected.invalid_block_ids and any(
                block in selected.invalid_block_ids
                for group in binding.block_ids
                for block in group
            ):
                raise ValueError(
                    "Native P terminal import requires request-scoped failure"
                )
            if warm.fences.keys() == {0, 1, 2, 3}:
                if self.requests.get(req_id) is not warm.request:
                    raise ValueError(
                        "Native P terminal completion lost its live request"
                    )
                accepted[req_id] = binding

        if original is not None or accepted:
            # A cancelled request still drains the complete incoming transfer.
            # Native HiSparse can then release its host mapping without replaying
            # the last page into resident memory for an already finished request.
            discarded = {
                req_id
                for req_id, binding in accepted.items()
                if RequestStatus.is_finished(binding.request.status)
                or any(
                    fence.receipt.status == "abort_settled"
                    for fence in self._warm_returns[req_id].fences.values()
                )
            }
            routed = replace(
                selected,
                finished_recving=(
                    (set(selected.finished_recving or ()) - self._warm_returns.keys())
                    | accepted.keys()
                )
                or None,
                finished_sending=(set(selected.finished_sending or ()) - protected)
                or None,
                failed_recving=(selected.failed_recving - self._warm_returns.keys())
                | discarded,
            )
            model_runner_output = replace(
                model_runner_output, kv_connector_output=routed
            )
        with owner.routed_prefill_return_completions(accepted):
            result = super().update_from_output(scheduler_output, model_runner_output)
        for req_id in accepted:
            warm = self._warm_returns.get(req_id)
            if warm is not None:
                warm.routed = True
        return result

    def _free_blocks(self, request: Request) -> None:
        super()._free_blocks(request)
        warm = self._warm_returns.get(request.request_id)
        if warm is not None and warm.request is request:
            self._warm_returns.pop(request.request_id)
