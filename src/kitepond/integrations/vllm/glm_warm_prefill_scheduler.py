# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compose native warm import with the same P cold/K0 admission and save path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from secrets import token_hex
from time import monotonic_ns
from typing import TYPE_CHECKING

from vllm.v1.request import RequestStatus

from .glm_cold_prefill_admission import ColdPrefillOffer, ColdPrefillPrompt
from .glm_cold_prefill_scheduler import (
    _OFFER_PARAM,
    GlmColdPrefillMixin,
    _native_prompt,
    _Prefill,
)
from .glm_prefill_bootstrap import stage_native_glm_k0_request
from .glm_prefill_return import GlmPrefillReturnScheduler
from .glm_terminal_return import GlmTerminalReturn
from .glm_warm_control import NAMESPACE, GlmWarmControl
from .glm_warm_decode_scheduler import GlmPublishedTerminalSource
from .prefill_return_receive import PrefillReturnFence

if TYPE_CHECKING:
    from vllm.v1.request import Request

_P_CONTROL = (
    "kitepond_prefill_submit_terminal_return",
    "kitepond_prefill_terminal_return_result",
    "kitepond_prefill_retire_terminal_return",
)
_RANKS = frozenset(range(4))
_RPC_TIMEOUT_SECONDS = 5.0


WARM_ADMISSION_PARAM = "kitepond_glm_warm_nonce"


@dataclass(frozen=True)
class _AuthorizedWarm:
    source: GlmPublishedTerminalSource
    offer: ColdPrefillOffer
    envelope: dict
    nonce: str


class GlmWarmPrefillScheduler(GlmColdPrefillMixin, GlmPrefillReturnScheduler):
    """Use one native P scheduler for ordinary cold and authenticated warm requests."""

    _warm_worker_rpc = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._cold_prefill_config is None:
            raise ValueError("Native warm P requires its existing K0/READY admission")
        self._warm_submissions: dict[str, GlmTerminalReturn] = {}
        self._authorized_warm: dict[str, _AuthorizedWarm] = {}
        self._warm_control = GlmWarmControl(self._cold_prefill_config.control)
        self._cold_prefill_owner.bind_control(NAMESPACE, self._warm_p_handler)

    def get_owner_worker_control_registration(self):
        parent = super().get_owner_worker_control_registration()

        def bind(rpc):
            if parent is not None:
                parent[1](rpc)
            self._warm_worker_rpc = rpc

        return ((*(() if parent is None else parent[0]), *_P_CONTROL), bind)

    def _verify_authorized_warm(self, lease, manifest, grant) -> None:
        authorized = self._authorized_warm.get(grant.request_id)
        if (
            authorized is None
            or (lease, manifest)
            != (authorized.source.lease, authorized.source.manifest)
            or grant != authorized.offer.grant
        ):
            raise ValueError(
                "Native warm P source has no current authenticated admission"
            )

    def _warm_p_handler(self, operation, payload):
        messages = self._warm_control
        if operation == "warm_authorize" and payload.keys() == {"source", "offer"}:
            source = messages.open("source", payload["source"])
            raw = payload["offer"]
            prompt = ColdPrefillPrompt(**raw["body"]["prompt"])
            offer = self._cold_prefill_config.control.open_offer(raw, prompt)
            if (
                prompt.epoch != source.lease.identity.epoch
                or not 0 < source.lease.requested_common_tokens < prompt.tokens - 1
                or prompt.request_id in self.requests
            ):
                raise ValueError(
                    "Native warm P offer changes the accepted source frontier"
                )
            now = monotonic_ns()
            for request_id, active in tuple(self._authorized_warm.items()):
                if now >= active.offer.arm_deadline_ns:
                    self._authorized_warm.pop(request_id)
            previous = self._authorized_warm.get(prompt.request_id)
            if previous is not None:
                if (previous.source, previous.offer) != (source, offer):
                    raise ValueError("Native warm P offer was already authorized")
                return {"nonce": previous.nonce}
            if len(self._authorized_warm) >= self._cold_prefill_config.max_pending:
                return {"status": "wait"}
            if self._prefill_return_verify is None:
                self.bind_terminal_return_offer_verifier(self._verify_authorized_warm)
            authorization = _AuthorizedWarm(source, offer, raw, token_hex(32))
            self._authorized_warm[prompt.request_id] = authorization
            return {"nonce": authorization.nonce}
        if operation == "warm_destination" and payload.keys() == {"request_id"}:
            request = self.requests.get(payload["request_id"])
            if (
                request is None
                or request.status is not RequestStatus.WAITING_FOR_REMOTE_KVS
            ):
                return {"destination": None}
            owned = self.terminal_return_destination(request)
            return {"destination": messages.seal("destination", owned)}
        if operation == "warm_submit" and payload.keys() == {"transfer", "timeout"}:
            transfer = messages.open("transfer", payload["transfer"])
            timeout = payload["timeout"]
            if type(timeout) not in (float, int) or not 0 < timeout <= 60:
                raise ValueError("Native warm P timeout must be bounded")
            self.submit_warm_terminal(transfer, timeout=timeout)
            return {"submitted": True}
        if operation in (
            "warm_poll",
            "warm_release",
            "warm_cancel",
        ) and payload.keys() == {"transfer"}:
            transfer = messages.open(
                "transfer", payload["transfer"], allow_expired=True
            )
            if operation == "warm_poll":
                return {
                    "fences": [
                        asdict(fence) for fence in self.warm_terminal_status(transfer)
                    ]
                }
            if operation == "warm_cancel":
                self.cancel_warm_terminal(transfer)
                return {"cancelled": True}
            self.retire_warm_terminal(transfer)
            return {"released": True}
        raise ValueError("Unsupported native warm P control operation")

    def add_request(self, request):
        params = request.kv_transfer_params or {}
        if WARM_ADMISSION_PARAM not in params:
            return super().add_request(request)
        authorized = self._authorized_warm.get(request.request_id)
        if authorized is None or params[WARM_ADMISSION_PARAM] != authorized.nonce:
            raise ValueError("Native warm P has no authorized request nonce")
        self.admit_warm_terminal(request, authorized.source, authorized.envelope)
        self._authorized_warm.pop(request.request_id)

    def _worker_rpc(self, method, *args):
        self._require_return_owner()
        if self._warm_worker_rpc is None:
            raise RuntimeError("Native warm P has no owner-scoped worker control")
        return tuple(
            self._warm_worker_rpc(method, timeout=_RPC_TIMEOUT_SECONDS, args=args)
        )

    def admit_warm_terminal(
        self,
        request: Request,
        source: GlmPublishedTerminalSource,
        signed_prefill_offer: dict,
    ) -> None:
        """Called by the trusted P owner before native request admission."""
        self._require_return_owner()
        deployment = self._cold_prefill_config
        collector = self._cold_prefill_collector
        params = request.kv_transfer_params or {}
        if (
            type(source) is not GlmPublishedTerminalSource
            or collector is None
            or deployment is None
            or _OFFER_PARAM in params
            or request.request_id in self.requests
            or request.request_id in self._cold_prefill_requests
        ):
            raise ValueError("Native warm P needs one fresh approved request")

        lease, manifest = source.lease, source.manifest
        prompt = _native_prompt(request, lease.identity.epoch)
        offer = deployment.control.open_offer(signed_prefill_offer, prompt)
        ticket = deployment.control.arm(offer, collector, self._cold_prefill_regions)
        try:
            stage_native_glm_k0_request(
                self.connector,
                request,
                offer.grant,
                ticket,
                imported_tokens=lease.requested_common_tokens,
                stage_warm=lambda: self.stage_terminal_return(
                    request, lease, manifest, offer.grant, before_admission=True
                ),
            )
        except BaseException:
            collector.cancel(ticket)
            raise
        self._cold_prefill_requests[request.request_id] = _Prefill(
            request, offer, ticket
        )
        super().add_request(request)

    def submit_warm_terminal(
        self, transfer: GlmTerminalReturn, *, timeout: float
    ) -> None:
        self.bind_terminal_return(transfer)
        request_id = transfer.destination.request_id
        self._warm_submissions[request_id] = transfer
        returned = self._worker_rpc(
            "kitepond_prefill_submit_terminal_return", transfer, timeout
        )
        if len(returned) != 4 or any(value is not None for value in returned):
            raise ValueError("Native warm P dispatch needs four initialized workers")

    def warm_terminal_status(
        self, transfer: GlmTerminalReturn
    ) -> tuple[PrefillReturnFence, ...]:
        request_id = transfer.destination.request_id
        if self._warm_submissions.get(request_id) != transfer:
            raise ValueError("Native warm P status differs from its active transfer")
        returned = self._worker_rpc("kitepond_prefill_terminal_return_result", transfer)
        if len(returned) != 4:
            raise ValueError("Native warm P status needs four initialized workers")
        fences = {}
        for fence in returned:
            if fence is None:
                continue
            if type(fence) is not PrefillReturnFence:
                raise ValueError("Native warm P status contains an invalid fence")
            rank = fence.receipt.local_rank
            if (
                rank not in _RANKS
                or rank in fences
                or fence.attempt != transfer.attempt(rank)
            ):
                raise ValueError("Native warm P status changed the admitted permit")
            fences[rank] = fence
        return tuple(fences[rank] for rank in sorted(fences))

    def cancel_warm_terminal(self, transfer: GlmTerminalReturn) -> None:
        """Cancel model scheduling while the workers drain their existing reads."""
        self._require_return_owner()
        request_id = transfer.destination.request_id
        if self._warm_submissions.get(request_id) != transfer:
            raise ValueError(
                "Native warm P cancellation differs from its active transfer"
            )
        self.finish_requests(request_id, RequestStatus.FINISHED_ABORTED)

    def retire_warm_terminal(self, transfer: GlmTerminalReturn) -> None:
        self._require_return_owner()
        request_id = transfer.destination.request_id
        if (
            self._warm_submissions.get(request_id) != transfer
            or request_id in self.requests
            or request_id in self._warm_returns
        ):
            raise ValueError("Native warm P still owns the admitted request")
        self._worker_rpc("kitepond_prefill_retire_terminal_return", transfer)
        self._warm_submissions.pop(request_id)
