# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""First-prefill lease operations on the actual P and D scheduler owners."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from kitepond.core.contracts import Wait
from kitepond.core.readiness import ReadyAttempt, ReadyReceipt, _digest
from kitepond.models.glm5next.shared_mla import (
    SHARED_MLA_CONFIG,
    SharedMlaRankPages,
    confirm_shared_mla_elision,
)

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from .glm_cold_prefill_admission import (
    ColdPrefillAdmission,
    ColdPrefillControl,
    ColdPrefillOffer,
    ColdPrefillPrompt,
)
from .glm_prefill_bootstrap import stage_native_glm_k0_request
from .glm_settlement import OPERATION as SETTLEMENT_OPERATION
from .glm_settlement import observe_glm_owner_settlement
from .hisparse_attachment import HiSparseConnector
from .ready_transport import ReadyCollector, ReadyTicket, ReadyWorker

_ADMISSION_CONFIG = "kitepond_glm_cold_prefill"
_KEY_ENV = "KITEPOND_GLM_COLD_PREFILL_KEY_FILE"
_OFFER_PARAM = "kitepond_cold_prefill_offer"
_NAMESPACE = "kitepond.glm-cold-prefill.v1"


@dataclass(frozen=True)
class _Deployment:
    control: ColdPrefillControl
    max_pending: int
    arm_timeout_seconds: float
    stage: str
    directory: Path | None

    @classmethod
    def from_scheduler(cls, scheduler, stage, owner):
        options = scheduler.vllm_config.additional_config
        raw = options.get(_ADMISSION_CONFIG)
        if raw is None:
            return None
        if (
            not isinstance(raw, Mapping)
            or raw.keys() != {"p_boot_id", "max_pending", "arm_timeout_seconds"}
            or scheduler.scheduler_config.async_scheduling is not False
            or owner.stage != stage
        ):
            raise ValueError("Cold prefill requires its explicit synchronous role")
        path = os.environ.get(_KEY_ENV)
        if not path or not Path(path).is_absolute():
            raise ValueError("Cold prefill needs an absolute private control key file")
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        with os.fdopen(os.open(path, flags), "rb") as handle:
            info = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) not in {0o400, 0o600}
                or info.st_size != 32
            ):
                raise ValueError(
                    "Cold prefill key must be a private 32-byte local file"
                )
            key = handle.read(33)
        directory = None
        if stage == "prefill":
            k0 = options.get("kitepond_glm_prefill_k0")
            if not isinstance(k0, Mapping) or not k0.get("ready_directory"):
                raise ValueError(
                    "Cold prefill requires the native P K0 READY directory"
                )
            directory = Path(k0["ready_directory"])
            if not directory.is_absolute():
                raise ValueError("Native P K0 READY directory must be absolute")
        return cls(
            ColdPrefillControl(
                d_engine_id=owner._engine_id,
                d_boot_id=owner._boot_id,
                p_boot_id=raw["p_boot_id"],
                key=key,
            ),
            raw["max_pending"],
            raw["arm_timeout_seconds"],
            stage,
            directory,
        )


def _compatible_owner(scheduler):
    connector = scheduler.connector
    if (
        type(connector) is not MultiConnector
        or len(connector.sub_connectors) != 2
        or type(connector.sub_connectors[0]) is not HiSparseConnector
    ):
        raise ValueError("Cold prefill needs the native HiSparse and Mooncake pair")
    return connector.sub_connectors[0]


def _native_prompt(request: Request, epoch: int) -> ColdPrefillPrompt:
    params = request.kv_transfer_params
    tokens = request.prompt_token_ids
    if (
        not isinstance(params, dict)
        or not isinstance(tokens, list)
        or len(tokens) != request.num_prompt_tokens
        or request.resumable
    ):
        raise ValueError("Cold prefill transport needs one native raw-token request")
    return ColdPrefillPrompt.from_tokens(
        request.request_id, params["transfer_id"], epoch, tokens
    )


@dataclass
class _Prefill:
    request: Request
    offer: ColdPrefillOffer
    ticket: ReadyTicket
    receipts: tuple[ReadyReceipt, ...] | None = None


class GlmColdPrefillMixin:
    """Check the D lease before staging P and collect its actual write fences."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cold_prefill_config = None
        if _ADMISSION_CONFIG not in (self.vllm_config.additional_config or {}):
            return
        self._cold_prefill_owner = _compatible_owner(self)
        deployment = _Deployment.from_scheduler(
            self, "prefill", self._cold_prefill_owner
        )
        assert deployment is not None
        self._cold_prefill_config = deployment
        self._cold_prefill_collector: ReadyCollector | None = None
        self._cold_prefill_requests: dict[str, _Prefill] = {}
        self._cold_prefill_regions: dict[int, str] = {}
        self._cold_prefill_owner.bind_control(_NAMESPACE, self._cold_prefill_handler)

    def get_worker_startup_callback(self):
        base = getattr(super(), "get_worker_startup_callback", None)
        inherited = base() if base is not None else None
        deployment = self._cold_prefill_config
        if deployment is None:
            return inherited

        def enroll(rpc):
            if inherited is not None:
                inherited(rpc)
            workers = {}
            regions = {}
            rows = rpc("kitepond_prefill_ready_identity", timeout=10.0)
            if len(rows) != 4:
                raise ValueError("Native P READY enrollment needs four workers")
            for row in rows:
                rank = row["rank"]
                if (
                    rank in workers
                    or rank not in {0, 1, 2, 3}
                    or not _digest(row["region_sha256"])
                ):
                    raise ValueError("Native P READY enrollment has a foreign rank")
                workers[rank] = ReadyWorker.from_pid(row["pid"])
                regions[rank] = row["region_sha256"]
            self._cold_prefill_collector = ReadyCollector(
                deployment.directory, workers, max_pending=deployment.max_pending
            )
            self._cold_prefill_regions = regions

        return enroll

    def add_request(self, request):
        deployment = self._cold_prefill_config
        params = request.kv_transfer_params or {}
        if deployment is not None and _OFFER_PARAM in params:
            collector = self._cold_prefill_collector
            envelope = params[_OFFER_PARAM]
            if (
                collector is None
                or request.request_id in self.requests
                or request.request_id in self._cold_prefill_requests
                or not isinstance(envelope, dict)
                or not isinstance(envelope.get("body"), dict)
                or not isinstance(envelope["body"].get("prompt"), dict)
            ):
                raise ValueError(
                    "Native P cold lease requires an enrolled fresh request"
                )
            prompt = _native_prompt(request, envelope["body"]["prompt"].get("epoch"))
            offer = deployment.control.open_offer(envelope, prompt)
            ticket = deployment.control.arm(
                offer, collector, self._cold_prefill_regions
            )
            try:
                stage_native_glm_k0_request(
                    self.connector, request, offer.grant, ticket
                )
            except Exception:
                collector.cancel(ticket)
                raise
            self._cold_prefill_requests[request.request_id] = _Prefill(
                request, offer, ticket
            )
        super().add_request(request)

    def _cold_prefill_handler(self, operation, payload):
        if operation == SETTLEMENT_OPERATION and not payload:
            return observe_glm_owner_settlement(self, self._cold_prefill_owner)
        if operation == "status" and not payload:
            return {
                "stage": "prefill",
                "max_pending": self._cold_prefill_config.max_pending,
                "registered_ranks": len(self._cold_prefill_regions),
                "active_leases": len(self._cold_prefill_requests),
                "ready_leases": sum(
                    item.receipts is not None
                    for item in self._cold_prefill_requests.values()
                ),
            }
        if operation != "read_ready":
            raise ValueError("Unsupported native P cold prefill operation")
        attempt = ReadyAttempt(**payload)
        state = self._cold_prefill_requests.get(attempt.request_id)
        if state is None or state.offer.attempt != attempt:
            raise ValueError("Native P READY query does not match a live lease")
        collector = self._cold_prefill_collector
        deployment = self._cold_prefill_config
        assert collector is not None and deployment is not None
        if state.receipts is None:
            state.receipts = collector.take(state.ticket)
        if state.receipts is None:
            return {"status": "wait"}
        return {
            "status": "ready",
            "receipt_batch": {
                "attempt": asdict(attempt),
                "receipts": [asdict(item) for item in state.receipts],
                "mac": deployment.control.ready_mac(state.offer, state.receipts),
            },
        }

    def _free_blocks(self, request):
        super()._free_blocks(request)
        if self._cold_prefill_config is None:
            return
        state = self._cold_prefill_requests.pop(request.request_id, None)
        if state is not None:
            self._cold_prefill_collector.cancel(state.ticket)
            self._cold_prefill_owner.retire_prefill_grant(request)

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            if (
                self._cold_prefill_config is not None
                and self._cold_prefill_collector is not None
            ):
                self._cold_prefill_collector.close()


class GlmColdPrefillScheduler(GlmColdPrefillMixin, Scheduler):
    """Native P scheduler for the single cold handoff verification profile."""


class GlmColdDecodeAdmissionMixin:
    """Reserve and adopt only D-owned native pages for this same-boot attempt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cold_decode_config = None
        self._cold_mla_rpc = None
        self._cold_mla_decisions: dict[str, tuple[str, ...] | None] = {}
        mla_preflight = (self.vllm_config.additional_config or {}).get(
            SHARED_MLA_CONFIG, False
        )
        if type(mla_preflight) is not bool:
            raise ValueError("GLM MLA preflight needs an explicit boolean")
        self._cold_mla_preflight = mla_preflight
        if _ADMISSION_CONFIG not in (self.vllm_config.additional_config or {}):
            return
        self._cold_decode_owner = _compatible_owner(self)
        deployment = _Deployment.from_scheduler(self, "decode", self._cold_decode_owner)
        assert deployment is not None
        self._cold_decode_config = deployment
        self._cold_decode_authority = self._cold_decode_owner.authority
        if self._cold_decode_authority is None:
            raise ValueError(
                "Native D cold reservation requires its existing allocator"
            )
        self._cold_decode_admission = ColdPrefillAdmission(
            self._cold_decode_authority,
            deployment.control,
            context_limit=self.vllm_config.model_config.max_model_len,
            max_pending=deployment.max_pending,
            arm_timeout_seconds=deployment.arm_timeout_seconds,
        )
        self._cold_decode_offered: dict[str, ColdPrefillOffer] = {}
        self._cold_decode_verified: set[str] = set()
        self._cold_decode_staged: dict[str, ColdPrefillOffer] = {}
        self._cold_decode_owner.bind_control(_NAMESPACE, self._cold_decode_handler)

    def get_owner_worker_control_registration(self):
        parent = super().get_owner_worker_control_registration()
        if not self._cold_mla_preflight:
            return parent
        if self._cold_decode_config is None:
            raise ValueError("Shared MLA preflight requires the native D cold owner")

        def bind(rpc):
            if parent is not None:
                parent[1](rpc)
            self._cold_mla_rpc = rpc

        return (
            (*(() if parent is None else parent[0]), "kitepond_prepare_shared_mla"),
            bind,
        )

    def _decide_shared_mla(self, offer, receipts, mac):
        rpc = self._cold_mla_rpc
        if rpc is None:
            raise ValueError("Shared MLA preflight has no native worker control")
        try:
            responses = rpc("kitepond_prepare_shared_mla", timeout=4.0, args=(offer,))
        except TimeoutError:
            return None
        if len(responses) != 4 or any(
            type(row) is not SharedMlaRankPages for row in responses
        ):
            return None
        rows = {row.descriptor.source_rank: row for row in responses}
        # Each observation was returned only after its own authenticated wire
        # comparison. P never supplies addresses or device state to this RPC.
        proof = confirm_shared_mla_elision(
            self._cold_decode_config.control,
            self._cold_decode_authority,
            offer,
            receipts,
            mac,
            rows,
            rows,
        )
        return None if proof is None else proof.rank_digests

    def _cold_decode_handler(self, operation, payload):
        if operation == SETTLEMENT_OPERATION and not payload:
            return observe_glm_owner_settlement(self, self._cold_decode_owner)
        admission = self._cold_decode_admission
        if operation == "status" and not payload:
            return {
                "stage": "decode",
                "max_pending": self._cold_decode_config.max_pending,
                "native_pool": asdict(self._cold_decode_authority.observe()),
                "active_leases": len(self._cold_decode_offered),
                "ready_leases": len(self._cold_decode_verified),
                "scheduled_imports": len(self._cold_decode_staged),
            }
        if operation == "reserve":
            prompt = ColdPrefillPrompt(**payload)
            offer = admission.offer(prompt)
            if isinstance(offer, Wait):
                return {"status": "wait", "reason": offer.reason}
            self._cold_decode_offered[prompt.request_id] = offer
            return {
                "status": "offered",
                "offer": self._cold_decode_config.control.export_offer(offer),
            }
        if operation != "confirm_ready" or payload.keys() != {
            "attempt",
            "receipts",
            "mac",
        }:
            raise ValueError("Unsupported native D cold prefill operation")
        attempt = ReadyAttempt(**payload["attempt"])
        offer = self._cold_decode_offered.get(attempt.request_id)
        if (
            offer is None
            or offer.attempt != attempt
            or not isinstance(payload["receipts"], list)
            or len(payload["receipts"]) != 4
        ):
            raise ValueError(
                "Native D READY did not identify the issued four-rank lease"
            )
        receipts = []
        for raw in payload["receipts"]:
            item = dict(raw)
            item["attempt"] = ReadyAttempt(**item["attempt"])
            receipts.append(ReadyReceipt(**item))
        receipts = tuple(receipts)
        admission.verify_ready(offer, receipts, payload["mac"])
        request_id = attempt.request_id
        if self._cold_mla_preflight and request_id not in self._cold_mla_decisions:
            self._cold_mla_decisions[request_id] = self._decide_shared_mla(
                offer, receipts, payload["mac"]
            )
        self._cold_decode_verified.add(request_id)
        result = {"status": "ready"}
        if self._cold_mla_preflight:
            result["mla_transport"] = (
                "same_pages" if self._cold_mla_decisions[request_id] else "full"
            )
        return result

    def add_request(self, request):
        params = request.kv_transfer_params or {}
        offered = (
            self._cold_decode_config is not None
            and request.request_id in self._cold_decode_offered
        )
        if offered:
            offer = self._cold_decode_offered[request.request_id]
            if (
                request.request_id not in self._cold_decode_verified
                or request.request_id in self.requests
                or params.get("do_remote_prefill") is not True
                or _native_prompt(request, offer.prompt.epoch) != offer.prompt
            ):
                raise ValueError(
                    "Native D request does not match its authenticated prefill"
                )
            self._cold_decode_owner.stage_host_grant(
                request,
                offer.grant,
                imported_tokens=offer.imported_tokens,
                mla_rank_digests=self._cold_mla_decisions.get(request.request_id),
            )
            self._cold_decode_staged[request.request_id] = offer
        super().add_request(request)

    def _forget_consumed_cold_lease(
        self, request_id: str, offer: ColdPrefillOffer
    ) -> None:
        if not self._cold_decode_authority.retains(offer.grant):
            self._cold_decode_admission.detach_adopted(offer)
            self._cold_decode_staged.pop(request_id)
            self._cold_decode_offered.pop(request_id)
            self._cold_decode_verified.discard(request_id)
            self._cold_mla_decisions.pop(request_id, None)

    def schedule(self, *args, **kwargs):
        result = super().schedule(*args, **kwargs)
        if self._cold_decode_config is not None:
            for request_id, offer in tuple(self._cold_decode_staged.items()):
                self._forget_consumed_cold_lease(request_id, offer)
        return result

    def _free_blocks(self, request):
        super()._free_blocks(request)
        if self._cold_decode_config is not None:
            offer = self._cold_decode_staged.get(request.request_id)
            if offer is not None:
                self._forget_consumed_cold_lease(request.request_id, offer)
