# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use the native D owner for warm source publication and rank settlement."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from threading import get_ident
from typing import TYPE_CHECKING

from .glm_return_bootstrap import DecodeTerminalRankResult
from .glm_terminal_manifest import GlmTerminalManifest, SignedTerminalRank
from .glm_terminal_return import (
    GlmTerminalReturn,
    PrefillTerminalReservation,
    TerminalReturnSettlements,
)
from .terminal_checkpoint import TerminalPinLease, TerminalTransferIdentity

if TYPE_CHECKING:
    from vllm.v1.core.sched.interface import SynchronousStepRPC

_D_CONTROL = (
    "kitepond_terminal_checkpoint",
    "kitepond_register_terminal_return",
    "kitepond_terminal_return_result",
    "kitepond_retire_terminal_return",
    "kitepond_discard_terminal_frame",
)
_SETTLE_INTERVAL_NS = 50_000_000
_RANKS = frozenset(range(4))


@dataclass(frozen=True)
class GlmPublishedTerminalSource:
    lease: TerminalPinLease
    manifest: GlmTerminalManifest


@dataclass
class _WarmTransfer:
    transfer: GlmTerminalReturn
    registered: bool = False
    statuses: dict[int, str] = field(default_factory=dict)
    complete: bool = False


@dataclass(frozen=True)
class GlmReturnStatus:
    received_ranks: tuple[int, ...]
    aborted_ranks: tuple[int, ...]
    complete: bool


class GlmDecodeWarmReturns:
    """Publish accepted D state and consume the four native worker dispositions."""

    def __init__(
        self,
        scheduler,
        worker_rpc: SynchronousStepRPC,
        require_destination: Callable[[PrefillTerminalReservation], None],
    ) -> None:
        owner = scheduler._terminal_checkpoints
        capture = scheduler._terminal_capture
        if owner is None or capture is None or not callable(require_destination):
            raise ValueError("Native warm D needs its enrolled terminal owner")
        self._scheduler, self._owner, self._capture = scheduler, owner, capture
        self._worker_rpc, self._destination = worker_rpc, require_destination
        self._thread = get_ident()
        self.settlements = TerminalReturnSettlements(owner)
        self._sources: dict[str, GlmPublishedTerminalSource] = {}
        self._returns: dict[str, _WarmTransfer] = {}
        self._next_poll_ns = 0

    def _require_owner(self) -> None:
        if self._thread != get_ident():
            raise RuntimeError("Native warm D belongs to the scheduler owner")

    def _rpc(self, method: str, *args):
        return tuple(
            self._worker_rpc(method, timeout=self._capture.rpc_timeout, args=args)
        )

    def arm(self, identity: TerminalTransferIdentity) -> None:
        self._require_owner()
        if type(identity) is not TerminalTransferIdentity:
            raise ValueError("Native warm capture requires its exact identity")
        request = self._scheduler.requests.get(identity.request_id)
        if request is None:
            raise ValueError("Native D request cannot arm warm terminal state")
        self._owner.arm(request, identity)

    def publish(
        self, identity: TerminalTransferIdentity
    ) -> GlmPublishedTerminalSource | None:
        """Return the same accepted source only after its native host fence."""
        self._require_owner()
        if type(identity) is not TerminalTransferIdentity:
            raise ValueError("Native warm source needs the published identity")
        request_id = identity.request_id
        lease = self._owner.lease_for(request_id)
        if lease is None:
            return None
        if lease.identity != identity:
            raise ValueError("Native warm source changed its boot, epoch or request")
        if identity.deadline_monotonic_ns <= self._capture.clock():
            self._expire_source(request_id, lease)
            return None
        existing = self._sources.get(request_id)
        if existing is not None:
            if existing.lease is not lease:
                raise ValueError("Native warm source changed its live lease")
            return existing
        fence = self._capture.terminal_host_prefix
        if fence is not None:
            try:
                fence.require_durable(lease)
            except BlockingIOError:
                return None
        imported = self._capture.imported_boundary(lease)
        ranks = self._rpc("kitepond_terminal_checkpoint", lease, imported)
        if (
            len(ranks) != 4
            or any(type(rank) is not SignedTerminalRank for rank in ranks)
            or {rank.checkpoint.rank for rank in ranks} != _RANKS
        ):
            raise ValueError("Native warm D source needs four distinct rank proofs")
        manifest = GlmTerminalManifest(
            lease.sha256, tuple(sorted(ranks, key=lambda rank: rank.checkpoint.rank))
        )
        self._scheduler._verify_committed_terminal_manifest(lease, manifest)
        source = GlmPublishedTerminalSource(lease, manifest)
        self._sources[request_id] = source
        return source

    def prepare(
        self,
        source: GlmPublishedTerminalSource,
        destination: PrefillTerminalReservation,
    ) -> GlmTerminalReturn:
        """Reserve one verified D lease for the exact allocated P request."""
        self._require_owner()
        if type(source) is not GlmPublishedTerminalSource:
            raise ValueError("Native warm transfer needs its published terminal source")
        req_id = source.lease.identity.request_id
        current = self._sources.get(req_id)
        if current != source or type(destination) is not PrefillTerminalReservation:
            raise ValueError("Native warm transfer changed the published source")
        assert current is not None
        route = self._returns.get(req_id)
        if route is None:
            self._destination(destination)
            identity = current.lease.identity
            if (
                destination.grant.engine_id,
                destination.grant.boot_id,
                destination.grant.epoch,
            ) != (identity.source_engine_id, identity.source_boot_id, identity.epoch):
                raise ValueError("Native warm target belongs to another D authority")
            permit = self._owner.begin_reads(current.lease, current.manifest)
            transfer = self.settlements.prepare(
                current.lease, permit, current.manifest, destination
            )
            route = self._returns[req_id] = _WarmTransfer(transfer)
        elif route.transfer.destination != destination or route.complete:
            raise ValueError("Native warm source is already bound to another target")
        if not route.registered:
            self._owner.require_read_open(route.transfer.lease, route.transfer.permit)
            registered = self._rpc("kitepond_register_terminal_return", route.transfer)
            if len(registered) != 4 or any(value is not None for value in registered):
                raise ValueError("Native warm D registration needs four workers")
            route.registered = True
        return route.transfer

    def poll(self, transfer: GlmTerminalReturn) -> GlmReturnStatus:
        """Advance the real D read lease only via its authenticated rank receipts."""
        self._require_owner()
        req_id = transfer.lease.identity.request_id
        route = self._returns.get(req_id)
        if route is None or route.transfer != transfer or not route.registered:
            raise ValueError("Native warm D poll has no registered lease")
        if not route.complete:
            returned = self._rpc("kitepond_terminal_return_result", route.transfer)
            if len(returned) != 4:
                raise ValueError("Native warm D poll needs the four enrolled workers")
            accepted = {}
            for item in returned:
                if item is None:
                    continue
                if type(item) is not DecodeTerminalRankResult:
                    raise ValueError("Native warm D poll returned an invalid receipt")
                rank = item.proof.attempt.rank
                if rank in accepted or item.proof.attempt != transfer.attempt(rank):
                    raise ValueError("Native warm D receipt changed its lease or rank")
                accepted[rank] = item
            for rank, item in accepted.items():
                outcome = item.proof.outcome
                previous = route.statuses.get(rank)
                if previous is not None:
                    if previous != outcome:
                        raise ValueError("Native warm D changed a settled rank")
                    continue
                route.complete = self.settlements.accept_native_worker_result(
                    route.transfer, item
                )
                route.statuses[rank] = outcome
        return GlmReturnStatus(
            tuple(sorted(route.statuses)),
            tuple(
                sorted(
                    rank for rank, value in route.statuses.items() if value != "acked"
                )
            ),
            route.complete,
        )

    def _expire_source(self, request_id: str, lease: TerminalPinLease) -> None:
        if self._owner.expire(lease):
            self._sources.pop(request_id, None)
            self._rpc("kitepond_discard_terminal_frame", lease)

    def cancel(self, source: GlmPublishedTerminalSource) -> bool:
        self._require_owner()
        req_id = source.lease.identity.request_id
        current = self._sources.get(req_id)
        if current is None or current != source:
            raise ValueError("Native warm cancellation changed the live source")
        if not self._owner.cancel(current.lease):
            return False
        self._sources.pop(req_id)
        self._rpc("kitepond_discard_terminal_frame", current.lease)
        return True

    def retire(self, transfer: GlmTerminalReturn) -> None:
        self._require_owner()
        req_id = transfer.lease.identity.request_id
        route = self._returns.get(req_id)
        if route is None or route.transfer != transfer or not route.complete:
            raise ValueError("Native warm D source is still waiting for its quorum")
        self._rpc("kitepond_retire_terminal_return", route.transfer)
        self._rpc("kitepond_discard_terminal_frame", route.transfer.lease)
        self._returns.pop(req_id)
        self._sources.pop(req_id)

    def tick(self) -> None:
        """Drain active worker receipts on the native loop with bounded cadence."""
        self._require_owner()
        if not self._sources:
            return
        now = time.monotonic_ns()
        if now < self._next_poll_ns:
            return
        self._next_poll_ns = now + _SETTLE_INTERVAL_NS
        for req_id, source in tuple(self._sources.items()):
            route = self._returns.get(req_id)
            if route is not None and route.registered and not route.complete:
                self.poll(route.transfer)
            elif route is None and source.lease.identity.deadline_monotonic_ns <= now:
                self._expire_source(req_id, source.lease)


class GlmWarmDecodeMixin:
    """Enroll reverse control at startup without replacing cold/native owners."""

    _warm_decode_returns: GlmDecodeWarmReturns | None = None

    @property
    def native_glm_warm_returns(self) -> GlmDecodeWarmReturns:
        owner = self._warm_decode_returns
        if owner is None:
            raise RuntimeError("Native GLM D warm return is not enabled")
        return owner

    def get_owner_worker_control_registration(self):
        parent = super().get_owner_worker_control_registration()
        additional = self.vllm_config.additional_config
        if (
            not isinstance(additional, Mapping)
            or "kitepond_glm_terminal_return" not in additional
        ):
            return parent

        def bind(rpc):
            if parent is not None:
                parent[1](rpc)
            from .glm_warm_control import NAMESPACE, GlmWarmControl

            deployment = getattr(self, "_cold_decode_config", None)
            if deployment is None:
                raise ValueError(
                    "Native warm D requires its existing authenticated admission"
                )
            loop = GlmDecodeWarmReturns(self, rpc, self._require_warm_destination)
            self.bind_terminal_transport_settlement(loop.settlements.verify)
            self._warm_decode_returns = loop
            self._warm_decode_control = GlmWarmControl(deployment.control)
            self._warm_decode_intents = {}
            self._cold_decode_owner.bind_control(NAMESPACE, self._warm_d_handler)

        return ((*(() if parent is None else parent[0]), *_D_CONTROL), bind)

    def _warm_d_handler(self, operation, payload):
        loop = self.native_glm_warm_returns
        messages = self._warm_decode_control
        if operation in ("warm_arm", "warm_source") and payload.keys() == {"identity"}:
            identity = TerminalTransferIdentity(**payload["identity"])
            if operation == "warm_source":
                source = loop.publish(identity)
                return {
                    "source": None
                    if source is None
                    else messages.seal("source", source)
                }
            offered = self._cold_decode_offered.get(identity.request_id)
            if (
                offered is None
                or identity.epoch != offered.grant.epoch
                or (identity.source_engine_id, identity.source_boot_id)
                != (offered.grant.engine_id, offered.grant.boot_id)
                or identity.deadline_monotonic_ns <= time.monotonic_ns()
            ):
                raise ValueError(
                    "Native warm D intent needs its current admitted lease"
                )
            if len(self._warm_decode_intents) >= self._cold_decode_config.max_pending:
                for request_id, old in tuple(self._warm_decode_intents.items()):
                    if (
                        old.deadline_monotonic_ns <= time.monotonic_ns()
                        and request_id not in self.requests
                    ):
                        self._warm_decode_intents.pop(request_id)
                    if (
                        len(self._warm_decode_intents)
                        < self._cold_decode_config.max_pending
                    ):
                        break
                if (
                    identity.request_id not in self._warm_decode_intents
                    and len(self._warm_decode_intents)
                    >= self._cold_decode_config.max_pending
                ):
                    return {"status": "wait"}
            previous = self._warm_decode_intents.get(identity.request_id)
            if previous is not None and previous != identity:
                raise ValueError("Native warm D intent was already registered")
            if previous is None and identity.request_id in self.requests:
                loop.arm(identity)
            self._warm_decode_intents[identity.request_id] = identity
            return {"armed": True}
        if operation == "warm_prepare" and payload.keys() == {"source", "destination"}:
            source = messages.open("source", payload["source"])
            destination = messages.open("destination", payload["destination"])
            return {
                "transfer": messages.seal("transfer", loop.prepare(source, destination))
            }
        if operation in ("warm_poll", "warm_release") and payload.keys() == {
            "transfer"
        }:
            transfer = messages.open(
                "transfer", payload["transfer"], allow_expired=True
            )
            if operation == "warm_poll":
                return asdict(loop.poll(transfer))
            loop.retire(transfer)
            self._warm_decode_intents.pop(transfer.lease.identity.request_id, None)
            return {"released": True}
        if operation == "warm_cancel" and payload.keys() == {"source"}:
            source = messages.open("source", payload["source"], allow_expired=True)
            released = loop.cancel(source)
            if released:
                self._warm_decode_intents.pop(source.lease.identity.request_id, None)
            return {"released": released}
        raise ValueError("Unsupported native warm D control operation")

    def add_request(self, request):
        loop = self._warm_decode_returns
        identity = (
            None if loop is None else self._warm_decode_intents.get(request.request_id)
        )
        if (
            identity is not None
            and identity.deadline_monotonic_ns <= time.monotonic_ns()
        ):
            raise ValueError(
                "Native warm D terminal reservation expired before admission"
            )
        super().add_request(request)
        if identity is not None:
            loop.arm(identity)

    def _require_warm_destination(
        self, destination: PrefillTerminalReservation
    ) -> None:
        offers = getattr(self, "_cold_decode_offered", {})
        authority = getattr(self, "_cold_decode_authority", None)
        offer = offers.get(destination.request_id)
        if (
            offer is None
            or authority is None
            or destination.grant != offer.grant
            or not authority.retains(offer.grant)
        ):
            raise ValueError("Native warm P target has no active D reservation")

    def schedule(self, *args, **kwargs):
        loop = self._warm_decode_returns
        if loop is not None:
            loop.tick()
        return super().schedule(*args, **kwargs)
