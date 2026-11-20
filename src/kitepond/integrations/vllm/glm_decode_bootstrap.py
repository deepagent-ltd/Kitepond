# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run cold GLM imports from exact vLLM decode allocation tickets."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import torch
from kitepond.core.readiness import ReadyAttempt, _digest
from kitepond.integrations.mooncake.live_receive import (
    AbortedRankWindows,
    FencedRankWindows,
    RankWindowAttempt,
)
from kitepond.integrations.mooncake.named_control import connect_glm_window_peer
from kitepond.models.glm5next.import_recipe import part_identity
from kitepond.models.glm5next.layout import CacheRole, Glm5NextPDLayout
from kitepond.models.glm5next.live_windows import GlmRankWindowPlan
from kitepond.models.glm5next.shared_mla import (
    SHARED_MLA_CONFIG,
    SharedMlaRankPages,
    VerifiedSharedMlaElision,
)

from vllm.logger import init_logger

from .decode_receive import DecodeReceiveTicket, DecodeReceiveWork
from .glm_live_receive import GlmNativeWindowReceiver

if TYPE_CHECKING:
    from .glm_cold_prefill_admission import ColdPrefillOffer
    from .glm_cold_restore_control import GlmWorkerColdRestore
    from .glm_native_views import GlmNativeStateViews

_MLA_PREFLIGHT_TIMEOUT = 2.0
_COLD_IMPORT_CONFIG = "kitepond_glm_cold_import"
logger = init_logger(__name__)


@dataclass(frozen=True)
class GlmColdImportProfile:
    prefill_bootstrap: str
    timeout_seconds: float

    @classmethod
    def from_config(cls, config) -> GlmColdImportProfile | None:
        options = config.additional_config
        raw = options.get(_COLD_IMPORT_CONFIG) if isinstance(options, Mapping) else None
        if raw is None:
            return None
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"prefill_bootstrap", "timeout_seconds"}
            or type(raw["prefill_bootstrap"]) is not str
            or not 0 < len(raw["prefill_bootstrap"]) <= 256
            or type(raw["timeout_seconds"]) not in (int, float)
            or not 0 < raw["timeout_seconds"] <= 60
            or config.scheduler_config.async_scheduling is not False
        ):
            raise ValueError(
                "Native cold import needs its explicit synchronous profile"
            )
        address = urlsplit(raw["prefill_bootstrap"])
        if (
            address.scheme != "http"
            or not address.hostname
            or not address.port
            or address.path
            or address.query
            or address.fragment
            or address.username is not None
            or address.password is not None
        ):
            raise ValueError("Native cold import needs its own P bootstrap endpoint")
        return cls(**raw)


@dataclass
class _Import:
    ticket: DecodeReceiveTicket
    future: Future
    started: bool = False
    reported_error: bool = False
    detached: bool = False


def forward_payload_receipt(
    ticket: DecodeReceiveTicket, plan: GlmRankWindowPlan
) -> dict:
    """Report exactly the payload of the already committed current wire plan."""
    by_role = {role.value: 0 for role in CacheRole}
    for row in plan.manifest():
        by_role[part_identity(row["role"]).cache_role.value] += row["nbytes"]
    omitted = 0
    if ticket.mla_rank_digests is not None:
        omitted = sum(
            part.nbytes
            for part in plan.final.manifest()
            if part_identity(part.role).cache_role is CacheRole.MLA
        )
    committed = sum(by_role.values())
    return {
        "request_id": ticket.request_id,
        "transfer_id": ticket.transfer_id,
        "rank": plan.final.rank,
        "tokens": ticket.imported_tokens,
        "mla_transport": "same_pages" if ticket.mla_rank_digests else "full",
        "payload_bytes_by_role": by_role,
        "committed_payload_bytes": committed,
        "shared_mla_omitted_bytes": omitted,
        "full_equivalent_bytes": committed + omitted,
    }


class GlmDecodeImportRuntime:
    """Schedule rank reads on the native Mooncake receiver loop."""

    def __init__(
        self,
        multi,
        runner,
        views: GlmNativeStateViews,
        scope,
        profile: GlmColdImportProfile,
        *,
        p_engine_id: str,
        model_sha256: str,
        config_sha256: str,
        main_stream: torch.cuda.Stream | None,
        receive_stream: torch.cuda.Stream | None,
        capacity: int,
    ) -> None:
        self._hisparse, mooncake = multi.sub_connectors
        worker = mooncake.connector_worker
        descriptor = views.descriptor
        if (
            not _digest(model_sha256)
            or not _digest(config_sha256)
            or scope.reverse
            or not p_engine_id
            or type(capacity) is not int
            or capacity < 1
            or worker.is_kv_consumer
            or worker.is_kv_producer
            or worker.tp_rank != views.rank
            or scope.rank != views.rank
            or views.rank != descriptor.source_rank
            or worker.engine_id != descriptor.authority_engine_id
            or scope.decode_boot_id != descriptor.authority_boot_id
            or (main_stream is None) != (receive_stream is None)
        ):
            raise ValueError("Native D cold import belongs to another initialized rank")
        self.worker, self.views, self.scope = worker, views, scope
        self.runner, self.profile = runner, profile
        self.p_engine_id = p_engine_id
        self._model_sha256, self._config_sha256 = model_sha256, config_sha256
        self._main_stream, self._receive_stream = main_stream, receive_stream
        self._capacity = capacity
        self._receiver = GlmNativeWindowReceiver(multi, slots=2)
        self._observe_forward_payload = worker.vllm_config.additional_config.get(
            SHARED_MLA_CONFIG, False
        )
        self._imports: dict[str, _Import] = {}
        self._current_tickets: set[str] = set()
        self._finished: set[str] = set()
        self._detached: GlmWorkerColdRestore | None = None

    def bind_detached_restore(self, restore: GlmWorkerColdRestore) -> None:
        if self._detached is not None or restore.views is not self.views:
            raise ValueError("Detached GLM import has a different rank authority")
        self._detached = restore

    def _layout(self, request_id: str, epoch: int, tokens: int) -> Glm5NextPDLayout:
        return Glm5NextPDLayout(
            request_id,
            self._model_sha256,
            self._config_sha256,
            0,
            epoch,
            tokens,
            tokens,
            0,
            "decode_ready",
        )

    def prepare_shared_mla(self, offer: ColdPrefillOffer) -> SharedMlaRankPages | None:
        """Match the actual local projection to the authenticated P observation."""
        grant = offer.grant
        if (
            grant.engine_id != self.worker.engine_id
            or grant.boot_id != self.scope.decode_boot_id
            or grant.prefix_tokens
            or grant.readonly_pages
            or grant.copy_tail is not None
        ):
            return None
        identity = offer.attempt
        try:
            local = self.views.project_shared_mla(
                self._layout(
                    identity.request_id, identity.epoch, offer.imported_tokens
                ),
                grant.writable_pages,
                identity,
            )
        except ValueError:
            return None
        attempt = RankWindowAttempt(
            identity.request_id,
            identity.transfer_id,
            identity.epoch,
            self.views.rank,
            self.scope.source_boot_id,
            identity.lease_sha256,
            offer.prompt.prompt_sha256,
            local.sha256,
        )

        async def query() -> bool:
            async with asyncio.timeout(_MLA_PREFLIGHT_TIMEOUT):
                async with connect_glm_window_peer(
                    self.worker,
                    self.p_engine_id,
                    self.profile.prefill_bootstrap,
                    self.scope,
                    discovery_timeout=_MLA_PREFLIGHT_TIMEOUT,
                ) as peer:
                    return await peer.prepare_mla(attempt) == local.sha256

        future = asyncio.run_coroutine_threadsafe(query(), self.worker.receiver_loop)
        try:
            return (
                local if future.result(timeout=_MLA_PREFLIGHT_TIMEOUT + 0.5) else None
            )
        except (TimeoutError, ValueError, OSError):
            # This query holds no remote registered window and mutates no cache.
            future.cancel()
            return None

    def _plan(self, ticket: DecodeReceiveTicket):
        if (
            ticket.prefix_tokens
            or ticket.copy_tail is not None
            or (ticket.engine_id, ticket.boot_id)
            != (self.worker.engine_id, self.scope.decode_boot_id)
        ):
            raise ValueError("Native D cold import needs its own cold allocation")
        layout = self._layout(ticket.request_id, ticket.epoch, ticket.imported_tokens)
        final, groups = self.views.for_import(layout, ticket.block_ids)
        self.views.require_owned_layout(final, groups)
        plan = GlmRankWindowPlan(final, 0)
        if ticket.mla_rank_digests is not None:
            attempt = ReadyAttempt(
                ticket.request_id, ticket.transfer_id, ticket.epoch, ticket.grant_sha256
            )
            plan = plan.for_forward_shared_mla(
                VerifiedSharedMlaElision(attempt, ticket.mla_rank_digests),
                self.views.descriptor,
                attempt,
            )
        return plan, groups

    async def _receive(
        self,
        ticket: DecodeReceiveTicket,
        plan: GlmRankWindowPlan,
        groups: dict[str, int],
        fence: torch.cuda.Event | None,
    ) -> None:
        timeout = self.profile.timeout_seconds
        mailbox = self._hisparse.decode_receive_mailbox()
        ready = False
        try:
            async with connect_glm_window_peer(
                self.worker,
                self.p_engine_id,
                self.profile.prefill_bootstrap,
                self.scope,
                discovery_timeout=timeout,
            ) as peer:
                ready = True
                await self._receiver.receive(
                    ticket,
                    plan,
                    groups,
                    peer,
                    source_boot_id=self.scope.source_boot_id,
                    timeout=timeout,
                    stream=self._receive_stream,
                    cow_event=fence,
                    seed_imported_tail=self.views.seed_imported_tail,
                    read_shared_mla=True,
                    forward_abort_peer=peer if self.scope.forward_abort else None,
                )
        except Exception:
            if not ready:
                # Discovery failed before any remote write could be submitted.
                mailbox._start_import(ticket)
                mailbox._post_fenced(ticket, "abort_settled")
            raise
        if self._observe_forward_payload:
            logger.info(
                "Kitepond forward committed payload %s",
                json.dumps(forward_payload_receipt(ticket, plan), sort_keys=True),
            )

    async def _abort_unopened_preflight(
        self, ticket: DecodeReceiveTicket, error
    ) -> None:
        """A mismatched allocation cannot downgrade after shared MLA was chosen."""
        mailbox = self._hisparse.decode_receive_mailbox()
        mailbox._start_import(ticket)
        # No data window was opened on this worker. This fixed abort-only
        # manifest can cancel an unopened authenticated P source but cannot
        # act on another live attempt which already has a different manifest.
        attempt = RankWindowAttempt(
            ticket.request_id,
            ticket.transfer_id,
            ticket.epoch,
            self.views.rank,
            self.scope.source_boot_id,
            ticket.grant_sha256,
            ticket.destination_sha256,
            ticket.grant_sha256,
            ticket.mla_rank_digests,
        )
        timeout = self.profile.timeout_seconds
        async with asyncio.timeout(timeout):
            async with connect_glm_window_peer(
                self.worker,
                self.p_engine_id,
                self.profile.prefill_bootstrap,
                self.scope,
                discovery_timeout=timeout,
            ) as peer:
                stopped = await peer.abort(attempt, self._receiver._pool.pool_id)
                if (
                    type(stopped) is not AbortedRankWindows
                    or stopped.attempt != attempt
                ):
                    raise ValueError("Shared MLA mismatch lacks a native abort fence")
                await peer.settle(FencedRankWindows(attempt, "abort_settled"))
        mailbox._post_fenced(ticket, "abort_settled")
        raise error

    def hisparse_metadata(self):
        return self._hisparse._get_connector_metadata()

    async def _receive_detached(
        self,
        ticket: DecodeReceiveTicket,
        fence: torch.cuda.Event | None,
    ) -> None:
        restore = self._detached
        assert restore is not None
        mailbox = self._hisparse.decode_receive_mailbox()
        pool = self._receiver._pool
        try:
            window = await pool.acquire(self.profile.timeout_seconds)
        except BaseException:
            restore.abort_unstarted(mailbox, ticket)
            raise
        try:
            restore.import_ticket(
                mailbox,
                ticket,
                window.tensor,
                stream=self._receive_stream,
                writes_before_import=fence,
            )
        finally:
            await pool.release(window, writes_settled=restore.fence_settled(ticket))

    def bind_step(self, metadata, request_ids=(), finished=()) -> None:
        tickets = metadata.tickets if isinstance(metadata, DecodeReceiveWork) else ()
        self._current_tickets = {ticket.request_id for ticket in tickets}
        if len(self._current_tickets) != len(tickets) or len(tickets) > self._capacity:
            raise ValueError("Native D cold ticket batch exceeds its rank capacity")
        self.retire_finished(finished)
        for supplied in tickets:
            self._submit(supplied)
        for request_id in request_ids:
            active = self._imports.get(request_id)
            if active is not None and not active.started:
                # Scheduler eligibility follows the native rank proof. Wait for
                # the local coroutine to return after publishing that proof.
                active.future.result(timeout=1)
                slot = self.runner.req_states.req_id_to_index[request_id]
                self.runner.start_imported_frontier(
                    request_id, active.ticket.imported_tokens, slot
                )
                active.started = True

    def _submit(self, supplied: DecodeReceiveTicket) -> None:
        mailbox = self._hisparse.decode_receive_mailbox()
        ticket = mailbox.ticket_for(
            supplied.request_id, supplied.transfer_id, supplied.epoch
        )
        if ticket != supplied:
            raise ValueError("Native D cold import changed its admitted ticket")
        active = self._imports.get(ticket.request_id)
        if active is not None:
            if active.ticket != ticket:
                raise ValueError("Native D cold import changed its allocation")
            return
        fence = None
        if self._main_stream is not None:
            fence = torch.cuda.Event()
            fence.record(self._main_stream)
        has_claim = self._detached is not None and self._detached.has_claim(ticket)
        if ticket.detached_cold != has_claim:
            raise ValueError(
                "Native D detached ticket lacks its exact local restore claim"
            )
        if ticket.detached_cold:
            body = self._receive_detached(ticket, fence)
        else:
            try:
                plan, groups = self._plan(ticket)
            except ValueError as error:
                if ticket.mla_rank_digests is None or not self.scope.forward_abort:
                    raise
                body = self._abort_unopened_preflight(ticket, error)
            else:
                body = self._receive(ticket, plan, groups, fence)
        future = asyncio.run_coroutine_threadsafe(body, self.worker.receiver_loop)
        self._imports[ticket.request_id] = _Import(
            ticket, future, detached=ticket.detached_cold
        )

    def retire_finished(self, finished) -> None:
        self._finished.update(self._imports.keys() & finished)
        for request_id, active in tuple(self._imports.items()):
            if not active.future.done():
                continue
            if (
                not active.reported_error
                and (error := active.future.exception()) is not None
            ):
                logger.warning(
                    "Native GLM cold import failed (%s)", type(error).__name__
                )
                active.reported_error = True
            if request_id in self._finished and request_id not in self._current_tickets:
                if active.started:
                    self.runner.drop_imported_frontier(request_id)
                if active.detached:
                    assert self._detached is not None
                    self._detached.retire_fenced(active.ticket)
                self._imports.pop(request_id)
                self._finished.discard(request_id)

    def close(self) -> None:
        if any(not active.future.done() for active in self._imports.values()):
            raise RuntimeError("Native D cold import still owns in-flight writes")
        self._receiver.close()
