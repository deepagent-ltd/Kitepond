# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offer GLM P bytes only from the post-fence native source lease."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from kitepond.core.readiness import ReadyReceipt
from kitepond.integrations.mooncake.live_receive import (
    NativeRankWindowWriter,
    RankWindowAttempt,
    mooncake_receive_pool,
)
from kitepond.integrations.mooncake.window import manifest_digest
from kitepond.models.glm5next.cold_restore import ColdRestoreLayout
from kitepond.models.glm5next.import_recipe import part_identity
from kitepond.models.glm5next.layout import CacheRole, Glm5NextPDLayout
from kitepond.models.glm5next.live_windows import GlmRankWindowPlan
from kitepond.models.glm5next.shared_mla import (
    SharedMlaRankPages,
    VerifiedSharedMlaElision,
)

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorWorker,
    SendBlockMeta,
)

from .glm_native_views import GlmNativeStateViews
from .prefill_fence import PrefillWrite

_MAX_FENCED = 128


@dataclass(frozen=True)
class GlmPrefillExport:
    """Exact native final frontier and optional detached KDA export."""

    layout: Glm5NextPDLayout
    local_cached_tokens: int
    checkpoint_layout: Glm5NextPDLayout | None = None
    checkpoint_blocks: tuple[tuple[int, ...], ...] | None = None
    native_kda_blocks: tuple[tuple[str, int], ...] | None = None

    def __post_init__(self) -> None:
        if (self.checkpoint_layout is None) != (self.checkpoint_blocks is None):
            raise ValueError("GLM prefill checkpoint requires native block tables")


class GlmPrefillReadySources:
    """Bridge exact local READY records to stock retained sender blocks."""

    def __init__(
        self,
        worker: MooncakeConnectorWorker,
        decode_host: str,
        *,
        mla_preflight: bool = False,
    ) -> None:
        self.worker = worker
        self.decode_host = decode_host
        self._mla_preflight = mla_preflight
        self.rank = worker.tp_rank
        self.pool = mooncake_receive_pool(worker.engine, 1)
        self._lock = threading.Lock()
        self._views: GlmNativeStateViews | None = None
        self._export: (
            Callable[[PrefillWrite, ReadyReceipt], GlmPrefillExport] | None
        ) = None
        self._fenced: dict[str, tuple[ReadyReceipt, GlmPrefillExport]] = {}
        self._writers: dict[str, NativeRankWindowWriter] = {}

    def bind_export(
        self,
        views: GlmNativeStateViews,
        export: Callable[[PrefillWrite, ReadyReceipt], GlmPrefillExport],
    ) -> None:
        if (
            self._views is not None
            or type(views) is not GlmNativeStateViews
            or views.rank != self.rank
            or not callable(export)
        ):
            raise ValueError("GLM prefill requires its initialized native rank views")
        self._views, self._export = views, export

    def record_fenced(
        self, records: tuple[tuple[PrefillWrite, ReadyReceipt], ...]
    ) -> None:
        """Called by the existing save fence after native target and draft events."""
        views, export = self._views, self._export
        if views is None or export is None:
            raise ValueError("GLM prefill has no native export owner")
        accepted = {}
        for write, receipt in records:
            if (
                write.ticket.attempt != write.grant.attempt
                or receipt.attempt != write.grant.attempt
                or receipt.rank != views.rank
                or write.grant.region != views.descriptor
                or receipt.region_sha256 != views.descriptor.fingerprint()
                or receipt.target_valid_tokens != write.grant.tokens
                or receipt.draft_valid_tokens != write.grant.tokens
            ):
                raise ValueError(
                    "GLM prefill source differs from its native READY grant"
                )
            selected = export(write, receipt)
            if type(selected) is not GlmPrefillExport:
                raise ValueError(
                    "GLM prefill owner did not supply an exact export plan"
                )
            layout = selected.layout
            layout.require_decode_ready()
            if (
                layout.request_id != receipt.attempt.request_id
                or layout.epoch != receipt.attempt.epoch
                or layout.target_valid_tokens != receipt.target_valid_tokens
                or type(selected.local_cached_tokens) is not int
                or not 0 <= selected.local_cached_tokens < layout.target_valid_tokens
                or selected.local_cached_tokens % 1024
                or receipt.attempt.transfer_id in accepted
            ):
                raise ValueError(
                    "GLM source plan differs from its accepted native boundary"
                )
            accepted[receipt.attempt.transfer_id] = (receipt, selected)
        with self._lock:
            if (
                self._fenced.keys() & accepted.keys()
                or len(self._fenced) + len(accepted) > _MAX_FENCED
            ):
                raise ValueError("GLM prefill writer registry is duplicated or full")
            self._fenced.update(accepted)

    def _require_sender(self) -> None:
        if asyncio.get_running_loop() is not self.worker.sender_loop:
            raise RuntimeError("GLM source ownership belongs to the native sender loop")

    def lookup(self, attempt: RankWindowAttempt) -> NativeRankWindowWriter | None:
        self._require_sender()
        current = self._writers.get(attempt.transfer_id)
        return current if current is not None and current.attempt == attempt else None

    def _fenced_source(
        self, attempt: RankWindowAttempt
    ) -> tuple[ReadyReceipt, GlmPrefillExport] | None:
        with self._lock:
            proof = self._fenced.get(attempt.transfer_id)
        if proof is None:
            return None
        receipt, _ = proof
        if (
            receipt.attempt.request_id,
            receipt.attempt.epoch,
            receipt.attempt.lease_sha256,
        ) != (attempt.request_id, attempt.epoch, attempt.grant_sha256):
            return None
        return proof

    async def cancel_unopened(
        self, attempt: RankWindowAttempt, destination: tuple[str, int, str]
    ) -> bool:
        """Remove an admitted source before any writer could submit DMA."""
        self._require_sender()
        key = attempt.transfer_id
        if destination[0] != self.decode_host or attempt.rank != self.rank:
            return False
        proof = self._fenced_source(attempt)
        native = self.worker.reqs_need_send
        send = native.get(key)
        if proof is None or type(send) is not SendBlockMeta:
            return False
        await asyncio.wait_for(
            send.ready.wait(), float(envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT)
        )
        if (
            native.get(key) is not send
            or key in self._writers
            or send.p_req_id != attempt.request_id
            or send.sending
            or send.sent
        ):
            return False
        # The named handler serialized admission and cancellation for this
        # transfer. Outside the handler, native expiration shares this loop.
        with self._lock:
            if self._fenced.get(key) is not proof:
                return False
            del self._fenced[key]
        native.pop(key)
        self.worker.finished_sending_reqs.add(send.p_req_id)
        return True

    async def _retained_source(self, attempt: RankWindowAttempt):
        self._require_sender()
        if attempt.rank != self.rank:
            return None
        proof = self._fenced_source(attempt)
        native = self.worker.reqs_need_send
        send = native.get(attempt.transfer_id)
        if proof is None or type(send) is not SendBlockMeta:
            return None
        await asyncio.wait_for(
            send.ready.wait(), float(envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT)
        )
        if (
            native.get(attempt.transfer_id) is not send
            or attempt.transfer_id in self._writers
            or send.p_req_id != attempt.request_id
            or not any(send.local_block_ids)
            or send.sending
            or send.sent
            or time.perf_counter() >= send.expire_time
        ):
            return None
        return (*proof, send)

    def _plan(
        self, send: SendBlockMeta, selected: GlmPrefillExport
    ) -> GlmRankWindowPlan:
        views = self._views
        assert views is not None
        cache = self.worker.kv_cache_config
        group_ids = cache.transfer_group_ids
        compact = tuple(tuple(group) for group in send.local_block_ids)
        if len(compact) != len(group_ids):
            raise ValueError("GLM source lost the native Mooncake transfer group table")
        by_id = dict(zip(group_ids, compact, strict=True))
        # Mooncake excludes native local hot/resident replicas. GLM views
        # address the original group IDs, so those slots are restored as empty.
        native_blocks = tuple(
            by_id.get(group_id, ()) for group_id in range(len(cache.kv_cache_groups))
        )
        final, groups = views.for_import(selected.layout, native_blocks)
        if selected.native_kda_blocks is not None:
            final = views.verify_k0_prefill_export(final, selected.native_kda_blocks)
        checkpoint = None
        if selected.checkpoint_layout is not None:
            assert selected.checkpoint_blocks is not None
            if len(selected.checkpoint_blocks) != len(native_blocks):
                raise ValueError("GLM detached source is missing native groups")
            checkpoint = ColdRestoreLayout(
                selected.checkpoint_layout,
                self.rank,
                [
                    replace(layer, block_ids=selected.checkpoint_blocks[groups[name]])
                    for name, layer in final.layers.items()
                    if layer.identity.cache_role
                    in (CacheRole.KDA_CONV, CacheRole.KDA_RECURRENT)
                ],
            )
        return GlmRankWindowPlan(
            final, selected.local_cached_tokens, checkpoint=checkpoint
        )

    async def prepare_shared_mla(self, attempt: RankWindowAttempt) -> str | None:
        source = await self._retained_source(attempt)
        if source is None:
            return None
        receipt, selected, send = source
        if selected.local_cached_tokens or selected.checkpoint_layout is not None:
            return None
        plan = self._plan(send, selected)
        assert self._views is not None
        return SharedMlaRankPages.observe_enrolled(
            plan.final, self._views.descriptor, receipt.attempt
        ).sha256

    async def open_writer(
        self, attempt: RankWindowAttempt, destination: tuple[str, int, str]
    ) -> NativeRankWindowWriter | None:
        self._require_sender()
        if destination[0] != self.decode_host or attempt.rank != self.rank:
            return None
        known = self.lookup(attempt)
        if known is not None:
            return known
        source = await self._retained_source(attempt)
        if source is None:
            return self.lookup(attempt)
        receipt, selected, send = source
        native = self.worker.reqs_need_send
        plan = self._plan(send, selected)
        if attempt.mla_rank_digests is not None:
            if not self._mla_preflight:
                return None
            assert self._views is not None
            plan = plan.for_forward_shared_mla(
                VerifiedSharedMlaElision(receipt.attempt, attempt.mla_rank_digests),
                self._views.descriptor,
                receipt.attempt,
            )
        manifest = plan.manifest()
        if manifest_digest(manifest) != attempt.manifest_sha256:
            return None

        def release() -> None:
            if native.get(attempt.transfer_id) is not send or send.sending != 1:
                raise ValueError("GLM source lost its retained native lease")
            send.sending -= 1
            native.pop(attempt.transfer_id)
            self.worker.finished_sending_reqs.add(send.p_req_id)
            self._writers.pop(attempt.transfer_id)
            with self._lock:
                self._fenced.pop(attempt.transfer_id)

        def pack(window, payload) -> int:
            if (
                native.get(attempt.transfer_id) is not send
                or send.sending != 1
                or time.perf_counter() >= send.expire_time
            ):
                raise TimeoutError("Native P Mooncake source lease expired")
            copied = plan.pack(window, payload)
            if selected.native_kda_blocks is not None:
                _canonicalize_window_tail(
                    window, payload, selected.layout.target_valid_tokens
                )
            return copied

        writer = NativeRankWindowWriter(
            attempt,
            manifest,
            self.worker.engine,
            self.pool,
            destination=destination,
            pack=pack,
            release_source=release,
            timeout=float(envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT),
        )
        send.sending += 1
        self._writers[attempt.transfer_id] = writer
        return writer

    async def expire(self) -> None:
        """Retire abandoned sources only after Mooncake has finished reading."""
        self._require_sender()
        native = self.worker.reqs_need_send
        now = time.perf_counter()
        for key, writer in tuple(self._writers.items()):
            send = native.get(key)
            if send is None:
                raise ValueError("GLM source lost its native lease before retirement")
            if send.expire_time <= now:
                await writer.abandon()
        threshold = time.monotonic_ns() - int(
            envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT * 1e9
        )
        with self._lock:
            for key in self._fenced.keys() - native.keys():
                if self._fenced[key][0].completed_monotonic_ns <= threshold:
                    del self._fenced[key]

    def has_fenced_sources(self) -> bool:
        with self._lock:
            return bool(self._fenced)

    def close(self) -> None:
        if self._writers:
            raise RuntimeError("GLM prefill still owns live Mooncake sources")
        self.pool.close()


def _canonicalize_window_tail(window, payload, boundary: int) -> None:
    # Only the unfinished compressed pool is reusable. Registered scratch can
    # be cleared without writing back to the held physical model ring.
    active = {position % 8 for position in range(boundary - boundary % 4, boundary)}
    offset = 0
    for part in window.entries:
        end = offset + part["nbytes"]
        if part_identity(part["role"]).cache_role is CacheRole.KPOOL_TAIL:
            rows = payload[offset:end].view(8, -1)
            for row in range(8):
                if row not in active:
                    rows[row].zero_()
        offset = end
