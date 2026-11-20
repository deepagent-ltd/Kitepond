# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep a verified FSL2 rank restore until native D has its import ticket."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from kitepond.core.contracts import Wait
from kitepond.models.glm5next.host_cold_extent import ColdExtentLease
from kitepond.models.glm5next.import_recipe import part_identity
from kitepond.models.glm5next.layout import CacheRole

from .decode_receive import DecodeReceiveMailbox, DecodeReceiveTicket
from .glm_cold_catalog import VerifiedColdCheckpoint
from .glm_decode_restore import apply_verified_native_host_import
from .host_authority import HostRankSettlement, NativeHostGrant
from .native_glm_cold_provider import NativeGlmRankMemory

if TYPE_CHECKING:
    import torch

    from .glm_native_views import GlmNativeStateViews


class DetachedGlmColdRank:
    """Restore RAM before device allocation, then import under a native ticket."""

    def __init__(
        self,
        memory: NativeGlmRankMemory,
        views: GlmNativeStateViews,
        grant: NativeHostGrant,
        verified: VerifiedColdCheckpoint,
        *,
        require_native: Callable[[], None],
    ) -> None:
        descriptor = memory.region.descriptor
        if (
            type(grant) is not NativeHostGrant
            or type(verified) is not VerifiedColdCheckpoint
            or (grant.engine_id, grant.boot_id)
            != (descriptor.authority_engine_id, descriptor.authority_boot_id)
            or descriptor != views.descriptor
            or memory.rank != views.rank
            or grant.boot_id != verified.target_boot_id
            or grant.prefix_tokens != 0
            or grant.readonly_pages
            or grant.copy_tail is not None
            or grant.reservation_tokens
            <= verified.checkpoint.layout.target_valid_tokens
            or not grant.writable_pages
            or not callable(require_native)
        ):
            raise ValueError("Detached restore must own the current D writable pages")
        self.memory, self.views, self.grant, self.verified = (
            memory,
            views,
            grant,
            verified,
        )
        self._require_native = require_native
        self._lease: ColdExtentLease | None = None
        self._data = None
        self._stage = None
        self._authority = None
        self._started_import = False

    def _adopt(self, lease: ColdExtentLease) -> None:
        if self._lease is not None:
            raise ValueError("Detached cold auxiliary bytes are already adopted")
        self._lease = lease

    def prepare(self):
        """Return the ordinary native FSL2 write authority over this rank only."""
        if self._data is not None:
            raise ValueError("Detached rank admission may prepare only once")
        checkpoint = self.verified.checkpoint
        layers = self.views.detached_host_layers(self.grant.writable_pages)
        aux = self.memory.reserve_aux(
            checkpoint,
            "RESTORE",
            evidence=self.verified,
            require_native=self._require_native,
            adopt_restore=self._adopt,
        )
        data = self.memory.bind_detached_restore(checkpoint, layers, aux)
        if isinstance(data, Wait):
            return data
        stage = self.memory.reserve_scratch(checkpoint)
        if isinstance(stage, Wait):
            data.settle("MISS")
            return stage
        try:
            authority = self.memory.make_authority(
                checkpoint,
                data,
                stage,
                self.verified.checksums(self.memory.rank),
            )
        except BaseException:
            try:
                stage.release()
            finally:
                data.settle("MISS")
            raise
        self._data, self._stage, self._authority = data, stage, authority
        return authority

    def settle_cold(self, state: str) -> None:
        if self._data is None or self._stage is None or self._authority is None:
            raise ValueError("Detached rank has no unsettled cold write")
        self._authority.require_settled_restore(verified=(state == "VERIFIED"))
        self._stage.release()
        self._stage = None
        self._data.settle("MISS" if state == "CANCELLED" else state)

    def host_ready(self) -> HostRankSettlement:
        """Attest the verified, still-owned host span before scheduler attachment."""
        if self._data is None or self._lease is None or self._stage is not None:
            raise ValueError("Detached rank bytes have not finished cold readback")
        self._data.require_live()
        return HostRankSettlement(self.grant.sha256, self.memory.rank, "ready", 0, 0)

    def verify_shared_readback(self) -> HostRankSettlement:
        if self._authority is None or self._stage is None:
            raise ValueError("Detached rank has no outstanding shared restore")
        self._authority.verify_shared_restore()
        self.settle_cold("VERIFIED")
        return self.host_ready()

    def import_ready(
        self,
        mailbox: DecodeReceiveMailbox,
        ticket: DecodeReceiveTicket,
        slab: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
        writes_before_import: torch.cuda.Event | None = None,
    ) -> None:
        """Publish loaded only through the real D import and after rank readback."""
        data, grant, checkpoint = self._data, self.grant, self.verified.checkpoint
        if (
            data is None
            or self._lease is None
            or self._stage is not None
            or self._started_import
            or type(ticket) is not DecodeReceiveTicket
            or (ticket.request_id, ticket.transfer_id, ticket.epoch)
            != (grant.request_id, grant.transfer_id, grant.epoch)
            or (ticket.engine_id, ticket.boot_id) != (grant.engine_id, grant.boot_id)
            or ticket.grant_sha256 != grant.sha256
            or ticket.imported_tokens != checkpoint.layout.target_valid_tokens
            or mailbox.ticket_for(ticket.request_id, ticket.transfer_id, ticket.epoch)
            != ticket
        ):
            raise ValueError("Detached cold import needs its verified native D ticket")
        data.require_live()
        # FSL2 already holds the accepted source recurrent state. The new
        # native request's first non-speculative step reads its own base slot.
        layout = replace(
            checkpoint.layout,
            request_id=ticket.request_id,
            epoch=ticket.epoch,
            committed_kda_slot=0,
        )
        cache, groups = self.views.for_import(layout, ticket.block_ids)
        self.views.require_owned_layout(cache, groups)
        for binding in data.bindings:
            part = binding.part
            if part_identity(part.role).cache_role is CacheRole.MLA and binding != (
                self.memory._mla_binding(layout, cache.layers[part.role], part)
            ):
                raise ValueError("Native D ticket changed the installed MLA host pages")
        self._started_import = True
        apply_verified_native_host_import(
            mailbox,
            ticket,
            cache,
            groups,
            self.memory.region,
            data.bindings,
            slab,
            source_region_sha256=self.memory.region_sha256,
            stream=stream,
            cow_event=writes_before_import,
            seed_imported_tail=self.views.seed_imported_tail,
        )
        self.memory.release_restored_aux(self._lease)
        self._lease = None

    def abandon_before_import(self) -> None:
        """Release a verified attempt whose native ticket has not started writing."""
        if self._stage is not None or self._started_import or self._lease is None:
            raise ValueError("Detached rank has no unconsumed verified auxiliary lease")
        self.memory.release_restored_aux(self._lease)
        self._lease = None

    def release_after_abort(
        self, mailbox: DecodeReceiveMailbox, ticket: DecodeReceiveTicket
    ) -> None:
        proof = mailbox.settled_proof(ticket)
        if (
            self._lease is None
            or ticket.grant_sha256 != self.grant.sha256
            or proof is None
            or proof.outcome != "abort_settled"
        ):
            raise ValueError("Detached device import has no settled native abort")
        self.memory.release_restored_aux(self._lease)
        self._lease = None
