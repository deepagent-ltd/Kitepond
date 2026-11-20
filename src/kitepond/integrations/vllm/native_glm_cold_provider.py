# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind accepted GLM rank views to existing cold-storage RAM leases."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from time import monotonic_ns
from typing import TYPE_CHECKING, Literal, Protocol

import torch
from kitepond.backends.fsl2 import BundleIdentity, PartSpec
from kitepond.core.contracts import Wait
from kitepond.models.glm5next.cold_authority import NativeRegionColdAuthority
from kitepond.models.glm5next.cold_lifecycle import GlmColdCheckpoint, GlmColdRamGrant
from kitepond.models.glm5next.cold_restore import (
    ColdRestoreLayout,
    RestoreLayer,
    native_rank_rows,
)
from kitepond.models.glm5next.host_cold_extent import (
    ColdExtentLease,
    NativeColdExtent,
    Zone,
)
from kitepond.models.glm5next.import_recipe import (
    ByteSpan,
    ColdBinding,
    part_identity,
    part_role,
)
from kitepond.models.glm5next.layout import CacheRole, Glm5NextPDLayout
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion

from .glm_cold_catalog import VerifiedColdCheckpoint
from .glm_cold_runtime import GlmColdDirectoryImport
from .glm_host_backing import describe_glm_cold_backing, describe_glm_host_backing
from .glm_terminal_manifest import FencedTerminalSource
from .native_glm_cold_directory import RetainedColdRam

if TYPE_CHECKING:
    from .glm_terminal_worker import GlmTerminalWorker

Mode = Literal["STORE", "RESTORE"]


@dataclass(frozen=True)
class NativeAuxReservation:
    """Pre-enrolled aux spans and their native lease.

    In RESTORE mode the verifier must cover the reserved MLA pages as well.
    """

    identity: BundleIdentity
    region_sha256: str
    mode: Mode
    bindings: tuple[ColdBinding, ...]
    require_live: Callable[[], None]
    settle: Callable[[str], None]
    extent_lease: ColdExtentLease | None = None


@dataclass(frozen=True)
class NativeColdRankBytes:
    identity: BundleIdentity
    region_sha256: str
    mode: Mode
    bindings: tuple[ColdBinding, ...]
    require_live: Callable[[], None]
    settle: Callable[[str], None]


@dataclass(frozen=True)
class NativeColdScratch:
    identity: BundleIdentity
    region_sha256: str
    span: ByteSpan
    require_live: Callable[[], None]
    release: Callable[[], None]
    extent_lease: ColdExtentLease | None = None


class NativeGlmRankMemory:
    """Recognize native MLA views and auxiliary subranges of one rank region."""

    def __init__(
        self,
        region: NativeHiSparseMemfdRegion,
        registered_mla: torch.Tensor,
        *,
        mla_bytes: int,
    ) -> None:
        if (
            type(region) is not NativeHiSparseMemfdRegion
            or region.mapping.closed
            or len(region.mapping) != region.descriptor.nbytes
            or type(mla_bytes) is not int
            or not 0 < mla_bytes <= region.descriptor.nbytes
            or not isinstance(registered_mla, torch.Tensor)
            or registered_mla.device.type != "cpu"
            or registered_mla.element_size() != 1
            or registered_mla.storage_offset() != 0
            or not registered_mla.is_contiguous()
            or registered_mla.numel() < mla_bytes
            or describe_glm_host_backing(registered_mla) != region.descriptor
        ):
            raise ValueError("Cold rank requires its registered native MLA extent")
        self.region = region
        self.rank = region.descriptor.source_rank
        self.region_sha256 = region.descriptor.fingerprint()
        self.mla_bytes = mla_bytes
        self._registered = registered_mla
        self._extent: NativeColdExtent | None = None
        self._restored: dict[int, ColdExtentLease] = {}

    @classmethod
    def from_registered_cold_backing(
        cls, backing: torch.Tensor
    ) -> NativeGlmRankMemory | Wait:
        extent = describe_glm_cold_backing(backing)
        if extent is None:
            return Wait("NATIVE_COLD_EXTENT_NOT_ENROLLED")
        if backing.untyped_storage().nbytes() != extent.layout.mla_padded_bytes:
            raise ValueError("Native cold enrollment changed its MLA tensor extent")
        memory = cls(extent.region, backing, mla_bytes=extent.layout.mla_bytes)
        memory._extent = extent
        return memory

    def reserve_aux(
        self,
        checkpoint: GlmColdCheckpoint,
        mode: Mode,
        *,
        evidence: GlmColdDirectoryImport | VerifiedColdCheckpoint,
        require_native: Callable[[], None],
        adopt_restore: Callable[[ColdExtentLease], None] | None = None,
    ) -> NativeAuxReservation | Wait:
        """Reserve rank aux bytes while the exact source or destination is live.

        A successful restore hands its lease to the native destination owner,
        which releases it after successful import or destination rollback.
        """
        extent = self._extent
        if extent is None:
            return Wait("NATIVE_COLD_EXTENT_NOT_ENROLLED")
        if (
            mode not in ("STORE", "RESTORE")
            or not callable(require_native)
            or (mode == "RESTORE") != callable(adopt_restore)
        ):
            raise ValueError("Native auxiliary range needs a live rank owner")
        descriptor = self.region.descriptor
        if mode == "STORE":
            if type(evidence) is not GlmColdDirectoryImport:
                raise ValueError("Native cold source needs its accepted terminal")
            identity = evidence.lease.identity
            if (
                evidence.checkpoint != checkpoint
                or identity.source_engine_id != descriptor.authority_engine_id
                or identity.source_boot_id != descriptor.authority_boot_id
                or evidence.manifest.ranks[self.rank].checkpoint.authority_sha256
                != self.region_sha256
            ):
                raise ValueError("Native cold source belongs to a different authority")
        elif (
            type(evidence) is not VerifiedColdCheckpoint
            or evidence.checkpoint != checkpoint
            or evidence.target_boot_id != descriptor.authority_boot_id
        ):
            raise ValueError("Native cold destination changed its verified boot")
        parts = tuple(
            part
            for part in checkpoint.plans[self.rank].parts
            if part_identity(part.role).cache_role is not CacheRole.MLA
        )
        require_native()
        lease = extent.reserve("aux", sum(part.nbytes for part in parts))
        if isinstance(lease, Wait):
            return lease
        try:
            offset = lease.offset
            bindings = []
            for part in parts:
                span = ByteSpan(self.rank, lease.allocation_id, offset, part.nbytes)
                bindings.append(ColdBinding(part, (span,)))
                offset += part.nbytes
        except BaseException:
            extent.release(lease)
            raise

        def live() -> None:
            extent.verify(lease)
            require_native()

        settled = False

        def settle(state: str) -> None:
            nonlocal settled
            if settled:
                raise ValueError("Native auxiliary reservation was already settled")
            if mode == "RESTORE" and state == "VERIFIED":
                try:
                    live()
                    assert adopt_restore is not None
                    adopt_restore(lease)
                except BaseException:
                    extent.release(lease)
                    raise
                else:
                    self._restored[lease.allocation_id] = lease
                finally:
                    settled = True
            elif state in ("FAILED", "MISS", "RELEASED"):
                extent.release(lease)
                settled = True
            else:
                raise ValueError("Unknown native auxiliary settlement")

        return NativeAuxReservation(
            checkpoint.identity,
            self.region_sha256,
            mode,
            tuple(bindings),
            live,
            settle,
            lease,
        )

    def release_restored_aux(self, lease: ColdExtentLease) -> None:
        if (
            self._extent is None
            or type(lease) is not ColdExtentLease
            or self._restored.get(lease.allocation_id) is not lease
        ):
            raise ValueError("Lease is not an adopted native auxiliary reservation")
        self._extent.release(lease)
        del self._restored[lease.allocation_id]

    def reserve_scratch(
        self, checkpoint: GlmColdCheckpoint
    ) -> NativeColdScratch | Wait:
        extent = self._extent
        if extent is None:
            return Wait("NATIVE_COLD_EXTENT_NOT_ENROLLED")
        lease = extent.reserve(
            "scratch", max(part.nbytes for part in checkpoint.plans[self.rank].parts)
        )
        if isinstance(lease, Wait):
            return lease
        return NativeColdScratch(
            checkpoint.identity,
            self.region_sha256,
            ByteSpan(self.rank, lease.allocation_id, lease.offset, lease.nbytes),
            lambda: extent.verify(lease),
            lambda: extent.release(lease),
            lease,
        )

    def _mla_binding(
        self, layout: Glm5NextPDLayout, layer: RestoreLayer, part: PartSpec
    ) -> ColdBinding:
        tensor = layer.tensor
        width = part.shape[1]
        first, end = part.index, part.index + part.shape[0]
        rows = tensor.shape[2]
        last = native_rank_rows(
            layout.valid_tokens(layer.identity), CacheRole.MLA, self.rank
        )
        if (
            tensor.device.type != "cpu"
            or tensor.untyped_storage().data_ptr()
            != self._registered.untyped_storage().data_ptr()
            or tensor.shape[-1] != width
            or tensor.stride(-1) != 1
            or tensor.stride(-2) != width
            or first < 0
            or end > last
        ):
            raise ValueError("Cold MLA does not address its registered native rows")
        spans = []
        while first < end:
            page, row = divmod(first, rows)
            block = layer.block_ids[page]
            count = min(rows - row, end - first)
            offset = tensor.storage_offset() + block * tensor.stride(0) + row * width
            if not (
                0 < block < tensor.shape[0]
                and 0 <= offset <= self.mla_bytes - count * width
            ):
                raise ValueError("Cold MLA span escapes the native host pool")
            spans.append(ByteSpan(self.rank, block, offset, count * width))
            first += count
        return ColdBinding(part, tuple(spans))

    def bind_detached_restore(
        self,
        checkpoint: GlmColdCheckpoint,
        layers: tuple[RestoreLayer, ...],
        aux: NativeAuxReservation | Wait,
    ) -> NativeColdRankBytes | Wait:
        """Bind owned host pages before this request receives device state slots."""
        if isinstance(aux, Wait):
            return aux
        try:
            if (
                self._extent is None
                or type(aux) is not NativeAuxReservation
                or aux.identity != checkpoint.identity
                or aux.region_sha256 != self.region_sha256
                or aux.mode != "RESTORE"
                or type(layers) is not tuple
                or any(
                    type(layer) is not RestoreLayer
                    or layer.identity.cache_role is not CacheRole.MLA
                    for layer in layers
                )
            ):
                raise ValueError(
                    "Detached cold restore needs exact native rank ownership"
                )
            local = {part_role(layer.identity): layer for layer in layers}
            auxiliary = {binding.part: binding for binding in aux.bindings}
            plan = checkpoint.plans[self.rank]
            expected_aux = tuple(
                part
                for part in plan.parts
                if part_identity(part.role).cache_role is not CacheRole.MLA
            )
            expected_mla = {part.role for part in plan.parts if part not in auxiliary}
            if (
                len(local) != len(layers)
                or local.keys() != expected_mla
                or tuple(auxiliary) != expected_aux
                or len(auxiliary) != len(aux.bindings)
            ):
                raise ValueError("Detached cold restore changed its accepted parts")
            for binding in aux.bindings:
                for span in binding.spans:
                    self._aux_range(span)
                    self._require_enrolled_span(span, aux.extent_lease, "aux")
            assert aux.extent_lease is not None
            self._extent.verify(aux.extent_lease)
            aux.require_live()
            return NativeColdRankBytes(
                checkpoint.identity,
                self.region_sha256,
                "RESTORE",
                tuple(
                    auxiliary[part]
                    if part in auxiliary
                    else self._mla_binding(checkpoint.layout, local[part.role], part)
                    for part in plan.parts
                ),
                aux.require_live,
                aux.settle,
            )
        except Exception:
            aux.settle("MISS")
            raise

    def _aux_range(self, span: ByteSpan) -> None:
        if (
            type(span) is not ByteSpan
            or span.rank != self.rank
            or not self.mla_bytes
            <= span.offset
            <= self.region.descriptor.nbytes - span.nbytes
        ):
            raise ValueError("Cold auxiliary or scratch span escapes its native extent")

    def _require_enrolled_span(
        self, span: ByteSpan, lease: ColdExtentLease | None, zone: Zone
    ) -> None:
        if self._extent is None:
            return
        if (
            type(lease) is not ColdExtentLease
            or lease.zone != zone
            or span.block_id != lease.allocation_id
            or not lease.offset
            <= span.offset
            <= lease.offset + lease.nbytes - span.nbytes
        ):
            raise ValueError("Cold byte span escapes its exact native reservation")

    def _bind(
        self,
        checkpoint: GlmColdCheckpoint,
        cache: ColdRestoreLayout,
        aux: NativeAuxReservation,
        mode: Mode,
    ) -> NativeColdRankBytes:
        plan = checkpoint.plans[self.rank]
        if (
            type(cache) is not ColdRestoreLayout
            or cache.rank != self.rank
            or cache.layout != checkpoint.layout
            or type(aux) is not NativeAuxReservation
            or aux.identity != checkpoint.identity
            or aux.region_sha256 != self.region_sha256
            or aux.mode != mode
            or cache.manifest() != plan.parts
        ):
            raise ValueError("Cold rank changed its accepted native layout or scope")
        auxiliary = {item.part: item for item in aux.bindings}
        expected = tuple(
            part
            for part in plan.parts
            if part_identity(part.role).cache_role is not CacheRole.MLA
        )
        if tuple(auxiliary) != expected or len(auxiliary) != len(aux.bindings):
            raise ValueError("Cold rank lacks its complete native auxiliary state")
        spans = sorted(
            (span for binding in aux.bindings for span in binding.spans),
            key=lambda span: span.offset,
        )
        for span in spans:
            self._aux_range(span)
            self._require_enrolled_span(span, aux.extent_lease, "aux")
        if self._extent is not None:
            assert aux.extent_lease is not None
            self._extent.verify(aux.extent_lease)
        if any(
            left.offset + left.nbytes > right.offset
            for left, right in zip(spans, spans[1:])
        ):
            raise ValueError("Native auxiliary reservations overlap")
        aux.require_live()
        return NativeColdRankBytes(
            checkpoint.identity,
            self.region_sha256,
            mode,
            tuple(
                auxiliary[part]
                if part in auxiliary
                else self._mla_binding(cache.layout, cache.layers[part.role], part)
                for part in plan.parts
            ),
            aux.require_live,
            aux.settle,
        )

    def prepare_terminal(
        self,
        entry: GlmColdDirectoryImport,
        source: FencedTerminalSource,
        aux: NativeAuxReservation | Wait,
        *,
        require_source: Callable[[], None],
    ) -> NativeColdRankBytes | Wait:
        """Worker-side capture while the exact signed terminal read remains open."""
        if isinstance(aux, Wait):
            return aux
        try:
            entry.require_exact_checkpoint()
            signed = entry.manifest.ranks[self.rank].checkpoint
            identity = entry.lease.identity
            if (
                type(source) is not FencedTerminalSource
                or signed.rank != self.rank
                or (
                    identity.source_engine_id,
                    identity.source_boot_id,
                )
                != (
                    self.region.descriptor.authority_engine_id,
                    self.region.descriptor.authority_boot_id,
                )
                or signed.authority_sha256 != self.region_sha256
                or source.authority_sha256 != self.region_sha256
                or source.representation_sha256
                != signed.plan.representation_fingerprint
                or source.tail_positions != signed.tail_positions
            ):
                raise ValueError("Cold rank source differs from its signed terminal")
            require_source()
            result = self._bind(entry.checkpoint, source.layout, aux, "STORE")
            for binding in aux.bindings:
                require_source()
                aux.require_live()
                payload = source.read_chunk(binding.part)
                if (
                    payload.device.type != "cpu"
                    or payload.dtype != torch.uint8
                    or payload.ndim != 1
                    or not payload.is_contiguous()
                    or payload.numel() != binding.part.nbytes
                ):
                    raise ValueError("Canonical GLM cold bytes changed shape or dtype")
                with memoryview(payload.numpy()) as content:
                    cursor = 0
                    for span in binding.spans:
                        end = cursor + span.nbytes
                        self.region.mapping[span.offset : span.offset + span.nbytes] = (
                            content[cursor:end]
                        )
                        cursor = end
            require_source()
            aux.require_live()
            return result
        except Exception:
            aux.settle("FAILED")
            raise

    def prepare_from_worker(
        self,
        worker: GlmTerminalWorker,
        entry: GlmColdDirectoryImport,
        aux: NativeAuxReservation | Wait,
    ) -> NativeColdRankBytes | Wait:
        """Use the worker's process-local, permit-checked native source."""
        if isinstance(aux, Wait):
            return aux
        access = getattr(worker, "terminal_window_source", None)
        if not callable(access):
            aux.settle("FAILED")
            return Wait("NATIVE_COLD_WORKER_SOURCE_WAIT")
        try:
            views = worker.kitepond_native_state_views()
            cache = worker.model_runner.kv_cache_config
            if (
                views.rank != self.rank
                or views.descriptor != self.region.descriptor
                or self.mla_bytes
                != cache.hisparse_host_num_blocks * cache.hisparse_host_block_stride
            ):
                raise ValueError("Cold worker has a different native rank backing")
            args = (entry.lease, entry.permit, entry.manifest)
            source = access(*args)
            request_id = entry.lease.identity.request_id
            frame = worker._kitepond_terminal_frames[request_id]
            deadline = entry.lease.identity.deadline_monotonic_ns
        except Exception:
            aux.settle("FAILED")
            raise

        def require_source():
            # The immutable permit and manifest were checked once. Long
            # checkpoints need only recheck the live frame and deadline here.
            if (
                worker._kitepond_terminal_frames.get(request_id) is not frame
                or frame.source is not source
                or monotonic_ns() >= deadline
            ):
                raise ValueError("Cold worker changed its authorized terminal source")

        return self.prepare_terminal(entry, source, aux, require_source=require_source)

    def bind_restore(
        self,
        checkpoint: GlmColdCheckpoint,
        destination: ColdRestoreLayout,
        aux: NativeAuxReservation | Wait,
    ) -> NativeColdRankBytes | Wait:
        if isinstance(aux, Wait):
            return aux
        try:
            return self._bind(checkpoint, destination, aux, "RESTORE")
        except Exception:
            aux.settle("MISS")
            raise

    def make_authority(
        self,
        checkpoint: GlmColdCheckpoint,
        data: NativeColdRankBytes,
        stage: NativeColdScratch,
        checksums: dict[PartSpec, str] | None,
    ) -> NativeRegionColdAuthority:
        if (
            type(data) is not NativeColdRankBytes
            or type(stage) is not NativeColdScratch
            or not data.identity == stage.identity == checkpoint.identity
            or not data.region_sha256 == stage.region_sha256 == self.region_sha256
        ):
            raise ValueError("Native cold bytes or scratch changed rank authority")
        self._aux_range(stage.span)
        self._require_enrolled_span(stage.span, stage.extent_lease, "scratch")
        if self._extent is not None:
            assert stage.extent_lease is not None
            self._extent.verify(stage.extent_lease)

        def live():
            data.require_live()
            stage.require_live()

        return NativeRegionColdAuthority(
            region=self.region,
            identity=checkpoint.identity,
            plan=checkpoint.plans[self.rank],
            bindings=data.bindings,
            stage=stage.span,
            verify_reference=live,
            mode=data.mode,
            checksums=checksums,
        )


class NativeColdRankPort(Protocol):
    memory: NativeGlmRankMemory

    def retain_terminal(
        self, entry: GlmColdDirectoryImport
    ) -> NativeColdRankBytes | Wait:
        """Retain already prepared, authenticated worker bytes."""

    def scratch(
        self, checkpoint: GlmColdCheckpoint, mode: Mode
    ) -> NativeColdScratch | Wait:
        """Acquire one temporary native scratch reservation."""

    def reserve_restore(
        self, checkpoint: GlmColdCheckpoint, verified: VerifiedColdCheckpoint
    ) -> NativeColdRankBytes | Wait:
        """Acquire same-authority host pages and native auxiliary destinations."""


class GlmNativeColdRankProvider:
    """Aggregate existing native rank leases for the cold scheduler directory."""

    def __init__(self, ranks: tuple[NativeColdRankPort, ...]) -> None:
        if (
            type(ranks) is not tuple
            or tuple(port.memory.rank for port in ranks) != tuple(range(4))
            or len({port.memory.region.descriptor.layout_sha256 for port in ranks}) != 1
            or len(
                {
                    (
                        port.memory.region.descriptor.authority_engine_id,
                        port.memory.region.descriptor.authority_boot_id,
                    )
                    for port in ranks
                }
            )
            != 1
        ):
            raise ValueError("Cold provider requires four ranks of one authority")
        self.ranks = ranks
        self.region_fingerprints = tuple(port.memory.region_sha256 for port in ranks)
        self.layout_sha256 = ranks[0].memory.region.descriptor.layout_sha256
        self._held: dict[int, tuple[NativeColdRankBytes, ...]] = {}

    def retain_terminal(self, entry: GlmColdDirectoryImport) -> RetainedColdRam | Wait:
        checkpoint = entry.checkpoint
        data = []
        with ExitStack() as rollback:
            for port in self.ranks:
                local = port.retain_terminal(entry)
                if isinstance(local, Wait):
                    return local
                if type(local) is not NativeColdRankBytes:
                    raise TypeError("Cold terminal source needs its typed byte lease")
                rollback.callback(local.settle, "FAILED")
                if (
                    local.identity != checkpoint.identity
                    or local.region_sha256 != port.memory.region_sha256
                    or local.mode != "STORE"
                    or tuple(item.part for item in local.bindings)
                    != checkpoint.plans[port.memory.rank].parts
                ):
                    raise ValueError("Cold terminal rank lost its confirmed bytes")
                local.require_live()
                data.append(local)
            retained = tuple(data)
            rollback.pop_all()

        def live():
            if id(source) not in self._held:
                raise ValueError("Cold retained rank lease is stale")
            for local in retained:
                local.require_live()

        def release():
            live()
            with ExitStack() as cleanup:
                for local in retained:
                    cleanup.callback(local.settle, "RELEASED")
            del self._held[id(source)]

        source = RetainedColdRam(
            checkpoint.identity,
            tuple(port.memory.region for port in self.ranks),
            live,
            release,
        )
        self._held[id(source)] = retained
        return source

    def snapshot(
        self,
        checkpoint: GlmColdCheckpoint,
        source: RetainedColdRam,
        *,
        max_inflight_bytes: int,
    ) -> GlmColdRamGrant | Wait:
        source.require_live()
        held = self._held.get(id(source))
        if source.identity != checkpoint.identity or held is None:
            raise ValueError("Cold snapshot borrowed a foreign retained checkpoint")
        return self._operation(checkpoint, held, max_inflight_bytes, verified=None)

    def reserve_restore(
        self,
        checkpoint: GlmColdCheckpoint,
        *,
        max_inflight_bytes: int,
        verified_catalog: VerifiedColdCheckpoint | None = None,
    ) -> GlmColdRamGrant | Wait:
        if verified_catalog is None:
            return Wait("NATIVE_COLD_CATALOG_REQUIRED")
        if (
            type(verified_catalog) is not VerifiedColdCheckpoint
            or verified_catalog.checkpoint != checkpoint
            or verified_catalog.target_boot_id
            != self.ranks[0].memory.region.descriptor.authority_boot_id
        ):
            raise ValueError("Cold destination changed its verified checkpoint")
        with ExitStack() as rollback:
            data = []
            for port in self.ranks:
                local = port.reserve_restore(checkpoint, verified_catalog)
                if isinstance(local, Wait):
                    return local
                if type(local) is not NativeColdRankBytes:
                    raise TypeError("Cold restore needs its typed byte lease")
                rollback.callback(local.settle, "MISS")
                if local.mode != "RESTORE":
                    raise ValueError("Cold restore needs an exact native write lease")
                data.append(local)
            grant = self._operation(
                checkpoint, tuple(data), max_inflight_bytes, verified=verified_catalog
            )
            if isinstance(grant, Wait):
                return grant
            rollback.pop_all()
            return grant

    def _operation(self, checkpoint, data, budget, *, verified):
        mode = "STORE" if verified is None else "RESTORE"
        with ExitStack() as rollback:
            ranks, charged = [], 0
            for rank, (port, local) in enumerate(zip(self.ranks, data, strict=True)):
                stage = port.scratch(checkpoint, mode)
                if isinstance(stage, Wait):
                    return stage
                rollback.callback(stage.release)
                ranks.append(
                    port.memory.make_authority(
                        checkpoint,
                        local,
                        stage,
                        verified.checksums(rank) if verified is not None else None,
                    )
                )
                charged += stage.span.nbytes
            largest = max(
                part.nbytes for plan in checkpoint.plans for part in plan.parts
            )
            if charged + min(largest, 65536) > budget:
                return Wait("NATIVE_COLD_STAGING_BUDGET_WAIT")
            cleanup = rollback.pop_all()

        def settle(state):
            with ExitStack() as settlement:
                if mode == "RESTORE":
                    for local in data:
                        settlement.callback(local.settle, state)
                cleanup.close()

        return GlmColdRamGrant(checkpoint, tuple(ranks), self.layout_sha256, settle)
