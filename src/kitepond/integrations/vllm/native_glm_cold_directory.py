# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Own verified GLM entries over the native scheduler's existing RAM pool."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from functools import partial
from time import monotonic_ns
from typing import TYPE_CHECKING, Protocol

from kitepond.backends.fsl2 import BundleIdentity
from kitepond.core.contracts import Wait
from kitepond.models.glm5next.cold_authority import NativeRegionColdAuthority
from kitepond.models.glm5next.cold_lifecycle import GlmColdCheckpoint, GlmColdRamGrant
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion

from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
from vllm.v1.request import Request

from .glm_cold_catalog import GlmColdCatalog, VerifiedColdCheckpoint
from .glm_cold_runtime import ColdDirectoryScope, GlmColdDirectoryImport
from .terminal_checkpoint import StockTerminalCheckpoints, TerminalPinLease

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock


@dataclass(frozen=True)
class RetainedColdRam:
    """Persistent rank refs. Release this entry's refs without closing the pool."""

    identity: BundleIdentity
    regions: tuple[NativeHiSparseMemfdRegion, ...]
    require_live: Callable[[], None]
    release: Callable[[], None]

    def __post_init__(self) -> None:
        if not callable(self.require_live) or not callable(self.release):
            raise TypeError("Native RAM requires verification and release callbacks")


class NativeColdRankProvider(Protocol):
    region_fingerprints: tuple[str, ...]

    def retain_terminal(self, entry: GlmColdDirectoryImport) -> RetainedColdRam | Wait:
        """Retain complete accepted native bytes after all rank writes settle.

        The existing HiSparse MLA pages are reused. The provider supplies
        already materialized Indexer/KDA/KPool and any auxiliary spans under
        the same checkpoint identity. Wait retains no new storage reference.
        """

    def snapshot(
        self,
        checkpoint: GlmColdCheckpoint,
        source: RetainedColdRam,
        *,
        max_inflight_bytes: int,
    ) -> GlmColdRamGrant | Wait:
        """Acquire four exact source leases and bounded temporary staging."""

    def reserve_restore(
        self,
        checkpoint: GlmColdCheckpoint,
        *,
        max_inflight_bytes: int,
        verified_catalog: VerifiedColdCheckpoint | None = None,
    ) -> GlmColdRamGrant | Wait:
        """Reserve real same-owner RAM and return four exact write leases."""


@dataclass(eq=False, frozen=True)
class ColdRamRead:
    """Live business reference to one retained checkpoint."""

    checkpoint: GlmColdCheckpoint
    source: RetainedColdRam


@dataclass
class _Resident:
    source: RetainedColdRam
    blocks: tuple[KVCacheBlock, ...]


@dataclass
class _Entry:
    checkpoint: GlmColdCheckpoint
    last_use_ns: int
    resident: _Resident | None
    readers: set[ColdRamRead] = field(default_factory=set)
    io: object | None = None
    activity: int = 0
    verified_catalog: VerifiedColdCheckpoint | None = None
    accepted_prefix_sha256: str | None = None


class NativeGlmColdDirectory:
    """Metadata directory with references to existing host pages."""

    def __init__(
        self,
        terminal: StockTerminalCheckpoints,
        provider: NativeColdRankProvider,
        resolve_scope: Callable[[Request], ColdDirectoryScope | Wait],
        *,
        max_entries: int,
        clock: Callable[[], int] = monotonic_ns,
        catalog: GlmColdCatalog | None = None,
    ) -> None:
        if not isinstance(terminal, StockTerminalCheckpoints):
            raise TypeError("Cold directory requires the native terminal owner")
        terminal._require_owner()
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("Cold directory needs a finite entry budget")
        self.engine_id, self.boot_id = terminal._identity
        if catalog is not None and (
            type(catalog) is not GlmColdCatalog
            or (catalog.engine_id, catalog.boot_id) != terminal._identity
        ):
            raise ValueError("Cold catalog belongs to another native engine or boot")
        self.catalog = catalog
        self._host_pool = get_hisparse_coordinator(
            terminal._manager
        ).get_host_block_pool()
        if not self.engine_id or not self.boot_id or self._host_pool is None:
            raise ValueError("Cold directory requires the live D host pool")
        fingerprints = provider.region_fingerprints
        if (
            type(fingerprints) is not tuple
            or len(fingerprints) != 4
            or any(
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
                for value in fingerprints
            )
            or len(set(fingerprints)) != 4
        ):
            raise ValueError("Cold directory requires four enrolled native RAM regions")
        self._rank_fingerprints = fingerprints
        self.terminal, self.provider = terminal, provider
        self.resolve_scope, self.clock = resolve_scope, clock
        self.max_entries = max_entries
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._last_clock = 0

    def _now(self) -> int:
        now = self.clock()
        if type(now) is not int or now < self._last_clock:
            raise ValueError("Cold directory business clock must remain monotonic")
        self._last_clock = now
        return now

    def _entry(self, checkpoint: GlmColdCheckpoint) -> _Entry:
        self.terminal._require_owner()
        entry = self._entries.get(checkpoint.namespace)
        if entry is None or entry.checkpoint != checkpoint:
            raise ValueError("Cold checkpoint is stale or belongs to another directory")
        return entry

    def _region_matches(self, region: NativeHiSparseMemfdRegion, rank: int) -> bool:
        if type(region) is not NativeHiSparseMemfdRegion or region.mapping.closed:
            return False
        descriptor = region.descriptor
        return (
            descriptor.authority_engine_id,
            descriptor.authority_boot_id,
            descriptor.source_rank,
        ) == (
            self.engine_id,
            self.boot_id,
            rank,
        ) and descriptor.fingerprint() == self._rank_fingerprints[rank]

    def _release_resident(self, entry: _Entry) -> None:
        resident = entry.resident
        if resident is None:
            return
        resident.source.release()
        self._host_pool.free_blocks(reversed(resident.blocks))
        entry.resident = None

    def cold_scope(self, lease: TerminalPinLease) -> ColdDirectoryScope | Wait:
        return self.resolve_scope(self.terminal._require_live(lease).request)

    def import_accepted_terminal(self, entry: GlmColdDirectoryImport) -> Wait | None:
        if type(entry) is not GlmColdDirectoryImport:
            raise TypeError("Cold directory import requires verified terminal evidence")
        checkpoint, lease = entry.checkpoint, entry.lease
        self.terminal.require_read_open(lease, entry.permit)
        scope = self.cold_scope(lease)
        if isinstance(scope, Wait):
            return scope
        if type(scope) is not ColdDirectoryScope or (
            scope.namespace,
            scope.generation,
        ) != (checkpoint.namespace, checkpoint.generation):
            raise ValueError("Cold directory import changed its authorized scope")
        entry.require_exact_checkpoint()
        if self.catalog is not None:
            self.catalog.require_layout_identity(checkpoint.layout)
        current = self._entries.get(checkpoint.namespace)
        if current is not None:
            if current.checkpoint == checkpoint:
                return None
            if current.checkpoint.generation >= checkpoint.generation:
                raise ValueError("Cold directory generation cannot move backward")
            if current.readers or current.io is not None:
                return Wait("GLM_DIRECTORY_IN_USE")
        elif len(self._entries) >= self.max_entries:
            return Wait("GLM_DIRECTORY_FULL")

        hold = self.terminal._require_live(lease)
        [blocks] = [owned for pool, owned in hold.refs if pool is self._host_pool]
        if not blocks or any(
            block.is_null
            or block.ref_cnt <= 0
            or self._host_pool.blocks[block.block_id] is not block
            for block in blocks
        ):
            raise ValueError("Terminal checkpoint lost its native host references")

        now = self._now()
        source = self.provider.retain_terminal(entry)
        if isinstance(source, Wait):
            return source
        if type(source) is not RetainedColdRam:
            raise TypeError("Native terminal provider did not retain four RAM ranks")
        try:
            if (
                source.identity != checkpoint.identity
                or type(source.regions) is not tuple
                or len(source.regions) != 4
                or any(
                    not self._region_matches(region, rank)
                    for rank, region in enumerate(source.regions)
                )
            ):
                raise ValueError("Native terminal RAM changed checkpoint, boot or rank")
            source.require_live()
            self._host_pool.touch(blocks)
            try:
                if current is not None:
                    self._release_resident(current)
            except Exception:
                self._host_pool.free_blocks(reversed(blocks))
                raise
            self._entries[checkpoint.namespace] = _Entry(
                checkpoint, now, _Resident(source, blocks)
            )
        except Exception:
            source.release()
            raise
        return None

    def acquire_ram_read(self, checkpoint: GlmColdCheckpoint) -> ColdRamRead | Wait:
        entry = self._entry(checkpoint)
        if entry.resident is None:
            return Wait("GLM_CHECKPOINT_ON_NVME")
        entry.resident.source.require_live()
        now = self._now()
        read = ColdRamRead(checkpoint, entry.resident.source)
        entry.last_use_ns = now
        entry.activity += 1
        entry.readers.add(read)
        return read

    def release_ram_read(self, read: ColdRamRead) -> None:
        entry = self._entry(read.checkpoint)
        if read not in entry.readers:
            raise ValueError("Cold RAM read reference is stale or foreign")
        now = self._now()
        entry.readers.remove(read)
        entry.last_use_ns = now
        entry.activity += 1

    def bind_resident_prefix(
        self, checkpoint: GlmColdCheckpoint, prefix_sha256: str
    ) -> None:
        entry = self._entry(checkpoint)
        if (
            type(prefix_sha256) is not str
            or len(prefix_sha256) != 64
            or any(char not in "0123456789abcdef" for char in prefix_sha256)
            or entry.resident is None
            or entry.io is not None
            or entry.accepted_prefix_sha256 not in (None, prefix_sha256)
        ):
            raise ValueError("Native directory prefix changed or has started I/O")
        entry.accepted_prefix_sha256 = prefix_sha256

    def resident_checkpoint(
        self, scope: ColdDirectoryScope
    ) -> GlmColdCheckpoint | Wait:
        self.terminal._require_owner()
        if type(scope) is not ColdDirectoryScope:
            raise ValueError("Cold operator requires an exact session generation")
        entry = self._entries.get(scope.namespace)
        if entry is None or entry.checkpoint.generation != scope.generation:
            return Wait("GLM_COLD_GENERATION_UNAVAILABLE")
        if entry.resident is None:
            return Wait("GLM_CHECKPOINT_ON_NVME")
        if entry.readers or entry.io is not None:
            return Wait("GLM_LIVE_REFERENCE_WAIT")
        entry.resident.source.require_live()
        return entry.checkpoint

    def last_business_use_ns(self, checkpoint: GlmColdCheckpoint) -> int:
        return self._entry(checkpoint).last_use_ns

    def expired_verified_checkpoints(
        self, *, idle_cutoff_ns: int, limit: int
    ) -> Iterator[GlmColdCheckpoint]:
        self.terminal._require_owner()
        for _ in range(min(limit, len(self._entries))):
            namespace, entry = next(iter(self._entries.items()))
            self._entries.move_to_end(namespace)
            if (
                entry.resident is not None
                and not entry.readers
                and entry.io is None
                and entry.last_use_ns <= idle_cutoff_ns
            ):
                yield entry.checkpoint

    def _protect_job(
        self,
        entry: _Entry,
        checkpoint: GlmColdCheckpoint,
        grant: GlmColdRamGrant | Wait,
        *,
        idle_cutoff_ns: int | None = None,
    ) -> GlmColdRamGrant | Wait:
        if isinstance(grant, Wait):
            return grant
        if type(grant) is not GlmColdRamGrant:
            raise TypeError("Native provider must return a typed checkpoint grant")
        resident = entry.resident
        mode = "STORE" if idle_cutoff_ns is not None else "RESTORE"
        if (
            grant.checkpoint is not checkpoint
            or type(grant.ranks) is not tuple
            or len(grant.ranks) != 4
            or any(
                type(authority) is not NativeRegionColdAuthority
                or authority.identity != checkpoint.identity
                or authority.plan != checkpoint.plans[rank]
                or authority.mode != mode
                or not self._region_matches(authority.region, rank)
                or (
                    resident is not None
                    and authority.region is not resident.source.regions[rank]
                )
                for rank, authority in enumerate(grant.ranks)
            )
        ):
            grant.settle("FAILED" if mode == "STORE" else "MISS")
            raise ValueError("Cold provider changed the active rank or checkpoint")
        operation, observed, activity = object(), entry.last_use_ns, entry.activity
        entry.io = operation

        def settle(state):
            self.terminal._require_owner()
            if entry.io is not operation:
                raise ValueError("Cold operation is already settled or foreign")
            grant.settle(state)
            entry.io = None
            if (
                state == "COMMITTED"
                and idle_cutoff_ns is not None
                and not entry.readers
                and entry.activity == activity
                and entry.last_use_ns == observed <= idle_cutoff_ns
            ):
                self._release_resident(entry)

        if mode == "STORE" and self.catalog is not None:
            return replace(
                grant,
                settle=settle,
                on_durable_publication=partial(
                    self.catalog.publish,
                    checkpoint,
                    accepted_prefix_sha256=entry.accepted_prefix_sha256,
                ),
            )
        verified = entry.verified_catalog
        return replace(
            grant,
            settle=settle,
            expected_aggregate_sha256=(
                verified.aggregate_sha256
                if verified is not None
                else grant.expected_aggregate_sha256
            ),
        )

    def register_verified_cold(
        self, request: Request, receipt: VerifiedColdCheckpoint
    ) -> GlmColdCheckpoint | Wait:
        """Authorize metadata for the current native request before RAM admission."""
        self.terminal._require_owner()
        if (
            type(request) is not Request
            or self.catalog is None
            or not self.catalog.verify(receipt)
        ):
            raise ValueError("Cold catalog receipt is unavailable or from another boot")
        checkpoint = receipt.checkpoint
        scope = self.resolve_scope(request)
        if isinstance(scope, Wait):
            return scope
        if scope != ColdDirectoryScope(checkpoint.namespace, checkpoint.generation):
            raise ValueError("Cold catalog receipt does not authorize this session")
        current = self._entries.get(checkpoint.namespace)
        if current is not None:
            if current.checkpoint == checkpoint:
                current.verified_catalog = receipt
                return current.checkpoint
            if current.checkpoint.generation >= checkpoint.generation:
                raise ValueError("Cold catalog generation cannot move backward")
            if (
                current.resident is not None
                or current.io is not None
                or current.readers
            ):
                return Wait("GLM_DIRECTORY_IN_USE")
        elif len(self._entries) >= self.max_entries:
            return Wait("GLM_DIRECTORY_FULL")
        self._entries[checkpoint.namespace] = _Entry(
            checkpoint, self._now(), None, verified_catalog=receipt
        )
        return checkpoint

    def acquire_snapshot(
        self,
        checkpoint: GlmColdCheckpoint,
        *,
        observed_last_use_ns: int,
        idle_cutoff_ns: int,
        max_inflight_bytes: int,
    ) -> GlmColdRamGrant | Wait:
        entry = self._entry(checkpoint)
        resident = entry.resident
        if (
            resident is None
            or entry.readers
            or entry.io is not None
            or not entry.last_use_ns == observed_last_use_ns <= idle_cutoff_ns
        ):
            return Wait("GLM_LIVE_REFERENCE_WAIT")
        resident.source.require_live()
        grant = self.provider.snapshot(
            checkpoint, resident.source, max_inflight_bytes=max_inflight_bytes
        )
        return self._protect_job(
            entry, checkpoint, grant, idle_cutoff_ns=idle_cutoff_ns
        )

    def reserve_restore(
        self, checkpoint: GlmColdCheckpoint, *, max_inflight_bytes: int
    ) -> GlmColdRamGrant | Wait:
        entry = self._entry(checkpoint)
        if entry.resident is not None or entry.io is not None:
            return Wait("GLM_DIRECTORY_IN_USE")
        verified = entry.verified_catalog
        if verified is None:
            grant = self.provider.reserve_restore(
                checkpoint, max_inflight_bytes=max_inflight_bytes
            )
        else:
            grant = self.provider.reserve_restore(
                checkpoint,
                max_inflight_bytes=max_inflight_bytes,
                verified_catalog=verified,
            )
        return self._protect_job(entry, checkpoint, grant)

    def forget_cold(self, checkpoint: GlmColdCheckpoint) -> None:
        """Retire in-memory lookup metadata after the native catalog forgets it."""
        entry = self._entry(checkpoint)
        if entry.resident is not None or entry.readers or entry.io is not None:
            raise ValueError("Cold directory entry still has native references")
        del self._entries[checkpoint.namespace]
