# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reserve cold ranges in the same native HiSparse authority memfd."""

from __future__ import annotations

import mmap
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Literal

from kitepond.core.contracts import Wait

from .shared_region import NativeHiSparseMemfdRegion

Zone = Literal["aux", "scratch"]
_ALIGNMENT = 64


@dataclass(frozen=True)
class ColdExtentBudget:
    aux_bytes_per_rank: int
    scratch_bytes_per_rank: int
    max_total_bytes: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int
            or not 0 < value <= sys.maxsize
            or value % mmap.PAGESIZE
            for value in (
                self.aux_bytes_per_rank,
                self.scratch_bytes_per_rank,
                self.max_total_bytes,
            )
        ):
            raise ValueError("Native cold budgets need positive page-aligned bytes")

    def layout(self, mla_bytes: int) -> ColdExtentLayout:
        if type(mla_bytes) is not int or mla_bytes <= 0:
            raise ValueError("Native cold extent requires the real MLA size")
        padded = (mla_bytes + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
        aux_end = padded + self.aux_bytes_per_rank
        extent = aux_end + self.scratch_bytes_per_rank
        if 4 * extent > self.max_total_bytes:
            raise ValueError("Four native ranks exceed the explicit cold extent budget")
        return ColdExtentLayout(
            mla_bytes, padded, aux_end, extent, self.max_total_bytes
        )


@dataclass(frozen=True)
class ColdExtentLayout:
    mla_bytes: int
    mla_padded_bytes: int
    aux_end: int
    extent: int
    max_total_bytes: int

    def bounds(self, zone: Zone) -> tuple[int, int]:
        if zone == "aux":
            return self.mla_padded_bytes, self.aux_end
        if zone == "scratch":
            return self.aux_end, self.extent
        raise ValueError("Unknown native cold range partition")


@dataclass(frozen=True, eq=False)
class ColdExtentLease:
    """One owner-local byte range, valid until the exact lease is released."""

    allocation_id: int
    zone: Zone
    offset: int
    nbytes: int


class NativeColdExtent:
    """Allocate metadata-only ranges under the one memfd owner."""

    def __init__(
        self,
        region: NativeHiSparseMemfdRegion,
        layout: ColdExtentLayout,
        retired: Callable[[], None],
    ) -> None:
        if (
            type(region) is not NativeHiSparseMemfdRegion
            or type(layout) is not ColdExtentLayout
            or not region.owner
            or region.descriptor.owner_pid != os.getpid()
            or region.mapping.closed
            or (
                len(region.mapping),
                region.descriptor.nbytes,
            )
            != (layout.extent, layout.extent)
            or not 0
            < layout.mla_bytes
            <= layout.mla_padded_bytes
            < layout.aux_end
            < layout.extent
            or 4 * layout.extent > layout.max_total_bytes
            or any(
                value % mmap.PAGESIZE
                for value in (layout.mla_padded_bytes, layout.aux_end, layout.extent)
            )
            or not callable(retired)
        ):
            raise ValueError("Cold ranges require their one native region owner")
        self.region, self.layout = region, layout
        self.region_sha256 = region.descriptor.fingerprint()
        self._descriptor = region.descriptor
        self._mapping = region.mapping
        self._leases: dict[int, ColdExtentLease] = {}
        self._next_id = 1
        self._lock = Lock()
        self._retiring = False
        self._retired = False
        self._on_retired = retired

    def _require_owner(self) -> None:
        if (
            self._descriptor.owner_pid != os.getpid()
            or not self.region.owner
            or self.region.descriptor is not self._descriptor
            or self.region.mapping is not self._mapping
            or self._mapping.closed
        ):
            raise ValueError("Cold range belongs to another native owner")

    def reserve(self, zone: Zone, nbytes: int) -> ColdExtentLease | Wait:
        if type(nbytes) is not int or nbytes <= 0:
            raise ValueError("Cold range needs its exact positive byte count")
        start, end = self.layout.bounds(zone)
        with self._lock:
            self._require_owner()
            if self._retiring:
                return Wait("NATIVE_COLD_EXTENT_RETIRING")
            current = start
            leases = sorted(
                (lease for lease in self._leases.values() if lease.zone == zone),
                key=lambda lease: lease.offset,
            )
            for lease in leases:
                if current + nbytes <= lease.offset:
                    break
                stop = lease.offset + lease.nbytes
                current = (stop + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
            if current + nbytes > end:
                return Wait("NATIVE_COLD_EXTENT_CAPACITY_WAIT")
            lease = ColdExtentLease(self._next_id, zone, current, nbytes)
            self._leases[lease.allocation_id] = lease
            self._next_id += 1
            return lease

    def _require_lease(self, lease: ColdExtentLease) -> None:
        self._require_owner()
        if (
            type(lease) is not ColdExtentLease
            or self._leases.get(lease.allocation_id) is not lease
        ):
            raise ValueError("Cold native range is stale or foreign")

    def verify(self, lease: ColdExtentLease) -> None:
        with self._lock:
            self._require_lease(lease)

    def _complete_retire(self) -> None:
        if self._retiring and not self._leases and not self._retired:
            self._on_retired()
            self._retired = True

    def release(self, lease: ColdExtentLease) -> None:
        with self._lock:
            self._require_lease(lease)
            del self._leases[lease.allocation_id]
            self._complete_retire()

    def retire_after_unpin(self) -> None:
        """Reject new reservations and keep the original fd for live leases."""
        with self._lock:
            self._require_owner()
            self._retiring = True
            self._complete_retire()
