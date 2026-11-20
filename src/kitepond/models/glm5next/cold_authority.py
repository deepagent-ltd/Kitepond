# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use native GLM memfd spans and one owned staging range for FSL2 I/O."""

from collections.abc import Callable, Mapping
from hashlib import sha256
from threading import Lock
from typing import Literal

import torch
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj

from kitepond.backends.fsl2 import (
    BundleIdentity,
    BundlePlan,
    ColdSnapshotLease,
    MemoryBorrow,
    NativeColdAuthority,
    PartSpec,
    RamRestoreLease,
)

from .import_recipe import ByteSpan, ColdBinding
from .shared_region import NativeHiSparseMemfdRegion

Mode = Literal["STORE", "RESTORE"]


class NativeRegionColdAuthority(NativeColdAuthority):
    """One trusted native reference supplies rank bytes and staging.

    The caller's verifier checks the active physical page lease. Read and
    write reservations remain held until FSL2 stops touching both ranges.
    """

    def __init__(
        self,
        *,
        region: NativeHiSparseMemfdRegion,
        identity: BundleIdentity,
        plan: BundlePlan,
        bindings: tuple[ColdBinding, ...],
        stage: ByteSpan,
        verify_reference: Callable[[], None],
        mode: Mode,
        checksums: Mapping[PartSpec, str] | None = None,
    ) -> None:
        parts = {binding.part: binding.spans for binding in bindings}
        max_part = max(part.nbytes for part in plan.parts)
        if (
            tuple(parts) != plan.parts
            or len(parts) != len(bindings)
            or stage.rank != region.descriptor.source_rank
            or stage.offset + stage.nbytes > region.descriptor.nbytes
            or not max_part <= stage.nbytes <= 4 * 2**20
            or not callable(verify_reference)
            or mode not in {"STORE", "RESTORE"}
        ):
            raise ValueError("Cold lease does not bind one native staging range")
        bounds = []
        for part, spans in parts.items():
            if part.rank != stage.rank:
                raise ValueError("Cold ranks cannot share one row staging range")
            for span in spans:
                end = span.offset + span.nbytes
                if (
                    span.rank != stage.rank
                    or end > region.descriptor.nbytes
                    or (
                        span.offset < stage.offset + stage.nbytes and stage.offset < end
                    )
                ):
                    raise ValueError("Cold data and staging buffers overlap or escape")
                bounds.append((span.offset, end))
        bounds.sort()
        if any(left[1] > right[0] for left, right in zip(bounds, bounds[1:])):
            raise ValueError("Cold native source ranges alias physical bytes")
        if mode == "RESTORE" and (
            checksums is None or checksums.keys() != parts.keys()
        ):
            raise ValueError("Native cold restore needs each verified part checksum")
        verify_reference()
        self.region = region
        self.identity = identity
        self.plan = plan
        self.stage = stage
        self.mode = mode
        self._binding = parts
        self._verify_reference = verify_reference
        self._checksums = dict(checksums or {})
        self._written: set[PartSpec] = set()
        self._active = False
        self._lock = Lock()
        self._finished = False
        self._verified_restore = False
        self.lease = (
            ColdSnapshotLease(self, object())
            if mode == "STORE"
            else RamRestoreLease(self, object())
        )

    def _check(self, lease, identity, plan, mode):
        if (
            lease is not self.lease
            or self.mode != mode
            or self._finished
            or identity != self.identity
            or plan != self.plan
        ):
            raise ValueError("Foreign or completed native cold lease")
        self._verify_reference()

    def check_store(self, lease, identity, plan):
        self._check(lease, identity, plan, "STORE")

    def check_restore(self, lease, identity, plan):
        self._check(lease, identity, plan, "RESTORE")

    def _copy(self, part: PartSpec, staging: memoryview, *, into_native: bool) -> None:
        current = 0
        for span in self._binding[part]:
            with memoryview(self.region.mapping)[
                span.offset : span.offset + span.nbytes
            ] as native:
                end = current + span.nbytes
                if into_native:
                    native[:] = staging[current:end]
                else:
                    staging[current:end] = native
            current = end

    def _borrow(self, lease, part, mode: Mode) -> MemoryBorrow:
        with self._lock:
            self._check(lease, self.identity, self.plan, mode)
            if self._active or part not in self._binding:
                raise RuntimeError("Native cold staging already active or out of scope")
            self._active = True
        try:
            return self._borrow_staging(part, mode)
        except BaseException:
            with self._lock:
                self._active = False
            raise

    def _borrow_staging(self, part: PartSpec, mode: Mode) -> MemoryBorrow:
        start = self.stage.offset
        staged = memoryview(self.region.mapping)[start : start + part.nbytes]
        try:
            if mode == "STORE":
                self._copy(part, staged, into_native=False)
            tensor = torch.frombuffer(staged, dtype=torch.uint8)
            tensor = tensor.view(getattr(torch, part.dtype[6:])).reshape(part.shape)
            memory = TensorMemoryObj(
                tensor,
                MemoryObjMetadata(
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    address=tensor.data_ptr(),
                    phy_size=part.nbytes,
                    ref_count=1,
                ),
                parent_allocator=None,
            )

            def settled():
                try:
                    if (
                        mode == "RESTORE"
                        and sha256(staged).hexdigest() == self._checksums[part]
                    ):
                        self._verify_reference()
                        self._copy(part, staged, into_native=True)
                        self._written.add(part)
                finally:
                    staged.release()
                    with self._lock:
                        self._active = False

            borrowed = MemoryBorrow(self, memory, settled)
        except BaseException:
            staged.release()
            raise
        return borrowed

    def borrow_source(self, lease, part):
        return self._borrow(lease, part, "STORE")

    def borrow_destination(self, lease, part):
        return self._borrow(lease, part, "RESTORE")

    def _finish(self, lease, mode: Mode, *, require_verified: bool = False) -> None:
        with self._lock:
            self._check(lease, self.identity, self.plan, mode)
            if self._active:
                raise RuntimeError(
                    "Native cold authority still has a storage reference"
                )
            if require_verified and self._written != set(self.plan.parts):
                raise ValueError("Native restore did not verify every byte span")
            self._finished = True
            self._verified_restore = require_verified

    def finish_store(self, lease):
        self._finish(lease, "STORE")

    def finish_restore(self, lease, verified):
        self._finish(lease, "RESTORE", require_verified=verified)

    def require_settled_restore(self, *, verified: bool) -> None:
        """Require I/O closure before the rank releases its RAM reservations."""
        with self._lock:
            if (
                self.mode != "RESTORE"
                or self._active
                or (verified and not self._verified_restore)
            ):
                raise ValueError("Native rank FSL2 readback has not settled")

    def verify_shared_restore(self) -> None:
        """Verify ranges written by the FSL2 owner through its separate mapping."""
        with self._lock:
            self._check(self.lease, self.identity, self.plan, "RESTORE")
            if self._active:
                raise ValueError("Native rank still has an active storage borrower")
            for part, spans in self._binding.items():
                checksum = sha256()
                for span in spans:
                    with memoryview(self.region.mapping)[
                        span.offset : span.offset + span.nbytes
                    ] as data:
                        checksum.update(data)
                if checksum.hexdigest() != self._checksums[part]:
                    raise ValueError(
                        "Shared native FSL2 bytes do not match the catalog"
                    )
            self._finished = True
            self._verified_restore = True
