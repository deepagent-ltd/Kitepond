# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real FSL2 compatibility with only a CPU buffer owner and the Kitepond package."""

import pytest
from kitepond.core.contracts import Wait


def test_fsl2_reopens_cold_bundle_and_loads_into_new_caller_ram(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("lmcache")
    from kitepond.backends.fsl2 import (
        BundleIdentity,
        BundlePlan,
        ColdSnapshotLease,
        ColdStorageAdapter,
        MemoryBorrow,
        NativeColdAuthority,
        PartSpec,
        RamRestoreLease,
    )
    from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj

    def owned(data):
        return TensorMemoryObj(
            data,
            MemoryObjMetadata(
                shape=data.shape,
                dtype=data.dtype,
                address=data.data_ptr(),
                phy_size=data.nbytes,
                ref_count=1,
            ),
            parent_allocator=None,
        )

    identity = BundleIdentity("test", "a" * 64, "b" * 64, 1, 64)
    part = PartSpec("ordered-history", 0, 0, (64,), "torch.uint8", 64)
    plan = BundlePlan("test-fsl2-v1", "c" * 64, (part.role,), (part,))

    class Authority(NativeColdAuthority):
        def __init__(self, data):
            self.memory = owned(data)
            self.snapshot = ColdSnapshotLease(self, object())
            self.restore = RamRestoreLease(self, object())
            self.stored = False
            self.restored = None

        @staticmethod
        def _check(issued, lease, actual_identity, actual_plan):
            if (lease, actual_identity, actual_plan) != (issued, identity, plan):
                raise PermissionError("Foreign or stale lease")

        def check_store(self, lease, actual_identity, actual_plan):
            self._check(self.snapshot, lease, actual_identity, actual_plan)

        def borrow_source(self, lease, spec):
            assert lease is self.snapshot and spec == part
            return MemoryBorrow(self, self.memory)

        def finish_store(self, lease):
            assert lease is self.snapshot
            assert self.memory.get_ref_count() == 1
            self.stored = True

        def check_restore(self, lease, actual_identity, actual_plan):
            self._check(self.restore, lease, actual_identity, actual_plan)

        def borrow_destination(self, lease, spec):
            assert lease is self.restore and spec == part
            return MemoryBorrow(self, self.memory)

        def finish_restore(self, lease, verified):
            assert lease is self.restore
            assert self.memory.get_ref_count() == 1
            self.restored = verified

    original = torch.arange(64, dtype=torch.uint8)
    source = Authority(original)
    writer = ColdStorageAdapter(tmp_path, source, max_inflight_bytes=65536)
    try:
        ticket = writer.begin_store(identity, plan, source.snapshot)
        assert writer.wait(ticket, 10) == "COMMITTED"
        assert source.stored is True
    finally:
        assert writer.close(10)

    destination = Authority(torch.zeros_like(original))
    reader = ColdStorageAdapter(tmp_path, destination, max_inflight_bytes=65536)
    try:
        assert reader.begin_restore(identity, plan, None) == Wait("RAM_ADMISSION_WAIT")
        ticket = reader.begin_restore(identity, plan, destination.restore)
        assert reader.wait(ticket, 10) == "VERIFIED"
        assert destination.restored is True
        assert torch.equal(destination.memory.raw_tensor, original)
        assert destination.memory.raw_tensor.data_ptr() != original.data_ptr()
    finally:
        assert reader.close(10)
    assert not torch.cuda.is_initialized()
