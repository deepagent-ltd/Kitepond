# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import mmap

import pytest
from kitepond.backends.fsl2 import (
    BundleIdentity,
    BundlePlan,
    ColdStorageAdapter,
    PartSpec,
)
from kitepond.models.glm5next.cold_authority import NativeRegionColdAuthority
from kitepond.models.glm5next.import_recipe import ByteSpan, ColdBinding, part_role
from kitepond.models.glm5next.layout import CacheIdentity, CacheRole
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion


def test_fsl2_persists_actual_stock_glm_bytes_into_fresh_authority(tmp_path):
    role = part_role(CacheIdentity("layers.1.mla", CacheRole.MLA, "target"))
    part = PartSpec(role, 2, 0, (4, 656), "torch.uint8", 4 * 656)
    identity = BundleIdentity("test", "a" * 64, "b" * 64, 4, 4)
    plan = BundlePlan("stock-glm-role-v1", "c" * 64, (role,), (part,))
    pages = mmap.PAGESIZE
    first = ByteSpan(2, 1, 0, 2 * 656)
    second = ByteSpan(2, 2, pages, 2 * 656)
    stage = ByteSpan(2, 3, pages * 2, pages)
    bindings = (ColdBinding(part, (first, second)),)
    payload = bytes(i % 247 for i in range(part.nbytes))

    def region(boot):
        return NativeHiSparseMemfdRegion.create(
            pages * 3,
            authority_engine_id="D",
            authority_boot_id=boot,
            source_rank=2,
            layout_sha256="c" * 64,
        )

    def authority(memfd, **kwargs):
        def check():
            assert not memfd.mapping.closed

        return NativeRegionColdAuthority(
            region=memfd,
            identity=identity,
            plan=plan,
            bindings=bindings,
            stage=stage,
            verify_reference=check,
            **kwargs,
        )

    source = region("first")
    inode = source.descriptor.inode
    source.mapping[: first.nbytes] = payload[: first.nbytes]
    source.mapping[second.offset : second.offset + second.nbytes] = payload[
        first.nbytes :
    ]
    try:
        author = authority(source, mode="STORE")
        store = ColdStorageAdapter(tmp_path, author, max_inflight_bytes=65536)
        try:
            ticket = store.begin_store(identity, plan, author.lease)
            assert store.wait(ticket, 10) == "COMMITTED"
        finally:
            assert store.close(10)
        with pytest.raises(ValueError, match="completed"):
            author.check_store(author.lease, identity, plan)
    finally:
        source.close()

    destination = region("fresh")
    try:
        assert destination.descriptor.inode != inode
        author = authority(
            destination,
            mode="RESTORE",
            checksums={part: hashlib.sha256(payload).hexdigest()},
        )
        store = ColdStorageAdapter(tmp_path, author, max_inflight_bytes=65536)
        try:
            ticket = store.begin_restore(identity, plan, author.lease)
            assert store.wait(ticket, 10) == "VERIFIED"
        finally:
            assert store.close(10)
        assert destination.mapping[: first.nbytes] == payload[: first.nbytes]
        assert (
            destination.mapping[second.offset : second.offset + second.nbytes]
            == payload[first.nbytes :]
        )
        assert destination.mapping[first.nbytes : second.offset] == bytes(
            second.offset - first.nbytes
        )
    finally:
        destination.close()
