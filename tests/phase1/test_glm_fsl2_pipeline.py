# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from contextlib import ExitStack, contextmanager

import torch
from kitepond.backends.fsl2 import (
    BundleIdentity,
    BundlePlan,
    ColdStorageAdapter,
    digest,
)
from kitepond.backends.fsl2_aggregate import FourRankManifestStore
from kitepond.models.glm5next.cold_authority import NativeRegionColdAuthority
from kitepond.models.glm5next.cold_restore import verify_cold_import_readback
from kitepond.models.glm5next.import_recipe import (
    ByteSpan,
    ColdBinding,
    compose_import_recipe,
    copy_import_window,
)
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion
from test_glm_cold_restore import _cache

STAGING = 4 * 2**20
METADATA = {
    "schema": "glm-ranked-readback-fixture",
    "model_fingerprint": "b" * 64,
    "config_fingerprint": "c" * 64,
    "boundary_tokens": 1059,
}
IDENTITY = BundleIdentity("qualified-layout", "b" * 64, digest(METADATA), 17, 1059)


def _close_store(adapter):
    assert adapter.close(10)


@contextmanager
def _checkpoint_ranks(root, *, restore):
    with ExitStack() as cleanup:
        ranks = []
        for rank in range(4):
            layout = _cache(width=656, rank=rank)
            parts = layout.manifest()
            plan = BundlePlan(
                "glm-logical-role-shapes-v1",
                "d" * 64,
                tuple(dict.fromkeys(part.role for part in parts)),
                parts,
            )
            region = NativeHiSparseMemfdRegion.create(
                STAGING * 2,
                authority_engine_id="D",
                authority_boot_id="restore" if restore else "capture",
                source_rank=rank,
                layout_sha256=plan.representation_fingerprint,
            )
            cleanup.callback(region.close)
            hashes = {}
            bindings = []
            offset = 0
            for ordinal, part in enumerate(parts):
                payload = bytes([rank * 31 + ordinal + 1]) * part.nbytes
                hashes[part] = hashlib.sha256(payload).hexdigest()
                if not restore:
                    region.mapping[offset : offset + part.nbytes] = payload
                span = ByteSpan(rank, ordinal + 1, offset, part.nbytes)
                bindings.append(ColdBinding(part, (span,)))
                offset += part.nbytes
            assert offset <= STAGING

            def verify_reference(region=region):
                assert not region.mapping.closed

            owner = NativeRegionColdAuthority(
                region=region,
                identity=IDENTITY,
                plan=plan,
                bindings=tuple(bindings),
                stage=ByteSpan(rank, len(parts) + 1, STAGING, STAGING),
                verify_reference=verify_reference,
                mode="RESTORE" if restore else "STORE",
                checksums=hashes if restore else None,
            )
            storage = ColdStorageAdapter(
                root / f"rank-{rank}", owner, max_inflight_bytes=STAGING + 65536
            )
            cleanup.callback(_close_store, storage)
            ranks.append((storage, owner, plan, layout, tuple(bindings), hashes))
        yield tuple(ranks)


def test_all_glm_roles_recover_from_one_committed_four_rank_fsl2_manifest(tmp_path):
    with _checkpoint_ranks(tmp_path, restore=False) as captures:
        plans = tuple(rank[2] for rank in captures)
        aggregate = FourRankManifestStore(tmp_path, tuple(rank[0] for rank in captures))
        for backend, owner, plan, *_ in captures:
            ticket = backend.begin_store(IDENTITY, plan, owner.lease)
            assert backend.wait(ticket, 10) == "COMMITTED"
        published = aggregate.publish(IDENTITY, plans, METADATA)
        assert len(published["ranks"]) == 4

    with _checkpoint_ranks(tmp_path, restore=True) as receivers:
        plans = tuple(rank[2] for rank in receivers)
        aggregate = FourRankManifestStore(
            tmp_path, tuple(rank[0] for rank in receivers)
        )
        assert aggregate.discover(IDENTITY, plans) == published
        slab = torch.empty(STAGING, dtype=torch.uint8)
        for backend, owner, plan, layout, bindings, hashes in receivers:
            ticket = backend.begin_restore(IDENTITY, plan, owner.lease)
            assert backend.wait(ticket, 10) == "VERIFIED"
            recipe = compose_import_recipe(layout.manifest(), bindings)
            region = owner.region
            scope = region.descriptor.fingerprint()
            for window in recipe:
                payload = copy_import_window(region, window, slab, region_sha256=scope)
                layout.scatter_window(
                    tuple(binding.part for binding in window), payload
                )
            report = verify_cold_import_readback(
                layout, recipe, region, slab, region_sha256=scope
            )
            assert report["status"] == "MATCH"
            assert set(report["counts"]) == {
                "mla",
                "indexer",
                "kda_conv",
                "kda_recurrent",
                "kpool_tail",
            }
            assert all(
                hashlib.sha256(layout.read_chunk(part).numpy()).hexdigest() == expected
                for part, expected in hashes.items()
            )
