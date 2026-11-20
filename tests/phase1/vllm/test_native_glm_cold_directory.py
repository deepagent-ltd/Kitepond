# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import ExitStack
from dataclasses import replace
from hashlib import sha256

import pytest
from kitepond.core.contracts import Wait
from kitepond.core.policy import IdlePolicy, LaneBudget
from kitepond.integrations.vllm.glm_cold_catalog import (
    GlmColdCatalog,
    GlmColdCatalogIdentityError,
)
from kitepond.integrations.vllm.glm_cold_runtime import (
    ColdDirectoryScope,
    GlmColdRuntime,
)
from kitepond.integrations.vllm.native_glm_cold_directory import (
    NativeGlmColdDirectory,
    RetainedColdRam,
)
from kitepond.models.glm5next.cold_authority import NativeRegionColdAuthority
from kitepond.models.glm5next.cold_lifecycle import GlmColdRamGrant
from kitepond.models.glm5next.import_recipe import ByteSpan, ColdBinding
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion
from test_glm_cold_lifecycle import BUDGET, STAGE_BYTES, _wait
from test_glm_terminal_manifest import (
    active_manifest as active_manifest,
    capture as capture,
    fenced_source,
)
from test_stock_terminal_checkpoint import receipt

RAM_LAYOUT = "d" * 64


class CpuRankProvider:
    """Use existing native memfd spans for a fixture's full accepted format."""

    def __init__(self, cleanup, lease, manifest):
        self.regions = tuple(
            NativeHiSparseMemfdRegion.create(
                3 * STAGE_BYTES,
                authority_engine_id=lease.identity.source_engine_id,
                authority_boot_id=lease.identity.source_boot_id,
                source_rank=rank,
                layout_sha256=RAM_LAYOUT,
            )
            for rank in range(4)
        )
        self.region_fingerprints = tuple(
            region.descriptor.fingerprint() for region in self.regions
        )
        self.parts = []
        for rank, region in enumerate(self.regions):
            cleanup.callback(region.close)
            plan = manifest.ranks[rank].checkpoint.plan
            sources, targets, checksums = [], [], {}
            offset = 0
            for index, part in enumerate(plan.parts, 1):
                payload = bytes([rank * 17 + index]) * part.nbytes
                region.mapping[offset : offset + part.nbytes] = payload
                source = ByteSpan(rank, index, offset, part.nbytes)
                target = replace(source, offset=offset + STAGE_BYTES)
                sources.append(ColdBinding(part, (source,)))
                targets.append(ColdBinding(part, (target,)))
                checksums[part] = sha256(payload).hexdigest()
                offset += part.nbytes
            assert offset <= STAGE_BYTES
            self.parts.append((tuple(sources), tuple(targets), checksums))
        self.ready = False
        self.resident = False
        self.ram_full = True
        self.io_open = False
        self.retired = 0

    def _require_resident(self):
        assert self.resident

    def retain_terminal(self, entry):
        if not self.ready:
            return Wait("NATIVE_AUX_STATE_NOT_RESIDENT")
        assert not self.resident
        self.resident = True

        def release():
            self._require_resident()
            assert not self.io_open
            self.resident = False
            self.retired += 1

        return RetainedColdRam(
            entry.checkpoint.identity, self.regions, self._require_resident, release
        )

    def _reserve(self, checkpoint, mode, max_inflight_bytes):
        assert max_inflight_bytes == BUDGET and not self.io_open
        self.io_open = True
        storing = mode == "STORE"

        def live():
            assert self.io_open
            if storing:
                self._require_resident()

        ranks = tuple(
            NativeRegionColdAuthority(
                region=region,
                identity=checkpoint.identity,
                plan=checkpoint.plans[rank],
                bindings=self.parts[rank][0 if storing else 1],
                stage=ByteSpan(rank, 1000, 2 * STAGE_BYTES, STAGE_BYTES),
                verify_reference=live,
                mode=mode,
                checksums=None if storing else self.parts[rank][2],
            )
            for rank, region in enumerate(self.regions)
        )

        def settle(_state):
            live()
            assert all(not rank._active for rank in ranks)
            self.io_open = False

        return GlmColdRamGrant(checkpoint, ranks, RAM_LAYOUT, settle)

    def snapshot(self, checkpoint, source, *, max_inflight_bytes):
        assert source.regions == self.regions
        return self._reserve(checkpoint, "STORE", max_inflight_bytes)

    def reserve_restore(self, checkpoint, *, max_inflight_bytes):
        if self.ram_full:
            return Wait("RAM_ADMISSION_WAIT")
        return self._reserve(checkpoint, "RESTORE", max_inflight_bytes)

    def verify_restored_bytes(self):
        for rank, region in enumerate(self.regions):
            _, targets, hashes = self.parts[rank]
            for binding in targets:
                (span,) = binding.spans
                data = region.mapping[span.offset : span.offset + span.nbytes]
                assert sha256(data).hexdigest() == hashes[binding.part]


def test_native_directory_retains_real_pages_and_rechecks_business_activity(
    tmp_path, active_manifest
):
    _, owner, lease, manifest, received, _, complete, _ = active_manifest
    complete[0] = True
    permit = owner.begin_reads(lease, manifest)
    layout = fenced_source(lease, 0, "1" * 64).layout.layout
    clock = [25]
    scope = ColdDirectoryScope("actual-native-owner", 3)
    native_req = owner._require_live(lease).request

    def scope_for(request):
        assert request is native_req
        return scope

    with ExitStack() as cleanup:
        provider = CpuRankProvider(cleanup, lease, manifest)
        root = tmp_path / "cold"
        catalog = GlmColdCatalog(
            root,
            engine_id=lease.identity.source_engine_id,
            boot_id=lease.identity.source_boot_id,
            model_fingerprint=layout.model_fingerprint,
            config_fingerprint=layout.config_fingerprint,
            signing_key=bytes(range(32)),
        )
        directory = NativeGlmColdDirectory(
            owner,
            provider,
            scope_for,
            max_entries=8,
            clock=lambda: clock[0],
            catalog=catalog,
        )
        idle = IdlePolicy(15)
        lane = LaneBudget(1, BUDGET, BUDGET)
        runtime = GlmColdRuntime(
            root,
            directory,
            owner,
            register_quiescent=lambda **_: None,
            cadence_ns=20_000_000,
            idle=idle,
            foreground=lane,
            background=lane,
            clock=lambda: clock[0],
        )
        hold = owner._require_live(lease)
        blocks = tuple(
            block
            for pool, group in hold.refs
            if pool is directory._host_pool
            for block in group
        )
        assert blocks
        before = tuple(block.ref_cnt for block in blocks)
        assert runtime.import_terminal(lease, permit, manifest, layout) == Wait(
            "NATIVE_AUX_STATE_NOT_RESIDENT"
        )
        assert tuple(block.ref_cnt for block in blocks) == before
        assert not (tmp_path / "cold").exists()
        provider.ready = True
        for model, config in (
            ("f" * 64, layout.config_fingerprint),
            (layout.model_fingerprint, "f" * 64),
        ):
            directory.catalog = GlmColdCatalog(
                root,
                engine_id=lease.identity.source_engine_id,
                boot_id=lease.identity.source_boot_id,
                model_fingerprint=model,
                config_fingerprint=config,
                signing_key=bytes(range(32)),
            )
            with pytest.raises(GlmColdCatalogIdentityError):
                runtime.import_terminal(lease, permit, manifest, layout)
            assert tuple(block.ref_cnt for block in blocks) == before
            assert not provider.resident and not provider.io_open
            assert not root.exists()
        directory.catalog = catalog
        checkpoint = runtime.import_terminal(lease, permit, manifest, layout)
        assert not isinstance(checkpoint, Wait)
        assert tuple(block.ref_cnt for block in blocks) == tuple(x + 1 for x in before)
        assert runtime.import_terminal(lease, permit, manifest, layout) == checkpoint
        assert tuple(block.ref_cnt for block in blocks) == tuple(x + 1 for x in before)
        for rank in range(4):
            settled = receipt(permit, rank)
            received.add(settled)
            assert owner.settle(lease, permit, settled) == (rank == 3)
        assert owner.lease_for(lease.identity.request_id) is None
        native_holds = tuple(block.ref_cnt for block in blocks)
        assert all(native_holds)

        scheduler = runtime.scheduler
        clock[0] += idle.idle_ns
        read = directory.acquire_ram_read(checkpoint)
        assert not isinstance(read, Wait)
        assert not tuple(
            directory.expired_verified_checkpoints(
                idle_cutoff_ns=clock[0], limit=1
            )
        )
        directory.release_ram_read(read)
        with pytest.raises(ValueError, match="stale or foreign"):
            directory.release_ram_read(read)

        clock[0] += idle.idle_ns
        store = scheduler.submit("STORE", checkpoint)
        scheduler.tick(now_ns=clock[0])
        assert scheduler.view(store).state == "PENDING" and provider.io_open
        read = directory.acquire_ram_read(checkpoint)
        directory.release_ram_read(read)
        assert _wait(scheduler, store, clock[0]) == "COMMITTED"
        assert provider.resident and provider.retired == 0
        assert tuple(block.ref_cnt for block in blocks) == native_holds
        receipt_metadata = catalog.reopen(scope)
        assert receipt_metadata is not None and receipt_metadata.checkpoint == checkpoint

        clock[0] += idle.idle_ns
        store = scheduler.submit("STORE", checkpoint)
        assert _wait(scheduler, store, clock[0]) == "COMMITTED"
        assert not provider.resident and provider.retired == 1
        assert tuple(block.ref_cnt for block in blocks) == tuple(
            count - 1 for count in native_holds
        )
        assert directory.acquire_ram_read(checkpoint) == Wait("GLM_CHECKPOINT_ON_NVME")

        restore = scheduler.submit("RESTORE", checkpoint)
        scheduler.tick(now_ns=clock[0])
        assert scheduler.view(restore).wait_reason == "RAM_ADMISSION_WAIT"
        assert not provider.io_open
        provider.ram_full = False
        assert _wait(scheduler, restore, clock[0]) == "VERIFIED"
        provider.verify_restored_bytes()
        directory.forget_cold(checkpoint)
        with pytest.raises(ValueError, match="stale"):
            directory.reserve_restore(checkpoint, max_inflight_bytes=BUDGET)
