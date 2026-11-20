# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import ExitStack
from dataclasses import replace
from hashlib import sha256
from threading import Event
from time import monotonic, monotonic_ns, sleep

import pytest
from kitepond.backends.fsl2 import BundlePlan, ColdStorageAdapter
from kitepond.backends.fsl2_aggregate import FourRankManifestStore
from kitepond.core.contracts import Wait
from kitepond.core.policy import ColdJobScheduler, IdlePolicy, LaneBudget
from kitepond.integrations.vllm.lifecycle_control import LifecycleControl
from kitepond.models.glm5next.cold_authority import NativeRegionColdAuthority
from kitepond.models.glm5next.cold_lifecycle import (
    GlmColdCheckpoint,
    GlmColdRamGrant,
    GlmFSL2ColdAdapter,
)
from kitepond.models.glm5next.idle_owner_loop import GlmIdleColdLoop
from kitepond.models.glm5next.import_recipe import ByteSpan, ColdBinding
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion
from test_glm_cold_restore import _cache

STAGE_BYTES = 2**21
RAM_LAYOUT = "d" * 64
BUDGET = 4 * STAGE_BYTES + 65536


class Owner:
    """Exercise real lease-bound spans in the same memfds across a cold cycle."""

    def __init__(
        self, cleanup, checkpoint=None, *, boot_id="same-live-process", seed_source=True
    ):
        if checkpoint is None:
            caches = tuple(_cache(width=656, rank=rank) for rank in range(4))
            plans = tuple(
                BundlePlan(
                    "glm-native-state-v1",
                    "e" * 64,
                    tuple(dict.fromkeys(part.role for part in parts)),
                    parts,
                )
                for parts in (cache.manifest() for cache in caches)
            )
            checkpoint = GlmColdCheckpoint(
                "scope", 42, caches[0].layout, "a" * 64, plans
            )
        self.checkpoint = checkpoint
        self.regions = tuple(
            NativeHiSparseMemfdRegion.create(
                STAGE_BYTES * 3,
                authority_engine_id="D",
                authority_boot_id=boot_id,
                source_rank=rank,
                layout_sha256=RAM_LAYOUT,
            )
            for rank in range(4)
        )
        self.source = []
        self.destination = []
        self.checksums = []
        for rank, (region, plan) in enumerate(
            zip(self.regions, checkpoint.plans, strict=True)
        ):
            cleanup.callback(region.close)
            source, destination, checksums = [], [], {}
            offset = 0
            for ordinal, part in enumerate(plan.parts, 1):
                payload = bytes([rank * 31 + ordinal]) * part.nbytes
                if seed_source:
                    region.mapping[offset : offset + part.nbytes] = payload
                checksums[part] = sha256(payload).hexdigest()
                source.append(
                    ColdBinding(part, (ByteSpan(rank, ordinal, offset, part.nbytes),))
                )
                destination.append(
                    ColdBinding(
                        part,
                        (
                            ByteSpan(
                                rank, ordinal + 100, offset + STAGE_BYTES, part.nbytes
                            ),
                        ),
                    )
                )
                offset += part.nbytes
            assert offset <= STAGE_BYTES
            self.source.append(tuple(source))
            self.destination.append(tuple(destination))
            self.checksums.append(checksums)
        self.last_use = 1
        self.readers = 0
        self.resident = True
        self.ram_full = False
        self.held = None
        self.receipts = []

    def last_business_use_ns(self, checkpoint):
        return self.last_use

    def _grant(self, checkpoint, mode):
        assert self.held is None
        token = object()
        self.held = token
        observed = self.last_use
        captures = mode == "STORE"

        def live():
            assert self.held is token

        authorities = tuple(
            NativeRegionColdAuthority(
                region=region,
                identity=checkpoint.identity,
                plan=plan,
                bindings=bindings,
                stage=ByteSpan(rank, 201, STAGE_BYTES * 2, STAGE_BYTES),
                verify_reference=live,
                mode=mode,
                checksums=None if captures else self.checksums[rank],
            )
            for rank, (region, plan, bindings) in enumerate(
                zip(
                    self.regions,
                    checkpoint.plans,
                    self.source if captures else self.destination,
                    strict=True,
                )
            )
        )

        def settle(state):
            live()
            assert all(not authority._active for authority in authorities)
            self.receipts.append(state)
            if state == "COMMITTED" and observed == self.last_use and not self.readers:
                # Release the source slots while keeping their pool mappings alive.
                for region in self.regions:
                    region.mapping[:STAGE_BYTES] = bytes(STAGE_BYTES)
                self.resident = False
            if state == "VERIFIED":
                self.resident = True
            self.held = None

        return GlmColdRamGrant(checkpoint, authorities, RAM_LAYOUT, settle)

    def acquire_snapshot(
        self, checkpoint, *, observed_last_use_ns, idle_cutoff_ns, max_inflight_bytes
    ):
        assert max_inflight_bytes == BUDGET
        if (
            checkpoint is not self.checkpoint
            or self.readers
            or not self.resident
            or not observed_last_use_ns == self.last_use <= idle_cutoff_ns
        ):
            return Wait("GLM_LIVE_REFERENCE_WAIT")
        return self._grant(checkpoint, "STORE")

    def reserve_restore(self, checkpoint, *, max_inflight_bytes):
        assert max_inflight_bytes == BUDGET
        if self.ram_full:
            return Wait("RAM_ADMISSION_WAIT")
        assert not self.resident and not self.readers
        return self._grant(checkpoint, "RESTORE")


def _scheduler(root, owner):
    budget = LaneBudget(1, BUDGET, BUDGET)
    return ColdJobScheduler(
        GlmFSL2ColdAdapter(root, owner),
        idle=IdlePolicy(15),
        foreground=budget,
        background=budget,
    )


def _wait(scheduler, job, now_ns):
    deadline = monotonic() + 10
    while monotonic() < deadline:
        scheduler.tick(now_ns=now_ns)
        state = scheduler.view(job).state
        if state not in {"QUEUED", "PENDING"}:
            return scheduler.reap(job).state
        sleep(0.002)
    pytest.fail(f"Cold job did not settle: {scheduler.view(job)}")


def _store(scheduler, checkpoint, now_ns):
    return _wait(scheduler, scheduler.submit("STORE", checkpoint), now_ns)


def test_idle_demote_reserve_and_restore_in_the_same_live_pool(tmp_path):
    with ExitStack() as cleanup:
        owner = Owner(cleanup)
        scheduler = _scheduler(tmp_path, owner)
        now = owner.last_use + scheduler.idle.idle_ns
        store = scheduler.submit("STORE", owner.checkpoint)
        scheduler.tick(now_ns=now - 1)
        assert scheduler.view(store).wait_reason == "BUSINESS_IDLE_WAIT"

        owner.readers = 1
        scheduler.tick(now_ns=now)
        assert scheduler.view(store).wait_reason == "GLM_LIVE_REFERENCE_WAIT"
        assert owner.held is None and owner.resident
        owner.readers = 0
        scheduler.tick(now_ns=now)
        assert scheduler.view(store).state == "PENDING"
        assert owner.held is not None and owner.resident
        assert not list((tmp_path / "aggregates").glob("*.json"))
        assert _wait(scheduler, store, now) == "COMMITTED"
        assert not owner.resident and owner.held is None
        assert len(list((tmp_path / "aggregates").glob("*.json"))) == 1
        for rank, region in enumerate(owner.regions):
            manifests = tmp_path / f"rank-{rank}" / "manifests"
            assert len(list(manifests.glob("*.json"))) == 1
            assert region.mapping[:STAGE_BYTES] == bytes(STAGE_BYTES)

        restore = scheduler.submit("RESTORE", owner.checkpoint)
        owner.ram_full = True
        scheduler.tick(now_ns=now)
        assert scheduler.view(restore).wait_reason == "RAM_ADMISSION_WAIT"
        assert owner.held is None and not scheduler.view(restore).native_started
        owner.ram_full = False
        assert _wait(scheduler, restore, now) == "VERIFIED"
        assert owner.held is None and owner.resident
        assert owner.receipts == ["COMMITTED", "VERIFIED"]
        pairs = zip(owner.regions, owner.destination, strict=True)
        for rank, (region, bindings) in enumerate(pairs):
            assert region.descriptor.authority_boot_id == "same-live-process"
            for binding in bindings:
                results = [
                    region.mapping[span.offset : span.offset + span.nbytes]
                    for span in binding.spans
                ]
                assert (
                    sha256(b"".join(results)).hexdigest()
                    == owner.checksums[rank][binding.part]
                )


def test_retry_reuses_only_complete_ranks_and_defers_ram_release(tmp_path, monkeypatch):
    with ExitStack() as cleanup:
        owner = Owner(cleanup)
        scheduler = _scheduler(tmp_path, owner)
        checkpoint = owner.checkpoint
        now = owner.last_use + scheduler.idle.idle_ns
        begin = ColdStorageAdapter.begin_store

        def fail_second_rank(adapter, *args):
            if adapter.root.name == "rank-1":
                raise OSError("injected rank submission failure")
            return begin(adapter, *args)

        monkeypatch.setattr(ColdStorageAdapter, "begin_store", fail_second_rank)
        assert _store(scheduler, checkpoint, now) == "FAILED"
        assert owner.resident and owner.held is None
        assert not list((tmp_path / "aggregates").glob("*.json"))
        assert list((tmp_path / "rank-0" / "manifests").glob("*.json"))
        monkeypatch.setattr(ColdStorageAdapter, "begin_store", begin)
        assert _store(scheduler, checkpoint, now) == "COMMITTED"
        assert owner.receipts == ["FAILED", "COMMITTED"]
        assert not owner.resident and owner.held is None


def test_epoch_layout_and_rank_plan_mismatch_start_no_rank_reads(tmp_path, monkeypatch):
    with ExitStack() as cleanup:
        owner = Owner(cleanup)
        scheduler = _scheduler(tmp_path, owner)
        checkpoint = owner.checkpoint
        now = owner.last_use + scheduler.idle.idle_ns
        assert _store(scheduler, checkpoint, now) == "COMMITTED"

        def no_rank_read(*args, **kwargs):
            pytest.fail("An incompatible cold descriptor reached physical FSL2 reads")

        monkeypatch.setattr(ColdStorageAdapter, "begin_restore", no_rank_read)
        for incompatible in (
            replace(checkpoint, generation=checkpoint.generation + 1),
            replace(checkpoint, layout=replace(checkpoint.layout, epoch=9)),
            replace(
                checkpoint, layout=replace(checkpoint.layout, committed_kda_slot=1)
            ),
            replace(
                checkpoint,
                plans=tuple(
                    replace(plan, representation_schema="other")
                    for plan in checkpoint.plans
                ),
            ),
        ):
            restore = scheduler.submit("RESTORE", incompatible)
            assert _wait(scheduler, restore, now) == "MISS"
            assert owner.held is None and not owner.resident


def test_restore_rank_digest_is_rechecked_before_io(tmp_path, monkeypatch):
    with ExitStack() as cleanup:
        owner = Owner(cleanup)
        scheduler = _scheduler(tmp_path, owner)
        checkpoint = owner.checkpoint
        now = owner.last_use + scheduler.idle.idle_ns
        assert _store(scheduler, checkpoint, now) == "COMMITTED"
        begin = ColdStorageAdapter.begin_restore

        def mismatch(adapter, *args, expected_manifest_digest):
            assert expected_manifest_digest is not None
            result = begin(adapter, *args, expected_manifest_digest="0" * 64)
            assert isinstance(result, Wait) and adapter.ticket is None
            return result

        monkeypatch.setattr(ColdStorageAdapter, "begin_restore", mismatch)
        assert _wait(scheduler, scheduler.submit("RESTORE", checkpoint), now) == "MISS"
        assert not owner.resident and owner.held is None


def test_cancel_keeps_ram_until_aggregate_finishes(tmp_path, monkeypatch):
    entered, release = Event(), Event()
    publish = FourRankManifestStore.publish

    def blocked_publish(store, *args):
        entered.set()
        assert release.wait(10)
        return publish(store, *args)

    monkeypatch.setattr(FourRankManifestStore, "publish", blocked_publish)
    with ExitStack() as cleanup:
        owner = Owner(cleanup)
        scheduler = _scheduler(tmp_path, owner)
        now = owner.last_use + scheduler.idle.idle_ns
        store = scheduler.submit("STORE", owner.checkpoint)
        try:
            deadline = monotonic() + 10
            while not entered.is_set() and monotonic() < deadline:
                scheduler.tick(now_ns=now)
                sleep(0.002)
            assert entered.is_set()
            assert scheduler.cancel(store).state == "PENDING"
            assert owner.held is not None and owner.resident
        finally:
            release.set()
        assert _wait(scheduler, store, now) == "CANCELLED"
        assert owner.held is None and owner.resident


class ScanOwner(Owner):
    def __init__(self, cleanup, checkpoint=None):
        super().__init__(cleanup, checkpoint)
        self.scans = []
        self.acquired = 0

    def expired_verified_checkpoints(self, *, idle_cutoff_ns, limit):
        self.scans.append((idle_cutoff_ns, limit))
        if self.resident and self.last_use <= idle_cutoff_ns:
            # A repeated scan can return an equal descriptor object.
            yield self.checkpoint
            yield replace(self.checkpoint)

    def acquire_snapshot(self, checkpoint, **kwargs):
        result = super().acquire_snapshot(checkpoint, **kwargs)
        self.acquired += not isinstance(result, Wait)
        return result


def _idle_loop(root, owner, clock, *, max_jobs=128):
    return GlmIdleColdLoop(
        root,
        owner,
        idle=IdlePolicy(15),
        foreground=LaneBudget(1, BUDGET, BUDGET),
        background=LaneBudget(2, 2 * BUDGET, BUDGET),
        max_jobs=max_jobs,
        scan_limit=2,
        clock=lambda: clock[0],
    )


def _loop_until(loop, *, predicate=None):
    deadline = monotonic() + 10
    while monotonic() < deadline:
        settled = loop.tick()
        if (predicate is None and settled) or (predicate and predicate()):
            return settled
        sleep(0.002)
    pytest.fail("Native idle loop did not reach the expected state")


def test_owner_tick_demotes_an_expired_verified_checkpoint_once(tmp_path):
    with ExitStack() as cleanup:
        owner = ScanOwner(cleanup)
        clock = [owner.last_use]
        loop = _idle_loop(tmp_path, owner, clock)
        clock[0] += loop.scheduler.idle.idle_ns - 1
        assert not loop.tick()
        assert owner.acquired == 0 and owner.resident

        clock[0] += 1
        control = LifecycleControl(engine_id="D", boot_id="boot", cold=loop.scheduler)
        health = control.handle(
            "health",
            {
                "schema": "kitepond.lifecycle-query.v1",
                "engine_id": "D",
                "boot_id": "boot",
            },
        )
        assert health["observations"]["cold"]["STORE"]["active_jobs"] == 0
        assert owner.acquired == 0

        assert not loop.tick()
        assert owner.acquired == 1 and owner.held is not None and owner.resident
        [settled] = _loop_until(loop)
        assert settled.state == "COMMITTED"
        assert owner.acquired == 1 and owner.held is None and not owner.resident
        assert not loop.tick()
        assert loop.scheduler.observe_lane("STORE").metadata_entries == 0


def test_owner_tick_allows_new_business_use_while_cold_publication_drains(
    tmp_path, monkeypatch
):
    entered, release = Event(), Event()
    publish = FourRankManifestStore.publish

    def paused_publish(store, *args):
        entered.set()
        assert release.wait(10)
        return publish(store, *args)

    monkeypatch.setattr(FourRankManifestStore, "publish", paused_publish)
    with ExitStack() as cleanup:
        owner = ScanOwner(cleanup)
        clock = [owner.last_use]
        loop = _idle_loop(tmp_path, owner, clock, max_jobs=2)
        clock[0] += loop.scheduler.idle.idle_ns
        try:
            _loop_until(loop, predicate=entered.is_set)
            assert loop.scheduler.observe_lane("STORE").metadata_entries == 1
            assert owner.held is not None and owner.resident
            owner.readers = 1
            owner.last_use = clock[0] + 1
            clock[0] += 1
        finally:
            release.set()

        [settled] = _loop_until(loop)
        assert settled.state == "COMMITTED"
        assert owner.acquired == 1 and owner.held is None and owner.resident
        owner.readers = 0
        assert not loop.tick()
        assert owner.acquired == 1


def test_idle_scan_preserves_foreground_job_metadata(tmp_path):
    with ExitStack() as cleanup:
        owner = ScanOwner(cleanup)
        clock = [owner.last_use]
        loop = _idle_loop(tmp_path, owner, clock, max_jobs=1)
        clock[0] += loop.scheduler.idle.idle_ns
        assert not loop.tick()
        assert not owner.scans
        owner.ram_full = True
        restore = loop.scheduler.submit("RESTORE", owner.checkpoint)
        assert not loop.tick()
        assert loop.scheduler.view(restore).wait_reason == "RAM_ADMISSION_WAIT"


def test_explicit_idle_demote_preserves_real_clock_and_waits_for_reader(tmp_path):
    with ExitStack() as cleanup:
        owner = Owner(cleanup)
        owner.last_use = monotonic_ns()
        scheduler = _scheduler(tmp_path, owner)
        natural = scheduler.submit("STORE", owner.checkpoint)
        scheduler.tick()
        assert scheduler.view(natural).wait_reason == "BUSINESS_IDLE_WAIT"
        scheduler.cancel(natural)
        scheduler.reap(natural)
        job = scheduler.submit("STORE", owner.checkpoint, operator_idle=True)
        owner.readers = 1
        scheduler.tick()
        assert scheduler.view(job).wait_reason == "GLM_LIVE_REFERENCE_WAIT"
        assert owner.held is None and not (tmp_path / "aggregates").exists()

        owner.readers = 0
        deadline = monotonic() + 8
        while scheduler.view(job).state in {"QUEUED", "PENDING"}:
            assert monotonic() < deadline
            scheduler.tick()
            sleep(0.002)
        assert scheduler.reap(job).state == "COMMITTED"
        assert (
            not owner.resident
            and len(list((tmp_path / "aggregates").glob("*.json"))) == 1
        )
        assert monotonic_ns() - owner.last_use < scheduler.idle.idle_ns
