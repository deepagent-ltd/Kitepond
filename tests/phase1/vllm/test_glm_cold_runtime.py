# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import ExitStack
from dataclasses import replace
from time import monotonic, sleep

import pytest
from kitepond.core.contracts import Wait
from kitepond.core.policy import IdlePolicy, LaneBudget
from kitepond.integrations.vllm.glm_cold_runtime import (
    ColdDirectoryScope,
    GlmColdDirectoryImport,
    GlmColdRuntime,
)
from kitepond.models.glm5next.cold_lifecycle import GlmColdCheckpoint
from test_glm_cold_lifecycle import BUDGET, ScanOwner
from test_glm_terminal_manifest import active_manifest, capture, fenced_source
from test_stock_terminal_checkpoint import receipt


class ServiceDirectory(ScanOwner):
    """Fixture directory mixing signed native host pins and real FSL2 RAM spans."""

    engine_id = "D"
    boot_id = "D-boot"

    def __init__(self, cleanup, scheduler, lease, manifest, layout, clock):
        self._scheduler = scheduler
        self._native_lease = lease
        self._clock = clock
        self._scope = ColdDirectoryScope("service-session-7", 12)
        staged = GlmColdCheckpoint(
            self._scope.namespace,
            self._scope.generation,
            layout,
            manifest.sha256,
            tuple(rank.checkpoint.plan for rank in manifest.ranks),
        )
        super().__init__(cleanup, staged)
        self.resident = False
        self.pins = []
        self.accepted = None

    def cold_scope(self, lease):
        assert lease is self._native_lease
        return self._scope

    def import_accepted_terminal(self, entry: GlmColdDirectoryImport):
        assert type(entry) is GlmColdDirectoryImport
        assert entry.lease is self._native_lease
        assert entry.checkpoint == self.checkpoint
        if self.accepted is not None:
            assert entry.checkpoint == self.accepted.checkpoint
            return None
        native = self._scheduler.kv_cache_manager.coordinator.single_type_managers
        held = {}
        for group in entry.lease.groups:
            if group.location != "host":
                continue
            pool = native[group.group_id].block_pool
            for page in group.block_ids:
                block = pool.blocks[page]
                if not block.is_null:
                    assert block.ref_cnt > 0
                    held[id(pool), page] = (pool, block)
        assert held
        for pool, block in held.values():
            pool.touch((block,))
            self.pins.append((pool, block))
        self.accepted = entry
        self.checkpoint = entry.checkpoint
        self.last_use = self._clock[0]
        self.resident = True
        return None

    def acquire_snapshot(self, checkpoint, **kwargs):
        grant = super().acquire_snapshot(checkpoint, **kwargs)
        if isinstance(grant, Wait):
            return grant

        def settle(state):
            grant.settle(state)
            if state == "COMMITTED" and not self.resident:
                for pool, block in self.pins:
                    pool.free_blocks((block,))
                self.pins.clear()

        return replace(grant, settle=settle)


class EngineIdle:
    def __init__(self):
        self.handler = None
        self.interval_ns = None

    def register(self, *, interval_ns, callback):
        assert self.handler is None
        self.interval_ns, self.handler = interval_ns, callback

    def quiescent(self):
        return self.handler()


def test_verified_terminal_uses_the_native_directory_and_engine_idle_cadence(
    tmp_path, active_manifest
):
    scheduler, owner, lease, manifest, received, _, complete, _ = active_manifest
    complete[0] = True
    permit = owner.begin_reads(lease, manifest)
    layout = fenced_source(lease, 0, "1" * 64).layout.layout
    clock = [25]
    root = tmp_path / "cold"
    engine = EngineIdle()
    idle = IdlePolicy(15)
    lane = LaneBudget(1, BUDGET, BUDGET)
    cadence = 10

    with ExitStack() as cleanup:
        directory = ServiceDirectory(
            cleanup, scheduler, lease, manifest, layout, clock
        )
        runtime = GlmColdRuntime(
            root,
            directory,
            owner,
            register_quiescent=engine.register,
            cadence_ns=cadence,
            idle=idle,
            foreground=lane,
            background=lane,
            clock=lambda: clock[0],
        )
        assert engine.interval_ns == cadence
        with pytest.raises(ValueError, match="cannot open"):
            runtime.import_terminal(lease, replace(permit), manifest, layout)
        wrong = replace(manifest, ranks=tuple(reversed(manifest.ranks)))
        with pytest.raises(ValueError, match="changed its accepted"):
            runtime.import_terminal(lease, permit, wrong, layout)
        with pytest.raises(ValueError, match="changed its accepted"):
            runtime.import_terminal(
                lease, permit, manifest, replace(layout, epoch=layout.epoch + 1)
            )
        assert directory.accepted is None and not root.exists()

        imported = runtime.import_terminal(lease, permit, manifest, layout)
        assert imported.namespace == "service-session-7" and imported.generation == 12
        assert not root.exists() and directory.resident and directory.pins
        assert runtime.import_terminal(lease, permit, manifest, layout) == imported
        assert not root.exists()

        for rank in range(4):
            settled = receipt(permit, rank)
            received.add(settled)
            assert owner.settle(lease, permit, settled) == (rank == 3)
        assert owner.lease_for(lease.identity.request_id) is None
        assert all(block.ref_cnt > 0 for _, block in directory.pins)
        with pytest.raises(ValueError, match="stale"):
            runtime.import_terminal(lease, permit, manifest, layout)

        clock[0] = directory.last_use + idle.idle_ns - 1
        assert not engine.quiescent() and not root.exists()
        clock[0] += 1
        assert not engine.quiescent() and not root.exists()
        clock[0] += cadence - 1
        assert not engine.quiescent() and root.exists()
        assert directory.acquired == 1 and directory.resident and directory.pins
        assert not engine.quiescent() and directory.acquired == 1

        deadline = monotonic() + 10
        while monotonic() < deadline:
            clock[0] += cadence
            result = engine.quiescent()
            if result:
                break
            sleep(0.002)
        else:
            pytest.fail("Native cold runtime did not settle")
        [done] = result
        assert done.state == "COMMITTED"
        assert not directory.resident and not directory.pins
        assert len(list((root / "aggregates").glob("*.json"))) == 1
