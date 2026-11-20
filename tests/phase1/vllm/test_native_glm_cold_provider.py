# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import ExitStack
from dataclasses import replace
from hashlib import sha256
from time import monotonic, monotonic_ns, sleep
from types import SimpleNamespace

import pytest
import torch
from kitepond.core.contracts import Wait
from kitepond.core.policy import IdlePolicy, LaneBudget
from kitepond.integrations.vllm import glm_terminal_worker, native_glm_cold_provider
from kitepond.integrations.vllm.glm_cold_catalog import GlmColdCatalog
from kitepond.integrations.vllm.glm_cold_enrollment import NativeGlmColdRegions
from kitepond.integrations.vllm.glm_cold_rank_control import RPC_NAME
from kitepond.integrations.vllm.glm_cold_remote_store import NativeGlmColdStore
from kitepond.integrations.vllm.glm_cold_runtime import (
    ColdDirectoryScope,
    GlmColdDirectoryImport,
    GlmColdRuntime,
)
from kitepond.integrations.vllm.glm_host_backing import (
    _Launch,
    _open_extended_backing,
)
from kitepond.integrations.vllm.glm_nvme_control import (
    GlmNvmeOperator,
    seal_nvme_operation,
)
from kitepond.integrations.vllm.glm_terminal_manifest import (
    GlmTerminalCheckpointVerifier,
    GlmTerminalManifest,
    TerminalRankSigner,
    capture_terminal_rank,
)
from kitepond.integrations.vllm.native_glm_cold_directory import NativeGlmColdDirectory
from kitepond.integrations.vllm.native_glm_cold_provider import (
    GlmNativeColdRankProvider,
    NativeGlmRankMemory,
)
from kitepond.models.glm5next.cold_lifecycle import GlmColdCheckpoint
from kitepond.models.glm5next.cold_restore import ColdRestoreLayout
from kitepond.models.glm5next.frontier import digest
from kitepond.models.glm5next.host_cold_extent import ColdExtentBudget
from kitepond.models.glm5next.import_recipe import part_identity
from kitepond.models.glm5next.layout import CacheRole
from kitepond.models.glm5next.region_socket import join_region
from test_glm_cold_lifecycle import BUDGET, _wait
from test_glm_terminal_manifest import (
    BOUNDARY,
    FORMAT,
    ROLES,
    Fence,
    fenced_source,
    make_runner,
)
from test_stock_terminal_checkpoint import capture as capture
from test_stock_terminal_checkpoint import identity, receipt

from vllm import SamplingParams
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

MIB = 2**20
SCOPE = ColdDirectoryScope("native-fenced-session", 17)


class RankFixture:
    def __init__(self, rank, cleanup, directory):
        budget = ColdExtentBudget(10 * MIB, 2 * MIB, 80 * MIB)
        self.backing, retire = _open_extended_backing(
            _Launch("authority", directory, "D", "D-boot", "e" * 64, budget),
            dict(
                authority_engine_id="D",
                authority_boot_id="D-boot",
                source_rank=rank,
                layout_sha256="e" * 64,
            ),
            budget.layout(8 * MIB),
        )
        cleanup.callback(retire)
        self.memory = NativeGlmRankMemory.from_registered_cold_backing(self.backing)
        assert isinstance(self.memory, NativeGlmRankMemory)
        self.prepared = None
        self.destination = None
        self.verified = None
        self.scratch_open = False
        self.released = []

    def native_views(self, original, *, restore=False):
        offset = 4 * MIB if restore else 0
        layers = []
        for layer in original.layers.values():
            tensor = layer.tensor
            if layer.identity.cache_role is CacheRole.MLA:
                size = tensor.numel()
                view = self.backing[offset : offset + size].view(torch.uint8)
                view = view.view(tensor.shape)
                if not restore:
                    view.fill_(11 + self.memory.rank)
                offset += size
                layer = replace(layer, tensor=view)
            elif not restore:
                tensor.fill_(23 + self.memory.rank)
            layers.append(layer)
        assert offset <= (8 if restore else 4) * MIB
        return ColdRestoreLayout(original.layout, original.rank, layers)

    def aux(self, checkpoint, mode, evidence):
        def require_native():
            # These allocated views outlive the terminal owner's transport pins.
            assert self.backing is not None
            if mode == "RESTORE":
                assert self.destination is not None

        def adopt(lease):
            self.verified = self.destination
            self.restored_lease = lease

        reservation = self.memory.reserve_aux(
            checkpoint,
            mode,
            evidence=evidence,
            require_native=require_native,
            adopt_restore=adopt if mode == "RESTORE" else None,
        )
        assert not isinstance(reservation, Wait)

        def settle(state):
            reservation.settle(state)
            self.released.append(state)

        return replace(reservation, settle=settle)

    def retain_terminal(self, entry):
        return self.prepared or Wait("NATIVE_AUX_STATE_NOT_RESIDENT")

    def scratch(self, checkpoint, _mode):
        assert not self.scratch_open
        stage = self.memory.reserve_scratch(checkpoint)
        assert not isinstance(stage, Wait)
        self.scratch_open = True

        def release():
            assert self.scratch_open
            stage.release()
            self.scratch_open = False

        return replace(stage, release=release)

    def reserve_restore(self, checkpoint, verified):
        if self.destination is None:
            return Wait("RAM_ADMISSION_WAIT")

        self.destination = self.memory.bind_restore(
            checkpoint,
            self.destination,
            self.aux(checkpoint, "RESTORE", verified),
        )
        return self.destination


def _digest_binding(region, binding):
    h = sha256()
    for span in binding.spans:
        h.update(region.mapping[span.offset : span.offset + span.nbytes])
    return h.hexdigest()


def _remote_provider(
    rank_fixtures, entry, sources, keys, directory, cleanup, monkeypatch, clock
):
    borrowed, workers, signers, commands = [], [], [], []
    monkeypatch.setattr(
        glm_terminal_worker, "time", SimpleNamespace(monotonic_ns=clock)
    )
    monkeypatch.setattr(native_glm_cold_provider, "monotonic_ns", clock)
    for rank, fixture in enumerate(rank_fixtures):
        worker = object.__new__(glm_terminal_worker.GlmTerminalWorker)
        signer = TerminalRankSigner(rank, keys[rank])
        worker._require_terminal_source = lambda signer=signer: (signer, None)
        worker._kitepond_native_cold_memory = fixture.memory
        descriptor = fixture.memory.region.descriptor
        worker._kitepond_native_state_views = SimpleNamespace(
            rank=rank, descriptor=descriptor
        )
        worker._kitepond_terminal_frames = {
            entry.lease.identity.request_id: SimpleNamespace(
                lease=entry.lease,
                source=sources[rank],
                checkpoint=entry.manifest.ranks[rank],
            )
        }
        worker.model_runner = SimpleNamespace(
            kv_cache_config=SimpleNamespace(
                hisparse_host_num_blocks=1,
                hisparse_host_block_stride=fixture.memory.mla_bytes,
            )
        )
        region = join_region(
            directory,
            authority_engine_id="D",
            authority_boot_id="D-boot",
            source_rank=rank,
            layout_sha256=descriptor.layout_sha256,
        )
        cleanup.callback(region.close)
        borrowed.append(region)
        workers.append(worker)
        signers.append(signer)
    fixtures_layout = rank_fixtures[0].memory._extent.layout
    regions = NativeGlmColdRegions(
        fixtures_layout,
        tuple(borrowed),
        borrowed[0].descriptor.layout_sha256,
    )
    assert all(len(region.mapping) == fixtures_layout.extent for region in borrowed)

    def call(method, timeout, args):
        assert method == RPC_NAME and timeout == 5.0
        command, signatures = args
        assert command.boot_id == "D-boot"
        commands.append(command.verb)
        return [
            worker.kitepond_control_cold_rank(command, signatures) for worker in workers
        ]

    return NativeGlmColdStore(regions, call, tuple(signers)), workers, commands


@pytest.mark.parametrize(
    "variant",
    ("local", "remote", "archive"),
    ids=("full-roundtrip", "worker-store", "operator-archive"),
)
def test_verified_rank_views_bind_in_place_and_roundtrip_through_fsl2(
    tmp_path,
    capture,  # noqa: F811
    monkeypatch,
    variant,
):
    from kitepond.integrations.vllm.glm_terminal_manifest import FencedTerminalSource

    archive = variant == "archive"
    remote = variant != "local"
    clock = [monotonic_ns() if archive else 25]
    now = monotonic_ns if archive else lambda: clock[0]
    sockets = tmp_path / "socket"
    sockets.mkdir(mode=0o700)
    with ExitStack() as cleanup:
        ranks = tuple(RankFixture(rank, cleanup, sockets) for rank in range(4))
        keys = {rank: bytes([rank + 1]) * 32 for rank in range(4)}
        hashes = {rank: value.memory.region_sha256 for rank, value in enumerate(ranks)}
        verifier = GlmTerminalCheckpointVerifier(
            keys,
            hashes,
            lambda _: None,
            ROLES,
            representation_sha256=FORMAT,
            clock=lambda: clock[0],
        )
        scheduler, owner, scheduler_clock, attested = capture(
            host_layers=("mla.target", "mla.draft"),
            device_layers=(
                "idx.target",
                "idx.draft",
                "tail.target",
                "tail.draft",
                "kda",
            ),
            verify_native_checkpoint=verifier.verify,
        )
        scheduler_clock[0] = clock[0]
        tokens = [index % 63 + 1 for index in range(BOUNDARY)] + [33]
        request = Request(
            request_id="native-byte-owner",
            prompt_token_ids=tokens[:-1],
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
        )
        scheduler.add_request(request)
        terminal_identity = identity(request.request_id)
        if archive:
            terminal_identity = replace(
                terminal_identity, deadline_monotonic_ns=now() + 120_000_000_000
            )
        owner.arm(request, terminal_identity)
        step = scheduler.schedule()
        scheduler.update_from_output(
            step,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[tokens[-1]]],
            ),
        )
        lease = owner.lease_for(request.request_id)
        assert lease is not None
        sources, signed = [], []
        block_ids = next(g.block_ids for g in lease.groups if "kda" in g.layer_names)
        for rank, value in enumerate(ranks):
            original = fenced_source(lease, rank, hashes[rank])
            source = replace(original, layout=value.native_views(original.layout))
            sources.append(source)
            runner = make_runner(lease, tokens, Fence(), block_ids[:4])
            signed.append(
                capture_terminal_rank(
                    runner,
                    lease,
                    32,
                    TerminalRankSigner(rank, keys[rank]),
                    lambda *_, source=source: source,
                    clock=lambda: clock[0],
                )
            )
        manifest = GlmTerminalManifest(lease.sha256, tuple(signed))
        permit = None if archive else owner.begin_reads(lease, manifest)
        layout = sources[0].layout.layout
        checkpoint = GlmColdCheckpoint(
            SCOPE.namespace,
            SCOPE.generation,
            layout,
            manifest.sha256,
            tuple(proof.checkpoint.plan for proof in signed),
        )
        entry = GlmColdDirectoryImport(checkpoint, lease, permit, manifest)
        root = tmp_path / "cold"
        catalog = GlmColdCatalog(
            root,
            engine_id="D",
            boot_id="D-boot",
            model_fingerprint=layout.model_fingerprint,
            config_fingerprint=layout.config_fingerprint,
            signing_key=bytes(range(32)),
        )
        if archive:
            target = []

            def retain(actual_entry):
                target[:] = _remote_provider(
                    ranks,
                    actual_entry,
                    sources,
                    keys,
                    sockets,
                    cleanup,
                    monkeypatch,
                    now,
                )
                return target[0].retain_terminal(actual_entry)

            provider = SimpleNamespace(
                region_fingerprints=tuple(hashes.values()),
                retain_terminal=retain,
                snapshot=lambda *args, **kwargs: target[0].snapshot(*args, **kwargs),
            )
        elif remote:
            provider, workers, commands = _remote_provider(
                ranks,
                entry,
                sources,
                keys,
                sockets,
                cleanup,
                monkeypatch,
                lambda: clock[0],
            )
        else:
            provider = GlmNativeColdRankProvider(ranks)
        directory = NativeGlmColdDirectory(
            owner,
            provider,
            lambda _: SCOPE,
            max_entries=8,
            clock=now,
            catalog=catalog,
        )
        callbacks = []
        idle = IdlePolicy(15)
        lane = LaneBudget(1, BUDGET, BUDGET)
        runtime = GlmColdRuntime(
            root,
            directory,
            owner,
            register_quiescent=lambda **k: callbacks.append(k["callback"]),
            cadence_ns=20_000_000,
            idle=idle,
            foreground=lane,
            background=lane,
            clock=now,
        )
        if archive:
            operator_key = bytes([31] * 32)
            scheduler._cold_decode_owner = SimpleNamespace(
                _require_scheduler_owner=owner._require_owner
            )
            scheduler.native_glm_cold_store = runtime
            operator = GlmNvmeOperator(
                scheduler,
                SimpleNamespace(engine_id="D", boot_id="D-boot"),
                None,
                operator_key,
                max_pending=2,
            )
            owner.bind_read_observer(operator.on_terminal_read)
            base = dict(engine_id="D", boot_id="D-boot", request_id=request.request_id)

            def operation(name, **fields):
                return operator.handle(
                    name, seal_nvme_operation(name, {**base, **fields}, operator_key)
                )

            assert operation(
                "archive_arm",
                namespace=SCOPE.namespace,
                generation=SCOPE.generation,
                transfer_id=terminal_identity.transfer_id,
                epoch=terminal_identity.epoch,
                deadline_ns=now() + 60_000_000_000,
            ) == {"state": "armed"}
            permit = owner.begin_reads(lease, manifest)
            status = operation("status")
            assert not status["reason"], status
            assert status["state"] == "captured"
            assert status["boundary"] == BOUNDARY
            assert status["prefix_sha256"] == digest(tokens[:BOUNDARY])
            assert operation("migrate") == {
                "state": "wait",
                "reason": "GLM_LIVE_REFERENCE_WAIT",
            }
            for rank in range(4):
                settled = receipt(permit, rank)
                attested.add(settled)
                assert owner.settle(lease, permit, settled) == (rank == 3)
            read = directory.acquire_ram_read(checkpoint)
            assert not isinstance(read, Wait)
            assert operation("migrate") == {
                "state": "wait",
                "reason": "GLM_LIVE_REFERENCE_WAIT",
            }
            directory.release_ram_read(read)
            assert operation("migrate")["state"] == "demoting"
            [tick] = callbacks
            deadline = monotonic() + 10
            while monotonic() < deadline:
                tick()
                if operation("status")["job"]["state"] == "COMMITTED":
                    break
                sleep(0.002)
            else:
                pytest.fail(
                    "Operator demote did not reach durable publication: "
                    f"{operation('status')}"
                )
            candidate = catalog.reopen(SCOPE)
            assert candidate is not None and candidate.checkpoint == checkpoint
            assert candidate.accepted_prefix_sha256 == digest(tokens[:BOUNDARY])
            _, workers, commands = target
            assert commands == ["retain", "stage", "unstage", "release"]
            assert all(
                not worker._kitepond_native_cold_leases._retained for worker in workers
            )
            return
        if not remote:
            assert runtime.import_terminal(lease, permit, manifest, layout) == Wait(
                "NATIVE_AUX_STATE_NOT_RESIDENT"
            )
        assert not root.exists()

        expect = tuple(
            {
                part: sha256(source.read_chunk(part).numpy()).hexdigest()
                for part in checkpoint.plans[rank].parts
            }
            for rank, source in enumerate(sources)
        )
        original_read = FencedTerminalSource.read_chunk

        def auxiliary_only(source, part):
            assert part_identity(part.role).cache_role is not CacheRole.MLA
            return original_read(source, part)

        monkeypatch.setattr(FencedTerminalSource, "read_chunk", auxiliary_only)

        def require_source():
            owner.require_read_open(lease, permit)

        with pytest.raises(ValueError, match="signed terminal"):
            ranks[0].memory.prepare_terminal(
                entry,
                sources[1],
                ranks[0].aux(checkpoint, "STORE", entry),
                require_source=require_source,
            )
        forbidden = ranks[0].aux(checkpoint, "STORE", entry)
        original_first, *rest = forbidden.bindings
        (span,) = original_first.spans
        forbidden = replace(
            forbidden,
            bindings=(replace(original_first, spans=(replace(span, offset=0),)), *rest),
        )
        with pytest.raises(ValueError, match="escapes its native extent"):
            ranks[0].memory.prepare_terminal(
                entry,
                sources[0],
                forbidden,
                require_source=require_source,
            )
        if not remote:
            for rank, value in enumerate(ranks):
                value.prepared = value.memory.prepare_terminal(
                    entry,
                    sources[rank],
                    value.aux(checkpoint, "STORE", entry),
                    require_source=require_source,
                )
                for binding in value.prepared.bindings:
                    got = _digest_binding(value.memory.region, binding)
                    assert got == expect[rank][binding.part]
        admitted = runtime.import_terminal(lease, permit, manifest, layout)
        assert admitted == checkpoint
        for rank in range(4):
            settled = receipt(permit, rank)
            attested.add(settled)
            assert owner.settle(lease, permit, settled) == (rank == 3)
        assert owner.lease_for(request.request_id) is None

        clock[0] += idle.idle_ns
        [tick] = callbacks
        deadline = monotonic() + 10
        while monotonic() < deadline:
            clock[0] += 20_000_000
            states = tick()
            if states:
                break
            sleep(0.002)
        else:
            pytest.fail("Native rank bindings did not reach durable cold publication")
        assert [state.state for state in states] == ["COMMITTED"]
        candidate = catalog.reopen(SCOPE)
        assert candidate is not None
        if remote:
            assert commands == ["retain", "stage", "unstage", "release"]
            assert all(
                not worker._kitepond_native_cold_leases._retained for worker in workers
            )
            assert all(candidate.checksums(rank) == expect[rank] for rank in range(4))
            assert provider.reserve_restore(checkpoint) == Wait(
                "NATIVE_COLD_RESTORE_FENCE_UNAVAILABLE"
            )
            return
        assert all(
            not value.scratch_open and value.released[-1] == "RELEASED"
            for value in ranks
        )
        assert directory.register_verified_cold(request, candidate) == admitted

        restore = runtime.scheduler.submit("RESTORE", admitted)
        runtime.scheduler.tick(now_ns=clock[0])
        assert runtime.scheduler.view(restore).wait_reason == "RAM_ADMISSION_WAIT"
        for rank, value in enumerate(ranks):
            value.destination = value.native_views(sources[rank].layout, restore=True)
        assert _wait(runtime.scheduler, restore, clock[0]) == "VERIFIED"
        for rank, value in enumerate(ranks):
            assert value.verified is not None and not value.scratch_open
            for binding in value.verified.bindings:
                got = _digest_binding(value.memory.region, binding)
                assert got == expect[rank][binding.part]
            value.memory.release_restored_aux(value.restored_lease)
