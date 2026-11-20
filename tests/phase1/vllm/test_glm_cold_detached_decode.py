# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import sys
from contextlib import ExitStack
from dataclasses import replace
from hashlib import sha256
from time import monotonic, sleep

import pytest
import torch
from kitepond.backends.fsl2 import BundlePlan
from kitepond.core.contracts import Wait
from kitepond.core.policy import IdlePolicy, LaneBudget
from kitepond.integrations.vllm.decode_completion import GrantRoutedScheduler
from kitepond.integrations.vllm.decode_receive import (
    DecodeReceiveMailbox,
    DecodeReceiveWork,
)
from kitepond.integrations.vllm.glm_cold_decode_bridge import GlmColdDecodeBridge
from kitepond.integrations.vllm.glm_cold_detached_rank import DetachedGlmColdRank
from kitepond.integrations.vllm.glm_cold_prefill_scheduler import (
    GlmColdDecodeAdmissionMixin,
)
from kitepond.integrations.vllm.glm_cold_runtime import GlmColdRuntime
from kitepond.integrations.vllm.glm_host_backing import _Launch, _open_extended_backing
from kitepond.integrations.vllm.glm_native_views import (
    GlmNativeStateViews,
    _NativeLayer,
)
from kitepond.integrations.vllm.hisparse_attachment import ATTACHMENT_MODE
from kitepond.integrations.vllm.native_glm_cold_directory import NativeGlmColdDirectory
from kitepond.integrations.vllm.native_glm_cold_provider import NativeGlmRankMemory
from kitepond.integrations.vllm.terminal_checkpoint import StockTerminalCheckpoints
from kitepond.models.glm5next.cold_lifecycle import GlmColdCheckpoint, GlmColdRamGrant
from kitepond.models.glm5next.cold_restore import ColdRestoreLayout
from kitepond.models.glm5next.frontier import digest
from kitepond.models.glm5next.host_cold_extent import ColdExtentBudget
from kitepond.models.glm5next.import_recipe import part_role
from kitepond.models.glm5next.layout import CacheRole
from stock_hisparse_attachment_probe import BLOCK, _finish, _request, create_scheduler
from test_glm_cold_catalog import (
    NEW_BOOT,
    OLD_BOOT,
    SCOPE,
    _catalog,
    _run_process,
    _write,
)
from test_glm_cold_lifecycle import BUDGET
from test_glm_cold_restore import _cache

from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiKVConnectorWorkerMetadata,
)
from vllm.v1.hisparse.coordinator import (
    HiSparseCoordinator,
    get_hisparse_coordinator,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus

BOUNDARY = BLOCK + 1
MIB = 2**20
LAYOUT = "e" * 64


class ColdDecodeScheduler(GlmColdDecodeAdmissionMixin, GrantRoutedScheduler):
    pass


def _checkpoint(*, boundary=BOUNDARY, kda_source_slot=0):
    source = tuple(_cache(rank=rank) for rank in range(4))
    layout = replace(
        source[0].layout,
        target_valid_tokens=boundary,
        draft_valid_tokens=boundary,
        committed_kda_slot=kda_source_slot,
    )
    plans = []
    for rank, cache in enumerate(source):
        parts = ColdRestoreLayout(layout, rank, tuple(cache.layers.values())).manifest()
        plans.append(
            BundlePlan(
                "glm-native-state-v1",
                "e" * 64,
                tuple(part.role for part in parts),
                parts,
            )
        )
    return GlmColdCheckpoint(
        SCOPE.namespace, SCOPE.generation, layout, "a" * 64, tuple(plans)
    )


class FixtureRankPort:
    """Use real native memfds and worker views with CPU tensors in this probe."""

    def __init__(self, cleanup, directory, rank, pages):
        mla_bytes = pages * 256 * 656
        provisional = ColdExtentBudget(2 * MIB, 2 * MIB, 128 * MIB)
        layout = provisional.layout(mla_bytes)
        backing, drop = _open_extended_backing(
            _Launch("authority", directory, "D", NEW_BOOT, "b" * 64, provisional),
            dict(
                authority_engine_id="D",
                authority_boot_id=NEW_BOOT,
                source_rank=rank,
                layout_sha256=LAYOUT,
            ),
            layout,
        )
        cleanup.callback(drop)
        self.memory = NativeGlmRankMemory.from_registered_cold_backing(backing)
        assert isinstance(self.memory, NativeGlmRankMemory)
        self.views = object.__new__(GlmNativeStateViews)
        self.views.rank = rank
        self.views._groups = 4
        self.views.descriptor = self.memory.region.descriptor
        self.views._layers = {}
        self.views._tail_positions = {}
        self.views._tail_attempts = {}
        for layer in _cache(rank=rank).layers.values():
            role = layer.identity.cache_role
            if role is CacheRole.MLA:
                tensor = backing[:mla_bytes].view(torch.uint8).view(pages, 1, 256, 656)
                group = 0
            else:
                tensor = torch.zeros(
                    (32, *layer.tensor.shape[1:]), dtype=layer.tensor.dtype
                )
                group = 1
            if role is CacheRole.KPOOL_TAIL:
                self.views._tail_positions[layer.identity.layer_name] = torch.full(
                    (32, 8), -1, dtype=torch.int32
                )
            self.views._layers[part_role(layer.identity)] = _NativeLayer(
                group, layer.identity, tensor
            )


class DetachedRestoreProvider:
    def __init__(self, ranks, host):
        self.ranks, self.host = ranks, host
        self.region_fingerprints = tuple(rank.memory.region_sha256 for rank in ranks)
        self.grant = None
        self.prepared = ()
        self.approved = False
        self.read_count = 0

    def register_grant(self, grant, verified):
        assert self.grant is None and self.host.retains(grant)
        self.grant = grant

    def release_grant(self, grant):
        assert grant is self.grant
        if self.prepared and self.approved:
            for local in self.prepared:
                local.abandon_before_import()
        self.grant, self.prepared, self.approved = None, (), False

    def rank_ownership(self, grant):
        assert self.grant is grant
        return tuple(rank.host_ready() for rank in self.prepared)

    def reserve_restore(self, checkpoint, *, max_inflight_bytes, verified_catalog):
        grant = self.grant
        assert max_inflight_bytes == BUDGET
        assert verified_catalog.checkpoint is checkpoint and self.host.retains(grant)

        def live():
            assert all(
                self.host.pool.blocks[block_id].ref_cnt > 0
                for block_id in grant.writable_pages
            )

        ranks = tuple(
            DetachedGlmColdRank(
                port.memory,
                port.views,
                grant,
                verified_catalog,
                require_native=live,
            )
            for port in self.ranks
        )
        authorities = tuple(rank.prepare() for rank in ranks)
        assert not any(isinstance(authority, Wait) for authority in authorities)
        with pytest.raises(ValueError, match="readback has not settled"):
            ranks[0].settle_cold("VERIFIED")
        self.prepared = ranks

        def settle(state):
            for rank in ranks:
                rank.settle_cold(state)
            self.approved = state == "VERIFIED"

        return GlmColdRamGrant(checkpoint, authorities, LAYOUT, settle)


def _restore_in_fresh_d(root, stale):
    cfg_path = root.parent / "cold-native-decode"
    cfg_path.mkdir()
    key_path = cfg_path / "control.key"
    key_path.write_bytes(bytes(range(32)))
    key_path.chmod(0o600)
    os.environ["KITEPOND_GLM_COLD_PREFILL_KEY_FILE"] = str(key_path)
    make = create_scheduler.__wrapped__(cfg_path)
    scheduler = make(
        "decode",
        routed=True,
        scheduler_cls=ColdDecodeScheduler,
        extra_config={
            "kitepond": {
                "mode": ATTACHMENT_MODE,
                "stage": "decode",
                "authority_engine_id": "D",
                "boot_id": NEW_BOOT,
            },
            "kitepond_glm_cold_prefill": {
                "p_boot_id": "P-1",
                "max_pending": 2,
                "arm_timeout_seconds": 15,
            },
        },
    )
    with ExitStack() as cleanup:
        cleanup.callback(scheduler.shutdown)
        owner = scheduler._cold_decode_owner
        host = owner.authority
        free = host.pool.get_num_free_blocks()
        terminal = StockTerminalCheckpoints(
            scheduler.kv_cache_manager,
            4,
            source_engine_id="D",
            source_boot_id=NEW_BOOT,
            on_release=lambda _: None,
        )
        sockets = cfg_path / "s"
        sockets.mkdir(mode=0o700)
        ports = tuple(
            FixtureRankPort(cleanup, sockets, rank, len(host.pool.blocks))
            for rank in range(4)
        )
        provider = DetachedRestoreProvider(ports, host)
        catalog = _catalog(root)
        boundary = stale.checkpoint.layout.target_valid_tokens
        request = _request(boundary + 3, "fresh-D-request")
        token_digest = digest([3] * boundary)

        def audit(event, args):
            if (
                event == "open"
                and isinstance(args[0], (str, bytes))
                and "/payloads/" in os.fsdecode(args[0])
            ):
                assert provider.prepared and not scheduler.has_requests()
                assert provider.grant is not None
                assert all(
                    host.pool.blocks[block_id].ref_cnt > 0
                    for block_id in provider.grant.writable_pages
                )
                provider.read_count += 1

        sys.addaudithook(audit)
        assert not catalog.verify(stale)
        verified = catalog.reopen(SCOPE)
        assert verified is not None and verified.source_boot_id == OLD_BOOT
        assert verified.checkpoint.layout == stale.checkpoint.layout
        directory = NativeGlmColdDirectory(
            terminal,
            provider,
            lambda obj: SCOPE if obj is request else Wait("WRONG_SCOPE"),
            max_entries=4,
            clock=lambda: 5,
            catalog=catalog,
        )
        lane = LaneBudget(1, BUDGET, BUDGET)
        clock = [5]
        callbacks = []
        runtime = GlmColdRuntime(
            root,
            directory,
            terminal,
            register_quiescent=lambda **registration: callbacks.append(registration),
            cadence_ns=20_000_000,
            idle=IdlePolicy(15),
            foreground=lane,
            background=lane,
            clock=lambda: clock[0],
        )
        bridge = GlmColdDecodeBridge(
            scheduler,
            runtime,
            authorize_prefix=lambda *_: token_digest,
            register_grant=provider.register_grant,
            release_grant=provider.release_grant,
            confirm_rank_ownership=provider.rank_ownership,
        )
        with pytest.raises(ValueError, match="authorized tokens"):
            bridge.queue(
                _request_with_wrong_tokens(boundary),
                verified,
                transfer_id="t",
                epoch=7,
            )
        assert host.pool.get_num_free_blocks() == free
        cancelled = bridge.queue(request, verified, transfer_id="discarded", epoch=6)
        assert not isinstance(cancelled, Wait)
        assert bridge.cancel(cancelled) is True
        assert host.pool.get_num_free_blocks() == free and provider.read_count == 0
        ready = bridge.queue(request, verified, transfer_id="new-transfer", epoch=7)
        assert not isinstance(ready, Wait)
        assert host.pool.get_num_free_blocks() == free - (boundary + BLOCK - 1) // BLOCK
        assert (
            request.request_id not in scheduler.requests
            and not scheduler.has_requests()
        )
        assert isinstance(bridge.stage(ready), Wait) and provider.read_count == 0
        [registration] = callbacks
        assert registration["interval_ns"] == 20_000_000
        deadline = monotonic() + 12
        while runtime.scheduler.view(ready.job_id).state != "VERIFIED":
            assert monotonic() < deadline
            assert runtime.scheduler.view(ready.job_id).state in {"QUEUED", "PENDING"}
            clock[0] += registration["interval_ns"]
            registration["callback"]()
            sleep(0.002)
        assert provider.read_count > 0 and not scheduler.has_requests()
        for rank, local in enumerate(provider.prepared):
            assert local.host_ready().lease_sha256 == ready.grant.sha256
            for bound in local._data.bindings:
                data = b"".join(
                    local.memory.region.mapping[span.offset : span.offset + span.nbytes]
                    for span in bound.spans
                )
                assert sha256(data).hexdigest() == verified.checksums(rank)[bound.part]

        assert bridge.stage(ready) is True
        step = scheduler.schedule()
        binding = owner.pending_decode_import(request)
        assert (
            binding is not None
            and request.status is RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        assert scheduler.verified_decode_import(request) is None
        work = step.kv_connector_metadata.metadata[0]
        assert isinstance(work, DecodeReceiveWork)
        (ticket,) = work.tickets
        assert ticket.grant_sha256 == ready.grant.sha256
        assert ticket.request_id != verified.checkpoint.layout.request_id

        for rank, local in enumerate(provider.prepared):
            mailbox = DecodeReceiveMailbox(rank, NEW_BOOT)
            mailbox.bind((ticket,))
            slab = torch.empty(4 * MIB, dtype=torch.uint8)
            if rank == 0:
                with pytest.raises(ValueError, match="verified native D ticket"):
                    local.import_ready(
                        mailbox, replace(ticket, epoch=ticket.epoch + 1), slab
                    )
            local.import_ready(mailbox, ticket, slab)
            assert local._lease is None
            assert local.views._tail_attempts
            # Native GDN/KDA prefill selects column zero for the new request.
            (part,) = (
                part
                for part in verified.checkpoint.plans[rank].parts
                if local.views._layers[part.role].identity.cache_role
                is CacheRole.KDA_RECURRENT
            )
            native = local.views._layers[part.role]
            base, *draft = ticket.block_ids[native.group_id]
            assert len(draft) == 3
            recovered_kda = native.tensor[base].view(torch.uint8).numpy().tobytes()
            assert (
                sha256(recovered_kda).hexdigest() == verified.checksums(rank)[part]
            ), (
                f"source MTP row {verified.checkpoint.layout.committed_kda_slot} "
                f"did not initialize native base KDA on rank {rank}"
            )
            assert all(
                torch.count_nonzero(native.tensor[block]) == 0 for block in draft
            )
            worker_output = mailbox.drain(None)
            scheduler.update_from_output(
                step,
                ModelRunnerOutput.with_kv_conn_output_only(
                    KVConnectorOutput(
                        finished_recving={request.request_id},
                        kv_connector_worker_meta=MultiKVConnectorWorkerMetadata(
                            metadata=(worker_output, None)
                        ),
                    )
                ),
            )
            if rank < 3:
                assert scheduler.verified_decode_import(request) is None
        assert scheduler.verified_decode_import(request) is binding
        resumed = scheduler.schedule()
        assert resumed.num_scheduled_tokens == {request.request_id: 3}
        assert request.status is RequestStatus.RUNNING
        _finish(scheduler, request)
        coordinator = get_hisparse_coordinator(scheduler.kv_cache_manager)
        counts = dict.fromkeys(coordinator.pending_spills, 1)
        if counts:
            scheduler.update_from_output(
                resumed,
                ModelRunnerOutput.with_kv_conn_output_only(
                    KVConnectorOutput(
                        kv_connector_worker_meta=MultiKVConnectorWorkerMetadata(
                            metadata=(
                                HiSparseConnectorWorkerMetadata(counts, counts),
                                None,
                            )
                        )
                    ),
                ),
            )
        assert host.pool.get_num_free_blocks() == free
        return {
            "pid": os.getpid(),
            "regions": list(provider.region_fingerprints),
            "payload_reads": provider.read_count,
            "verified_rank_imports": len(provider.prepared),
            "native_suffix_tokens": resumed.num_scheduled_tokens[request.request_id],
            "host_free_initial": free,
            "host_free_final": host.pool.get_num_free_blocks(),
        }


def _request_with_wrong_tokens(boundary):
    req = _request(boundary + 3, "wrong-prefix")
    assert req.prompt_token_ids is not None
    req.prompt_token_ids[0] = 13
    return req


@pytest.mark.skipif(
    not callable(getattr(HiSparseCoordinator, "attach_external_host_blocks", None)),
    reason="this engine cannot attach verified cold grants to native requests",
)
@pytest.mark.parametrize("kda_source_slot", range(4))
def test_fresh_d_reads_nvme_before_native_admission_and_waits_for_four_ready_fences(
    tmp_path,
    kda_source_slot,
):
    root = tmp_path / "cold"
    checkpoint = _checkpoint(boundary=3 * BLOCK + 1, kda_source_slot=kda_source_slot)
    original = _run_process(_write, root, checkpoint)
    recovered = _run_process(_restore_in_fresh_d, root, original["old_sig"])
    assert recovered["pid"] != original["pid"]
    assert set(recovered["regions"]).isdisjoint(original["regions"])
    assert recovered["payload_reads"] > 0

    receipt = {
        "device": "cpu",
        "source_pid": original["pid"],
        "target_pid": recovered["pid"],
        "source_boot_id": OLD_BOOT,
        "source_kda_slot": kda_source_slot,
        "destination_kda_slot": 0,
        "target_boot_id": NEW_BOOT,
        "reserved_before_payload": True,
        "foreign_ticket_epoch_rejected": True,
        "required_rank_readiness": 4,
        **{
            name: value
            for name, value in recovered.items()
            if name not in {"pid", "regions"}
        },
    }
    (tmp_path / "cold-restore-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
