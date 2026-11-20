# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import ctypes
import json
import os
import subprocess
import time
from collections import deque
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import terminal_post_step_probe as terminal_probe
import torch
from kitepond.integrations.mooncake.live_receive import (
    AbortedRankWindows,
)
from kitepond.integrations.mooncake.named_control import GlmWindowControlScope
from kitepond.integrations.vllm import glm_warm_prefill_worker, hisparse_attachment
from kitepond.integrations.vllm.checkpoint import CheckpointReadReceipt
from kitepond.integrations.vllm.glm_cold_prefill_admission import (
    ColdPrefillAdmission,
    ColdPrefillPrompt,
)
from kitepond.integrations.vllm.glm_native_views import (
    GlmNativeStateViews,
    _NativeLayer,
)
from kitepond.integrations.vllm.glm_prefill_return import GlmPrefillReturnScheduler
from kitepond.integrations.vllm.glm_return_bootstrap import (
    GlmDecodeTerminalSources,
)
from kitepond.integrations.vllm.glm_terminal_return import (
    GlmNativeTerminalReturnWindows,
)
from kitepond.integrations.vllm.glm_warm_decode_scheduler import GlmDecodeWarmReturns
from kitepond.integrations.vllm.glm_warm_prefill_scheduler import (
    WARM_ADMISSION_PARAM,
    GlmWarmPrefillScheduler,
)
from kitepond.integrations.vllm.hisparse_attachment import (
    ATTACHMENT_MODE,
    HiSparseConnector,
    KitepondHiSparseAttachmentConnector,
)
from kitepond.integrations.vllm.host_authority import (
    HostRankSettlement,
    NativeHostAuthority,
)
from kitepond.integrations.vllm.prefill_return_receive import PREFILL_RETURN_BOOT_CONFIG
from kitepond.models.glm5next.cold_restore import ColdRestoreLayout, RestoreLayer
from kitepond.models.glm5next.layout import CacheRole
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion
from terminal_post_step_probe import BLOCK, TOKENS, _add_request
from test_glm_terminal_manifest import BOUNDARY
from test_stock_terminal_checkpoint import identity

from vllm.config import CacheConfig, KVTransferConfig, VllmConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.sampling_params import SamplingParams
from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

terminal_engine = terminal_probe.terminal_engine


class CpuMooncakeMemory:
    """Exercise the native registration/write API with bounded local memory."""

    def __init__(self):
        self.extents = {}

    def batch_register_memory(self, pointers, lengths):
        self.extents.update(zip(pointers, lengths, strict=True))
        return 0

    def batch_unregister_memory(self, pointers):
        for ptr in pointers:
            self.extents.pop(ptr)
        return 0

    def batch_transfer_sync_write(self, session, sources, destinations, sizes):
        assert session == "127.0.0.1:12345"
        for source, destination, size in zip(sources, destinations, sizes, strict=True):
            assert size <= min(self.extents[source], self.extents[destination])
            ctypes.memmove(destination, source, size)
        return 0


def _p_scheduler(engine, *, warm=None):
    cache = deepcopy(engine.scheduler.kv_cache_manager.kv_cache_config)
    config = VllmConfig(
        model_config=deepcopy(engine.vllm_config.model_config),
        cache_config=CacheConfig(
            block_size=BLOCK,
            enable_prefix_caching=False,
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=128,
            max_num_batched_tokens=128,
            max_num_seqs=2,
            is_encoder_decoder=False,
            async_scheduling=False,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="MultiConnector",
            engine_id="P",
            kv_role="kv_both",
            kv_connector_extra_config={
                "connectors": [
                    {
                        "kv_connector": "KitepondHiSparseAttachmentConnector",
                        "kv_connector_module_path": (
                            KitepondHiSparseAttachmentConnector.__module__
                        ),
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {"host_pool_gib": 1},
                    },
                    {"kv_connector": "MooncakeConnector", "kv_role": "kv_both"},
                ],
            },
        ),
        additional_config={
            PREFILL_RETURN_BOOT_CONFIG: "P-boot",
            "kitepond": {
                "mode": ATTACHMENT_MODE,
                "stage": "prefill",
                "authority_engine_id": "D",
                "boot_id": "D-boot",
            },
        },
    )
    config.cache_config.num_gpu_blocks = cache.num_blocks
    cls = GlmPrefillReturnScheduler
    if warm is not None:
        tmp_path, monkeypatch, _ = warm
        key = tmp_path / "prefill.key"
        key.write_bytes(b"f" * 32)
        key.chmod(0o600)
        directory = tmp_path / "ready"
        directory.mkdir(mode=0o700)
        monkeypatch.setenv("KITEPOND_GLM_COLD_PREFILL_KEY_FILE", str(key))
        config.additional_config.update(
            {
                "kitepond_glm_cold_prefill": {
                    "p_boot_id": "P-boot",
                    "max_pending": 2,
                    "arm_timeout_seconds": 30,
                },
                "kitepond_glm_prefill_k0": {"ready_directory": str(directory)},
                "kitepond_glm_window": {
                    "p_engine_id": "P",
                    "p_boot_id": "P-boot",
                    "d_engine_id": "D",
                    "d_boot_id": "D-boot",
                    "decode_host": "127.0.0.1",
                },
            }
        )
        config.kv_transfer_config.kv_connector_extra_config["connectors"][1][
            "kv_connector_module_path"
        ] = "kitepond.integrations.vllm.glm_prefill_bootstrap"
        cls = GlmWarmPrefillScheduler
    scheduler = cls(
        vllm_config=config,
        kv_cache_config=cache,
        structured_output_manager=StructuredOutputManager(config),
        log_stats=False,
        block_size=BLOCK,
    )
    if warm is not None:
        # The small cache groups stand in for real GLM KDA tensors. Select
        # the actual native Mamba ingress condition used by the deployed model.
        scheduler.connector.sub_connectors[1].connector_scheduler._has_mamba = True
    return scheduler


@pytest.fixture
def ready_worker_pids():
    processes = [subprocess.Popen(("sleep", "90")) for _ in range(4)]
    try:
        yield tuple(process.pid for process in processes)
    finally:
        for process in processes:
            process.terminate()
            process.wait(timeout=5)


def _p_native_destination(engine, source, *, warm=None):
    lease, manifest = source.lease, source.manifest
    p = _p_scheduler(engine, warm=warm)
    prompt = TOKENS + [3, 4]
    request = Request(
        request_id="prefill-return",
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={
                "kv_transfer_params": {
                    "do_remote_decode": True,
                    "transfer_id": "prefill-grant",
                },
            }
            if warm is not None
            else None,
        ),
        pooling_params=None,
    )
    coordinator = get_hisparse_coordinator(engine.scheduler.kv_cache_manager)
    host = NativeHostAuthority(
        coordinator.get_host_block_pool(),
        block_tokens=BLOCK,
        engine_id="D",
        boot_id="D-boot",
    )
    if warm is not None:

        def enroll(method, *, timeout, args=()):
            assert method == "kitepond_prefill_ready_identity"
            return tuple(
                {
                    "rank": rank,
                    "pid": warm[2][rank],
                    "region_sha256": f"{rank + 1:064x}",
                }
                for rank in range(4)
            )

        p.get_worker_startup_callback()(enroll)
        control = p._cold_prefill_config.control
        cold = ColdPrefillAdmission(
            host,
            control,
            context_limit=128,
            max_pending=2,
            arm_timeout_seconds=30,
        )
        issued = cold.offer(
            ColdPrefillPrompt.from_tokens(
                request.request_id, "prefill-grant", lease.identity.epoch, prompt
            )
        )
        grant = issued.grant
    else:
        p.add_request(request)
        grant = host.reserve(
            request.request_id,
            "prefill-grant",
            lease.identity.epoch,
            prefix_tokens=0,
            reservation_tokens=request.num_prompt_tokens,
            source=(),
        )
        host.dispatch(grant)
        with pytest.raises(ValueError, match="verified waiting offer"):
            p.stage_terminal_return(request, lease, manifest, grant)

    def verify_offer(actual_lease, actual_manifest, actual_grant):
        assert actual_grant == grant
        engine.scheduler._terminal_capture.verifier.verify(
            actual_lease, actual_manifest
        )

    if warm is not None:
        original_tokens = len(prompt)

        def wire(message):
            return json.loads(json.dumps(message))

        source_record = wire(p._warm_control.seal("source", source))
        signed_offer = wire(control.export_offer(issued))
        body = {"source": source_record, "offer": signed_offer}
        corrupt = {**source_record, "mac": "0" * 64}
        with pytest.raises(ValueError, match="not authenticated"):
            p._warm_p_handler("warm_authorize", {**body, "source": corrupt})
        issued_nonce = p._warm_p_handler("warm_authorize", body)
        assert p._warm_p_handler("warm_authorize", body) == issued_nonce
        request.kv_transfer_params[WARM_ADMISSION_PARAM] = issued_nonce["nonce"]
        p.add_request(request)
        assert request.num_prompt_tokens == original_tokens - 1
    else:
        p.bind_terminal_return_offer_verifier(verify_offer)
        with pytest.raises(ValueError, match="frontier or grant"):
            p.stage_terminal_return(request, lease, manifest, replace(grant, epoch=99))
        p.stage_terminal_return(request, lease, manifest, grant)
    pending = p.schedule()
    assert not pending.num_scheduled_tokens
    assert request.status is RequestStatus.WAITING_FOR_REMOTE_KVS
    reservation = p.terminal_return_destination(request)
    if warm is not None:
        response = p._warm_p_handler(
            "warm_destination", {"request_id": request.request_id}
        )
        assert (
            p._warm_control.open("destination", wire(response["destination"]))
            == reservation
        )
    return p, request, host, grant, reservation, pending


@pytest.fixture
def rank_regions():
    blocks, rows, width = 16, 256, 656
    nbytes = 2 * blocks * rows * width
    with ExitStack() as cleanup:
        records = []
        for rank in range(4):
            scope = dict(
                authority_engine_id="D",
                authority_boot_id="D-boot",
                source_rank=rank,
                layout_sha256="a" * 64,
            )
            d = NativeHiSparseMemfdRegion.create(nbytes, **scope)
            cleanup.callback(d.close)
            p = NativeHiSparseMemfdRegion.join_received_fd(
                d.descriptor, os.dup(d.fd), **scope
            )
            cleanup.callback(p.close)
            records.append((d, p))
        yield records


def _mla_view(region, model_role):
    return region.tensor().view(2, 16, 1, 256, 656)[int(model_role == "draft")]


@pytest.mark.parametrize("mode", ["complete", "cancel", "rank-timeout"])
def test_native_prefill_return_holds_d_until_all_four_p_read_fences(
    terminal_engine,
    monkeypatch,
    mode,
    rank_regions,
    tmp_path,
    ready_worker_pids,
):
    cancel_prefill = mode == "cancel"
    engine, owner, _ = terminal_engine
    executor = engine.model_executor
    for rank, worker in enumerate(executor.ranks):
        region = rank_regions[rank][0]
        signer, old_source = worker._kitepond_terminal_source

        def resolve(lease, observed, region=region, old_source=old_source):
            result = old_source(lease, observed)
            layers = [
                replace(layer, tensor=_mla_view(region, layer.identity.model_role))
                if layer.identity.cache_role is CacheRole.MLA
                else layer
                for layer in result.layout.layers.values()
            ]
            return replace(
                result,
                layout=ColdRestoreLayout(
                    result.layout.layout, result.layout.rank, layers
                ),
                authority_sha256=region.descriptor.fingerprint(),
            )

        worker._kitepond_terminal_source = (signer, resolve)
    monkeypatch.setattr(
        engine.scheduler._terminal_capture.verifier,
        "_authority",
        {
            rank: record[0].descriptor.fingerprint()
            for rank, record in enumerate(rank_regions)
        },
    )
    receipts_by_rank = {}
    registered = []
    reclaimed = []
    selected_destination = []

    def rpc(method, *, timeout, args=()):
        if method == "kitepond_register_terminal_return":
            registered.append(args[0])
            return (None,) * 4
        if method == "kitepond_terminal_return_result":
            assert registered == [args[0]]
            return tuple(receipts_by_rank.get(rank) for rank in (0, 1, 2, 3))
        if method in (
            "kitepond_retire_terminal_return",
            "kitepond_discard_terminal_frame",
        ):
            reclaimed.append((method, args[0]))
            return (None,) * 4
        return engine.collective_rpc(method, timeout=timeout, args=args)

    def require_destination(destination):
        assert selected_destination == [destination]

    loop = GlmDecodeWarmReturns(engine.scheduler, rpc, require_destination)
    settlements = loop.settlements
    monkeypatch.setattr(owner, "_verify_settlement", settlements.verify)
    request = _add_request(engine, "terminal", TOKENS[:-1])
    intent = replace(
        identity("terminal"),
        deadline_monotonic_ns=time.monotonic_ns() + 30_000_000_000,
    )
    loop.arm(intent)
    assert loop.publish(intent) is None
    assert engine.step()[1]
    lease = owner.lease_for(request.request_id)
    assert lease is not None and lease.requested_common_tokens == BOUNDARY
    executor.durable = True
    source = loop.publish(intent)
    assert source.lease is lease and loop.publish(intent) is source
    manifest = source.manifest
    p, p_request, host, grant, reservation, p_pending = _p_native_destination(
        engine,
        source,
        warm=(tmp_path, monkeypatch, ready_worker_pids) if mode == "complete" else None,
    )
    selected_destination.append(reservation)
    transfer = loop.prepare(source, reservation)
    permit = transfer.permit
    assert loop.prepare(source, reservation) is transfer
    assert loop.poll(transfer).received_ranks == ()
    with pytest.raises(ValueError, match="waiting for its quorum"):
        loop.retire(transfer)
    with pytest.raises(ValueError, match="changed the admitted offer"):
        p.bind_terminal_return(
            replace(transfer, destination=replace(reservation, boot_id="foreign"))
        )
    if type(p) is GlmWarmPrefillScheduler:
        names, bind = p.get_owner_worker_control_registration()
        assert "kitepond_prefill_submit_terminal_return" in names
        bind(lambda method, *, timeout, args=(): (None,) * 4)
        permit_record = json.loads(
            json.dumps(p._warm_control.seal("transfer", transfer))
        )
        assert p._warm_p_handler(
            "warm_submit", {"transfer": permit_record, "timeout": 3}
        ) == {
            "submitted": True,
        }
    else:
        p.bind_terminal_return(transfer)

    def update_prefill(step, output):
        p.update_from_output(
            step,
            ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                kv_connector_output=output,
            ),
        )

    # Ordinary flat flags from another connector never publish a warm return.
    update_prefill(
        p_pending,
        KVConnectorOutput(
            finished_recving={p_request.request_id},
            finished_sending={p_request.request_id},
        ),
    )
    assert not p.finished_recving_kv_req_ids
    assert p.requests[p_request.request_id] is p_request
    assert settlements.prepare(lease, permit, manifest, reservation) is transfer
    with pytest.raises(ValueError, match="another destination"):
        settlements.prepare(
            lease, permit, manifest, replace(reservation, boot_id="foreign")
        )
    with pytest.raises(ValueError, match="Unattested"):
        owner.settle(lease, permit, CheckpointReadReceipt(permit.sha256, 0, "acked", 0))

    target_groups = {
        name: group for group in reservation.groups for name in group.layer_names
    }
    observed_ranks = []
    p_workers = []
    for rank in range(4):
        connector = HiSparseConnector(
            p.vllm_config, KVConnectorRole.WORKER, p.kv_cache_manager.kv_cache_config
        )
        # The CPU fixture seeds only the native HiSparse DMA bookkeeping.
        native = connector.connector_worker
        native._enqueued_transfer_ids = []
        native._pending_transfer_events = deque()
        native._completed_host_copy_dst_ids = []
        mooncake = object.__new__(MooncakeConnector)
        mooncake.connector_worker = SimpleNamespace(
            engine=CpuMooncakeMemory(), is_kv_producer=False, is_kv_consumer=False
        )
        multi = object.__new__(MultiConnector)
        multi._role = KVConnectorRole.WORKER
        multi._connectors = [connector, mooncake]
        p_workers.append(multi)
    aggregator = KVOutputAggregator(expected_finished_count=4)
    p_slot = [0]
    monkeypatch.setattr(
        hisparse_attachment,
        "get_dcp_group",
        lambda: SimpleNamespace(rank_in_group=p_slot[0], world_size=4),
    )

    async def roundtrip(rank):
        p_slot[0] = rank
        worker = executor.ranks[rank]
        source = worker.terminal_window_source(lease, permit, manifest)
        with pytest.raises(ValueError, match="no verified native rank source"):
            worker.terminal_window_source(
                lease, replace(permit, manifest_sha256="a" * 64), manifest
            )
        source_layers = source.layout.layers
        targets = []
        for layer in source_layers.values():
            group = target_groups[layer.identity.layer_name]
            if layer.identity.cache_role is CacheRole.MLA:
                tensor = _mla_view(rank_regions[rank][1], layer.identity.model_role)
                assert tensor.data_ptr() != layer.tensor.data_ptr()
                assert set(group.block_ids).isdisjoint(layer.block_ids)
                layer.tensor[layer.block_ids[0]].fill_(rank + 17)
                assert tensor[layer.block_ids[0]].unique().tolist() == [rank + 17]
                assert not tensor[group.block_ids[0]].count_nonzero()
            else:
                tensor = torch.zeros(
                    (max(group.block_ids) + 1, *layer.tensor.shape[1:]),
                    dtype=layer.tensor.dtype,
                )
            targets.append(RestoreLayer(layer.identity, tensor, group.block_ids))
            if layer.identity.cache_role is CacheRole.KDA_CONV:
                raw = layer.tensor[layer.block_ids[0]]
                for index in range(6):
                    raw[index].fill_(index + 1 + rank)
            elif layer.identity.cache_role is CacheRole.KPOOL_TAIL:
                raw = layer.tensor[layer.block_ids[0]]
                for slot in range(8):
                    raw[:, slot, :].fill_(slot + 1 + rank)
        target = ColdRestoreLayout(
            replace(
                source.layout.layout,
                request_id=p_request.request_id,
                committed_kda_slot=0,
            ),
            rank,
            targets,
        )
        layer_groups = {
            role: target_groups[layer.identity.layer_name].group_id
            for role, layer in target.layers.items()
        }
        native_views = object.__new__(GlmNativeStateViews)
        native_views.rank = rank
        native_views._groups = len(p.kv_cache_manager.kv_cache_config.kv_cache_groups)
        native_views.descriptor = rank_regions[rank][1].descriptor
        native_views._layers = {
            role: _NativeLayer(layer_groups[role], layer.identity, layer.tensor)
            for role, layer in target.layers.items()
        }
        native_views._tail_attempts = {}
        native_views._tail_positions = {
            layer.identity.layer_name: torch.full(
                (layer.tensor.shape[0], 8), 9999, dtype=torch.int32
            )
            for layer in target.layers.values()
            if layer.identity.cache_role is CacheRole.KPOOL_TAIL
        }
        multi = p_workers[rank]
        engine_memory = multi.sub_connectors[1].connector_worker.engine
        rx = GlmNativeTerminalReturnWindows(multi, slots=2)
        installed = {}

        def bind_namespace(name, handler):
            assert name not in installed
            installed[name] = handler

        d_native = SimpleNamespace(
            sender_loop=asyncio.get_running_loop(),
            is_kv_consumer=False,
            tp_rank=rank,
            engine_id="D",
            engine=engine_memory,
            bind_named_control_handler=bind_namespace,
        )
        registry = GlmDecodeTerminalSources(
            d_native,
            GlmWindowControlScope("P-boot", "D-boot", rank, b"c" * 32, reverse=True),
            "127.0.0.1",
            capacity=1,
        )
        await registry.register(transfer, worker)
        await registry.register(transfer, worker)
        wrong_order = replace(
            transfer, destination=replace(reservation, boot_id="other")
        )
        with pytest.raises(ValueError, match="another deployment"):
            await registry.register(wrong_order, worker)
        assert len(installed) == 1
        attempt = transfer.attempt(rank)
        assert await registry.result(attempt) is None
        assert await registry._open(attempt, ("127.0.0.2", 12345, rx.pool_id)) is None
        writer = await registry._open(attempt, ("127.0.0.1", 12345, rx.pool_id))
        assert writer is not None
        rank_result = []
        skip_window_reply = mode == "rank-timeout" and rank == 0

        class NativePeer:
            async def fetch(self, *args):
                result = await writer.fetch(*args)
                if skip_window_reply:
                    raise TimeoutError("Injected loss after completed native write")
                return result

            async def abort(self, attempt, pool_id):
                assert pool_id == rx.pool_id
                await writer.abandon()
                return AbortedRankWindows(attempt)

            async def settle(self, proof):
                assert (
                    await writer.fenced_settlement(proof.attempt, proof.outcome)
                    == proof
                )
                await registry._settled(proof)
                rank_result.append(await registry.result(proof.attempt))

            async def commit(self, attempt, last_sequence):
                result = await writer.commit(attempt, last_sequence)
                assert owner.lease_for(request.request_id) is lease
                return result

        def after_p_fence(attempt, receipt):
            assert owner.lease_for(request.request_id) is lease
            assert p.requests[p_request.request_id] is p_request
            assert p_request.status is (
                RequestStatus.FINISHED_ABORTED
                if cancel_prefill and len(observed_ranks) >= 1
                else RequestStatus.WAITING_FOR_REMOTE_KVS
            )
            assert receipt.status == ("abort_settled" if skip_window_reply else "acked")
            if not skip_window_reply:
                for layer in target.layers.values():
                    if layer.identity.cache_role is CacheRole.KPOOL_TAIL:
                        name = layer.identity.layer_name
                        block = layer.block_ids[0]
                        expected = [-1] * 8
                        for position in range(BOUNDARY - BOUNDARY % 4, BOUNDARY):
                            expected[position % 8] = position
                        assert (
                            native_views._tail_positions[name][block].tolist()
                            == expected
                        )
                for part in target.manifest():
                    actual = target.read_chunk(part)
                    assert torch.equal(actual, source.read_chunk(part))
                    if part.role.endswith(":kda_conv"):
                        values = actual.view(torch.bfloat16).view(part.shape)
                        assert values[:, 0].tolist() == [
                            rank + 2,
                            rank + 3,
                            rank + 4,
                            0,
                            0,
                            0,
                        ]
                    elif part.role.endswith(":kpool_tail"):
                        values = actual.view(torch.bfloat16).view(part.shape)
                        assert values[:, 0, 0].tolist() == [
                            rank + 1,
                            rank + 2,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
            with pytest.raises(ValueError, match="changed its boot or destination"):
                settlements.accept_prefill_fence(
                    transfer,
                    attempt,
                    receipt,
                    peer_engine_id="P",
                    peer_boot_id="foreign",
                )
            worker_outputs = [
                ModelRunnerOutput(
                    req_ids=[],
                    req_id_to_index={},
                    kv_connector_output=KVConnectorOutput(
                        kv_connector_worker_meta=child.build_connector_worker_meta()
                    ),
                )
                for child in p_workers
            ]
            output = aggregator.aggregate(worker_outputs)
            pending = p.schedule()
            assert not pending.num_scheduled_tokens
            update_prefill(pending, output.kv_connector_output)
            observed_ranks.append(rank)

            if len(observed_ranks) < 4:
                assert not p.finished_recving_kv_req_ids
                assert p.requests[p_request.request_id] is p_request
                if cancel_prefill and len(observed_ranks) == 1:
                    p.finish_requests(
                        p_request.request_id, RequestStatus.FINISHED_ABORTED
                    )
                    assert p.requests[p_request.request_id] is p_request
                    assert p.connector.has_pending_push_work()
                    with pytest.raises(ValueError, match="holds writable host"):
                        p.connector.sub_connectors[0].retire_prefill_grant(p_request)
            (result,) = rank_result
            assert result.proof.attempt == attempt
            assert result.proof.outcome == receipt.status
            receipts_by_rank[rank] = result
            status = loop.poll(transfer)
            assert status.received_ranks == tuple(sorted(observed_ranks))
            assert status.complete == (len(observed_ranks) == 4)
            host.settle(
                grant,
                HostRankSettlement(reservation.grant.sha256, rank, "ready", 0, 0),
            )

        try:
            with pytest.raises(ValueError, match="changed its boot or destination"):
                settlements.accept_prefill_fence(
                    transfer,
                    transfer.attempt(rank),
                    CheckpointReadReceipt(permit.sha256, rank, "acked", 0),
                    peer_engine_id="P",
                    peer_boot_id="P-boot",
                )
            with pytest.raises(ValueError, match="foreign worker or boot"):
                await rx.receive_at_prefill(
                    replace(
                        transfer, destination=replace(reservation, boot_id="foreign")
                    ),
                    target,
                    layer_groups,
                    NativePeer(),
                    after_p_fence,
                    native_views=native_views,
                    stream=None,
                    timeout=3,
                )
            descriptor = native_views.descriptor
            native_views.descriptor = rank_regions[(rank + 1) % 4][1].descriptor
            with pytest.raises(ValueError, match="another D host authority"):
                await rx.receive_at_prefill(
                    transfer,
                    target,
                    layer_groups,
                    NativePeer(),
                    after_p_fence,
                    native_views=native_views,
                    stream=None,
                    timeout=3,
                )
            native_views.descriptor = descriptor
            witnesses = native_views._tail_positions
            native_views._tail_positions = {}
            with pytest.raises(ValueError, match="no live KPool tail block"):
                await rx.receive_at_prefill(
                    transfer,
                    target,
                    layer_groups,
                    NativePeer(),
                    after_p_fence,
                    native_views=native_views,
                    stream=None,
                    timeout=3,
                )
            native_views._tail_positions = witnesses

            class NativeDestination:
                def receive(
                    self, active, views, cache, groups, stream, notify, *, timeout
                ):
                    peer = NativePeer()

                    self.task = asyncio.create_task(
                        rx.receive_at_prefill(
                            active,
                            cache,
                            groups,
                            peer,
                            after_p_fence,
                            native_views=views,
                            stream=stream,
                            timeout=timeout,
                            reverse_peer=peer,
                            before_route=notify,
                        )
                    )
                    return self.task

                def close(self):
                    rx.close()

            destination = NativeDestination()
            monkeypatch.setattr(
                glm_warm_prefill_worker,
                "GlmPrefillTerminalDestination",
                lambda: destination,
            )
            imports = glm_warm_prefill_worker.GlmPrefillWarmImports(
                native_views, None, capacity=1
            )
            imports.submit(transfer, 3)
            imports.submit(transfer, 3)
            assert imports.result(transfer) is None
            route = SimpleNamespace(grant=reservation.grant, imported_tokens=BOUNDARY)
            with pytest.raises(ValueError, match="no verified local warm prefix"):
                imports.require_loaded(route)
            with pytest.raises(ValueError, match="not fenced and settled"):
                imports.retire(transfer)
            if skip_window_reply:
                with pytest.raises(TimeoutError, match="Injected loss"):
                    await destination.task
                with pytest.raises(ValueError, match="no verified local warm prefix"):
                    imports.require_loaded(route)
            else:
                await destination.task
                imports.require_loaded(route)
            assert imports.result(transfer).attempt == transfer.attempt(rank)
            imports.retire(transfer)
            imports.close()
        finally:
            (result,) = rank_result
            await registry.retire(result)
            registry.close()
        assert not engine_memory.extents

    async def transfer_all():
        for rank in (2, 0, 3, 1):
            await roundtrip(rank)

    asyncio.run(transfer_all())
    assert observed_ranks == [2, 0, 3, 1]
    assert owner.lease_for(request.request_id) is None
    assert request.request_id not in engine.scheduler.requests
    assert loop.poll(transfer).aborted_ranks == ((0,) if mode == "rank-timeout" else ())
    loop.retire(transfer)
    assert reclaimed == [
        ("kitepond_retire_terminal_return", transfer),
        ("kitepond_discard_terminal_frame", lease),
    ]
    assert host.observe().active_grants == 1
    if mode in ("cancel", "rank-timeout"):
        assert p_request.request_id not in p.requests
        if mode == "rank-timeout":
            assert p_request.status is RequestStatus.FINISHED_ERROR
        assert (
            p.connector.sub_connectors[0].pending_prefill_terminal_return(p_request)
            is None
        )
        assert not p.connector.has_pending_push_work()
        assert (
            p.connector.sub_connectors[0].retire_prefill_grant(p_request)
            is reservation.grant
        )
        assert host.cancel(reservation.grant)
        return

    assert p_request.request_id in p.finished_recving_kv_req_ids
    p_resume = p.schedule()
    assert p_resume.num_scheduled_tokens == {
        p_request.request_id: p_request.num_prompt_tokens - BOUNDARY
    }
    assert p_resume.scheduled_new_reqs[0].num_computed_tokens == BOUNDARY
