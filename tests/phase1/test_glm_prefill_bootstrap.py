# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch the real P alias and exercise its only authenticated READY route."""

import asyncio
import json
import multiprocessing
import os
import secrets
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_glm_cold_restore import _cache
from test_glm_named_control import _worker

P = "prefill-engine"
D = "decode-engine"
P_BOOT = "prefill-current"
D_BOOT = "decode-current"

# The native block table includes accelerator-local groups between wire groups.
TRANSFER_GROUP_IDS = (0, 2, 4, 6)
FINAL_BLOCKS = ((3, 1), (4,), (1,), (5,), (3, 0, 2), (6,), (3,))
CHECKPOINT_BLOCKS = ((), (), (), (), (1,), (), ())


def _native_plan(fill, *, with_checkpoint=True):
    from kitepond.models.glm5next.cold_restore import ColdRestoreLayout
    from kitepond.models.glm5next.layout import CacheRole
    from kitepond.models.glm5next.live_windows import GlmRankWindowPlan

    cache = _cache()
    role_groups = {
        CacheRole.MLA: 0,
        CacheRole.INDEXER: 2,
        CacheRole.KDA_CONV: 4,
        CacheRole.KDA_RECURRENT: 4,
        CacheRole.KPOOL_TAIL: 6,
    }
    groups = {
        name: role_groups[layer.identity.cache_role]
        for name, layer in cache.layers.items()
    }
    for offset, (name, layer) in enumerate(cache.layers.items()):
        assert layer.block_ids == FINAL_BLOCKS[groups[name]]
        for block, view in enumerate(layer.tensor):
            view.view(torch.uint8).fill_(fill + 5 * offset + block)
    layout = replace(
        cache.layout,
        target_valid_tokens=2051,
        draft_valid_tokens=2051,
        committed_kda_slot=cache.layout.committed_kda_slot if with_checkpoint else 0,
    )
    terminal = ColdRestoreLayout(layout, cache.rank, list(cache.layers.values()))
    saved = ColdRestoreLayout(
        replace(
            layout,
            target_valid_tokens=1024,
            draft_valid_tokens=1024,
            committed_kda_slot=0,
        ),
        cache.rank,
        [
            replace(layer, block_ids=CHECKPOINT_BLOCKS[groups[name]])
            for name, layer in cache.layers.items()
            if CHECKPOINT_BLOCKS[groups[name]]
        ],
    )
    return GlmRankWindowPlan(
        terminal, 0, checkpoint=saved if with_checkpoint else None
    ), groups


def _map_native_mla(plan, backing, *, producer=False):
    from kitepond.models.glm5next.cold_restore import ColdRestoreLayout
    from kitepond.models.glm5next.layout import CacheRole
    from kitepond.models.glm5next.live_windows import GlmRankWindowPlan

    original = plan.final
    layers = []
    for layer in original.layers.values():
        if layer.identity.cache_role is CacheRole.MLA:
            mapped = torch.frombuffer(
                backing.mapping,
                dtype=torch.uint8,
                count=layer.tensor.numel(),
            ).view_as(layer.tensor)
            if producer:
                mapped.copy_(layer.tensor)
            layer = replace(layer, tensor=mapped)
        layers.append(layer)
    return GlmRankWindowPlan(
        ColdRestoreLayout(original.layout, original.rank, layers),
        plan.local_tokens,
        checkpoint=plan.checkpoint,
    )


def _profile_connector(stage, rank, plan, groups, *, mla_preflight=False):
    from kitepond.integrations.vllm import glm_prefill_bootstrap as bootstrap

    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.kv_cache_interface import (
        HiSparseHotSpec,
        HiSparseResidentSpec,
        KVCacheGroupSpec,
    )

    def factory(config, cache):
        default = cache.kv_cache_groups[0].kv_cache_spec
        resident = HiSparseResidentSpec(block_size=256, page_size=256 * 656)
        hot = HiSparseHotSpec(block_size=256, page_size=256 * 656, blocks_per_request=1)
        cache = replace(
            cache,
            kv_cache_groups=[
                KVCacheGroupSpec(
                    sorted(
                        {
                            plan.final.layers[name].identity.layer_name
                            for name in groups
                            if groups[name] == group
                        }
                    ),
                    default,
                )
                if group in TRANSFER_GROUP_IDS
                else KVCacheGroupSpec(
                    [f"local.hisparse.{group}"],
                    hot if group == 3 else resident,
                    enable_kv_transfer=False,
                )
                for group in range(len(FINAL_BLOCKS))
            ],
        )
        assert cache.transfer_group_ids == TRANSFER_GROUP_IDS
        transfer = config.kv_transfer_config
        transfer.engine_id = P if stage == "prefill" else D
        transfer.kv_connector_module_path = bootstrap.__name__
        config.additional_config = {
            "kitepond_glm_shared_mla_preflight": mla_preflight,
            "kitepond": {
                "mode": "hisparse-attachment-probe",
                "stage": stage,
                "authority_engine_id": D,
                "boot_id": D_BOOT,
            },
            "kitepond_glm_window": {
                "p_engine_id": P,
                "p_boot_id": P_BOOT,
                "d_engine_id": D,
                "d_boot_id": D_BOOT,
                "decode_host": "127.0.0.1",
            },
        }
        group = SimpleNamespace(world_size=4, rank_in_group=rank)
        with patch.multiple(
            bootstrap, get_tp_group=lambda: group, get_dcp_group=lambda: group
        ):
            return KVConnectorFactory.create_connector(
                config, KVConnectorRole.WORKER, cache
            )

    return _worker("kv_both", rank, create=factory)


def _prefill_source_process(
    pipe, credentials, rank, ready_directory, mla_preflight=False
):
    from multiprocessing.reduction import DupFd

    from kitepond.core.readiness import ReadyAttempt
    from kitepond.integrations.vllm.glm_native_views import (
        GlmNativeStateViews,
        _NativeLayer,
    )
    from kitepond.integrations.vllm.glm_prefill_sources import GlmPrefillExport
    from kitepond.integrations.vllm.hisparse_attachment import HiSparseConnector
    from kitepond.integrations.vllm.prefill_fence import PrefillWrite
    from kitepond.integrations.vllm.ready_transport import ReadyPublisher
    from kitepond.models.glm5next.ready import PrefillWriteGrant
    from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion

    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
        HiSparseConnectorMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
        HiSparseConnectorWorker,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
        MooncakeBootstrapServer,
    )
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import RequestStatus

    os.environ["KITEPOND_GLM_WINDOW_KEYS_FILE"] = credentials
    plan, groups = _native_plan(40, with_checkpoint=not mla_preflight)
    alias = _profile_connector(
        "prefill", rank, plan, groups, mla_preflight=mla_preflight
    )
    native = alias.connector_worker
    assert native.side_channel_port == 0 and native._named_control_only
    bridge = MooncakeBootstrapServer("127.0.0.1", 0)
    bridge.start(timeout=10)
    alias._kv_transfer_config.kv_connector_extra_config["bootstrap_server_address"] = (
        f"127.0.0.1:{bridge.port}"
    )
    ready = threading.Event()
    listener = asyncio.run_coroutine_threadsafe(
        native._mooncake_sender_listener(ready), native.sender_loop
    )
    region = NativeHiSparseMemfdRegion.create(
        4 * 1024**2,
        authority_engine_id=D,
        authority_boot_id=D_BOOT,
        source_rank=rank,
        layout_sha256="c" * 64,
    )
    trace = []

    try:
        assert ready.wait(10), listener.exception(timeout=0)
        if mla_preflight:
            plan = _map_native_mla(plan, region, producer=True)
        borrowed = (region.descriptor, DupFd(region.fd)) if mla_preflight else None
        pipe.send(
            (
                "ready",
                f"http://127.0.0.1:{bridge.port}",
                region.descriptor.fingerprint(),
                borrowed,
            )
        )
        attempt, ready_ticket = pipe.recv()
        source = plan.final
        views = GlmNativeStateViews.__new__(GlmNativeStateViews)
        views.rank = rank
        views.descriptor = region.descriptor
        views._groups = len(FINAL_BLOCKS)
        views._layers = {
            name: _NativeLayer(groups[name], layer.identity, layer.tensor)
            for name, layer in source.layers.items()
        }
        hi = HiSparseConnector.__new__(HiSparseConnector)
        hi._role = KVConnectorRole.WORKER
        hi.stage = "prefill"
        hi._connector_metadata = HiSparseConnectorMetadata(
            None, (), (), {attempt.request_id: (object(),)}, True, {}
        )
        worker = HiSparseConnectorWorker.__new__(HiSparseConnectorWorker)
        hi.connector_worker = worker
        worker.is_host_writer = True
        worker._draft_mirror_pending = True
        worker._draft_layers = (3,)
        worker._post_forward_transfers = []
        worker._dma_submitted = False

        def enqueue(layers):
            trace.append("draft_submit")
            worker._dma_submitted = True

        worker._enqueue_row_dma = enqueue
        worker._submit_transfers = lambda _: None
        worker.host_write_event = SimpleNamespace(
            synchronize=lambda: trace.append("host_fence")
        )
        checkpoint = SimpleNamespace(
            wait_ready=lambda _: (
                trace.append("draft_fence") or SimpleNamespace(epoch=attempt.epoch)
            ),
            require_completed_boundary=lambda _: source.layout.target_valid_tokens,
        )
        identity = ReadyAttempt(
            attempt.request_id, attempt.transfer_id, attempt.epoch, attempt.grant_sha256
        )
        grant = PrefillWriteGrant(
            identity, region.descriptor, source.layout.target_valid_tokens
        )
        write = PrefillWrite(ready_ticket, grant)

        def export(accepted_write, receipt):
            assert accepted_write is write and receipt.attempt == identity
            assert trace == ["draft_submit", "host_fence", "draft_fence"]
            trace.append("export")
            return GlmPrefillExport(
                source.layout,
                plan.local_tokens,
                checkpoint_layout=None
                if plan.checkpoint is None
                else plan.checkpoint.layout,
                checkpoint_blocks=None
                if plan.checkpoint is None
                else CHECKPOINT_BLOCKS,
            )

        alias.bind_prefill_save_fence(
            hi,
            views,
            ReadyPublisher(ready_directory),
            lambda _: grant,
            checkpoint,
            collect_writes=lambda: (write,),
            export=export,
        )
        # Exercise native producer admission plus retained final block metadata.
        pending = mooncake_connector.MooncakeConnectorMetadata()
        pending.reqs_to_send[attempt.request_id] = (attempt.transfer_id, [])

        def native_record(meta):
            asyncio.run_coroutine_threadsafe(
                native.record_send_reqs(meta), native.sender_loop
            ).result(6)

        native_record(pending)
        pipe.send("awaiting-fence")
        assert pipe.recv() == "publish"
        alias.wait_for_save()
        scheduler = mooncake_connector.MooncakeConnectorScheduler(
            native.vllm_config, P, native.kv_cache_config
        )
        request = SimpleNamespace(
            request_id=attempt.request_id,
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
            kv_transfer_params={
                "do_remote_decode": True,
                "transfer_id": attempt.transfer_id,
            },
        )
        hold, _ = scheduler.request_finished(
            request, tuple(list(row) for row in FINAL_BLOCKS)
        )
        assert hold
        complete = scheduler.build_connector_meta(SchedulerOutput.make_empty())
        assert complete.reqs_to_send[attempt.request_id] == (
            attempt.transfer_id,
            [list(FINAL_BLOCKS[group]) for group in TRANSFER_GROUP_IDS],
        )
        native_record(complete)
        pipe.send("native-fence-ready")
        assert pipe.recv() == "preflight-read"
        pending_native = native.reqs_need_send[attempt.transfer_id]
        assert pending_native.ready.is_set() and not pending_native.sending
        assert not alias._sources._writers
        pipe.send("source-remains-unopened")
        assert pipe.recv() == "close"
        sent = alias.get_transfer_results(set()).finished_sending
        assert not native.reqs_need_send
        pipe.send((sent, trace, torch.cuda.is_initialized()))
    finally:
        alias.shutdown()
        native.sender_loop.close()
        native.receiver_loop.close()
        bridge.shutdown()
        region.close()
        pipe.close()


@pytest.mark.parametrize("mode", ["full", "shared", "changed_after_preflight"])
def test_prefill_alias_offers_bytes_only_after_native_fence(
    tmp_path, monkeypatch, mode
):
    mla_preflight = mode != "full"
    changed = mode == "changed_after_preflight"
    import zmq
    from kitepond.core.readiness import ReadyAttempt
    from kitepond.integrations.mooncake.live_receive import (
        RankWindowAttempt,
        mooncake_receive_pool,
    )
    from kitepond.integrations.mooncake.named_control import (
        connect_glm_window_peer,
    )
    from kitepond.integrations.mooncake.window import manifest_digest, plan_windows
    from kitepond.integrations.vllm.decode_receive import (
        DecodeReceiveMailbox,
        DecodeReceiveTicket,
    )
    from kitepond.integrations.vllm.glm_cold_prefill_admission import (
        ColdPrefillOffer,
        ColdPrefillPrompt,
    )
    from kitepond.integrations.vllm.glm_decode_bootstrap import (
        GlmColdImportProfile,
        GlmDecodeImportRuntime,
        forward_payload_receipt,
    )
    from kitepond.integrations.vllm.glm_live_receive import receive_glm_rank_window
    from kitepond.integrations.vllm.glm_native_views import (
        GlmNativeStateViews,
        _NativeLayer,
    )
    from kitepond.integrations.vllm.host_authority import NativeHostGrant
    from kitepond.integrations.vllm.ready_transport import ReadyCollector, ReadyWorker
    from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector
    from vllm.utils.network_utils import make_zmq_socket

    pytest.importorskip("mooncake.engine", exc_type=ImportError)
    pytest.importorskip(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.named_control"
    )
    keys = tmp_path / "keys.json"
    keys.write_text(
        json.dumps(
            {
                "schema": "kitepond.glm-window-keys.v1",
                "p_engine_id": P,
                "p_boot_id": P_BOOT,
                "d_engine_id": D,
                "d_boot_id": D_BOOT,
                "rank_keys": {str(rank): secrets.token_hex(32) for rank in range(4)},
            }
        )
    )
    keys.chmod(0o600)
    monkeypatch.setenv("KITEPOND_GLM_WINDOW_KEYS_FILE", str(keys))
    ready_directory = tmp_path / "ready"
    ready_directory.mkdir(mode=0o700)

    plan, groups = _native_plan(100, with_checkpoint=not mla_preflight)
    expected, _ = _native_plan(40, with_checkpoint=not mla_preflight)
    cache = plan.final
    rank = cache.rank
    decoder = _profile_connector("decode", rank, plan, groups)
    scope = decoder.window_scope()
    native = decoder.connector_worker
    pool = mooncake_receive_pool(native.engine, 2)
    prompt = ColdPrefillPrompt.from_tokens(
        cache.layout.request_id,
        "authorized-transfer",
        cache.layout.epoch,
        list(range(cache.layout.target_valid_tokens + 1)),
    )
    host_grant = NativeHostGrant(
        D,
        D_BOOT,
        prompt.request_id,
        prompt.transfer_id,
        prompt.epoch,
        0,
        prompt.tokens,
        (),
        FINAL_BLOCKS[0],
        None,
    )
    offer = ColdPrefillOffer(prompt, host_grant, P_BOOT, (1 << 63) - 1)
    ticket = DecodeReceiveTicket(
        cache.layout.request_id,
        "authorized-transfer",
        D,
        D_BOOT,
        cache.layout.epoch,
        host_grant.sha256,
        cache.layout.target_valid_tokens,
        FINAL_BLOCKS,
        0,
        None,
    )
    attempt = RankWindowAttempt(
        ticket.request_id,
        ticket.transfer_id,
        ticket.epoch,
        rank,
        scope.source_boot_id,
        ticket.grant_sha256,
        ticket.destination_sha256,
        manifest_digest(plan.manifest()),
    )
    mailbox = DecodeReceiveMailbox(rank, D_BOOT)
    ctx = multiprocessing.get_context("spawn")
    pipe, source = ctx.Pipe()
    child = ctx.Process(
        target=_prefill_source_process,
        args=(source, str(keys), rank, ready_directory, mla_preflight),
    )
    child.start()
    source.close()

    def read():
        assert pipe.poll(35), "Native P READY process did not respond"
        return pipe.recv()

    async def gate(bootstrap):
        address = await native.resolve_named_control_peer(P, bootstrap)
        with make_zmq_socket(
            native.async_zmq_ctx, address, zmq.DEALER, bind=False, linger=0
        ) as stock:
            packet = mooncake_connector.MooncakeXferMetadata(
                remote_hostname="127.0.0.1",
                remote_port=native.rpc_port,
                remote_tp_size=4,
                remote_tp_rank=rank,
                req_blocks={ticket.request_id: (ticket.transfer_id, [[1]])},
                kv_caches_base_addr=[],
                block_lens=[],
                kv_block_lens=[],
            )
            await stock.send(native._encoder.encode(packet))
            failure = native._xfer_resp_decoder.decode(await stock.recv())
            assert failure.err_msg == "This Mooncake producer requires named control"

        window = plan_windows(plan.manifest())[0]
        lease = await pool.acquire(5)
        try:
            with pytest.raises(ValueError, match="authentication failed"):
                async with connect_glm_window_peer(native, P, bootstrap, scope) as peer:
                    await peer.fetch(
                        attempt, window.to_metadata(), lease.descriptor(), -1
                    )
        finally:
            # The fixture knows P had no source export and never issued a write.
            await pool.release(lease, writes_settled=True)

    async def prepare(bootstrap):
        async with connect_glm_window_peer(native, P, bootstrap, scope) as peer:
            return await peer.prepare_mla(attempt)

    async def wrong_grant(bootstrap):
        altered = replace(attempt, grant_sha256="e" * 64)
        lease = await pool.acquire(5)
        try:
            window = plan_windows(plan.manifest())[0]
            with pytest.raises(ValueError, match="authentication failed"):
                async with connect_glm_window_peer(native, P, bootstrap, scope) as peer:
                    await peer.fetch(
                        altered, window.to_metadata(), lease.descriptor(), -1
                    )
        finally:
            await pool.release(lease, writes_settled=True)

    async def transfer(bootstrap):
        async with connect_glm_window_peer(native, P, bootstrap, scope) as peer:
            await receive_glm_rank_window(
                mailbox,
                pool,
                ticket,
                plan,
                groups,
                peer,
                source_boot_id=scope.source_boot_id,
                checkpoint_blocks=None
                if plan.checkpoint is None
                else CHECKPOINT_BLOCKS,
                timeout=6,
            )

    def on_decode(operation):
        return asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(operation, 20), native.receiver_loop
        ).result(25)

    collector = ReadyCollector(ready_directory, {rank: ReadyWorker.from_pid(child.pid)})
    borrowed_region = None
    try:
        state, bootstrap, region_fingerprint, borrowed = read()
        assert state == "ready"
        ready_attempt = ReadyAttempt(
            attempt.request_id, attempt.transfer_id, attempt.epoch, attempt.grant_sha256
        )
        ready_ticket = collector.arm(
            ready_attempt, {rank: region_fingerprint}, cache.layout.target_valid_tokens
        )
        pipe.send((attempt, ready_ticket))
        assert read() == "awaiting-fence"
        on_decode(gate(bootstrap))
        assert on_decode(prepare(bootstrap)) is None
        assert not mailbox.drain(None).receipts
        pipe.send("publish")
        assert read() == "native-fence-ready"
        (source_proof,) = collector.take(ready_ticket)
        assert source_proof.attempt == ready_attempt
        assert source_proof.region_sha256 == region_fingerprint
        proof = on_decode(prepare(bootstrap))
        assert (
            (type(proof) is str and len(proof) == 64)
            if mla_preflight
            else proof is None
        )
        pipe.send("preflight-read")
        assert read() == "source-remains-unopened"
        on_decode(wrong_grant(bootstrap))
        if mla_preflight:
            descriptor, fd = borrowed
            borrowed_region = NativeHiSparseMemfdRegion.join_received_fd(
                descriptor,
                fd.detach(),
                authority_engine_id=D,
                authority_boot_id=D_BOOT,
                source_rank=rank,
                layout_sha256=descriptor.layout_sha256,
            )
            full_shared = _map_native_mla(plan, borrowed_region)
            local = GlmNativeStateViews.__new__(GlmNativeStateViews)
            local.rank = rank
            local.descriptor = descriptor
            local._groups = len(FINAL_BLOCKS)
            local._layers = {
                name: _NativeLayer(groups[name], layer.identity, layer.tensor)
                for name, layer in full_shared.final.layers.items()
            }
            runtime = GlmDecodeImportRuntime.__new__(GlmDecodeImportRuntime)
            runtime.worker, runtime.views, runtime.scope = native, local, scope
            runtime.p_engine_id = P
            runtime.profile = GlmColdImportProfile(bootstrap, 6)
            runtime._model_sha256 = cache.layout.model_fingerprint
            runtime._config_sha256 = cache.layout.config_fingerprint
            observation = runtime.prepare_shared_mla(offer)
            assert observation is not None and observation.sha256 == proof
            # The scheduler all-rank quorum is tested with four true physical
            # mappings separately. This TCP fixture exercises one enrolled rank.
            ordered = ["f" * 64] * 4
            ordered[rank] = ("1" * 64) if changed else observation.sha256
            ticket = replace(ticket, mla_rank_digests=tuple(ordered))
            if changed:
                with pytest.raises(ValueError, match="rank changed") as mismatch:
                    runtime._plan(ticket)
                runtime._hisparse = SimpleNamespace(
                    decode_receive_mailbox=lambda: mailbox
                )
                runtime._receiver = SimpleNamespace(_pool=pool)
                mailbox.bind((ticket,))
                with pytest.raises(ValueError, match="rank changed"):
                    on_decode(runtime._abort_unopened_preflight(ticket, mismatch.value))
                (abort,) = mailbox.drain(None).receipts
                assert (abort.rank, abort.outcome) == (rank, "abort_settled")
            else:
                plan, groups = runtime._plan(ticket)
                assert len(plan.manifest()) < len(full_shared.manifest())
                before_bytes = sum(row["nbytes"] for row in full_shared.manifest())
                after_bytes = sum(row["nbytes"] for row in plan.manifest())
                from kitepond.models.glm5next.cold_restore import native_rank_rows
                from kitepond.models.glm5next.layout import CacheRole

                assert (
                    before_bytes - after_bytes
                    == native_rank_rows(ticket.imported_tokens, CacheRole.MLA, rank)
                    * 656
                )
        if not changed:
            mailbox.bind((ticket,))
            on_decode(transfer(bootstrap))
            (proof,) = mailbox.drain(None).receipts
            assert (proof.rank, proof.outcome) == (rank, "loaded")
            receipt = forward_payload_receipt(ticket, plan)
            assert receipt["rank"] == rank
            assert receipt["committed_payload_bytes"] == sum(
                row["nbytes"] for row in plan.manifest()
            )
            if mla_preflight:
                assert receipt["full_equivalent_bytes"] == before_bytes
                assert receipt["shared_mla_omitted_bytes"] == before_bytes - after_bytes
                assert receipt["payload_bytes_by_role"]["mla"] == 0
            else:
                assert receipt["shared_mla_omitted_bytes"] == 0
                assert receipt["payload_bytes_by_role"]["mla"] > 0
            observed = full_shared if mla_preflight else plan
            for window in plan_windows(observed.manifest()):
                payload = torch.empty(window.nbytes, dtype=torch.uint8)
                assert expected.pack(window, payload) == window.nbytes
                observed.verify_readback(window, payload)
        pool.close()
        pipe.send("close")
        sent, trace, initialized = read()
        assert sent == {ticket.request_id}
        assert trace == ["draft_submit", "host_fence", "draft_fence", "export"]
        assert not initialized and not torch.cuda.is_initialized()
    finally:
        collector.close()
        decoder.shutdown()
        native.receiver_loop.close()
        native.sender_loop.close()
        child.join(4)
        if child.is_alive():
            child.terminate()
            child.join(4)
        pipe.close()
        if borrowed_region is not None:
            borrowed_region.close()
    assert child.exitcode == 0
