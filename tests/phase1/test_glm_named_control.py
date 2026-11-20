# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run authenticated GLM control over the native Mooncake peer sockets."""

import asyncio
import multiprocessing
import secrets
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_glm_cold_restore import _live_model_plan


def _worker(role, rank, *, create=None):
    from vllm.config import CacheConfig, KVTransferConfig, VllmConfig
    from vllm.config.scheduler import SchedulerConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )

    config = VllmConfig(
        cache_config=CacheConfig(block_size=16, enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            max_model_len=32,
            max_num_batched_tokens=32,
            max_num_seqs=2,
            is_encoder_decoder=False,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="MooncakeConnector",
            kv_role=role,
            engine_id=role,
            kv_connector_extra_config={
                "mooncake_protocol": "tcp",
                "bootstrap_server_address": "127.0.0.1:1",
                "num_workers": 1,
            },
        ),
    )
    config.model_config = SimpleNamespace(
        use_mla=False, get_total_num_kv_heads=lambda: 1
    )
    cache = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["model.layers.0"],
                FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32
                ),
            )
        ],
    )
    with (
        patch.object(torch.accelerator, "current_device_index", return_value=0),
        patch.object(mooncake_connector.current_platform, "set_device"),
        patch.multiple(
            mooncake_connector,
            get_ip=lambda: "127.0.0.1",
            get_tensor_model_parallel_rank=lambda: rank,
            get_tensor_model_parallel_world_size=lambda: 4,
            get_pp_group=lambda: SimpleNamespace(rank_in_group=0),
            get_current_attn_backends=lambda config: [],
            should_launch_bootstrap_server=lambda config: False,
        ),
    ):
        if create is not None:
            return create(config, cache)
        return mooncake_connector.MooncakeConnectorWorker(config, role, cache)


def _source_process(pipe, scope):
    from kitepond.integrations.mooncake.live_receive import (
        NativeRankWindowWriter,
        mooncake_receive_pool,
    )
    from kitepond.integrations.mooncake.named_control import GlmWindowControlHandler

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
        MooncakeBootstrapServer,
    )

    native = _worker("kv_producer", scope.rank)
    pool = mooncake_receive_pool(native.engine, 1)
    source_plan = _live_model_plan(40)
    sources, released, ordinary = {}, [], []

    bootstrap = MooncakeBootstrapServer("127.0.0.1", 0)
    bootstrap.start(timeout=10)
    transfer = native.vllm_config.kv_transfer_config
    transfer.kv_connector_extra_config["bootstrap_server_address"] = (
        f"127.0.0.1:{bootstrap.port}"
    )

    async def ordinary_fixture(identity, socket, metadata):
        ordinary.append(tuple(metadata.req_blocks))
        await socket.send_multipart(
            (
                identity,
                native._encoder.encode(
                    mooncake_connector.MooncakeXferResponse(
                        status=mooncake_connector.MooncakeXferResponseStatus.FINISH,
                        ok_reqs=list(metadata.req_blocks),
                    )
                ),
            )
        )

    native.send_kv_to_decode = ordinary_fixture
    GlmWindowControlHandler(native, scope, sources.get)
    started = threading.Event()
    listener = asyncio.run_coroutine_threadsafe(
        native._mooncake_sender_listener(started), native.sender_loop
    )
    try:
        assert started.wait(10), listener.exception(timeout=0)
        pipe.send(("ready", f"http://127.0.0.1:{bootstrap.port}"))
        attempt, destination = pipe.recv()
        sources[attempt] = NativeRankWindowWriter(
            attempt,
            source_plan.manifest(),
            native.engine,
            pool,
            destination=destination,
            pack=source_plan.pack,
            release_source=lambda: released.append(attempt.rank),
            timeout=5,
        )
        pipe.send("bound")
        assert pipe.recv() == "close"
        pool.close()
    finally:
        native.shutdown()
        native.sender_loop.close()
        bootstrap.shutdown()
        assert bootstrap.server_thread is None
        pipe.send((released, ordinary, torch.cuda.is_initialized()))
        pipe.close()


def test_native_glm_control_tcp_handshake_and_authentication():
    import zmq
    from kitepond.integrations.mooncake.live_receive import (
        RankWindowAttempt,
        mooncake_receive_pool,
    )
    from kitepond.integrations.mooncake.named_control import (
        NAMESPACE,
        GlmWindowControlScope,
        bind_glm_window_peer,
        connect_glm_window_peer,
    )
    from kitepond.integrations.mooncake.window import manifest_digest, plan_windows
    from kitepond.integrations.vllm.decode_receive import (
        DecodeReceiveMailbox,
        DecodeReceiveTicket,
    )
    from kitepond.integrations.vllm.glm_live_receive import receive_glm_rank_window

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector
    from vllm.utils.network_utils import make_zmq_socket

    pytest.importorskip("mooncake.engine", exc_type=ImportError)
    framing = pytest.importorskip(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.named_control"
    )
    target = _live_model_plan(120)
    cache = target.final
    rank = cache.rank
    scope = GlmWindowControlScope("P-boot", "D-boot", rank, secrets.token_bytes(32))
    native = _worker("kv_consumer", rank)
    pool = mooncake_receive_pool(native.engine, 2)
    ticket = DecodeReceiveTicket(
        cache.layout.request_id,
        "worker-rank-transfer",
        "D-owner",
        scope.decode_boot_id,
        cache.layout.epoch,
        "a" * 64,
        cache.layout.target_valid_tokens,
        tuple(layer.block_ids for layer in cache.layers.values()),
        1024,
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
        manifest_digest(target.manifest()),
    )
    groups = {name: idx for idx, name in enumerate(cache.layers)}
    mailbox = DecodeReceiveMailbox(rank, scope.decode_boot_id)
    mailbox.bind((ticket,))
    ctx = multiprocessing.get_context("spawn")
    pipe, source_pipe = ctx.Pipe()
    child = ctx.Process(target=_source_process, args=(source_pipe, scope))
    child.start()
    source_pipe.close()

    def rendezvous():
        assert pipe.poll(35), "Native CPU control process did not respond"
        return pipe.recv()

    async def run(bootstrap_address):
        address = await native.resolve_named_control_peer(
            "kv_producer", bootstrap_address
        )
        assert address == native._remote_agents["kv_producer"][rank][0]
        with make_zmq_socket(
            native.async_zmq_ctx, address, zmq.DEALER, bind=False, linger=0
        ) as sock:
            native_request = mooncake_connector.MooncakeXferMetadata(
                remote_hostname="127.0.0.1",
                remote_port=native.rpc_port,
                remote_tp_size=4,
                remote_tp_rank=rank,
                req_blocks={"stock": ("normal", [[1]])},
                kv_caches_base_addr=[],
                block_lens=[],
                kv_block_lens=[],
            )
            await sock.send(native._encoder.encode(native_request))
            ordinary = native._xfer_resp_decoder.decode(await sock.recv())
            assert (
                ordinary.status == mooncake_connector.MooncakeXferResponseStatus.FINISH
            )
            assert ordinary.ok_reqs == ["stock"]

        test_window = plan_windows(target.manifest())[0]
        fake = GlmWindowControlScope(
            scope.source_boot_id, scope.decode_boot_id, rank, secrets.token_bytes(32)
        )
        lease = await pool.acquire(5)
        try:
            with (
                bind_glm_window_peer(native, address, fake) as rejected,
                pytest.raises(ValueError, match="authentication failed"),
            ):
                await asyncio.wait_for(
                    rejected.fetch(
                        attempt, test_window.to_metadata(), lease.descriptor(), -1
                    ),
                    6,
                )
        finally:
            await pool.release(lease, writes_settled=True)

        stale = GlmWindowControlScope(
            scope.source_boot_id, "earlier-D-boot", rank, scope.key
        )
        stale_packet = stale.seal(
            b"fetcher", attempt, "commit", 1, {"last_sequence": 0}
        )
        with make_zmq_socket(
            native.async_zmq_ctx, address, zmq.DEALER, bind=False, linger=0
        ) as sock:
            await sock.send(
                framing.encode_mooncake_control_message(NAMESPACE, stale_packet)
            )
            no_boot_match = native._xfer_resp_decoder.decode(await sock.recv())
            assert (
                no_boot_match.status
                == mooncake_connector.MooncakeXferResponseStatus.ERROR
            )
            assert no_boot_match.err_msg == "Invalid GLM Mooncake control envelope"

        async with connect_glm_window_peer(
            native, "kv_producer", bootstrap_address, scope
        ) as peer:
            await receive_glm_rank_window(
                mailbox,
                pool,
                ticket,
                target,
                groups,
                peer,
                source_boot_id=scope.source_boot_id,
                timeout=6,
            )
        (proof,) = mailbox.drain(None).receipts
        assert (proof.rank, proof.outcome) == (rank, "loaded")

        packet = scope.seal(
            b"fetcher",
            attempt,
            "commit",
            1,
            {"last_sequence": 0},
        )
        with make_zmq_socket(
            native.async_zmq_ctx, address, zmq.DEALER, bind=False, linger=0
        ) as sock:
            await sock.send(framing.encode_mooncake_control_message(NAMESPACE, packet))
            reject = native._xfer_resp_decoder.decode(await sock.recv())
            assert reject.status == mooncake_connector.MooncakeXferResponseStatus.ERROR
            assert reject.err_msg == "GLM Mooncake control replay"

    try:
        ready, address = rendezvous()
        assert ready == "ready"
        pipe.send((attempt, ("127.0.0.1", native.rpc_port, pool.pool_id)))
        assert rendezvous() == "bound"
        task = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(run(address), 22), native.receiver_loop
        )
        task.result(30)
        pool.close()
        pipe.send("close")
        released, ordinary, initialized = rendezvous()
        assert released == [rank] and ordinary == [("stock",)]
        assert not initialized and not torch.cuda.is_initialized()
    finally:
        native.shutdown()
        native.receiver_loop.close()
        child.join(4)
        if child.is_alive():
            child.terminate()
            child.join(4)
        pipe.close()
    assert child.exitcode == 0
