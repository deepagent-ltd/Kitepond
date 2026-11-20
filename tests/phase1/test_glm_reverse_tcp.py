# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise both vLLM Mooncake endpoints and real reverse CPU byte writes."""

import asyncio
import multiprocessing
import secrets
import threading
import time
from dataclasses import replace

import pytest
import torch
from kitepond.integrations.mooncake.live_receive import (
    FencedRankWindows,
    NativeRankWindowWriter,
    RankWindowAttempt,
    mooncake_receive_pool,
    receive_rank_windows,
)
from kitepond.integrations.mooncake.named_control import (
    GlmWindowControlHandler,
    GlmWindowControlScope,
    connect_glm_window_peer,
)
from kitepond.integrations.mooncake.window import manifest_digest
from kitepond.models.glm5next.live_windows import GlmRankWindowPlan
from test_glm_cold_restore import _live_model_plan
from test_glm_named_control import _worker


def _native(scope, engine_id):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector

    def create(config, cache):
        config.kv_transfer_config.engine_id = engine_id
        return mooncake_connector.MooncakeConnectorWorker(config, engine_id, cache)

    return _worker("kv_both", scope.rank, create=create)


def _source(pipe, scope, late, shared=None):
    from kitepond.models.glm5next.cold_restore import ColdRestoreLayout
    from kitepond.models.glm5next.layout import CacheRole
    from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
        MooncakeBootstrapServer,
    )

    worker = _native(scope, "D" if scope.reverse else "P")
    pool = mooncake_receive_pool(worker.engine, 1)
    base = _live_model_plan(40).final
    region = None
    if shared is not None:
        descriptor, ref = shared
        region = NativeHiSparseMemfdRegion.join_received_fd(
            descriptor,
            ref.detach(),
            authority_engine_id=descriptor.authority_engine_id,
            authority_boot_id=descriptor.authority_boot_id,
            source_rank=descriptor.source_rank,
            layout_sha256=descriptor.layout_sha256,
        )
        base = ColdRestoreLayout(
            replace(base.layout, committed_kda_slot=0),
            base.rank,
            [
                replace(layer, tensor=region.tensor().view(layer.tensor.shape))
                if layer.identity.cache_role is CacheRole.MLA
                else layer
                for layer in base.layers.values()
            ],
        )
    source = GlmRankWindowPlan(base, 0)
    routes, settled, closed, writes = {}, [], [], []
    bootstrap = MooncakeBootstrapServer("127.0.0.1", 0)
    bootstrap.start(timeout=10)
    worker.vllm_config.kv_transfer_config.kv_connector_extra_config[
        "bootstrap_server_address"
    ] = f"127.0.0.1:{bootstrap.port}"

    async def accept(proof):
        assert closed == [proof.attempt.rank]
        settled.append(proof)

    GlmWindowControlHandler(
        worker, scope, routes.get, settled=accept if scope.reverse else None
    )
    started = threading.Event()
    listener = asyncio.run_coroutine_threadsafe(
        worker._mooncake_sender_listener(started), worker.sender_loop
    )
    try:
        assert started.wait(10), listener.exception(timeout=0)
        pipe.send(f"http://127.0.0.1:{bootstrap.port}")
        attempt, destination = pipe.recv()

        def pack(window, payload):
            if late:
                time.sleep(0.9)
            written = source.pack(window, payload)
            writes.append(written)
            return written

        routes[attempt] = NativeRankWindowWriter(
            attempt,
            source.manifest(),
            worker.engine,
            pool,
            destination=destination,
            pack=pack,
            release_source=lambda: closed.append(attempt.rank),
            timeout=5,
        )
        pipe.send("registered")
        assert pipe.recv() == "done"
        assert routes[attempt].session.next_sequence >= 1
        assert pool.leased_slots == pool.quarantined_slots == 0
        pool.close()
    finally:
        worker.shutdown()
        worker.sender_loop.close()
        worker.receiver_loop.close()
        bootstrap.shutdown()
        pipe.send((settled, closed, writes, torch.cuda.is_initialized()))
        pipe.close()


@pytest.mark.parametrize("late", [False, True])
def test_native_reverse_route_and_safe_timeout(late):
    from kitepond.integrations.vllm.glm_live_receive import _NativeGlmSink

    pytest.importorskip("mooncake.engine", exc_type=ImportError)
    pytest.importorskip(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.named_control"
    )
    assert not torch.cuda.is_initialized()
    plan = GlmRankWindowPlan(_live_model_plan(120).final, 0)
    rank = plan.final.rank
    scope = GlmWindowControlScope(
        "P-boot", "D-boot", rank, secrets.token_bytes(32), reverse=True
    )
    attempt = RankWindowAttempt(
        plan.final.layout.request_id,
        "reverse-rank-transfer",
        plan.final.layout.epoch,
        rank,
        scope.decode_boot_id,
        "a" * 64,
        "b" * 64,
        manifest_digest(plan.manifest()),
    )
    forward = replace(scope, reverse=False)
    with pytest.raises(ValueError, match="qualified source rank"):
        forward.seal(b"fetcher", attempt, "commit", 1, {"last_sequence": 0})
    native = _native(scope, "P")
    pool = mooncake_receive_pool(native.engine, 2)
    ctx = multiprocessing.get_context("spawn")
    pipe, source_pipe = ctx.Pipe()
    child = ctx.Process(target=_source, args=(source_pipe, scope, late))
    child.start()
    source_pipe.close()
    proofs = []
    reads = []

    def recv():
        assert pipe.poll(35), "Native reverse TCP worker did not respond"
        return pipe.recv()

    async def run(bootstrap):
        async with connect_glm_window_peer(native, "D", bootstrap, scope) as peer:
            assert native._remote_agents["D"][rank][0].startswith("tcp://")
            sink = _NativeGlmSink(
                plan, None, None, lambda: reads.append("seeded"), needs_cow=False
            )
            transfer = receive_rank_windows(
                attempt,
                plan.manifest(),
                pool,
                peer,
                sink,
                proofs.append,
                timeout=0.67 if late else 5,
                reverse_peer=peer,
            )
            if late:
                with pytest.raises(TimeoutError):
                    await transfer
                assert proofs == ["abort_settled"]
                assert not reads
            else:
                await transfer
                assert proofs == ["loaded"]
                assert reads == ["seeded"]
            assert pool.quarantined_slots == pool.leased_slots == 0

    try:
        bootstrap = recv()
        pipe.send((attempt, (native.hostname, native.rpc_port, pool.pool_id)))
        assert recv() == "registered"
        asyncio.run_coroutine_threadsafe(run(bootstrap), native.receiver_loop).result(
            timeout=28
        )
        pipe.send("done")
        settled, closed, writes, cuda = recv()
        expected = "abort_settled" if late else "acked"
        assert settled == [FencedRankWindows(attempt, expected)]
        assert closed == [rank] and sum(writes) > 1024 and not cuda
        assert not torch.cuda.is_initialized()
        child.join(timeout=10)
        assert child.exitcode == 0
    finally:
        pool.close()
        native.shutdown()
        native.receiver_loop.close()
        native.sender_loop.close()
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
        pipe.close()


def test_disconnect_never_recycles_a_window_with_uncertain_remote_write():
    from kitepond.integrations.mooncake.window import RegisteredWindowPool

    plan = GlmRankWindowPlan(_live_model_plan(120).final, 0)
    attempt = RankWindowAttempt(
        plan.final.layout.request_id,
        "lost",
        plan.final.layout.epoch,
        plan.final.rank,
        "D-boot",
        "a" * 64,
        "b" * 64,
        manifest_digest(plan.manifest()),
    )
    pool = RegisteredWindowPool(2, lambda tensor: None, lambda tensor: None)
    observations = []

    class Unreachable:
        async def fetch(self, *args):
            await asyncio.sleep(30)

        async def abort(self, *args):
            await asyncio.sleep(30)

    async def run():
        peer = Unreachable()
        with pytest.raises(TimeoutError):
            await receive_rank_windows(
                attempt,
                plan.manifest(),
                pool,
                peer,
                object(),
                observations.append,
                timeout=0.01,
                reverse_peer=peer,
            )
        assert not observations
        assert pool.quarantined_slots == 1 and pool.leased_slots == 0
        healthy = await pool.acquire(1)
        assert healthy.slot == 1
        await pool.release(healthy, writes_settled=True)
        with pytest.raises(RuntimeError, match="live or uncertain"):
            pool.close()

    asyncio.run(run())
