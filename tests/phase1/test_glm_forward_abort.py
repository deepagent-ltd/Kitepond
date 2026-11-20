# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify signed forward abort and native Mooncake return with GPUs hidden."""

import asyncio
import secrets
import threading
import time

import pytest
import torch
from kitepond.core.readiness import ReadyAttempt, ReadyReceipt
from kitepond.integrations.mooncake.live_receive import (
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
from kitepond.integrations.mooncake.window import (
    RegisteredWindowPool,
    manifest_digest,
    plan_windows,
)
from kitepond.integrations.vllm.glm_prefill_sources import (
    GlmPrefillExport,
    GlmPrefillReadySources,
)
from test_glm_cold_restore import _live_model_plan
from test_glm_reverse_tcp import _native

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    SendBlockMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
)


class _TransferObserver:
    def __init__(self, engine, delay):
        self.engine = engine
        self.delay = delay
        self.timeline = []

    def __getattr__(self, key):
        return getattr(self.engine, key)

    def batch_transfer_sync_write(self, *args):
        self.timeline.append("write-start")
        time.sleep(self.delay)
        outcome = self.engine.batch_transfer_sync_write(*args)
        self.timeline.append("write-done")
        assert outcome == 0
        return outcome


class _NeverScatter:
    def __init__(self, timeline):
        self.timeline = timeline

    def fence(self):
        self.timeline.append("D-fence")

    def scatter(self, *_):
        raise AssertionError("A failed fetch must not scatter")

    def finish(self):
        raise AssertionError("A failed fetch must not finish")


@pytest.mark.parametrize("late", [False, True])
def test_real_forward_abort_with_and_without_active_writer(late):
    pytest.importorskip("mooncake.engine", exc_type=ImportError)
    pytest.importorskip(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.named_control"
    )
    source_plan = _live_model_plan(40)
    layout = source_plan.final.layout
    rank = source_plan.final.rank
    scope = GlmWindowControlScope(
        "P-forward", "D-forward", rank, secrets.token_bytes(32), forward_abort=True
    )
    attempt = RankWindowAttempt(
        layout.request_id,
        "D-forward-test",
        layout.epoch,
        rank,
        scope.source_boot_id,
        "a" * 64,
        "b" * 64,
        manifest_digest(source_plan.manifest()),
    )
    sender = _native(scope, "P")
    observed = _TransferObserver(sender.engine, 0.90)
    sender.engine = observed
    receiver = _native(scope, "D")
    source = GlmPrefillReadySources(sender, "127.0.0.1")
    pool = mooncake_receive_pool(receiver.engine, 2)
    send = SendBlockMeta(
        attempt.request_id, attempt.transfer_id, [[2]], asyncio.Event()
    )
    send.ready.set()
    sender.reqs_need_send[attempt.transfer_id] = send
    proof = ReadyReceipt(
        ReadyAttempt(
            attempt.request_id, attempt.transfer_id, attempt.epoch, attempt.grant_sha256
        ),
        rank,
        "f" * 64,
        layout.target_valid_tokens,
        layout.draft_valid_tokens,
        time.monotonic_ns(),
    )
    source._fenced[attempt.transfer_id] = (proof, GlmPrefillExport(layout, 0))

    async def refused(*_):
        raise ValueError("Known P fixture pre-write source error")

    GlmWindowControlHandler(
        sender,
        scope,
        source.lookup,
        admit_source=refused,
        cancel_unopened=source.cancel_unopened,
        reject_native_requests=True,
    )

    if late:

        def release():
            assert sender.reqs_need_send.pop(attempt.transfer_id) is send
            sender.finished_sending_reqs.add(send.p_req_id)
            source._writers.pop(attempt.transfer_id)
            source._fenced.pop(attempt.transfer_id)
            observed.timeline.append("P-retired")

        source._writers[attempt.transfer_id] = NativeRankWindowWriter(
            attempt,
            source_plan.manifest(),
            sender.engine,
            source.pool,
            destination=(receiver.hostname, receiver.rpc_port, pool.pool_id),
            pack=source_plan.pack,
            release_source=release,
            timeout=5,
        )
        send.sending = 1

    bootstrap = MooncakeBootstrapServer("127.0.0.1", 0)
    bootstrap.start(timeout=10)
    sender.vllm_config.kv_transfer_config.kv_connector_extra_config[
        "bootstrap_server_address"
    ] = f"127.0.0.1:{bootstrap.port}"
    started = threading.Event()
    listener = asyncio.run_coroutine_threadsafe(
        sender._mooncake_sender_listener(started), sender.sender_loop
    )

    async def receive():
        outcomes = []
        async with connect_glm_window_peer(
            receiver, "P", f"http://127.0.0.1:{bootstrap.port}", scope
        ) as peer:
            with pytest.raises(TimeoutError if late else ValueError):
                await receive_rank_windows(
                    attempt,
                    source_plan.manifest(),
                    pool,
                    peer,
                    _NeverScatter(observed.timeline),
                    outcomes.append,
                    timeout=0.67 if late else 5,
                    forward_abort_peer=peer,
                )
            assert outcomes == ["abort_settled"]
            assert pool.leased_slots == pool.quarantined_slots == 0
            assert attempt.transfer_id not in sender.reqs_need_send
            assert attempt.transfer_id not in source._fenced
            assert sender.finished_sending_reqs == {attempt.request_id}
            count = len(observed.timeline)
        async with connect_glm_window_peer(
            receiver, "P", f"http://127.0.0.1:{bootstrap.port}", scope
        ) as fresh_peer:
            slot = await pool.acquire(1)
            with pytest.raises(ValueError, match="authentication"):
                await fresh_peer.fetch(
                    attempt,
                    plan_windows(source_plan.manifest())[0].to_metadata(),
                    slot.descriptor(),
                    -1,
                )
            assert len(observed.timeline) == count
            await pool.release(slot, writes_settled=True)

    try:
        assert started.wait(10), listener.exception(timeout=0)
        asyncio.run_coroutine_threadsafe(receive(), receiver.receiver_loop).result(
            timeout=30
        )
        assert observed.timeline == (
            ["write-start", "write-done", "P-retired", "D-fence"]
            if late
            else ["D-fence"]
        )
        assert not torch.cuda.is_initialized()
    finally:
        pool.close()
        source.close()
        receiver.shutdown()
        sender.shutdown()
        bootstrap.shutdown()
        receiver.receiver_loop.close()
        receiver.sender_loop.close()
        sender.receiver_loop.close()
        sender.sender_loop.close()


def test_missing_authenticated_forward_abort_remains_quarantined():
    plan = _live_model_plan(40)
    attempt = RankWindowAttempt(
        "request",
        "unproved",
        0,
        0,
        "P",
        "a" * 64,
        "b" * 64,
        manifest_digest(plan.manifest()),
    )
    pool = RegisteredWindowPool(2, lambda _: None, lambda _: None)
    observations = []

    class NoProof:
        async def fetch(self, *_):
            raise ValueError("Unsigned native failure")

        async def abort(self, *_):
            raise ConnectionError("The named sender was lost")

    async def run():
        peer = NoProof()
        with pytest.raises(ConnectionError, match="lost"):
            await receive_rank_windows(
                attempt,
                plan.manifest(),
                pool,
                peer,
                _NeverScatter([]),
                observations.append,
                timeout=1,
                forward_abort_peer=peer,
            )
        assert not observations
        assert pool.quarantined_slots == 1 and pool.leased_slots == 0
        remaining = await pool.acquire(1)
        assert remaining.slot == 1
        await pool.release(remaining, writes_settled=True)
        with pytest.raises(RuntimeError, match="live or uncertain"):
            pool.close()

    asyncio.run(run())
