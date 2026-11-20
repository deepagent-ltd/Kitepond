# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fault a real authentically routed Mooncake TCP write acknowledgement."""

import asyncio
import json
import multiprocessing
import secrets
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_glm_cold_restore import _live_model_plan
from test_glm_named_control import _worker

_RANK = 2


def _producer(pipe, scope, folder):
    from kitepond.integrations.mooncake.live_receive import (
        NativeRankWindowWriter,
        mooncake_receive_pool,
    )
    from kitepond.integrations.mooncake.named_control import GlmWindowControlHandler
    from kitepond.integrations.mooncake.window import manifest_digest
    from lost_write_ack_injector import install_missing_write_ack

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
        MooncakeBootstrapServer,
    )

    native = _worker("kv_producer", scope.rank)
    pool = mooncake_receive_pool(native.engine, 1)
    plan = _live_model_plan(40)
    registered, released = {}, []

    async def no_unopened_cleanup(attempt, destination):
        # Both exact sources are registered by this worker; no raw fallback.
        return False

    GlmWindowControlHandler(
        native, scope, registered.get, cancel_unopened=no_unopened_cleanup
    )
    install_missing_write_ack(native, scope, registered.get, Path(folder))
    bootstrap = MooncakeBootstrapServer("127.0.0.1", 0)
    bootstrap.start(timeout=10)
    native.vllm_config.kv_transfer_config.kv_connector_extra_config[
        "bootstrap_server_address"
    ] = f"127.0.0.1:{bootstrap.port}"
    ready = threading.Event()
    listener = asyncio.run_coroutine_threadsafe(
        native._mooncake_sender_listener(ready), native.sender_loop
    )
    try:
        assert ready.wait(10), listener.exception(timeout=0)
        pipe.send(f"http://127.0.0.1:{bootstrap.port}")
        attempts, destination = pipe.recv()
        for attempt in attempts:
            assert attempt.manifest_sha256 == manifest_digest(plan.manifest())
            registered[attempt] = NativeRankWindowWriter(
                attempt,
                plan.manifest(),
                native.engine,
                pool,
                destination=destination,
                pack=plan.pack,
                release_source=lambda attempt=attempt: released.append(
                    attempt.request_id
                ),
                timeout=5,
            )
        pipe.send("registered")
        assert pipe.recv() == "close"
        pipe.send((released, torch.cuda.is_initialized(), pool.leased_slots))
    finally:
        pool.close()
        native.shutdown()
        native.sender_loop.close()
        bootstrap.shutdown()
        pipe.close()


def test_one_missing_native_write_ack_holds_pages_until_authenticated_abort(tmp_path):
    from kitepond.core.readiness import ReadyAttempt
    from kitepond.integrations.mooncake.live_receive import (
        RankWindowAttempt,
        mooncake_receive_pool,
    )
    from kitepond.integrations.mooncake.named_control import (
        GlmWindowControlScope,
        connect_glm_window_peer,
    )
    from kitepond.integrations.mooncake.window import manifest_digest, plan_windows
    from kitepond.integrations.vllm.decode_receive import (
        DecodeReceiveMailbox,
        DecodeReceiveTicket,
    )
    from kitepond.integrations.vllm.glm_live_receive import receive_glm_rank_window
    from lost_write_ack_injector import (
        ABORT,
        DROP,
        FENCED,
        HOLD,
        SETTLE,
        AckTarget,
        arm_fault,
        install_target_observation,
    )
    from phase1_lost_write_ack_probe import validate_evidence

    pytest.importorskip("mooncake.engine", exc_type=ImportError)
    pytest.importorskip(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.named_control"
    )
    scope = GlmWindowControlScope(
        "P-boot",
        "D-boot",
        _RANK,
        secrets.token_bytes(32),
        forward_abort=True,
    )
    native = _worker("kv_consumer", _RANK)
    pool = mooncake_receive_pool(native.engine, 2)
    target = _live_model_plan(120)
    source = _live_model_plan(40)
    cache = target.final
    groups = {name: index for index, name in enumerate(cache.layers)}
    initial = DecodeReceiveTicket(
        cache.layout.request_id,
        "lost-ack-1",
        "D-owner",
        scope.decode_boot_id,
        cache.layout.epoch,
        "a" * 64,
        cache.layout.target_valid_tokens,
        tuple(layer.block_ids for layer in cache.layers.values()),
        target.local_tokens,
        None,
    )
    # The model-shaped recipe is independent of API request identity. Keep its
    # native ColdRestoreLayout consistent for a truly distinct second request.
    from kitepond.models.glm5next.cold_restore import ColdRestoreLayout
    from kitepond.models.glm5next.live_windows import GlmRankWindowPlan

    fresh = replace(
        initial,
        request_id="followup-request",
        transfer_id="fresh-2",
        grant_sha256="b" * 64,
    )
    fresh_cache = ColdRestoreLayout(
        replace(cache.layout, request_id=fresh.request_id),
        cache.rank,
        list(cache.layers.values()),
    )
    fresh_plan = GlmRankWindowPlan(fresh_cache, target.local_tokens)

    def make_attempt(ticket, plan):
        return RankWindowAttempt(
            ticket.request_id,
            ticket.transfer_id,
            ticket.epoch,
            _RANK,
            scope.source_boot_id,
            ticket.grant_sha256,
            ticket.destination_sha256,
            manifest_digest(plan.manifest()),
        )

    lost, survivor = make_attempt(initial, target), make_attempt(fresh, fresh_plan)
    folder = tmp_path / "only-one-lost-write"
    folder.mkdir(mode=0o700)
    arm_fault(
        folder,
        AckTarget(
            ReadyAttempt(
                lost.request_id, lost.transfer_id, lost.epoch, lost.grant_sha256
            ),
            _RANK,
            scope.source_boot_id,
            scope.decode_boot_id,
        ),
    )
    mailbox = DecodeReceiveMailbox(_RANK, scope.decode_boot_id)
    ctx = multiprocessing.get_context("spawn")
    pipe, child_pipe = ctx.Pipe()
    child = ctx.Process(target=_producer, args=(child_pipe, scope, str(folder)))
    child.start()
    child_pipe.close()

    def read():
        assert pipe.poll(35), "Native P control process did not respond"
        return pipe.recv()

    async def transfer(bootstrap, ticket, plan, timeout):
        async with connect_glm_window_peer(
            native, "kv_producer", bootstrap, scope
        ) as peer:
            await receive_glm_rank_window(
                mailbox,
                pool,
                ticket,
                plan,
                groups,
                peer,
                source_boot_id=scope.source_boot_id,
                forward_abort_peer=peer,
                timeout=timeout,
            )

    async def exercise(bootstrap):
        async def native_receive(ticket, plan, layer_groups, fence):
            await transfer(bootstrap, ticket, plan, 4.0)

        runtime = SimpleNamespace(
            scope=scope,
            _receive=native_receive,
            _receiver=SimpleNamespace(_pool=pool),
            _hisparse=SimpleNamespace(decode_receive_mailbox=lambda: mailbox),
        )
        install_target_observation(runtime, folder)
        mailbox.bind((initial,))
        before = [layer.tensor.clone() for layer in target.final.layers.values()]
        task = asyncio.create_task(runtime._receive(initial, target, groups, None))
        async with asyncio.timeout(8):
            while not (folder / DROP).exists():
                await asyncio.sleep(0.01)
        dropped = json.loads((folder / DROP).read_bytes())
        assert dropped["attempt"]["rank"] == _RANK and dropped["native_write_completed"]
        assert dropped["source_released"] is False and not task.done()
        assert pool.leased_slots == 1 and pool.quarantined_slots == 0
        assert mailbox.settled_proof(initial) is None
        assert not (folder / ABORT).exists() and not (folder / SETTLE).exists()
        # The independently packed bytes are already in the actual D scratch.
        # No synthetic ACK or local model scatter caused those bytes.
        window = plan_windows(source.manifest())[0]
        expected = torch.empty(window.nbytes, dtype=torch.uint8)
        assert source.pack(window, expected) == dropped["nbytes"] == window.nbytes
        slot = pool._slots[dropped["slot"]]
        assert slot.generation == dropped["generation"]
        torch.testing.assert_close(slot.tensor[: window.nbytes], expected)
        with pytest.raises(RuntimeError, match="live or uncertain"):
            pool.close()
        with pytest.raises(ValueError, match="unsettled local writes"):
            mailbox.bind(())

        with pytest.raises(TimeoutError):
            await task
        aborted = mailbox.drain(None).receipts
        assert len(aborted) == 1 and aborted[0].outcome == "abort_settled"
        assert (aborted[0].rank, aborted[0].grant_sha256) == (
            _RANK,
            initial.grant_sha256,
        )
        for old, current in zip(before, target.final.layers.values(), strict=True):
            torch.testing.assert_close(old, current.tensor)
        assert pool.leased_slots == pool.quarantined_slots == 0
        abort, settle = (
            json.loads((folder / file).read_bytes()) for file in (ABORT, SETTLE)
        )
        assert dropped["attempt"] == abort["attempt"] == settle["attempt"]
        assert dropped["monotonic_ns"] < abort["monotonic_ns"] < settle["monotonic_ns"]
        assert abort["closed"] and settle["outcome"] == "abort_settled"

        assert (folder / HOLD).is_file() and (folder / FENCED).is_file()
        identity = ReadyAttempt(
            initial.request_id, initial.transfer_id, initial.epoch, initial.grant_sha256
        )
        claim = AckTarget(identity, _RANK, scope.source_boot_id, scope.decode_boot_id)
        observed = validate_evidence(folder, claim)
        mailbox.bind((fresh,))
        await runtime._receive(fresh, fresh_plan, groups, None)
        assert observed == validate_evidence(folder, claim)
        loaded = mailbox.drain(None).receipts
        assert len(loaded) == 1 and loaded[0].outcome == "loaded"
        assert loaded[0].grant_sha256 == fresh.grant_sha256
        assert json.loads((folder / DROP).read_bytes()) == dropped
        assert pool.leased_slots == pool.quarantined_slots == 0
        mailbox.bind(())
        pool.close()

    try:
        bootstrap = read()
        pipe.send(((lost, survivor), ("127.0.0.1", native.rpc_port, pool.pool_id)))
        assert read() == "registered"
        asyncio.run_coroutine_threadsafe(
            exercise(bootstrap), native.receiver_loop
        ).result(22)
        pipe.send("close")
        released, used_gpu, held = read()
        assert released == [initial.request_id, fresh.request_id] and held == 0
        assert not used_gpu and not torch.cuda.is_initialized()
    finally:
        native.shutdown()
        native.receiver_loop.close()
        pipe.close()
        child.join(4)
        if child.is_alive():
            child.terminate()
            child.join(4)
    assert child.exitcode == 0
