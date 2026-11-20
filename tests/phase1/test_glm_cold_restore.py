# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact CPU byte recipes and native tensor readback for GLM cold restore."""

from dataclasses import replace

import pytest
import torch
from kitepond.backends.fsl2 import PartSpec
from kitepond.models.glm5next.cold_restore import (
    ColdRestoreLayout,
    RestoreLayer,
    verify_cold_import_readback,
)
from kitepond.models.glm5next.import_recipe import (
    ByteSpan,
    ColdBinding,
    compose_import_recipe,
    copy_import_window,
    part_identity,
)
from kitepond.models.glm5next.layout import CacheIdentity, CacheRole, Glm5NextPDLayout
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion


def _cache(width=656, rank=2, layer_counts=None):
    layout = Glm5NextPDLayout(
        "new-request", "b" * 64, "c" * 64, 0, 4, 1059, 1059, 2, "decode_ready"
    )
    native = {
        CacheRole.MLA: ((4, 1, 256, width), torch.uint8, "draft", (3, 1)),
        CacheRole.INDEXER: ((4, 1, 768, 132), torch.uint8, "target", (1,)),
        CacheRole.KDA_CONV: ((4, 6, 6144), torch.bfloat16, "target", (3, 0, 2)),
        CacheRole.KDA_RECURRENT: (
            (4, 16, 128, 128),
            torch.float32,
            "target",
            (3, 0, 2),
        ),
        CacheRole.KPOOL_TAIL: ((4, 2, 8, 128), torch.bfloat16, "draft", (3,)),
    }
    layers = [
        RestoreLayer(
            CacheIdentity(
                f"decoder.{layer}.kda"
                if role in (CacheRole.KDA_CONV, CacheRole.KDA_RECURRENT)
                else f"decoder.{layer}.{role.value}",
                role,
                model_role if layer == 0 else "target",
            ),
            torch.zeros(shape, dtype=dtype),
            blocks,
        )
        for role, (shape, dtype, model_role, blocks) in native.items()
        for layer in range(1 if layer_counts is None else layer_counts[role])
    ]
    return ColdRestoreLayout(layout, rank, layers)


def _restored_parts(cache, region):
    parts = cache.manifest()
    payloads = {}
    physical = 4096
    bindings = []
    for ordinal, part in enumerate(parts, 1):
        data = torch.arange(part.nbytes, dtype=torch.uint8) ^ ((ordinal * 11) % 256)
        payloads[part] = data
        role = part_identity(part.role).cache_role
        chunks = [(part, data)]
        if role in (CacheRole.MLA, CacheRole.INDEXER):
            width = part.shape[1]
            chunks = [
                (
                    PartSpec(
                        part.role,
                        part.rank,
                        start,
                        (min(7, part.shape[0] - start), width),
                        part.dtype,
                        min(7, part.shape[0] - start) * width,
                    ),
                    data[start * width : min(start + 7, part.shape[0]) * width],
                )
                for start in range(0, part.shape[0], 7)
            ]
        # Each source part crosses two native spans with reversed offsets and
        # unequal lengths. A byte copy must retain logical span order.
        for source_part, bytes_in_part in reversed(chunks):
            nbytes = source_part.nbytes
            head = nbytes // 2 + 1
            first = physical + nbytes - head + 5
            region.mapping[first : first + head] = (
                bytes_in_part[:head].numpy().tobytes()
            )
            region.mapping[physical : physical + nbytes - head] = (
                bytes_in_part[head:].numpy().tobytes()
            )
            bindings.append(
                ColdBinding(
                    source_part,
                    (
                        ByteSpan(part.rank, 2, first, head),
                        ByteSpan(part.rank, 1, physical, nbytes - head),
                    ),
                )
            )
            physical += nbytes + 5
    return parts, payloads, tuple(reversed(bindings))


def _owned_region(cache, extent=4 * 2**20):
    return NativeHiSparseMemfdRegion.create(
        extent,
        authority_engine_id="restore-d",
        authority_boot_id="resume-boot",
        source_rank=cache.rank,
        layout_sha256="b" * 64,
    )


@pytest.mark.parametrize("width", [528, 656])
def test_cold_restore_reorders_native_spans_and_preserves_all_rows(width):
    cache = _cache(width)
    region = _owned_region(cache)
    try:
        parts, payloads, bindings = _restored_parts(cache, region)
        recipe = compose_import_recipe(parts, bindings)
        slab = torch.empty(2**21, dtype=torch.uint8)
        fingerprint = region.descriptor.fingerprint()
        for window in recipe:
            copied = copy_import_window(region, window, slab, region_sha256=fingerprint)
            packet = tuple(binding.part for binding in window)
            offset = 0
            for part in packet:
                assert torch.equal(
                    copied[offset : offset + part.nbytes], payloads[part]
                )
                offset += part.nbytes
            cache.scatter_window(packet, copied)

        result = verify_cold_import_readback(
            cache, recipe, region, slab, region_sha256=fingerprint
        )
        assert result == {
            "rank": 2,
            "status": "MATCH",
            "counts": {role.value: 1 for role in CacheRole},
            "mismatch": None,
        }
        by_role = {part_identity(part.role).cache_role: part for part in parts}
        tail = by_role[CacheRole.KPOOL_TAIL]
        assert torch.equal(
            cache.layers[tail.role].tensor[3].view(torch.uint8),
            payloads[tail]
            .view(torch.bfloat16)
            .reshape(tail.shape)
            .transpose(0, 1)
            .contiguous()
            .view(torch.uint8),
        )
        recurrent = by_role[CacheRole.KDA_RECURRENT]
        recurrent_cache = cache.layers[recurrent.role].tensor
        assert torch.equal(
            recurrent_cache[2].view(torch.uint8).flatten(), payloads[recurrent]
        )
        assert torch.count_nonzero(recurrent_cache[3]) == 0
        mla = by_role[CacheRole.MLA]
        mla_cache = cache.layers[mla.role].tensor
        assert torch.equal(mla_cache[1, 0, 0], payloads[mla].view(mla.shape)[256])
        assert torch.count_nonzero(mla_cache[2]) == 0
        assert not torch.cuda.is_initialized()
    finally:
        region.close()


@pytest.mark.parametrize("role", tuple(CacheRole))
def test_readback_rejects_actual_destination_corruption(role):
    cache = _cache()
    region = _owned_region(cache)
    try:
        parts, _, bindings = _restored_parts(cache, region)
        recipe = compose_import_recipe(parts, bindings)
        slab = torch.empty(2**21, dtype=torch.uint8)
        fingerprint = region.descriptor.fingerprint()
        for window in recipe:
            copied = copy_import_window(region, window, slab, region_sha256=fingerprint)
            cache.scatter_window(tuple(binding.part for binding in window), copied)
        corrupt = next(
            part for part in parts if part_identity(part.role).cache_role is role
        )
        data = cache.read_chunk(corrupt).clone()
        data[-1] ^= 1
        cache.scatter_window((corrupt,), data)
        bad = verify_cold_import_readback(
            cache, recipe, region, slab, region_sha256=fingerprint
        )
        assert bad["status"] == "MISMATCH"
        assert bad["mismatch"]["role"] == role.value
        assert bad["mismatch"]["expected_sha256"] != bad["mismatch"]["actual_sha256"]
        with pytest.raises(ValueError, match="complete rank/layer/role"):
            verify_cold_import_readback(
                cache, ((recipe[0][0],),), region, slab, region_sha256=fingerprint
            )
    finally:
        region.close()


def test_restore_reads_back_all_104_native_glm_role_bindings():
    counts = {
        CacheRole.MLA: 12,
        CacheRole.INDEXER: 12,
        CacheRole.KDA_CONV: 34,
        CacheRole.KDA_RECURRENT: 34,
        CacheRole.KPOOL_TAIL: 12,
    }
    cache = _cache(layer_counts=counts)
    region = _owned_region(cache, 64 * 2**20)
    try:
        parts, _, bindings = _restored_parts(cache, region)
        assert len(parts) == 104
        recipe = compose_import_recipe(parts, bindings)
        assert len(recipe) > 1
        slab = torch.empty(32 * 2**20, dtype=torch.uint8)
        authority = region.descriptor.fingerprint()
        for window in recipe:
            payload = copy_import_window(region, window, slab, region_sha256=authority)
            cache.scatter_window(tuple(binding.part for binding in window), payload)
        proof = verify_cold_import_readback(
            cache, recipe, region, slab, region_sha256=authority
        )
        assert proof["status"] == "MATCH"
        assert proof["counts"] == {role.value: count for role, count in counts.items()}
    finally:
        region.close()


def test_restore_rejects_incomplete_foreign_or_aliased_native_storage():
    cache = _cache()
    region = _owned_region(cache)
    try:
        parts, _, bindings = _restored_parts(cache, region)
        with pytest.raises(ValueError, match="format or row coverage"):
            compose_import_recipe(_cache(528).manifest(), bindings)
        with pytest.raises(ValueError, match="complete rank/layer/role"):
            compose_import_recipe(_cache(rank=3).manifest(), bindings)
        mla_index = next(
            index
            for index, binding in enumerate(bindings)
            if part_identity(binding.part.role).cache_role is CacheRole.MLA
        )
        split = bindings[mla_index]
        with pytest.raises(ValueError, match="overlap, hole"):
            compose_import_recipe(
                parts, (*bindings[:mla_index], *bindings[mla_index + 1 :])
            )
        with pytest.raises(ValueError, match="alias physical bytes"):
            same = (
                split.spans[0],
                replace(split.spans[1], offset=split.spans[0].offset),
            )
            compose_import_recipe(
                parts,
                (
                    *bindings[:mla_index],
                    ColdBinding(split.part, same),
                    *bindings[mla_index + 1 :],
                ),
            )
        with pytest.raises(ValueError, match="different authority region"):
            copy_import_window(
                region,
                (split,),
                torch.empty(2**21, dtype=torch.uint8),
                region_sha256="d" * 64,
            )
        duplicate = parts[0]
        with pytest.raises(ValueError, match="same native rows twice"):
            cache.scatter_window(
                (duplicate, duplicate),
                torch.empty(2 * duplicate.nbytes, dtype=torch.uint8),
            )
        assert not torch.count_nonzero(cache.layers[duplicate.role].tensor)
        slab = torch.zeros(split.part.nbytes, dtype=torch.uint8)
        offset = replace(split.spans[-1], offset=region.descriptor.nbytes)
        with pytest.raises(ValueError, match="authority rank/extent"):
            copy_import_window(
                region,
                (ColdBinding(split.part, (*split.spans[:-1], offset)),),
                slab,
                region_sha256=region.descriptor.fingerprint(),
            )
        assert torch.count_nonzero(slab) == 0
    finally:
        region.close()


@pytest.mark.parametrize("fail_after_scatter", [False, True])
def test_actual_native_d_restore_releases_only_checked_destination(
    fail_after_scatter, monkeypatch
):
    from kitepond.integrations.vllm.decode_receive import (
        DecodeReceiveMailbox,
        DecodeReceiveTicket,
    )
    from kitepond.integrations.vllm.glm_decode_restore import (
        apply_verified_native_host_import,
    )

    cache = _cache()
    region = _owned_region(cache)
    groups = {name: index for index, name in enumerate(cache.layers)}
    block_ids = tuple(layer.block_ids for layer in cache.layers.values())
    ticket = DecodeReceiveTicket(
        cache.layout.request_id,
        "checked-transfer",
        "restore-d",
        "resume-boot",
        cache.layout.epoch,
        "a" * 64,
        cache.layout.target_valid_tokens,
        block_ids,
        0,
        None,
    )
    mailbox = DecodeReceiveMailbox(cache.rank, ticket.boot_id)
    mailbox.bind((ticket,))
    assert mailbox.ticket_for(ticket.request_id, ticket.transfer_id, ticket.epoch) == (
        ticket
    )
    try:
        parts, payloads, bindings = _restored_parts(cache, region)
        slab = torch.empty(2**21, dtype=torch.uint8)
        expected = region.descriptor.fingerprint()
        changed = replace(ticket, imported_tokens=ticket.imported_tokens - 1)
        with pytest.raises(ValueError, match="admitted rank/layer views"):
            apply_verified_native_host_import(
                mailbox,
                changed,
                cache,
                groups,
                region,
                bindings,
                slab,
                source_region_sha256=expected,
            )
        assert not mailbox.drain(None).receipts

        def apply():
            apply_verified_native_host_import(
                mailbox,
                ticket,
                cache,
                groups,
                region,
                bindings,
                slab,
                source_region_sha256=expected,
            )

        if fail_after_scatter:
            scatter = cache.scatter_window

            def failing_scatter(parts, payload):
                scatter(parts, payload)
                raise RuntimeError("Injected failure after native scatter")

            monkeypatch.setattr(cache, "scatter_window", failing_scatter)
            with pytest.raises(RuntimeError, match="failure after native scatter"):
                apply()
        else:
            apply()
        for part in parts:
            assert torch.equal(cache.read_chunk(part), payloads[part])
        (receipt,) = mailbox.drain(None).receipts
        assert (
            receipt.rank,
            receipt.grant_sha256,
            receipt.destination_sha256,
            receipt.outcome,
        ) == (
            cache.rank,
            ticket.grant_sha256,
            ticket.destination_sha256,
            "abort_settled" if fail_after_scatter else "loaded",
        )
        with pytest.raises(ValueError, match="no available exact scheduler ticket"):
            apply_verified_native_host_import(
                mailbox,
                ticket,
                cache,
                groups,
                region,
                bindings,
                slab,
                source_region_sha256=expected,
            )
        assert not mailbox.drain(None).receipts
    finally:
        region.close()


def _live_model_plan(fill):
    from kitepond.models.glm5next.live_windows import GlmRankWindowPlan

    cache = _cache()
    for number, layer in enumerate(cache.layers.values()):
        layer.tensor.view(torch.uint8).fill_(fill + number)
    return GlmRankWindowPlan(cache, 1024)


def _native_live_source(pipe):
    import asyncio

    from kitepond.integrations.mooncake.live_receive import (
        NativeRankWindowWriter,
        mooncake_receive_pool,
    )
    from mooncake.engine import TransferEngine

    async def serve():
        engine = TransferEngine()
        assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "") == 0
        source = _live_model_plan(40)
        pool = mooncake_receive_pool(engine, 1)
        released = []
        pipe.send("ready")
        attempt, destination = pipe.recv()
        writer = NativeRankWindowWriter(
            attempt,
            source.manifest(),
            engine,
            pool,
            destination=destination,
            pack=source.pack,
            release_source=lambda: released.append(attempt.rank),
            timeout=5,
        )
        while True:
            operation, arguments = pipe.recv()
            if operation == "close":
                pool.close()
                pipe.send((released, torch.cuda.is_initialized()))
                return
            if operation == "fetch":
                pipe.send(await writer.fetch(*arguments))
            if operation == "commit":
                pipe.send(await writer.commit(*arguments))
            if operation == "abandon":
                await writer.abandon()
                pipe.send("abandoned")

    try:
        asyncio.run(serve())
    finally:
        pipe.close()


@pytest.mark.parametrize("mode", ["loaded", "ack_lost", "cancel_scatter"])
def test_native_glm_rank_rendezvous_loopback_without_gpu(mode, monkeypatch):
    import asyncio
    import multiprocessing
    import threading

    from kitepond.integrations.mooncake.live_receive import (
        RankWindowAttempt,
        mooncake_receive_pool,
    )
    from kitepond.integrations.mooncake.window import manifest_digest
    from kitepond.integrations.vllm.decode_receive import (
        DecodeReceiveMailbox,
        DecodeReceiveTicket,
    )
    from kitepond.integrations.vllm.glm_live_receive import receive_glm_rank_window

    TransferEngine = pytest.importorskip(
        "mooncake.engine", exc_type=ImportError
    ).TransferEngine
    assert not torch.cuda.is_initialized()
    engine = TransferEngine()
    assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "") == 0
    pool = mooncake_receive_pool(engine, 2)
    destination = ("127.0.0.1", engine.get_rpc_port(), pool.pool_id)
    target = _live_model_plan(120)
    cache = target.final
    groups = {name: idx for idx, name in enumerate(cache.layers)}
    ticket = DecodeReceiveTicket(
        cache.layout.request_id,
        "live-rank-transfer",
        "D-owner",
        "d-current-boot",
        cache.layout.epoch,
        "a" * 64,
        cache.layout.target_valid_tokens,
        tuple(layer.block_ids for layer in cache.layers.values()),
        1024,
        None,
    )
    mailbox = DecodeReceiveMailbox(cache.rank, ticket.boot_id)
    mailbox.bind((ticket,))

    ctx = multiprocessing.get_context("spawn")
    local_pipe, child_pipe = ctx.Pipe()
    child = ctx.Process(target=_native_live_source, args=(child_pipe,))
    child.start()
    child_pipe.close()
    seeded = []

    def receive():
        assert local_pipe.poll(35), "Native Mooncake CPU source timed out"
        return local_pipe.recv()

    class LoopbackPeer:
        async def fetch(self, *args):
            local_pipe.send(("fetch", args))
            reply = await asyncio.to_thread(receive)
            if mode == "ack_lost":
                raise TimeoutError("Injected lost ACK after native write")
            return reply

        async def commit(self, *args):
            local_pipe.send(("commit", args))
            return await asyncio.to_thread(receive)

    def seed(actual, key) -> None:
        assert actual is cache and key == ticket
        assert not mailbox.drain(None).receipts
        seeded.append(key.request_id)

    async def run():
        await receive_glm_rank_window(
            mailbox,
            pool,
            ticket,
            target,
            groups,
            LoopbackPeer(),
            source_boot_id="p-current-boot",
            timeout=10,
            seed_imported_tail=seed,
        )

    async def cancel_during_native_scatter():
        enter = asyncio.Event()
        exit_scatter = threading.Event()
        scatter = target.scatter
        loop = asyncio.get_running_loop()

        def controls(window, payload):
            # Do real model-shaped writes while holding the registered window.
            count = scatter(window, payload)
            loop.call_soon_threadsafe(enter.set)
            assert exit_scatter.wait(6)
            return count

        monkeypatch.setattr(target, "scatter", controls)
        task = asyncio.create_task(run())
        try:
            await asyncio.wait_for(enter.wait(), 8)
            task.cancel()
            await asyncio.sleep(0)
            assert pool.leased_slots == 1
            assert not mailbox.drain(None).receipts
        finally:
            exit_scatter.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        assert receive() == "ready"
        source_boot = "p-current-boot"
        local_pipe.send(
            (
                RankWindowAttempt(
                    ticket.request_id,
                    ticket.transfer_id,
                    ticket.epoch,
                    cache.rank,
                    source_boot,
                    ticket.grant_sha256,
                    ticket.destination_sha256,
                    manifest_digest(target.manifest()),
                ),
                destination,
            )
        )
        if mode == "ack_lost":
            with pytest.raises(TimeoutError, match="lost ACK"):
                asyncio.run(run())
            assert pool.quarantined_slots == 1
            with pytest.raises(RuntimeError, match="uncertain writes"):
                pool.close()
            assert not seeded and not mailbox.drain(None).receipts
            with pytest.raises(ValueError, match="unsettled local writes"):
                mailbox.bind(())
        elif mode == "cancel_scatter":
            asyncio.run(cancel_during_native_scatter())
            assert not seeded and pool.leased_slots == pool.quarantined_slots == 0
            (proof,) = mailbox.drain(None).receipts
            assert (proof.rank, proof.outcome) == (cache.rank, "abort_settled")
            pool.close()
            local_pipe.send(("abandon", ()))
            assert receive() == "abandoned"
        else:
            asyncio.run(run())
            pool.close()
            assert seeded == [ticket.request_id]
            (proof,) = mailbox.drain(None).receipts
            assert (proof.rank, proof.outcome) == (cache.rank, "loaded")
            from kitepond.models.glm5next.cold_restore import native_rank_rows

            for part in cache.manifest():
                index = groups[part.role]
                actual = cache.read_chunk(part)
                role = part_identity(part.role).cache_role
                if role in (CacheRole.MLA, CacheRole.INDEXER):
                    kept = native_rank_rows(1024, role, cache.rank) * part.shape[1]
                    assert torch.all(actual[:kept] == 120 + index)
                    actual = actual[kept:]
                assert torch.all(actual == 40 + index)
        local_pipe.send(("close", ()))
        released, cuda_initialized = receive()
        assert not cuda_initialized and not torch.cuda.is_initialized()
        assert released == ([] if mode == "ack_lost" else [cache.rank])
    finally:
        child.join(3)
        if child.is_alive():
            child.terminate()
            child.join(3)
        local_pipe.close()
    assert child.exitcode == 0


def test_live_glm_window_checkpoint_keeps_detached_native_state():
    from kitepond.integrations.mooncake.window import plan_windows
    from kitepond.models.glm5next.live_windows import GlmRankWindowPlan

    base = _cache()
    final = replace(base.layout, target_valid_tokens=2067, draft_valid_tokens=2067)
    live_layers = [
        replace(layer, block_ids=(3, 1, 2))
        if layer.identity.cache_role is CacheRole.MLA
        else layer
        for layer in base.layers.values()
    ]
    kda = tuple(
        layer
        for layer in live_layers
        if layer.identity.cache_role in (CacheRole.KDA_CONV, CacheRole.KDA_RECURRENT)
    )

    def windows_with_data(initial):
        mapped = [
            replace(layer, tensor=torch.zeros_like(layer.tensor))
            for layer in live_layers
        ]
        live = ColdRestoreLayout(final, base.rank, mapped)
        native_snapshot = [
            replace(layer, block_ids=(1, 1))
            for layer in mapped
            if layer.identity in {entry.identity for entry in kda}
        ]
        snapshot = ColdRestoreLayout(
            replace(
                final,
                target_valid_tokens=1024,
                draft_valid_tokens=1024,
                committed_kda_slot=1,
            ),
            base.rank,
            native_snapshot,
        )
        for end in (live, snapshot):
            for name, layer in end.layers.items():
                if end is live:
                    layer.tensor.view(torch.uint8).fill_(initial)
                else:
                    layer.tensor[1].view(torch.uint8).fill_(initial + 1)
        return GlmRankWindowPlan(live, 0, checkpoint=snapshot)

    source, target = windows_with_data(20), windows_with_data(70)
    manifest = source.manifest()
    assert manifest == target.manifest()
    assert sum(entry["purpose"] == "checkpoint" for entry in manifest) == 2
    for window in plan_windows(manifest):
        slab = torch.empty(window.nbytes, dtype=torch.uint8)
        source.pack(window, slab)
        target.scatter(window, slab)
        target.verify_readback(window, slab)
    for part in target.checkpoint.manifest():
        assert torch.all(target.checkpoint.read_chunk(part) == 21)
    for part in target.final.manifest():
        if part_identity(part.role).cache_role in (
            CacheRole.KDA_CONV,
            CacheRole.KDA_RECURRENT,
        ):
            assert torch.all(target.final.read_chunk(part) == 20)
