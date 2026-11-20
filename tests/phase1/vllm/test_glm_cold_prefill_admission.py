# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from json import dumps, loads
from multiprocessing import get_context
from time import monotonic, monotonic_ns, sleep

import pytest
from kitepond.core.contracts import Wait
from kitepond.core.readiness import ReadyReceipt
from kitepond.integrations.vllm.glm_cold_prefill_admission import (
    ColdPrefillAdmission,
    ColdPrefillControl,
    ColdPrefillPrompt,
)
from kitepond.integrations.vllm.host_authority import (
    HostRankSettlement,
    NativeHostAuthority,
)
from kitepond.integrations.vllm.ready_transport import (
    ReadyCollector,
    ReadyPublisher,
    ReadyWorker,
)

from vllm.v1.core.block_pool import BlockPool


def _control(**overrides):
    settings = dict(
        d_engine_id="decoder", d_boot_id="D-1", p_boot_id="P-1", key=bytes(range(32))
    )
    return ColdPrefillControl(**(settings | overrides))


def _authority(free_pages, **overrides):
    pool = BlockPool(
        num_gpu_blocks=free_pages + 1, enable_caching=False, hash_block_size=1024
    )
    native = NativeHostAuthority(
        pool, engine_id="decoder", boot_id="D-1", block_tokens=1024
    )
    control = _control()
    limits = dict(context_limit=8192, max_pending=4, arm_timeout_seconds=5)
    admission = ColdPrefillAdmission(native, control, **(limits | overrides))
    return native, control, admission


def _prompt(request="req-1", tokens=1025):
    return ColdPrefillPrompt.from_tokens(request, "kv-" + request, 19, range(tokens))


def test_native_capacity_and_signed_exact_boot_prompt_and_page_table():
    native, signer, admission = _authority(2)
    prompt = _prompt()
    offer = admission.offer(prompt)
    assert admission.offer(prompt) is offer
    assert offer.imported_tokens == 1024
    assert len(offer.grant.writable_pages) == 2
    assert native.observe().available_pages == 0
    assert admission.offer(_prompt("other", 2)) == Wait("RAM_ADMISSION_WAIT")

    wire = loads(dumps(signer.export_offer(offer)))
    assert signer.open_offer(wire, prompt) == offer
    assert signer.open_offer(wire, prompt).grant is not offer.grant
    with pytest.raises(ValueError, match="different prompt"):
        signer.open_offer(wire, replace(prompt, prompt_sha256="9" * 64))
    with pytest.raises(ValueError, match="different boot"):
        _control(p_boot_id="old-P").open_offer(wire, prompt)
    with pytest.raises(ValueError, match="different prompt"):
        signer.open_offer(wire, prompt, clock=lambda: offer.arm_deadline_ns)
    wire["body"]["grant"]["writable_pages"][0] += 1
    with pytest.raises(ValueError, match="not authenticated"):
        signer.open_offer(wire, prompt)


def _publish(connection, directory, rank, region):
    ticket, tokens = connection.recv()
    publisher = ReadyPublisher(directory)
    try:
        receipt = ReadyReceipt(
            ticket.attempt, rank, region, tokens, tokens, monotonic_ns()
        )
        connection.send(publisher.after_save_fence(ticket, lambda: receipt))
        assert connection.recv() == "close"
    finally:
        publisher.close()
        connection.close()


def test_real_rank_workers_prove_n_minus_one_and_only_d_can_adopt(tmp_path):
    native, signer, admission = _authority(3)
    prompt = _prompt()
    offer = admission.offer(prompt)
    remote_offer = signer.open_offer(signer.export_offer(offer), prompt)
    directory = tmp_path / "p"
    directory.mkdir(mode=0o700)
    context = get_context("fork")
    regions = {rank: f"{rank:064x}" for rank in range(4)}
    parents = []
    workers = []
    for rank, region in regions.items():
        parent, child = context.Pipe()
        proc = context.Process(target=_publish, args=(child, directory, rank, region))
        proc.start()
        child.close()
        parents.append(parent)
        workers.append(proc)
    collector = ReadyCollector(
        directory,
        {rank: ReadyWorker.from_pid(proc.pid) for rank, proc in enumerate(workers)},
    )
    try:
        ticket = signer.arm(remote_offer, collector, regions)
        for conn in parents[:3]:
            conn.send((ticket, remote_offer.imported_tokens))
            assert conn.poll(5) and conn.recv()
        assert collector.take(ticket) is None
        parents[3].send((ticket, remote_offer.imported_tokens))
        assert parents[3].poll(5) and parents[3].recv()
        deadline = monotonic() + 2
        receipts = collector.take(ticket)
        while receipts is None and monotonic() < deadline:
            sleep(0.01)
            receipts = collector.take(ticket)
        assert receipts is not None
        mac = signer.ready_mac(remote_offer, receipts)
        assert collector.take(ticket) is None
        with pytest.raises(ValueError, match="not authenticated"):
            admission.verify_ready(offer, receipts, "1" * 64)
        with pytest.raises(ValueError, match="no complete ready quorum"):
            native.adopt(offer.grant)
        with pytest.raises(ValueError, match="no longer owned"):
            admission.verify_ready(remote_offer, receipts, mac)
        owned = admission.verify_ready(offer, receipts, mac)
        assert owned is offer.grant
        attached = native.adopt(owned)
        admission.detach_adopted(offer)
        native.pool.free_blocks(attached)
        assert native.observe().available_pages == 3
    finally:
        collector.close()
        for conn, proc in zip(parents, workers, strict=True):
            if proc.is_alive():
                conn.send("close")
            conn.close()
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=3)


@pytest.mark.parametrize("already_ready", [False, True])
def test_cancel_never_recycles_before_four_abort_fences(already_ready):
    native, signer, admission = _authority(2)
    offer = admission.offer(_prompt())
    receipts = tuple(
        ReadyReceipt(
            offer.attempt,
            rank,
            "a" * 64,
            offer.imported_tokens,
            offer.imported_tokens,
            monotonic_ns(),
        )
        for rank in range(4)
    )
    mac = signer.ready_mac(offer, receipts)
    if already_ready:
        admission.verify_ready(offer, receipts, mac)
    settled = tuple(
        HostRankSettlement(offer.grant.sha256, rank, "abort_settled", 0, 0)
        for rank in range(4)
    )
    assert not admission.settle_abort(offer, settled[:3])
    assert native.observe().available_pages == 0
    with pytest.raises(ValueError, match="not authenticated"):
        admission.verify_ready(offer, receipts, mac)
    assert admission.settle_abort(offer, settled[3:])
    assert native.observe().available_pages == 2
    with pytest.raises(ValueError, match="no longer owned"):
        admission.verify_ready(offer, receipts, mac)
