# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import os
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import test_stock_terminal_checkpoint as stock_terminal
import torch
from kitepond.integrations.vllm.glm_terminal_manifest import (
    FencedTerminalSource,
    GlmTerminalCheckpointVerifier,
    GlmTerminalManifest,
    SignedTerminalRank,
    TerminalRankSigner,
    capture_terminal_rank,
)
from kitepond.integrations.vllm.glm_terminal_worker import (
    GlmTerminalRunner,
    _SampleFrame,
)
from kitepond.models.glm5next.cold_restore import ColdRestoreLayout, RestoreLayer
from kitepond.models.glm5next.import_recipe import part_identity, part_role
from kitepond.models.glm5next.layout import CacheIdentity, CacheRole, Glm5NextPDLayout
from test_stock_terminal_checkpoint import counts, identity, receipt

from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

capture = stock_terminal.capture

ROLES = (
    CacheIdentity("mla.target", CacheRole.MLA, "target"),
    CacheIdentity("mla.draft", CacheRole.MLA, "draft"),
    CacheIdentity("idx.target", CacheRole.INDEXER, "target"),
    CacheIdentity("idx.draft", CacheRole.INDEXER, "draft"),
    CacheIdentity("tail.target", CacheRole.KPOOL_TAIL, "target"),
    CacheIdentity("tail.draft", CacheRole.KPOOL_TAIL, "draft"),
    CacheIdentity("kda", CacheRole.KDA_CONV, "target"),
    CacheIdentity("kda", CacheRole.KDA_RECURRENT, "target"),
)
BOUNDARY = 66
FORMAT = "e" * 64


class Fence:
    done = False

    def synchronize(self):
        self.done = True


def make_runner(lease, tokens, fence, recurrent_blocks):
    runner = GlmTerminalRunner.__new__(GlmTerminalRunner)
    start, token = BOUNDARY - 2, lease.identity.request_id
    observed = SimpleNamespace(
        spec_sequence_masks_cpu=torch.tensor([True]),
        spec_state_indices_tensor=torch.tensor([recurrent_blocks]),
    )
    runner._frames = {token: _SampleFrame(0, 7, fence, 0, {"kda": observed})}
    frame = torch.zeros((1, 26), dtype=torch.int32)
    frame[0, :2] = torch.tensor([start, start + 1])
    frame[0, 4:6] = torch.tensor(tokens[start:BOUNDARY])
    frame[0, 8:10] = torch.tensor([start, start + 1])
    frame[0, 12:14] = torch.tensor(tokens[start + 1 : BOUNDARY + 1])
    frame[0, 16] = 1
    frame[0, 18] = 2
    frame[0, 20:22] = torch.tensor([0, 2])
    frame[0, 24:26] = 7
    state = torch.tensor([[BOUNDARY, BOUNDARY, 1, 1, 1, 32, start, BOUNDARY, 0, 2, 1]])
    runner.frontier = SimpleNamespace(frame=frame, state=state)
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.tensor([tokens])),
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([BOUNDARY])),
    )
    return runner


def fenced_source(lease, rank, authority, *, bad_tail=False):
    owner = {
        name: group.block_ids for group in lease.groups for name in group.layer_names
    }
    layers = []
    tails = []
    for role in ROLES:
        blocks = owner[role.layer_name]
        scope = role.cache_role
        nblocks = max(blocks) + 1
        if scope is CacheRole.MLA:
            shape, dtype = (1, 256, 656), torch.uint8
        elif scope is CacheRole.INDEXER:
            shape, dtype = (1, 64, 132), torch.uint8
        elif scope is CacheRole.KPOOL_TAIL:
            shape, dtype = (2, 8, 128), torch.bfloat16
            tails.append((part_role(role), () if bad_tail else (64, 65)))
        elif scope is CacheRole.KDA_CONV:
            shape, dtype = (6, 6144), torch.bfloat16
        else:
            shape, dtype = (16, 128, 128), torch.float32
        layer_tensor = torch.zeros((nblocks, *shape), dtype=dtype)
        layers.append(RestoreLayer(role, layer_tensor, blocks))
    identity = lease.identity
    layout = ColdRestoreLayout(
        Glm5NextPDLayout(
            identity.request_id,
            identity.model_fingerprint,
            identity.config_fingerprint,
            producer_owner=0,
            epoch=identity.epoch,
            target_valid_tokens=BOUNDARY,
            draft_valid_tokens=BOUNDARY,
            committed_kda_slot=1,
            boundary_status="decode_ready",
        ),
        rank,
        layers,
    )
    return FencedTerminalSource(layout, authority, FORMAT, tuple(sorted(tails)))


def sign_rank(pipe, lease, tokens, rank, key, authority, *, bad_tail=False):
    try:
        groups = {name: g.block_ids for g in lease.groups for name in g.layer_names}
        actual_selectors = groups["kda"][:4]
        fence = Fence()
        runner = make_runner(lease, tokens, fence, actual_selectors)

        def native_source(active, observed):
            assert fence.done
            assert active is lease
            assert observed.kda_selectors["kda"]["spec_state_indices_tensor"] == [
                list(actual_selectors)
            ]
            return fenced_source(active, rank, authority, bad_tail=bad_tail)

        signed = capture_terminal_rank(
            runner,
            lease,
            32,
            TerminalRankSigner(rank, key),
            native_source,
            clock=lambda: 10 + rank,
        )
        assert lease.identity.request_id not in runner._frames
        pipe.send((os.getpid(), signed))
    except BaseException as exc:
        pipe.send((os.getpid(), exc))
    finally:
        pipe.close()


@pytest.fixture
def active_manifest(capture):
    clock, host_complete, wait_checks = [25], [False], []
    keys = {rank: bytes([rank + 1]) * 32 for rank in range(4)}
    sources = {rank: str(rank + 1) * 64 for rank in range(4)}

    def native_host_prefix(lease):
        if not host_complete[0]:
            raise ValueError("Native terminal host prefix still has pending writes")
        wait_checks.append(lease.sha256)

    verifier = GlmTerminalCheckpointVerifier(
        keys,
        sources,
        native_host_prefix,
        ROLES,
        representation_sha256=FORMAT,
        clock=lambda: clock[0],
    )
    scheduler, owner, scheduler_clock, received = capture(
        host_layers=("mla.target", "mla.draft"),
        device_layers=("idx.target", "idx.draft", "tail.target", "tail.draft", "kda"),
        verify_native_checkpoint=verifier.verify,
    )
    scheduler_clock[0] = clock[0]
    tokens = [index % 63 + 1 for index in range(BOUNDARY)] + [33]
    request = Request(
        request_id="signed-terminal",
        prompt_token_ids=tokens[:-1],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    scheduler.add_request(request)
    owner.arm(request, identity(request.request_id))
    step = scheduler.schedule()
    output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[tokens[-1]]],
    )
    scheduler.update_from_output(step, output)
    lease = owner.lease_for(request.request_id)
    assert lease is not None and lease.requested_common_tokens == BOUNDARY
    mp = multiprocessing.get_context("fork")
    signers = {}
    pids = set()
    for rank in (3, 1, 0, 2):
        inbound, outbound = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=sign_rank,
            args=(outbound, lease, tokens, rank, keys[rank], sources[rank]),
        )
        proc.start()
        outbound.close()
        try:
            assert inbound.poll(10), "CPU rank did not issue a receipt"
            pid, signed = inbound.recv()
            if isinstance(signed, BaseException):
                raise signed
            pids.add(pid)
            signers[rank] = signed
        finally:
            inbound.close()
            proc.join(5)
            if proc.is_alive():
                proc.terminate()
                proc.join()
        assert proc.exitcode == 0
    assert len(pids) == 4
    proof = GlmTerminalManifest(lease.sha256, tuple(signers[r] for r in range(4)))
    return scheduler, owner, lease, proof, received, keys, host_complete, wait_checks


def test_native_terminal_quorum_drives_exact_stock_transport_lifetime(active_manifest):
    scheduler, owner, lease, proof, received, _, host_complete, checks = active_manifest
    with pytest.raises(ValueError, match="pending writes"):
        owner.begin_reads(lease, proof)
    host_complete[0] = True
    permit = owner.begin_reads(lease, proof)
    assert permit.manifest_sha256 == proof.sha256
    assert checks == [lease.sha256]
    assert len({value.checkpoint.target_sha256 for value in proof.ranks}) == 1
    assert len({value.checkpoint.draft_sha256 for value in proof.ranks}) == 1
    assert [
        sum(
            p.shape[0]
            for p in item.checkpoint.plan.parts
            if p.role == "target:mla.target:mla"
        )
        for item in proof.ranks
    ] == [18, 16, 16, 16]
    for signed in proof.ranks:
        [select] = signed.checkpoint.kda
        assert select.accepted_row == 1 and select.conv_block != select.recurrent_block
        assert signed.checkpoint.tail_positions[0][1] == (64, 65)
    req_id = lease.identity.request_id
    request = scheduler.requests[req_id]
    assert all(counts(scheduler, request)[:2])
    for rank in (2, 3, 1, 0):
        settled = receipt(permit, rank)
        received.add(settled)
        assert owner.settle(lease, permit, settled) == (rank == 0)
        assert (req_id in scheduler.requests) == (rank != 0)


@pytest.mark.parametrize(
    "corruption",
    ["duplicate", "signature", "role", "frontier", "lease", "tail", "format"],
)
def test_verifier_rejects_partial_wrong_or_stale_signed_evidence(
    active_manifest, corruption
):
    _, owner, lease, proof, _, keys, host_complete, checks = active_manifest
    host_complete[0] = True
    rank0, *remaining = proof.ranks
    if corruption == "duplicate":
        changed = replace(proof, ranks=(rank0, *remaining[1:], rank0))
    elif corruption == "lease":
        changed = replace(proof, lease_sha256="f" * 64)
    elif corruption == "signature":
        bad = replace(rank0, signature=remaining[0].signature)
        changed = replace(proof, ranks=(bad, *remaining))
    else:
        payload = rank0.checkpoint
        if corruption == "role":
            parts = tuple(
                p
                for p in payload.plan.parts
                if part_identity(p.role).cache_role is not CacheRole.INDEXER
            )
            plan = replace(
                payload.plan,
                parts=parts,
                required_roles=tuple(sorted({p.role for p in parts})),
            )
            payload = replace(payload, plan=plan)
        elif corruption == "tail":
            payload = replace(payload, tail_positions=())
        elif corruption == "format":
            parts = tuple(
                replace(p, shape=(p.shape[0], 528), nbytes=p.shape[0] * 528)
                if part_identity(p.role).cache_role is CacheRole.MLA
                else p
                for p in payload.plan.parts
            )
            payload = replace(payload, plan=replace(payload.plan, parts=parts))
        else:
            payload = replace(payload, draft_sha256="f" * 64)
        bad = SignedTerminalRank(
            payload, TerminalRankSigner(0, keys[0])._sign(asdict(payload))
        )
        changed = replace(proof, ranks=(bad, *remaining))
    with pytest.raises(ValueError):
        owner.begin_reads(lease, changed)
    assert checks == []
    assert owner.lease_for(lease.identity.request_id) is lease
