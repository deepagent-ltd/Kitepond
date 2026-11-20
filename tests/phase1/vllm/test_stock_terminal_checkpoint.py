# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch
from kitepond.integrations.vllm.checkpoint import CheckpointReadReceipt
from kitepond.integrations.vllm.terminal_checkpoint import (
    StockTerminalCheckpointScheduler,
    TerminalTransferIdentity,
)

from vllm.config import CacheConfig, KVTransferConfig, ModelConfig, VllmConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnector,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

BLOCK = 16
MANIFEST = "c" * 64


class CpuFixtureHiSparseConnector(HiSparseConnector):
    """Select the stock scheduler connector without starting its GPU worker."""


@pytest.fixture
def capture(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "opt",
                "architectures": ["OPTForCausalLM"],
                "vocab_size": 64,
                "hidden_size": 16,
                "word_embed_proj_dim": 16,
                "ffn_dim": 32,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "max_position_embeddings": 128,
                "pad_token_id": 0,
            }
        )
    )
    model = ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=128,
    )

    def setup(
        *,
        enable_proofs=True,
        host_layers=("source",),
        device_layers=("index",),
        verify_native_checkpoint=None,
    ):
        cfg = VllmConfig(
            model_config=model,
            cache_config=CacheConfig(block_size=BLOCK, enable_prefix_caching=False),
            scheduler_config=SchedulerConfig(
                max_model_len=128,
                max_num_batched_tokens=128,
                max_num_seqs=2,
                is_encoder_decoder=False,
                async_scheduling=False,
            ),
            kv_transfer_config=KVTransferConfig(
                kv_connector="CpuFixtureHiSparseConnector",
                kv_connector_module_path=__name__,
                kv_role="kv_both",
                engine_id="D",
                kv_connector_extra_config={"host_pool_gib": 1},
            ),
        )
        spec = FullAttentionSpec(
            block_size=BLOCK,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        )
        groups = [
            KVCacheGroupSpec(
                list(host_layers),
                spec,
                role=KVCacheGroupRole.HISPARSE_SOURCE,
                host_resident=True,
            ),
            KVCacheGroupSpec(
                list(device_layers), spec, role=KVCacheGroupRole.HISPARSE_INDEXER
            ),
            KVCacheGroupSpec(
                ["resident"],
                HiSparseResidentSpec(block_size=BLOCK, page_size=4 * BLOCK),
                enable_kv_transfer=False,
            ),
            KVCacheGroupSpec(
                ["hot"],
                HiSparseHotSpec(
                    block_size=BLOCK,
                    page_size=4 * BLOCK,
                    blocks_per_request=2,
                ),
                enable_kv_transfer=False,
            ),
        ]
        cache = KVCacheConfig(
            num_blocks=32,
            hisparse_host_num_blocks=16,
            kv_cache_tensors=[],
            kv_cache_groups=groups,
        )
        cfg.cache_config.num_gpu_blocks = cache.num_blocks
        register_all_kvcache_specs(cfg)
        scheduler = StockTerminalCheckpointScheduler(
            vllm_config=cfg,
            kv_cache_config=cache,
            structured_output_manager=StructuredOutputManager(cfg),
            log_stats=False,
            block_size=BLOCK,
        )
        clock = [1]
        permitted = set()

        def verify_checkpoint(lease, proof):
            if proof != (lease.sha256, MANIFEST):
                raise ValueError("Wrong fixture terminal proof")
            return MANIFEST

        def verify_receipt(permit, receipt):
            if receipt not in permitted:
                raise ValueError("Missing fixture transport attestation")

        check = (
            dict(
                verify_checkpoint=verify_native_checkpoint or verify_checkpoint,
                verify_settlement=verify_receipt,
            )
            if enable_proofs
            else {}
        )
        owner = scheduler.bind_terminal_checkpoints(
            source_boot_id="D-boot", clock=lambda: clock[0], **check
        )
        return scheduler, owner, clock, permitted

    return setup


def identity(request_id, *, transfer=None, boot="D-boot"):
    return TerminalTransferIdentity(
        request_id,
        transfer or f"reverse-{request_id}",
        2,
        "D",
        boot,
        "a" * 64,
        "b" * 64,
        deadline_monotonic_ns=100,
    )


def submit(scheduler, owner, request_id="r", *, arm=True):
    request = Request(
        request_id=request_id,
        prompt_token_ids=list(range(1, 17)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    scheduler.add_request(request)
    if arm:
        owner.arm(request, identity(request_id))
    step = scheduler.schedule()
    output = ModelRunnerOutput(
        req_ids=[request_id],
        req_id_to_index={request_id: 0},
        sampled_token_ids=[[33]],
    )
    return request, step, output


def counts(scheduler, request):
    blocks = scheduler.kv_cache_manager.get_blocks(request.request_id).blocks
    return [[block.ref_cnt for block in group if not block.is_null] for group in blocks]


def receipt(permit, rank, status="acked"):
    return CheckpointReadReceipt(permit.sha256, rank, status, 0)


def test_real_native_terminal_holds_native_non_hot_refs_until_quorum(capture):
    scheduler, owner, _, attested = capture()
    request, step, output = submit(scheduler, owner)
    before = counts(scheduler, request)
    assert all(before[group_id] for group_id in (0, 1, 2))
    reply = scheduler.update_from_output(step, output)
    assert reply[0].outputs[0].new_token_ids == [33]
    assert request.status is RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler.requests[request.request_id] is request
    lease = owner.lease_for(request.request_id)
    assert lease is not None
    assert lease.requested_common_tokens == 16 and lease.visible_tokens == 17
    assert lease.next_token_id == 33
    assert [(g.group_id, g.location, g.layer_names) for g in lease.groups] == [
        (0, "host", ("source",)),
        (1, "device", ("index",)),
    ]
    after = counts(scheduler, request)
    assert after[:2] == [[value + 1 for value in group] for group in before[:2]]
    assert after[2:] == before[2:]
    assert lease.step_index == scheduler.current_step
    with pytest.raises(ValueError, match="Wrong fixture terminal proof"):
        owner.begin_reads(lease, ("x", MANIFEST))
    permit = owner.begin_reads(lease, (lease.sha256, MANIFEST))
    owner.require_read_open(lease, permit)
    with pytest.raises(ValueError, match="still owns"):
        CheckpointReadReceipt(permit.sha256, 3, "abort_settled", 1)
    received = receipt(permit, 3)
    with pytest.raises(ValueError, match="did not authorize"):
        owner.settle(lease, replace(permit), received)
    with pytest.raises(ValueError, match="attestation"):
        owner.settle(lease, permit, received)
    attested.add(received)
    assert owner.settle(lease, permit, received) is False
    assert owner.settle(lease, permit, received) is False
    with pytest.raises(ValueError, match="Conflicting"):
        owner.settle(lease, permit, receipt(permit, 3, "abort_settled"))
    for rank in (1, 0, 2):
        settled = receipt(permit, rank, "abort_settled" if rank == 0 else "acked")
        attested.add(settled)
        assert owner.settle(lease, permit, settled) is (rank == 2)
    assert owner.settle(lease, permit, received)
    assert owner.lease_for(request.request_id) is None
    assert request.request_id not in scheduler.requests
    assert counts(scheduler, request) == [[], [], [], []]


def test_cancel_or_timeout_before_read_releases_request_refs(capture):
    scheduler, owner, clock, _ = capture(enable_proofs=False)
    request, step, output = submit(scheduler, owner)
    scheduler.update_from_output(step, output)
    lease = owner.lease_for(request.request_id)
    assert lease is not None
    with pytest.raises(RuntimeError, match="verifiers are unavailable"):
        owner.begin_reads(lease, object())
    assert not owner.expire(lease)
    clock[0] = 100
    assert owner.expire(lease)
    assert request.request_id not in scheduler.requests
    assert owner.lease_for(request.request_id) is None
    with pytest.raises(ValueError, match="stale"):
        owner.cancel(lease)


def test_started_read_timeout_retains_request_until_abort_receipts(capture):
    scheduler, owner, clock, attested = capture()
    request, step, output = submit(scheduler, owner)
    scheduler.update_from_output(step, output)
    lease = owner.lease_for(request.request_id)
    assert lease is not None
    permit = owner.begin_reads(lease, (lease.sha256, MANIFEST))
    held_refs = counts(scheduler, request)
    clock[0] = 100
    assert not owner.expire(lease)
    assert not owner.cancel(lease)
    with pytest.raises(ValueError, match="cannot open"):
        owner.require_read_open(lease, permit)
    assert counts(scheduler, request) == held_refs
    for rank in (2, 0, 3, 1):
        done = receipt(permit, rank, "abort_settled")
        attested.add(done)
        assert owner.settle(lease, permit, done) is (rank == 1)
    assert request.request_id not in scheduler.requests


def test_native_abort_and_stale_completion_cannot_capture(capture):
    scheduler, owner, _, _ = capture()
    request, step, output = submit(scheduler, owner)
    with pytest.raises(ValueError, match="tracked step"):
        scheduler.update_from_output(replace(step), output)
    with ThreadPoolExecutor(max_workers=1) as thread:
        future = thread.submit(owner.lease_for, request.request_id)
        with pytest.raises(RuntimeError, match="belong to the scheduler"):
            future.result()
    scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
    scheduler.update_from_output(step, output)
    assert request.request_id not in scheduler.requests
    assert owner.lease_for(request.request_id) is None


@pytest.mark.parametrize("native_first", [False, True])
def test_native_connector_retirement_is_independent(capture, monkeypatch, native_first):
    scheduler, owner, _, _ = capture()
    native = scheduler.connector.request_finished_all_groups

    def held(request, blocks):
        _, payload = native(request, blocks)
        return True, payload

    monkeypatch.setattr(scheduler.connector, "request_finished_all_groups", held)
    request, step, output = submit(scheduler, owner)
    before = counts(scheduler, request)
    scheduler.update_from_output(step, output)
    lease = owner.lease_for(request.request_id)
    assert lease is not None
    native_done = KVConnectorOutput(finished_sending={request.request_id})
    if native_first:
        scheduler._update_from_kv_xfer_finished(native_done)
        assert request.request_id in scheduler.requests
    assert owner.cancel(lease)
    if not native_first:
        assert request.request_id in scheduler.requests
        assert counts(scheduler, request) == before
        scheduler._update_from_kv_xfer_finished(native_done)
    assert request.request_id not in scheduler.requests


def test_wrong_boot_and_duplicate_active_transfer_are_rejected(capture):
    scheduler, owner, _, _ = capture()
    req, step, output = submit(scheduler, owner, arm=False)
    with pytest.raises(ValueError, match="cannot arm"):
        owner.arm(req, identity(req.request_id, boot="obsolete"))
    scheduler.update_from_output(step, output)
    assert req.request_id not in scheduler.requests

    req, step, output = submit(scheduler, owner, request_id="terminal")
    scheduler.update_from_output(step, output)
    waiting = Request(
        request_id="next",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    scheduler.add_request(waiting)
    with pytest.raises(ValueError, match="cannot arm"):
        owner.arm(waiting, identity("next", transfer="reverse-terminal"))
    lease = owner.lease_for(req.request_id)
    assert lease is not None and owner.cancel(lease)
