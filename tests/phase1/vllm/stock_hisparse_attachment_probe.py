# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run against the isolated engine seam proposal or the untouched stock control."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from kitepond.integrations.vllm.decode_completion import (
    DecodeImportRankReceipt,
    GrantRoutedScheduler,
)
from kitepond.integrations.vllm.hisparse_attachment import ATTACHMENT_MODE
from kitepond.integrations.vllm.host_authority import (
    HostRankSettlement,
    NativeHostAuthority,
)
from kitepond.integrations.vllm.profile import UnsupportedKitepondProfile

from vllm.config import CacheConfig, KVTransferConfig, ModelConfig, VllmConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnector,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.hisparse.coordinator import HiSparseCoordinator, get_hisparse_coordinator
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


@pytest.fixture
def create_scheduler(tmp_path):
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
        ),
        encoding="utf-8",
    )

    def make(
        stage,
        mode=ATTACHMENT_MODE,
        *,
        routed=False,
        force_async=False,
        extra_config=None,
        scheduler_cls=None,
    ):
        model = ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="float32",
            max_model_len=128,
        )
        identity = "D" if stage == "decode" else "P"
        attachment = {
            "kv_connector": "KitepondHiSparseAttachmentConnector",
            "kv_connector_module_path": (
                "kitepond.integrations.vllm.hisparse_attachment"
            ),
            "kv_role": "kv_both",
        }
        if routed:
            # This CPU request fixture already supplies native cache groups.
            # Legacy name avoids automatic CUDA discovery for its OPT metadata.
            transfer = KVTransferConfig(
                kv_connector="MultiConnector",
                kv_role="kv_both",
                engine_id=identity,
                kv_connector_extra_config={
                    "connectors": [
                        attachment,
                        {"kv_connector": "MooncakeConnector", "kv_role": "kv_consumer"},
                    ]
                },
            )
        else:
            transfer = KVTransferConfig(**attachment, engine_id=identity)
        config = VllmConfig(
            model_config=model,
            cache_config=CacheConfig(block_size=BLOCK, enable_prefix_caching=False),
            scheduler_config=SchedulerConfig(
                max_model_len=128,
                max_num_batched_tokens=128,
                max_num_seqs=2,
                is_encoder_decoder=False,
                async_scheduling=False,
            ),
            kv_transfer_config=transfer,
            additional_config={
                "kitepond": {
                    "mode": mode,
                    "stage": stage,
                    "authority_engine_id": "D",
                    "boot_id": "boot",
                },
                **(extra_config or {}),
            },
        )
        full = FullAttentionSpec(
            block_size=BLOCK,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        )
        groups = [
            KVCacheGroupSpec(
                ["source"],
                full,
                host_resident=True,
                role=KVCacheGroupRole.HISPARSE_SOURCE,
            ),
            KVCacheGroupSpec(["index"], full, role=KVCacheGroupRole.HISPARSE_INDEXER),
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
        config.cache_config.num_gpu_blocks = cache.num_blocks
        register_all_kvcache_specs(config)
        scheduler_cls = scheduler_cls or (GrantRoutedScheduler if routed else Scheduler)
        if force_async:
            config.scheduler_config.async_scheduling = True
        return scheduler_cls(
            vllm_config=config,
            kv_cache_config=cache,
            structured_output_manager=StructuredOutputManager(config),
            log_stats=False,
            block_size=BLOCK,
        )

    return make


def _request(tokens, request_id="request"):
    return Request(
        request_id=request_id,
        prompt_token_ids=[3] * tokens,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )


def _reserve(authority, request, tokens):
    return authority.reserve(
        request.request_id,
        f"transfer-{request.request_id}",
        17,
        prefix_tokens=0,
        reservation_tokens=tokens,
        source=(),
    )


def _finish(scheduler, request):
    scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)


def _settled_grant(scheduler, request):
    authority = scheduler.connector.authority
    grant = _reserve(authority, request, BLOCK)
    authority.dispatch(grant)
    for rank in range(4):
        assert authority.settle(
            grant, HostRankSettlement(grant.sha256, rank, "ready", 0, 0)
        ) == (rank == 3)
    return grant


def _update(scheduler, step, completion):
    return scheduler.update_from_output(
        step, ModelRunnerOutput.with_kv_conn_output_only(completion)
    )


def test_untouched_stock_rejects_attachment_mode(create_scheduler):
    assert not hasattr(HiSparseCoordinator, "attach_external_host_blocks")
    with pytest.raises(UnsupportedKitepondProfile, match="attachment is unavailable"):
        create_scheduler("decode")


def test_probe_configuration_is_explicit(create_scheduler):
    with pytest.raises(UnsupportedKitepondProfile, match="Explicit verified"):
        create_scheduler("prefill", mode="full")


def test_prefill_attaches_through_real_lookup_reuses_on_retry_and_retires(
    create_scheduler,
):
    scheduler = create_scheduler("prefill")
    connector = scheduler.connector
    device = scheduler.kv_cache_manager.block_pool
    free = device.get_num_free_blocks()
    d_pool = BlockPool(num_gpu_blocks=16, enable_caching=False, hash_block_size=BLOCK)
    authority = NativeHostAuthority(
        d_pool, engine_id="D", boot_id="boot", block_tokens=BLOCK
    )
    free_d = d_pool.get_num_free_blocks()
    request = _request(4)
    scheduler.add_request(request)
    assert not scheduler.schedule().num_scheduled_tokens

    grant = _reserve(authority, request, 2 * BLOCK)
    connector.stage_host_grant(request, grant)
    with pytest.raises(ValueError, match="ID changed"):
        connector.get_num_new_matched_tokens(_request(4), 0)

    withheld = device.get_new_blocks(free)
    assert not scheduler.schedule().num_scheduled_tokens
    device.free_blocks(withheld)
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {request.request_id: 4}
    assert step.scheduled_new_reqs[0].block_ids[0] == [grant.writable_pages[0]]
    assert d_pool.get_num_free_blocks() == free_d - 2
    with pytest.raises(ValueError, match="not eligible"):
        connector.retire_prefill_grant(request)
    _finish(scheduler, request)
    assert request.request_id not in scheduler.requests
    assert connector.retire_prefill_grant(request) is grant
    assert d_pool.get_num_free_blocks() == free_d - 2
    assert authority.cancel(grant)
    assert d_pool.get_num_free_blocks() == free_d
    assert device.get_num_free_blocks() == free


def test_decode_retry_reuses_owned_grant_and_abort_waits_for_verified_completion(
    create_scheduler,
):
    scheduler = create_scheduler("decode")
    connector = scheduler.connector
    pool = connector.authority.pool
    free = pool.get_num_free_blocks()
    device = scheduler.kv_cache_manager.block_pool
    request = _request(BLOCK + 4)
    grant = _settled_grant(scheduler, request)
    connector.stage_host_grant(request, grant, imported_tokens=BLOCK)
    withheld = device.get_new_blocks(device.get_num_free_blocks())
    scheduler.add_request(request)
    assert not scheduler.schedule().num_scheduled_tokens
    assert connector.pending_decode_import(request) is None
    assert pool.get_num_free_blocks() == free - 1
    device.free_blocks(withheld)

    step = scheduler.schedule()
    binding = connector.pending_decode_import(request)
    assert binding.request is request and binding.grant is grant
    assert binding.block_ids[0] == grant.writable_pages
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert not step.num_scheduled_tokens
    assert not torch.cuda.is_initialized()
    _finish(scheduler, request)
    assert request.request_id in scheduler.requests
    assert pool.get_num_free_blocks() == free - 1

    failed = KVConnectorOutput(
        finished_recving={request.request_id}, failed_recving={request.request_id}
    )
    with pytest.raises(ValueError, match="matching verified"):
        _update(scheduler, step, failed)
    assert pool.get_num_free_blocks() == free - 1

    def verify(actual_binding, output):
        assert actual_binding is binding and output is failed

    connector.bind_decode_completion_verifier(verify)
    _update(scheduler, step, failed)
    assert request.request_id not in scheduler.requests
    assert connector.pending_decode_import(request) is None
    assert pool.get_num_free_blocks() == free


def test_decode_requires_host_quorum_before_native_attachment(create_scheduler):
    scheduler = create_scheduler("decode")
    connector = scheduler.connector
    request = _request(BLOCK + 4)
    grant = _reserve(connector.authority, request, BLOCK)
    connector.authority.dispatch(grant)
    connector.stage_host_grant(request, grant, imported_tokens=BLOCK)
    scheduler.add_request(request)
    with pytest.raises(ValueError, match="ready quorum"):
        scheduler.schedule()
    coordinator = get_hisparse_coordinator(scheduler.kv_cache_manager)
    assert not coordinator.host_manager.req_to_blocks
    for rank in range(4):
        connector.authority.settle(
            grant, HostRankSettlement(grant.sha256, rank, "ready", 0, 0)
        )
    assert not scheduler.schedule().num_scheduled_tokens
    assert connector.pending_decode_import(request).grant is grant


def test_decode_native_attachment_conflict_preserves_grant_for_retry(create_scheduler):
    scheduler = create_scheduler("decode")
    connector = scheduler.connector
    pool = connector.authority.pool
    free = pool.get_num_free_blocks()
    request = _request(BLOCK + 4)
    grant = _settled_grant(scheduler, request)
    connector.stage_host_grant(request, grant, imported_tokens=BLOCK)
    scheduler.add_request(request)
    # A real prior native allocation makes the request ineligible for attach.
    assert scheduler.kv_cache_manager.allocate_slots(request, num_new_tokens=4)
    assert pool.get_num_free_blocks() == free - 2
    with pytest.raises(ValueError, match="idle request"):
        scheduler.schedule()
    assert pool.get_num_free_blocks() == free - 2
    assert connector.pending_decode_import(request) is None
    scheduler.kv_cache_manager.free(request)
    assert pool.get_num_free_blocks() == free - 1

    step = scheduler.schedule()
    binding = connector.pending_decode_import(request)
    assert binding.grant is grant
    assert binding.block_ids[0] == grant.writable_pages

    loaded = KVConnectorOutput(finished_recving={request.request_id})

    def verify(actual, output):
        if actual is not binding or output is not loaded:
            raise ValueError("wrong worker fence")

    connector.bind_decode_completion_verifier(verify)
    with pytest.raises(ValueError, match="wrong worker fence"):
        _update(
            scheduler, step, KVConnectorOutput(finished_recving={request.request_id})
        )
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert connector.pending_decode_import(request) is binding
    _update(scheduler, step, loaded)
    assert connector.pending_decode_import(request) is None
    next_step = scheduler.schedule()
    assert next_step.num_scheduled_tokens == {request.request_id: 4}
    _finish(scheduler, request)
    assert request.request_id not in scheduler.requests
    coordinator = get_hisparse_coordinator(scheduler.kv_cache_manager)
    transfers = tuple(coordinator.pending_spills)
    assert transfers and pool.get_num_free_blocks() == free - 1
    counts = dict.fromkeys(transfers, 1)
    _update(
        scheduler,
        next_step,
        KVConnectorOutput(
            kv_connector_worker_meta=HiSparseConnectorWorkerMetadata(counts, counts)
        ),
    )
    assert pool.get_num_free_blocks() == free


def test_module_path_selects_one_named_hisparse_child_and_preserves_stock_budget():
    from kitepond.integrations.vllm.hisparse_attachment import (
        HiSparseConnector,
        KitepondHiSparseAttachmentConnector,
    )

    from vllm.config.kv_transfer import hisparse_host_pool_gib
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
        HiSparseConnector as NativeHiSparseConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.stats import (
        HiSparseKVConnectorStats,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )

    module = "kitepond.integrations.vllm.hisparse_attachment"
    child = {
        "kv_connector": "HiSparseConnector",
        "kv_connector_module_path": module,
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"host_pool_gib": 4},
    }
    transfer = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        engine_id="D",
        kv_connector_extra_config={"connectors": [child]},
    )
    assert transfer.has_connector("HiSparseConnector")
    assert hisparse_host_pool_gib(transfer) == 4
    assert KVConnectorFactory.supports_hma_config(transfer)
    (selected,) = MultiConnector._get_connector_classes_and_configs(
        SimpleNamespace(kv_transfer_config=transfer)
    )
    assert selected[0] is HiSparseConnector is KitepondHiSparseAttachmentConnector
    assert selected[0].__name__ == "HiSparseConnector"
    assert issubclass(selected[0], NativeHiSparseConnector)
    assert selected[1].kv_transfer_config.engine_id == "D"
    assert (
        KVConnectorFactory.get_connector_class_by_name("HiSparseConnector")
        is NativeHiSparseConnector
    )

    stats = HiSparseKVConnectorStats()
    stats.record_snapshot(3, 2, 1312)
    decoded = MultiConnector.build_kv_connector_stats(
        {selected[0].__name__: stats.to_dict()}
    )
    assert decoded is not None
    assert (
        decoded.reduce()["HiSparseConnector"]["HiSparse host-to-device bytes"] == 1312
    )

    legacy = child | {"kv_connector": "KitepondHiSparseAttachmentConnector"}
    named = KVTransferConfig(**legacy)
    assert not named.has_connector("HiSparseConnector")
    assert hisparse_host_pool_gib(named) is None
    assert KVConnectorFactory.get_connector_class(named) is HiSparseConnector

    transfer.kv_connector_extra_config["connectors"].append(
        {
            key: value
            for key, value in child.items()
            if key != "kv_connector_module_path"
        }
    )
    with pytest.raises(ValueError, match="Only one HiSparseConnector"):
        hisparse_host_pool_gib(transfer)


def _quorum(binding, *, outcome="loaded"):
    return tuple(
        DecodeImportRankReceipt(
            binding.grant.sha256,
            binding.destination_sha256,
            rank,
            outcome,
            0,
            0,
        )
        for rank in range(4)
    )


def test_stock_mooncake_ids_do_not_settle_protected_decode_grants(create_scheduler):
    scheduler = create_scheduler("decode", routed=True)
    attachment, mooncake = scheduler.connector.sub_connectors
    assert type(mooncake) is MooncakeConnector
    protected = _request(BLOCK + 4, "protected")
    grant = _reserve(attachment.authority, protected, BLOCK)
    attachment.authority.dispatch(grant)
    for rank in range(4):
        attachment.authority.settle(
            grant, HostRankSettlement(grant.sha256, rank, "ready", 0, 0)
        )
    attachment.stage_host_grant(protected, grant, imported_tokens=BLOCK)
    scheduler.add_request(protected)
    ordinary = _request(BLOCK + 4, "ordinary")
    ordinary.kv_transfer_params = {
        "do_remote_prefill": True,
        "remote_engine_id": "P",
        "remote_bootstrap_addr": "127.0.0.1:9",
        "transfer_id": "native-mooncake-transfer",
    }
    scheduler.add_request(ordinary)
    step = scheduler.schedule()
    binding = attachment.pending_decode_import(protected)
    assert binding is not None
    assert ordinary.status == protected.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert "ordinary" in step.kv_connector_metadata.metadata[1].reqs_to_recv["P"]
    assert "protected" not in step.kv_connector_metadata.metadata[1].reqs_to_recv["P"]

    # The unmodified worker aggregator can produce the same ID from unrelated
    # events. Its result has lost child/transfer identity.
    aggregated = KVOutputAggregator(expected_finished_count=4).aggregate(
        [
            ModelRunnerOutput.with_kv_conn_output_only(
                KVConnectorOutput(
                    finished_sending={"protected"},
                    finished_recving={"ordinary", "protected"},
                    failed_recving={"protected"},
                )
            )
            for _ in range(4)
        ]
    )
    scheduler.update_from_output(step, aggregated)
    assert scheduler.finished_recving_kv_req_ids == {"ordinary"}
    assert not scheduler.failed_recving_kv_req_ids
    assert attachment.pending_decode_import(protected) is binding
    assert (
        protected.request_id
        in scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks
    )
    assert "protected" in aggregated.kv_connector_output.failed_recving

    receipts = ()
    scheduler.bind_verified_decode_imports(
        lambda active: receipts if active is binding else None
    )
    good = _quorum(binding)
    receipts = tuple(replace(receipt, grant_sha256="a" * 64) for receipt in good)
    _update(scheduler, step, KVConnectorOutput(finished_recving={"protected"}))
    assert "protected" not in scheduler.finished_recving_kv_req_ids
    receipts = good[:-1] + (replace(good[-1], destination_sha256="b" * 64),)
    _update(scheduler, step, KVConnectorOutput(finished_recving={"protected"}))
    assert attachment.pending_decode_import(protected) is binding
    assert "protected" not in scheduler.finished_recving_kv_req_ids

    receipts = good
    _update(scheduler, step, None)
    assert scheduler.finished_recving_kv_req_ids == {"ordinary", "protected"}
    assert attachment.pending_decode_import(protected) is None
    next_step = scheduler.schedule()
    assert next_step.num_scheduled_tokens["protected"] == 4


def test_protected_abort_releases_only_after_current_settled_d_quorum(create_scheduler):
    scheduler = create_scheduler("decode", routed=True)
    attachment, _ = scheduler.connector.sub_connectors
    pool = attachment.authority.pool
    free = pool.get_num_free_blocks()
    request = _request(BLOCK + 4, "protected")
    grant = _reserve(attachment.authority, request, BLOCK)
    attachment.authority.dispatch(grant)
    for rank in range(4):
        attachment.authority.settle(
            grant, HostRankSettlement(grant.sha256, rank, "ready", 0, 0)
        )
    attachment.stage_host_grant(request, grant, imported_tokens=BLOCK)
    scheduler.add_request(request)
    step = scheduler.schedule()
    binding = attachment.pending_decode_import(request)
    assert binding is not None
    with pytest.raises(ValueError, match="request-scoped transfer failures"):
        _update(
            scheduler,
            step,
            KVConnectorOutput(
                invalid_block_ids={binding.block_ids[0][0]},
            ),
        )
    _finish(scheduler, request)
    assert scheduler.requests.get(request.request_id) is request
    assert pool.get_num_free_blocks() == free - 1

    receipts = _quorum(binding, outcome="abort_settled")[:-1]
    scheduler.bind_verified_decode_imports(
        lambda active: receipts if active is binding else None
    )
    false_failure = KVConnectorOutput(
        finished_recving={request.request_id}, failed_recving={request.request_id}
    )
    _update(scheduler, step, false_failure)
    assert scheduler.requests.get(request.request_id) is request
    assert pool.get_num_free_blocks() == free - 1

    receipts = _quorum(binding, outcome="abort_settled")
    _update(scheduler, step, None)
    assert request.request_id not in scheduler.requests
    assert attachment.pending_decode_import(request) is None
    assert scheduler.verified_decode_import(request) is None
    assert pool.get_num_free_blocks() == free


def test_routed_scheduler_rejects_async_mode(create_scheduler):
    with pytest.raises(ValueError, match="requires --no-async-scheduling"):
        create_scheduler("decode", routed=True, force_async=True)


def test_native_receipt_metadata_routes_only_the_attached_four_ranks(create_scheduler):
    from kitepond.integrations.vllm.decode_receive import (
        DecodeReceiveProofs,
        DecodeReceiveTicket,
        DecodeReceiveWork,
    )

    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiKVConnectorWorkerMetadata,
    )

    scheduler = create_scheduler("decode", routed=True)
    attachment, _ = scheduler.connector.sub_connectors
    request = _request(BLOCK + 4, "native-post-fence")
    grant = _reserve(attachment.authority, request, BLOCK)
    attachment.authority.dispatch(grant)
    for rank in range(4):
        attachment.authority.settle(
            grant, HostRankSettlement(grant.sha256, rank, "ready", 0, 0)
        )
    attachment.stage_host_grant(request, grant, imported_tokens=BLOCK)
    scheduler.add_request(request)
    step = scheduler.schedule()
    binding = attachment.pending_decode_import(request)
    expected = DecodeReceiveTicket.from_binding(binding)
    work = step.kv_connector_metadata.metadata[0]
    assert isinstance(work, DecodeReceiveWork)
    assert work.tickets == (expected,)
    assert expected.destination_sha256 == binding.destination_sha256
    receipts = _quorum(binding)

    def packet(receipt, slot=0):
        children = [None, None]
        children[slot] = DecodeReceiveProofs({}, {}, receipts=(receipt,))
        return ModelRunnerOutput.with_kv_conn_output_only(
            KVConnectorOutput(
                finished_recving={request.request_id},
                kv_connector_worker_meta=MultiKVConnectorWorkerMetadata(
                    metadata=tuple(children),
                ),
            )
        )

    aggregate = KVOutputAggregator(expected_finished_count=4)
    scheduler.update_from_output(
        step, aggregate.aggregate([packet(receipt) for receipt in receipts[:3]])
    )
    assert attachment.pending_decode_import(request) is binding
    scheduler.update_from_output(step, packet(receipts[3], slot=1))
    assert attachment.pending_decode_import(request) is binding
    assert scheduler.verified_decode_import(request) is None
    scheduler.update_from_output(step, packet(receipts[3]))
    assert attachment.pending_decode_import(request) is None
    assert request.request_id in scheduler.finished_recving_kv_req_ids
    assert scheduler.verified_decode_import(request) is binding
    assert (
        scheduler.verified_decode_import(_request(BLOCK + 4, request.request_id))
        is None
    )
    _finish(scheduler, request)
    assert scheduler.verified_decode_import(request) is None


def test_native_cold_decode_consumes_only_authenticated_prefill(
    create_scheduler, tmp_path, monkeypatch
):
    from dataclasses import asdict
    from time import monotonic_ns

    from kitepond.core.readiness import ReadyReceipt
    from kitepond.integrations.vllm.glm_cold_prefill_admission import (
        ColdPrefillControl,
        ColdPrefillPrompt,
    )
    from kitepond.integrations.vllm.glm_cold_prefill_scheduler import (
        GlmColdDecodeAdmissionMixin,
    )

    class ColdDecodeScheduler(GlmColdDecodeAdmissionMixin, GrantRoutedScheduler):
        pass

    key = tmp_path / "control.key"
    key.write_bytes(bytes(range(32)))
    key.chmod(0o600)
    monkeypatch.setenv("KITEPOND_GLM_COLD_PREFILL_KEY_FILE", str(key))
    scheduler = create_scheduler(
        "decode",
        routed=True,
        scheduler_cls=ColdDecodeScheduler,
        extra_config={
            "kitepond_glm_cold_prefill": {
                "p_boot_id": "P-1",
                "max_pending": 1,
                "arm_timeout_seconds": 15,
            },
        },
    )
    owner = scheduler.connector.sub_connectors[0]
    control = scheduler.connector.get_control_handler("kitepond.glm-cold-prefill.v1")
    assert control is not None
    prompt = ColdPrefillPrompt.from_tokens(
        "cold-D", "transfer-cold", 1, [3] * (BLOCK + 1)
    )
    initially_free = owner.authority.observe().available_pages
    refused = _request(BLOCK + 1, request_id=prompt.request_id)
    refused.kv_transfer_params = {
        "do_remote_prefill": True,
        "transfer_id": prompt.transfer_id,
    }
    reserved = control("reserve", asdict(prompt))
    assert reserved["status"] == "offered"
    assert scheduler.connector.get_control_handler("foreign-namespace") is None
    assert owner.authority.observe().available_pages == initially_free - 2
    with pytest.raises(ValueError, match="does not match its authenticated prefill"):
        scheduler.add_request(refused)
    assert refused.request_id not in scheduler.requests

    signer = ColdPrefillControl(
        d_engine_id="D", d_boot_id="boot", p_boot_id="P-1", key=key.read_bytes()
    )
    offer = signer.open_offer(reserved["offer"], prompt)
    receipts = tuple(
        ReadyReceipt(offer.attempt, rank, "a" * 64, BLOCK, BLOCK, monotonic_ns())
        for rank in range(4)
    )
    batch = {
        "attempt": asdict(offer.attempt),
        "receipts": [asdict(item) for item in receipts],
        "mac": signer.ready_mac(offer, receipts),
    }
    assert control("confirm_ready", batch) == {"status": "ready"}
    wrong = _request(BLOCK, request_id=prompt.request_id)
    wrong.kv_transfer_params = refused.kv_transfer_params
    with pytest.raises(ValueError, match="does not match its authenticated prefill"):
        scheduler.add_request(wrong)

    refused.kv_transfer_params.update(
        remote_engine_id="P", remote_bootstrap_addr="http://127.0.0.1:18162"
    )
    scheduler.add_request(refused)
    scheduler.schedule()
    binding = owner.pending_decode_import(refused)
    assert binding is not None and binding.request is refused
    assert binding.imported_tokens == BLOCK
    assert owner.get_loaded_kv_cache_group_ids(refused) == (0, 1)
    assert binding.grant.sha256 == offer.grant.sha256
    assert not owner.authority.retains(binding.grant)
    assert owner.authority.observe().active_grants == 0
    assert refused.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.shutdown()
