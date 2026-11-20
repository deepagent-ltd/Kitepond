# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise public loader entry points using stock vLLM on CPU."""

import json
import os
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch
from kitepond.integrations.vllm.compat import inspect_engine
from kitepond.integrations.vllm.connector import KitepondConnector, RegistrationMetadata
from kitepond.integrations.vllm.hisparse_binding import (
    bind_decode_host,
    bind_prefill_host,
)
from kitepond.integrations.vllm.profile import UnsupportedKitepondProfile
from kitepond.integrations.vllm.ready_transport import (
    ReadyAttempt,
    ReadyCollector,
    ReadyPublisher,
    ReadyReceipt,
    ReadyWorker,
)
from kitepond.integrations.vllm.scheduler import KitepondScheduler
from kitepond.integrations.vllm.worker_hooks import WorkerLifecycleBridge

from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.scheduler import SchedulerConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    KVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
)
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.single_type_kv_cache_manager import (
    HiSparseSourceManager,
    MambaManager,
    register_all_kvcache_specs,
)
from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
from vllm.v1.hisparse.runtime import (
    drop_hisparse_host_pool,
    get_hisparse_host_pool,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
from vllm.v1.worker.worker_base import WorkerWrapperBase

INTEGRATION = "kitepond.integrations.vllm"


class _CpuDeviceLifecycle:
    """CPU-only stand-in for GPU device startup and teardown."""

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config
        self.events = []

    def init_device(self):
        self.events.append("device")

    def shutdown(self):
        self.events.append(self.kitepond_worker_registration())


class CpuStockExtensionWorker(_CpuDeviceLifecycle):
    """Use the real stock wrapper and a CPU device lifecycle."""


class CpuLegacyBridge(WorkerLifecycleBridge, _CpuDeviceLifecycle):
    """Use the legacy explicit worker lifecycle bridge with a CPU device."""


class _DeferredMetadata(KVConnectorWorkerMetadata):
    # Test marker only. It carries no readiness or ownership claim.
    def aggregate(self, other):
        return self


class DeferredMarkerConnector(KitepondConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.finalized = False

    def wait_for_save(self):
        super().wait_for_save()

        def finish_fixture_save():
            if not isinstance(self._get_connector_metadata(), RegistrationMetadata):
                raise AssertionError(
                    "Stock step metadata was cleared before its save fence"
                )
            self.finalized = True
            return ReadyReceipt(
                self.ticket.attempt, 0, self.region, 4, 4, time.monotonic_ns()
            )

        self.publisher.after_save_fence(self.ticket, finish_fixture_save)

    def build_connector_worker_meta(self):
        return _DeferredMetadata() if self.finalized else None


class ProbeStateManager(MambaManager):
    """Test a separately held native block without changing native block tables."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_hold_requests = set()
        self.test_holds = {}

    def get_num_blocks_to_allocate(self, request_id, *args, **kwargs):
        required = super().get_num_blocks_to_allocate(request_id, *args, **kwargs)
        return required + int(
            request_id in self.test_hold_requests and request_id not in self.test_holds
        )

    def allocate_new_blocks(self, request_id, *args, **kwargs):
        allocated = super().allocate_new_blocks(request_id, *args, **kwargs)
        if request_id in self.test_hold_requests and request_id not in self.test_holds:
            (self.test_holds[request_id],) = self.block_pool.get_new_blocks(1)
        return allocated

    def pop_blocks_for_free(self, request_id):
        freed = super().pop_blocks_for_free(request_id)
        self.test_hold_requests.discard(request_id)
        block = self.test_holds.pop(request_id, None)
        return freed if block is None else [*freed, block]


@dataclass(frozen=True, kw_only=True)
class ProbeStateSpec(MambaSpec):
    """A distinct test spec that preserves native Mamba allocation behavior."""


class StockRegistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.model_directory.cleanup)
        cls.model = cls.model_directory.name
        (Path(cls.model) / "config.json").write_text(
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
                    "max_position_embeddings": 32,
                    "bos_token_id": 1,
                    "eos_token_id": 2,
                    "pad_token_id": 0,
                }
            ),
            encoding="utf-8",
        )

    def config(self, mode="registration-probe"):
        model = ModelConfig(
            model=self.model,
            tokenizer=self.model,
            skip_tokenizer_init=True,
            dtype="float32",
            max_model_len=32,
        )
        return VllmConfig(
            model_config=model,
            cache_config=CacheConfig(block_size=16, enable_prefix_caching=False),
            scheduler_config=SchedulerConfig(
                max_model_len=32,
                max_num_batched_tokens=32,
                max_num_seqs=2,
                is_encoder_decoder=model.is_encoder_decoder,
                async_scheduling=False,
                scheduler_cls=f"{INTEGRATION}.scheduler.KitepondScheduler",
            ),
            kv_transfer_config=KVTransferConfig(
                kv_connector="KitepondConnector",
                kv_connector_module_path=f"{INTEGRATION}.connector",
                kv_role="kv_both",
            ),
            additional_config={"kitepond": {"mode": mode}},
        )

    def cache_spec(self, config):
        spec = FullAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32
        )
        config.cache_config.num_gpu_blocks = 8
        register_all_kvcache_specs(config)
        return KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
        )

    def make_scheduler(self, config, cache):
        scheduler_cls = config.scheduler_config.get_scheduler_cls()
        self.assertIs(scheduler_cls, KitepondScheduler)
        return scheduler_cls(
            vllm_config=config,
            kv_cache_config=cache,
            structured_output_manager=StructuredOutputManager(config),
            log_stats=False,
            block_size=16,
        )

    def test_stock_scheduler_routes_one_request_and_native_release(self):
        config = self.config()
        cache = self.cache_spec(config)
        scheduler = self.make_scheduler(config, cache)
        connector = scheduler.get_kv_connector()
        self.assertIs(type(connector), KitepondConnector)
        self.assertTrue(connector.registration_report()["manager_bound"])
        self.assertFalse(config.scheduler_config.disable_hybrid_kv_cache_manager)
        request = Request(
            request_id="stock-probe",
            prompt_token_ids=[1, 8, 9, 3],
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
        )
        pool = scheduler.kv_cache_manager.block_pool
        free_before = pool.get_num_free_blocks()
        scheduler.add_request(request)
        step = scheduler.schedule()
        self.assertLess(pool.get_num_free_blocks(), free_before)
        self.assertEqual(step.num_scheduled_tokens, {"stock-probe": 4})
        self.assertIsInstance(step.kv_connector_metadata, RegistrationMetadata)
        scheduler.finish_requests(["stock-probe"], RequestStatus.FINISHED_ABORTED)
        self.assertFalse(scheduler.requests)
        self.assertEqual(pool.get_num_free_blocks(), free_before)
        self.assertEqual(
            connector.registration_report()["callbacks"],
            {"bind_manager": 1, "lookup": 1, "allocate": 1, "finish": 1},
        )
        self.assertFalse(torch.cuda.is_initialized())

    def test_factory_builds_separate_worker_and_refuses_full_mode(self):
        config = self.config()
        cache = self.cache_spec(config)
        worker = KVConnectorFactory.create_connector(
            config, KVConnectorRole.WORKER, cache
        )
        self.assertEqual(
            worker.registration_report(),
            {
                "role": "WORKER",
                "manager_bound": False,
                "callbacks": {},
            },
        )
        config.additional_config["kitepond"]["mode"] = "full"
        with self.assertRaises(UnsupportedKitepondProfile):
            KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER, cache)
        with self.assertRaises(UnsupportedKitepondProfile):
            self.make_scheduler(config, cache)

    def test_stock_worker_wrapper_uses_versioned_lifecycle(self):
        config = self.config()
        facts = inspect_engine()
        if facts["worker_extension_init_callback"]:
            self.assertTrue(facts["worker_extension_destroy_callback"])
            config.parallel_config.worker_cls = f"{__name__}.CpuStockExtensionWorker"
            config.parallel_config.worker_extension_cls = (
                f"{INTEGRATION}.worker_hooks.KitepondWorkerExtension"
            )
        else:
            config.parallel_config.worker_cls = f"{__name__}.CpuLegacyBridge"
        wrapper = WorkerWrapperBase(rpc_rank=0)
        wrapper.init_worker([{"vllm_config": config}])
        self.assertEqual(wrapper.kitepond_worker_registration(), "not_initialized")
        wrapper.init_device()
        self.assertEqual(wrapper.worker.events, ["device"])
        self.assertEqual(wrapper.kitepond_worker_registration(), "initialized")
        wrapper.shutdown()
        self.assertEqual(wrapper.worker.events, ["device", "closed"])
        self.assertEqual(wrapper.kitepond_worker_registration(), "closed")

    def test_stock_deferred_finalize_does_not_emit_late_metadata(self):
        config = self.config()
        config.kv_transfer_config.kv_connector = "DeferredMarkerConnector"
        config.kv_transfer_config.kv_connector_module_path = __name__
        cache = self.cache_spec(config)
        scheduler = self.make_scheduler(config, cache)
        step = scheduler.schedule()
        with tempfile.TemporaryDirectory() as rank_dir:
            directory = Path(rank_dir)
            receiver = ReadyCollector(directory, {0: ReadyWorker.from_pid(os.getpid())})
            publisher = ReadyPublisher(directory)
            self.addCleanup(receiver.close)
            self.addCleanup(publisher.close)
            region = "b" * 64
            ticket = receiver.arm(
                ReadyAttempt("req", "transfer", 3, "a" * 64), {0: region}, 4
            )
            rendezvous = directory / "rank"
            with set_current_vllm_config(config):
                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    local_rank=0,
                    distributed_init_method=rendezvous.as_uri(),
                    backend="gloo",
                )
                try:
                    initialize_model_parallel(backend="gloo")
                    ensure_kv_transfer_initialized(config, cache)
                    active = ActiveKVConnector(config, {})
                    active.kv_connector.publisher = publisher
                    active.kv_connector.ticket = ticket
                    active.kv_connector.region = region
                    active.pre_forward(step)
                    active.finish_forward()
                    output = active.post_forward(set(), defer_finalize=True)
                    self.assertFalse(active.kv_connector.finalized)
                    self.assertIsNone(output.kv_connector_worker_meta)
                    self.assertIsNone(receiver.take(ticket))
                    active.finalize_step()
                    self.assertTrue(active.kv_connector.finalized)
                    (receipt,) = receiver.take(ticket)
                    self.assertEqual(receipt.attempt, ticket.attempt)
                    self.assertEqual(receipt.region_sha256, region)
                    self.assertIsNone(receiver.take(ticket))
                    self.assertIsInstance(
                        active.kv_connector.build_connector_worker_meta(),
                        _DeferredMetadata,
                    )
                    self.assertIsNone(output.kv_connector_worker_meta)
                    self.assertFalse(active.kv_connector.has_connector_metadata())
                finally:
                    ensure_kv_transfer_shutdown()
                    destroy_model_parallel()
                    destroy_distributed_environment()

    def test_additive_state_spec_uses_native_hybrid_allocator(self):
        config = self.config()
        config.cache_config.enable_prefix_caching = True
        config.cache_config.mamba_cache_mode = "align"
        cache = self.cache_spec(config)
        base_spec = MambaSpec(
            block_size=16,
            shapes=((8,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        )
        KVCacheSpecRegistry.register(
            ProbeStateSpec, ProbeStateManager, uniform_type_base_spec=MambaSpec
        )
        spec = ProbeStateSpec(
            block_size=16,
            shapes=((8,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        )
        self.assertIs(KVCacheSpecRegistry.get_manager_class(base_spec), MambaManager)
        self.assertIs(KVCacheSpecRegistry.get_manager_class(spec), ProbeStateManager)
        cache.kv_cache_groups.append(KVCacheGroupSpec(["state"], spec))

        scheduler = self.make_scheduler(config, cache)
        pool = scheduler.kv_cache_manager.block_pool
        coordinator = scheduler.kv_cache_manager.coordinator
        self.assertIsInstance(coordinator, HybridKVCacheCoordinator)
        state_manager = coordinator.single_type_managers[1]
        self.assertIs(type(state_manager), ProbeStateManager)
        self.assertIs(state_manager.block_pool, pool)
        self.assertFalse(
            hasattr(scheduler.kv_cache_manager, "plan_native_mamba_snapshot")
        )

        request = Request(
            request_id="hybrid-addon",
            prompt_token_ids=[1, 8, 9, 3],
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=lambda _request: (),
        )
        free_before = pool.get_num_free_blocks()
        # Leave exactly two blocks, enough for the ordinary attention and state
        # groups, but not the reservation added by the external state manager.
        withheld = pool.get_new_blocks(free_before - 2)
        state_manager.test_hold_requests.add(request.request_id)
        scheduler.add_request(request)
        blocked = scheduler.schedule()
        self.assertFalse(blocked.num_scheduled_tokens)
        self.assertFalse(state_manager.test_holds)
        self.assertEqual(pool.get_num_free_blocks(), 2)

        pool.free_blocks(withheld)
        step = scheduler.schedule()
        self.assertEqual(step.num_scheduled_tokens, {request.request_id: 4})
        hold = state_manager.test_holds[request.request_id]
        self.assertEqual(hold.ref_cnt, 1)
        self.assertNotIn(hold, state_manager.req_to_blocks[request.request_id])
        self.assertEqual(pool.get_num_free_blocks(), free_before - 3)
        scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
        self.assertFalse(state_manager.test_holds)
        self.assertEqual(pool.get_num_free_blocks(), free_before)

    def test_native_hisparse_host_instances_have_distinct_authority(self):
        config_a, config_b = self.config(), self.config()
        cache = self.cache_spec(config_a)
        config_b.cache_config.num_gpu_blocks = cache.num_blocks
        cache.hisparse_host_num_blocks = 16
        cache.kv_cache_groups[0].role = KVCacheGroupRole.HISPARSE_SOURCE
        cache.kv_cache_groups[0].host_resident = True

        manager_a = self.make_scheduler(config_a, cache).kv_cache_manager
        manager_b = self.make_scheduler(config_b, cache).kv_cache_manager
        owner = bind_decode_host(manager_a, engine_id="D", boot_id="test-epoch")
        a = get_hisparse_coordinator(manager_a)
        self.assertIs(owner.pool, a.get_host_block_pool())
        self.assertNotEqual(owner.pool, manager_a.block_pool)
        with self.assertRaisesRegex(RuntimeError, "host pool provider"):
            bind_prefill_host(manager_b, engine_id="D", boot_id="test-epoch")
        b = get_hisparse_coordinator(manager_b)
        self.assertIs(type(a.host_manager), HiSparseSourceManager)
        self.assertIs(type(b.host_manager), HiSparseSourceManager)
        self.assertIsNot(a.host_manager.block_pool, b.host_manager.block_pool)
        self.assertEqual(a.host_manager.block_pool.get_num_free_blocks(), 15)
        self.assertFalse(hasattr(a.host_manager, "bind_borrowed_authority"))
        with self.assertRaisesRegex(AssertionError, "Conflicting manager registration"):
            KVCacheSpecRegistry.register_role_manager(
                KVCacheGroupRole.HISPARSE_SOURCE, ProbeStateManager
            )

        pool_a = get_hisparse_host_pool(config_a, cache)
        pool_b = get_hisparse_host_pool(config_b, cache)
        self.addCleanup(drop_hisparse_host_pool, config_a)
        self.addCleanup(drop_hisparse_host_pool, config_b)
        self.assertIs(get_hisparse_host_pool(config_a, cache), pool_a)
        self.assertIsNot(pool_a, pool_b)
        self.assertIsNone(pool_a.backing)
        self.assertFalse(hasattr(pool_a, "bind_authority_region"))

    def test_stock_gap_report_matches_actual_cache_manager(self):
        info = inspect_engine()
        config = self.config()
        scheduler = self.make_scheduler(config, self.cache_spec(config))
        self.assertTrue(info["connector_module_path"])
        self.assertIn("kv_cache_config", info["connector_init_args"])
        self.assertEqual(info["full_cache_mode"], "unavailable")
        self.assertFalse(info["post_fence_metadata_hook"])
        self.assertTrue(info["cache_spec_registration"])
        self.assertFalse(info["exact_snapshot_reservation"])
        self.assertFalse(info["shared_host_allocator_binding"])
        with self.assertRaises(AttributeError):
            _ = scheduler.kv_cache_manager.plan_native_mamba_snapshot
        self.assertFalse(info["shared_host_authority_binding"])
        self.assertFalse(info["engine_connector_control"])
        self.assertFalse(info["hisparse_dcp_provider_gate"])
        self.assertFalse(info["hisparse_dcp_independent_hybrid_pages"])
        self.assertFalse(info["native_synchronous_step_callback"])
        self.assertFalse(info["native_worker_startup_callback"])
        self.assertFalse(info["native_engine_idle_callback"])
        self.assertFalse(info["native_terminal_host_fence"])
        self.assertFalse(info["mooncake_named_control"])
        self.assertFalse(info["scheduler_owned_hisparse_warmup"])
        print("KITEPOND_STOCK_INTERFACE", json.dumps(info, sort_keys=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
