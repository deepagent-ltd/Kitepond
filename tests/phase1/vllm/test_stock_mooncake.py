# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the stock Mooncake connector with controlled CPU hardware boundaries."""

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from kitepond.integrations.vllm.connector import KitepondConnector, RegistrationMetadata
from kitepond.integrations.vllm.mooncake import mooncake_forward_config
from kitepond.integrations.vllm.profile import UnsupportedKitepondProfile

from vllm.config import CacheConfig, VllmConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus

TRANSFER_ID = "synthetic-forward-xfer"


_NATIVE_ENGINE = mooncake_connector.TransferEngine


class _CpuEngineBoundary:
    """Observe host memory registration, using native TE when installed."""

    def __init__(self):
        self.native = _NATIVE_ENGINE() if _NATIVE_ENGINE is not None else None
        self.registered = {}

    def initialize(self, hostname, handshake, protocol, device_name):
        if self.native is None:
            return 0
        return self.native.initialize(hostname, handshake, protocol, device_name)

    def get_rpc_port(self):
        if self.native is None:
            return 1234
        return self.native.get_rpc_port()

    def batch_register_memory(self, pointers, capacities):
        result = (
            0
            if self.native is None
            else self.native.batch_register_memory(pointers, capacities)
        )
        if result == 0:
            self.registered.update(zip(pointers, capacities, strict=True))
        return result

    def batch_unregister_memory(self, pointers):
        result = (
            0 if self.native is None else self.native.batch_unregister_memory(pointers)
        )
        if result == 0:
            for pointer in pointers:
                del self.registered[pointer]
        return result


def _config(role):
    config = VllmConfig(
        cache_config=CacheConfig(block_size=16, enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            max_model_len=32,
            max_num_batched_tokens=32,
            max_num_seqs=2,
            is_encoder_decoder=False,
        ),
        kv_transfer_config=mooncake_forward_config(
            role,
            engine_id=f"fixture-{role}",
            mooncake_options={
                "mooncake_protocol": "tcp",
                "bootstrap_server_address": "127.0.0.1:1",
            },
        ),
        additional_config={"kitepond": {"mode": "registration-probe"}},
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
    return config, cache


def _request(request_id, **route):
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(16)),
        sampling_params=SamplingParams(
            max_tokens=1, extra_args={"kv_transfer_params": route}
        ),
        pooling_params=None,
    )


class StockMooncakeTest(unittest.TestCase):
    def connector(self, config, cache, role):
        connector = KVConnectorFactory.create_connector(config, role, cache)
        self.assertIs(type(connector), MultiConnector)
        kitepond, native = connector.sub_connectors
        self.assertIs(type(native), mooncake_connector.MooncakeConnector)
        self.assertIs(type(kitepond), KitepondConnector)
        self.assertIs(kitepond.role, role)
        return connector, native

    def worker(self, config, cache):
        config.model_config = SimpleNamespace(
            use_mla=False, get_total_num_kv_heads=lambda: 1
        )
        with (
            patch.object(torch.accelerator, "current_device_index", return_value=0),
            patch.object(mooncake_connector.current_platform, "set_device"),
            patch.multiple(
                mooncake_connector,
                TransferEngine=_CpuEngineBoundary,
                get_ip=lambda: "127.0.0.1",
                get_tensor_model_parallel_rank=lambda: 0,
                get_tensor_model_parallel_world_size=lambda: 1,
                get_pp_group=lambda: SimpleNamespace(rank_in_group=0),
                get_current_attn_backends=lambda config: [],
            ),
        ):
            connector, native = self.connector(config, cache, KVConnectorRole.WORKER)
        worker = native.connector_worker
        self.assertIs(type(worker), mooncake_connector.MooncakeConnectorWorker)
        self.addCleanup(self.stop_fixture, worker)
        self.assertIsNone(native.connector_scheduler)
        if _NATIVE_ENGINE is not None:
            self.assertIs(type(worker.engine.native), _NATIVE_ENGINE)
        return connector, native, worker

    @staticmethod
    def stop_fixture(worker):
        worker.shutdown()
        loop = worker.receiver_loop if worker.is_kv_consumer else worker.sender_loop
        loop.close()
        worker.engine.native = None

    def test_wrong_profile_is_rejected_before_native_worker_initialization(self):
        config, cache = _config("kv_consumer")
        config.additional_config = {}
        with (
            patch.object(mooncake_connector, "MooncakeConnectorWorker") as worker,
            self.assertRaises(UnsupportedKitepondProfile),
        ):
            KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER, cache)
        worker.assert_not_called()

    def test_producer_metadata_and_expiry_never_certify_delivery(self):
        config, cache = _config("kv_producer")
        scheduler, native = self.connector(config, cache, KVConnectorRole.SCHEDULER)
        self.assertTrue(native.connector_scheduler.is_kv_producer)
        self.assertIsNone(native.connector_worker)

        request = _request("p-fixture", do_remote_decode=True, transfer_id=TRANSFER_ID)
        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (0, False))
        scheduler.update_state_after_alloc(
            request, KVCacheBlocks(([KVCacheBlock(2)],)), 0
        )
        initial = scheduler.build_connector_meta(SchedulerOutput.make_empty())
        self.assertIs(type(initial.metadata[0]), RegistrationMetadata)

        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        self.assertEqual(
            scheduler.request_finished_all_groups(request, ([2],)),
            (True, None),
        )
        final = scheduler.build_connector_meta(SchedulerOutput.make_empty())
        self.assertEqual(
            final.metadata[1].reqs_to_send, {"p-fixture": (TRANSFER_ID, [[2]])}
        )

        composite, native, worker = self.worker(config, cache)
        self.assertIs(type(native).shutdown, KVConnectorBase_V1.shutdown)

        for metadata in (initial, final):
            composite.bind_connector_metadata(metadata)
            self.assertIs(native._get_connector_metadata(), metadata.metadata[1])
            composite.start_load_kv(None)
            composite.clear_connector_metadata()

        async def expire_unsent():
            send = worker.reqs_need_send[TRANSFER_ID]
            self.assertEqual(send.local_block_ids, [[2]])
            self.assertTrue(send.ready.is_set())
            send.expire_time = time.perf_counter() - 1

        asyncio.run_coroutine_threadsafe(expire_unsent(), worker.sender_loop).result(3)
        expired = composite.get_transfer_results(set())
        self.assertEqual(expired.finished_sending, {"p-fixture"})
        self.assertEqual(expired.finished_recving, set())
        self.assertEqual(worker.reqs_need_send, {})

    def test_consumer_routes_native_failure_and_unregisters_host_fixture(self):
        config, cache = _config("kv_consumer")
        scheduler, native = self.connector(config, cache, KVConnectorRole.SCHEDULER)
        self.assertTrue(native.connector_scheduler.is_kv_consumer)
        request = _request(
            "d-fixture",
            do_remote_prefill=True,
            remote_engine_id="p-fixture",
            remote_bootstrap_addr="127.0.0.1:1",
            transfer_id=TRANSFER_ID,
        )
        matched, asynchronous = scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual((matched, asynchronous), (16, True))
        scheduler.update_state_after_alloc(
            request, KVCacheBlocks(([KVCacheBlock(3)],)), matched
        )
        self.assertFalse(request.kv_transfer_params["do_remote_prefill"])
        metadata = scheduler.build_connector_meta(SchedulerOutput.make_empty())
        pull = metadata.metadata[1].reqs_to_recv["p-fixture"]["d-fixture"]
        self.assertEqual((pull.transfer_id, pull.local_block_ids), (TRANSFER_ID, [[3]]))

        composite, native, worker = self.worker(config, cache)
        tensor = torch.empty(
            8 * cache.kv_cache_groups[0].kv_cache_spec.page_size_bytes,
            dtype=torch.uint8,
        )
        composite.register_kv_caches({"model.layers.0": tensor})
        self.assertEqual(worker.engine.registered, {tensor.data_ptr(): tensor.numel()})
        composite.unregister_kv_caches()
        self.assertEqual(worker.engine.registered, {})
        composite.restore_kv_caches()
        self.assertEqual(worker.engine.registered, {tensor.data_ptr(): tensor.numel()})
        completed = threading.Event()

        async def fail_bootstrap(engine_id, pull_metas):
            worker._handle_failed_recv(pull_metas, tuple(pull_metas), "CPU fixture")
            completed.set()

        with patch.object(worker, "handle_new_engine_id", fail_bootstrap):
            composite.bind_connector_metadata(metadata)
            self.assertIs(native._get_connector_metadata(), metadata.metadata[1])
            composite.start_load_kv(None)
            self.assertTrue(completed.wait(timeout=3))
            composite.clear_connector_metadata()
        result = composite.get_transfer_results(set())
        self.assertEqual(result.finished_recving, {"d-fixture"})
        self.assertEqual(composite.get_block_ids_with_load_errors(), {3})
        composite.unregister_kv_caches()
        self.assertEqual(worker.engine.registered, {})
