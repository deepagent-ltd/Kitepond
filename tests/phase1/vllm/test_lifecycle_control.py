# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise read-only lifecycle controls on native CPU cache state."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, patch

from kitepond.core.policy import (
    ColdJobScheduler,
    IdlePolicy,
    LaneBudget,
    NativeModelColdAdapter,
    NativeModelColdJob,
)
from kitepond.integrations.vllm.connector import KitepondConnector
from kitepond.integrations.vllm.host_authority import NativeHostAuthority
from kitepond.integrations.vllm.lifecycle_control import (
    LIFECYCLE_NAMESPACE,
    LifecycleControl,
    LifecycleQueryClient,
)

from vllm.config import CacheConfig, KVTransferConfig, VllmConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import KVCacheConfig

IDENTITY = {"engine_id": "decode-1", "boot_id": "native-1"}
QUERY = {"schema": "kitepond.lifecycle-query.v1", **IDENTITY}


def _make_connector(role):
    config = VllmConfig(
        cache_config=CacheConfig(block_size=16, enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            max_model_len=32,
            max_num_batched_tokens=32,
            max_num_seqs=2,
            is_encoder_decoder=False,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="KitepondConnector",
            kv_connector_module_path="kitepond.integrations.vllm.connector",
            kv_role="kv_consumer",
            engine_id=IDENTITY["engine_id"],
        ),
        additional_config={"kitepond": {"mode": "registration-probe"}},
    )
    cache = KVCacheConfig(num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[])
    return KVConnectorFactory.create_connector(config, role, cache)


class _PendingRestore(NativeModelColdJob):
    def __init__(self):
        super().__init__("RESTORE", io_budget_bytes=128)
        self.polls = 0

    def poll(self):
        self.polls += 1
        return "PENDING"

    def cancel(self):
        raise AssertionError("The fixture has no cancellation")

    def reap(self):
        raise AssertionError("A pending restore cannot be reaped")


class _NativeColdFixture(NativeModelColdAdapter):
    def __init__(self):
        self.job = _PendingRestore()

    def last_business_use_ns(self, handle):
        raise AssertionError("Read-only queries should not access business time")

    def begin_restore(self, handle, *, max_inflight_bytes):
        return self.job

    def begin_store(self, handle, **kwargs):
        raise AssertionError("Read-only queries should not start a store")


class LifecycleControlTest(unittest.TestCase):
    def test_scheduler_factory_requires_explicit_control_and_same_engine_identity(self):
        connector = _make_connector(KVConnectorRole.SCHEDULER)
        worker = _make_connector(KVConnectorRole.WORKER)
        self.assertIs(type(connector), KitepondConnector)
        self.assertIsNone(connector.get_control_handler(LIFECYCLE_NAMESPACE))
        self.assertIsNone(worker.get_control_handler(LIFECYCLE_NAMESPACE))
        wrong = LifecycleControl(engine_id="other", boot_id=IDENTITY["boot_id"])
        with self.assertRaisesRegex(ValueError, "different connector"):
            connector.bind_lifecycle_control(wrong)
        owner = LifecycleControl(**IDENTITY)
        with self.assertRaisesRegex(RuntimeError, "scheduler"):
            worker.bind_lifecycle_control(owner)
        connector.bind_lifecycle_control(owner)
        connector.bind_lifecycle_control(owner)
        self.assertIsNone(connector.get_control_handler("undeclared"))
        with self.assertRaisesRegex(RuntimeError, "already bound"):
            connector.bind_lifecycle_control(LifecycleControl(**IDENTITY))
        report = connector.get_control_handler(LIFECYCLE_NAMESPACE)("health", QUERY)
        self.assertEqual(report["observations"]["ram"], {"status": "unavailable"})
        self.assertEqual(
            report["observations"]["cold"]["RESTORE"],
            {"status": "unavailable", "mode": "RESTORE"},
        )

    def test_native_ram_query_reads_pages_without_reserving_or_settling(self):
        pool = BlockPool(num_gpu_blocks=6, enable_caching=True, hash_block_size=1024)
        authority = NativeHostAuthority(pool, block_tokens=1024, **IDENTITY)
        grant = authority.reserve(
            "request",
            "transfer",
            1,
            prefix_tokens=0,
            reservation_tokens=1025,
            source=(),
        )
        authority.dispatch(grant)
        self.assertFalse(authority.cancel(grant))
        before = [block.ref_cnt for block in pool.blocks]
        owner = LifecycleControl(host=authority, **IDENTITY)
        with (
            patch.object(
                pool, "get_new_blocks", side_effect=AssertionError("allocate")
            ),
            patch.object(pool, "free_blocks", side_effect=AssertionError("release")),
        ):
            health = owner.handle("health", QUERY)["observations"]["ram"]
            estimate = owner.handle(
                "ram_admission", dict(QUERY, reservation_tokens=3073)
            )["observations"]["ram"]
        self.assertEqual(
            health,
            {
                "status": "observed",
                "block_tokens": 1024,
                "available_pages": 3,
                "active_grants": 1,
                "awaiting_settlement": 1,
                "grant_block_references": 2,
            },
        )
        self.assertEqual(
            (estimate["required_unshared_pages"], estimate["capacity"]), (4, "wait")
        )
        self.assertEqual([block.ref_cnt for block in pool.blocks], before)

    def test_cold_lane_observation_does_not_poll_start_or_reap_jobs(self):
        model = _NativeColdFixture()
        cold = ColdJobScheduler(
            model,
            idle=IdlePolicy(),
            foreground=LaneBudget(1, 128, 128),
            background=LaneBudget(1, 64, 64),
            max_jobs=3,
        )
        active = cold.submit("RESTORE", object())
        cold.tick(now_ns=1)
        queued = cold.submit("RESTORE", object())
        begin, pending = cold.view(active), cold.view(queued)
        owner = LifecycleControl(cold=cold, **IDENTITY)
        with (
            patch.object(
                model, "begin_restore", side_effect=AssertionError("started I/O")
            ),
            patch.object(model.job, "poll", side_effect=AssertionError("polled job")),
        ):
            report = owner.handle("cold_admission", dict(QUERY, mode="RESTORE"))[
                "observations"
            ]["cold"]
            other = owner.handle("cold_admission", dict(QUERY, mode="STORE"))[
                "observations"
            ]["cold"]
        self.assertEqual(
            (
                report["capacity"],
                report["active_jobs"],
                report["charged_bytes"],
                report["queued_jobs"],
            ),
            ("lane_full", 1, 128, 1),
        )
        self.assertEqual(other["capacity"], "available")
        self.assertEqual(model.job.polls, 1)
        self.assertEqual((cold.view(active), cold.view(queued)), (begin, pending))
        cold.submit("STORE", object())
        full = owner.handle("cold_admission", dict(QUERY, mode="STORE"))[
            "observations"
        ]["cold"]
        self.assertEqual(full["capacity"], "metadata_full")

    def test_closed_schema_rejects_mutations_stale_boots_and_foreign_threads(self):
        owner = LifecycleControl(**IDENTITY)
        for op, payload in (
            ("reserve", QUERY),
            ("health", dict(QUERY, boot_id="stale")),
            ("health", dict(QUERY, force=True)),
            ("health", dict(QUERY, schema="legacy")),
            ("ram_admission", dict(QUERY, reservation_tokens=True)),
            ("cold_admission", dict(QUERY, mode="UNKNOWN")),
        ):
            with self.subTest(op=op, payload=payload), self.assertRaises(ValueError):
                owner.handle(op, payload)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(owner.handle, "health", QUERY)
            with self.assertRaisesRegex(RuntimeError, "scheduler owner"):
                future.result(timeout=2)


class LifecycleClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_typed_dispatch_rejects_result_from_wrong_generation(self):
        owner = LifecycleControl(**IDENTITY)

        class Core:
            async def kv_connector_control_async(self, namespace, operation, payload):
                if namespace != LIFECYCLE_NAMESPACE:
                    raise AssertionError("wrong namespace")
                return owner.handle(operation, payload)

        core = Core()
        client = LifecycleQueryClient(core, **IDENTITY)
        self.assertEqual(
            (await client.ram_admission(1))["observations"]["ram"],
            {"status": "unavailable"},
        )
        with (
            patch.object(
                core,
                "kv_connector_control_async",
                new=AsyncMock(
                    return_value=owner.handle("health", QUERY) | {"boot_id": "stale"}
                ),
            ),
            self.assertRaisesRegex(ValueError, "owner identity"),
        ):
            await client.health()
