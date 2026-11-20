# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owner settlement remains scoped when worker event-loop evidence is absent."""

import os
import unittest
from threading import get_ident
from time import monotonic_ns
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from kitepond.integrations.vllm.cold_prefill_http import (
    CONTROL_NAMESPACE,
    CONTROL_PATH,
    TOKEN_ENV,
    cold_prefill_ingress,
)
from kitepond.integrations.vllm.glm_cold_prefill_scheduler import (
    GlmColdDecodeAdmissionMixin,
    GlmColdPrefillMixin,
)
from kitepond.integrations.vllm.glm_settlement import (
    OPERATION,
    observe_glm_owner_settlement,
)
from kitepond.integrations.vllm.host_authority import HostAuthorityView
from kitepond.ops.glm_settlement import _combine, _role_source, _validate

AUTH = "a" * 64


def model(role):
    recipe = {
        "p_engine_id": "local-p",
        "d_engine_id": "local-d",
        "p_boot_id": "pb",
        "d_boot_id": "db",
        "decode_host": "127.0.0.1",
    }
    profile = {
        "mooncake_api": recipe,
        "authority_engine_id": "local-d",
        "authority_boot_id": "db",
        "roles": {
            name: {"argv": ["--host", "127.0.0.1", "--port", port]}
            for name, port in (("prefill", "18163"), ("decode", "18162"))
        },
    }
    entry_thread = get_ident()

    def require_owner():
        if get_ident() != entry_thread:
            raise RuntimeError("wrong owner")

    owner = SimpleNamespace(
        stage=role,
        _engine_id="local-d",
        _boot_id="db",
        _require_scheduler_owner=require_owner,
        _pending={},
        _routed=None,
        _routed_prefill=None,
        authority=SimpleNamespace(observe=lambda: HostAuthorityView(256, 19, 0, 0, 0)),
    )
    queues = SimpleNamespace(
        _reqs_need_recv={}, _reqs_need_send={}, _reqs_not_processed=set()
    )
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(additional_config={"kitepond_glm_window": recipe}),
        connector=SimpleNamespace(
            sub_connectors=[owner, SimpleNamespace(connector_scheduler=queues)],
        ),
        _cold_prefill_owner=owner,
        _cold_decode_owner=owner,
        _cold_prefill_requests={},
        _cold_prefill_collector=SimpleNamespace(_pending={}),
        _cold_decode_offered={},
        _cold_decode_verified=set(),
        _cold_decode_staged={},
        _cold_decode_admission=SimpleNamespace(_aborts={}),
        _cold_mla_decisions={},
    )
    return scheduler, owner, profile


class SettlementTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_app_dispatch_for_both_owner_roles(self):
        for role, method in (
            ("prefill", GlmColdPrefillMixin._cold_prefill_handler),
            ("decode", GlmColdDecodeAdmissionMixin._cold_decode_handler),
        ):
            scheduler, _, profile = model(role)
            calls = []

            async def dispatch(
                namespace,
                operation,
                payload,
                calls=calls,
                handler=method,
                native=scheduler,
            ):
                self.assertEqual(namespace, CONTROL_NAMESPACE)
                calls.append(operation)
                return handler(native, operation, payload)

            app = FastAPI()
            app.middleware("http")(cold_prefill_ingress)
            app.state.engine_client = SimpleNamespace(
                engine_core=SimpleNamespace(kv_connector_control_async=dispatch)
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
            ) as client:
                auth = {"Authorization": f"Bearer {AUTH}"}
                body = {"operation": OPERATION, "payload": {}}
                with patch.dict(os.environ, {TOKEN_ENV: AUTH}):
                    self.assertEqual(
                        (await client.post(CONTROL_PATH, json=body)).status_code,
                        401,
                    )
                    self.assertEqual(
                        (
                            await client.post(
                                CONTROL_PATH,
                                json=body | {"payload": {"arbitrary": 1}},
                                headers=auth,
                            )
                        ).status_code,
                        400,
                    )
                    before = monotonic_ns()
                    reply = await client.post(CONTROL_PATH, json=body, headers=auth)
                    after = monotonic_ns()
                self.assertEqual(reply.status_code, 200)
                self.assertEqual(calls, [OPERATION])
                self.assertEqual(
                    _validate(
                        reply.json(), _role_source(profile, role)[1], before, after
                    ),
                    reply.json(),
                )
                self.assertEqual(
                    reply.json()["components"]["native_rank_transport"]["state"],
                    "unavailable",
                )

    def test_cold_jobs_real_refs_and_expired_records_are_observed_without_reaping(self):
        scheduler, owner, _ = model("decode")
        owner.authority.observe = lambda: HostAuthorityView(256, 19, 10, 0, 9)
        scheduler._cold_decode_offered["gone"] = SimpleNamespace(arm_deadline_ns=1)
        scheduler._cold_decode_admission._aborts["gone"] = {}
        scheduler._pending_host_writes = {"one"}
        scheduler._terminal_frames = {"one": None}
        scheduler._terminal_native_can_release = {"old": False, "new": True}
        scheduler._grant_completion_owner = owner
        scheduler._native_decode_proofs = {"id": {}}
        scheduler._settled_decode_imports = {"other-id": object()}
        expired = SimpleNamespace(deadline_monotonic_ns=1)
        active = SimpleNamespace(
            closing=True,
            permit=object(),
            refs=((object(), (1, 2, 3)),),
            lease=SimpleNamespace(identity=expired),
        )
        scheduler._terminal_checkpoints = SimpleNamespace(
            _active={"old": active},
            _armed={},
            _require_owner=owner._require_scheduler_owner,
        )
        old = SimpleNamespace(
            resident=SimpleNamespace(blocks=(1, 2)),
            readers={object()},
            io=None,
        )
        pending = SimpleNamespace(resident=None, readers=set(), io=object())
        scheduler._native_glm_cold_store = SimpleNamespace(
            _require_owner=owner._require_scheduler_owner,
            directory=SimpleNamespace(_entries={"old": old, "pending": pending}),
            scheduler=SimpleNamespace(
                observe_lane=lambda mode: SimpleNamespace(
                    active_jobs=int(mode == "STORE"),
                    queued_jobs=int(mode == "RESTORE"),
                    charged_bytes=2048 if mode == "STORE" else 0,
                ),
            ),
        )

        def op(stage, **fields):
            return (
                SimpleNamespace(
                    stage=stage,
                    body={"deadline_ns": 1},
                    job_id=None,
                    request=None,
                    cold_admission=None,
                    cancel=False,
                ).__dict__
                | fields
            )

        scheduler._native_glm_nvme_operator = SimpleNamespace(
            _require_owner=owner._require_scheduler_owner,
            _operations={
                "history": SimpleNamespace(**op("committed")),
                "live": SimpleNamespace(**op("reading", cold_admission=object())),
            },
        )
        got = observe_glm_owner_settlement(scheduler, owner)
        blocks = got["components"]
        self.assertEqual(
            blocks["host_authority"]["counts"]["grant_block_references"], 9
        )
        self.assertEqual(blocks["forward"]["counts"]["abort_quarantine"], 1)
        self.assertEqual(blocks["forward"]["counts"]["expired_grants"], 1)
        self.assertEqual(blocks["terminal"]["counts"]["block_references"], 3)
        self.assertEqual(blocks["terminal"]["counts"]["expired_holds"], 1)
        self.assertEqual(
            blocks["terminal"]["counts"]["connector_block_release_pending"], 1
        )
        self.assertEqual(
            blocks["native_decode_completion"]["counts"]["grant_proofs_pending"], 1
        )
        self.assertEqual(blocks["cold_ram"]["counts"]["resident_block_references"], 2)
        self.assertEqual(blocks["fsl2_io"]["counts"]["store_charged_bytes"], 2048)
        self.assertEqual(blocks["nvme_operations"]["counts"]["active"], 1)
        self.assertEqual(blocks["nvme_operations"]["counts"]["expired_active"], 1)
        self.assertIn("gone", scheduler._cold_decode_admission._aborts)
        self.assertEqual(
            scheduler._native_glm_cold_store.directory._entries["old"], old
        )
        self.assertIs(
            _combine({"decode": got}, {})["owner_reference_counts_zero"], False
        )
        self.assertIsNone(
            _combine({}, {"decode": "HTTPStatusError"})["owner_reference_counts_zero"]
        )

    def test_missing_worker_proof_leaves_global_zero_unknown(self):
        scheduler, owner, pair = model("decode")
        owner._pending.clear()
        reading = observe_glm_owner_settlement(scheduler, owner)
        paired = _combine({"decode": reading}, {})
        self.assertIsNone(paired["owner_reference_counts_zero"])
        self.assertIn("decode.native_rank_transport", paired["unavailable_components"])
        _, expected = _role_source(pair, "decode")
        when = reading["observed_monotonic_ns"]
        with self.assertRaisesRegex(ValueError, "identity"):
            _validate(reading | {"boot_id": "other"}, expected, when, when)

    async def test_wrong_thread_is_rejected_before_any_owner_reads(self):
        import asyncio

        scheduler, owner, _ = model("prefill")
        with self.assertRaisesRegex(RuntimeError, "wrong owner"):
            await asyncio.to_thread(observe_glm_owner_settlement, scheduler, owner)
