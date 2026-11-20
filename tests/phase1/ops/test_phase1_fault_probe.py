# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI
from kitepond.core.readiness import ReadyAttempt, ReadyReceipt
from kitepond.models.glm5next.frontier import digest
from kitepond.ops.wheel import sha256_file
from phase1_fault_probe import (
    SCHEMA,
    Pair,
    _check_confirm_ready,
    cancelled_tokens,
    old_proof,
    tool_step_one,
)

from vllm.entrypoints.scale_out.factories import register_scale_out_api_routers
from vllm.entrypoints.serve.tokenize.protocol import TokenizeChatRequest


class FaultProbeTest(unittest.IsolatedAsyncioTestCase):
    async def test_stock_abort_path_can_be_registered_but_disabled(self):
        for enabled in (False, True):
            app = FastAPI()
            app.state.args = SimpleNamespace(
                tokens_only=False, enable_scale_out=enabled
            )
            abort = AsyncMock()
            app.state.engine_client = SimpleNamespace(abort=abort)
            register_scale_out_api_routers(app, ("generate",))
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
            ) as client:
                pair = Pair.__new__(Pair)
                pair.client, pair.api_key = client, None
                if enabled:
                    self.assertEqual(await pair.abort("cmpl-test-cancel-0"), 200)
                    await asyncio.sleep(0)
                    abort.assert_awaited_once_with(["cmpl-test-cancel-0"])
                else:
                    with self.assertRaisesRegex(ValueError, "enable-scale-out"):
                        await pair.abort("cmpl-test-cancel-0")
                    abort.assert_not_awaited()

    def test_ready_confirmation_accepts_only_native_transport_choices(self):
        ready = {"status": "ready"}
        self.assertIsNone(_check_confirm_ready(ready))
        for transport in ("full", "same_pages"):
            self.assertEqual(
                _check_confirm_ready(ready | {"mla_transport": transport}),
                transport,
            )
        for reply in (
            ready | {"mla_transport": None},
            ready | {"mla_transport": "bypass"},
            ready | {"mla_transport": "full", "other": True},
            {"status": "pending", "mla_transport": "full"},
        ):
            with self.assertRaises(ValueError):
                _check_confirm_ready(reply)

    def test_abort_result_rejects_frontend_only_abort_or_reset(self):
        def snapshot(count, epoch):
            return {
                "metrics": {
                    "components": {
                        "decode": {
                            "epoch": epoch,
                            "metrics": {"generated_tokens_total": count},
                        }
                    }
                }
            }

        self.assertEqual(cancelled_tokens(snapshot(31, "d0"), snapshot(37, "d0")), 6)
        for row in (snapshot(31, "d0"), snapshot(2100, "d0"), snapshot(37, "d1")):
            with self.assertRaisesRegex(ValueError, "Native D"):
                cancelled_tokens(snapshot(31, "d0"), row)

    def test_old_wire_is_bound_to_exact_source_and_fresh_owner(self):
        prompt = [1, 2, 3]
        attempt = ReadyAttempt("cmpl-controlled-0", "transfer", 11, "a" * 64)
        batch = {
            "attempt": asdict(attempt),
            "mac": "b" * 64,
            "receipts": [
                asdict(ReadyReceipt(attempt, r, "c" * 64, 2, 2, 4)) for r in range(4)
            ],
        }
        artifact = {"wheel_sha256": "d" * 64}
        pair = SimpleNamespace(
            artifact=artifact, boots={"decode": "fresh"}, plan_sha="new-plan"
        )
        with tempfile.TemporaryDirectory() as temp:
            prior, wire = (Path(temp) / name for name in ("prior.json", "wire.json"))
            wire.write_text(json.dumps(batch))
            old = {
                "schema": SCHEMA,
                "case": "capture-old-ack",
                "artifact": artifact,
                "boots": {"decode": "old"},
                "plan_sha256": "old-plan",
                "operation": {
                    "api_id": "controlled",
                    "transfer_id": "transfer",
                    "epoch": 11,
                    "lease_sha256": "a" * 64,
                    "prompt_sha256": digest(prompt),
                    "old_ready_accepted_and_generated": True,
                    "ready_wire_sha256": sha256_file(wire),
                },
            }
            prior.write_text(json.dumps(old))
            self.assertEqual(old_proof(pair, prompt, prior, wire)[2], batch)
            moved = SimpleNamespace(
                artifact=artifact, boots={"decode": "old"}, plan_sha="new-plan"
            )
            with self.assertRaisesRegex(ValueError, "prior old D boot"):
                old_proof(moved, prompt, prior, wire)
            wire.write_text(json.dumps(batch | {"mac": "e" * 64}))
            with self.assertRaisesRegex(ValueError, "exact signed batch"):
                old_proof(pair, prompt, prior, wire)

    def test_frozen_native_chat_tool_schema_preserves_real_call_id(self):
        fixture = json.loads(
            (
                Path(__file__).parent / "fixtures/phase1_tool_closed_loop.json"
            ).read_bytes()
        )
        req = fixture["request"]
        tokenized = TokenizeChatRequest.model_validate(
            {"model": "glm", "messages": req["messages"], "tools": req["tools"]}
        )
        self.assertEqual(tokenized.tools[0].function.name, "lookup_item")
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_native_fixture",
                    "function": {
                        "name": "lookup_item",
                        "arguments": '{"item":{"sku":"铜-甲"}}',
                    },
                }
            ],
        }
        assistant, stub = tool_step_one(
            {"finish_reason": "tool_calls", "message": message}, fixture["oracle"]
        )
        self.assertIs(assistant, message)
        self.assertEqual(stub["tool_call_id"], "call_native_fixture")
        self.assertEqual(json.loads(stub["content"])["unit_price"], 7)
        message["tool_calls"][0]["function"]["arguments"] = (
            '{"item":{},"item":{"sku":"铜-甲"}}'
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            tool_step_one(
                {"finish_reason": "tool_calls", "message": message}, fixture["oracle"]
            )
