# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conformance test for the native two-turn private HTTP driver."""

import asyncio
import json
from dataclasses import asdict
from time import monotonic_ns

import httpx
import pytest
from kitepond.core.readiness import ReadyReceipt
from kitepond.integrations.vllm.cold_prefill_http import CONTROL_PATH, WARM_CONTROL_PATH
from kitepond.integrations.vllm.glm_cold_prefill_admission import (
    ColdPrefillControl,
    ColdPrefillOffer,
    ColdPrefillPrompt,
)
from kitepond.integrations.vllm.host_authority import NativeHostGrant
from kitepond.integrations.vllm.terminal_checkpoint import _sha256
from kitepond.ops.glm_cold_probe import ProbeConfig, ProbeFailure
from kitepond.ops.glm_warm_probe import DecodeIdentity, run_warm_probe


@pytest.mark.parametrize("case", ["success", "wrong_terminal_token", "rank_aborted"])
def test_raw_session_keeps_native_frontier_and_settles(case):
    prompt = [11, 12, 13, 14, 15]
    suffix = [21, 22, 23, 24]
    outputs = [[16, 17], [25, 26]]
    transcript = prompt + outputs[0]
    boundary = len(transcript) - 1
    expected_prompt = transcript + suffix
    token = "b" * 64
    release = asyncio.Event()
    waiting = asyncio.Event()
    identity = None
    routes = {}
    trace = []
    p_turn = 0
    d_turn = 0
    scope = DecodeIdentity("D", "d-boot", "d" * 64, "c" * 64)
    config = ProbeConfig(
        8001,
        8002,
        "http://127.0.0.1:8101",
        "P",
        "fixture",
        output_tokens=2,
        capture_token_ids=True,
        seed=677,
    )
    signer = ColdPrefillControl(
        d_engine_id=scope.engine_id,
        d_boot_id=scope.boot_id,
        p_boot_id="p-boot",
        key=b"a" * 32,
    )

    def reply(data):
        return httpx.Response(200, json=data)

    def completion(body, tokens: list[int], output: list[int]):
        return reply(
            {
                "id": f"cmpl-{body['request_id']}",
                "choices": [{"finish_reason": "length", "token_ids": output}],
                "usage": {
                    "prompt_tokens": len(tokens),
                    "completion_tokens": len(output),
                    "total_tokens": len(tokens) + len(output),
                },
            }
        )

    def sealed(body):
        return {"body": body, "mac": "e" * 64}

    async def application(req):
        nonlocal identity, p_turn, d_turn
        is_p = req.url.port == config.prefill_port
        path = req.url.path
        role = "P" if is_p else "D"
        if path == "/metrics":
            assert is_p
            suffix_work = len(expected_prompt) - 1 - boundary if p_turn == 2 else 0
            retained = boundary if p_turn == 2 else 0
            rows = {
                "local_compute": len(prompt) - 1 + suffix_work,
                "local_cache_hit": 0,
                "external_kv_transfer": retained,
            }
            return httpx.Response(
                200,
                text="\n".join(
                    f'vllm:prompt_tokens_by_source_total{{source="{key}"}} {value}'
                    for key, value in rows.items()
                ),
            )
        body = json.loads(req.content)
        if path == "/v1/completions":
            trace.append((role, "completion"))
            params = body["kv_transfer_params"]
            native_id = f"cmpl-{body['request_id']}-0"
            offer = routes[native_id]
            tokens = prompt if (p_turn if is_p else d_turn) == 0 else expected_prompt
            assert body["prompt"] == tokens and body["seed"] == 677
            if is_p:
                assert params["do_remote_decode"] and body["max_tokens"] == 1
                if p_turn == 0:
                    assert (
                        signer.open_offer(
                            params["kitepond_cold_prefill_offer"], offer.prompt
                        )
                        == offer
                    )
                else:
                    assert params["kitepond_glm_warm_nonce"] == "f" * 64
                    assert "kitepond_cold_prefill_offer" not in params
                    waiting.set()
                    await release.wait()
                p_turn += 1
                return completion(body, tokens, [0])
            assert params["do_remote_prefill"] and body["return_token_ids"]
            assert identity is not None
            output = outputs[d_turn]
            d_turn += 1
            return completion(body, tokens, output)

        operation, payload = body["operation"], body["payload"]
        trace.append((role, operation))
        assert req.headers["authorization"] == f"Bearer {token}"
        if path == CONTROL_PATH:
            if operation == "reserve":
                assert not is_p
                native = ColdPrefillPrompt(**payload)
                grant = NativeHostGrant(
                    "D",
                    "d-boot",
                    native.request_id,
                    native.transfer_id,
                    native.epoch,
                    0,
                    native.tokens,
                    (),
                    (len(routes) + 10,),
                    None,
                )
                offer = ColdPrefillOffer(
                    native, grant, "p-boot", monotonic_ns() + 10**9
                )
                routes[native.request_id] = offer
                return reply({"status": "offered", "offer": signer.export_offer(offer)})
            offer = routes[
                payload["request_id"] if is_p else payload["attempt"]["request_id"]
            ]
            if operation == "read_ready":
                assert is_p
                receipts = [
                    ReadyReceipt(
                        offer.attempt,
                        rank,
                        f"{rank:064x}",
                        offer.imported_tokens,
                        offer.imported_tokens,
                        monotonic_ns(),
                    )
                    for rank in range(4)
                ]
                return reply(
                    {
                        "status": "ready",
                        "receipt_batch": {
                            "attempt": asdict(offer.attempt),
                            "receipts": [asdict(row) for row in receipts],
                            "mac": "a" * 64,
                        },
                    }
                )
            assert operation == "confirm_ready" and not is_p
            return reply({"status": "ready"})

        assert path == WARM_CONTROL_PATH
        if operation == "warm_arm":
            assert not is_p and not d_turn
            identity = payload["identity"]
            assert identity["epoch"] == next(iter(routes.values())).grant.epoch
            return reply({"armed": True})
        if operation == "warm_source":
            assert d_turn == 1 and payload["identity"] == identity
            return reply(
                {
                    "source": sealed(
                        {
                            "lease": {
                                "identity": identity,
                                "requested_common_tokens": boundary,
                                "visible_tokens": len(transcript),
                                "prefix_sha256": _sha256(transcript[:boundary]),
                                "next_token_id": 999
                                if case == "wrong_terminal_token"
                                else transcript[boundary],
                            },
                            "manifest": {
                                "ranks": [
                                    {"checkpoint": {"rank": rank}} for rank in range(4)
                                ]
                            },
                        }
                    )
                }
            )
        if operation == "warm_authorize":
            assert is_p and d_turn == 1
            assert payload["offer"]["body"]["prompt"]["epoch"] == identity["epoch"]
            assert payload["offer"]["body"]["prompt"]["tokens"] == len(expected_prompt)
            return reply({"nonce": "f" * 64})
        if operation == "warm_destination":
            assert is_p
            if not waiting.is_set():
                return reply({"destination": None})
            return reply({"destination": sealed({"request_id": payload["request_id"]})})
        if operation == "warm_prepare":
            assert not is_p and waiting.is_set() and not release.is_set()
            return reply(
                {
                    "transfer": sealed(
                        {
                            "destination": payload["destination"]["body"],
                        }
                    )
                }
            )
        if operation == "warm_submit":
            assert is_p and waiting.is_set() and not release.is_set()
            release.set()
            return reply({"submitted": True})
        if operation == "warm_poll":
            assert p_turn == d_turn == 2
            if is_p:
                return reply(
                    {
                        "fences": [
                            {"receipt": {"local_rank": rank, "status": "acked"}}
                            for rank in range(4)
                        ]
                    }
                )
            return reply(
                {
                    "received_ranks": [0, 1, 2, 3],
                    "aborted_ranks": [3] if case == "rank_aborted" else [],
                    "complete": True,
                }
            )
        assert operation == "warm_release" and p_turn == d_turn == 2
        return reply({"released": True})

    async def execute():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(application)
        ) as client:
            return await run_warm_probe(client, config, scope, prompt, suffix, token)

    if case != "success":
        with pytest.raises(
            ProbeFailure,
            match=(
                "terminal checkpoint differs"
                if case == "wrong_terminal_token"
                else "reverse worker settled as aborted"
            ),
        ):
            asyncio.run(execute())
        assert not any(operation == "warm_release" for _, operation in trace)
        if case == "wrong_terminal_token":
            assert p_turn == d_turn == 1
        return

    report = asyncio.run(execute())
    assert report["second_prompt_token_ids"] == expected_prompt
    assert report["first_turn"]["output_token_ids"] == outputs[0]
    assert report["second_turn"]["output_token_ids"] == outputs[1]
    assert report["settled_reverse_ranks"] == [0, 1, 2, 3]
    assert report["reused_prefix_tokens"] == boundary
    assert (
        report["prefill_suffix_computed_tokens"] == len(expected_prompt) - 1 - boundary
    )
    assert report["prefill_released"] and report["decode_released"]
    assert trace[-2:] == [("P", "warm_release"), ("D", "warm_release")]
