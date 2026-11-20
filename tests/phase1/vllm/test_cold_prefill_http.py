# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for the pinned native API, request ID and typed Core wire."""

from dataclasses import asdict
from queue import SimpleQueue
from time import monotonic_ns
from types import SimpleNamespace

import httpx
import pytest
from kitepond.core.readiness import ReadyAttempt, ReadyReceipt
from kitepond.integrations.vllm.cold_prefill_http import (
    CONTROL_NAMESPACE,
    CONTROL_PATH,
    MAX_CONTROL_BYTES,
    TOKEN_ENV,
    WARM_CONTROL_PATH,
)
from kitepond.integrations.vllm.glm_cold_prefill_admission import (
    ColdPrefillAdmission,
    ColdPrefillControl,
    ColdPrefillPrompt,
)
from kitepond.integrations.vllm.glm_warm_control import NAMESPACE as WARM_NAMESPACE
from kitepond.integrations.vllm.host_authority import NativeHostAuthority
from kitepond.ops.glm_cold_probe import ProbeConfig, ProbeFailure, run_probe
from starlette.requests import Request

from vllm.entrypoints.common.cli_args import make_arg_parser
from vllm.entrypoints.launchers.app import build_app
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest, EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.core_client import AsyncMPClient, _process_utility_output
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

_TOKEN = "a" * 64
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}
_MIDDLEWARE = "kitepond.integrations.vllm.cold_prefill_http.cold_prefill_ingress"


class _LoopbackNativeWire(AsyncMPClient):
    """Use the pinned native request/result codec with an in-process CPU queue."""

    def __init__(self, handler, namespaces=(CONTROL_NAMESPACE, WARM_NAMESPACE)):
        self.core_engines = (b"core",)
        self.client_index = 0
        self.utility_results = {}
        self.encoder = MsgpackEncoder()
        self.calls = []

        self.core = EngineCoreProc.__new__(EngineCoreProc)
        self.core.shutdown_state = EngineShutdownState.RUNNING
        self.core.output_queue = SimpleQueue()
        connector = SimpleNamespace(
            get_control_handler=lambda name: handler if name in namespaces else None
        )
        self.core.scheduler = SimpleNamespace(get_kv_connector=lambda: connector)

    async def _send_input_message(self, message, engine):
        assert engine == b"core"
        assert message[0] == EngineCoreRequestType.UTILITY.value
        fields = MsgpackDecoder().decode(message[1:])
        assert fields[2] == "kv_connector_control"
        self.calls.append(fields[3][:2])
        self.core._handle_client_request(EngineCoreRequestType.UTILITY, fields)
        idx, outputs = self.core.output_queue.get_nowait()
        assert idx == self.client_index
        result = MsgpackDecoder(EngineCoreOutputs).decode(self.encoder.encode(outputs))
        _process_utility_output(result.utility_output, self.utility_results)

    def _ensure_output_queue_task(self):
        pass


@pytest.fixture
def app_factory(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, _TOKEN)
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(
        ["--host", "127.0.0.1", "--middleware", _MIDDLEWARE]
    )

    def make(handler):
        app = build_app(args, supported_tasks=())
        wire = _LoopbackNativeWire(handler)
        app.state.engine_client = SimpleNamespace(engine_core=wire)
        return app, wire

    return make


@pytest.fixture
def native_app_factory(app_factory):
    if not callable(
        getattr(AsyncMPClient, "kv_connector_control_async", None)
    ) or not callable(getattr(EngineCoreProc, "kv_connector_control", None)):
        pytest.skip("this stock build has no typed connector-control utility")
    return app_factory


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _reserve():
    payload = asdict(ColdPrefillPrompt.from_tokens("req", "xfer", 9, [31, 51]))
    return {"operation": "reserve", "payload": payload}


@pytest.mark.anyio
async def test_native_boundary_only_routes_pin_and_preserves_normal_fastapi(
    native_app_factory,
):
    app, wire = native_app_factory(
        lambda op, payload: {"routed": op, "token_ids": payload["tokens"]}
    )

    @app.post("/ordinary")
    async def ordinary():
        return {"unchanged": True}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(CONTROL_PATH, json=_reserve(), headers=_AUTH)
        assert response.json() == {"routed": "reserve", "token_ids": 2}
        assert wire.calls == [[CONTROL_NAMESPACE, "reserve"]]
        assert (await client.post("/ordinary")).json() == {"unchanged": True}


@pytest.mark.anyio
async def test_native_warm_control_is_bound_to_one_private_namespace(
    native_app_factory,
):
    app, wire = native_app_factory(lambda op, payload: {"routed": op, **payload})
    body = {"operation": "warm_destination", "payload": {"request_id": "turn-2"}}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        assert (await client.post(WARM_CONTROL_PATH, json=body)).status_code == 401
        assert (
            await client.post(CONTROL_PATH, json=body, headers=_AUTH)
        ).status_code == 400
        assert (
            await client.post(WARM_CONTROL_PATH, json=_reserve(), headers=_AUTH)
        ).status_code == 400
        response = await client.post(WARM_CONTROL_PATH, json=body, headers=_AUTH)
        assert response.json() == {"routed": "warm_destination", "request_id": "turn-2"}
        assert wire.calls == [[WARM_NAMESPACE, "warm_destination"]]


@pytest.mark.anyio
async def test_untrusted_or_malformed_calls_never_enter_native_core(
    app_factory, monkeypatch
):
    app, wire = app_factory(None)

    async def stream_large():
        for _ in range(2):
            yield b"x" * MAX_CONTROL_BYTES

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        assert (await client.post(CONTROL_PATH, json=_reserve())).status_code == 401
        assert (await client.get(CONTROL_PATH, headers=_AUTH)).status_code == 405
        wrong = {"Authorization": "Bearer " + "b" * 64}
        assert (
            await client.post(CONTROL_PATH, json=_reserve(), headers=wrong)
        ).status_code == 401
        spoofed = {**_AUTH, "X-Forwarded-For": "127.0.0.1"}
        assert (
            await client.post(CONTROL_PATH, json=_reserve(), headers=spoofed)
        ).status_code == 404
        assert (
            await client.post(
                CONTROL_PATH + "?namespace=other", json=_reserve(), headers=_AUTH
            )
        ).status_code == 400
        for body in (
            {**_reserve(), "namespace": "arbitrary"},
            {"operation": "call_utility_async", "payload": {}},
            {"operation": "confirm_ready", "payload": {}},
            {
                "operation": "reserve",
                "payload": {**_reserve()["payload"], "namespace": "other"},
            },
        ):
            assert (
                await client.post(CONTROL_PATH, json=body, headers=_AUTH)
            ).status_code == 400

        headers = {**_AUTH, "Content-Type": "application/json"}
        for raw in (
            b'{"operation":"reserve","operation":"read_ready","payload":{}}',
            b'{"operation":"reserve","payload":{"epoch":NaN}}',
        ):
            assert (
                await client.post(CONTROL_PATH, content=raw, headers=headers)
            ).status_code == 400
        assert (
            await client.post(
                CONTROL_PATH, content=b"z" * (MAX_CONTROL_BYTES + 1), headers=headers
            )
        ).status_code == 413
        assert (
            await client.post(CONTROL_PATH, content=stream_large(), headers=headers)
        ).status_code == 413
        assert (
            await client.post(CONTROL_PATH, content="{}", headers=_AUTH)
        ).status_code == 415
        monkeypatch.delenv(TOKEN_ENV)
        assert (
            await client.post(CONTROL_PATH, json=_reserve(), headers=_AUTH)
        ).status_code == 404

    monkeypatch.setenv(TOKEN_ENV, _TOKEN)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("192.0.2.1", 4000)),
        base_url="http://127.0.0.1",
    ) as client:
        assert (
            await client.post(CONTROL_PATH, json=_reserve(), headers=_AUTH)
        ).status_code == 404

    assert wire.calls == []


@pytest.mark.anyio
async def test_missing_native_capability_and_backend_failure_are_private(
    native_app_factory,
):
    def fail(op, fields):
        raise ValueError("private fixture details")

    app, wire = native_app_factory(fail)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(CONTROL_PATH, json=_reserve(), headers=_AUTH)
        assert response.status_code == 502
        assert response.json() == {"error": "connector_control_rejected"}
        app.state.engine_client = SimpleNamespace(engine_core=SimpleNamespace())
        unavailable = await client.post(CONTROL_PATH, json=_reserve(), headers=_AUTH)
        assert unavailable.status_code == 503
    assert len(wire.calls) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case", ["success", "success_8k", "wrong_route", "bad_mac", "wait", "timeout"]
)
async def test_one_request_script_using_native_api_and_test_only_signatures(
    native_app_factory, monkeypatch, case
):
    monkeypatch.setenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")
    route = "http://127.0.0.1:8330"
    signer = ColdPrefillControl(
        d_engine_id="D", d_boot_id="D-cpu", p_boot_id="P-cpu", key=bytes(range(32))
    )
    authority = NativeHostAuthority(
        BlockPool(num_gpu_blocks=10, enable_caching=False, hash_block_size=1024),
        engine_id="D",
        boot_id="D-cpu",
        block_tokens=1024,
    )
    admission = ColdPrefillAdmission(
        authority, signer, context_limit=8192, max_pending=1, arm_timeout_seconds=5
    )
    tokens = list(range(21, 21 + (8192 if case == "success_8k" else 19)))
    recorded = {}
    steps = []

    def d_control(op, payload):
        steps.append(("D", op))
        if op == "reserve":
            if case == "wait":
                return {"status": "wait", "reason": "RAM_ADMISSION_WAIT"}
            offer = admission.offer(ColdPrefillPrompt(**payload))
            recorded["offer"] = offer
            return {"status": "offered", "offer": signer.export_offer(offer)}
        assert op == "confirm_ready"
        receipts = tuple(
            ReadyReceipt(**{**item, "attempt": ReadyAttempt(**item["attempt"])})
            for item in payload["receipts"]
        )
        admission.verify_ready(recorded["offer"], receipts, payload["mac"])
        recorded["d_ready"] = True
        return {"status": "ready"}

    def p_control(op, payload):
        steps.append(("P", op))
        assert op == "read_ready"
        offer, receipts = recorded["p_state"]
        assert payload == asdict(offer.attempt)
        if case == "timeout" or steps.count(("P", op)) == 1:
            return {"status": "wait"}
        mac = signer.ready_mac(offer, receipts) if case != "bad_mac" else "b" * 64
        return {
            "status": "ready",
            "receipt_batch": {
                "attempt": payload,
                "receipts": [asdict(r) for r in receipts],
                "mac": mac,
            },
        }

    p_app, _ = native_app_factory(p_control)
    d_app, _ = native_app_factory(d_control)

    def receive(request, raw):
        parsed = CompletionRequest.model_validate(raw)
        assert parsed.prompt == tokens and parsed.n == 1
        assert not parsed.stream and parsed.ignore_eos
        assert request.headers["authorization"] == "Bearer api-fixture"
        base = OpenAIServingCompletion._base_request_id(request, parsed.request_id)
        core = EngineCoreRequest(
            f"cmpl-{base}-0",
            tokens,
            None,
            parsed.to_sampling_params(parsed.max_tokens),
            None,
            0,
            None,
            None,
            None,
        )
        InputProcessor.assign_request_id(core)
        assert core.request_id == recorded["offer"].prompt.request_id
        return parsed, core, core.sampling_params.extra_args["kv_transfer_params"]

    def completion(base, inputs, outputs, **extra):
        return {
            "id": f"cmpl-{base}",
            "choices": [{"finish_reason": "length"}],
            "usage": {
                "prompt_tokens": inputs,
                "completion_tokens": outputs,
                "total_tokens": inputs + outputs,
            },
            **extra,
        }

    @p_app.post("/v1/completions")
    async def p_complete(request: Request):
        steps.append(("P", "completion"))
        parsed, core, params = receive(request, await request.json())
        assert parsed.max_tokens == 1 and params["do_remote_decode"] is True
        prompt = ColdPrefillPrompt.from_tokens(
            core.request_id,
            params["transfer_id"],
            recorded["offer"].prompt.epoch,
            tokens,
        )
        offer = signer.open_offer(params["kitepond_cold_prefill_offer"], prompt)
        receipts = tuple(
            ReadyReceipt(
                offer.attempt,
                rank,
                f"{rank:064x}",
                offer.imported_tokens,
                offer.imported_tokens,
                monotonic_ns(),
            )
            for rank in range(4)
        )
        recorded["p_state"] = offer, receipts
        extra = (
            {"kv_transfer_params": {"remote_engine_id": "not-P"}}
            if case == "wrong_route"
            else {}
        )
        if case == "success_8k":
            extra = {
                "kv_transfer_params": {
                    "remote_engine_id": "P",
                    "remote_bootstrap_addr": route,
                    "transfer_id": params["transfer_id"],
                    "irrelevant": "do not forward",
                }
            }
        return completion(parsed.request_id, len(tokens) - 1, 1, **extra)

    @d_app.post("/v1/completions")
    async def d_complete(request: Request):
        steps.append(("D", "completion"))
        parsed, _, params = receive(request, await request.json())
        assert recorded["d_ready"]
        assert parsed.max_tokens == 4
        assert params == {
            "do_remote_prefill": True,
            "remote_bootstrap_addr": route,
            "remote_engine_id": "P",
            "transfer_id": recorded["offer"].prompt.transfer_id,
        }
        result = completion(parsed.request_id, len(tokens), 4)
        assert bool(parsed.return_token_ids) is (case == "success_8k")
        if case == "success_8k":
            assert parsed.seed == 677
            result["choices"][0]["token_ids"] = [51, 52, 53, 54]
        return result

    async def local_apps(scope, receive, send):
        host = dict(scope["headers"])[b"host"]
        await (p_app if host.endswith(b":8001") else d_app)(scope, receive, send)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=local_apps)
    ) as client:
        config = ProbeConfig(
            8001,
            8002,
            route,
            "P",
            "GLM-fixture",
            output_tokens=4,
            timeout_seconds=0.5 if case == "timeout" else 30,
            capture_token_ids=case == "success_8k",
            seed=677 if case == "success_8k" else None,
        )
        if case in {"success", "success_8k"}:
            report = await run_probe(client, config, tokens, _TOKEN, "api-fixture")
            assert report["status"] == "completed"
            assert (
                report["verified_ready_ranks"] == 4
                and report["ready_tokens_per_rank"] == len(tokens) - 1
            )
            assert (
                report["prompt_tokens"] == len(tokens) and report["output_tokens"] == 4
            )
            assert report["request_id"] == recorded["offer"].prompt.request_id
            if case == "success_8k":
                assert report["output_token_ids"] == [51, 52, 53, 54]
            else:
                assert "output_token_ids" not in report
            assert steps == [
                ("D", "reserve"),
                ("P", "completion"),
                ("P", "read_ready"),
                ("P", "read_ready"),
                ("D", "confirm_ready"),
                ("D", "completion"),
            ]
        else:
            with pytest.raises(ProbeFailure) as error:
                await run_probe(client, config, tokens, _TOKEN, "api-fixture")
            assert (
                error.value.phase
                == {
                    "bad_mac": "D READY confirmation",
                    "wrong_route": "P completion",
                    "wait": "D reserve",
                    "timeout": "P READY",
                }[case]
            )
            assert ("D", "completion") not in steps
            assert steps.count(("D", "reserve")) == 1


def test_request_randomization_default_unchanged(monkeypatch):
    monkeypatch.delenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", raising=False)
    native = EngineCoreRequest(
        "cmpl-default-0", [1, 2], None, None, None, 0, None, None, None
    )
    InputProcessor.assign_request_id(native)
    assert native.external_req_id == "cmpl-default-0"
    assert native.request_id.startswith("cmpl-default-0-")


@pytest.mark.anyio
async def test_nvme_http_uses_its_separate_token_namespace_and_native_core(
    native_app_factory, monkeypatch
):
    from kitepond.integrations.vllm.glm_nvme_control import NAMESPACE
    from kitepond.integrations.vllm.glm_nvme_http import CONTROL_PATH as NVME_PATH
    from kitepond.integrations.vllm.glm_nvme_http import TOKEN_ENV as NVME_TOKEN_ENV

    token = "b" * 64
    monkeypatch.setenv(NVME_TOKEN_ENV, token)
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(
        [
            "--host",
            "127.0.0.1",
            "--middleware",
            "kitepond.integrations.vllm.glm_nvme_http.nvme_ingress",
        ]
    )
    app = build_app(args, supported_tasks=())
    wire = _LoopbackNativeWire(lambda op, payload: {"op": op}, (NAMESPACE,))
    app.state.engine_client = SimpleNamespace(engine_core=wire)
    http_auth = {"Authorization": "Bearer " + token}
    payload = {"operation": "status", "payload": {"body": {}, "mac": "a" * 64}}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        assert (
            await client.post(NVME_PATH, headers=_AUTH, json=payload)
        ).status_code == 401
        assert (
            await client.post(NVME_PATH, headers=http_auth, json=_reserve())
        ).status_code == 400
        assert (
            await client.post(NVME_PATH, headers=http_auth, json=payload)
        ).json() == {"op": "status"}
    assert wire.calls == [[NAMESPACE, "status"]]
