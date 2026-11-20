# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private, loopback-only ingress for the GLM cold prefill connector."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
from hmac import compare_digest

from kitepond.core.readiness import ReadyAttempt, ReadyReceipt
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse

from .control_transport import ConnectorControlTransport, ConnectorControlUnavailable
from .glm_cold_prefill_admission import ColdPrefillPrompt
from .glm_settlement import OPERATION as SETTLEMENT_OPERATION

CONTROL_NAMESPACE = "kitepond.glm-cold-prefill.v1"
CONTROL_PATH = "/internal/kitepond/glm-cold-prefill/v1"
WARM_CONTROL_PATH = "/internal/kitepond/glm-warm-return/v1"
_WARM_CONTROL_BYTES = 4 * 1024 * 1024
TOKEN_ENV = "KITEPOND_GLM_COLD_HTTP_TOKEN"
MAX_CONTROL_BYTES = 32 * 1024
CONTROL_TIMEOUT_SECONDS = 10
_OPERATIONS = frozenset(
    {"reserve", "read_ready", "confirm_ready", "status", SETTLEMENT_OPERATION}
)
_HEX = frozenset("0123456789abcdef")


def is_control_token(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def check_ready_batch(payload: dict) -> ReadyAttempt:
    """Check the wire shape, leaving authenticity to the owning connector."""
    if payload.keys() != {"attempt", "receipts", "mac"}:
        raise ValueError("Invalid READY batch")
    attempt = ReadyAttempt(**payload["attempt"])
    raw = payload["receipts"]
    if (
        not isinstance(raw, list)
        or len(raw) != 4
        or not is_control_token(payload["mac"])
    ):
        raise ValueError("Invalid READY batch")
    receipts = [
        ReadyReceipt(**{**item, "attempt": ReadyAttempt(**item["attempt"])})
        for item in raw
    ]
    if {receipt.rank for receipt in receipts} != {0, 1, 2, 3} or any(
        receipt.attempt != attempt for receipt in receipts
    ):
        raise ValueError("Invalid READY quorum")
    return attempt


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON field")
        value[key] = item
    return value


def _invalid_constant(_: str) -> None:
    raise ValueError("Non-finite JSON number")


def parse_control_envelope(raw: bytearray) -> tuple[str, dict]:
    envelope = json.loads(
        raw,
        object_pairs_hook=_json_object,
        parse_constant=_invalid_constant,
    )
    if not isinstance(envelope, dict) or envelope.keys() != {"operation", "payload"}:
        raise ValueError("Invalid control fields")
    operation, payload = envelope["operation"], envelope["payload"]
    if not isinstance(operation, str) or not isinstance(payload, dict):
        raise ValueError("Invalid control operation")
    return operation, payload


def _parse_warm_body(raw: bytearray) -> tuple[str, dict]:
    from .glm_warm_control import OPERATIONS

    operation, payload = parse_control_envelope(raw)
    if operation not in OPERATIONS:
        raise ValueError("Invalid warm control operation")
    return operation, payload


def _parse_body(raw: bytearray) -> tuple[str, dict]:
    operation, payload = parse_control_envelope(raw)
    if operation not in _OPERATIONS:
        raise ValueError("Invalid control operation")
    if operation == "reserve":
        ColdPrefillPrompt(**payload)
    elif operation == "read_ready":
        ReadyAttempt(**payload)
    elif operation == "confirm_ready":
        check_ready_batch(payload)
    elif payload:
        raise ValueError("Read-only control takes no request fields")
    return operation, payload


def _reject(code: int, error: str) -> JSONResponse:
    return JSONResponse({"error": error}, status_code=code)


def _is_direct_local(request: Request) -> bool:
    if not request.client or any(
        name in {"forwarded", "x-real-ip"} or name.startswith("x-forwarded-")
        for name in request.headers
    ):
        return False
    try:
        return ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        return False


async def cold_prefill_ingress(request: Request, call_next):
    """Load using vLLM's normal --middleware import path on private P/D instances."""
    if request.scope["path"] == WARM_CONTROL_PATH:
        from .glm_warm_control import NAMESPACE

        return await private_control_ingress(
            request,
            call_next,
            path=WARM_CONTROL_PATH,
            namespace=NAMESPACE,
            token_env=TOKEN_ENV,
            parse_body=_parse_warm_body,
            byte_limit=_WARM_CONTROL_BYTES,
        )
    return await private_control_ingress(
        request,
        call_next,
        path=CONTROL_PATH,
        namespace=CONTROL_NAMESPACE,
        token_env=TOKEN_ENV,
        parse_body=_parse_body,
    )


async def private_control_ingress(
    request: Request,
    call_next,
    *,
    path: str,
    namespace: str,
    token_env: str,
    parse_body,
    byte_limit: int = MAX_CONTROL_BYTES,
):
    """Share the narrow loopback-only HTTP transport for explicit control routes."""
    if request.scope["path"] != path:
        return await call_next(request)

    token = os.environ.get(token_env)
    if not is_control_token(token) or not _is_direct_local(request):
        return _reject(404, "not_found")
    if request.method != "POST":
        return JSONResponse(
            {"error": "method_not_allowed"}, status_code=405, headers={"Allow": "POST"}
        )
    if request.scope["query_string"]:
        return _reject(400, "invalid_control_request")

    authorization = request.headers.getlist("authorization")
    scheme, _, supplied = (
        authorization[0] if len(authorization) == 1 else ""
    ).partition(" ")
    if (
        scheme.lower() != "bearer"
        or not is_control_token(supplied)
        or not compare_digest(token, supplied)
    ):
        return _reject(401, "authentication_required")

    media = request.headers.getlist("content-type")
    if (
        len(media) != 1
        or media[0].partition(";")[0].strip().lower() != "application/json"
        or "content-encoding" in request.headers
    ):
        return _reject(415, "json_required")

    length = request.headers.getlist("content-length")
    if length:
        if (
            len(length) != 1
            or len(length[0]) > 20
            or not length[0].isascii()
            or not length[0].isdecimal()
        ):
            return _reject(400, "invalid_control_request")
        if int(length[0]) > byte_limit:
            return _reject(413, "control_request_too_large")

    try:
        async with asyncio.timeout(CONTROL_TIMEOUT_SECONDS):
            raw = bytearray()
            async for chunk in request.stream():
                if len(raw) + len(chunk) > byte_limit:
                    return _reject(413, "control_request_too_large")
                raw.extend(chunk)
            if length and int(length[0]) != len(raw):
                return _reject(400, "invalid_control_request")
            operation, payload = parse_body(raw)
    except TimeoutError:
        return _reject(408, "control_body_timeout")
    except (ClientDisconnect, KeyError, TypeError, ValueError, RecursionError):
        return _reject(400, "invalid_control_request")

    engine = getattr(request.app.state, "engine_client", None)
    try:
        control = ConnectorControlTransport(
            getattr(engine, "engine_core", None), namespace=namespace
        )
    except ConnectorControlUnavailable:
        return _reject(503, "connector_control_unavailable")
    try:
        async with asyncio.timeout(CONTROL_TIMEOUT_SECONDS):
            result = await control.call(operation, payload)
        if not isinstance(result, dict):
            return _reject(502, "invalid_connector_control_reply")
        return JSONResponse(result)
    except TimeoutError:
        return _reject(504, "connector_control_timeout")
    except Exception:
        # Native typed RPC can wrap a rejected operation in an exception containing
        # internal state. Keep the detailed failure in the engine log.
        return _reject(502, "connector_control_rejected")
