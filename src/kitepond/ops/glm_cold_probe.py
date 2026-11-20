# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send one raw-token request through the private single-host 4P4D path."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import math
import os
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic, time_ns
from typing import ClassVar
from urllib.parse import urlsplit

import httpx

from kitepond.integrations.vllm.cold_prefill_http import (
    CONTROL_PATH,
    TOKEN_ENV,
    check_ready_batch,
    is_control_token,
)
from kitepond.integrations.vllm.glm_cold_prefill_admission import (
    ColdPrefillOffer,
    ColdPrefillPrompt,
    _identifier,
)
from kitepond.integrations.vllm.host_authority import NativeHostGrant

_API_KEY_ENV = "KITEPOND_GLM_PROBE_VLLM_API_KEY"
_MAX_PROMPT_TOKENS = 8192
_MAX_PROMPT_FILE_BYTES = 128 * 1024


@dataclass(frozen=True)
class ProbeConfig:
    max_prompt_tokens: ClassVar[int] = _MAX_PROMPT_TOKENS
    max_output_tokens: ClassVar[int] = 128
    max_timeout_seconds: ClassVar[float] = 120

    prefill_port: int
    decode_port: int
    prefill_bootstrap: str
    prefill_engine_id: str
    model: str
    output_tokens: int = 16
    timeout_seconds: float = 30
    capture_token_ids: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if (
            any(
                type(port) is not int or not 1 <= port <= 65535
                for port in (self.prefill_port, self.decode_port)
            )
            or self.prefill_port == self.decode_port
        ):
            raise ValueError("P and D need distinct local API ports")
        route = urlsplit(self.prefill_bootstrap)
        if (
            route.scheme != "http"
            or route.port is None
            or route.port < 1
            or route.username is not None
            or route.password is not None
            or route.path
            or route.query
            or route.fragment
            or not route.hostname
        ):
            raise ValueError("The P bootstrap must be an HTTP address with a port")
        ip = ipaddress.ip_address(route.hostname)
        if not (ip.is_loopback or ip.is_private) or ip.is_unspecified:
            raise ValueError("The P bootstrap must be a local numeric address")
        if not _identifier(self.prefill_engine_id) or not self.model:
            raise ValueError("A P engine ID and served model are required")
        if (
            type(self.output_tokens) is not int
            or not 1 <= self.output_tokens <= self.max_output_tokens
        ):
            raise ValueError(
                f"The one-request probe allows 1 to {self.max_output_tokens} output tokens"
            )
        if (
            type(self.timeout_seconds) not in (float, int)
            or not math.isfinite(self.timeout_seconds)
            or not 0.1 <= self.timeout_seconds <= self.max_timeout_seconds
        ):
            raise ValueError(
                f"The probe deadline must be between 0.1 and {self.max_timeout_seconds} seconds"
            )
        if type(self.capture_token_ids) is not bool:
            raise ValueError("Token capture must be explicitly enabled or disabled")
        if self.seed is not None and (
            type(self.seed) is not int or not -(2**63) <= self.seed < 2**63
        ):
            raise ValueError("The probe seed must be a signed 64-bit integer")


PrefillProbeHook = Callable[
    [ColdPrefillPrompt, ColdPrefillOffer, dict, dict], Awaitable[dict]
]
DecodeProbeHook = Callable[[ColdPrefillPrompt, ColdPrefillOffer], Awaitable[None]]


class ProbeFailure(RuntimeError):
    """A failed test retains the attempted identity for engine log correlation."""

    def __init__(self, phase: str, request_id: str, transfer_id: str, reason: str):
        self.phase = phase
        self.request_id = request_id
        self.transfer_id = transfer_id
        self.reason = reason
        super().__init__(f"{phase}: {reason}")

    def report(self) -> dict:
        return {
            "status": "failed",
            "phase": self.phase,
            "request_id": self.request_id,
            "transfer_id": self.transfer_id,
            "reason": self.reason,
        }


async def _post(client, port: int, path: str, body: dict, token: str | None) -> dict:
    response = await client.post(
        f"http://127.0.0.1:{port}{path}",
        json=body,
        headers={"Authorization": f"Bearer {token}"} if token else None,
        follow_redirects=False,
    )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("Invalid response object")
    return result


async def _control(client, port: int, operation: str, payload: dict, token: str):
    return await _post(
        client, port, CONTROL_PATH, {"operation": operation, "payload": payload}, token
    )


def _open_offer(response: dict, prompt: ColdPrefillPrompt) -> ColdPrefillOffer:
    if response.get("status") != "offered":
        raise ValueError("No native D grant")
    offer = response["offer"]
    if offer.keys() != {"body", "mac"} or not is_control_token(offer["mac"]):
        raise ValueError("Invalid native D grant envelope")
    body = offer["body"]
    if body.keys() != {"prompt", "grant", "prefill_boot_id", "arm_deadline_ns"}:
        raise ValueError("Invalid native D grant body")
    if ColdPrefillPrompt(**body["prompt"]) != prompt:
        raise ValueError("D returned a different prompt identity")
    grant = dict(body["grant"])
    if "nonce" not in grant:
        raise ValueError("D returned no lease nonce")
    grant["readonly_pages"] = tuple(grant["readonly_pages"])
    grant["writable_pages"] = tuple(grant["writable_pages"])
    # P validates the MAC with the native boot-scoped signer. The test driver
    # only checks shape and binds subsequent reads and writes to the same lease.
    return ColdPrefillOffer(
        prompt,
        NativeHostGrant(**grant),
        body["prefill_boot_id"],
        body["arm_deadline_ns"],
    )


def _check_completion(
    response: dict, api_id: str, input_tokens: set[int], output_tokens: int
) -> int:
    if response.get("id") != f"cmpl-{api_id}":
        raise ValueError("The native API returned a different completion ID")
    choices = response["choices"]
    usage = response["usage"]
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or choices[0]["finish_reason"] != "length"
        or type(prompt_tokens) is not int
        or prompt_tokens not in input_tokens
        or type(completion_tokens) is not int
        or completion_tokens != output_tokens
        or usage["total_tokens"] != prompt_tokens + completion_tokens
    ):
        raise ValueError("Native completion usage or finish reason differs")
    return prompt_tokens


def _decode_route(config: ProbeConfig, p_response: dict, transfer_id: str) -> dict:
    route = {
        "transfer_id": transfer_id,
        "remote_engine_id": config.prefill_engine_id,
        "remote_bootstrap_addr": config.prefill_bootstrap,
    }
    native = p_response.get("kv_transfer_params")
    if native is not None and (
        not isinstance(native, dict)
        or any(
            field in native and native[field] != value for field, value in route.items()
        )
    ):
        raise ValueError("P returned a different native transfer route")
    return {"do_remote_prefill": True, **route}


async def run_probe(
    client: httpx.AsyncClient,
    config: ProbeConfig,
    prompt_tokens: list[int],
    token: str,
    api_key: str | None = None,
    *,
    session_epoch: int | None = None,
    prefill_hook: PrefillProbeHook | None = None,
    before_decode: DecodeProbeHook | None = None,
) -> dict:
    """Use only received grants and READY receipts. Mutating calls run once."""
    if not is_control_token(token):
        raise ValueError("A private 64-digit hexadecimal control token is required")
    if (
        not isinstance(prompt_tokens, list)
        or not 2 <= len(prompt_tokens) <= config.max_prompt_tokens
    ):
        raise ValueError(
            f"Expected one raw prompt with 2 to {config.max_prompt_tokens} token IDs"
        )
    api_id = f"kp-{secrets.token_hex(8)}"
    transfer_id = f"kp-xfer-{secrets.token_hex(8)}"
    prompt = ColdPrefillPrompt.from_tokens(
        f"cmpl-{api_id}-0",
        transfer_id,
        time_ns() if session_epoch is None else session_epoch,
        prompt_tokens,
    )
    completion = {
        "model": config.model,
        "request_id": api_id,
        "prompt": prompt_tokens,
        "temperature": 0,
        "stream": False,
        "n": 1,
        "ignore_eos": True,
    }
    if config.seed is not None:
        completion["seed"] = config.seed
    started = monotonic()
    output_ids = None
    mla_transport = None
    phase = "D reserve"

    try:
        async with asyncio.timeout(config.timeout_seconds):
            offered = await _control(
                client, config.decode_port, "reserve", asdict(prompt), token
            )
            if offered.get("status") == "wait":
                raise ProbeFailure(
                    phase, prompt.request_id, transfer_id, "D capacity wait"
                )
            offer = _open_offer(offered, prompt)

            phase = "P completion"
            p_body = {
                **completion,
                "max_tokens": 1,
                "kv_transfer_params": {
                    "do_remote_decode": True,
                    "transfer_id": transfer_id,
                    "kitepond_cold_prefill_offer": offered["offer"],
                },
            }
            if prefill_hook is None:
                p_result = await _post(
                    client, config.prefill_port, "/v1/completions", p_body, api_key
                )
            else:
                p_result = await prefill_hook(prompt, offer, offered["offer"], p_body)
            p_usage = _check_completion(
                p_result, api_id, {prompt.tokens, offer.imported_tokens}, 1
            )
            decode_params = _decode_route(config, p_result, transfer_id)

            phase = "P READY"
            while True:
                ready = await _control(
                    client,
                    config.prefill_port,
                    "read_ready",
                    asdict(offer.attempt),
                    token,
                )
                if ready.get("status") != "wait":
                    break
                await asyncio.sleep(0.1)
            if ready.get("status") != "ready":
                raise ValueError("The native P owners did not publish READY")
            batch = ready["receipt_batch"]
            if check_ready_batch(batch) != offer.attempt or any(
                item["target_valid_tokens"] != offer.imported_tokens
                or item["draft_valid_tokens"] != offer.imported_tokens
                for item in batch["receipts"]
            ):
                raise ValueError("P READY is not the offered four-rank prompt boundary")

            phase = "D READY confirmation"
            confirmed = await _control(
                client, config.decode_port, "confirm_ready", batch, token
            )
            if confirmed.get("status") != "ready":
                raise ValueError("D did not authenticate the P READY quorum")
            mla_transport = confirmed.get("mla_transport")
            if mla_transport not in (None, "full", "same_pages"):
                raise ValueError("D returned an invalid MLA transport decision")

            if before_decode is not None:
                phase = "D pre-completion control"
                await before_decode(prompt, offer)

            phase = "D completion"
            d_result = await _post(
                client,
                config.decode_port,
                "/v1/completions",
                {
                    **completion,
                    "max_tokens": config.output_tokens,
                    "kv_transfer_params": decode_params,
                    "return_token_ids": config.capture_token_ids,
                },
                api_key,
            )
            _check_completion(d_result, api_id, {prompt.tokens}, config.output_tokens)
            if config.capture_token_ids:
                output_ids = d_result["choices"][0].get("token_ids")
                if (
                    not isinstance(output_ids, list)
                    or len(output_ids) != config.output_tokens
                    or any(
                        type(token_id) is not int or token_id < 0
                        for token_id in output_ids
                    )
                ):
                    raise ValueError("D did not return the requested token ID sequence")
    except httpx.HTTPStatusError as exc:
        raise ProbeFailure(
            phase, prompt.request_id, transfer_id, f"HTTP {exc.response.status_code}"
        ) from None
    except (TimeoutError, httpx.TimeoutException):
        raise ProbeFailure(
            phase, prompt.request_id, transfer_id, "deadline expired"
        ) from None
    except httpx.HTTPError:
        raise ProbeFailure(
            phase, prompt.request_id, transfer_id, "local HTTP failed"
        ) from None
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        raise ProbeFailure(
            phase, prompt.request_id, transfer_id, "invalid phase response"
        ) from None

    return {
        "status": "completed",
        "request_id": prompt.request_id,
        "transfer_id": transfer_id,
        "prompt_tokens": prompt.tokens,
        "prefill_api_prompt_tokens": p_usage,
        "verified_ready_ranks": 4,
        "ready_tokens_per_rank": offer.imported_tokens,
        "output_tokens": config.output_tokens,
        "probe_wall_seconds": round(monotonic() - started, 3),
        **({"output_token_ids": output_ids} if output_ids is not None else {}),
        **({"mla_transport": mla_transport} if mla_transport is not None else {}),
    }


def _read_prompt(path: Path) -> list[int]:
    with path.open("rb") as handle:
        raw = handle.read(_MAX_PROMPT_FILE_BYTES + 1)
    if len(raw) > _MAX_PROMPT_FILE_BYTES:
        raise ValueError("The raw token file is too large")
    return json.loads(raw)


async def _run_cli(args) -> dict:
    config = ProbeConfig(
        args.prefill_port,
        args.decode_port,
        args.prefill_bootstrap,
        args.prefill_engine_id,
        args.model,
        args.output_tokens,
        args.timeout,
        args.capture_token_ids,
        args.seed,
    )
    async with httpx.AsyncClient(trust_env=False, timeout=None) as client:
        return await run_probe(
            client,
            config,
            _read_prompt(args.prompt_file),
            os.environ.get(TOKEN_ENV, ""),
            os.environ.get(_API_KEY_ENV),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-port", required=True, type=int)
    parser.add_argument("--decode-port", required=True, type=int)
    parser.add_argument("--prefill-bootstrap", required=True)
    parser.add_argument("--prefill-engine-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--capture-token-ids", action="store_true")
    parser.add_argument("--seed", type=int)
    try:
        result = asyncio.run(_run_cli(parser.parse_args()))
    except ProbeFailure as exc:
        print(json.dumps(exc.report(), sort_keys=True))
        return 1
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
