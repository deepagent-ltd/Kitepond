# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe one private native GLM session over cold P→D and warm D→P→D."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns, time_ns

import httpx
import regex as re

from kitepond.integrations.vllm.cold_prefill_http import (
    TOKEN_ENV,
    WARM_CONTROL_PATH,
    is_control_token,
)
from kitepond.integrations.vllm.glm_cold_prefill_admission import _identifier
from kitepond.integrations.vllm.terminal_checkpoint import _sha256
from kitepond.models.glm5next.fingerprint import config_fingerprint

from .glm_cold_probe import (
    _API_KEY_ENV,
    _MAX_PROMPT_TOKENS,
    ProbeConfig,
    ProbeFailure,
    _post,
    _read_prompt,
    run_probe,
)

_RANKS = [0, 1, 2, 3]
_SOURCES = {"local_compute", "local_cache_hit", "external_kv_transfer"}
_SOURCE_COUNTER = "vllm:prompt_tokens_by_source_total{"


@dataclass(frozen=True)
class DecodeIdentity:
    engine_id: str
    boot_id: str
    model_fingerprint: str
    config_fingerprint: str

    def __post_init__(self) -> None:
        if not (
            _identifier(self.engine_id)
            and _identifier(self.boot_id)
            and is_control_token(self.model_fingerprint)
            and is_control_token(self.config_fingerprint)
        ):
            raise ValueError("Native D boot and both model fingerprints are required")


async def _control(client, port, operation, payload, token):
    return await _post(
        client,
        port,
        WARM_CONTROL_PATH,
        {"operation": operation, "payload": payload},
        token,
    )


def _body(envelope):
    if (
        type(envelope) is not dict
        or envelope.keys() != {"body", "mac"}
        or type(envelope["body"]) is not dict
        or not is_control_token(envelope["mac"])
    ):
        raise ValueError("Warm owner returned an invalid signed envelope")
    return envelope["body"]


async def _await_envelope(control, port, operation, payload, key, *, request=None):
    while True:
        result = await control(port, operation, payload)
        if result.keys() != {key}:
            raise ValueError("Warm owner returned an unexpected reply")
        envelope = result[key]
        if envelope is not None:
            _body(envelope)
            return envelope
        if request is not None and request.done():
            await request
            raise ValueError("P completed before its warm destination was admitted")
        await asyncio.sleep(0.1)


async def _prefill_counters(client, port):
    response = await client.get(
        f"http://127.0.0.1:{port}/metrics", follow_redirects=False
    )
    response.raise_for_status()
    sources = dict.fromkeys(_SOURCES, 0)
    observed = set()
    for row in response.text.splitlines():
        if not row.startswith(_SOURCE_COUNTER):
            continue
        sample, value = row.rsplit(maxsplit=1)
        label = re.search(r'(?:\{|,)source="([^"]+)"', sample)
        if label is not None and label[1] in sources:
            name = label[1]
            tokens = float(value)
            if not tokens.is_integer() or tokens < 0:
                raise ValueError("Prompt-source counter returned invalid tokens")
            observed.add(name)
            sources[name] += int(tokens)
    if observed != _SOURCES:
        raise ValueError("Native prompt-source counters are unavailable")
    return sources


def _accepted_prefix(source, identity, transcript):
    body = _body(source)
    lease = body["lease"]
    manifest = body["manifest"]
    common = lease["requested_common_tokens"]
    if (
        lease["identity"] != identity
        or lease["visible_tokens"] != len(transcript)
        or type(common) is not int
        or not 0 < common < len(transcript)
        or lease["prefix_sha256"] != _sha256(transcript[:common])
        or lease["next_token_id"] != transcript[common]
        or [rank["checkpoint"]["rank"] for rank in manifest["ranks"]] != _RANKS
    ):
        raise ValueError("D terminal checkpoint differs from the accepted transcript")
    return common


async def _finish_return(control, config, transfer):
    command = {"transfer": transfer}
    p = await control(config.prefill_port, "warm_poll", command)
    fences = p["fences"]
    if sorted(fence["receipt"]["local_rank"] for fence in fences) != _RANKS or any(
        fence["receipt"]["status"] != "acked" for fence in fences
    ):
        raise ValueError("P did not record all four reverse readback fences")
    while True:
        d = await control(config.decode_port, "warm_poll", command)
        if d["aborted_ranks"]:
            raise ValueError("A D reverse worker settled as aborted")
        if d["complete"] is True:
            if d["received_ranks"] != _RANKS:
                raise ValueError("D reverse settlement lost a rank")
            return d["received_ranks"]
        await asyncio.sleep(0.1)


async def _release_return(control, config, transfer):
    command = {"transfer": transfer}
    # A P finished-sending notification is asynchronous to the D completion.
    # Retry only the owner's explicit rejection while pending, never a lost reply.
    for delay in (0.1, 0.1, 0.2, 0.3, 0.5, None):
        try:
            p = await control(config.prefill_port, "warm_release", command)
            if p != {"released": True}:
                raise ValueError("P did not retire its reverse worker leases")
            break
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 502 or delay is None:
                raise
            await asyncio.sleep(delay)
    d = await control(config.decode_port, "warm_release", command)
    if d != {"released": True}:
        raise ValueError("D did not retire its terminal worker frames")


async def run_warm_probe(
    client: httpx.AsyncClient,
    config: ProbeConfig,
    decoder: DecodeIdentity,
    prompt: list[int],
    suffix: list[int],
    token: str,
    api_key: str | None = None,
) -> dict:
    """Run two generations while preserving the exact prior accepted token IDs."""
    if (
        not is_control_token(token)
        or not config.capture_token_ids
        or not isinstance(prompt, list)
        or not isinstance(suffix, list)
        or not suffix
        or len(prompt) + len(suffix) + config.output_tokens > _MAX_PROMPT_TOKENS
        or any(type(item) is not int or item < 0 for item in suffix)
    ):
        raise ValueError("Warm probe needs token capture and a bounded raw suffix")

    async def control(port, operation, payload):
        return await _control(client, port, operation, payload, token)

    epoch = time_ns()
    identity = None
    transfer = None
    phase = "warm D terminal arm"
    request_id = ""
    transfer_id = ""

    async def arm(native_prompt, offer):
        nonlocal identity, request_id, transfer_id
        request_id = native_prompt.request_id
        transfer_id = f"kp-return-{secrets.token_hex(8)}"
        grant = offer.grant
        if (grant.engine_id, grant.boot_id) != (decoder.engine_id, decoder.boot_id):
            raise ValueError("D offer changed the configured engine or boot")
        identity = {
            "request_id": request_id,
            "transfer_id": transfer_id,
            "epoch": epoch,
            "source_engine_id": decoder.engine_id,
            "source_boot_id": decoder.boot_id,
            "model_fingerprint": decoder.model_fingerprint,
            "config_fingerprint": decoder.config_fingerprint,
            "deadline_monotonic_ns": monotonic_ns() + int(config.timeout_seconds * 1e9),
        }
        if await control(config.decode_port, "warm_arm", {"identity": identity}) != {
            "armed": True
        }:
            raise ValueError("D rejected its terminal arm")

    try:
        async with asyncio.timeout(config.timeout_seconds):
            first = await run_probe(
                client,
                config,
                prompt,
                token,
                api_key,
                session_epoch=epoch,
                before_decode=arm,
            )
            phase = "warm D terminal source"
            assert identity is not None
            source = await _await_envelope(
                control,
                config.decode_port,
                "warm_source",
                {"identity": identity},
                "source",
            )
            transcript = [*prompt, *first["output_token_ids"]]
            boundary = _accepted_prefix(source, identity, transcript)
            prompt2 = [*transcript, *suffix]

            async def prefill(native_prompt, offer, offered, p_body):
                nonlocal phase, transfer
                phase = "warm P authorize"
                result = await control(
                    config.prefill_port,
                    "warm_authorize",
                    {"source": source, "offer": offered},
                )
                nonce = result.get("nonce")
                if not is_control_token(nonce):
                    raise ValueError("P did not authorize its warm admission")
                params = dict(p_body["kv_transfer_params"])
                del params["kitepond_cold_prefill_offer"]
                params["kitepond_glm_warm_nonce"] = nonce
                task = asyncio.create_task(
                    _post(
                        client,
                        config.prefill_port,
                        "/v1/completions",
                        p_body | {"kv_transfer_params": params},
                        api_key,
                    )
                )
                try:
                    phase = "warm P destination"
                    destination = await _await_envelope(
                        control,
                        config.prefill_port,
                        "warm_destination",
                        {"request_id": native_prompt.request_id},
                        "destination",
                        request=task,
                    )
                    if _body(destination)["request_id"] != offer.prompt.request_id:
                        raise ValueError("P allocated a different warm request")
                    phase = "warm D prepare"
                    result = await control(
                        config.decode_port,
                        "warm_prepare",
                        {"source": source, "destination": destination},
                    )
                    transfer = result["transfer"]
                    if _body(transfer)["destination"] != _body(destination):
                        raise ValueError("Warm transfer changed the P allocation")
                    phase = "warm P submit"
                    result = await control(
                        config.prefill_port,
                        "warm_submit",
                        {
                            "transfer": transfer,
                            "timeout": min(30, config.timeout_seconds),
                        },
                    )
                    if result != {"submitted": True}:
                        raise ValueError("P did not submit its reverse transfer")
                    phase = "warm P suffix"
                    return await task
                except httpx.HTTPStatusError as exc:
                    raise ProbeFailure(
                        phase,
                        native_prompt.request_id,
                        native_prompt.transfer_id,
                        f"HTTP {exc.response.status_code}",
                    ) from None
                except (KeyError, TypeError, ValueError) as exc:
                    raise ProbeFailure(
                        phase,
                        native_prompt.request_id,
                        native_prompt.transfer_id,
                        str(exc),
                    ) from None
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)

            phase = "warm P counters before second turn"
            before = await _prefill_counters(client, config.prefill_port)
            second = await run_probe(
                client,
                config,
                prompt2,
                token,
                api_key,
                session_epoch=epoch,
                prefill_hook=prefill,
            )
            phase = "warm P/D native settlement"
            assert transfer is not None
            ranks = await _finish_return(control, config, transfer)
            phase = "warm P/D native release"
            await _release_return(control, config, transfer)
            phase = "warm P suffix accounting"
            after = await _prefill_counters(client, config.prefill_port)
            counters = {name: after[name] - before[name] for name in before}
            expected_suffix = second["ready_tokens_per_rank"] - boundary
            if not 0 < expected_suffix < len(prompt2) or counters != {
                "local_compute": expected_suffix,
                "local_cache_hit": 0,
                "external_kv_transfer": boundary,
            }:
                raise ValueError(f"P prompt-source counters differ: {counters}")
    except httpx.HTTPStatusError as exc:
        raise ProbeFailure(
            phase, request_id, transfer_id, f"HTTP {exc.response.status_code}"
        ) from None
    except (TimeoutError, httpx.TimeoutException):
        raise ProbeFailure(phase, request_id, transfer_id, "deadline expired") from None
    except httpx.HTTPError:
        raise ProbeFailure(
            phase, request_id, transfer_id, "local HTTP failed"
        ) from None
    except (KeyError, TypeError, ValueError) as exc:
        raise ProbeFailure(phase, request_id, transfer_id, str(exc)) from None

    return {
        "status": "completed",
        "session_epoch": epoch,
        "first_turn": first,
        "second_turn": second,
        "second_prompt_token_ids": prompt2,
        "reused_prefix_tokens": boundary,
        "prefill_suffix_computed_tokens": expected_suffix,
        "prefill_second_turn_source_tokens": counters,
        "settled_reverse_ranks": ranks,
        "prefill_released": True,
        "decode_released": True,
    }


def _config_fingerprint(path: Path) -> str:
    from vllm.transformers_utils.config import get_config, get_hf_text_config

    config = get_config(path.resolve(strict=True), trust_remote_code=False)
    return config_fingerprint(get_hf_text_config(config).to_dict())


async def _run_cli(args) -> dict:
    config = ProbeConfig(
        args.prefill_port,
        args.decode_port,
        args.prefill_bootstrap,
        args.prefill_engine_id,
        args.model,
        args.output_tokens,
        args.timeout,
        capture_token_ids=True,
        seed=args.seed,
    )
    decoder = DecodeIdentity(
        args.decode_engine_id,
        args.decode_boot_id,
        args.model_fingerprint,
        args.config_fingerprint or _config_fingerprint(args.model_path),
    )
    async with httpx.AsyncClient(trust_env=False, timeout=None) as client:
        return await run_warm_probe(
            client,
            config,
            decoder,
            _read_prompt(args.prompt_file),
            _read_prompt(args.suffix_file),
            os.environ.get(TOKEN_ENV, ""),
            os.environ.get(_API_KEY_ENV),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-port", required=True, type=int)
    parser.add_argument("--decode-port", required=True, type=int)
    parser.add_argument("--prefill-bootstrap", required=True)
    parser.add_argument("--prefill-engine-id", required=True)
    parser.add_argument("--decode-engine-id", required=True)
    parser.add_argument("--decode-boot-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    fingerprint = parser.add_mutually_exclusive_group(required=True)
    fingerprint.add_argument("--config-fingerprint")
    fingerprint.add_argument("--model-path", type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--suffix-file", required=True, type=Path)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=110)
    parser.add_argument("--seed", type=int, default=677)
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
