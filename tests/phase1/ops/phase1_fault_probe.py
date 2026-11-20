# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise native cancel, stale READY replay and tool/precision on a private pair."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
from dataclasses import asdict, replace
from pathlib import Path
from time import monotonic_ns, time_ns

import httpx
from kitepond.integrations.vllm.cold_prefill_http import (
    TOKEN_ENV,
    check_ready_batch,
    is_control_token,
)
from kitepond.integrations.vllm.glm_cold_prefill_admission import ColdPrefillPrompt
from kitepond.models.glm5next.frontier import digest
from kitepond.ops.capture import capture_snapshot
from kitepond.ops.cli import _read_manifest
from kitepond.ops.glm_cold_probe import (
    _API_KEY_ENV,
    ProbeConfig,
    _check_completion,
    _control,
    _decode_route,
    _open_offer,
    _post,
    run_probe,
)
from kitepond.ops.wheel import sha256_file
from phase1_mixed_lab import MODEL_LABEL, PORTS, _capture_sources
from phase1_soak_driver import _emit, _no_native_holds, _role_idle, _roles_status

SCHEMA = "kitepond.glm-native-lab-fault.v1"
CHAT = "/v1/chat/completions"
COMPLETION = "/v1/completions"
ABORT = "/inference/v1/abort_requests"
RANKS = [0, 1, 2, 3]
SEED = 677


def require(test, message):
    if not test:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def _check_confirm_ready(reply):
    required = {"status"}
    optional = {"mla_transport"}
    require(
        reply.keys() == required
        or (
            reply.keys() == required | optional
            and reply["mla_transport"] in ("full", "same_pages")
        ),
        "D returned invalid native READY metadata",
    )
    require(reply["status"] == "ready", "D did not confirm native READY")
    return reply.get("mla_transport")


async def stage(pair, tokens, api_id, transfer, epoch, body, path, before_confirm=None):
    """Use the existing signed reserve, native P generate, READY, and D confirm."""
    chat = path == CHAT
    public_id = f"{'chatcmpl' if chat else 'cmpl'}-{api_id}"
    native_id = public_id if chat else f"{public_id}-0"
    prompt = ColdPrefillPrompt.from_tokens(native_id, transfer, epoch, tokens)
    offered = await pair.control("decode", "reserve", asdict(prompt))
    offer = _open_offer(offered, prompt)
    p_response = await pair.post(
        "prefill",
        path,
        body
        | {
            "request_id": api_id,
            "max_tokens": 1,
            "ignore_eos": True,
            "kv_transfer_params": {
                "do_remote_decode": True,
                "transfer_id": transfer,
                "kitepond_cold_prefill_offer": offered["offer"],
            },
        },
    )
    usage = p_response["usage"]
    require(
        p_response["id"] == public_id
        and usage["completion_tokens"] == 1
        and usage["prompt_tokens"] in (len(tokens), offer.imported_tokens)
        and len(p_response["choices"]) == 1
        and p_response["choices"][0]["finish_reason"] == "length",
        "Prefix owner response does not match the issued native lease",
    )
    while True:
        result = await pair.control("prefill", "read_ready", asdict(offer.attempt))
        if result.get("status") != "wait":
            break
        await asyncio.sleep(0.05)
    batch = result["receipt_batch"]
    require(
        result["status"] == "ready"
        and check_ready_batch(batch) == offer.attempt
        and all(
            row["target_valid_tokens"] == row["draft_valid_tokens"] == len(tokens) - 1
            for row in batch["receipts"]
        ),
        "No exact native four-rank READY boundary",
    )
    if before_confirm is not None:
        await before_confirm(offer, batch)
    reply = await pair.control("decode", "confirm_ready", batch)
    _check_confirm_ready(reply)
    return (
        offered,
        offer,
        batch,
        body
        | {
            "request_id": api_id,
            "kv_transfer_params": _decode_route(pair.config, p_response, transfer),
        },
    )


class Pair:
    def __init__(self, args, client, auth, api_key):
        self.client, self.auth, self.api_key = client, auth, api_key
        plan = read(args.pair)
        self.plan_sha = sha256_file(args.pair)
        self.roles = plan["roles"]
        self.runtime = args.runtime
        model = plan["checkpoint_identity"]["model_sha256"]
        peer = plan["mooncake_api"]
        self.boots = {"prefill": peer["p_boot_id"], "decode": peer["d_boot_id"]}
        self.manifest_sha = sha256_file(args.manifest)
        release = _read_manifest(args.manifest)
        wheel = release["wheel"]
        require(
            Path(wheel["filename"]).name == wheel["filename"]
            and sha256_file(args.manifest.with_name(wheel["filename"]))
            == wheel["sha256"]
            and release["base"]["image"].endswith(plan["base_image"]),
            "Release wheel does not match the pair artifact",
        )
        native = (args.pair.parent / "NATIVE_COMMIT").read_text().strip()
        require(
            len(native) == 40 and all(c in "abcdef0123456789" for c in native),
            "Expected exact native commit",
        )
        self.artifact = {
            "release_manifest_sha256": self.manifest_sha,
            "wheel_sha256": wheel["sha256"],
            "base_image": release["base"]["image"],
            "native_commit": native,
            "model_sha256": model,
        }
        self.config = ProbeConfig(
            PORTS["prefill"],
            PORTS["decode"],
            f"http://127.0.0.1:{plan['roles']['prefill']['mooncake_bootstrap_port']}",
            peer["p_engine_id"],
            MODEL_LABEL,
            timeout_seconds=args.timeout,
            capture_token_ids=True,
            seed=SEED,
        )
        self.sources = args.output / "sources.json"
        _emit(self.sources, _capture_sources(args, plan))

    async def control(self, role, operation, payload):
        return await _control(self.client, PORTS[role], operation, payload, self.auth)

    async def post(self, role, path, payload):
        return await _post(self.client, PORTS[role], path, payload, self.api_key)

    async def snapshot(self):
        sample, roles = await asyncio.gather(
            asyncio.to_thread(capture_snapshot, self.sources, self.manifest_sha),
            _roles_status(self.client, self.config, self.auth),
        )
        return sample, dict(zip(PORTS, roles, strict=True))

    async def drain(self):
        async with asyncio.timeout(25):
            while True:
                sample, roles = await self.snapshot()
                if _role_idle(sample) and _no_native_holds(*roles.values()):
                    return {"metrics": sample, "native": roles}
                await asyncio.sleep(0.25)

    async def abort(self, native_id=None):
        response = await self.client.post(
            f"http://127.0.0.1:{PORTS['decode']}{ABORT}",
            json={"request_ids": [native_id] if native_id is not None else []},
            headers=self.api_headers(),
            follow_redirects=False,
        )
        if response.status_code == 403:
            raise ValueError("Stock abort route requires private --enable-scale-out")
        response.raise_for_status()
        require(response.status_code == 200, "Unexpected native abort route response")
        return response.status_code

    def api_headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None


def raw_request(tokens):
    return {
        "model": MODEL_LABEL,
        "prompt": tokens,
        "temperature": 0,
        "stream": False,
        "ignore_eos": True,
        "n": 1,
        "seed": SEED,
        "return_token_ids": True,
    }


def identities():
    return f"kp-fault-{secrets.token_hex(8)}", f"kp-xfer-{secrets.token_hex(8)}"


async def _cancel_stream(pair, d_request, offer, output):
    native_id = offer.prompt.request_id
    public_id = native_id.removesuffix("-0")
    emitted = 0
    requested = False
    finished = None
    done = False
    endpoint = f"http://127.0.0.1:{PORTS['decode']}{COMPLETION}"
    async with pair.client.stream(
        "POST",
        endpoint,
        json=d_request | {"stream": True, "max_tokens": 2048},
        headers=pair.api_headers(),
        follow_redirects=False,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line == "data: [DONE]":
                done = True
                break
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            require(
                event.get("id") == public_id and len(event["choices"]) == 1,
                "D stream changed request identity",
            )
            choice = event["choices"][0]
            token_ids = choice.get("token_ids") or []
            emitted += len(token_ids)
            if choice.get("finish_reason") is not None:
                finished = choice["finish_reason"]
            if token_ids and not requested and finished is None:
                sample, roles = await pair.snapshot()
                d_metrics = sample["components"]["decode"]["metrics"]
                require(
                    d_metrics.get("running") == 1, "No attributed in-flight D request"
                )
                _emit(
                    output / "injection.json",
                    {
                        "request_id": native_id,
                        "lease_sha256": offer.grant.sha256,
                        "stream_tokens_before_cancel": emitted,
                        "observed_before_cancel": {"metrics": sample, "native": roles},
                        "abort_path": ABORT,
                        "dispatch_monotonic_ns": monotonic_ns(),
                        "http_status": await pair.abort(native_id),
                    },
                )
                requested = True
    require(
        requested and finished == "abort" and done and 0 < emitted < 2048,
        "D did not complete the targeted stream with native abort",
    )
    return emitted


def cancelled_tokens(before, after):
    prior = before["metrics"]["components"]["decode"]
    final = after["metrics"]["components"]["decode"]
    key = "generated_tokens_total"
    before_count, after_count = prior["metrics"][key], final["metrics"][key]
    require(
        prior["epoch"] == final["epoch"]
        and type(before_count) is int
        and type(after_count) is int
        and 0 < after_count - before_count < 2048,
        "Native D did not stop below the cancelled sequence limit",
    )
    return after_count - before_count


async def cancel(pair, tokens, output):
    require(2 <= len(tokens) <= 6144, "Keep room for an actual abortable decode")
    # This harmless stock request checks the real endpoint before issuing a grant.
    await pair.abort()
    before_target = await pair.drain()
    api_id, transfer = identities()
    _, offer, _, body = await stage(
        pair, tokens, api_id, transfer, time_ns(), raw_request(tokens), COMPLETION
    )
    token_count = await _cancel_stream(pair, body, offer, output)
    after_target = await pair.drain()
    _emit(output / "after-abort-before-followup.json", after_target)
    observed_delta = cancelled_tokens(before_target, after_target)
    return {
        "native_finish_reason": "abort",
        "tokens_before_stream_end": token_count,
        "native_generated_tokens_delta": observed_delta,
        "follow_up": await recover(pair, tokens),
    }


async def recover(pair, prompt):
    # Use a differently prefixed request to exercise a fresh transaction after drain.
    await pair.drain()
    tokens = prompt.copy()
    tokens[0] = tokens[0] - 1 if tokens[0] else 1
    reply = await run_probe(pair.client, pair.config, tokens, pair.auth, pair.api_key)
    return {
        "request_id": reply["request_id"],
        "closed_ranks": RANKS,
        "input_tokens": reply["prompt_tokens"],
        "output_tokens": reply["output_tokens"],
        "output_token_sha256": digest(reply["output_token_ids"]),
    }


async def capture_old_ack(pair, tokens, output):
    api_id, transfer = identities()
    epoch = time_ns()
    _, offer, batch, body = await stage(
        pair, tokens, api_id, transfer, epoch, raw_request(tokens), COMPLETION
    )
    result = await pair.post("decode", COMPLETION, body | {"max_tokens": 8})
    _check_completion(result, api_id, {len(tokens)}, 8)
    _emit(output / "private-ready-wire.json", batch)
    return {
        "api_id": api_id,
        "transfer_id": transfer,
        "epoch": epoch,
        "lease_sha256": offer.grant.sha256,
        "prompt_sha256": digest(tokens),
        "old_ready_accepted_and_generated": True,
        "ready_wire_sha256": sha256_file(output / "private-ready-wire.json"),
    }


def old_proof(pair, tokens, prior, wire):
    old, batch = read(prior), read(wire)
    expected = old["operation"]
    require(
        old["schema"] == SCHEMA
        and old["case"] == "capture-old-ack"
        and old["artifact"] == pair.artifact
        and old["boots"]["decode"] != pair.boots["decode"]
        and old["plan_sha256"] != pair.plan_sha
        and expected["old_ready_accepted_and_generated"] is True
        and expected["prompt_sha256"] == digest(tokens)
        and expected["ready_wire_sha256"] == sha256_file(wire),
        "Replay requires the same artifact, prior old D boot and exact signed batch",
    )
    attempt = check_ready_batch(batch)
    require(
        attempt.request_id == f"cmpl-{expected['api_id']}-0"
        and attempt.transfer_id == expected["transfer_id"]
        and attempt.epoch == expected["epoch"]
        and attempt.lease_sha256 == expected["lease_sha256"],
        "Old signed READY is not bound to the genuine old request",
    )
    return old, expected, batch


async def replay_ack(pair, tokens, output, prior, wire):
    old, expected, batch = old_proof(pair, tokens, prior, wire)
    epoch = time_ns()
    require(epoch > expected["epoch"], "Fresh request epoch must advance")

    async def replay(offer, fresh):
        attempt = check_ready_batch(fresh)
        require(
            attempt.request_id == batch["attempt"]["request_id"]
            and attempt.transfer_id == batch["attempt"]["transfer_id"]
            and attempt.lease_sha256 != expected["lease_sha256"]
            and attempt.epoch == epoch,
            "Replay needs a real new lease with the same logical request identity",
        )
        before = await pair.control("decode", "status", {})
        require(
            before["active_leases"] == 1 and before["ready_leases"] == 0,
            "No fresh D lease",
        )
        try:
            await pair.control("decode", "confirm_ready", batch)
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            disposition = error.response.json().get("error")
        else:
            raise ValueError("New D accepted the old signed READY")
        after = await pair.control("decode", "status", {})
        _emit(
            output / "injection.json",
            {
                "old_decode_boot": old["boots"]["decode"],
                "new_decode_boot": pair.boots["decode"],
                "old_request_epoch": expected["epoch"],
                "new_request_epoch": epoch,
                "prior_ready_wire_sha256": expected["ready_wire_sha256"],
                "new_lease_sha256": offer.grant.sha256,
                "status": status,
                "disposition": disposition,
                "native_before": before,
                "native_after": after,
            },
        )
        require(
            status == 502
            and disposition == "connector_control_rejected"
            and before == after,
            "Old ACK did not fail at D owner without mutating the fresh lease",
        )

    _, _, _, d_body = await stage(
        pair,
        tokens,
        expected["api_id"],
        expected["transfer_id"],
        epoch,
        raw_request(tokens),
        COMPLETION,
        before_confirm=replay,
    )
    valid = await pair.post("decode", COMPLETION, d_body | {"max_tokens": 8})
    _check_completion(valid, expected["api_id"], {len(tokens)}, 8)
    return {
        "old_ready_rejected": True,
        "fresh_owner_unchanged": True,
        "fresh_quorum_admitted": True,
        "fresh_output_token_sha256": digest(valid["choices"][0]["token_ids"]),
        "follow_up": await recover(pair, tokens),
    }


def decode_json(text):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, "Tool returned duplicate JSON keys")
            result[k] = v
        return result

    return json.loads(text, object_pairs_hook=pairs)


def tool_step_one(choice, oracle):
    message = choice["message"]
    calls = message["tool_calls"]
    require(
        choice["finish_reason"] == "tool_calls" and len(calls) == 1,
        "Expected exactly one parsed native function call",
    )
    call = calls[0]
    function = call["function"]
    expected = oracle["expected_calls"][0]
    require(
        call["type"] == "function"
        and function["name"] == expected["name"]
        and decode_json(function["arguments"]) == expected["arguments"],
        "Tool call does not match the frozen local stub",
    )
    return message, {
        "role": "tool",
        "tool_call_id": call["id"],
        "content": json.dumps(
            oracle["stub"][expected["name"]]["result"], ensure_ascii=False
        ),
    }


async def chat_turn(pair, fixture):
    token_fields = {
        "model": MODEL_LABEL,
        "messages": fixture["messages"],
        "tools": fixture["tools"],
    }
    p, d = await asyncio.gather(
        pair.post("prefill", "/tokenize", token_fields),
        pair.post("decode", "/tokenize", token_fields),
    )
    tokens = p["tokens"]
    require(
        tokens == d["tokens"] and len(tokens) == p["count"] == d["count"],
        "P and D production template tokens differ",
    )
    api_id, transfer = identities()
    raw = fixture | {
        "model": MODEL_LABEL,
        "temperature": 0,
        "n": 1,
        "stream": False,
        "seed": SEED,
        "return_token_ids": True,
    }
    _, _, batch, body = await stage(
        pair, tokens, api_id, transfer, time_ns(), raw, CHAT
    )
    resp = await pair.post("decode", CHAT, body | {"max_tokens": 2048})
    events = resp["choices"]
    require(
        resp["id"] == f"chatcmpl-{api_id}"
        and resp["prompt_token_ids"] == tokens
        and resp["usage"]["prompt_tokens"] == len(tokens)
        and len(events) == 1
        and len(events[0]["token_ids"]) == resp["usage"]["completion_tokens"] > 0,
        "Chat renderer or D physical request accounting differed",
    )
    return events[0], {
        "prompt_sha256": digest(tokens),
        "native_ready_sha256": hashlib.sha256(
            json.dumps(batch, sort_keys=True).encode()
        ).hexdigest(),
        "output_token_sha256": digest(events[0]["token_ids"]),
        "prompt_tokens": len(tokens),
        "output_tokens": len(events[0]["token_ids"]),
    }


async def tool_accuracy(pair, tokens, output, fixture_path, oracle_path):
    fixture = read(fixture_path)
    reference = read(oracle_path)
    want = reference["output_token_ids"]
    require(
        type(want) is list
        and 1 <= len(want) <= 128
        and all(type(token) is int and token >= 0 for token in want),
        "Missing sealed original-engine generated IDs",
    )
    for role in PORTS:
        argv = pair.roles[role]["argv"]
        require(
            "--enable-auto-tool-choice" in argv
            and "--tool-call-parser" in argv
            and argv[argv.index("--tool-call-parser") + 1] == "glm47"
            and "--reasoning-parser" in argv
            and argv[argv.index("--reasoning-parser") + 1] == "glm45",
            "The sealed candidate lacks the native GLM tool/reasoning parser",
        )
    require(
        fixture["case_id"] == "p3-tool-closed-loop"
        and fixture["request"]["tool_choice"] == "auto"
        and reference["prompt_sha256"] == digest(tokens)
        and reference["seed"] == SEED
        and reference["base_image"] == pair.artifact["base_image"]
        and reference["model_sha256"] == pair.artifact["model_sha256"],
        "The stock token oracle or the frozen local tool fixture changed",
    )
    first, step1 = await chat_turn(pair, fixture["request"])
    assistant, stub = tool_step_one(first, fixture["oracle"])
    await pair.drain()
    request2 = fixture["request"] | {
        "messages": [*fixture["request"]["messages"], assistant, stub],
    }
    terminal, step2 = await chat_turn(pair, request2)
    last = terminal["message"]
    require(
        terminal["finish_reason"] == "stop"
        and not last.get("tool_calls")
        and decode_json(last["content"])
        == fixture["oracle"]["expected_final"]["expected"],
        "Model did not compute exact JSON using the actual stub result",
    )
    await pair.drain()
    greedy = await run_probe(
        pair.client,
        replace(pair.config, output_tokens=len(want)),
        tokens,
        pair.auth,
        pair.api_key,
    )
    got = greedy["output_token_ids"]
    _emit(
        output / "token-comparison.json",
        {
            "scope": "exact_seeded_greedy_continuation_only",
            "reference_sha256": sha256_file(oracle_path),
            "input_sha256": digest(tokens),
            "target": got,
            "original_engine": want,
        },
    )
    require(
        want == got,
        "P/D greedy IDs differed from the separately pinned original reference",
    )
    return {
        "tool_verified": True,
        "token_comparison_retained": True,
        "numeric_acceptance_met": True,
        "scope": "frozen_stub_roundtrip_and_exact_greedy_ids",
        "tool_fixture_sha256": sha256_file(fixture_path),
        "tool_turns": [step1, step2],
        "oracle_sha256": sha256_file(oracle_path),
    }


async def run(args):
    token = os.environ.get(TOKEN_ENV)
    require(is_control_token(token), "Missing private connector token")
    args.runtime = args.runtime.resolve(strict=True)
    require(
        args.output.parent.resolve() == args.runtime, "Keep fault evidence in this boot"
    )
    prompt = read(args.prompt)
    require(
        isinstance(prompt, list)
        and 2 <= len(prompt) <= 6144
        and all(type(x) is int and x >= 0 for x in prompt),
        "Expected bounded frozen raw prompt",
    )
    args.output.mkdir(mode=0o700)
    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
        pair = Pair(args, client, token, os.environ.get(_API_KEY_ENV))
        before = await pair.drain()
        _emit(args.output / "before.json", before)
        async with asyncio.timeout(args.timeout):
            operation = await {
                "cancel": lambda: cancel(pair, prompt, args.output),
                "capture-old-ack": lambda: capture_old_ack(pair, prompt, args.output),
                "replay-ack": lambda: replay_ack(
                    pair, prompt, args.output, args.prior, args.wire
                ),
                "tool-accuracy": lambda: tool_accuracy(
                    pair, prompt, args.output, args.fixture, args.oracle
                ),
            }[args.case]()
        after = await pair.drain()
        _emit(args.output / "after.json", after)
        result = {
            "schema": SCHEMA,
            "case": args.case,
            "artifact": pair.artifact,
            "plan_sha256": pair.plan_sha,
            "boots": pair.boots,
            "operation": operation,
            "before_sha256": sha256_file(args.output / "before.json"),
            "after_sha256": sha256_file(args.output / "after.json"),
            "scope": "observed_native_pair_overlay_lab_only",
            "release_qualification": "not_evaluated",
        }
        _emit(args.output / "result.json", result)
        return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "case", choices=("cancel", "capture-old-ack", "replay-ack", "tool-accuracy")
    )
    for name in ("manifest", "pair", "prompt", "runtime", "output"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    for name in ("prior", "wire", "fixture", "oracle"):
        cli.add_argument(f"--{name}", type=Path)
    cli.add_argument("--timeout", type=int, default=110)
    args = cli.parse_args()
    try:
        require(10 <= args.timeout <= 120, "Keep the native bounded operator deadline")
        require(
            args.case != "replay-ack" or (args.prior and args.wire),
            "Missing old settled evidence",
        )
        require(
            args.case != "tool-accuracy" or (args.fixture and args.oracle),
            "Missing frozen tool and pinned original-engine reference",
        )
        result = asyncio.run(run(args))
    except (
        OSError,
        TimeoutError,
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        httpx.HTTPError,
    ) as exc:
        cli.exit(2, f"Private lab did not finish ({type(exc).__name__}): {exc}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
