# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture authenticated GLM scheduler-owner settlement from both roles."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from time import monotonic_ns

import httpx

from kitepond.integrations.vllm.cold_prefill_http import TOKEN_ENV, is_control_token
from kitepond.integrations.vllm.glm_settlement import OPERATION, SCHEMA

from .cli import _read_manifest
from .glm_cold_probe import _control
from .wheel import sha256_file

PAIR_SCHEMA = "kitepond.glm-pair-owner-settlement.v1"
_COMMON = {"native_scheduler_connections", "native_rank_transport", "warm", "forward"}
_COMPONENTS = {
    "prefill": _COMMON | {"ready_collector"},
    "decode": _COMMON
    | {
        "host_authority",
        "native_decode_completion",
        "terminal",
        "cold_ram",
        "fsl2_io",
        "nvme_operations",
    },
}


def _role_source(pair, role):
    recipe = pair["mooncake_api"]
    name = "p" if role == "prefill" else "d"
    argv = pair["roles"][role]["argv"]
    port = int(argv[argv.index("--port") + 1])
    if (
        argv[argv.index("--host") + 1] != "127.0.0.1"
        or not 0 < port < 65536
        or recipe["d_engine_id"] != pair["authority_engine_id"]
        or recipe["d_boot_id"] != pair["authority_boot_id"]
    ):
        raise ValueError("Owner settlement requires the exact local pair plan")
    return port, {
        "schema": SCHEMA,
        "role": role,
        "engine_id": recipe[f"{name}_engine_id"],
        "boot_id": recipe[f"{name}_boot_id"],
        "authority_engine_id": recipe["d_engine_id"],
        "authority_boot_id": recipe["d_boot_id"],
    }


def _validate(reply, expected, started, finished):
    """Reject identity changes, incomplete schemas and foreign timestamps."""
    keys = {"observed_monotonic_ns", "components"}
    if (
        set(reply) != set(expected) | keys
        or any(reply[key] != value for key, value in expected.items())
        or type(reply["observed_monotonic_ns"]) is not int
        or not started <= reply["observed_monotonic_ns"] <= finished
        or type(reply["components"]) is not dict
        or reply["components"].keys() != _COMPONENTS[expected["role"]]
    ):
        raise ValueError("Unexpected native owner settlement identity")
    for component in reply["components"].values():
        if (
            type(component) is not dict
            or component.keys() != {"state", "counts"}
            or component["state"] not in ("observed", "unavailable", "disabled")
            or type(component["counts"]) is not dict
            or bool(component["counts"]) != (component["state"] == "observed")
            or any(type(v) is not int or v < 0 for v in component["counts"].values())
        ):
            raise ValueError("Invalid native owner settlement component")
    return reply


def _combine(roles, errors):
    unavailable = sorted(
        f"{role}.{component}"
        for role, reply in roles.items()
        for component, part in reply["components"].items()
        if part["state"] == "unavailable"
    )
    decode = roles.get("decode", {}).get("components", {})
    # These are independently acquired model block references. A positive
    # reference disproves zero even while native rank coverage is incomplete.
    held = any(
        decode.get(component, {}).get("counts", {}).get(key, 0) > 0
        for component, key in (
            ("host_authority", "grant_block_references"),
            ("terminal", "block_references"),
            ("cold_ram", "resident_block_references"),
        )
    )
    return {
        "schema": PAIR_SCHEMA,
        "capture_scope": "authorized_owner_thread",
        "roles": roles,
        "unavailable_role_errors": errors,
        "unavailable_components": unavailable,
        # A sequence of individual scheduler samples does not prove transport
        # retirement across rank event loops. Keep global zero unknown.
        "owner_reference_counts_zero": False if held else None,
    }


async def sample_pair(client: httpx.AsyncClient, pair: dict, token: str) -> dict:
    if not is_control_token(token):
        raise ValueError("The private GLM settlement bearer is required")
    roles = {role: _role_source(pair, role) for role in _COMPONENTS}
    if roles["prefill"][0] == roles["decode"][0]:
        raise ValueError("P and D must expose distinct control endpoints")

    async def capture(role, port, expected):
        started = monotonic_ns()
        try:
            reply = await _control(client, port, OPERATION, {}, token)
            return role, _validate(reply, expected, started, monotonic_ns()), None
        except (ValueError, KeyError, TypeError, httpx.HTTPError) as error:
            return role, None, type(error).__name__

    captures = await asyncio.gather(
        *(capture(role, *source) for role, source in roles.items())
    )
    return _combine(
        {role: value for role, value, _ in captures if value is not None},
        {role: failure for role, _, failure in captures if failure is not None},
    )


async def _run(args):
    release = _read_manifest(args.manifest)
    pair = json.loads(args.pair.read_bytes())
    wheel = release["wheel"]
    if (
        not release["base"]["image"].endswith(pair["base_image"])
        or Path(wheel["filename"]).name != wheel["filename"]
        or sha256_file(args.manifest.with_name(wheel["filename"])) != wheel["sha256"]
    ):
        raise ValueError("Local role and release artifact evidence differ")
    async with httpx.AsyncClient(trust_env=False, timeout=12) as client:
        sample = await sample_pair(client, pair, os.environ.get(TOKEN_ENV))
    return {
        **sample,
        "release_manifest_sha256": sha256_file(args.manifest),
        "pair_sha256": sha256_file(args.pair),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, sort_keys=True))
    return 2 if result["unavailable_role_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
