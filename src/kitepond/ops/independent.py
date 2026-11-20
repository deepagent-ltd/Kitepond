# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate the separate optimized GLM artifact on the fixed SM120 workloads."""

import csv
import json
import math
from pathlib import Path

from kitepond.ops.acceptance import _mapping, _read, _require
from kitepond.ops.wheel import is_hex

INDEX_SCHEMA = "kitepond.independent-glm-performance-index.v1"
PROFILE = "glm53-flash-nvfp4-sm120-dp8"
EXPECTED = {
    "prefill": (128, 262144, 128),
    "decode": (32, 33690, 32768),
}
REQUIRED_CHECKS = {
    "client_tokens",
    "exact_server_counters",
    "fixed_output",
    "no_errors",
    "positive_duration",
    "prefill_input_2048",
    "quiet_before_after",
    "total_records",
}
INACTIVE_FLAGS = {
    "--enable-prefix-caching",
    "--kv-transfer-config",
    "--enable-hisparse",
    "--worker-extension-cls",
}


def _image_id(value: str) -> bool:
    return (
        isinstance(value, str) and value.startswith("sha256:") and is_hex(value[7:], 64)
    )


def _read_text(root: Path, reference: dict) -> str:
    return _read(root, reference).read_text(encoding="utf-8")


def _option(command: list, flag: str, expected: str) -> bool:
    return (
        command.count(flag) == 1
        and command.index(flag) + 1 < len(command)
        and command[command.index(flag) + 1] == expected
    )


def _plan_identity(plan: dict, digest: str) -> dict:
    _require(
        plan.get("schema") == "kitepond-phase1-independent-glm-lane.v1",
        "Unsupported standalone engine receipt",
    )
    source = _mapping(plan.get("source"), "Standalone source")
    _require(
        is_hex(source.get("commit"), 40)
        and is_hex(source.get("math_commit"), 40)
        and _image_id(plan.get("image")),
        "Standalone image or source identity is incomplete",
    )
    native = _mapping(plan.get("native_sha256"), "Native artifact fingerprints")
    _require(
        bool(native) and all(is_hex(value, 64) for value in native.values()),
        "Standalone native artifacts are not fingerprinted",
    )
    model = _mapping(plan.get("model"), "Standalone model")
    fingerprints = _mapping(model.get("metadata_sha256"), "Model fingerprints")
    _require(
        {"config.json", "model.safetensors.index.json", "tokenizer.json"}
        <= set(fingerprints)
        and all(is_hex(value, 64) for value in fingerprints.values())
        and isinstance(model.get("served_model_id"), str)
        and len(model.get("indexed_shard_bytes", {})) == 33,
        "Standalone model identity is incomplete",
    )
    server = plan.get("server")
    _require(
        isinstance(server, list) and all(isinstance(value, str) for value in server),
        "Missing standalone engine configuration",
    )
    _require(
        _option(server, "--host", "127.0.0.1")
        and _option(server, "--port", "18160")
        and _option(server, "--tensor-parallel-size", "1")
        and _option(server, "--data-parallel-size", "8")
        and _option(server, "--pipeline-parallel-size", "1")
        and "--enable-expert-parallel" in server
        and "--no-enable-prefix-caching" in server
        and not any(value.partition("=")[0] in INACTIVE_FLAGS for value in server),
        "The standalone workload used a different runtime profile",
    )
    return {
        "engine_receipt_sha256": digest,
        "engine_image": plan["image"],
        "source_commit": source["commit"],
        "math_commit": source["math_commit"],
        "model_id": model["served_model_id"],
        "model_metadata_sha256": fingerprints,
    }


def _negative_probe(text: str) -> None:
    records = [
        json.loads(line) for line in text.splitlines() if line.startswith('{"abi":')
    ]
    _require(
        "CLI_DRY_CHECK_PASS" in text.splitlines()
        and len(records) == 1
        and records[0]["optional_dependencies_imported"]["kitepond"] is False,
        "Missing standalone Kitepond package and import negative evidence",
    )


def _observe_container(root: Path, record: dict, identity: dict) -> str:
    fields = _read_text(root, record["container"]).split()
    _require(
        len(fields) == 6
        and is_hex(fields[0], 64)
        and fields[1] == identity["engine_image"]
        and fields[2].isdigit()
        and int(fields[2]) > 0
        and fields[3:] == ["none", "true", "{}"],
        "Observed container did not match the sealed image and private loopback",
    )
    process = [
        line
        for line in _read_text(root, record["processes"]).splitlines()
        if "vllm.entrypoints.cli.main serve" in line
    ]
    _require(
        any(
            "--host 127.0.0.1" in line
            and "--port 18160" in line
            and "--data-parallel-size 8" in line
            and "--no-enable-prefix-caching" in line
            and not INACTIVE_FLAGS.intersection(line.split())
            for line in process
        ),
        "Observed standalone process identity is missing or unexpected",
    )
    devices = list(
        csv.reader(
            _read_text(root, record["devices"]).splitlines(), skipinitialspace=True
        )
    )
    _require(
        len(devices) == 8
        and all(len(row) == 4 for row in devices)
        and {row[0] for row in devices} == set("01234567")
        and all(
            row[1].startswith("GPU-") and row[2] == "NVIDIA RTX 6000D"
            for row in devices
        )
        and len({row[1] for row in devices}) == 8,
        "Observed engine lacks eight distinct observed RTX 6000D devices",
    )
    model = _read(root, record["model"], parse=True).get("data")
    _require(
        isinstance(model, list)
        and len(model) == 1
        and model[0]["id"] == identity["model_id"]
        and model[0]["max_model_len"] == 1048576,
        "Observed API model identity did not match",
    )
    return fields[0]


def _workload(root: Path, name: str, record: dict, plan: dict, identity: dict) -> dict:
    count, input_tokens, output_tokens = EXPECTED[name]
    _require(
        record.get("engine_receipt_sha256") == identity["engine_receipt_sha256"],
        "Workload uses a different engine receipt",
    )
    args = plan["benchmark"][name]
    _require(
        _option(args, "--num-prompts", str(count))
        and _option(args, "--max-concurrency", str(count)),
        "Workload does not match the frozen request and concurrency count",
    )
    container_id = _observe_container(root, record, identity)
    if name == "decode":
        _require(
            _image_id(_read_text(root, record["client_image"]).strip()),
            "Missing pinned benchmark client identity",
        )
    native_closed = _read_text(root, record["native_closed"]).strip()
    prefix = "Native closed delta by rank: "
    pairs = [
        item.partition("=") for item in native_closed.removeprefix(prefix).split(", ")
    ]
    _require(
        native_closed.startswith(prefix)
        and len(pairs) == 8
        and all(
            separator and value.isascii() and value.isdecimal()
            for _, separator, value in pairs
        ),
        "Malformed native close count receipt",
    )
    closed_by_rank = {rank: value for rank, _, value in pairs}
    _require(
        set(closed_by_rank) == set("01234567")
        and all(int(value) > 0 for value in closed_by_rank.values())
        and sum(map(int, closed_by_rank.values())) == count,
        "Missing observed native success counts for all ranks",
    )
    result = _read(root, record["result"], parse=True)
    checks = _mapping(result.get("checks"), "Workload checks")
    _require(
        result.get("case") == name
        and result.get("completed") == count
        and result.get("failed") == 0
        and result.get("input_tokens") == input_tokens
        and result.get("output_tokens") == output_tokens
        and all(checks.get(key) is True for key in REQUIRED_CHECKS),
        "Workload failed the observed request or idle checks",
    )
    _require(
        result.get("metrics_delta")
        == {
            "vllm:prompt_tokens_total": input_tokens,
            "vllm:generation_tokens_total": output_tokens,
            "vllm:prefix_cache_hits_total": 0,
            "vllm:external_prefix_cache_hits_total": 0,
            "vllm:num_preemptions_total": 0,
        },
        "Workload did not match measured tokens and cache counters",
    )
    duration = result.get("duration_seconds")
    _require(
        type(duration) in (int, float)
        and math.isfinite(duration)
        and duration > 0
        and math.isclose(result["input_tok_s"], input_tokens / duration)
        and math.isclose(result["output_tok_s"], output_tokens / duration),
        "Workload throughput does not match actual tokens and duration",
    )
    latency = _mapping(result.get("ms"), "Workload latency")
    metrics = ("ttft", "tpot", "itl", "e2el") if name == "decode" else ("ttft", "e2el")
    for metric in metrics:
        _require(
            all(
                type(latency[metric][stat]) in (int, float)
                and math.isfinite(latency[metric][stat])
                and latency[metric][stat] >= 0
                for stat in ("mean", "p90", "p99")
            ),
            f"No observed latency statistics for {metric}",
        )
    return {
        "status": "PASS",
        "container_id": container_id,
        "qualified_summary_sha256": record["result"]["sha256"],
        "measurement": result,
    }


def evaluate_independent_glm(path: Path) -> dict:
    """Read and qualify independent engine evidence, with no release wheel."""
    root = path.resolve(strict=True).parent
    index = _mapping(json.loads(path.read_bytes()), "Standalone evidence index")
    _require(
        index.get("schema") == INDEX_SCHEMA
        and index.get("measurement_profile") == PROFILE,
        "Unsupported standalone acceptance index",
    )
    reference = index["engine_receipt"]
    plan = _read(root, reference, parse=True)
    identity = _plan_identity(plan, reference["sha256"])
    _negative_probe(_read_text(root, index["negative_package_probe"]))
    cases = _mapping(index.get("cases"), "Standalone cases")
    _require(set(cases) <= set(EXPECTED), "Unknown standalone workload")
    if set(cases) == set(EXPECTED):
        left, right = (cases[name].get("devices") for name in EXPECTED)
        _require(
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.get("sha256") == right.get("sha256"),
            "Standalone workloads report different physical devices",
        )
    results = {}
    for name in EXPECTED:
        if name not in cases:
            results[name] = {"status": "NOT_RUN"}
            continue
        try:
            results[name] = _workload(root, name, cases[name], plan, identity)
        except (KeyError, OSError, TypeError, ValueError) as issue:
            results[name] = {"status": "FAIL", "reason": str(issue)}
    return {
        "schema": "kitepond.independent-glm-performance-report.v1",
        "identity": identity,
        "cases": results,
        "complete": all(item["status"] == "PASS" for item in results.values()),
    }
