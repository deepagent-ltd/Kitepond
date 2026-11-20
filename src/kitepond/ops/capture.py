# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture stock vLLM loopback metrics and scoped Kitepond adapter receipts."""

import hashlib
import http.client
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from kitepond.ops.report import COUNTERS, GAUGES, SNAPSHOT_SCHEMA, validate_snapshot

SOURCES_SCHEMA = "kitepond.ops-sources.v1"
RECEIPT_SCHEMA = "kitepond.ops-receipt.v1"
_PROCESS_STAT = Path("/proc")
_BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
_ROLES = {"prefill", "decode"}
_ADAPTER_COMPONENTS = {"handoff", "settlement", "storage", "residency"}
_MAX_EXPORT_BYTES = 16 * 1024 * 1024
_MAX_RECEIPT_BYTES = 64 * 1024
_SELECTOR = {
    "vllm:num_requests_running": ("running", ""),
    "vllm:num_requests_waiting": ("waiting", ""),
    "vllm:prompt_tokens_total": ("logical_input_tokens_total", ""),
    "vllm:prompt_tokens_by_source_total": (
        "computed_input_tokens_total",
        "local_compute",
    ),
    "vllm:generation_tokens_total": ("generated_tokens_total", ""),
    "vllm:request_decode_time_seconds_sum": ("output_interval_seconds_total", ""),
}
_PROMPT_BY_SOURCE = "vllm:prompt_tokens_by_source_total"


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive(value) -> bool:
    return type(value) is int and value > 0


@dataclass(frozen=True)
class EngineSource:
    port: int
    pid: int
    process_start_ticks: int
    engine_epoch: str
    model_name: str
    engine_ids: tuple[str, ...]

    @classmethod
    def parse(cls, value) -> "EngineSource":
        if not isinstance(value, dict):
            raise ValueError("Engine source must be an object")
        target = dict(value)
        engines = target.get("engine_ids")
        if (
            not all(
                _positive(target.get(key))
                for key in ("port", "pid", "process_start_ticks")
            )
            or target["port"] > 65535
            or not all(
                _nonempty(target.get(key)) for key in ("engine_epoch", "model_name")
            )
            or not isinstance(engines, list)
            or not engines
            or any(not _nonempty(index) for index in engines)
            or len(set(engines)) != len(engines)
            or set(target) != set(cls.__dataclass_fields__)
        ):
            raise ValueError("Invalid engine source identity")
        target["engine_ids"] = tuple(sorted(engines))
        return cls(**target)


@dataclass(frozen=True)
class AdapterSource:
    path: str
    epoch: str
    scope: str
    max_age_seconds: int

    @classmethod
    def parse(cls, value) -> "AdapterSource":
        if (
            not isinstance(value, dict)
            or set(value) != set(cls.__dataclass_fields__)
            or not all(_nonempty(value[key]) for key in ("path", "epoch", "scope"))
            or not Path(value["path"]).is_absolute()
            or not _positive(value["max_age_seconds"])
        ):
            raise ValueError("Invalid adapter receipt source identity")
        return cls(**value)


def _role_export(text: str, role: str, source: EngineSource) -> tuple[dict, dict]:
    try:
        from prometheus_client.parser import text_string_to_metric_families
    except ImportError as issue:
        raise ValueError("Prometheus parser dependency is unavailable") from issue

    counters_for_role = COUNTERS[role]
    allowed = set(GAUGES[role]) | set(counters_for_role)
    selected = {
        (name, tag): field
        for name, (field, tag) in _SELECTOR.items()
        if field in allowed
    }
    if role == "prefill":
        selected[(_PROMPT_BY_SOURCE, "local_cache_hit")] = "local_apc"
        selected[(_PROMPT_BY_SOURCE, "external_kv_transfer")] = "external_kv_transfer"
    if role == "decode":
        native = COUNTERS["decode_residency"]
        selected.update({(f"vllm:{field}", ""): field for field in native})
        counters_for_role += native

    series: dict[str, dict] = {name: {} for name in selected.values()}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            labels = sample.labels
            key = (sample.name, labels.get("source", ""))
            field = selected.get(key)
            index = labels.get("engine")
            if (
                field is None
                or labels.get("model_name") != source.model_name
                or index not in source.engine_ids
            ):
                continue
            expected_labels = {"model_name", "engine"}
            if key[1]:
                expected_labels.add("source")
            value = sample.value
            valid = (
                math.isfinite(value) and value >= 0 and labels.keys() == expected_labels
            )
            if field.endswith("_seconds_total"):
                reading = value if valid else None
            else:
                reading = int(value) if valid and int(value) == value else None
            series[field][index] = None if index in series[field] else reading

    complete = {
        name: values
        for name, values in series.items()
        if len(values) == len(source.engine_ids) and None not in values.values()
    }
    local = complete.pop("local_apc", None)
    external = complete.pop("external_kv_transfer", None)
    if local is not None and external is not None and not any(external.values()):
        complete["reused_input_tokens_total"] = local
    metrics = {
        name: sum(values[index] for index in sorted(values))
        for name, values in complete.items()
    }
    for name, total in tuple(metrics.items()):
        if name.endswith("_seconds_total") and not math.isfinite(total):
            del metrics[name]
            del complete[name]
    counters = {
        name: values for name, values in complete.items() if name in counters_for_role
    }
    return metrics, counters


def _process_start(pid: int) -> int:
    stat = (_PROCESS_STAT / str(pid) / "stat").read_text(encoding="ascii")
    _, closing, fields = stat.rpartition(")")
    if not closing or len(fields.split()) <= 19:
        raise ValueError("Malformed process stat")
    return int(fields.split()[19])


def _request(port: int, path: str) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request("GET", path)
        with connection.getresponse() as response:
            payload = b""
            if path == "/metrics" and response.status == 200:
                payload = response.read(_MAX_EXPORT_BYTES + 1)
                if len(payload) > _MAX_EXPORT_BYTES:
                    raise ValueError("Prometheus export exceeds its maximum size")
            return response.status, payload
    finally:
        connection.close()


def _same_process(source: EngineSource) -> bool:
    try:
        return _process_start(source.pid) == source.process_start_ticks
    except (OSError, ValueError):
        return False


def _capture_engine(role: str, source: EngineSource, boot: str) -> dict:
    scope = _digest((role, source.port, source.model_name, source.engine_ids))
    missing = {"scope": scope, "metrics": {}, "source_status": "process_unavailable"}
    if not _same_process(source):
        return missing
    sample = {
        "epoch": _digest(
            (boot, source.engine_epoch, source.pid, source.process_start_ticks)
        ),
        "scope": scope,
        "metrics": {},
        "source_status": "unavailable",
    }
    try:
        status, _ = _request(source.port, "/health")
        if status in (200, 503):
            sample["metrics"]["healthy"] = status == 200
    except (OSError, http.client.HTTPException, ValueError):
        pass
    try:
        status, body = _request(source.port, "/metrics")
        if status == 200:
            values, series = _role_export(body.decode(), role, source)
            sample["metrics"].update(values)
            sample["counter_series"] = series
    except (OSError, http.client.HTTPException, ValueError):
        pass
    if not _same_process(source):
        return missing
    measurements = set(sample["metrics"])
    if measurements:
        core = set(GAUGES[role]) | {
            "logical_input_tokens_total"
            if role == "prefill"
            else "generated_tokens_total"
        }
        sample["source_status"] = "observed" if core <= measurements else "partial"
    return sample


def _decode_residency(decode: dict) -> dict:
    """Commit residency readings under the same D process and scrape scope."""
    fields = COUNTERS["decode_residency"]
    values = decode["metrics"]
    series = decode.get("counter_series", {})
    metrics = {field: values.pop(field) for field in fields if field in values}
    selected = {field: series.pop(field) for field in fields if field in series}
    if len(metrics) == len(fields):
        status = "observed"
    elif metrics:
        status = "partial"
    elif decode["source_status"] == "process_unavailable":
        status = "process_unavailable"
    else:
        status = "unavailable"
    return {
        "epoch": decode.get("epoch"),
        "scope": decode["scope"],
        "metrics": metrics,
        "counter_series": selected,
        "source_status": status,
    }


def _read_json(path: Path) -> dict:
    with path.open("rb") as file:
        block = file.read(_MAX_RECEIPT_BYTES + 1)
    if len(block) > _MAX_RECEIPT_BYTES:
        raise ValueError("Telemetry JSON exceeds its maximum size")
    data = json.loads(block)
    if not isinstance(data, dict):
        raise ValueError("Telemetry JSON must contain an object")
    return data


def decode_adapter_receipt(
    value: dict,
    *,
    component: str,
    source: AdapterSource,
    artifact: str,
    deployment: str,
    boot: str,
    now_ns: int,
) -> dict:
    expected = {
        "schema": RECEIPT_SCHEMA,
        "component": component,
        "release_manifest_sha256": artifact,
        "deployment": deployment,
        "boot_id": boot,
        "epoch": source.epoch,
        "scope": source.scope,
    }
    if any(value.get(key) != selected for key, selected in expected.items()):
        raise ValueError("Adapter receipt identity mismatch")
    recorded = value.get("monotonic_ns")
    if (
        not _positive(recorded)
        or not 0 <= now_ns - recorded <= source.max_age_seconds * 10**9
    ):
        raise ValueError("Adapter receipt is expired or from a different clock")
    if not isinstance(value.get("metrics"), dict):
        raise ValueError("Adapter receipt requires an explicit metrics object")
    return {
        "epoch": source.epoch,
        "scope": source.scope,
        "metrics": value["metrics"],
        "source_status": "observed",
    }


def capture_snapshot(source_path: Path, release_manifest_sha256: str) -> dict:
    settings = _read_json(source_path)
    deployment = settings.get("deployment")
    if (
        settings.get("schema") != SOURCES_SCHEMA
        or settings.get("release_manifest_sha256") != release_manifest_sha256
        or not _nonempty(deployment)
    ):
        raise ValueError("Telemetry sources do not match the specified release")
    roles = settings.get("roles")
    adapters = settings.get("adapters", {})
    if (
        not isinstance(roles, dict)
        or set(roles) != _ROLES
        or not isinstance(adapters, dict)
        or adapters.keys() - _ADAPTER_COMPONENTS
    ):
        raise ValueError(
            "Specify both engine roles and only declared adapter components"
        )
    engines = {role: EngineSource.parse(value) for role, value in roles.items()}
    receipts = {
        component: AdapterSource.parse(value) for component, value in adapters.items()
    }
    if engines["prefill"].port == engines["decode"].port:
        raise ValueError("Prefill and decode require distinct loopback ports")

    boot = _BOOT_ID.read_text(encoding="ascii").strip()
    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "source": "observed",
        "sources_sha256": _digest(settings),
        "release_manifest_sha256": release_manifest_sha256,
        "deployment": deployment,
        "collector": {"epoch": boot, "monotonic_ns": 0},
        "components": {
            role: _capture_engine(role, source, boot)
            for role, source in engines.items()
        },
    }
    snapshot["components"]["decode_residency"] = _decode_residency(
        snapshot["components"]["decode"]
    )
    for component, source in receipts.items():
        try:
            document = _read_json(Path(source.path))
            data = decode_adapter_receipt(
                document,
                component=component,
                source=source,
                artifact=release_manifest_sha256,
                deployment=deployment,
                boot=boot,
                now_ns=time.monotonic_ns(),
            )
            snapshot["components"][component] = data
            validate_snapshot(snapshot)
        except (OSError, ValueError):
            snapshot["components"][component] = {
                "scope": source.scope,
                "metrics": {},
                "source_status": "receipt_unavailable",
            }
    snapshot["collector"]["monotonic_ns"] = time.monotonic_ns()
    return validate_snapshot(snapshot)
