# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare two operational snapshots without contacting a running engine."""

import math

from kitepond.ops.wheel import is_hex

SNAPSHOT_SCHEMA = "kitepond.ops-snapshot.v1"
REPORT_SCHEMA = "kitepond.ops-report.v1"

GAUGES = {
    "prefill": ("healthy", "running", "waiting"),
    "decode": ("healthy", "running", "waiting"),
    "decode_residency": (),
    "handoff": (),
    "settlement": ("held_references", "pending_retirement", "oldest_reference_age_ns"),
    "residency": (),
    "storage": ("ram_resident_bytes", "nvme_payload_bytes"),
}
COUNTERS = {
    "prefill": (
        "logical_input_tokens_total",
        "computed_input_tokens_total",
        "reused_input_tokens_total",
        "native_compute_seconds_total",
    ),
    "decode": ("generated_tokens_total", "output_interval_seconds_total"),
    "decode_residency": (
        "hisparse_cache_hits_total",
        "hisparse_cache_misses_total",
        "hisparse_host_to_device_bytes_total",
    ),
    "handoff": (
        "ready_events_total",
        "ready_control_seconds_total",
        "payload_transfer_bytes_total",
        "payload_transfer_seconds_total",
    ),
    "settlement": ("retired_objects_total", "closed_checkpoints_total"),
    "storage": ("nvme_written_bytes_total", "nvme_read_bytes_total"),
    "residency": (
        "hot_page_hits_total",
        "host_recalled_pages_total",
        "host_recalled_bytes_total",
    ),
}
SCOPED = {"settlement", "storage", "residency", "decode_residency"}


def _object(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _identifier(value, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty identifier")


def _reading(value, name: str) -> None:
    if value is None:
        return
    if name == "healthy":
        valid = type(value) is bool
    elif name.endswith("_seconds_total"):
        try:
            valid = type(value) in (int, float) and value >= 0 and math.isfinite(value)
        except OverflowError:
            valid = False
    else:
        valid = type(value) is int and value >= 0
    if not valid:
        raise ValueError(f"Invalid {name} reading")


def validate_snapshot(sample) -> dict:
    sample = _object(sample, "Snapshot")
    if sample.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("Unsupported operational snapshot schema")
    if not is_hex(sample.get("release_manifest_sha256"), 64):
        raise ValueError("Snapshot requires the sealed release manifest SHA-256")
    if sample.get("sources_sha256") is not None and not is_hex(
        sample["sources_sha256"], 64
    ):
        raise ValueError("Invalid telemetry sources SHA-256")
    _identifier(sample.get("deployment"), "deployment")
    if sample.get("source") not in ("observed", "synthetic"):
        raise ValueError("Snapshot source must be observed or synthetic")
    collector = _object(sample.get("collector"), "collector")
    _identifier(collector.get("epoch"), "collector epoch")
    timestamp = collector.get("monotonic_ns")
    if type(timestamp) is not int or timestamp < 0:
        raise ValueError("Collector timestamp must be a nonnegative integer")
    for name, record in _object(sample.get("components"), "components").items():
        if name not in GAUGES:
            raise ValueError(f"Unknown component {name!r}")
        component = _object(record, name)
        for key in ("epoch", "scope"):
            if component.get(key) is not None:
                _identifier(component[key], f"{name} {key}")
        status = component.get("source_status")
        if status is not None and status not in (
            "observed",
            "partial",
            "unavailable",
            "process_unavailable",
            "receipt_unavailable",
        ):
            raise ValueError(f"Invalid {name} source status")
        readings = _object(component.get("metrics", {}), "metrics")
        for metric, value in readings.items():
            if metric not in GAUGES[name] + COUNTERS[name]:
                raise ValueError(f"Unknown {name} metric {metric!r}")
            _reading(value, metric)
        for metric, values in _object(
            component.get("counter_series", {}), "counter series"
        ).items():
            if (
                metric not in COUNTERS[name]
                or not isinstance(values, dict)
                or not values
            ):
                raise ValueError(f"Invalid {name} counter series")
            for index, value in values.items():
                _identifier(index, "counter series identity")
                if value is None:
                    raise ValueError("Counter series requires observed values")
                _reading(value, metric)
            if sum(values[index] for index in sorted(values)) != readings.get(metric):
                raise ValueError("Counter series does not match the exported total")
    return sample


def _interval(current: dict, previous: dict | None, artifact_matches: bool) -> dict:
    if not artifact_matches:
        return {"status": "artifact_mismatch"}
    if previous is None:
        return {"status": "baseline_missing"}
    for field, mismatch in (
        ("release_manifest_sha256", "artifact_changed"),
        ("deployment", "deployment_changed"),
        ("source", "source_changed"),
    ):
        if current[field] != previous[field]:
            return {"status": mismatch}
    now, before = current["collector"], previous["collector"]
    if now["epoch"] != before["epoch"]:
        return {"status": "collector_restarted"}
    elapsed = now["monotonic_ns"] - before["monotonic_ns"]
    if elapsed <= 0:
        return {"status": "non_increasing_time"}
    return {"status": "observed", "elapsed_ns": elapsed}


def _component_interval(name: str, current: dict, previous: dict) -> str:
    if current.get("epoch") is None or previous.get("epoch") is None:
        return "component_epoch_missing"
    if current["epoch"] != previous["epoch"]:
        return "component_restarted"
    scope_now, scope_before = current.get("scope"), previous.get("scope")
    if name in SCOPED and (scope_now is None or scope_before is None):
        return "scope_missing"
    if scope_now != scope_before:
        return "scope_changed"
    return "observed"


def _series_status(current: dict, previous: dict, metric: str) -> str:
    after = current.get("counter_series", {}).get(metric, {})
    before = previous.get("counter_series", {}).get(metric, {})
    if after.keys() != before.keys():
        return "series_changed"
    if any(after[index] < value for index, value in before.items()):
        return "counter_reset"
    return "observed"


def _report_metrics(name: str, current: dict, previous: dict, interval: str) -> dict:
    after = current.get("metrics", {})
    before = previous.get("metrics", {})
    result = {}
    for metric in GAUGES[name] + COUNTERS[name]:
        value = after.get(metric)
        entry = (
            {"status": "missing"}
            if value is None
            else {
                "status": "observed",
                "value": value,
            }
        )
        if metric in COUNTERS[name]:
            old = before.get(metric)
            if interval != "observed":
                entry["delta"] = {"status": interval}
            elif value is None:
                entry["delta"] = {"status": "current_missing"}
            elif old is None:
                entry["delta"] = {"status": "baseline_missing"}
            elif value < old:
                entry["delta"] = {"status": "counter_reset"}
            else:
                status = _series_status(current, previous, metric)
                entry["delta"] = {"status": status}
                if status == "observed":
                    entry["delta"]["value"] = value - old
        result[metric] = entry
    return result


def _coverage(metrics: dict, names) -> str:
    return (
        "observed"
        if all(metrics[name]["status"] == "observed" for name in names)
        else "missing"
    )


def _scoped_coverage(component: dict, names) -> str:
    if component["scope"] is None:
        return "scope_missing"
    if component["epoch"] is None:
        return "component_epoch_missing"
    return _coverage(component["metrics"], names)


def operational_report(current, previous, expected_manifest_sha256: str) -> dict:
    """Describe operational observations and their attributable intervals."""
    current = validate_snapshot(current)
    if previous is not None:
        previous = validate_snapshot(previous)
    artifact_matches = current["release_manifest_sha256"] == expected_manifest_sha256
    interval = _interval(current, previous, artifact_matches)

    components = {}
    baselines = {} if previous is None else previous["components"]
    for name in GAUGES:
        component = current["components"].get(name, {})
        prior = baselines.get(name, {})
        comparison = interval["status"]
        if comparison == "observed":
            comparison = (
                _component_interval(name, component, prior)
                if name in baselines
                else "baseline_missing"
            )
        components[name] = {
            "epoch": component.get("epoch"),
            "scope": component.get("scope"),
            "source_status": component.get("source_status"),
            "interval_status": comparison,
            "metrics": _report_metrics(name, component, prior, comparison),
        }

    roles = [components[name]["metrics"] for name in ("prefill", "decode")]
    health = [role["healthy"].get("value") for role in roles]
    role_health = "missing" if None in health else "healthy"
    if False in health:
        role_health = "unhealthy"
    settlement = components["settlement"]
    storage = components["storage"]
    return {
        "schema": REPORT_SCHEMA,
        "release_manifest_sha256": current["release_manifest_sha256"],
        "expected_manifest_sha256": expected_manifest_sha256,
        "artifact_matches": artifact_matches,
        "source": current["source"],
        "sources_sha256": current.get("sources_sha256"),
        "deployment": current["deployment"],
        "collector": current["collector"],
        "interval": interval,
        "components": components,
        "gate_evidence": {
            "role_health": role_health,
            "inflight": (
                "observed"
                if all(
                    _coverage(role, ("running", "waiting")) == "observed"
                    for role in roles
                )
                else "missing"
            ),
            "phase_counters": {
                name: _coverage(components[name]["metrics"], COUNTERS[name])
                for name in ("prefill", "decode", "handoff")
            },
            "decode_residency": _scoped_coverage(
                components["decode_residency"], COUNTERS["decode_residency"]
            ),
            "references_and_retirement": _scoped_coverage(
                settlement,
                ("held_references", "pending_retirement", "retired_objects_total"),
            ),
            "checkpoint_closure": _scoped_coverage(
                settlement, ("closed_checkpoints_total",)
            ),
            "storage": _scoped_coverage(
                storage, GAUGES["storage"] + COUNTERS["storage"]
            ),
            "release_qualification": "not_evaluated",
        },
    }
