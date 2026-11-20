# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a shareable diagnostic report without engine or GPU access."""

import hashlib
import json
from pathlib import Path

from kitepond.ops.acceptance import FAULTS
from kitepond.ops.report import COUNTERS, GAUGES, SCOPED, operational_report
from kitepond.ops.wheel import is_hex

SCHEMA = "kitepond.ops-diagnosis.v1"
CASE_STATES = {"PASS", "FAIL", "NOT_RUN"}
INSTALL_MATCHES = ("kitepond", "python", "vllm", "vllm_origin")


def _read(path: Path | None) -> tuple[dict | None, str | None]:
    if path is None:
        return None, None
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Diagnostic input must contain a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _stock_verdict(sample: dict | None, artifact: str) -> str:
    if sample is None:
        return "missing"
    if sample.get("manifest_sha256") != artifact:
        raise ValueError("Doctor report belongs to another release manifest")
    checks = sample.get("matches")
    if not isinstance(checks, dict):
        raise ValueError("Doctor report must include installation checks")
    if sample.get("compatible_metadata") is not True or not all(
        checks.get(key) is True for key in INSTALL_MATCHES
    ):
        return "incompatible_installation"
    capabilities = sample.get("engine_source_capabilities")
    if capabilities is None:
        return "engine_hooks_unchecked"
    if not isinstance(capabilities, dict):
        raise ValueError("Doctor engine hooks must be a JSON object")
    if capabilities.get("connector_module_path") is not True:
        return "external_connector_unavailable"
    if capabilities.get("full_cache_mode") == "unavailable":
        return "registration_probe_only"
    return "runtime_qualification_required"


def _acceptance_cases(
    sample: dict | None, manifest: dict, artifact: str
) -> dict[str, dict]:
    if sample is None:
        return {}
    identity = sample.get("identity", {})
    expected = {
        "release_manifest_sha256": artifact,
        "wheel_sha256": manifest["wheel"]["sha256"],
        "source_commit": manifest["source_commit"],
        "base_image": manifest["base"]["image"],
    }
    if (
        sample.get("schema") != "kitepond.release-acceptance-report.v1"
        or not isinstance(identity, dict)
        or any(identity.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("Acceptance report belongs to another release manifest")
    entries = sample.get("cases")
    if not isinstance(entries, dict):
        raise ValueError("Acceptance report requires case statuses")
    cases = {}
    for name, entry in entries.items():
        state = entry.get("status") if isinstance(entry, dict) else None
        if not isinstance(state, str) or state not in CASE_STATES:
            raise ValueError("Invalid acceptance case status")
        attempts = entry.get("attempts", [])
        if not isinstance(attempts, list) or any(
            not isinstance(attempt, dict) for attempt in attempts
        ):
            raise ValueError("Invalid acceptance attempt list")
        rows = [row for row in attempts if row.get("status") != "NOT_RUN"]
        result = rows[-1].get("result_sha256") if rows else None
        if state == "PASS" and (
            not rows or rows[-1].get("status") != "PASS" or not is_hex(result, 64)
        ):
            raise ValueError("Accepted case lacks an original result digest")
        cases[name] = {
            "status": state,
            "result_sha256": result if is_hex(result, 64) else None,
        }
    return cases


def _fingerprint(value: str | None) -> str | None:
    return None if value is None else hashlib.sha256(value.encode()).hexdigest()


def _readings(report: dict | None, component: str) -> dict:
    fields = GAUGES[component] + COUNTERS[component]
    if report is None:
        return {field: {"status": "missing"} for field in fields}
    selected = report["components"][component]
    readings = {}
    for field in fields:
        value = selected["metrics"][field]
        if value["status"] == "missing":
            readings[field] = {"status": "missing"}
            continue
        status = "observed"
        if report["source"] == "synthetic":
            status = "synthetic"
        elif component in SCOPED and (not selected["scope"] or not selected["epoch"]):
            status = "unattributed"
        readings[field] = {"status": status, "value": value["value"]}
        if "delta" in value and status == "observed":
            readings[field]["delta"] = value["delta"]
    return readings


def diagnostic_report(
    manifest: dict,
    digest: str,
    *,
    current: Path | None = None,
    previous: Path | None = None,
    doctor: Path | None = None,
    acceptance: Path | None = None,
) -> dict:
    """Project the five Phase 1 evidence areas onto stable fields."""
    observed, current_digest = _read(current)
    baseline, previous_digest = _read(previous)
    startup, doctor_digest = _read(doctor)
    accepted, acceptance_digest = _read(acceptance)
    if baseline is not None and observed is None:
        raise ValueError("A previous snapshot requires a current snapshot")
    report = (
        None if observed is None else operational_report(observed, baseline, digest)
    )
    if report is not None and not report["artifact_matches"]:
        raise ValueError("Current snapshot belongs to another release manifest")
    cases = _acceptance_cases(accepted, manifest, digest)
    verdict = _stock_verdict(startup, digest)
    missing_case = {"status": "NOT_RUN", "result_sha256": None}
    return {
        "schema": SCHEMA,
        "release_manifest_sha256": digest,
        "inputs_sha256": {
            "current": current_digest,
            "previous": previous_digest,
            "doctor": doctor_digest,
            "acceptance": acceptance_digest,
        },
        "run_scope": {
            "source": None if report is None else report["source"],
            "deployment_fingerprint": (
                None if report is None else _fingerprint(report["deployment"])
            ),
            "interval": {"status": "missing"} if report is None else report["interval"],
            "components": {
                name: {
                    "epoch_fingerprint": (
                        None
                        if report is None
                        else _fingerprint(report["components"][name]["epoch"])
                    ),
                    "scope_fingerprint": (
                        None
                        if report is None
                        else _fingerprint(report["components"][name]["scope"])
                    ),
                    "source_status": (
                        None
                        if report is None
                        else report["components"][name].get("source_status")
                    ),
                }
                for name in GAUGES
            },
        },
        "core": {"settlement": _readings(report, "settlement")},
        "stock_integration": {
            "startup_source_verdict": verdict,
            "ordinary_cases": {
                name: cases.get(name, missing_case)["status"]
                for name in (
                    "c1",
                    "c32",
                    "c64",
                    "bidirectional",
                    "p_ready",
                    "tool_and_accuracy",
                )
            },
        },
        "storage_and_transport": {
            "transfer": _readings(report, "handoff"),
            "storage": _readings(report, "storage"),
            "residency": _readings(report, "residency"),
            "decode_residency": _readings(report, "decode_residency"),
            "cold_restart_receipt": cases.get("cold_restart", missing_case),
            "long_hisparse_receipt": cases.get("hisparse_long", missing_case),
        },
        "stability": {
            "workload_cases": {
                name: cases.get(name, missing_case)["status"]
                for name in ("ram_saturation", *FAULTS, "mixed_soak")
            },
            "prefill": _readings(report, "prefill"),
            "decode": _readings(report, "decode"),
        },
        "deployment": {
            "artifact_provenance": cases.get("artifact_provenance", missing_case)[
                "status"
            ],
            "stock_install": cases.get("stock_install", missing_case)["status"],
            "rollback_receipt": cases.get("rollback", missing_case),
        },
    }
