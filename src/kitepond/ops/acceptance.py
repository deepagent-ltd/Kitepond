# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate existing release evidence without contacting a serving process."""

import hashlib
import json
from pathlib import Path

from kitepond.ops.report import operational_report
from kitepond.ops.wheel import inspect_wheel, is_hex, sha256_file

INDEX_SCHEMA = "kitepond.release-acceptance-index.v1"
ATTEMPT_SCHEMA = "kitepond.release-acceptance-attempt.v1"
ROLES = ("prefill", "decode")
RANKS = [0, 1, 2, 3]
CONCURRENCY = {"c1": 1, "c32": 32, "c64": 64}
CPU_CASES = ("artifact_provenance", "stock_install")
FAULTS = (
    "cancel",
    "lost_write_ack",
    "prefill_process_exit",
    "decode_process_exit",
    "late_epoch_ack",
    "nvme_partial_write",
    "nvme_corruption",
    "nvme_read_failure",
)
GPU_CASES = (
    *CONCURRENCY,
    "bidirectional",
    "p_ready",
    "hisparse_long",
    "tool_and_accuracy",
    "ram_saturation",
    "cold_restart",
    *FAULTS,
    "mixed_soak",
    "rollback",
)
CASES = CPU_CASES + GPU_CASES
GATES = {
    "ownership": ("c1", "p_ready", "ram_saturation"),
    "vllm_integration": (
        "stock_install",
        *CONCURRENCY,
        "hisparse_long",
        "tool_and_accuracy",
    ),
    "backends": ("bidirectional", "cold_restart"),
    "stability": (*FAULTS, "mixed_soak"),
    "deployment": ("artifact_provenance", "stock_install", "rollback"),
}
MODEL_FILES = {
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(data, name: str) -> dict:
    _require(isinstance(data, dict), f"{name} must be an object")
    return data


def _identifier(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _read(root: Path, reference: dict, *, parse: bool = False):
    reference = _mapping(reference, "Evidence file reference")
    name = reference.get("file")
    digest = reference.get("sha256")
    _require(
        _identifier(name) and is_hex(digest, 64), "Invalid evidence file reference"
    )
    relative = Path(name)
    _require(
        not relative.is_absolute() and ".." not in relative.parts,
        "Unsafe evidence path",
    )
    path = (root / relative).resolve(strict=True)
    _require(
        path.is_relative_to(root) and path.is_file(),
        "Evidence file is outside the bundle",
    )
    if not parse:
        _require(sha256_file(path) == digest, f"Digest mismatch for {name}")
        return path
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == digest, f"Digest mismatch for {name}")
    return _mapping(json.loads(data), name)


def _identity(manifest: dict, digest: str, plan: dict, config_sha256: str) -> dict:
    expected = {
        "release_manifest_sha256": digest,
        "wheel_sha256": manifest["wheel"]["sha256"],
        "source_commit": manifest["source_commit"],
        "base_image": manifest["base"]["image"],
    }
    _require(
        plan.get("manifest_sha256") == digest
        and plan.get("wheel_sha256") == expected["wheel_sha256"]
        and plan.get("source_commit") == expected["source_commit"]
        and plan.get("base") == manifest["base"],
        "Deployment plan does not match the sealed release",
    )
    model = _mapping(plan.get("model_files"), "Model fingerprint")
    _require(
        set(model) >= MODEL_FILES
        and all(is_hex(value, 64) for value in model.values()),
        "Deployment plan lacks the model fingerprints",
    )
    roles = _mapping(plan.get("roles"), "Deployment role devices")
    _require(
        set(roles) == set(ROLES)
        and all(
            isinstance(items, list) and len(items) == 4 for items in roles.values()
        ),
        "Deployment requires four devices per role",
    )
    indices = roles["prefill"] + roles["decode"]
    _require(
        all(type(index) is int and index >= 0 for index in indices)
        and len(set(indices)) == 8,
        "Deployment requires eight distinct GPU indices",
    )
    return {**expected, "model_files": model, "configuration_sha256": config_sha256}


def _all_true(proof: dict, *names: str) -> None:
    for name in names:
        _require(proof.get(name) is True, f"Missing positive proof: {name}")


def _same_ranks(proof: dict, key: str) -> None:
    ranks = proof.get(key)
    _require(
        isinstance(ranks, dict)
        and set(ranks) == set(ROLES)
        and all(
            items == RANKS and all(type(rank) is int for rank in items)
            for items in ranks.values()
        ),
        f"{key} must identify all four ranks for both roles",
    )


def _runtime_report(snapshot: dict, previous: dict | None, identity: dict) -> dict:
    report = operational_report(snapshot, previous, identity["release_manifest_sha256"])
    _require(
        report["artifact_matches"] and report["source"] == "observed",
        "Snapshot lacks matching observed provenance",
    )
    for role in ROLES:
        component = report["components"][role]
        _require(
            _identifier(component["epoch"]), f"{role} snapshot has no process epoch"
        )
        _require(
            all(
                component["metrics"][metric]["status"] == "observed"
                for metric in ("healthy", "running", "waiting")
            ),
            f"{role} snapshot lacks health or queue readings",
        )
    return report


def _roles(report: dict) -> dict:
    return {role: report["components"][role]["epoch"] for role in ROLES}


def _idle(report: dict) -> None:
    _require(
        all(
            report["components"][role]["metrics"]["healthy"]["value"] is True
            and report["components"][role]["metrics"]["running"]["value"] == 0
            and report["components"][role]["metrics"]["waiting"]["value"] == 0
            for role in ROLES
        ),
        "P and D must be healthy and idle at this boundary",
    )


def _delta(report: dict, component: str, name: str) -> float:
    delta = report["components"][component]["metrics"][name]["delta"]
    _require(
        delta["status"] == "observed", f"Unattributed counter delta: {component}.{name}"
    )
    return delta["value"]


def _settled(report: dict) -> None:
    settlement = report["components"]["settlement"]
    _require(
        _identifier(settlement["epoch"])
        and _identifier(settlement["scope"])
        and settlement["metrics"]["held_references"].get("value") == 0
        and settlement["metrics"]["pending_retirement"].get("value") == 0,
        "Final settlement is unavailable or still holds references",
    )


def _stock_install_reference(observed: dict, manifest_digest: str) -> dict:
    _require(
        observed.get("vllm_preserved") is True
        and observed.get("engine_imported") is False
        and observed.get("gpu_qualified") is False,
        "Expected the original CPU stock installation receipt",
    )
    checks = _mapping(observed.get("checks"), "Stock installation checks")
    verify = _mapping(checks.get("verify"), "Stock wheel verification")
    _require(
        verify.get("wheel_verified") is True
        and verify.get("manifest_sha256") == manifest_digest,
        "Stock installation references another manifest",
    )
    stock = _mapping(observed.get("stock"), "Independent stock fingerprint")
    _require(
        _identifier(stock.get("version"))
        and _identifier(stock.get("origin"))
        and type(stock.get("vllm_file_count")) is int
        and stock["vllm_file_count"] > 0
        and is_hex(stock.get("vllm_files_sha256"), 64),
        "Missing independent stock file identity",
    )
    _validate_doctor(checks.get("doctor"), manifest_digest, stock)
    return stock


def _validate_doctor(observed, manifest_digest: str, stock: dict, *, hooks=False):
    doctor = _mapping(observed, "Engine doctor evidence")
    matches = _mapping(doctor.get("matches"), "Engine doctor checks")
    _require(
        doctor.get("manifest_sha256") == manifest_digest
        and doctor.get("compatible_metadata") is True
        and all(
            matches.get(key) is True
            for key in ("kitepond", "python", "vllm", "vllm_origin")
        ),
        "Engine doctor installation differs from the intended manifest",
    )
    _require(
        doctor.get("engine_files") == stock,
        "Engine doctor file identity differs from independent stock",
    )
    if hooks:
        source = _mapping(
            doctor.get("engine_source_capabilities"), "Engine source hook evidence"
        )
        _require(
            source.get("scope") == "source_signatures"
            and source.get("distribution_version") == stock["version"],
            "Engine doctor source hook inventory differs from stock version",
        )


def _require_stock_runtime(root: Path, reference, identity: dict, runtime: dict):
    evidence = _mapping(reference, "Runtime stock provenance")
    _require(
        evidence.keys() == {"stock", *ROLES},
        "Runtime stock provenance requires stock, P and D",
    )
    manifest_digest = identity["release_manifest_sha256"]
    stock = _stock_install_reference(
        _read(root, evidence["stock"], parse=True), manifest_digest
    )
    for role, epoch in _roles(runtime).items():
        source = _mapping(evidence[role], f"{role} engine doctor provenance")
        _require(
            source.get("role") == role
            and source.get("epoch") == epoch
            and source.get("base_image") == identity["base_image"],
            "Engine doctor evidence is attributed to a different role, boot or image",
        )
        _validate_doctor(
            _read(root, source["doctor"], parse=True),
            manifest_digest,
            stock,
            hooks=True,
        )


def _gpu_evidence(root: Path, evidence: dict, identity: dict, rollback: dict | None):
    devices = _mapping(evidence.get("gpu_devices"), "GPU device assignment")
    _require(set(devices) == set(ROLES), "GPU evidence requires P and D device lists")
    _require(
        all(
            isinstance(devices[role], list) and len(devices[role]) == 4
            for role in ROLES
        ),
        "GPU evidence requires four device UUIDs per role",
    )
    uuids = devices["prefill"] + devices["decode"]
    _require(
        all(
            isinstance(item, str) and item.startswith("GPU-") and len(item) > 4
            for item in uuids
        )
        and len(set(uuids)) == 8,
        "GPU evidence requires eight distinct device UUIDs",
    )
    snapshots = _mapping(evidence.get("snapshots"), "Runtime snapshots")
    before = _read(root, snapshots["before"], parse=True)
    after = _read(root, snapshots["after"], parse=True)
    start = _runtime_report(before, None, identity)
    end = _runtime_report(after, before, rollback or identity)
    _require(
        evidence.get("before_role_epochs") == _roles(start),
        "Incorrect initial P/D epochs",
    )
    _require(
        evidence.get("after_role_epochs") == _roles(end), "Incorrect final P/D epochs"
    )
    if rollback is None:
        _require(
            start["deployment"] == end["deployment"],
            "P/D snapshots came from different deployments",
        )
    provenance = _mapping(
        evidence.get("stock_engine_provenance"), "Stock engine evidence"
    )
    _require_stock_runtime(root, provenance["before"], identity, start)
    _require_stock_runtime(root, provenance["after"], rollback or identity, end)
    return snapshots, start, end, after


def _evaluate_runtime(
    case: str,
    root: Path,
    receipt: dict,
    identity: dict,
    rollback: dict | None,
    soak_seconds: int | None,
) -> None:
    proof = _mapping(receipt.get("proof"), "Runtime proof")
    snapshots, before, after, final_sample = _gpu_evidence(
        root, receipt, identity, rollback if case == "rollback" else None
    )
    _idle(before)
    _idle(after)
    _settled(after)
    changed = {role for role in ROLES if _roles(before)[role] != _roles(after)[role]}
    expected_change = {
        "cold_restart": set(ROLES),
        "rollback": set(ROLES),
        "prefill_process_exit": {"prefill"},
        "decode_process_exit": {"decode"},
        "late_epoch_ack": {"decode"},
    }.get(case, set())
    if case in ("prefill_process_exit", "decode_process_exit", "late_epoch_ack"):
        _require(expected_change <= changed, "The targeted role epoch did not restart")
    else:
        _require(
            changed == expected_change,
            "Observed P/D process epoch change does not match the case",
        )
    if not changed:
        _require(
            after["interval"]["status"] == "observed",
            "No attributable runtime interval",
        )

    if case in CONCURRENCY:
        required = CONCURRENCY[case]
        _require(
            type(proof.get("peak_concurrency")) is int
            and proof["peak_concurrency"] >= required,
            "Insufficient observed request concurrency",
        )
        _require(
            type(proof.get("successful_requests")) is int
            and proof["successful_requests"] >= required,
            "Insufficient successful requests",
        )
        _all_true(proof, "native_token_counts_match", "outputs_valid")
        _same_ranks(proof, "closed_ranks")
        for component, counter in (
            ("prefill", "logical_input_tokens_total"),
            ("decode", "generated_tokens_total"),
            ("settlement", "closed_checkpoints_total"),
        ):
            _require(
                _delta(after, component, counter) > 0, f"No measured {counter} increase"
            )
    elif case == "bidirectional":
        _all_true(
            proof,
            "p_consumed_prior_d",
            "d_consumed_prior_p",
            "incremental_bytes_accounted",
        )
        _require(
            _delta(after, "prefill", "reused_input_tokens_total") > 0,
            "No P history consumption",
        )
    elif case == "p_ready":
        _same_ranks(proof, "fenced_ranks")
        _all_true(proof, "publication_after_gpu_fence", "no_early_consumption")
        _require(
            _delta(after, "handoff", "ready_events_total") > 0,
            "No observed READY event",
        )
    elif case == "hisparse_long":
        _all_true(
            proof,
            "long_request_completed",
            "bounded_device_working_set",
            "host_recall_observed",
        )
    elif case == "tool_and_accuracy":
        _all_true(
            proof,
            "tool_verified",
            "token_comparison_retained",
            "numeric_acceptance_met",
        )
    elif case == "ram_saturation":
        during = _runtime_report(
            _read(root, snapshots["queued"], parse=True), None, identity
        )
        _require(
            during["deployment"] == after["deployment"]
            and _roles(during) == _roles(after),
            "Queue observation came from a different P/D run",
        )
        _require(
            any(
                during["components"][role]["metrics"]["waiting"]["value"] > 0
                for role in ROLES
            ),
            "No observed queue under RAM pressure",
        )
        _all_true(proof, "limit_reached", "request_resumed", "no_duplicate_backing")
    elif case == "cold_restart":
        initial = _read(root, snapshots["after_restart"], parse=True)
        fresh = _runtime_report(initial, None, identity)
        restore = _runtime_report(final_sample, initial, identity)
        _idle(fresh)
        _require(
            fresh["deployment"] == after["deployment"]
            and _roles(fresh) == _roles(after),
            "Cold restore is not bound to the fresh P/D epochs",
        )
        _require(
            _delta(restore, "storage", "nvme_read_bytes_total") > 0,
            "No actual NVMe read after restart",
        )
        _require(
            _delta(restore, "decode", "generated_tokens_total") > 0,
            "No model output after cold restore",
        )
        _all_true(
            proof,
            "empty_cold_root_before_store",
            "persistent_commit_complete",
            "ram_owner_absent_after_restart",
        )
        _same_ranks(proof, "restored_ranks")
        stored = proof.get("stored_payload_sha256")
        _require(
            is_hex(stored, 64) and stored == proof.get("restored_payload_sha256"),
            "Cold payload bytes were not matched",
        )
    elif case in FAULTS:
        _all_true(
            proof,
            "injection_observed",
            "expected_disposition",
            "model_recovered",
            "lease_released",
        )
        if case == "late_epoch_ack":
            replay = _mapping(proof.get("replay"), "Late ACK replay")
            _require(
                replay.get("source_decode_epoch") == _roles(before)["decode"]
                and replay.get("target_decode_epoch") == _roles(after)["decode"],
                "Late ACK replay is not bound to the old and fresh decode epochs",
            )
            _read(root, replay["signed_ack"])
            _all_true(replay, "old_ack_previously_accepted", "fresh_lease_unchanged")
    elif case == "mixed_soak":
        _require(
            type(soak_seconds) is int and soak_seconds > 0,
            "No duration set in the acceptance index",
        )
        interval = after["interval"]
        _require(
            interval["status"] == "observed"
            and interval["elapsed_ns"] >= soak_seconds * 1_000_000_000,
            "Observed continuous soak duration is below the declared policy",
        )
        _all_true(
            proof,
            "mixed_population_covered",
            "resource_guard_passed",
            "natural_and_injected_errors_separated",
        )
        _require(
            _delta(after, "decode", "generated_tokens_total") > 0,
            "No model output during soak",
        )
    elif case == "rollback":
        _all_true(
            proof,
            "ordered_drain_and_stop",
            "prior_wheel_verified",
            "warm_and_cold_results_verified",
            "old_generation_recovered",
        )
        elapsed = proof.get("recovery_seconds")
        _require(
            type(elapsed) is int and elapsed > 0,
            "Missing measured rollback completion time",
        )


def _attempt(
    case: str,
    root: Path,
    receipt: dict,
    identity: dict,
    rollback: dict | None,
    soak_seconds: int | None,
    fault_events: set[str],
    fault_results: set[str],
) -> dict:
    _require(
        receipt.get("schema") == ATTEMPT_SCHEMA and receipt.get("case") == case,
        "Incorrect attempt schema or case",
    )
    run = receipt.get("run")
    _require(_identifier(run), "Missing run identity")
    _require(
        receipt.get("identity")
        == (
            {
                key: identity[key]
                for key in (
                    "release_manifest_sha256",
                    "wheel_sha256",
                    "source_commit",
                    "base_image",
                )
            }
            if case in CPU_CASES
            else identity
        ),
        "Wrong release, image, model, or configuration identity",
    )
    mode = receipt.get("mode")
    _require(mode in ("cpu", "gpu", "fixture"), "Unknown attempt mode")
    _read(root, receipt["driver"])
    result = receipt["result"]
    _read(root, result)
    report = {"run": run, "mode": mode, "result_sha256": result["sha256"]}
    if receipt.get("status") == "FAIL":
        return {**report, "status": "FAIL", "reason": "Driver reported failure"}
    _require(
        receipt.get("status") == "PASS", "Expected PASS or FAIL in the driver attempt"
    )
    expected_mode = "cpu" if case in CPU_CASES else "gpu"
    if mode != expected_mode:
        return {
            **report,
            "status": "NOT_RUN",
            "reason": f"Requires {expected_mode} evidence",
        }
    if case == "stock_install":
        _stock_install_reference(
            _read(root, result, parse=True), identity["release_manifest_sha256"]
        )
    elif case == "artifact_provenance":
        _all_true(
            _mapping(receipt.get("proof"), "Artifact proof"),
            "reproduced",
            "lock_verified",
            "licenses_reviewed",
        )
    else:
        if case == "rollback":
            _require(
                rollback is not None and receipt.get("rollback_identity") == rollback,
                "Rollback baseline is not verified",
            )
        if case in FAULTS:
            event = receipt["injection"]
            _read(root, event)
            injection = event["sha256"]
            _require(
                injection not in fault_events and result["sha256"] not in fault_results,
                "Fault arms require independent injection and result artifacts",
            )
            fault_events.add(injection)
            fault_results.add(result["sha256"])
        _evaluate_runtime(case, root, receipt, identity, rollback, soak_seconds)
    return {**report, "status": "PASS"}


def _frozen_release(root: Path, files: dict, read_manifest) -> tuple[dict, str]:
    manifest_path = _read(root, files["manifest"])
    manifest = read_manifest(manifest_path)
    _require(
        inspect_wheel(_read(root, files["wheel"])) == manifest["wheel"],
        "Wheel does not match its sealed manifest",
    )
    plan = _read(root, files["plan"], parse=True)
    config_path = _read(root, files["configuration"])
    return (
        _identity(
            manifest, files["manifest"]["sha256"], plan, sha256_file(config_path)
        ),
        files["manifest"]["sha256"],
    )


def _gate_status(outcomes: dict, cases: tuple[str, ...]) -> str:
    states = {outcomes[case]["status"] for case in cases}
    if states == {"PASS"}:
        return "PASS"
    return "FAIL" if "FAIL" in states else "NOT_RUN"


def evaluate_release(
    index_path: Path,
    manifest_path: Path,
    wheel_path: Path,
    read_manifest,
) -> dict:
    """Match direct evidence against the fixed Phase 1 release cases."""
    root = index_path.resolve(strict=True).parent
    index = _mapping(json.loads(index_path.read_bytes()), "Acceptance index")
    _require(index.get("schema") == INDEX_SCHEMA, "Unsupported acceptance index schema")
    identity, digest = _frozen_release(
        root, _mapping(index.get("release"), "Release files"), read_manifest
    )
    _require(
        sha256_file(manifest_path) == digest
        and sha256_file(wheel_path) == identity["wheel_sha256"],
        "Acceptance index targets a different command-line release",
    )
    rollback = None
    if index.get("rollback_baseline") is not None:
        rollback, _ = _frozen_release(
            root, _mapping(index["rollback_baseline"], "Rollback files"), read_manifest
        )
        _require(
            rollback != identity and rollback["model_files"] == identity["model_files"],
            "Rollback requires a distinct configuration or artifact for the same model",
        )
    declared = _mapping(index.get("cases"), "Acceptance cases")
    _require(set(declared) <= set(CASES), "Unknown release case")
    fault_events: set[str] = set()
    fault_results: set[str] = set()
    outcomes = {}
    for case in CASES:
        references = declared.get(case, [])
        _require(isinstance(references, list), f"Expected an attempt list for {case}")
        attempts = []
        runs = set()
        for reference in references:
            receipt = {}
            try:
                receipt = _read(root, reference, parse=True)
                _require(
                    receipt.get("run") not in runs,
                    "Duplicate run identity for this case",
                )
                runs.add(receipt.get("run"))
                row = _attempt(
                    case,
                    root,
                    receipt,
                    identity,
                    rollback,
                    index.get("soak_seconds"),
                    fault_events,
                    fault_results,
                )
            except (KeyError, OSError, TypeError, ValueError) as issue:
                row = {
                    "status": "FAIL",
                    "run": receipt.get("run"),
                    "reason": str(issue),
                }
            attempts.append(row)
        actual = [row for row in attempts if row["status"] != "NOT_RUN"]
        outcomes[case] = {
            "status": actual[-1]["status"] if actual else "NOT_RUN",
            "attempts": attempts,
        }
    return {
        "schema": "kitepond.release-acceptance-report.v1",
        "identity": identity,
        "rollback_identity": rollback,
        "complete": all(row["status"] == "PASS" for row in outcomes.values()),
        "cases": outcomes,
        "gates": {
            name: {"status": _gate_status(outcomes, cases), "cases": list(cases)}
            for name, cases in GATES.items()
        },
    }
