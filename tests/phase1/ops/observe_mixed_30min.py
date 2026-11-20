# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reconcile a 30-minute mixed observation without rewriting the 2-hour run."""

import argparse
import json
import math
from collections import Counter
from hashlib import sha256
from pathlib import Path

SCHEMA = "kitepond.glm-observed-mixed-window.v1"
CUTOFF_SCHEMA = "kitepond.glm-observed-operator-cutoff.v1"
PHASES = ("cold", "warm", "nvme")
METRICS = {
    "prefill": "logical_input_tokens_total",
    "decode": "generated_tokens_total",
}
SECOND = 10**9
TARGET = 1800


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def count(value):
    require(type(value) is int and value >= 0, "Missing nonnegative measured count")
    return value


def read(path):
    return json.loads(path.read_bytes())


def digest(data):
    return sha256(data).hexdigest()


def file_digest(path):
    return digest(path.read_bytes())


def records(path):
    """An append in progress contributes only after its newline is written."""
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        lines.pop()
    return [json.loads(line) for line in lines], lines


def snapshot(value):
    require(
        value["schema"] == "kitepond.ops-snapshot.v1" and value["source"] == "observed",
        "Collector snapshot is unobserved",
    )
    collector = value["collector"]
    identity = [
        collector["epoch"],
        value["deployment"],
        value["release_manifest_sha256"],
        value["sources_sha256"],
    ]
    absolute = {}
    scopes = {}
    series = {}
    for role, metric in METRICS.items():
        component = value["components"][role]
        require(component["source_status"] == "observed", "Native role unavailable")
        scopes[role] = [component["epoch"], component["scope"]]
        values = component["counter_series"][metric]
        require(type(values) is dict and values, "Native counter series unavailable")
        series[role] = {key: count(x) for key, x in values.items()}
        absolute[role] = count(component["metrics"][metric])
        require(
            sum(series[role].values()) == absolute[role], "Native series sum differs"
        )
    identity.extend(scopes.values())
    require(
        all(type(value) is str and value for value in identity[:4])
        and all(type(v) is str and v for scope in scopes.values() for v in scope),
        "Exact collector and role epochs are required",
    )
    return count(collector["monotonic_ns"]), identity, absolute, series


def progression(prior, current):
    require(
        prior[0] < current[0] and current[1] == prior[1],
        "Collector, artifact, scope, or live engine epoch changed",
    )
    for role in METRICS:
        old, new = prior[3][role], current[3][role]
        require(
            old.keys() == new.keys() and all(new[key] >= old[key] for key in old),
            "Native counter reset or series changed",
        )


def expect_idle(value):
    return all(
        type(value["components"][role]["metrics"].get(name)) is int
        and value["components"][role]["metrics"][name] == 0
        for role in METRICS
        for name in ("running", "waiting")
    )


def native_idle(native):
    p, d = native["prefill"], native["decode"]
    pool = d["native_pool"]
    return (
        p["stage"] == "prefill"
        and d["stage"] == "decode"
        and p["registered_ranks"] == 4
        and all(
            type(group.get(key)) is int and group[key] == 0
            for group, keys in (
                (p, ("active_leases", "ready_leases")),
                (d, ("active_leases", "ready_leases", "scheduled_imports")),
                (
                    pool,
                    ("active_grants", "awaiting_settlement", "grant_block_references"),
                ),
            )
            for key in keys
        )
    )


def guard_errors(raw, current, base, plan):
    limits = plan["resource_limits"]
    errors = []
    for role in METRICS:
        metrics = raw["components"][role]["metrics"]
        if metrics.get("healthy") is not True:
            errors.append(f"{role}_health_unavailable")
        waiting = metrics.get("waiting")
        if (
            type(waiting) is not int
            or not 0 <= waiting <= limits["max_waiting_per_role"]
        ):
            errors.append(f"{role}_wait_limit")
    host = count(current["host_available_bytes"])
    if host < limits["min_host_available_bytes"]:
        errors.append("host_available_below_plan")
    cards = current["device_memory"]
    require(
        set(cards) == set(limits["gpu_uuids"]) and len(cards) == 8,
        "Expected eight actual resource samples",
    )
    for value in cards.values():
        used, total = count(value["used_bytes"]), count(value["total_bytes"])
        require(total > 0 and used <= total, "Invalid GPU memory counters")
        if used / total > limits["max_device_used_fraction"]:
            errors.append("device_usage_above_plan")
    groups = current["cgroup_memory_events"]
    require(
        set(groups)
        == set(limits.get("memory_event_paths", {}))
        == set(base["cgroup_memory_events"]),
        "Resource sources changed",
    )
    for name, now in groups.items():
        past = base["cgroup_memory_events"][name]
        if now is None or past is None or now["inode"] != past["inode"]:
            errors.append("cgroup_oom_accounting_unavailable")
            continue
        rises = [count(now[key]) - count(past[key]) for key in ("oom", "oom_kill")]
        if any(rise < 0 for rise in rises):
            errors.append("cgroup_oom_counter_reset")
        elif any(rises):
            errors.append("cgroup_oom_observed")
    return sorted(set(errors))


def completed(ordinal, value):
    phase = value["phase"]
    require(
        phase == PHASES[ordinal % len(PHASES)], "Mixed workload phase order differs"
    )
    turns = value["settled_turns"]
    require(
        len(turns) == (1 if phase == "cold" else 2)
        and all(
            row["verified_ready_ranks"] == 4
            and count(row["prefill_api_prompt_tokens"])
            == count(row["ready_tokens_per_rank"]) + 1
            for row in turns
        ),
        "Completed phase lacks its exact four-rank READY boundary",
    )
    logical = sum(row["ready_tokens_per_rank"] for row in turns)
    api = sum(row["prefill_api_prompt_tokens"] for row in turns)
    restore = count(value["native_nvme_restore_output_tokens"])
    source = value["prefill_source_delta"]
    external = count(source["external_kv_transfer"])
    aggregate = value["checkpoint_aggregate_sha256"]
    out = sum(count(row["output_tokens"]) for row in turns) + restore
    require(
        api == count(value["prefill_api_prompt_tokens"])
        and out == count(value["output_tokens"])
        and source["local_cache_hit"] == 0
        and count(source["local_compute"]) + external == logical
        and (external == 0) == (phase == "cold")
        and (restore > 0) == (phase == "nvme")
        and (
            (phase != "nvme" and aggregate is None)
            or (
                type(aggregate) is str
                and len(aggregate) == 64
                and all(c in "0123456789abcdef" for c in aggregate)
            )
        ),
        "Completed phase disagrees with native source or API token accounting",
    )
    return {"phase": phase, "logical": logical, "api": api, "output": out}


def request_history(rows, start):
    prior = start
    parsed = []
    for index, row in enumerate(rows):
        now, end = count(row["started_ns"]), count(row["ended_ns"])
        require(
            type(row["ordinal"]) is int
            and row["ordinal"] == index
            and prior <= now < end,
            "C1 request ledger has a gap or overlapping records",
        )
        doc = row["request"]
        good = doc["status"] == "completed"
        parsed.append((now, end, completed(index, doc) if good else None))
        prior = end
    return parsed


def progress(rows, edge):
    chunk = [row for _, end, row in rows if end <= edge]
    selected = [row for row in chunk if row is not None]
    phases = Counter(row["phase"] for row in selected)
    return {
        "completed": len(selected),
        "failed": len(chunk) - len(selected),
        "completed_by_phase": {phase: phases[phase] for phase in PHASES},
        "prefill_api_tokens": sum(row["api"] for row in selected),
        "expected_native_logical_p_tokens": sum(row["logical"] for row in selected),
        "expected_native_d_output_tokens": sum(row["output"] for row in selected),
        "recorded_rows": len(chunk),
    }


def read_cutoff(path, start, hashes, rows):
    if path is None:
        return None
    cut = read(path)
    require(
        cut["schema"] == CUTOFF_SCHEMA
        and cut["kind"] == "authorized_30_minute_limit"
        and all(cut[key] == value for key, value in hashes.items())
        and count(cut["start_monotonic_ns"]) == start
        and count(cut["target_monotonic_ns"]) == start + TARGET * SECOND
        and count(cut["stop_monotonic_ns"]) >= cut["target_monotonic_ns"]
        and type(cut["driver_exited"]) is bool
        and count(cut["finished_client_rows"]) == len(rows)
        and cut["last_completion_monotonic_ns"] == rows[-1][1]
        and rows[-1][0] <= cut["stop_monotonic_ns"],
        "Cutoff does not bind the exact stopped client and original workload",
    )
    return cut


def build(args):
    directory = args.mixed_dir
    hashes = {
        "manifest_sha256": file_digest(args.manifest),
        "pair_sha256": file_digest(args.pair),
        "run_plan_sha256": file_digest(directory / "run-plan.json"),
    }
    run_plan = read(directory / "run-plan.json")
    plan = run_plan["plan"]
    wanted = plan["active_seconds"]
    require(
        wanted == 7200
        and plan["concurrency"] == 1
        and plan["max_failures"] == 0
        and run_plan["run_scope"] == "observed_local_native_lab_mixed"
        and run_plan["manifest_sha256"] == hashes["manifest_sha256"]
        and run_plan["sources_file_sha256"] == file_digest(args.sources)
        and type(plan["sample_seconds"]) is int
        and plan["sample_seconds"] > 0,
        "Expected the original, unchanged two-hour C1 mixed plan",
    )
    pair = read(args.pair)
    initial_raw = read(directory / "before.json")
    baseline = snapshot(initial_raw)
    original_native = read(directory / "before-native-status.json")
    baseline_resources = read(directory / "before-resources.json")
    require(
        initial_raw["release_manifest_sha256"] == hashes["manifest_sha256"]
        and initial_raw["deployment"] == pair["authority_engine_id"]
        and expect_idle(initial_raw)
        and native_idle(original_native)
        and not guard_errors(initial_raw, baseline_resources, baseline_resources, plan),
        "The original pair baseline was not measured ready and idle",
    )
    raw, ledger_lines = records(directory / "request-results.jsonl")
    require(raw, "The mixed client has no completed record to identify its start")
    start = count(raw[0]["started_ns"])
    ledger = request_history(raw, start)
    require(baseline[0] <= start, "Mixed requests precede the original baseline")
    cutoff = read_cutoff(args.cutoff, start, hashes, ledger)
    stop = cutoff["stop_monotonic_ns"] if cutoff else math.inf

    ticks, trend_lines = records(directory / "trend.jsonl")
    prior = baseline
    guard_through = baseline[0]
    guard_failure = []
    gap = 0
    live = []
    for index, row in enumerate(ticks):
        sampled = snapshot(row["snapshot"])
        if sampled[0] > stop:
            break
        progression(prior, sampled)
        gap = max(gap, sampled[0] - prior[0])
        prior = sampled
        errors = guard_errors(
            row["snapshot"], row["resources"], baseline_resources, plan
        )
        if errors:
            guard_failure = errors
            break
        guard_through = sampled[0]
        stats = row["client_counts"]
        client = {
            name: count(stats[name]) for name in ("started", "completed", "failed")
        }
        require(
            0 <= client["started"] - client["completed"] - client["failed"] <= 1,
            "C1 client counts did not come from the single in-flight slot",
        )
        live.append((sampled, sampled[0], "historical_trend", index + 1, client))

    drain = {
        "observed": False,
        "native_import_and_lease_drained": None,
        "owner_reference_counts_zero": None,
    }
    if cutoff is not None and cutoff["drain"]["observed"] is True:
        evidence = cutoff["drain"]
        end = evidence["snapshot"]
        sample = snapshot(end)
        progression(prior, sample)
        require(sample[0] >= stop, "Drain sample predates the operator cutoff")
        settled = (
            expect_idle(end)
            and native_idle(evidence["native"])
            and all(
                end["components"][role]["metrics"].get("healthy") is True
                for role in METRICS
            )
        )
        require(
            evidence["settled"] is settled, "Drain witness contradicts observed status"
        )
        drain = {
            **drain,
            "observed": True,
            "native_import_and_lease_drained": settled
            if cutoff["driver_exited"]
            else None,
            "snapshot_monotonic_ns": sample[0],
            "driver_exited": cutoff["driver_exited"],
        }
        resources = evidence.get("resources")
        if resources is not None:
            observed = guard_errors(end, resources, baseline_resources, plan)
            require(
                observed == sorted(evidence.get("guard_errors", [])),
                "Cutoff resource guard witness differs",
            )
            if not guard_failure and not observed:
                gap = max(gap, stop - prior[0])
                guard_through = sample[0]
            elif observed:
                guard_failure = observed
        if cutoff["driver_exited"]:
            final_progress = progress(ledger, sample[0])
            if final_progress["recorded_rows"] == len(ledger):
                live.append(
                    (
                        sample,
                        stop,
                        "cutoff_snapshot",
                        len(ticks),
                        {
                            "completed": final_progress["completed"],
                            "failed": final_progress["failed"],
                        },
                    )
                )

    candidates = []
    activity_budget = 2 * plan["sample_seconds"] * SECOND
    for sample, active_end, method, seen, counts in live:
        row = progress(ledger, sample[0])
        absolute = sample[2]
        logical = absolute["prefill"] - baseline[2]["prefill"]
        output = absolute["decode"] - baseline[2]["decode"]
        matched = (
            counts["failed"] == row["failed"] == 0
            and counts["completed"] == row["completed"]
            and logical == row["expected_native_logical_p_tokens"]
            and output == row["expected_native_d_output_tokens"]
        )
        other_activity = any(
            active_end - activity_budget <= ts[0] <= active_end
            and state.get("started", 0) - state["completed"] - state["failed"] == 1
            for ts, _, source, _, state in live
            if source == "historical_trend"
        )
        recent_completion = any(
            active_end - activity_budget <= ended <= active_end
            for _, ended, _ in ledger
        )
        admitted = [(began, ended) for began, ended, _ in ledger if began <= active_end]
        idle_gap = max(
            (
                now[0] - prior[1]
                for prior, now in zip(admitted, admitted[1:], strict=False)
            ),
            default=0,
        )
        candidates.append(
            {
                **row,
                "client_activity_observed_near_edge": other_activity
                or recent_completion,
                "max_completed_request_dispatch_gap_seconds": idle_gap / SECOND,
                "observed_dispatch_gaps_within_sample_budget": idle_gap
                <= activity_budget,
                "api_engine_counts_match": matched,
                "native_logical_p_tokens": logical,
                "native_d_output_tokens": output,
                "seconds": (active_end - start) / SECOND,
                "counter_snapshot_source": method,
                "counter_sample_ns": sample[0],
                "active_end_monotonic_ns": active_end,
                "trend_rows": seen,
            }
        )
    covered = [p for p in candidates if p["seconds"] >= TARGET]
    clean = not guard_failure and gap <= 2 * plan["sample_seconds"] * SECOND
    eligible = [
        item
        for item in covered
        if item["api_engine_counts_match"]
        and item["client_activity_observed_near_edge"]
        and item["observed_dispatch_gaps_within_sample_budget"]
        and all(item["completed_by_phase"].values())
        and clean
        and item["active_end_monotonic_ns"] <= guard_through
    ]
    chosen = eligible or covered or candidates
    edge = chosen[0] if eligible else chosen[-1] if chosen else None
    elapsed = None if cutoff is None else (stop - start) / SECOND
    return {
        "schema": SCHEMA,
        "status": (
            "observed_complete"
            if eligible
            else "unreconciled"
            if covered or elapsed is not None
            else "too_short"
        ),
        "original_plan": {"active_seconds": wanted, "concurrency": 1, **hashes},
        "target_seconds": TARGET,
        "first_request_monotonic_ns": start,
        "latest_counter_sample_ns": candidates[-1]["counter_sample_ns"]
        if candidates
        else None,
        "operator_cutoff_ns": cutoff["stop_monotonic_ns"] if cutoff else None,
        "operator_client_window": None
        if cutoff is None
        else {
            "seconds": elapsed,
            "driver_exited": cutoff["driver_exited"],
            **progress(ledger, stop),
        },
        "observed_window": edge,
        "resource_guard": {
            "last_guard_sample_ns": guard_through,
            "seconds_from_first_request": max(0, guard_through - start) / SECOND,
            "maximum_capture_gap_seconds": gap / SECOND,
            "errors": guard_failure,
            "scope": "observed_periodic_and_available_cutoff_samples",
        },
        "cutoff_drain": drain,
        "source_observation": {
            "collector_epoch": baseline[1][0],
            "role_epochs": dict(zip(METRICS, baseline[1][4:], strict=True)),
            "request_ledger_prefix_sha256": digest(
                b"".join(ledger_lines[: edge["recorded_rows"]])
            )
            if edge
            else None,
            "trend_prefix_sha256": digest(b"".join(trend_lines[: edge["trend_rows"]]))
            if edge
            else None,
            "cutoff_sha256": file_digest(args.cutoff) if args.cutoff else None,
        },
        "release_qualification": "not_evaluated_observed_short_window",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mixed-dir", "manifest", "pair", "sources"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cutoff", type=Path)
    try:
        result = build(parser.parse_args())
    except (OSError, TypeError, KeyError, ValueError) as error:
        result = {
            "schema": SCHEMA,
            "status": "invalid",
            "detail": f"{type(error).__name__}: {error}",
            "release_qualification": "not_evaluated_observed_short_window",
        }
    print(json.dumps(result, indent=2, allow_nan=False, sort_keys=True))
    return {"observed_complete": 0, "too_short": 2, "unreconciled": 3, "invalid": 4}[
        result["status"]
    ]


if __name__ == "__main__":
    raise SystemExit(main())
