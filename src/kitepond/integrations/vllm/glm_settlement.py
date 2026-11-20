# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read bounded GLM settlement state on the current scheduler owner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from time import monotonic_ns

SCHEMA = "kitepond.glm-owner-settlement.v1"
OPERATION = "settlement_status"
_MAX_CONTROL_LEDGER = 4096
_TERMINAL_NVME = frozenset(
    (
        "admitted",
        "cancelled",
        "capture_failed",
        "capture_wait",
        "committed",
        "failed",
        "miss",
    )
)


def _counts(**readings: int) -> dict:
    if any(type(value) is not int or value < 0 for value in readings.values()):
        raise ValueError("Invalid GLM owner settlement count")
    return {"state": "observed", "counts": readings}


def _omitted(enabled: bool) -> dict:
    return {"state": "unavailable" if enabled else "disabled", "counts": {}}


def _check_bound(*ledgers) -> None:
    if any(len(ledger) > _MAX_CONTROL_LEDGER for ledger in ledgers):
        raise ValueError("GLM owner settlement exceeds its control snapshot budget")


def _expired(rows, clock, deadline):
    return sum(deadline(row) <= clock for row in rows)


def _connection(scheduler, owner):
    # The exact declaration in this profile is two scheduler-side children.
    # Missing Mooncake state cannot be represented by a zero-length substitute.
    children = getattr(scheduler.connector, "sub_connectors", ())
    if len(children) != 2 or children[0] is not owner:
        return _omitted(True)
    native = getattr(children[1], "connector_scheduler", None)
    queues = (
        getattr(native, "_reqs_need_recv", None),
        getattr(native, "_reqs_need_send", None),
        getattr(native, "_reqs_not_processed", None),
    )
    if not all(isinstance(q, (dict, set)) for q in queues):
        return _omitted(True)
    return _counts(
        hisparse_grants=len(owner._pending),
        hisparse_d_routed=len(owner._routed or ()),
        hisparse_p_routed=len(owner._routed_prefill or ()),
        mooncake_recv_queued=len(queues[0]),
        mooncake_send_queued=len(queues[1]),
        mooncake_not_processed=len(queues[2]),
    )


def _prefill(scheduler, now):
    state = scheduler._cold_prefill_requests
    collector = scheduler._cold_prefill_collector
    _check_bound(state)
    cold = _counts(
        requests=len(state),
        expired_requests=_expired(
            state.values(), now, lambda row: row.offer.arm_deadline_ns
        ),
        ready_requests=sum(row.receipts is not None for row in state.values()),
    )
    if collector is None:
        ready = _omitted(True)
    else:
        rows = collector._pending
        _check_bound(rows)
        ready = _counts(
            attempts=len(rows),
            expired_attempts=_expired(rows.values(), now, lambda row: row.deadline_ns),
        )

    authorizations = getattr(scheduler, "_authorized_warm", None)
    warm_native = getattr(scheduler, "_warm_returns", None)
    submissions = getattr(scheduler, "_warm_submissions", None)
    if (
        isinstance(authorizations, dict)
        and isinstance(warm_native, dict)
        and isinstance(submissions, dict)
    ):
        _check_bound(authorizations, warm_native, submissions)
        warm = _counts(
            authorized=len(authorizations),
            expired_authorized=_expired(
                authorizations.values(), now, lambda row: row.offer.arm_deadline_ns
            ),
            native_destinations=len(warm_native),
            submitted=len(submissions),
        )
    else:
        configured = (
            "kitepond_glm_prefill_return_boot_id"
            in scheduler.vllm_config.additional_config
        )
        warm = _omitted(configured)
    return {"forward": cold, "ready_collector": ready, "warm": warm}


def _d_forward(scheduler, now):
    state = scheduler._cold_decode_offered
    admission = scheduler._cold_decode_admission
    _check_bound(state, admission._aborts)
    return _counts(
        grants=len(state),
        ready=len(scheduler._cold_decode_verified),
        staged=len(scheduler._cold_decode_staged),
        expired_grants=_expired(
            state.values(), now, lambda offer: offer.arm_deadline_ns
        ),
        abort_quarantine=len(admission._aborts),
        mla_decisions=len(scheduler._cold_mla_decisions),
    )


def _d_completion(scheduler, owner):
    if getattr(scheduler, "_grant_completion_owner", None) is not owner:
        return _omitted(True)
    proofs = scheduler._native_decode_proofs
    settled = scheduler._settled_decode_imports
    _check_bound(proofs, settled)
    return _counts(
        grant_proofs_pending=len(proofs),
        retained_settled_import_bindings=len(settled),
    )


def _terminal(scheduler, now):
    terminal = getattr(scheduler, "_terminal_checkpoints", None)
    enabled = (
        scheduler.vllm_config.additional_config.get("kitepond_glm_native_terminal")
        is True
    )
    if terminal is None:
        return _omitted(enabled)
    terminal._require_owner()
    holds = terminal._active
    armed = terminal._armed
    _check_bound(holds, armed)
    return _counts(
        armed=len(armed),
        expired_armed=_expired(
            (entry[1] for entry in armed.values()),
            now,
            lambda identity: identity.deadline_monotonic_ns,
        ),
        holds=len(holds),
        block_references=sum(
            len(blocks) for hold in holds.values() for _, blocks in hold.refs
        ),
        read_permits=sum(hold.permit is not None for hold in holds.values()),
        closing=sum(hold.closing for hold in holds.values()),
        expired_holds=_expired(
            holds.values(), now, lambda hold: hold.lease.identity.deadline_monotonic_ns
        ),
        host_writes_pending=len(scheduler._pending_host_writes),
        captured_frames=len(scheduler._terminal_frames),
        connector_block_release_pending=sum(
            not ready for ready in scheduler._terminal_native_can_release.values()
        ),
    )


def _d_warm(scheduler, now):
    warm = getattr(scheduler, "_warm_decode_returns", None)
    configured = (
        "kitepond_glm_terminal_return" in scheduler.vllm_config.additional_config
    )
    if warm is None:
        return _omitted(configured)
    warm._require_owner()
    intents = scheduler._warm_decode_intents
    _check_bound(intents, warm._sources, warm._returns)
    return _counts(
        intents=len(intents),
        expired_intents=_expired(
            intents.values(), now, lambda row: row.deadline_monotonic_ns
        ),
        sources=len(warm._sources),
        transfer_routes=len(warm._returns),
        closed_routes_waiting_retire=sum(
            row.complete for row in warm._returns.values()
        ),
    )


def _d_cold(scheduler, now):
    runtime = getattr(scheduler, "_native_glm_cold_store", None)
    operator = getattr(scheduler, "_native_glm_nvme_operator", None)
    options = scheduler.vllm_config.additional_config
    if runtime is None:
        unavailable = _omitted("kitepond_glm_cold" in options)
        return {
            "cold_ram": unavailable,
            "fsl2_io": unavailable.copy(),
            "nvme_operations": _omitted("kitepond_glm_nvme_control" in options),
        }
    runtime._require_owner()
    rows = runtime.directory._entries
    _check_bound(rows)
    ram = _counts(
        resident_entries=sum(row.resident is not None for row in rows.values()),
        resident_block_references=sum(
            len(row.resident.blocks)
            for row in rows.values()
            if row.resident is not None
        ),
        outstanding_readers=sum(len(row.readers) for row in rows.values()),
        directory_io_entries=sum(row.io is not None for row in rows.values()),
    )
    store, restore = (
        runtime.scheduler.observe_lane(name) for name in ("STORE", "RESTORE")
    )
    io = _counts(
        store_active=store.active_jobs,
        store_queued=store.queued_jobs,
        store_charged_bytes=store.charged_bytes,
        restore_active=restore.active_jobs,
        restore_queued=restore.queued_jobs,
        restore_charged_bytes=restore.charged_bytes,
    )
    if operator is None:
        operations = _omitted("kitepond_glm_nvme_control" in options)
    else:
        operator._require_owner()
        _check_bound(operator._operations)
        active = tuple(
            row
            for row in operator._operations.values()
            if row.stage not in _TERMINAL_NVME
            or row.job_id is not None
            or row.cold_admission is not None
            or row.request is not None
        )
        operations = _counts(
            active=len(active),
            expired_active=_expired(active, now, lambda row: row.body["deadline_ns"]),
            cancel_requested=sum(row.cancel for row in active),
            cold_admissions=sum(row.cold_admission is not None for row in active),
        )
    return {"cold_ram": ram, "fsl2_io": io, "nvme_operations": operations}


def observe_glm_owner_settlement(scheduler, owner) -> dict:
    """No collect, tick, page allocation, native I/O wait or expiry processing."""
    owner._require_scheduler_owner()
    stage = owner.stage
    options = scheduler.vllm_config.additional_config
    peers = options.get("kitepond_glm_window")
    if not isinstance(peers, Mapping) or stage not in ("prefill", "decode"):
        raise ValueError("Scoped GLM settlement requires its native role identity")
    engine, boot = (
        ("p_engine_id", "p_boot_id")
        if stage == "prefill"
        else ("d_engine_id", "d_boot_id")
    )
    if (
        not all(
            type(peers.get(key)) is str and 0 < len(peers[key]) <= 128
            for key in (engine, boot)
        )
        or peers.get("d_engine_id") != owner._engine_id
        or peers.get("d_boot_id") != owner._boot_id
    ):
        raise ValueError("GLM settlement role cannot change engine or boot")
    now = monotonic_ns()
    parts = {"native_scheduler_connections": _connection(scheduler, owner)}
    if stage == "prefill":
        parts.update(_prefill(scheduler, now))
    else:
        parts.update(
            host_authority=_counts(**asdict(owner.authority.observe())),
            forward=_d_forward(scheduler, now),
            native_decode_completion=_d_completion(scheduler, owner),
            terminal=_terminal(scheduler, now),
            warm=_d_warm(scheduler, now),
        )
        parts.update(_d_cold(scheduler, now))
    # Worker event-loop state is independent of the scheduler metadata queues.
    # A closed file or zero local queues cannot assert those rank windows settled.
    parts["native_rank_transport"] = _omitted(True)
    return {
        "schema": SCHEMA,
        "role": stage,
        "engine_id": peers[engine],
        "boot_id": peers[boot],
        "authority_engine_id": owner._engine_id,
        "authority_boot_id": owner._boot_id,
        "observed_monotonic_ns": now,
        "components": parts,
    }
