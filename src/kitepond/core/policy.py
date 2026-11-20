# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded cold-job admission; the model owns RAM and native I/O leases."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from kitepond.core.contracts import Wait

Mode = Literal["STORE", "RESTORE"]
IDLE_MINUTES = frozenset({15, 30, 45, 60, 120, 180, 240})
_TERMINAL = {
    "STORE": frozenset({"COMMITTED", "FAILED", "CANCELLED"}),
    "RESTORE": frozenset({"VERIFIED", "MISS", "FAILED", "CANCELLED"}),
}


@dataclass(frozen=True)
class IdlePolicy:
    idle_minutes: int = 60

    def __post_init__(self) -> None:
        if type(self.idle_minutes) is not int or self.idle_minutes not in IDLE_MINUTES:
            raise ValueError("Unsupported business idle deadline")

    @property
    def idle_ns(self) -> int:
        return self.idle_minutes * 60 * 1_000_000_000


@dataclass(frozen=True)
class LaneBudget:
    """I/O staging grants, not the size of the persistent or RAM cache."""

    max_active_jobs: int
    max_inflight_bytes: int
    per_job_bytes: int

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not int or value <= 0
                for value in (
                    self.max_active_jobs,
                    self.max_inflight_bytes,
                    self.per_job_bytes,
                )
            )
            or self.per_job_bytes > self.max_inflight_bytes
        ):
            raise ValueError("Invalid cold I/O lane budget")

    def has_credit(self, active_jobs: int, charged_bytes: int) -> bool:
        return (
            active_jobs < self.max_active_jobs
            and charged_bytes + self.per_job_bytes <= self.max_inflight_bytes
        )


class NativeModelColdJob(ABC):
    """Opaque aggregate over actual native leases and completion results.

    A GLM job may contain four rank-local FSL2 tickets. The model must bound
    their total live staging by this grant, including verification scratch.
    VERIFIED means RAM bytes verified, never GPU/import execution readiness.
    """

    def __init__(self, mode: Mode, io_budget_bytes: int) -> None:
        if mode not in _TERMINAL or type(io_budget_bytes) is not int:
            raise ValueError("Invalid native model cold job")
        if io_budget_bytes <= 0:
            raise ValueError("Invalid native model cold job byte grant")
        self.mode, self.io_budget_bytes = mode, io_budget_bytes

    @abstractmethod
    def poll(self) -> str:
        """Return PENDING or actual terminal state; do not synthesize completion."""

    @abstractmethod
    def cancel(self) -> None:
        """Request cancellation without dropping native I/O/read references."""

    @abstractmethod
    def reap(self) -> None:
        """Retire only this completed job's model-owned control bookkeeping."""


class NativeModelColdAdapter(ABC):
    """Model-owned eligibility and same-pool leases, not a second directory.

    begin methods must return a job once any native I/O starts. Wait/exception
    means no new I/O or acquired lease remains. The model atomically validates
    business generation/time and native refs; a caller-supplied active flag is
    never authority. RAM-full restore returns Wait before NVMe I/O starts.
    """

    @abstractmethod
    def last_business_use_ns(self, handle: object) -> int:
        """Actual business acquire/use/release time on CLOCK_MONOTONIC."""

    @abstractmethod
    def begin_store(
        self,
        handle: object,
        *,
        observed_last_use_ns: int,
        idle_cutoff_ns: int,
        max_inflight_bytes: int,
    ) -> NativeModelColdJob | Wait:
        """Recheck time unchanged, then issue native eligible snapshot leases."""

    @abstractmethod
    def begin_restore(
        self, handle: object, *, max_inflight_bytes: int
    ) -> NativeModelColdJob | Wait:
        """Reserve actual same-pool RAM before starting a native restore job."""


@dataclass(frozen=True)
class JobView:
    job_id: int
    mode: Mode
    state: str
    wait_reason: str | None
    cancel_requested: bool
    native_started: bool
    io_budget_bytes: int


@dataclass(frozen=True)
class ColdLaneView:
    budget: LaneBudget
    active_jobs: int
    charged_bytes: int
    queued_jobs: int
    metadata_entries: int
    metadata_limit: int

    @property
    def capacity(self) -> str:
        if self.metadata_entries >= self.metadata_limit:
            return "metadata_full"
        if not self.budget.has_credit(self.active_jobs, self.charged_bytes):
            return "lane_full"
        return "available"


@dataclass
class _Entry:
    job_id: int
    mode: Mode
    handle: object
    state: str = "QUEUED"
    wait_reason: str | None = None
    cancel_requested: bool = False
    native: NativeModelColdJob | None = None
    granted_bytes: int = 0
    operator_idle: bool = False


class ColdJobScheduler:
    """Owner-loop driven admission; no threads, RAM pool, scanner or allocator.

    Only queued/active/completed job metadata is held until explicit reap.
    Scans, I/O and cache hits do not write or refresh business idle time.
    """

    def __init__(
        self,
        model: NativeModelColdAdapter,
        *,
        idle: IdlePolicy,
        foreground: LaneBudget,
        background: LaneBudget,
        max_jobs: int = 128,
    ) -> None:
        if not isinstance(model, NativeModelColdAdapter):
            raise TypeError("Cold scheduler requires a model-owned adapter")
        if type(max_jobs) is not int or max_jobs <= 0:
            raise ValueError("Invalid bounded cold-job metadata capacity")
        self.model, self.idle = model, idle
        self.budgets = {"RESTORE": foreground, "STORE": background}
        self.max_jobs = max_jobs
        self._entries: dict[int, _Entry] = {}
        self._next_id = 1
        self._last_tick_ns = 0

    def has_job(self, handle: object) -> bool:
        return any(entry.handle is handle for entry in self._entries.values())

    def submit(self, mode: Mode, handle: object, *, operator_idle: bool = False) -> int:
        if (
            mode not in _TERMINAL
            or handle is None
            or type(operator_idle) is not bool
            or (operator_idle and mode != "STORE")
        ):
            raise ValueError("Invalid model cold job")
        if len(self._entries) >= self.max_jobs:
            raise BufferError("Cold-job metadata admission is full")
        if self.has_job(handle):
            raise ValueError("Reap the existing job for this opaque model handle")
        job_id = self._next_id
        self._next_id += 1
        self._entries[job_id] = _Entry(
            job_id, mode, handle, operator_idle=operator_idle
        )
        return job_id

    def view(self, job_id: int) -> JobView:
        entry = self._entries[job_id]
        return JobView(
            entry.job_id,
            entry.mode,
            entry.state,
            entry.wait_reason,
            entry.cancel_requested,
            entry.native is not None,
            entry.granted_bytes,
        )

    def usage(self, mode: Mode) -> tuple[int, int]:
        if mode not in _TERMINAL:
            raise ValueError("Invalid cold I/O lane")
        active = [
            entry
            for entry in self._entries.values()
            if entry.mode == mode and entry.state == "PENDING"
        ]
        return len(active), sum(entry.granted_bytes for entry in active)

    def observe_lane(self, mode: Mode) -> ColdLaneView:
        """Read lane credits and queue metadata without running model callbacks."""
        active, charged = self.usage(mode)
        queued = sum(
            entry.mode == mode and entry.state == "QUEUED"
            for entry in self._entries.values()
        )
        return ColdLaneView(
            self.budgets[mode],
            active,
            charged,
            queued,
            len(self._entries),
            self.max_jobs,
        )

    def cancel(self, job_id: int) -> JobView:
        entry = self._entries[job_id]
        entry.cancel_requested = True
        if entry.state == "QUEUED":
            entry.state, entry.wait_reason = "CANCELLED", None
        elif entry.state == "PENDING":
            assert entry.native is not None
            entry.native.cancel()
        return self.view(job_id)

    def reap(self, job_id: int) -> JobView:
        entry = self._entries[job_id]
        if entry.state not in _TERMINAL[entry.mode]:
            raise RuntimeError("Actual native completion is still pending")
        if entry.native is not None:
            entry.native.reap()
        result = self.view(job_id)
        del self._entries[job_id]
        return result

    def _poll(self, entry: _Entry) -> None:
        assert entry.native is not None
        state = entry.native.poll()
        if state != "PENDING" and state not in _TERMINAL[entry.mode]:
            raise RuntimeError("Invalid native model completion state")
        entry.state = state

    def tick(self, *, now_ns: int | None = None) -> None:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if type(now_ns) is not int or now_ns < self._last_tick_ns:
            raise ValueError("Business idle clock must be monotonic")
        self._last_tick_ns = now_ns
        for entry in self._entries.values():
            if entry.state == "PENDING":
                self._poll(entry)
        # Foreground is considered first, with an independent grant and job cap.
        for mode in ("RESTORE", "STORE"):
            budget = self.budgets[mode]
            for entry in self._entries.values():
                if entry.mode != mode or entry.state != "QUEUED":
                    continue
                active, charged = self.usage(mode)
                if not budget.has_credit(active, charged):
                    entry.wait_reason = "COLD_IO_ADMISSION_WAIT"
                    continue
                if mode == "STORE":
                    last_use = self.model.last_business_use_ns(entry.handle)
                    if type(last_use) is not int or not 0 <= last_use <= now_ns:
                        raise ValueError("Invalid model business monotonic timestamp")
                    cutoff = (
                        now_ns if entry.operator_idle else now_ns - self.idle.idle_ns
                    )
                    if last_use > cutoff:
                        entry.wait_reason = "BUSINESS_IDLE_WAIT"
                        continue
                    native = self.model.begin_store(
                        entry.handle,
                        observed_last_use_ns=last_use,
                        idle_cutoff_ns=cutoff,
                        max_inflight_bytes=budget.per_job_bytes,
                    )
                else:
                    native = self.model.begin_restore(
                        entry.handle, max_inflight_bytes=budget.per_job_bytes
                    )
                if isinstance(native, Wait):
                    entry.wait_reason = native.reason
                    continue
                # Retain the returned native job before any contract checks or poll.
                # A broken adapter cannot make in-flight I/O disappear from usage.
                entry.native = native
                entry.state, entry.wait_reason = "PENDING", None
                entry.granted_bytes = budget.per_job_bytes
                if (
                    not isinstance(native, NativeModelColdJob)
                    or native.mode != mode
                    or native.io_budget_bytes > budget.per_job_bytes
                ):
                    raise RuntimeError("Native model job violated its granted lane")
                self._poll(entry)
