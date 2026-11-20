# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Advance expired native GLM checkpoints from the scheduler owner loop."""

from collections.abc import Callable, Iterable
from itertools import islice
from pathlib import Path
from threading import get_ident
from time import monotonic_ns
from typing import Protocol

from kitepond.backends.fsl2 import BundleIdentity
from kitepond.core.policy import ColdJobScheduler, IdlePolicy, JobView, LaneBudget

from .cold_lifecycle import GlmColdCheckpoint, GlmColdRamOwner, GlmFSL2ColdAdapter


class GlmIdleRamOwner(GlmColdRamOwner, Protocol):
    def expired_verified_checkpoints(
        self, *, idle_cutoff_ns: int, limit: int
    ) -> Iterable[GlmColdCheckpoint]:
        """Inspect at most one bounded slice of the existing native directory.

        Yield current accepted RAM entries without taking new page references
        or refreshing business time. Rotate through larger directories in the
        native owner. Exclude demoted entries and entries the owner has deferred.
        """


class GlmIdleColdLoop:
    """Opt-in, single-owner tick for idle stores and explicit restore jobs."""

    def __init__(
        self,
        root: Path,
        owner: GlmIdleRamOwner,
        *,
        idle: IdlePolicy,
        foreground: LaneBudget,
        background: LaneBudget,
        max_jobs: int = 128,
        scan_limit: int = 32,
        clock: Callable[[], int] = monotonic_ns,
    ) -> None:
        if type(scan_limit) is not int or scan_limit <= 0:
            raise ValueError("Idle scan needs a positive candidate bound")
        self.owner = owner
        self.scheduler = ColdJobScheduler(
            GlmFSL2ColdAdapter(root, owner),
            idle=idle,
            foreground=foreground,
            background=background,
            max_jobs=max_jobs,
        )
        self.scan_limit, self.clock = scan_limit, clock
        self._thread = get_ident()
        self._last_tick = 0
        self._stores: dict[BundleIdentity, int] = {}

    def _admit(self, cutoff: int) -> None:
        stores = self.scheduler.observe_lane("STORE")
        restores = self.scheduler.observe_lane("RESTORE")
        reserve = max(
            0,
            restores.budget.max_active_jobs
            - restores.active_jobs
            - restores.queued_jobs,
        )
        lane = min(
            stores.budget.max_active_jobs - stores.active_jobs,
            (stores.budget.max_inflight_bytes - stores.charged_bytes)
            // stores.budget.per_job_bytes,
        )
        room = min(
            lane - stores.queued_jobs,
            stores.metadata_limit - stores.metadata_entries - reserve,
        )
        if room <= 0 or cutoff < 0:
            return
        entries = self.owner.expired_verified_checkpoints(
            idle_cutoff_ns=cutoff, limit=self.scan_limit
        )
        for checkpoint in islice(entries, self.scan_limit):
            if type(checkpoint) is not GlmColdCheckpoint:
                raise ValueError("Native idle scan returned an unverified handle")
            identity = checkpoint.identity
            if identity in self._stores or self.scheduler.has_job(checkpoint):
                continue
            job_id = self.scheduler.submit("STORE", checkpoint)
            self._stores[identity] = job_id
            room -= 1
            if room == 0:
                break

    def tick(self) -> tuple[JobView, ...]:
        """Advance on the scheduler owner, including when there are no requests."""
        if get_ident() != self._thread:
            raise RuntimeError("Cold jobs must advance on the native scheduler owner")
        now = self.clock()
        if type(now) is not int or now < self._last_tick:
            raise ValueError("Business idle clock must be monotonic")
        self._last_tick = now

        self._admit(now - self.scheduler.idle.idle_ns)
        self.scheduler.tick(now_ns=now)
        settled = []
        for identity, job_id in tuple(self._stores.items()):
            view = self.scheduler.view(job_id)
            if view.state == "PENDING":
                continue
            if view.state == "QUEUED":
                # A stale scan or busy native ref has started no I/O.
                # Leave retries to the next native scan instead of filling the queue.
                self.scheduler.cancel(job_id)
                self.scheduler.reap(job_id)
            else:
                settled.append(self.scheduler.reap(job_id))
            del self._stores[identity]
        return tuple(settled)
