# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise portable policy with one CPU ownership model."""

import unittest
from dataclasses import dataclass

from kitepond.core.contracts import Wait
from kitepond.core.foreground import ForegroundInterestTable
from kitepond.core.policy import (
    ColdJobScheduler,
    IdlePolicy,
    LaneBudget,
    NativeModelColdAdapter,
    NativeModelColdJob,
)

NS = 1_000_000_000
NOW = 1_800 * NS


@dataclass(eq=False)
class Handle:
    last_use_ns: int
    live_reads: int = 0
    on_disk: bool = False
    borrowed: bool = False


class OwnerJob(NativeModelColdJob):
    def __init__(self, mode, budget, owner, handle):
        super().__init__(mode, budget)
        self.owner = owner
        self.handle = handle
        self.state = "PENDING"
        self.cancel_requested = False

    def poll(self):
        return self.state

    def cancel(self):
        self.cancel_requested = True

    def finish(self, state):
        if self.state != "PENDING":
            raise RuntimeError("Already finished")
        self.state = state
        if state == "COMMITTED":
            self.handle.on_disk = True
        if state != "VERIFIED":
            self.handle.borrowed = False

    def reap(self):
        if self.state == "PENDING":
            raise RuntimeError("I/O still pending")
        self.owner.jobs.pop(self.handle)


class Owner(NativeModelColdAdapter):
    def __init__(self):
        self.jobs = {}
        self.ram_full = False

    def last_business_use_ns(self, handle):
        return handle.last_use_ns

    def start(self, handle, mode, budget):
        if handle.borrowed:
            raise RuntimeError("Duplicate native borrow")
        handle.borrowed = True
        job = OwnerJob(mode, budget, self, handle)
        self.jobs[handle] = job
        return job

    def begin_store(
        self, handle, *, observed_last_use_ns, idle_cutoff_ns, max_inflight_bytes
    ):
        if (
            handle.last_use_ns != observed_last_use_ns
            or handle.last_use_ns > idle_cutoff_ns
        ):
            return Wait("BUSINESS_TIME_CHANGED")
        if handle.live_reads:
            return Wait("NATIVE_REFERENCES_BUSY")
        return self.start(handle, "STORE", max_inflight_bytes)

    def begin_restore(self, handle, *, max_inflight_bytes):
        if not handle.on_disk:
            return Wait("COLD_MISS")
        if self.ram_full:
            return Wait("RAM_ADMISSION_WAIT")
        return self.start(handle, "RESTORE", max_inflight_bytes)


class ColdJobSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.owner = Owner()
        self.core = ColdJobScheduler(
            self.owner,
            idle=IdlePolicy(15),
            foreground=LaneBudget(1, 32, 32),
            background=LaneBudget(1, 16, 16),
            max_jobs=3,
        )

    def test_idle_store_and_restore_hold_one_native_owner(self):
        handle = Handle(NOW - 899 * NS)
        job = self.core.submit("STORE", handle)
        self.core.tick(now_ns=NOW)
        self.assertEqual(self.core.view(job).wait_reason, "BUSINESS_IDLE_WAIT")
        self.assertFalse(handle.borrowed)
        self.core.tick(now_ns=NOW + NS)
        self.assertTrue(handle.borrowed)
        self.assertEqual(self.core.usage("STORE"), (1, 16))
        self.owner.jobs[handle].finish("COMMITTED")
        self.core.tick(now_ns=NOW + NS)
        self.assertEqual(self.core.reap(job).state, "COMMITTED")

        self.owner.ram_full = True
        restore = self.core.submit("RESTORE", handle)
        self.core.tick(now_ns=NOW + NS)
        self.assertEqual(self.core.view(restore).wait_reason, "RAM_ADMISSION_WAIT")
        self.assertFalse(handle.borrowed)
        self.assertEqual(self.core.usage("RESTORE"), (0, 0))

        self.owner.ram_full = False
        self.core.tick(now_ns=NOW + NS)
        self.owner.jobs[handle].finish("VERIFIED")
        self.core.tick(now_ns=NOW + NS)
        self.assertEqual(self.core.reap(restore).state, "VERIFIED")
        self.assertTrue(handle.borrowed)
        self.assertEqual(handle.last_use_ns, NOW - 899 * NS)

    def test_inflight_cancel_waits_actual_completion(self):
        handle = Handle(0)
        job = self.core.submit("STORE", handle)
        self.core.tick(now_ns=NOW)
        view = self.core.cancel(job)
        self.assertEqual(view.state, "PENDING")
        self.assertTrue(self.owner.jobs[handle].cancel_requested)
        self.assertTrue(handle.borrowed)
        with self.assertRaisesRegex(RuntimeError, "completion"):
            self.core.reap(job)
        self.owner.jobs[handle].finish("CANCELLED")
        self.core.tick(now_ns=NOW)
        self.assertEqual(self.core.reap(job).state, "CANCELLED")
        self.assertFalse(handle.borrowed)
        self.assertEqual(self.owner.jobs, {})

    def test_live_references_block_store_until_owner_releases(self):
        handle = Handle(0, live_reads=1)
        job = self.core.submit("STORE", handle)
        self.core.tick(now_ns=NOW)
        self.assertEqual(self.core.view(job).wait_reason, "NATIVE_REFERENCES_BUSY")
        self.assertFalse(handle.borrowed)
        handle.live_reads = 0
        self.core.tick(now_ns=NOW)
        self.assertTrue(handle.borrowed)

    def test_restore_and_store_have_separate_limits(self):
        store = self.core.submit("STORE", Handle(0))
        restore = self.core.submit("RESTORE", Handle(0, on_disk=True))
        waiting = self.core.submit("STORE", Handle(0))
        self.core.tick(now_ns=NOW)
        self.assertTrue(self.core.view(store).native_started)
        self.assertTrue(self.core.view(restore).native_started)
        self.assertEqual(self.core.view(waiting).wait_reason, "COLD_IO_ADMISSION_WAIT")
        self.assertEqual(self.core.usage("STORE"), (1, 16))
        self.assertEqual(self.core.usage("RESTORE"), (1, 32))
        self.assertEqual(self.core.cancel(waiting).state, "CANCELLED")
        self.assertFalse(self.core.reap(waiting).native_started)

    def test_owner_failure_reaps_without_fabricating_success(self):
        handle = Handle(0, on_disk=True)
        restore = self.core.submit("RESTORE", handle)
        self.core.tick(now_ns=NOW)
        self.owner.jobs[handle].finish("FAILED")
        self.core.tick(now_ns=NOW)
        self.assertEqual(self.core.reap(restore).state, "FAILED")
        self.assertFalse(handle.borrowed)

    def test_duplicate_handle_and_clock_rollback_are_rejected(self):
        handle = Handle(0)
        self.core.submit("STORE", handle)
        with self.assertRaisesRegex(ValueError, "existing job"):
            self.core.submit("RESTORE", handle)
        self.core.tick(now_ns=NOW)
        with self.assertRaisesRegex(ValueError, "monotonic"):
            self.core.tick(now_ns=NOW - 1)
        self.assertTrue(handle.borrowed)


class ForegroundInterestTest(unittest.TestCase):
    def test_metadata_expiry_does_not_release_native_state(self):
        table = ForegroundInterestTable(limit=1, lease_ns=10)
        handle = Handle(0, borrowed=True)
        first = table.begin(handle, "probe", now_ns=100)
        again = table.begin(handle, "probe", now_ns=103)
        self.assertIs(first, again)
        wait = table.begin(Handle(0), "other", now_ns=104)
        self.assertEqual(wait, Wait("FOREGROUND_LOOKUP_ADMISSION_WAIT"))
        self.assertEqual(table.expire(now_ns=113), 1)
        self.assertTrue(handle.borrowed)
        self.assertIsNone(table.renew(first.token, now_ns=114))
