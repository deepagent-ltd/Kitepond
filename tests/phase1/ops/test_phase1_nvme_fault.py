# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
import stat
import tempfile
import unittest
from pathlib import Path

from kitepond.ops.wheel import sha256_file
from phase1_nvme_fault import metered_delta, mutate, native_idle, output_count


class FaultFileScopeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "clone.data"
        self.path.write_bytes(b"committed-payload")
        self.path.chmod(0o600)
        self.info = {
            "mode": 0o600,
            "payload_sha256": sha256_file(self.path),
            "payload_bytes": self.path.stat().st_size,
        }

    def check_recovered(self):
        self.assertEqual(sha256_file(self.path), self.info["payload_sha256"])
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_output_budget_is_bound_to_oracle_without_assuming_eight(self):
        oracle = {"expected_token_ids": list(range(32))}
        self.assertEqual(output_count(oracle, 32), 32)
        for expected in (8, 0, 129, True):
            with (
                self.subTest(expected=expected),
                self.assertRaisesRegex(ValueError, "differs from the sealed oracle"),
            ):
                output_count(oracle, expected)
        with self.assertRaises(ValueError):
            output_count({"expected_token_ids": [True]}, 1)

    def test_corruption_restored_after_async_cancellation(self):
        with (
            self.assertRaises(asyncio.CancelledError),
            mutate(self.path, self.info, "corrupt") as evidence,
        ):
            self.assertNotEqual(evidence["sha256_mutated"], evidence["sha256_before"])
            self.assertEqual(self.path.stat().st_size, self.info["payload_bytes"])
            raise asyncio.CancelledError()
        self.check_recovered()

    @unittest.skipIf(os.geteuid() == 0, "Unprivileged UID required for EACCES")
    def test_permission_scope_does_not_change_bytes_and_restores_access(self):
        with (
            self.assertRaises(asyncio.CancelledError),
            mutate(self.path, self.info, "unreadable") as evidence,
        ):
            self.assertEqual(evidence["local_errno"], 13)
            self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0)
            with self.assertRaises(PermissionError):
                self.path.open("rb")
            raise asyncio.CancelledError()
        self.check_recovered()

    def test_refuses_any_hardlinked_part(self):
        extra = self.path.with_name("hard-link.data")
        extra.hardlink_to(self.path)
        with (
            self.assertRaisesRegex(ValueError, "exact cloned private part"),
            mutate(self.path, self.info, "corrupt"),
        ):
            self.fail("Hard link was mutated")
        self.check_recovered()

    def test_forensic_zero_requires_same_native_epoch_scope(self):
        def capture(value, epoch):
            return {
                "decode": {
                    "scope": "D-only",
                    "epoch": epoch,
                    "metrics": {"generated_tokens_total": value},
                },
            }

        before = capture(8, "boot")
        self.assertEqual(metered_delta(before, capture(8, "boot")), 0)
        self.assertEqual(metered_delta(before, capture(16, "boot")), 8)
        for invalid in [capture(None, "boot"), capture(0, "boot"), capture(8, "next")]:
            with self.assertRaisesRegex(ValueError, "scope changed"):
                metered_delta(before, invalid)

    def test_native_drain_requires_zero_grants_not_only_idle_queue(self):
        sample = {"metrics": {"healthy": True, "running": 0, "waiting": 0}}
        owner = {
            "stage": "decode",
            "active_leases": 0,
            "ready_leases": 0,
            "scheduled_imports": 0,
            "native_pool": {
                "active_grants": 0,
                "awaiting_settlement": 0,
                "grant_block_references": 0,
            },
        }
        self.assertTrue(native_idle(sample, owner))
        owner["native_pool"]["awaiting_settlement"] = 1
        self.assertFalse(native_idle(sample, owner))
