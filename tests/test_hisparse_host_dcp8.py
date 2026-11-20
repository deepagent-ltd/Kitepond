# SPDX-License-Identifier: Apache-2.0
import fcntl
import json
import mmap
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from kitepond import hisparse_host as dp_provider
from kitepond import hisparse_host_dcp8 as provider
from test_hisparse_host import _modules


def _worker_test_boundaries(rank):
    torch = ModuleType("torch")
    torch.int8 = "int8"
    torch.frombuffer = lambda region, *, dtype: memoryview(region).cast("b")
    return (
        patch.dict("sys.modules", {"torch": torch}),
        patch.object(provider, "_owner_rank", return_value=rank),
        patch.object(provider, "_validate_source"),
    )


def _try_rank_lease(results, policy, rank):
    driver, topology, planner = _worker_test_boundaries(rank)
    with (
        patch.dict(os.environ, {provider.POLICY_ENV: policy}),
        driver,
        topology,
        planner,
    ):
        try:
            backing, release = provider.acquire(
                SimpleNamespace(instance_id="contender"), None, mmap.PAGESIZE
            )
        except RuntimeError:
            results.send(False)
        else:
            release()
            del backing
            results.send(True)


class DCP8HostBackingTest(unittest.TestCase):
    def setUp(self):
        path = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.policy = path / "dcp8.json"
        self._write_policy()
        self.enterContext(
            patch.dict(os.environ, {provider.POLICY_ENV: str(self.policy)})
        )
        for boundary in _worker_test_boundaries(0):
            self.enterContext(boundary)
        self.config = SimpleNamespace(instance_id="engine")

    def _write_policy(self, *, rank=mmap.PAGESIZE, total=8 * mmap.PAGESIZE):
        self.policy.write_text(
            json.dumps({"rank_bytes": rank, "total_bytes": total}),
            encoding="utf-8",
        )

    def test_total_and_per_rank_caps(self):
        self._write_policy(total=8 * mmap.PAGESIZE - 1)
        with self.assertRaisesRegex(ValueError, "must fit in total_bytes"):
            provider.acquire(self.config, None, mmap.PAGESIZE)
        self._write_policy()
        with self.assertRaisesRegex(ValueError, "rank budget"):
            provider.acquire(self.config, None, mmap.PAGESIZE + 1)
        self.assertEqual(list(self.policy.parent.glob("*.lock")), [])

    def test_same_rank_blocked_in_other_process_until_last_view_dies(self):
        backing, release = provider.acquire(self.config, None, mmap.PAGESIZE)
        alias = backing[:1]
        release()
        release()
        del backing

        def contender(rank):
            ctx = multiprocessing.get_context("spawn")
            results, sender = ctx.Pipe(duplex=False)
            worker = ctx.Process(
                target=_try_rank_lease, args=(sender, str(self.policy), rank)
            )
            worker.start()
            sender.close()
            try:
                self.assertTrue(results.poll(10), "lease contender timed out")
                allowed = results.recv()
            finally:
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
                results.close()
            self.assertEqual(worker.exitcode, 0)
            return allowed

        self.assertFalse(contender(0))
        self.assertTrue(contender(1))
        del alias
        self.assertTrue(contender(0))

    def test_mapping_failure_releases_rank(self):
        with (
            patch("kitepond.hisparse_host.mmap.mmap", side_effect=OSError("failed")),
            self.assertRaisesRegex(OSError, "failed"),
        ):
            provider.acquire(self.config, None, mmap.PAGESIZE)
        backing, release = provider.acquire(self.config, None, mmap.PAGESIZE)
        self.assertEqual(len(backing), mmap.PAGESIZE)
        release()


    def test_dp_dcp_same_policy_share_rank_budget(self):
        with (
            patch.dict(os.environ, {dp_provider.POLICY_ENV: str(self.policy)}),
            patch.dict("sys.modules", _modules()),
        ):
            lease = dp_provider.acquire(
                instance_id="dp-worker", dp_rank=0, size_bytes=mmap.PAGESIZE
            )
            alias = lease.buffer[:1]
            lease.release()
            del lease
            with self.assertRaisesRegex(RuntimeError, "already holds a host lease"):
                provider.acquire(self.config, None, mmap.PAGESIZE)

            del alias
            backing, release = provider.acquire(self.config, None, mmap.PAGESIZE)
            release()
            with self.assertRaisesRegex(RuntimeError, "already holds a host lease"):
                dp_provider.acquire(
                    instance_id="dp-worker", dp_rank=0, size_bytes=mmap.PAGESIZE
                )
            del backing
            dp_provider.acquire(
                instance_id="dp-worker", dp_rank=0, size_bytes=mmap.PAGESIZE
            ).release()

    def test_existing_legacy_lock_names_block_the_new_provider(self):
        for suffix in ("rank0", "dcp-rank0"):
            lock = self.policy.with_name(f"{self.policy.name}.{suffix}.lock")
            with lock.open("ab") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "already holds a host lease"):
                    provider.acquire(self.config, None, mmap.PAGESIZE)
            backing, release = provider.acquire(self.config, None, mmap.PAGESIZE)
            release()
            del backing
