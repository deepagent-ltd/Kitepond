# SPDX-License-Identifier: Apache-2.0
import ctypes
import json
import mmap
import multiprocessing
import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from kitepond.hisparse_host import POLICY_ENV, acquire


@dataclass(frozen=True)
class _Lease:
    buffer: object
    release: object


def _modules():
    runtime = ModuleType("vllm.v1.hisparse.runtime")
    runtime.HiSparseHostMemoryLease = _Lease
    torch = ModuleType("torch")
    torch.int8 = "int8"
    torch.frombuffer = lambda region, *, dtype: memoryview(region).cast("b")
    return {runtime.__name__: runtime, torch.__name__: torch}


def _attempt_acquire(queue, policy, dp_rank):
    with (
        patch.dict(os.environ, {POLICY_ENV: policy}),
        patch.dict("sys.modules", _modules()),
    ):
        try:
            lease = acquire(
                instance_id="other-worker", dp_rank=dp_rank, size_bytes=mmap.PAGESIZE
            )
        except RuntimeError:
            queue.put(False)
        else:
            lease.release()
            queue.put(True)


class HostBackingTest(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.policy = Path(directory) / "host.json"
        self._write_policy()
        self.enterContext(patch.dict(os.environ, {POLICY_ENV: str(self.policy)}))
        self.enterContext(patch.dict("sys.modules", _modules()))

    def _write_policy(self, *, rank_bytes=mmap.PAGESIZE, total_bytes=8 * mmap.PAGESIZE):
        self.policy.write_text(
            json.dumps({"rank_bytes": rank_bytes, "total_bytes": total_bytes})
        )

    def _acquire(self, rank=0, size=mmap.PAGESIZE):
        return acquire(instance_id="engine", dp_rank=rank, size_bytes=size)

    def test_opt_in_and_budget_validation(self):
        with patch.dict(os.environ):
            del os.environ[POLICY_ENV]
            with self.assertRaisesRegex(ValueError, "absolute policy file"):
                self._acquire()
        self._write_policy(total_bytes=7 * mmap.PAGESIZE)
        with self.assertRaisesRegex(ValueError, "must fit in total_bytes"):
            self._acquire()
        self._write_policy()
        with self.assertRaisesRegex(ValueError, "rank budget"):
            self._acquire(size=2 * mmap.PAGESIZE)
        with self.assertRaisesRegex(ValueError, "DP rank"):
            self._acquire(rank=8)
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            self._acquire(size=mmap.PAGESIZE - 1)

    def test_exact_alignment_and_independent_backing(self):
        leases = [self._acquire(rank) for rank in range(8)]
        buffers = [lease.buffer for lease in leases]
        addresses = [ctypes.addressof(ctypes.c_byte.from_buffer(buf)) for buf in buffers]
        self.assertEqual(len(set(addresses)), 8)
        for rank, (address, buf) in enumerate(zip(addresses, buffers)):
            self.assertEqual(address % mmap.PAGESIZE, 0)
            self.assertEqual(buf.nbytes, mmap.PAGESIZE)
            self.assertEqual(buf[0], 0)
            buf[0] = rank + 1
        self.assertEqual([buf[0] for buf in buffers], list(range(1, 9)))
        for lease in leases:
            lease.release()

    def test_lock_remains_until_alias_dies(self):
        lease = self._acquire()
        alias = lease.buffer[:1]
        lease.release()
        lease.release()
        del lease
        with self.assertRaisesRegex(RuntimeError, "already holds a host lease"):
            self._acquire()
        del alias
        self._acquire().release()

    def test_duplicate_request_from_separate_worker(self):
        ctx = multiprocessing.get_context("spawn")
        lease = self._acquire()

        def contender():
            results = ctx.Queue()
            worker = ctx.Process(
                target=_attempt_acquire,
                args=(results, str(self.policy), 0),
            )
            worker.start()
            try:
                acquired = results.get(timeout=10)
            finally:
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
                results.close()
            self.assertEqual(worker.exitcode, 0)
            return acquired

        self.assertFalse(contender())
        lease.release()
        del lease
        self.assertTrue(contender())

    def test_mapping_failure_releases_rank(self):
        with patch("kitepond.hisparse_host.mmap.mmap", side_effect=OSError("failed")):
            with self.assertRaisesRegex(OSError, "failed"):
                self._acquire()
        self._acquire().release()
