# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise the genuine LMCache FSL2 buffered writer and rollback on CPU."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from phase1_fsl2_partial_hook import archive_counts, create_arm, install, read
from phase1_partial_write_probe import validate_failure


class PartialFSL2Test(unittest.TestCase):
    def test_actual_first_rename_second_staged_then_failed_and_repaired(self):
        import torch
        from kitepond.backends.fsl2 import (
            BundleIdentity,
            BundlePlan,
            ColdSnapshotLease,
            ColdStorageAdapter,
            MemoryBorrow,
            NativeColdAuthority,
            PartSpec,
            RamRestoreLease,
        )
        from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj

        specs = (
            PartSpec("mla-part", 0, 0, (32768,), "torch.uint8", 32768),
            PartSpec("state-part", 0, 1, (16384,), "torch.uint8", 16384),
        )
        plan = BundlePlan(
            "fsl2-real-cpu-test", "b" * 64, tuple(s.role for s in specs), specs
        )
        fresh = BundleIdentity("partial", "a" * 64, "c" * 64, 1, 19)
        next_gen = BundleIdentity("partial", "a" * 64, "c" * 64, 2, 19)

        def memory(spec, value):
            tensor = torch.full(spec.shape, value, dtype=torch.uint8)
            return TensorMemoryObj(
                tensor,
                MemoryObjMetadata(
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    address=tensor.data_ptr(),
                    phy_size=tensor.nbytes,
                    ref_count=1,
                ),
                parent_allocator=None,
            )

        class Authority(NativeColdAuthority):
            def __init__(self):
                self.source = {s: memory(s, i + 7) for i, s in enumerate(specs)}
                self.target = {s: memory(s, 0) for s in specs}
                self.store, self.restore = (
                    ColdSnapshotLease(self, object()),
                    RamRestoreLease(self, object()),
                )
                self.outcomes = []

            def check_store(self, lease, identity, current_plan):
                assert (
                    lease is self.store
                    and identity in (fresh, next_gen)
                    and plan == current_plan
                )

            def borrow_source(self, lease, part):
                return MemoryBorrow(self, self.source[part])

            def finish_store(self, lease):
                assert lease is self.store and all(
                    m.get_ref_count() == 1 for m in self.source.values()
                )
                self.outcomes.append("store")

            def abort_store(self, lease):
                assert lease is self.store and all(
                    m.get_ref_count() == 1 for m in self.source.values()
                )
                self.outcomes.append("abort")

            def check_restore(self, lease, identity, current_plan):
                assert (
                    lease is self.restore
                    and identity == next_gen
                    and plan == current_plan
                )

            def borrow_destination(self, lease, part):
                return MemoryBorrow(self, self.target[part])

            def finish_restore(self, lease, verified):
                assert lease is self.restore and verified
                self.outcomes.append("restore")

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            root, evidence = work / "store", work / "fault"
            root.mkdir(mode=0o700)
            evidence.mkdir(mode=0o700)
            arm = create_arm(root, evidence, "cpu-file-fixture")
            undo = install(arm)
            authority = Authority()
            writer = ColdStorageAdapter(
                root / "rank-0", authority, max_inflight_bytes=2**20
            )
            helper = Path(__file__).with_name("phase1_fsl2_partial_hook.py")
            watcher = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(helper),
                    "--observe",
                    str(arm),
                    "--timeout",
                    "10",
                ]
            )
            try:
                failed = writer.begin_store(fresh, plan, authority.store)
                self.assertEqual(writer.wait(failed, 12), "FAILED")
                self.assertEqual(watcher.wait(timeout=5), 0)
                self.assertEqual(authority.outcomes, ["abort"])
                proof = read(evidence / "injected.json")
                validate_failure(
                    "cpu-file-fixture", proof, ["captured", "demoting", "failed"]
                )
                self.assertEqual(proof["first"]["bytes"], 32768)
                self.assertEqual(proof["second_temporary"]["bytes"], 16384)
                self.assertIsNone(writer.discover(fresh, plan))
                self.assertFalse(any(archive_counts(root).values()))
                self.assertEqual(writer.inflight_bytes, 0)

                healed = writer.begin_store(next_gen, plan, authority.store)
                self.assertEqual(writer.wait(healed, 10), "COMMITTED")
                self.assertIsNone(writer.discover(fresh, plan))
                lookup = writer.begin_restore(next_gen, plan, authority.restore)
                self.assertEqual(writer.wait(lookup, 10), "VERIFIED")
                self.assertEqual(authority.outcomes, ["abort", "store", "restore"])
                for spec in specs:
                    self.assertTrue(
                        torch.equal(
                            authority.target[spec].raw_tensor,
                            authority.source[spec].raw_tensor,
                        )
                    )
                print(
                    "REAL_FSL2_PARTIAL_ROLLBACK",
                    proof["first"]["bytes"],
                    proof["second_temporary"]["bytes"],
                    failed.state,
                    healed.state,
                    lookup.state,
                )
            finally:
                if watcher.poll() is None:
                    watcher.terminate()
                    watcher.wait(timeout=5)
                self.assertTrue(writer.close(10))
                undo()
        self.assertFalse(torch.cuda.is_initialized())
