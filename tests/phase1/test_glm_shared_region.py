# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import mmap
import os
import subprocess
import sys
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import kitepond
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion

JOIN = """
import json
import sys
sys.path.insert(0, sys.argv[1])
from kitepond.models.glm5next.shared_region import (
    NativeHiSparseMemfdRegion,
    NativeHiSparseRegionDescriptor,
)
descriptor = NativeHiSparseRegionDescriptor(**json.loads(sys.argv[2]))
region = NativeHiSparseMemfdRegion.join_received_fd(
    descriptor, int(sys.argv[3]), authority_engine_id="D-engine",
    authority_boot_id="D-boot", source_rank=2, layout_sha256="a" * 64,
)
try:
    assert not region.owner
    assert region.mapping[:4] == b"KITE"
    region.mapping[4:8] = b"POND"
finally:
    region.close()
assert "torch" not in sys.modules and "vllm" not in sys.modules
"""


class SharedRegionTest(unittest.TestCase):
    def setUp(self):
        self.region = NativeHiSparseMemfdRegion.create(
            mmap.PAGESIZE,
            authority_engine_id="D-engine",
            authority_boot_id="D-boot",
            source_rank=2,
            layout_sha256="a" * 64,
        )
        self.addCleanup(self.region.close)

    def test_new_process_joins_and_updates_the_same_pages(self):
        self.region.mapping[:4] = b"KITE"
        package_root = Path(kitepond.__file__).resolve().parents[1]
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                JOIN,
                str(package_root),
                json.dumps(asdict(self.region.descriptor)),
                str(self.region.fd),
            ],
            pass_fds=(self.region.fd,),
            check=True,
            timeout=10,
        )
        self.assertEqual(self.region.mapping[:8], b"KITEPOND")
        self.assertEqual(os.fstat(self.region.fd).st_size, mmap.PAGESIZE)

    def test_stale_descriptor_rejects_and_closes_only_received_fd(self):
        received = os.dup(self.region.fd)
        stale = replace(self.region.descriptor, authority_boot_id="old-boot")
        with self.assertRaisesRegex(ValueError, "different owner or layout"):
            NativeHiSparseMemfdRegion.join_received_fd(
                stale,
                received,
                authority_engine_id="D-engine",
                authority_boot_id="D-boot",
                source_rank=2,
                layout_sha256="a" * 64,
            )
        with self.assertRaises(OSError):
            os.fstat(received)
        self.assertEqual(os.fstat(self.region.fd).st_ino, stale.inode)
