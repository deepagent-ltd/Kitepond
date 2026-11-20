# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mmap
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import kitepond
from kitepond.models.glm5next.region_socket import RegionPublisher
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion

CHILD = """
import pathlib
import sys
sys.path.insert(0, sys.argv[1])
from kitepond.models.glm5next.region_socket import join_region
try:
    region = join_region(
        pathlib.Path(sys.argv[2]), authority_engine_id="D",
        authority_boot_id=sys.argv[3],
        source_rank=2, layout_sha256="b" * 64,
    )
except ValueError:
    assert sys.argv[3] == "wrong"
else:
    try:
        assert sys.argv[3] == "same" and region.mapping[:4] == b"same"
        region.mapping[4:9] = b"pages"
    finally:
        region.close()
"""


def test_child_process_attaches_exact_d_generation_without_inheriting_fds(tmp_path):
    directory = tmp_path / "socket"
    directory.mkdir(mode=0o700)
    region = NativeHiSparseMemfdRegion.create(
        mmap.PAGESIZE,
        authority_engine_id="D",
        authority_boot_id="same",
        source_rank=2,
        layout_sha256="b" * 64,
    )
    owner = RegionPublisher(directory, region)
    package = Path(kitepond.__file__).parents[1]
    region.mapping[:4] = b"same"
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            for boot in ("wrong", "same"):
                ready = executor.submit(owner.serve_one, accept_timeout=10)
                subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        "-B",
                        "-c",
                        CHILD,
                        str(package),
                        str(directory),
                        boot,
                    ],
                    check=True,
                    timeout=10,
                )
                assert ready.result() is (boot == "same")
        assert region.mapping[:9] == b"samepages"
    finally:
        owner.close()
        region.close()
    assert not owner.path.exists()
