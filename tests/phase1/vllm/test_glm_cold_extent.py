# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mmap
import os
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from time import monotonic, sleep
from types import SimpleNamespace

import pytest
import torch
from kitepond.core.contracts import Wait
from kitepond.integrations.vllm import glm_host_backing as host
from kitepond.integrations.vllm.native_glm_cold_provider import NativeGlmRankMemory
from kitepond.models.glm5next.host_cold_extent import ColdExtentBudget

from vllm.v1.hisparse import runtime
from vllm.v1.kv_cache_interface import KVCacheGroupRole, MLAAttentionSpec

PAGE = mmap.PAGESIZE


def geometry(directory, role, *, boot="boot", enabled=True):
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=528,
        dtype=torch.uint8,
        state_content_bytes=528,
    )
    # The trailing byte exercises the same native page rounding on both paths.
    size = spec.page_size_bytes + 1
    options = {
        "role": role,
        "socket_directory": str(directory),
        "authority_engine_id": "D",
        "authority_boot_id": boot,
        "model_sha256": "e" * 64,
    }
    padded = (size + PAGE - 1) // PAGE * PAGE
    if enabled:
        options["cold_extent"] = {
            "aux_bytes_per_rank": 2 * PAGE,
            "scratch_bytes_per_rank": PAGE,
            "max_total_bytes": 4 * (padded + 3 * PAGE),
        }
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=4,
            cp_kv_cache_interleave_size=4,
        ),
        model_config=SimpleNamespace(architectures=["Glm5NextForCausalLM"]),
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(name="BLHNC")
        ),
        additional_config={
            "hisparse_host_buffer_provider": (
                "kitepond.integrations.vllm.glm_host_backing.provide_glm_host_backing"
            ),
            "kitepond_glm_host": options,
        },
    )
    cache = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                host_resident=True,
                role=KVCacheGroupRole.HISPARSE_SOURCE,
                layer_names=["mla"],
                kv_cache_spec=spec,
            )
        ],
        kv_cache_tensors=[
            SimpleNamespace(
                host_resident=True,
                size=size,
                layers=["mla"],
                offset=0,
                layer_stride=spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        hisparse_host_num_blocks=1,
        hisparse_host_block_stride=size,
        hisparse_shared_host_pool=False,
    )
    return config, cache, size, padded


@pytest.mark.skipif(
    not callable(getattr(runtime.HiSparseHostPool, "after_unpin", None)),
    reason="native post-unpin pool retirement is not available",
)
def test_four_native_registrations_keep_mla_views_across_processes(
    tmp_path, monkeypatch
):
    directory = tmp_path / "s"
    directory.mkdir(mode=0o700)
    attachments = []
    monkeypatch.setattr(
        runtime, "pin_tensor", lambda tensor: attachments.append(tensor)
    )
    config, cache, size, padded = geometry(directory, "authority")
    with ExitStack() as cleanup:
        extents = []
        for rank in range(4):
            group = SimpleNamespace(rank_in_group=rank, world_size=4)
            monkeypatch.setattr(host, "get_dcp_group", lambda group=group: group)
            monkeypatch.setattr(host, "get_tp_group", lambda group=group: group)
            pool = runtime.HiSparseHostPool(config, cache)
            assert pool.allocate(size).nbytes == size
            extent = host.describe_glm_cold_backing(pool.registered)
            cleanup.callback(pool.after_unpin)
            assert extent is not None
            assert pool.registered is attachments[rank]
            assert pool.registered.nbytes == padded
            assert pool.registered.untyped_storage().nbytes() == padded
            assert os.fstat(extent.region.fd).st_size == padded + 3 * PAGE
            assert len(extent.region.mapping) == padded + 3 * PAGE
            pool.registered[0] = 10 + rank
            extent.region.mapping[padded : padded + 3] = b"aux"
            extent.region.mapping[padded + 2 * PAGE : padded + 2 * PAGE + 3] = b"tmp"
            extents.append(extent)

        fingerprints = [extent.region_sha256 for extent in extents]
        assert len(set(fingerprints)) == 4
        assert len({extent.region.descriptor.layout_sha256 for extent in extents}) == 1
        subprocess.run(
            [sys.executable, __file__, str(directory), *fingerprints],
            check=True,
            timeout=45,
        )
        for rank, extent in enumerate(extents):
            assert attachments[rank][1].item() == 20 + rank
            assert extent.region.mapping[padded : padded + 3] == b"aux"
            assert (
                extent.region.mapping[padded + 2 * PAGE : padded + 2 * PAGE + 3]
                == b"tmp"
            )


def test_aux_and_scratch_reservations_drain_before_native_fd_retirement(tmp_path):
    directory = tmp_path / "s"
    directory.mkdir(mode=0o700)
    config, cache, size, padded = geometry(directory, "authority")
    backing, retire = host._open_worker_backing(config, cache, size, 0)
    extent = host.describe_glm_cold_backing(backing)
    assert extent is not None
    endpoint = directory / "host-region-0.sock"
    aux = extent.reserve("aux", PAGE)
    second_aux = extent.reserve("aux", PAGE)
    scratch = extent.reserve("scratch", PAGE)
    assert (aux.offset, second_aux.offset, scratch.offset) == (
        padded,
        padded + PAGE,
        padded + 2 * PAGE,
    )
    assert extent.reserve("aux", 1) == Wait("NATIVE_COLD_EXTENT_CAPACITY_WAIT")
    assert extent.reserve("scratch", 1) == Wait("NATIVE_COLD_EXTENT_CAPACITY_WAIT")
    with pytest.raises(ValueError, match="stale or foreign"):
        extent.release(replace(aux))
    retire()
    retire()
    assert extent.reserve("scratch", 1) == Wait("NATIVE_COLD_EXTENT_RETIRING")
    for lease in (second_aux, scratch):
        extent.release(lease)
        assert endpoint.exists()
        assert os.fstat(extent.region.fd).st_size == padded + 3 * PAGE
    extent.verify(aux)
    extent.release(aux)
    deadline = monotonic() + 3
    while endpoint.exists() and monotonic() < deadline:
        sleep(0.02)
    assert not endpoint.exists()
    with pytest.raises(ValueError, match="stale or foreign"):
        extent.verify(aux)


def test_budget_is_explicit_and_absence_keeps_the_legacy_registration(tmp_path):
    directory = tmp_path / "s"
    directory.mkdir(mode=0o700)
    config, cache, size, padded = geometry(directory, "authority", enabled=False)
    backing, retire = host._open_worker_backing(config, cache, size, 0)
    try:
        assert host.describe_glm_host_backing(backing).nbytes == padded
        assert backing.untyped_storage().nbytes() == padded
        assert host.describe_glm_cold_backing(backing) is None
        assert NativeGlmRankMemory.from_registered_cold_backing(backing) == Wait(
            "NATIVE_COLD_EXTENT_NOT_ENROLLED"
        )
    finally:
        retire()
    with pytest.raises(ValueError, match="Four native ranks"):
        ColdExtentBudget(PAGE, PAGE, 4 * (padded + PAGE)).layout(size)
    with pytest.raises(ValueError, match="page-aligned"):
        ColdExtentBudget(1, PAGE, 4 * (padded + 2 * PAGE))


def check_independent_producer(directory, fingerprints):
    config, cache, size, padded = geometry(directory, "producer")
    for rank, fingerprint in enumerate(fingerprints):
        backing = host._open_worker_backing(config, cache, size, rank)
        descriptor = host.describe_glm_host_backing(backing)
        assert descriptor.owner_pid != os.getpid()
        assert descriptor.fingerprint() == fingerprint
        assert backing.nbytes == padded == backing.untyped_storage().nbytes()
        assert backing[0].item() == 10 + rank
        assert host.describe_glm_cold_backing(backing) is None
        backing[1] = 20 + rank
    wrong, _, _, _ = geometry(directory, "producer", boot="wrong")
    with pytest.raises(ValueError, match="exactly one native region FD"):
        host._open_worker_backing(wrong, cache, size, 0)
    default, _, _, _ = geometry(directory, "producer", enabled=False)
    with pytest.raises(ValueError, match="exactly one native region FD"):
        host._open_worker_backing(default, cache, size, 0)


if __name__ == "__main__":
    check_independent_producer(Path(sys.argv[1]), sys.argv[2:])
