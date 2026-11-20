# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Back one GLM worker's HiSparse host extent with its D authority's memfd."""

from __future__ import annotations

import hashlib
import json
import logging
import mmap
import os
import threading
import weakref
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from kitepond.models.glm5next.host_cold_extent import (
    ColdExtentBudget,
    ColdExtentLayout,
    NativeColdExtent,
)
from kitepond.models.glm5next.region_socket import RegionPublisher, join_region
from kitepond.models.glm5next.shared_region import (
    NativeHiSparseMemfdRegion,
    NativeHiSparseRegionDescriptor,
)

from vllm.distributed.parallel_state import get_dcp_group, get_tp_group
from vllm.v1.hisparse.runtime import check_hisparse_host_memory
from vllm.v1.kv_cache_interface import (
    KVCacheGroupRole,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Launch:
    role: str
    socket_directory: Path
    authority_engine_id: str
    authority_boot_id: str
    model_sha256: str
    cold_extent: ColdExtentBudget | None = None

    @classmethod
    def from_config(cls, vllm_config) -> _Launch:
        options = vllm_config.additional_config
        raw = options.get("kitepond_glm_host") if isinstance(options, dict) else None
        fields = {
            "role",
            "socket_directory",
            "authority_engine_id",
            "authority_boot_id",
            "model_sha256",
        }
        if (
            not isinstance(raw, dict)
            or set(raw) not in (fields, fields | {"cold_extent"})
            or any(not isinstance(raw[key], str) or not raw[key] for key in fields)
        ):
            raise ValueError("Kitepond needs an exact GLM host launcher binding")
        cold = None
        if "cold_extent" in raw:
            try:
                cold = ColdExtentBudget(**raw["cold_extent"])
            except (TypeError, ValueError) as error:
                raise ValueError("Invalid explicit GLM cold extent budget") from error
        launch = cls(
            **{key: raw[key] for key in fields if key != "socket_directory"},
            socket_directory=Path(raw["socket_directory"]),
            cold_extent=cold,
        )
        if (
            launch.role not in {"authority", "producer"}
            or not launch.socket_directory.is_absolute()
            or len(launch.model_sha256) != 64
            or any(char not in "0123456789abcdef" for char in launch.model_sha256)
        ):
            raise ValueError("Invalid GLM host launcher binding")
        return launch


def _host_layout(vllm_config, cache, size: int, launch: _Launch) -> str:
    parallel = vllm_config.parallel_config
    topology = (
        parallel.tensor_parallel_size,
        parallel.pipeline_parallel_size,
        parallel.data_parallel_size,
        parallel.prefill_context_parallel_size,
        parallel.decode_context_parallel_size,
        parallel.cp_kv_cache_interleave_size,
    )
    if topology != (4, 1, 1, 1, 4, 4):
        raise ValueError(
            "GLM shared host backing requires TP4/DP1/PP1/DCP4/interleave4"
        )
    architectures = set(vllm_config.model_config.architectures)
    if not architectures.intersection(
        {"Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"}
    ):
        raise ValueError("GLM shared host backing requires a GLM5Next architecture")

    groups = [group for group in cache.kv_cache_groups if group.host_resident]
    tensors = [tensor for tensor in cache.kv_cache_tensors if tensor.host_resident]
    host_blocks = cache.hisparse_host_num_blocks
    host_stride = cache.hisparse_host_block_stride
    device_layout = vllm_config.cache_config.get_resolved_kv_cache_layout().name
    if (
        cache.hisparse_shared_host_pool
        or device_layout != "BLHNC"
        or len(groups) != 1
        or groups[0].role is not KVCacheGroupRole.HISPARSE_SOURCE
        or type(host_blocks) is not int
        or host_blocks <= 0
        or type(host_stride) is not int
        or host_stride <= 0
        or type(size) is not int
        or size != host_blocks * host_stride
        or {tensor.size for tensor in tensors} != {size}
    ):
        raise ValueError("GLM native rank-local host allocation is inconsistent")

    group = groups[0]
    rows = {}
    spans = []
    for tensor in tensors:
        for index, name in enumerate(tensor.layers):
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            offset = tensor.offset + index * tensor.layer_stride
            if (
                name in rows
                or not isinstance(spec, MLAAttentionSpec)
                or not spec.dcp_sharded
                or spec.block_size != 256
                or spec.state_content_size_bytes not in {528, 656}
                or offset < 0
                or tensor.layer_stride < spec.page_size_bytes
                or tensor.block_stride < spec.page_size_bytes
                or offset
                + (host_blocks - 1) * tensor.block_stride
                + spec.page_size_bytes
                > size
            ):
                raise ValueError("Invalid GLM MLA host-layer placement")
            spans.append(
                (
                    offset,
                    offset
                    + (host_blocks - 1) * tensor.block_stride
                    + spec.page_size_bytes,
                )
            )
            rows[name] = {
                "offset": offset,
                "layer_stride": tensor.layer_stride,
                "block_stride": tensor.block_stride,
                "state_content_bytes": spec.state_content_size_bytes,
                "spec": asdict(spec),
            }
    spans.sort()
    if any(end > start for (_, end), (start, _) in zip(spans, spans[1:])):
        raise ValueError("GLM native host layers overlap")
    if not rows or set(rows) != set(group.layer_names):
        raise ValueError("GLM host backing does not cover its declared MLA layers")

    # The coordinates are rank-local. Rank identity is independently checked
    # by the descriptor and socket so all four ranks share a layout identity.
    layout = {
        "schema": "kitepond.glm5next.host-layout.v1",
        "model_sha256": launch.model_sha256,
        "topology": topology,
        "host_layout": "LBNHC",
        "device_layout": device_layout,
        "host_blocks": host_blocks,
        "host_stride": host_stride,
        "host_bytes": size,
        "layers": rows,
    }
    if launch.cold_extent is not None:
        layout["cold_extent"] = asdict(launch.cold_extent.layout(size))
    wire = json.dumps(layout, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(wire.encode()).hexdigest()


def _serve_authority(publisher: RegionPublisher, stop: threading.Event) -> None:
    try:
        while not stop.is_set():
            try:
                publisher.serve_one(accept_timeout=0.2)
            except TimeoutError:
                continue
    except OSError:
        if not stop.is_set():
            logger.exception("GLM host authority socket stopped serving")
    finally:
        try:
            publisher.close()
        finally:
            os.close(publisher.region.fd)


def _open_extended_backing(
    launch: _Launch, scope: dict, layout: ColdExtentLayout
) -> torch.Tensor | tuple[torch.Tensor, Callable[[], None]]:
    if launch.role == "producer":
        region = join_region(
            launch.socket_directory, **scope, map_bytes=layout.mla_padded_bytes
        )
        try:
            if region.descriptor.nbytes != layout.extent:
                raise ValueError("GLM producer received the wrong cold extent")
            backing = region.tensor().view(torch.int8)
        finally:
            os.close(region.fd)
        backing._kitepond_glm_host_descriptor = region.descriptor
        return backing

    check_hisparse_host_memory(layout.extent)
    region = NativeHiSparseMemfdRegion.create(layout.extent, **scope)
    publisher = None
    try:
        publisher = RegionPublisher(launch.socket_directory, region)
        backing = torch.frombuffer(
            region.mapping, dtype=torch.int8, count=layout.mla_padded_bytes
        )
        stop = threading.Event()
        extent = NativeColdExtent(region, layout, stop.set)
        backing._kitepond_glm_host_descriptor = region.descriptor
        backing._kitepond_glm_cold_extent = extent
        threading.Thread(
            name=f"kitepond-host-authority-{region.descriptor.source_rank}",
            target=_serve_authority,
            args=(publisher, stop),
            daemon=True,
        ).start()
    except BaseException:
        try:
            if publisher is not None:
                publisher.close()
        finally:
            os.close(region.fd)
        raise
    return backing, extent.retire_after_unpin


def _open_worker_backing(
    vllm_config, cache, size: int, rank: int
) -> torch.Tensor | tuple[torch.Tensor, Callable[[], None]]:
    launch = _Launch.from_config(vllm_config)
    scope = {
        "authority_engine_id": launch.authority_engine_id,
        "authority_boot_id": launch.authority_boot_id,
        "source_rank": rank,
        "layout_sha256": _host_layout(vllm_config, cache, size, launch),
    }
    if launch.cold_extent is not None:
        return _open_extended_backing(launch, scope, launch.cold_extent.layout(size))
    extent = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
    if launch.role == "producer":
        region = join_region(launch.socket_directory, **scope)
        try:
            if region.descriptor.nbytes != extent:
                raise ValueError("GLM producer received the wrong authority extent")
            backing = region.tensor().view(torch.int8)
        finally:
            # mmap stays alive through Torch's storage buffer, independently
            # of this process's received descriptor.
            os.close(region.fd)
    else:
        check_hisparse_host_memory(extent)
        region = NativeHiSparseMemfdRegion.create(extent, **scope)
        publisher = None
        try:
            publisher = RegionPublisher(launch.socket_directory, region)
            backing = region.tensor().view(torch.int8)
            stop = threading.Event()
            weakref.finalize(backing, stop.set)
            threading.Thread(
                name=f"kitepond-host-authority-{rank}",
                target=_serve_authority,
                args=(publisher, stop),
                daemon=True,
            ).start()
        except BaseException:
            try:
                if publisher is not None:
                    publisher.close()
            finally:
                os.close(region.fd)
            raise

    backing._kitepond_glm_host_descriptor = region.descriptor
    return backing if launch.role == "producer" else (backing, stop.set)


def provide_glm_host_backing(
    vllm_config, kv_cache_config, size: int
) -> torch.Tensor | tuple[torch.Tensor, Callable[[], None]]:
    """VLLM startup provider. The native engine owns CUDA pin and unpin order."""
    dcp, tp = get_dcp_group(), get_tp_group()
    rank = dcp.rank_in_group
    if dcp.world_size != 4 or tp.world_size != 4 or tp.rank_in_group != rank:
        raise ValueError("GLM host provider requires aligned native DCP4/TP4 ranks")
    return _open_worker_backing(vllm_config, kv_cache_config, size, rank)


def describe_glm_host_backing(backing: torch.Tensor) -> NativeHiSparseRegionDescriptor:
    """Read the local provider descriptor from the registered native tensor."""
    descriptor = getattr(backing, "_kitepond_glm_host_descriptor", None)
    if not isinstance(descriptor, NativeHiSparseRegionDescriptor):
        raise ValueError("Tensor was not registered by the GLM host provider")
    descriptor.validate()
    return descriptor


def describe_glm_cold_backing(backing: torch.Tensor) -> NativeColdExtent | None:
    """Read the optional rank extent only from the native D registration."""
    extent = getattr(backing, "_kitepond_glm_cold_extent", None)
    if extent is not None and (
        type(extent) is not NativeColdExtent
        or extent.region.descriptor != describe_glm_host_backing(backing)
        or not extent.region.owner
    ):
        raise ValueError("Cold extent differs from its registered native D backing")
    return extent
