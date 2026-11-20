# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Match verified cold parts to bounded, exact GLM native restore windows."""

from __future__ import annotations

import os
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from kitepond.integrations.mooncake.window import (
    MAX_CHUNK_BYTES,
    WINDOW_BYTES,
    WINDOW_CHUNKS,
    plan_windows,
)

from .layout import CacheIdentity, CacheRole
from .shared_region import NativeHiSparseMemfdRegion

if TYPE_CHECKING:
    from kitepond.backends.fsl2 import PartSpec

# The wire name participates in the signed Mooncake/FSL2 part manifests.
_INDEXER_WIRE_ROLE = "indexer_fp8_soa_v2"
ROW_WIDTHS = {CacheRole.MLA: (528, 656), CacheRole.INDEXER: (132,)}
STATE_FORMATS = {
    CacheRole.KDA_CONV: ((6, 6144), torch.bfloat16),
    CacheRole.KDA_RECURRENT: ((16, 128, 128), torch.float32),
    CacheRole.KPOOL_TAIL: ((8, 2, 128), torch.bfloat16),
}


def part_identity(role: str) -> CacheIdentity:
    model_role, layer, cache_role = role.split(":", 2)
    if cache_role == CacheRole.INDEXER.value:
        raise ValueError("Legacy GLM Indexer parts do not include exact scale bytes")
    role_type = (
        CacheRole.INDEXER if cache_role == _INDEXER_WIRE_ROLE else CacheRole(cache_role)
    )
    return CacheIdentity(layer, role_type, model_role)


def part_role(identity: CacheIdentity) -> str:
    role = identity.cache_role
    wire_role = _INDEXER_WIRE_ROLE if role is CacheRole.INDEXER else role.value
    return f"{identity.model_role}:{identity.layer_name}:{wire_role}"


def validate_part(part: PartSpec) -> CacheIdentity:
    from kitepond.backends.fsl2 import PartSpec

    if (
        type(part) is not PartSpec
        or not 0 <= part.rank < 4
        or part.nbytes > MAX_CHUNK_BYTES
    ):
        raise ValueError("Invalid GLM cold part rank or bounded byte count")
    identity = part_identity(part.role)
    role = identity.cache_role
    if role in ROW_WIDTHS:
        if (
            len(part.shape) != 2
            or part.shape[1] not in ROW_WIDTHS[role]
            or part.dtype != "torch.uint8"
        ):
            raise ValueError("GLM cold part changed its native row format")
    else:
        shape, dtype = STATE_FORMATS[role]
        if (
            part.index != 0
            or part.shape != shape
            or part.dtype != str(dtype)
            or (role is not CacheRole.KPOOL_TAIL and identity.model_role != "target")
        ):
            raise ValueError("GLM cold part must hold the exact whole native state")
    return identity


@dataclass(frozen=True)
class ByteSpan:
    rank: int
    block_id: int
    offset: int
    nbytes: int

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not int
                for value in (self.rank, self.block_id, self.offset, self.nbytes)
            )
            or not 0 <= self.rank < 4
            or self.block_id <= 0
            or self.offset < 0
            or self.nbytes <= 0
        ):
            raise ValueError("Invalid native cold byte span")


@dataclass(frozen=True)
class ColdBinding:
    part: PartSpec
    spans: tuple[ByteSpan, ...]

    def __post_init__(self) -> None:
        validate_part(self.part)
        if (
            type(self.spans) is not tuple
            or any(
                type(span) is not ByteSpan or span.rank != self.part.rank
                for span in self.spans
            )
            or sum(span.nbytes for span in self.spans) != self.part.nbytes
        ):
            raise ValueError("Native cold byte spans changed rank or length")


def slice_native_spans(
    spans: tuple[ByteSpan, ...], start: int, nbytes: int
) -> tuple[ByteSpan, ...]:
    """Select an exact logical byte interval without materializing a part."""
    if type(start) is not int or type(nbytes) is not int or start < 0 or nbytes <= 0:
        raise ValueError("Invalid cold span interval")
    end = start + nbytes
    result = []
    offset = 0
    for span in spans:
        limit = offset + span.nbytes
        left, right = max(start, offset), min(end, limit)
        if left < right:
            result.append(
                ByteSpan(
                    span.rank, span.block_id, span.offset + left - offset, right - left
                )
            )
        offset = limit
        if offset >= end:
            return tuple(result)
    raise ValueError("Cold span interval exceeds native storage")


def _parts_by_role(parts: Sequence[PartSpec]) -> dict:
    groups = defaultdict(list)
    for part in parts:
        validate_part(part)
        groups[(part.role, part.rank)].append(part)
    for key, rows in groups.items():
        rows.sort(key=lambda part: part.index)
        index = 0
        first = rows[0]
        for row in rows:
            if (
                row.index != index
                or row.shape[1:] != first.shape[1:]
                or row.dtype != first.dtype
            ):
                raise ValueError(
                    f"Cold role {key} has an overlap, hole or format drift"
                )
            index += row.shape[0]
    return groups


def _compose_part(part: PartSpec, rows, storage) -> ColdBinding:
    identity = part_identity(part.role)
    if identity.cache_role not in ROW_WIDTHS:
        if tuple(rows) != (part,):
            raise ValueError("Cold whole-state part changed before restore")
        return storage[part]
    width = part.shape[1]
    begin, end = part.index, part.index + part.shape[0]
    spans = []
    first = bisect_right(rows, begin, key=lambda row: row.index) - 1
    for row in rows[first:]:
        if row.index >= end:
            break
        left = max(begin, row.index)
        right = min(end, row.index + row.shape[0])
        if left < right:
            spans.extend(
                slice_native_spans(
                    storage[row].spans,
                    (left - row.index) * width,
                    (right - left) * width,
                )
            )
    return ColdBinding(part, tuple(spans))


def validate_restore_coverage(
    entries: Sequence[PartSpec], parts: Sequence[PartSpec]
) -> dict:
    """Match all native layer identities, widths and contiguous valid rows."""
    expected_groups = _parts_by_role(entries)
    source_groups = _parts_by_role(parts)
    if expected_groups.keys() != source_groups.keys():
        raise ValueError("Cold restore changed its complete rank/layer/role set")

    for key, expected in expected_groups.items():
        actual = source_groups[key]
        if (
            actual[0].shape[1:] != expected[0].shape[1:]
            or actual[0].dtype != expected[0].dtype
            or sum(part.shape[0] for part in actual)
            != sum(part.shape[0] for part in expected)
        ):
            raise ValueError("Cold restore changed native format or row coverage")

    return source_groups


def compose_import_recipe(
    entries: Sequence[PartSpec], source_bindings: Sequence[ColdBinding]
) -> tuple[tuple[ColdBinding, ...], ...]:
    """Validate full per-rank coverage, then re-chunk without copying payloads."""
    if not entries or len({entry.rank for entry in entries}) != 1:
        raise ValueError("Cold restore requires one exact destination rank")
    source = {binding.part: binding for binding in source_bindings}
    if len(source) != len(source_bindings):
        raise ValueError("Duplicate cold part binding")
    source_groups = validate_restore_coverage(entries, tuple(source))
    physical = sorted(
        (span.offset, span.offset + span.nbytes)
        for binding in source_bindings
        for span in binding.spans
    )
    if any(end > start for (_, end), (start, _) in zip(physical, physical[1:])):
        raise ValueError("Cold source spans alias physical bytes")

    canonical, offset = [], 0
    for part in entries:
        canonical.append({"offset": offset, "nbytes": part.nbytes})
        offset += part.nbytes
    return tuple(
        tuple(
            _compose_part(part, source_groups[(part.role, part.rank)], source)
            for part in entries[window.first_chunk : window.end_chunk]
        )
        for window in plan_windows(canonical)
    )


def copy_import_window(
    region: NativeHiSparseMemfdRegion,
    bindings: tuple[ColdBinding, ...],
    destination: torch.Tensor,
    *,
    region_sha256: str,
) -> torch.Tensor:
    """Fill one caller-owned CPU slab from a held native rank-local region."""
    nbytes = sum(binding.part.nbytes for binding in bindings)
    if (
        type(region) is not NativeHiSparseMemfdRegion
        or not 0 < len(bindings) <= WINDOW_CHUNKS
        or destination.device.type != "cpu"
        or destination.dtype != torch.uint8
        or destination.ndim != 1
        or not destination.is_contiguous()
        or destination.storage_offset() % 4
        or not 0 < nbytes <= min(destination.numel(), WINDOW_BYTES)
    ):
        raise ValueError("Cold import needs one bounded aligned CPU window")
    desc = region.descriptor
    physical = os.fstat(region.fd)
    if desc.fingerprint() != region_sha256 or (
        physical.st_dev,
        physical.st_ino,
        physical.st_size,
    ) != (desc.device_id, desc.inode, desc.nbytes):
        raise ValueError("Cold import received a different authority region")
    spans = tuple(span for binding in bindings for span in binding.spans)
    if any(
        span.rank != desc.source_rank or span.offset + span.nbytes > desc.nbytes
        for span in spans
    ):
        raise ValueError("Cold import span exceeds its authority rank/extent")

    base = region.tensor()
    pos = 0
    for span in spans:
        stop = pos + span.nbytes
        destination[pos:stop].copy_(base[span.offset : span.offset + span.nbytes])
        pos = stop
    return destination[:pos]
