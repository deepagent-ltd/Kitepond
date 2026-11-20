# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM rank-local native checkpoint layout and bounded byte readback."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from kitepond.integrations.mooncake.window import (
    MAX_CHUNK_BYTES,
    WINDOW_BYTES,
    WINDOW_CHUNKS,
)

from .import_recipe import (
    ROW_WIDTHS,
    STATE_FORMATS,
    ColdBinding,
    copy_import_window,
    part_role,
    validate_part,
    validate_restore_coverage,
)
from .layout import CacheIdentity, CacheRole, Glm5NextPDLayout
from .shared_region import NativeHiSparseMemfdRegion

if TYPE_CHECKING:
    from kitepond.backends.fsl2 import PartSpec


_INDEXER_KEY_BYTES = 128
_INDEXER_SCALE_BYTES = 4


def _indexer_fields(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the native FP8 Indexer page's separate key and scale arrays."""
    blocks, _, rows, row_bytes = tensor.shape
    native = tensor.view(blocks, rows * row_bytes)
    keys_end = rows * _INDEXER_KEY_BYTES
    keys = native[:, :keys_end].view(blocks, rows, _INDEXER_KEY_BYTES)
    scales = native[:, keys_end:].view(blocks, rows, _INDEXER_SCALE_BYTES)
    return keys, scales


@dataclass(frozen=True)
class RestoreLayer:
    """Caller-owned native view and its current request's logical block table."""

    identity: CacheIdentity
    tensor: torch.Tensor
    block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        role = self.identity.cache_role
        tensor = self.tensor
        if role in ROW_WIDTHS:
            if (
                tensor.dtype != torch.uint8
                or tensor.ndim != 4
                or tensor.shape[1] != 1
                or not tensor.shape[2]
                or tensor.shape[3] not in ROW_WIDTHS[role]
            ):
                raise ValueError("Native GLM row view changed shape or dtype")
        else:
            shape, dtype = STATE_FORMATS[role]
            if role is CacheRole.KPOOL_TAIL:
                shape = (2, 8, 128)
            if (
                tensor.dtype != dtype
                or tuple(tensor.shape[1:]) != shape
                or (
                    role is not CacheRole.KPOOL_TAIL
                    and self.identity.model_role != "target"
                )
            ):
                raise ValueError("Native GLM state view changed shape or dtype")
        if (
            tensor.ndim == 0
            or not tensor.shape[0]
            or type(self.block_ids) is not tuple
            or not self.block_ids
            or any(type(block_id) is not int for block_id in self.block_ids)
        ):
            raise ValueError("GLM restore needs the native request's block table")


def native_rank_rows(valid: int, role: CacheRole, rank: int) -> int:
    """Return the native DCP4/interleave4 row frontier for the layer."""
    if role is CacheRole.INDEXER:
        return max(0, (valid // 4 + 3 - rank) // 4)
    if role is CacheRole.MLA:
        full, rest = divmod(valid, 16)
        return full * 4 + min(4, max(0, rest - rank * 4))
    raise ValueError("Only native MLA and Indexer tensors have row frontiers")


class ColdRestoreLayout:
    """Resolve exact logical parts against one rank's reserved native storage."""

    def __init__(
        self,
        layout: Glm5NextPDLayout,
        rank: int,
        layers: Sequence[RestoreLayer],
    ) -> None:
        layout.require_decode_ready()
        if (
            type(rank) is not int
            or not 0 <= rank < 4
            or not 0 < layout.target_valid_tokens <= 1048576
            or not layers
        ):
            raise ValueError("Invalid GLM cold restore rank or boundary")
        self.layout, self.rank = layout, rank
        self.layers = MappingProxyType(
            {part_role(layer.identity): layer for layer in layers}
        )
        if len(self.layers) != len(layers):
            raise ValueError("Duplicate native GLM cache identity")
        if len({id(layer.tensor) for layer in layers}) != len(layers):
            raise ValueError("Duplicate native GLM cache view")
        for layer in layers:
            if layer.identity.cache_role in ROW_WIDTHS:
                nrows = self._local_rows(layer.identity)
                row_step = layer.tensor.shape[2]
                count = (nrows + row_step - 1) // row_step
                blocks = layer.block_ids[:count]
                if (
                    len(blocks) != count
                    or len(set(blocks)) != count
                    or any(not 0 < block < layer.tensor.shape[0] for block in blocks)
                ):
                    raise ValueError("Cold import destination row blocks are invalid")

    def _local_rows(self, identity: CacheIdentity) -> int:
        return native_rank_rows(
            self.layout.valid_tokens(identity), identity.cache_role, self.rank
        )

    def manifest(self, *, part_bytes: int = MAX_CHUNK_BYTES) -> tuple[PartSpec, ...]:
        """Describe target native bytes without taking a reference to old storage."""
        from kitepond.backends.fsl2 import PartSpec

        if type(part_bytes) is not int or not 0 < part_bytes <= MAX_CHUNK_BYTES:
            raise ValueError("GLM restore chunk exceeds the native byte limit")
        parts = []
        for name, layer in sorted(self.layers.items()):
            role = layer.identity.cache_role
            if role in ROW_WIDTHS:
                width = layer.tensor.shape[-1]
                step = part_bytes // width
                if not step:
                    raise ValueError("GLM restore chunk cannot hold a native row")
                total = self._local_rows(layer.identity)
                for start in range(0, total, step):
                    count = min(step, total - start)
                    parts.append(
                        PartSpec(
                            name,
                            self.rank,
                            start,
                            (count, width),
                            "torch.uint8",
                            count * width,
                        )
                    )
            else:
                shape, dtype = STATE_FORMATS[role]
                nbytes = prod(shape) * dtype.itemsize
                if nbytes > part_bytes:
                    raise ValueError("GLM restore cannot split an exact native state")
                parts.append(PartSpec(name, self.rank, 0, shape, str(dtype), nbytes))
        return tuple(parts)

    def _select(self, part: PartSpec) -> tuple[RestoreLayer, int | tuple]:
        identity = validate_part(part)
        layer = self.layers.get(part.role)
        if part.rank != self.rank or layer is None:
            raise ValueError("Cold import part is not the destination rank/layer")
        role = identity.cache_role
        tensor = layer.tensor
        if role in ROW_WIDTHS:
            if part.shape[1] != tensor.shape[-1]:
                raise ValueError("Cold import cannot repack a native GLM row")
            start, end = part.index, part.index + part.shape[0]
            rows_per_block = tensor.shape[2]
            head, last = start // rows_per_block, (end - 1) // rows_per_block
            if end > self._local_rows(identity) or last >= len(layer.block_ids):
                raise ValueError("Cold import rows exceed the destination prefix")
            blocks = layer.block_ids[head : last + 1]
            if len(set(blocks)) != len(blocks) or any(
                not 0 < block < tensor.shape[0] for block in blocks
            ):
                raise ValueError("Cold import destination row blocks are invalid")
            local_rows = torch.arange(start, end, device=tensor.device)
            table = torch.tensor(blocks, dtype=torch.int64, device=tensor.device)
            physical = table[local_rows // rows_per_block - head]
            return layer, (physical, 0, local_rows % rows_per_block)
        slot = (
            self.layout.committed_kda_slot
            if role is CacheRole.KDA_RECURRENT
            else self.layout.conv_base_slot
        )
        if (
            slot >= len(layer.block_ids)
            or not 0 < layer.block_ids[slot] < tensor.shape[0]
        ):
            raise ValueError("Cold import has no valid committed native state slot")
        return layer, layer.block_ids[slot]

    def read_chunk(self, part: PartSpec) -> torch.Tensor:
        """Pack only the selected destination bytes in canonical cold order."""
        layer, indexes = self._select(part)
        if layer.identity.cache_role is CacheRole.INDEXER:
            pages, _, rows = indexes
            keys, scales = _indexer_fields(layer.tensor)
            result = torch.cat((keys[pages, rows], scales[pages, rows]), dim=-1)
        else:
            result = layer.tensor[indexes]
        if layer.identity.cache_role is CacheRole.KPOOL_TAIL:
            result = result.transpose(0, 1)
        return result.contiguous().view(torch.uint8).flatten().cpu()

    def scatter_window(self, parts: Sequence[PartSpec], payload: torch.Tensor) -> None:
        """Write one fully checked window into existing native views."""
        size = sum(part.nbytes for part in parts)
        if (
            not 0 < len(parts) <= WINDOW_CHUNKS
            or payload.device.type != "cpu"
            or payload.dtype != torch.uint8
            or payload.ndim != 1
            or not payload.is_contiguous()
            or payload.storage_offset() % 4
            or not 0 < payload.numel() == size <= WINDOW_BYTES
        ):
            raise ValueError("Cold restore needs exact bounded CPU bytes")
        destinations = [self._select(part) for part in parts]
        rows = sorted(
            (part.role, part.index, part.index + part.shape[0]) for part in parts
        )
        if any(
            role == next_role and stop > next_start
            for (role, _, stop), (next_role, next_start, _) in zip(rows, rows[1:])
        ):
            raise ValueError("Cold restore window writes the same native rows twice")
        start = 0
        with torch.no_grad():
            for part, (layer, indexes) in zip(parts, destinations):
                end = start + part.nbytes
                data = payload[start:end].view(layer.tensor.dtype).reshape(part.shape)
                data = data.to(layer.tensor.device)
                role = layer.identity.cache_role
                if role is CacheRole.INDEXER:
                    pages, _, rows = indexes
                    keys, scales = _indexer_fields(layer.tensor)
                    keys[pages, rows] = data[:, :_INDEXER_KEY_BYTES]
                    scales[pages, rows] = data[:, _INDEXER_KEY_BYTES:]
                else:
                    if role is CacheRole.KPOOL_TAIL:
                        data = data.transpose(0, 1)
                    layer.tensor[indexes] = data
                start = end


def verify_cold_import_readback(
    cache: ColdRestoreLayout,
    recipe: Sequence[tuple[ColdBinding, ...]],
    region: NativeHiSparseMemfdRegion,
    slab: torch.Tensor,
    *,
    region_sha256: str,
) -> dict:
    """Compare actual destination and verified source bytes by window."""
    validate_restore_coverage(
        cache.manifest(), tuple(binding.part for window in recipe for binding in window)
    )
    counts: dict[str, int] = {}
    ordinal = 0
    for window in recipe:
        expected = copy_import_window(region, window, slab, region_sha256=region_sha256)
        offset = 0
        for binding in window:
            part = binding.part
            role = validate_part(part).cache_role.value
            count = part.nbytes
            source = expected[offset : offset + count]
            actual = cache.read_chunk(part)
            if actual.numel() != count:
                raise ValueError(
                    "Cold destination readback changed its native byte count"
                )
            counts[role] = counts.get(role, 0) + 1
            ordinal += 1
            if not torch.equal(source, actual):
                return {
                    "rank": cache.rank,
                    "status": "MISMATCH",
                    "counts": counts,
                    "mismatch": {
                        "role": role,
                        "count": ordinal,
                        "expected_sha256": hashlib.sha256(source.numpy()).hexdigest(),
                        "actual_sha256": hashlib.sha256(actual.numpy()).hexdigest(),
                    },
                }
            offset += count
    if not ordinal:
        raise ValueError("Cold readback requires selected native bytes")
    return {"rank": cache.rank, "status": "MATCH", "counts": counts, "mismatch": None}
