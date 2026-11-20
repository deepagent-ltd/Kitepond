# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-local GLM window plan using already-admitted native cache views."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy, deepcopy
from dataclasses import asdict, replace
from itertools import groupby
from typing import TYPE_CHECKING

import torch

from kitepond.integrations.mooncake.window import TransferWindow

from .cold_restore import ColdRestoreLayout, native_rank_rows
from .import_recipe import ROW_WIDTHS, part_identity
from .layout import CacheRole

if TYPE_CHECKING:
    from kitepond.backends.fsl2 import PartSpec
    from kitepond.core.readiness import ReadyAttempt

    from .shared_mla import VerifiedSharedMlaElision
    from .shared_region import NativeHiSparseRegionDescriptor

_KDA = {CacheRole.KDA_CONV, CacheRole.KDA_RECURRENT}


def rank_window_records(parts: Sequence[tuple[str, PartSpec]]) -> tuple[dict, ...]:
    records, offset = [], 0
    for scope, part in parts:
        records.append(
            {
                "purpose": scope,
                "offset": offset,
                **asdict(part),
                "shape": list(part.shape),
            }
        )
        offset += part.nbytes
    return tuple(records)


class GlmRankWindowPlan:
    """Freeze identical source and destination byte recipes independently."""

    def __init__(
        self,
        final: ColdRestoreLayout,
        local_tokens: int,
        *,
        checkpoint: ColdRestoreLayout | None = None,
    ):
        valid = final.layout.target_valid_tokens
        if (
            type(local_tokens) is not int
            or not 0 <= local_tokens < valid
            or local_tokens % 1024
        ):
            raise ValueError("GLM window plan needs its exact native local prefix")
        self.final, self.checkpoint = final, checkpoint
        self.local_tokens = local_tokens
        if checkpoint is not None:
            self._check_checkpoint()

        selected = []
        for part in final.manifest():
            role = part_identity(part.role).cache_role
            if role in ROW_WIDTHS:
                low = native_rank_rows(local_tokens, role, final.rank)
                stop = part.index + part.shape[0]
                start = max(low, part.index)
                if start >= stop:
                    continue
                rows, width = stop - start, part.shape[1]
                part = replace(
                    part, index=start, shape=(rows, width), nbytes=rows * width
                )
            selected.append(("final", part))

        if checkpoint is not None:
            selected.extend(
                ("checkpoint", part)
                for part in checkpoint.manifest()
                if part_identity(part.role).cache_role in _KDA
            )
        self._parts: tuple[tuple[str, PartSpec], ...] = tuple(selected)
        self._records = rank_window_records(self._parts)

    def _check_checkpoint(self) -> None:
        final, checkpoint = self.final, self.checkpoint
        assert checkpoint is not None
        boundary = checkpoint.layout.target_valid_tokens
        if (
            final.rank != checkpoint.rank
            or not self.local_tokens
            < boundary
            <= final.layout.target_valid_tokens - 1024
            or boundary % 1024
            or replace(
                checkpoint.layout,
                target_valid_tokens=final.layout.target_valid_tokens,
                draft_valid_tokens=final.layout.draft_valid_tokens,
                committed_kda_slot=final.layout.committed_kda_slot,
            )
            != final.layout
        ):
            raise ValueError("GLM detached checkpoint differs from the native boundary")
        final_kda = {
            part.role: part
            for part in final.manifest()
            if part_identity(part.role).cache_role in _KDA
        }
        snapshot_kda = {
            part.role: part
            for part in checkpoint.manifest()
            if part_identity(part.role).cache_role in _KDA
        }
        if not final_kda or final_kda.keys() != snapshot_kda.keys():
            raise ValueError("GLM detached checkpoint omitted native KDA layers")
        for name, part in final_kda.items():
            live, live_block = final._select(part)
            saved, saved_block = checkpoint._select(snapshot_kda[name])
            if live.tensor is not saved.tensor or live_block == saved_block:
                raise ValueError(
                    "GLM detached checkpoint changed native storage ownership"
                )

    def for_forward_shared_mla(
        self,
        proof: VerifiedSharedMlaElision | None,
        region: NativeHiSparseRegionDescriptor,
        attempt: ReadyAttempt,
    ) -> GlmRankWindowPlan:
        """Keep the existing full recipe until the admitted four-rank proof."""
        if proof is None:
            return self
        if self.local_tokens or self.checkpoint is not None:
            raise ValueError("Only the cold forward wire plan supports shared MLA")
        proof.validate_local(self.final, region, attempt)
        result = copy(self)
        result._parts = tuple(
            (scope, part)
            for scope, part in self._parts
            if part_identity(part.role).cache_role is not CacheRole.MLA
        )
        result._records = rank_window_records(result._parts)
        return result

    def manifest(self) -> tuple[dict, ...]:
        return deepcopy(self._records)

    def _layers_for(self, scope: str) -> ColdRestoreLayout:
        cache = self.final if scope == "final" else self.checkpoint
        assert cache is not None
        return cache

    def _check_payload(self, window: TransferWindow, payload: torch.Tensor) -> None:
        local = self._records[window.first_chunk : window.end_chunk]
        if (
            not local
            or local != window.entries
            or payload.device.type != "cpu"
            or payload.ndim != 1
            or payload.dtype != torch.uint8
            or not payload.is_contiguous()
            or payload.numel() != window.nbytes
        ):
            raise ValueError("GLM window payload differs from its admitted recipe")

    def scatter(
        self,
        window: TransferWindow,
        payload: torch.Tensor,
        *,
        read_shared_mla: bool = False,
    ) -> int:
        self._check_payload(window, payload)
        offset = 0
        selected = self._parts[window.first_chunk : window.end_chunk]

        def destination(part):
            scope, spec = part
            shared = (
                read_shared_mla and part_identity(spec.role).cache_role is CacheRole.MLA
            )
            return scope, shared

        for (scope, shared), groups in groupby(selected, key=destination):
            parts = tuple(part for _, part in groups)
            end = offset + sum(part.nbytes for part in parts)
            if not shared:
                self._layers_for(scope).scatter_window(parts, payload[offset:end])
            offset = end
        return offset

    def verify_readback(self, window: TransferWindow, payload: torch.Tensor) -> None:
        self._check_payload(window, payload)
        offset = 0
        for scope, part in self._parts[window.first_chunk : window.end_chunk]:
            end = offset + part.nbytes
            if not torch.equal(
                self._layers_for(scope).read_chunk(part), payload[offset:end]
            ):
                raise ValueError("GLM live window native readback differs")
            offset = end

    def pack(
        self,
        window: TransferWindow,
        payload: torch.Tensor,
        *,
        read_final: Callable[[PartSpec], torch.Tensor] | None = None,
    ) -> int:
        self._check_payload(window, payload)
        offset = 0
        for scope, part in self._parts[window.first_chunk : window.end_chunk]:
            end = offset + part.nbytes
            read = (
                read_final
                if scope == "final" and read_final is not None
                else self._layers_for(scope).read_chunk
            )
            data = read(part)
            if (
                data.device.type != "cpu"
                or data.dtype != torch.uint8
                or data.shape != (part.nbytes,)
                or not data.is_contiguous()
            ):
                raise ValueError("GLM window source changed its canonical byte view")
            payload[offset:end].copy_(data)
            offset = end
        return offset
