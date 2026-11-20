# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native GLM DCP4 HiSparse layout, addressing, and rank-local accounting."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from kitepond.models.glm5next.layout import CacheRole
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SparseCacheRole

DCP_RANKS = 4
DCP_INTERLEAVE = 4
MANAGER_BLOCK_TOKENS = 256
MLA_KERNEL_BLOCK_ROWS = 64
MLA_ROW_WIDTHS = frozenset({528, 656})
INDEXER_TOKENS_PER_STATE = 4
INDEXER_ROW_WIDTH = 132


@dataclass(frozen=True)
class NativeCachePlane:
    """Per-rank physical cache page for one group of GLM layers."""

    layer_names: tuple[str, ...]
    row_bytes: int
    page_rows: int
    kernel_page_rows: int

    @property
    def page_bytes(self) -> int:
        return len(self.layer_names) * self.page_rows * self.row_bytes


@dataclass(frozen=True)
class RankPageAddress:
    request_page: int
    physical_block: int
    physical_row: int
    kernel_block: int
    kernel_row: int


@dataclass(frozen=True)
class HostShardPlan:
    rank_budget_bytes: int
    blocks_per_rank: int
    shard_bytes: int
    aggregate_bytes: int
    logical_tokens: int


@dataclass(frozen=True)
class SparseAccessDelta:
    direct_rows: int = 0
    hot_hits: int = 0
    host_recalls: int = 0
    host_read_bytes: int = 0


@dataclass(frozen=True)
class GlmHiSparseGeometry:
    mla: NativeCachePlane
    indexer: NativeCachePlane

    @classmethod
    def from_native_specs(
        cls,
        native_specs: Mapping[str, MLAAttentionSpec],
        *,
        dcp_size: int,
        interleave_size: int,
    ) -> "GlmHiSparseGeometry":
        if (dcp_size, interleave_size) != (DCP_RANKS, DCP_INTERLEAVE):
            raise ValueError("GLM HiSparse geometry requires DCP4/interleave4")

        def plane(
            role: SparseCacheRole,
            tokens_per_state: int,
            widths: frozenset[int],
            kernel_rows: int,
        ) -> NativeCachePlane:
            layers = [
                (name, spec)
                for name, spec in native_specs.items()
                if isinstance(spec, MLAAttentionSpec) and spec.cache_role is role
            ]
            actual_widths = {spec.state_content_size_bytes for _, spec in layers}
            if len(actual_widths) != 1 or not actual_widths <= widths:
                raise ValueError(f"Unsupported {role.value} row width")
            (row_bytes,) = actual_widths
            page_rows = MANAGER_BLOCK_TOKENS // tokens_per_state
            for name, spec in layers:
                if (
                    not name
                    or spec.block_size != MANAGER_BLOCK_TOKENS
                    or not spec.dcp_sharded
                    or spec.dtype is not torch.uint8
                    or spec.num_heads != 1
                    or spec.tokens_per_state != tokens_per_state
                    or spec.num_states != page_rows
                    or spec.page_size_bytes != page_rows * row_bytes
                ):
                    raise ValueError(
                        f"Unsupported native {role.value} geometry: {name}"
                    )
            return NativeCachePlane(
                tuple(name for name, _ in layers), row_bytes, page_rows, kernel_rows
            )

        return cls(
            mla=plane(SparseCacheRole.SPARSE, 1, MLA_ROW_WIDTHS, MLA_KERNEL_BLOCK_ROWS),
            indexer=plane(
                SparseCacheRole.INDEXER,
                INDEXER_TOKENS_PER_STATE,
                frozenset({INDEXER_ROW_WIDTH}),
                MLA_KERNEL_BLOCK_ROWS,
            ),
        )

    @property
    def logical_page_tokens(self) -> int:
        return MANAGER_BLOCK_TOKENS * DCP_RANKS

    def plan_host_shards(
        self, total_budget_bytes: int, max_model_len: int
    ) -> HostShardPlan:
        """Divide one configured RAM budget among the four distinct MLA shards."""
        if type(total_budget_bytes) is not int or total_budget_bytes <= 0:
            raise ValueError("Host budget must be a positive byte count")
        if type(max_model_len) is not int or max_model_len <= 0:
            raise ValueError("Maximum context must be a positive token count")
        rank_budget = total_budget_bytes // DCP_RANKS
        blocks = rank_budget // self.mla.page_bytes
        logical_page = self.logical_page_tokens
        # Native block zero is reserved. Leave room for one copy-on-write tail.
        if blocks < (max_model_len + logical_page - 1) // logical_page + 2:
            raise ValueError("Host budget cannot hold one maximum context plus COW")
        shard_bytes = blocks * self.mla.page_bytes
        return HostShardPlan(
            rank_budget,
            blocks,
            shard_bytes,
            DCP_RANKS * shard_bytes,
            (blocks - 2) * logical_page,
        )

    def address(
        self,
        role: CacheRole,
        token_position: int,
        block_table: Sequence[int],
        *,
        rank: int,
        mla_block_stride_rows: int | None = None,
    ) -> RankPageAddress | None:
        """Resolve a request token, or the token completing an indexer pool.

        MLA defaults to dense layer-compact pages. Pass the actual row stride
        from a block-outermost MLA view when resolving that view's kernel tiles.
        """
        if (
            type(token_position) is not int
            or token_position < 0
            or type(rank) is not int
            or not 0 <= rank < DCP_RANKS
        ):
            raise ValueError(
                "A nonnegative token position and a DCP4 rank are required"
            )
        page, offset = divmod(token_position, self.logical_page_tokens)
        if role is CacheRole.MLA:
            plane = self.mla
            group, within_group = divmod(offset, DCP_INTERLEAVE)
            owner = group % DCP_RANKS
            physical_row = group // DCP_RANKS * DCP_INTERLEAVE + within_group
        elif role is CacheRole.INDEXER:
            if mla_block_stride_rows is not None:
                raise ValueError("An MLA block stride only applies to MLA addresses")
            plane = self.indexer
            pool, within_pool = divmod(offset, INDEXER_TOKENS_PER_STATE)
            if within_pool != INDEXER_TOKENS_PER_STATE - 1:
                return None
            physical_row, owner = divmod(pool, DCP_RANKS)
        else:
            raise ValueError("Addressing requires an MLA or completed indexer pool")
        if owner != rank:
            return None
        if page >= len(block_table):
            raise ValueError("Logical request page has no native block")
        block = block_table[page]
        if type(block) is not int or block <= 0:
            raise ValueError("Logical request page has no live native block")
        block_stride_rows = plane.page_rows
        if mla_block_stride_rows is not None:
            if (
                type(mla_block_stride_rows) is not int
                or mla_block_stride_rows < plane.page_rows
                or mla_block_stride_rows % plane.kernel_page_rows
            ):
                raise ValueError(
                    "MLA block stride must fit aligned physical kernel tiles"
                )
            block_stride_rows = mla_block_stride_rows
        tile, kernel_row = divmod(
            block * block_stride_rows + physical_row, plane.kernel_page_rows
        )
        return RankPageAddress(page, block, physical_row, tile, kernel_row)

    def access_delta(
        self,
        branch: Literal["direct", "swap"],
        *,
        direct_rows: int = 0,
        hot_hits: int = 0,
        host_recalls: int = 0,
        mla_layers: Sequence[str] = (),
    ) -> SparseAccessDelta:
        """Calculate one rank's observed direct or swapping-path counters."""
        if any(
            type(n) is not int or n < 0 for n in (direct_rows, hot_hits, host_recalls)
        ):
            raise ValueError("Sparse access counters must be nonnegative integers")
        if branch == "direct" and not (hot_hits or host_recalls or mla_layers):
            return SparseAccessDelta(direct_rows=direct_rows)
        if (
            branch == "swap"
            and not direct_rows
            and mla_layers
            and len(set(mla_layers)) == len(mla_layers)
            and set(mla_layers) <= set(self.mla.layer_names)
        ):
            return SparseAccessDelta(
                hot_hits=hot_hits,
                host_recalls=host_recalls,
                host_read_bytes=host_recalls * len(mla_layers) * self.mla.row_bytes,
            )
        raise ValueError("Sparse access counters do not match the observed branch")
