# SPDX-License-Identifier: Apache-2.0
"""Provide rank-local host backing for the qualified TP8/DCP8 HiSparse path."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .hisparse_host import _acquire_rank_backing

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

POLICY_ENV = "KITEPOND_HISPARSE_DCP8_HOST_POOL_POLICY"
DCP_SIZE = 8


def _owner_rank(config: VllmConfig) -> int:
    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_dcp_group, get_tp_group
    from vllm.v1.hisparse.dcp_layout import validate_hisparse_dcp8_config

    validate_hisparse_dcp8_config(config)
    if config.parallel_config.data_parallel_index != 0:
        raise ValueError("HiSparse DCP8 host backing requires the sole DP owner")
    if not dist.is_initialized() or dist.get_world_size() != DCP_SIZE:
        raise ValueError(
            "HiSparse DCP8 host backing requires eight initialized workers"
        )
    tp, dcp = get_tp_group(), get_dcp_group()
    rank, global_rank = dcp.rank_in_group, dist.get_rank()
    if (
        tp.world_size != DCP_SIZE
        or dcp.world_size != DCP_SIZE
        or len(set(dcp.ranks)) != DCP_SIZE
        or tp.ranks != dcp.ranks
        or not 0 <= rank < DCP_SIZE
        or tp.rank_in_group != rank
        or tp.rank != global_rank
        or dcp.rank != global_rank
        or dcp.ranks[rank] != global_rank
    ):
        raise ValueError("HiSparse DCP8 host backing has inconsistent TP/DCP ownership")
    return rank


def _validate_source(config: VllmConfig, cache: KVCacheConfig, size: int) -> None:
    import torch
    from vllm.v1.kv_cache_interface import (
        MLAAttentionSpec,
        SparseCacheRole,
        iter_layer_specs,
    )

    if (
        cache.hisparse_shared_host_pool
        or len(cache.host_group_ids) != 1
        or not config.cache_config.get_resolved_kv_cache_layout().is_block_outermost
    ):
        raise ValueError(
            "HiSparse DCP8 host backing requires a rank-local source group"
        )
    (source_id,) = cache.host_group_ids
    source_group = cache.kv_cache_groups[source_id]
    source = source_group.kv_cache_spec
    is_full_glm = getattr(config.attention_config, "enable_glm53_full_dcp8", False)
    # The engine revalidates the exact selected model/layout in _owner_rank().
    page_rows, row_bytes = (64, 656) if is_full_glm else (256, 528)
    per_layer = tuple(iter_layer_specs(source))
    # The Full profile has 78 target layers and one source for its native draft.
    full_draft_layers = int(getattr(config, "speculative_config", None) is not None)
    pages = cache.hisparse_host_num_blocks
    stride = cache.hisparse_host_block_stride
    host_tensors = [tensor for tensor in cache.kv_cache_tensors if tensor.host_resident]
    if (
        source.block_size != page_rows
        or not source.dcp_sharded
        or not per_layer
        or (
            is_full_glm
            and (
                len(source_group.layer_names) != 78 + full_draft_layers
                or len(per_layer) != len(source_group.layer_names)
            )
        )
        or any(
            not isinstance(spec, MLAAttentionSpec)
            or not spec.dcp_sharded
            or spec.block_size != page_rows
            or spec.cache_role != SparseCacheRole.SPARSE
            or spec.cache_dtype_str != "fp8_ds_mla"
            or spec.tokens_per_state != 1
            or spec.page_size_bytes != page_rows * row_bytes
            or (
                is_full_glm
                and (
                    spec.head_size != 576
                    or spec.num_kv_heads != 1
                    or spec.dtype != torch.uint8
                )
            )
            for spec in per_layer
        )
        or (
            is_full_glm
            and sum(spec.is_draft for spec in per_layer) != full_draft_layers
        )
        or type(pages) is not int
        or pages <= 0
        or stride != source.page_size_bytes
        or type(size) is not int
        or size != pages * stride
        or not host_tensors
        or any(tensor.size != size for tensor in host_tensors)
    ):
        raise ValueError(
            "HiSparse DCP8 host backing does not match the packed source layout"
        )


def acquire(
    config: VllmConfig, cache: KVCacheConfig, size: int
) -> tuple[torch.Tensor, Callable[[], None]]:
    """Return legacy positional-provider output for old and new vLLM runtimes."""
    rank = _owner_rank(config)
    _validate_source(config, cache, size)
    return _acquire_rank_backing(
        instance_id=config.instance_id,
        rank=rank,
        size=size,
        policy_env=POLICY_ENV,
        owner="DCP8",
    )
