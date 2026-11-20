# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from kitepond.integrations.vllm import glm_sparse_backend
from kitepond.integrations.vllm.glm_sparse_backend import (
    GlmHiSparseDCPAttentionImpl,
    GlmHiSparseDCPBackend,
    GlmSparseAttentionBinding,
)
from kitepond.integrations.vllm.glm_sparse_registration import (
    GLM_HISPARSE_BACKEND,
    register_glm_hisparse_backend,
)

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_opt_in_registry_keeps_native_sm120_layout_selection():
    stock = AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120
    path = stock.get_path()
    custom = register_glm_hisparse_backend()
    assert custom.name == GLM_HISPARSE_BACKEND
    assert register_glm_hisparse_backend() is custom
    assert custom.get_class() is GlmHiSparseDCPBackend
    assert stock.get_path() == path
    assert stock.get_class() is FlashInferMLASparseSM120Backend
    assert GlmHiSparseDCPBackend.get_name() == stock.get_class().get_name()
    assert GlmHiSparseDCPBackend.supports_hisparse_dcp_layout(656, 4, 4)
    assert not GlmHiSparseDCPBackend.supports_hisparse_dcp_layout(528, 4, 4)
    assert not GlmHiSparseDCPAttentionImpl.lse_base_on_e


def test_glm_backend_consumes_rank_local_rows_in_both_layers(monkeypatch):
    group = object.__new__(HiSparseMLAIndexGroup)
    group.logical_topk_indices = torch.full((3, 2176), -1, dtype=torch.int32)
    group.logical_topk_indices[0, :6] = torch.tensor([3, 5, 1, 17, 20, 1])
    group.logical_topk_indices[1, :3] = torch.tensor([4, 8, 12])
    group.logical_topk_indices[0, 2048:2051] = torch.tensor([32, 33, 34])
    marks = []
    group.set_logical_topk_ready = lambda layer: marks.append(layer)
    raw = torch.zeros((2, 64, 656), dtype=torch.uint8)
    counts = torch.tensor([4, 0], dtype=torch.int32)
    q = torch.empty((2, 64, 512), dtype=torch.bfloat16)
    workspace = torch.empty(16, dtype=torch.uint8)
    metadata = SimpleNamespace(cp_kv_cache_interleave_size=4)
    layer = object()
    invoked = []

    def resolve(bound_group, layer_index, local, meta):
        assert bound_group is group
        assert meta is metadata
        assert local[0, :8].tolist() == [3, 1, 5, 1, 8, 9, 10, -1]
        assert (local[1] == -1).all()
        assert marks == [0]
        invoked.append(layer_index)
        return SimpleNamespace(kv_cache=raw, physical_topk=local, valid_counts=counts)

    expected = (torch.zeros_like(q), torch.full(q.shape[:2], float("-inf")))

    def attention(query, cache, indices, valid, scratch, *, qk_nope_head_dim, scale):
        assert query is q and cache is raw and valid is counts and scratch is workspace
        assert (qk_nope_head_dim, scale) == (256, 0.5)
        return expected

    monkeypatch.setattr(glm_sparse_backend, "run_sm120_rank_attention", attention)
    binding = GlmSparseAttentionBinding.create(group, resolve)

    def native_layer(index):
        instance = object.__new__(GlmHiSparseDCPAttentionImpl)
        instance.dcp_world_size, instance.dcp_rank, instance.pcp_world_size = 4, 0, 1
        instance.index_group, instance.index_group_index = group, index
        instance.qk_nope_head_dim, instance.qk_rope_head_dim = 256, 0
        instance.kv_lora_rank, instance.kv_scale_format = 512, "arbitrary_fp32"
        instance._workspace_buffer, instance.scale = workspace, 0.5
        return instance

    leader = native_layer(0)
    follower = native_layer(1)
    with pytest.raises(RuntimeError, match="resolver is not bound"):
        leader.forward_mqa(q, raw, metadata, layer)
    leader.bind_kitepond_hisparse(binding)
    follower.bind_kitepond_hisparse(binding)
    with pytest.raises(RuntimeError, match="immutable"):
        follower.bind_kitepond_hisparse(
            GlmSparseAttentionBinding.create(group, resolve)
        )
    assert leader.forward_mqa(q, raw, metadata, layer) is expected
    assert follower.forward_mqa(q, raw, metadata, layer) is expected
    assert invoked == [0, 1]
    assert group.logical_topk_indices[0, :6].tolist() == [3, 5, 1, 17, 20, 1]


def test_metadata_builder_carries_native_per_rank_prefill_plan(monkeypatch):
    native = SimpleNamespace(num_prefills=2, num_decodes=1, prefill=None)
    common = object()
    token_plan = object()
    calls = []

    def build(self, prefix, metadata, fast):
        assert (prefix, metadata, fast) == (0, common, True)
        return native

    def plan(metadata, *, num_decodes, num_prefills, dcp_rank):
        assert metadata is common
        calls.append((num_decodes, num_prefills, dcp_rank))
        return token_plan

    monkeypatch.setattr(
        glm_sparse_backend.FlashInferMLASparseMetadataBuilder, "build", build
    )
    monkeypatch.setattr(glm_sparse_backend, "plan_rank_local_prefill", plan)
    monkeypatch.setattr(
        glm_sparse_backend, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=3)
    )
    builder = object.__new__(GlmHiSparseDCPBackend.get_builder_cls())
    builder.dcp_world_size = 4
    result = builder.build(0, common, True)
    assert result.rank_local_prefill is token_plan
    assert calls == [(1, 2, 3)]
    builder.dcp_world_size = 1
    assert builder.build(0, common, True) is native
    assert calls == [(1, 2, 3)]
