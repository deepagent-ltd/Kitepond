# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from kitepond.models.glm5next import sparse_attention


@pytest.mark.parametrize("row_bytes", [528, 656])
def test_rank_local_attention_passes_native_rows_and_clears_empty_rank_lse(
    monkeypatch, row_bytes
):
    cache = torch.zeros((2, 64, row_bytes), dtype=torch.uint8)
    valid = torch.tensor([2, 0], dtype=torch.int32)

    def kernel(**kwargs):
        assert kwargs["kv_cache"].data_ptr() == cache.data_ptr()
        assert kwargs["kv_cache"].shape == (2, 1, 64, row_bytes)
        assert kwargs["seq_lens"] is valid
        assert kwargs["return_lse"] is True and kwargs["return_lse_base"] == "base2"
        assert kwargs["backend"] == "sparse"
        kwargs["out"].fill_(3)
        return kwargs["out"], torch.ones((2, 64), dtype=torch.float32)

    monkeypatch.setattr(
        sparse_attention, "flashinfer_trtllm_batch_decode_with_kv_cache_mla", kernel
    )
    output, lse = sparse_attention.run_sm120_rank_attention(
        torch.zeros((2, 64, 512), dtype=torch.bfloat16),
        cache,
        torch.tensor([[0, 5], [-1, -1]], dtype=torch.int32),
        valid,
        torch.zeros(1024, dtype=torch.uint8),
        qk_nope_head_dim=128,
        scale=0.1,
    )
    assert (output[0] == 3).all() and (output[1] == 0).all()
    assert (lse[0] == 1).all() and torch.isneginf(lse[1]).all()
