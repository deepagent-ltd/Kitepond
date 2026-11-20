# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from kitepond.models.glm5next.hisparse_dcp import localize_hisparse_dcp_topk
from kitepond.models.glm5next.hisparse_mirror import localize_row_mirrors

from vllm.v1.hisparse.types import SparseKVRowMirror


def test_rank_keeps_full_global_topk_order_and_duplicates_in_local_output():
    global_topk = torch.tensor([[7, 24, -1, 5, 7, 23, 0]], dtype=torch.int32)
    expected = global_topk.clone()
    local, counts = localize_hisparse_dcp_topk(global_topk, dcp_rank=1)
    assert torch.equal(global_topk, expected)
    assert local.tolist() == [[3, 1, 3, 7, -1, -1, -1]]
    assert counts.tolist() == [4]


def test_host_mirror_splits_a_logical_page_into_exact_rank_local_intervals():
    physical = 256
    logical = physical * 4
    source = SparseKVRowMirror((logical * 3 + 3,), logical * 7 + 3, 21)
    seen = []
    for rank in range(4):
        (local,) = localize_row_mirrors(
            (source,), block_size=physical, dcp_world_size=4, dcp_rank=rank
        )
        tokens = [t for t in range(3, 24) if (t // 4) % 4 == rank]
        offset = (tokens[0] // 16) * 4 + tokens[0] % 4
        assert local.source_starts == (physical * 3 + offset,)
        assert local.destination_start == physical * 7 + offset
        assert local.num_rows == len(tokens)
        seen.extend(tokens)
    assert sorted(seen) == list(range(3, 24))
    with pytest.raises(ValueError, match="logical page"):
        bad = SparseKVRowMirror((logical - 2,), logical - 2, 4)
        localize_row_mirrors((bad,), block_size=physical, dcp_world_size=4, dcp_rank=0)
