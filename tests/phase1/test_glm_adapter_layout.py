# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from kitepond.models.glm5next.layout import (
    CacheIdentity,
    CacheRole,
    Glm5NextPDLayout,
    copy_kpool_tail,
    pack_kv_rows,
    slice_kda_conv,
    slice_kda_recurrent,
)


def layout():
    return Glm5NextPDLayout(
        request_id="layout-test",
        model_fingerprint="a" * 64,
        config_fingerprint="b" * 64,
        producer_owner=0,
        epoch=2,
        target_valid_tokens=9,
        draft_valid_tokens=9,
        committed_kda_slot=2,
        boundary_status="decode_ready",
    )


def test_native_mla_rows_preserve_all_bytes():
    for width in (528, 656):
        original = torch.arange(9 * (width + 16), dtype=torch.int64)
        original = (original % 251).to(torch.uint8).reshape(9, width + 16)
        view = original[:, 8 : 8 + width]
        identity = CacheIdentity("attention", CacheRole.MLA, "target")
        packed = pack_kv_rows(
            view, torch.arange(9), identity=identity, layout=layout(), dcp_rank=1
        )
        assert packed.local_row_ids.tolist() == list(range(4))
        assert packed.payload.shape == (4, width)
        assert torch.equal(packed.payload, view[4:8])


def test_indexer_uses_only_complete_rows():
    source = torch.arange(5 * 132).to(torch.uint8).reshape(5, 132)
    identity = CacheIdentity("indexer", CacheRole.INDEXER, "draft")
    packed = pack_kv_rows(
        source, torch.arange(5), identity=identity, layout=layout(), dcp_rank=2
    )
    assert packed.local_row_ids.numel() == 0
    assert packed.payload.shape == (0, 132)


def test_exact_state_preserves_conv_groups_recurrent_heads_and_tail():
    state = layout()
    conv = (torch.arange(6 * 24576).reshape(6, 24576) % 193).to(torch.bfloat16)
    recurrent = torch.arange(64 * 128 * 128, dtype=torch.float32).reshape(64, 128, 128)
    tail = (torch.arange(8 * 2 * 128).reshape(8, 2, 128) % 193).to(torch.bfloat16)
    selected_conv = slice_kda_conv(
        conv,
        identity=CacheIdentity("kda", CacheRole.KDA_CONV, "target"),
        layout=state,
        dcp_rank=1,
        source_committed_slot=0,
    )
    assert torch.equal(
        selected_conv,
        torch.cat(
            (conv[:, 2048:4096], conv[:, 10240:12288], conv[:, 18432:20480]), dim=1
        ),
    )
    selected_recurrent = slice_kda_recurrent(
        recurrent,
        identity=CacheIdentity("kda", CacheRole.KDA_RECURRENT, "target"),
        layout=state,
        dcp_rank=1,
        source_committed_slot=2,
    )
    assert torch.equal(selected_recurrent, recurrent[16:32])
    selected_tail = copy_kpool_tail(
        tail,
        identity=CacheIdentity("indexer.tail", CacheRole.KPOOL_TAIL, "draft"),
        layout=state,
    )
    assert torch.equal(selected_tail, tail)
    assert selected_tail.data_ptr() != tail.data_ptr()
