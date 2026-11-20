# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
import torch
from kitepond.integrations.mooncake.window import manifest_digest, plan_windows
from kitepond.integrations.vllm.glm_terminal_manifest import FencedTerminalSource
from kitepond.models.glm5next.cold_restore import ColdRestoreLayout, RestoreLayer
from kitepond.models.glm5next.import_recipe import part_identity
from kitepond.models.glm5next.layout import CacheIdentity, CacheRole, Glm5NextPDLayout
from kitepond.models.glm5next.live_windows import GlmRankWindowPlan, rank_window_records

_INDEXER = CacheIdentity("model.blocks.1.indexer", CacheRole.INDEXER, "target")
_PAGE_ROWS, _ROW_BYTES = 64, 132
_KEYS, _SCALE = 128, 4


def _layout(boundary):
    return Glm5NextPDLayout(
        "native", "b" * 64, "c" * 64, 0, 3, boundary, boundary, 0, "decode_ready"
    )


def _cache(layout, rank, blocks, fill):
    tensor = torch.full((8, 1, _PAGE_ROWS, _ROW_BYTES), fill, dtype=torch.uint8)
    ref = RestoreLayer(_INDEXER, tensor, blocks)
    return ColdRestoreLayout(layout, rank, (ref,)), tensor.view(8, -1)


def _populate_native(pages, blocks, boundary, rank):
    rows = []
    for pool in range(rank, boundary // 4, 4):
        local = pool // 4
        # Mirror the stock CUDA KPool writer: keys first, then scales at 8192.
        page, offset = divmod(local, _PAGE_ROWS)
        k = (torch.arange(_KEYS, dtype=torch.int) + 31 + pool) % 255
        ks = k.to(torch.uint8)
        scale = torch.tensor([0.125 + pool / 256], dtype=torch.float32).view(
            torch.uint8
        )
        dst_key = offset * _KEYS
        dst_scale = _PAGE_ROWS * _KEYS + offset * _SCALE
        pages[blocks[page], dst_key : dst_key + _KEYS] = ks
        pages[blocks[page], dst_scale : dst_scale + _SCALE] = scale
        rows.append(torch.cat((ks, scale)))
    return torch.stack(rows)


@pytest.mark.parametrize("boundary", (1032, 2064, 4128))
@pytest.mark.parametrize("rank", range(4))
def test_indexer_streams_only_actual_stock_key_and_scale_bytes(boundary, rank):
    layout = _layout(boundary)
    source_blocks = (7, 3, 5, 2, 1)
    destination_blocks = (1, 4, 6, 2, 7)
    source, source_pages = _cache(layout, rank, source_blocks, 211)
    expected = _populate_native(source_pages, source_blocks, boundary, rank)
    fenced = FencedTerminalSource(source, "a" * 64, "b" * 64, ())

    cold, cold_pages = _cache(replace(layout, epoch=4), rank, destination_blocks, 221)
    reference = torch.full_like(cold_pages, 221)
    _populate_native(reference, destination_blocks, boundary, rank)

    # FSL2 can split across physical pages. Verify direct stock offsets,
    # so using the same wrong read and write cannot make the probe pass.
    for part in source.manifest(part_bytes=7 * _ROW_BYTES):
        payload = fenced.read_chunk(part)
        want = expected[part.index : part.index + part.shape[0]].flatten()
        assert torch.equal(payload, want), (
            f"rank {rank} boundary {boundary}: wire bytes differ from stock Indexer keys/scales"
        )
        cold.scatter_window((part,), payload)
    assert torch.equal(cold_pages, reference)

    # The positive warm reverse and ordinary forward use the same format.
    outgoing = GlmRankWindowPlan(source, 0)
    for read_final in (None, fenced.read_chunk):
        live, live_pages = _cache(layout, rank, destination_blocks, 221)
        incoming = GlmRankWindowPlan(live, 0)
        assert outgoing.manifest() == incoming.manifest()
        for window in plan_windows(outgoing.manifest()):
            payload = torch.empty(window.nbytes, dtype=torch.uint8)
            assert (
                outgoing.pack(window, payload, read_final=read_final) == window.nbytes
            )
            assert incoming.scatter(window, payload) == window.nbytes
            incoming.verify_readback(window, payload)
        assert torch.equal(live_pages, reference)


def test_legacy_indexer_wire_cannot_match_a_corrected_peer():
    native, _ = _cache(_layout(2064), 0, (7, 3, 5), 211)
    (current,) = native.manifest()
    legacy = replace(current, role=f"target:{_INDEXER.layer_name}:indexer")
    assert legacy.nbytes == current.nbytes
    assert manifest_digest(rank_window_records((("final", legacy),))) != (
        manifest_digest(GlmRankWindowPlan(native, 0).manifest())
    )
    assert part_identity(current.role) == _INDEXER
    with pytest.raises(ValueError, match="Legacy GLM Indexer parts"):
        part_identity(legacy.role)
