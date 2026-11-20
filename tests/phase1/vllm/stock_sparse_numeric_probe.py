# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native 656-byte SM120 rank partials against a dense FP32 oracle."""

import json
import math

import torch
from kitepond.models.glm5next.hisparse_dcp import localize_hisparse_dcp_topk
from kitepond.models.glm5next.sparse_attention import run_sm120_rank_attention

from vllm import _custom_ops as ops
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import _get_workspace_buffer


def unpack_native_rows(rows):
    values = rows[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = rows[:, 512:528].contiguous().view(torch.float32)
    return (values.view(-1, 4, 128) * scales.unsqueeze(-1)).reshape(-1, 512)


@torch.inference_mode()
def run():
    torch.manual_seed(913)
    torch.cuda.set_per_process_memory_fraction(0.025, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    assert torch.cuda.get_device_capability(device) == (12, 0)
    query = torch.randn((3, 64, 512), dtype=torch.bfloat16, device=device) * 0.20
    scale = 0.125

    # Each rank has a native B256 page. Place rows on either side of B64 tiles.
    physical_slots = [
        0,
        1,
        62,
        63,
        64,
        65,
        126,
        127,
        128,
        191,
        192,
        255,
        3,
        66,
        193,
        254,
    ]
    cache_mapping = torch.tensor(physical_slots, dtype=torch.long, device=device)
    raw_slots = len(physical_slots)
    queries = [
        [0, 4, 7, 8, 15, 16, 32, 48, 36, 36, 47],
        [1, 2, 3, 17, 32, 34, 35],
        [6, 14, 30, 46],
    ]
    global_topk = torch.full((len(queries), 2048), -1, dtype=torch.int32, device=device)
    for i, chosen in enumerate(queries):
        global_topk[i, : len(chosen)] = torch.tensor(chosen, device=device)
    raw_nope = torch.empty((raw_slots, 0), dtype=torch.bfloat16, device=device)
    write_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    workspace = _get_workspace_buffer(device)
    per_rank_out, per_rank_lse, observed_counts, unpacked, physical = [], [], [], [], []
    for rank in range(4):
        values = torch.randn((raw_slots, 512), dtype=torch.bfloat16, device=device)
        raw_cache = torch.full((1, 256, 656), 255, dtype=torch.uint8, device=device)
        ops.concat_and_cache_mla(
            values, raw_nope, raw_cache, cache_mapping, "fp8_ds_mla", write_scale
        )
        physical.append(raw_cache)
        flat = raw_cache.view(256, 656)
        chosen = flat[cache_mapping]
        assert (chosen[:, 528:] == 0).all()
        assert (flat[2] == 255).all()
        unpacked.append(unpack_native_rows(chosen))
        local_positions, counts = localize_hisparse_dcp_topk(global_topk, dcp_rank=rank)
        indices = torch.where(
            local_positions >= 0,
            cache_mapping[local_positions.clamp_min(0)].int(),
            -1,
        )
        output, lse = run_sm120_rank_attention(
            query,
            raw_cache.view(-1, 64, 656),
            indices,
            counts,
            workspace,
            qk_nope_head_dim=256,
            scale=scale,
        )
        assert torch.isfinite(output).all()
        assert torch.isneginf(lse[counts == 0]).all()
        assert (output[counts == 0] == 0).all()
        per_rank_out.append(output.float())
        per_rank_lse.append(lse.float())
        observed_counts.append(counts.tolist())

    outputs, lses = torch.stack(per_rank_out), torch.stack(per_rank_lse)
    merged_lse = torch.logsumexp(lses * math.log(2), dim=0) / math.log(2)
    weights = torch.exp2(lses - merged_lse[None, ...]).unsqueeze(-1)
    merged = (weights * outputs).sum(dim=0)
    direct, direct_lse = [], []
    for i, chosen in enumerate(queries):
        values = torch.stack(
            [unpacked[pos // 4 % 4][pos // 16 * 4 + pos % 4] for pos in chosen]
        )
        logits = query[i].float() @ values.T * scale
        direct.append(torch.softmax(logits, dim=-1) @ values)
        direct_lse.append(torch.logsumexp(logits, dim=-1) / math.log(2))
    direct = torch.stack(direct)
    direct_lse = torch.stack(direct_lse)
    joined_topk = torch.full_like(global_topk, -1)
    for i, chosen in enumerate(queries):
        joined_topk[i, : len(chosen)] = torch.tensor(
            [
                pos // 4 % 4 * 256 + physical_slots[pos // 16 * 4 + pos % 4]
                for pos in chosen
            ],
            dtype=torch.int32,
            device=device,
        )
    joined_out, joined_lse = run_sm120_rank_attention(
        query,
        torch.cat(physical).view(-1, 64, 656),
        joined_topk,
        torch.tensor([len(row) for row in queries], dtype=torch.int32, device=device),
        workspace,
        qk_nope_head_dim=256,
        scale=scale,
    )
    # Isolate rank reduction error from the stock kernel's numerical error
    # against independently dequantized FP32 attention.
    shard_out_err = (merged - joined_out.float()).abs()
    shard_lse_err = (merged_lse - joined_lse.float()).abs()
    dense_err = (merged - direct).abs()
    max_out_error = dense_err.max().item()
    dense_tolerance = 0.045 + 0.035 * direct.abs()
    max_lse_error = (merged_lse - direct_lse).abs().max().item()
    print(
        json.dumps(
            {
                "scope": "one SM120 GPU, four synthetic native sparse shards",
                "storage_row_bytes": 656,
                "physical_page_rows": 256,
                "kernel_page_rows": 64,
                "counts_by_rank": observed_counts,
                "max_abs_output_error": max_out_error,
                "max_abs_base2_lse_error": max_lse_error,
                "stock_writer_zero_rope_tail": True,
                "empty_shard_output_zero_lse_neg_inf": True,
                "dense_out_p99_error": torch.quantile(dense_err.flatten(), 0.99).item(),
                "dense_out_strict_failures": int((dense_err > dense_tolerance).sum()),
                "stock_unsharded_out_max_diff": shard_out_err.max().item(),
                "stock_unsharded_lse_max_diff": shard_lse_err.max().item(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    torch.testing.assert_close(merged, joined_out.float(), atol=0.02, rtol=0.015)
    torch.testing.assert_close(merged_lse, joined_lse.float(), atol=0.015, rtol=0.006)
    torch.testing.assert_close(merged_lse, direct_lse, atol=0.06, rtol=0.015)


if __name__ == "__main__":
    run()
