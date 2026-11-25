#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# != 1 ]]; then
    printf 'Usage: %s FULL_GLM_MODEL_DIRECTORY\n' "$0" >&2
    exit 2
fi

capture='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,96,128,256,512,1024,2048]}'
transfer='{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":384}}'
speculation='{"method":"mtp","num_speculative_tokens":3}'

exec kitepond serve "$1" --enable \
    --served-model-name glm-5.3 --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 8 --decode-context-parallel-size 8 --enable-expert-parallel \
    --cp-kv-cache-interleave-size 4 --dcp-comm-backend ag_rs --no-dcp-q-replicate \
    --distributed-executor-backend mp --block-size 64 --kv-cache-dtype fp8_ds_mla \
    --no-enable-prefix-caching --no-enable-extensible-kv-cache \
    --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.90 --dtype bfloat16 --cudagraph-metrics \
    --attention-backend FLASHINFER_MLA_SPARSE_SM120 --compilation-config "$capture" \
    --kv-transfer-config "$transfer" --speculative-config "$speculation"
