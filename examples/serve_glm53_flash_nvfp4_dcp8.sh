#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    printf 'Usage: %s MODEL_DIR KITEPOND_HOST_POLICY_JSON [PORT]\n' "$0" >&2
    exit 2
fi

model=$1
export KITEPOND_HISPARSE_DCP8_HOST_POOL_POLICY=$2
port=${3:-8000}
set --
host_pool_gib=384
gpu_memory_utilization=0.90
case "${KITEPOND_DCP8_MTP:-0}" in
    0) ;;
    3)
        set -- --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
        host_pool_gib=16
        gpu_memory_utilization=0.92
        ;;
    *)
        printf 'KITEPOND_DCP8_MTP must be 0 or 3\n' >&2
        exit 2
        ;;
esac

test -r "$model/config.json"
test -r "$KITEPOND_HISPARSE_DCP8_HOST_POOL_POLICY"
export HF_HUB_OFFLINE=1 VLLM_USE_RUST_FRONTEND=0
export VLLM_NO_USAGE_STATS=1 PYTHONHASHSEED=13417 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

capture='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,28,32,36,40,44,48,52,56,60,64,96,128,192,256,384,512,768,1024,1536,2048]}'
attention='{"backend_per_kind":{"mla_attention":"FLASHINFER_MLA_SPARSE_SM120"},"hisparse_config":{"enable_dcp8":true,"pack_hot_groups":true,"indexer_block_size":2048}}'
transfer=$(printf '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":%s}}' "$host_pool_gib")
provider='{"hisparse_host_buffer_provider":"kitepond.hisparse_host_dcp8.acquire"}'

exec kitepond serve "$model" --enable \
    --served-model-name glm-5.3-flash --host 127.0.0.1 --port "$port" --api-server-count 1 \
    --tensor-parallel-size 8 --decode-context-parallel-size 8 --enable-expert-parallel \
    --data-parallel-size 1 --data-parallel-size-local 1 --data-parallel-backend mp \
    --pipeline-parallel-size 1 --prefill-context-parallel-size 1 \
    --gpu-memory-utilization "$gpu_memory_utilization" --max-model-len 131072 --max-num-seqs 16 \
    --max-num-batched-tokens 2048 --enable-chunked-prefill --scheduler-reserve-full-isl \
    --compilation-config "$capture" --kv-cache-dtype fp8_ds_mla --block-size 256 \
    --cp-kv-cache-interleave-size 4 --dcp-comm-backend ag_rs \
    --all2all-backend allgather_reducescatter --distributed-executor-backend mp \
    --no-enable-prefix-caching --no-enable-extensible-kv-cache \
    --disable-flashinfer-autotune --language-model-only --mamba-cache-mode none \
    --cudagraph-metrics --tool-call-parser glm47 --reasoning-parser glm47 \
    --enable-auto-tool-choice --attention-config "$attention" \
    --kv-transfer-config "$transfer" --additional-config "$provider" \
    "$@"
