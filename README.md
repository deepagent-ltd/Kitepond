# Kitepond 鸢池

**[中文](README.zh-CN.md)** · **[English](README.md)**

Kitepond (鸢池) is a tiered KV cache manager for vLLM. It manages KV residency, reuse,
and reclamation across GPU memory, host RAM, and NVMe SSDs according to how
actively the cache is used. Kitepond adds combined HiSparse and DCP8 support
for GLM sparse MLA on SM120 in vLLM 0.34.0, so a single request can benefit
from both. HiSparse recalls historical KV from host RAM on demand during
decode and reduces GPU residency. DCP shards a request's KV across GPUs
along the sequence dimension, increasing the KV space available to that
request. LMCache FSL2 saves cold KV to SSD and reloads it into RAM when a
session resumes. In disaggregated prefill and decode deployments, Kitepond
uses Mooncake for KV transfer and handles cache handoff and GLM session
state restoration. Current integration and validation focus on SM120.

鸢池（Kitepond）是面向 vLLM 的分层 KV Cache 管理组件，按冷热统一调度 KV 在 GPU 显存、主存与
NVMe SSD 之间的驻留、复用与回收。它为 vLLM 0.34.0 补齐了 SM120 上 GLM 稀疏 MLA 的
HiSparse + DCP8 组合支持，让两种扩容方式在同一请求上叠加。HiSparse 在解码时从主存
按需召回历史 KV，压低单条请求的 GPU 常驻占用。DCP 沿序列维度将同一请求的 KV 分到多张卡，
扩大该请求可用的 KV 空间。冷会话的 KV 通过 LMCache FSL2 写入 SSD，恢复时读回主存。
P/D 分离部署时，鸢池使用 Mooncake 传输 KV，并承担缓存交接与 GLM 会话状态恢复。
当前适配与验证聚焦 SM120。

## Why Kitepond

- **Combine two ways to expand KV capacity for one request.** DCP distributes
  historical KV across GPUs, and HiSparse shrinks the resident working set on
  each GPU. On eight RTX 6000D GPUs serving GLM-5.3-Flash NVFP4 at 90% GPU memory
  utilization, enabling Kitepond on TP8 + EP8 + DCP8 cut total GPU KV memory
  for one 1 Mi-token request by **84.5%, from 6.304 GiB to 0.976 GiB**. At the
  same GPU KV budget and with sufficient host RAM, capacity for that
  configuration grew **6.42×, from 58.40 to 375.05 Mi tokens**.
- **Run HiSparse and DCP8 together on SM120.** Kitepond adds rank-local KV
  sharding, host recall, and cross-GPU attention reduction for GLM sparse MLA
  in vLLM 0.34.0. It also integrates external host memory allocation and
  release. [vLLM 0.34.0 configuration check](https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/config/vllm.py#L1810-L1822)
- **Manage active and cold KV across three tiers.** HiSparse manages the GPU
  working set and host recall for active sessions. LMCache FSL2 stores cold
  KV on NVMe SSD and reloads it into host RAM when a session resumes.
  Kitepond coordinates retention, reuse, and reclamation across the tiers.
- **Reuse historical KV and model state across prefill and decode.** Mooncake
  handles KV transfer. Kitepond coordinates cache ownership, incremental
  reuse, and GLM session state handoff between the stages.

## Capacity and performance

On a fully populated eight-card RTX 6000D node with 672 GB of GPU memory
and `--gpu-memory-utilization 0.90`, Kitepond reduced
the GPU memory charged per 1 Mi-token GLM-5.3-Flash NVFP4 request from
**7.359 to 1.387 GiB across the node** for DP8 + EP8. At a matched GPU KV
budget, the node's KV capacity increased **5.3×**, from
24.35 to 129.21 Mi tokens.

| GLM-5.3-Flash NVFP4 topology | GPU measurement scope | GPU KV memory for one 1 Mi-token request, off → on | Node capacity at matched GPU KV budget, off → on |
| --- | --- | ---: | ---: |
| DP8 + EP8 | One GPU that owns the request | 7.359 → 1.387 GiB | 24.35 → 129.21 Mi tokens |
| TP8 + EP8 + DCP1 | Per GPU, limiting admission for the TP group | 6.685 → 0.579 GiB | 6.42 → 36.42 Mi tokens |
| TP8 + EP8 + DCP8 | Sum of the eight GPU shards | 6.304 → 0.976 GiB | 58.40 → 375.05 Mi tokens |

DP8 node capacity sums the independent DP ranks, and each request uses its
owning rank's pool. With DCP1, all eight TP ranks admit the same context.
The rank with the smallest capacity limits the group, and node capacity
counts the group once. DCP8 shards one request's historical KV across
eight GPUs. The last column assumes sufficient host RAM and matches the
GPU KV budget using logical capacity for replicated TP KV. A node with
2 TB of host RAM can provision enough KV backing for the DP8 + EP8 result.
One Mi is 1,048,576 tokens.

The launch examples below use a 384 GiB host KV pool and a 131,072-token context
cap to reproduce the verified setup. For other deployments, you can evaluate
`--max-model-len 1048576` or `--max-num-seqs 128`. DP8 applies
`--max-num-seqs` per rank. Validate capacity and latency on the target node
before raising these serving limits.

The same study covered Flash FP8 and full GLM-5.3 ModelOpt NVFP4 on a
vLLM 0.34.0 compatibility build. The supported wheel configurations are
listed below. Sustained prefill throughput, steady-state decode throughput,
TTFT, and TPOT under long-context recall are still being benchmarked.

## Verified setup

The launch instructions below use the official Linux amd64 vLLM 0.34.0 image,
eight NVIDIA RTX 6000D GPUs, and the GLM-5.3-Flash NVFP4 checkpoint.
Each profile passed request generation at 90% GPU memory utilization and a
131,072-token context limit.

| Single-node profile | MTP draft tokens | HiSparse host RAM |
| --- | ---: | --- |
| DP8 + EP8 | 3 | 48 GiB per DP rank, 384 GiB total |
| TP8 + EP8 + DCP1 | 3 | One 384 GiB shared pool |
| TP8 + EP8 + DCP8 | 0 | 48 GiB per rank, 384 GiB total |

Allow additional host RAM for the model loader, workers, and operating
system. DP8 requests and their KV remain on their assigned DP rank. The
instructions in this README cover single-node serving. Other checkpoints,
GPUs, and P/D deployment require separate qualification.

## Install and run the wheel

From v0.2.0 onward, download the WHL and `SHA256SUMS` from the matching
[GitHub Release](https://github.com/deepagent-ltd/Kitepond/releases).

Have `kitepond-0.2.0-py3-none-any.whl` and the complete model checkpoint on
the GPU host. Set the two paths in this example and start the container.

```bash
VLLM_IMAGE=vllm/vllm-openai@sha256:cf5b2fdd087eb4f6d261736967cc5cde974bd22ff4df50365f8fa419b6b4c484
MODEL_DIR=/absolute/path/to/GLM-5.3-FLASH-NVFP4
WHEEL_DIR=/absolute/path/to/wheel-directory

docker run --rm -it --gpus all --network host --shm-size 16g \
  --mount "type=bind,source=$MODEL_DIR,target=/models/GLM-5.3-FLASH-NVFP4,readonly" \
  --mount "type=bind,source=$WHEEL_DIR,target=/artifacts,readonly" \
  --entrypoint /bin/bash "$VLLM_IMAGE"
```

In the container, install the wheel and start the DP8 + EP8 profile.
`kitepond serve MODEL [vLLM options] --enable` loads the matching compatibility
code for the server process and uses the CUDA extensions from the pinned base
image. Keep the native vLLM tensor, data, expert, and decode context parallelism
flags. Other commands and `serve` without `--enable` use the original vLLM CLI.
With `--enable`, Kitepond checks the local checkpoint, GPU type, and topology
before loading weights. It supplies the required HiSparse settings for that
topology and reports conflicting configuration values as startup errors.

```bash
uv pip install --system --offline --no-index --no-deps \
  /artifacts/kitepond-0.2.0-py3-none-any.whl
kitepond serve --enable --help=all
```

To enable Kitepond on an existing launch, change the command prefix and add
`--enable`. In the example below, `vllm_args` is the Bash array of options
from your existing command, including `-dp`, `-tp`, and
`--decode-context-parallel-size` where applicable.

```bash
# Existing command: vllm serve "$MODEL" "${vllm_args[@]}"
kitepond serve "$MODEL" "${vllm_args[@]}" --enable
```

For example, start DP8 with `kitepond serve "$MODEL" -dp 8 --enable`. Start
TP8 with `kitepond serve "$MODEL" -tp 8 --enable` and add
`--decode-context-parallel-size 8` for DCP8. Kitepond fills in EP if omitted
because the verified configurations require it. The TP8/DCP1 profile needs
`--shm-size 416g` with the default host pool.

The complete DP8 + EP8 launch below can be run in the container.

```bash
policy_dir=$(mktemp -d)
export KITEPOND_HISPARSE_HOST_POOL_POLICY="$policy_dir/policy.json"
cat > "$KITEPOND_HISPARSE_HOST_POOL_POLICY" <<'JSON'
{"rank_bytes":51539607552,"total_bytes":412316860416}
JSON

export HF_HUB_OFFLINE=1 VLLM_USE_RUST_FRONTEND=0
export VLLM_NO_USAGE_STATS=1 PYTHONHASHSEED=13417 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

capture='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,28,32,36,40,44,48,52,56,60,64,96,128,192,256,384,512,768,1024,1536,2048]}'
transfer='{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":48,"host_pool_allocator":"kitepond.hisparse_host.acquire"}}'
speculate='{"method":"mtp","num_speculative_tokens":3}'
attention='{"hisparse_config":{"indexer_block_size":8192}}'

kitepond serve /models/GLM-5.3-FLASH-NVFP4 --enable \
  --served-model-name glm-5.3-flash --host 127.0.0.1 --port 8000 --api-server-count 1 \
  --tensor-parallel-size 1 --decode-context-parallel-size 1 --enable-expert-parallel \
  --data-parallel-size 8 --data-parallel-size-local 8 --data-parallel-backend mp \
  --pipeline-parallel-size 1 --prefill-context-parallel-size 1 \
  --gpu-memory-utilization 0.90 --max-model-len 131072 --max-num-seqs 16 \
  --max-num-batched-tokens 2048 --enable-chunked-prefill --scheduler-reserve-full-isl \
  --compilation-config "$capture" --kv-cache-dtype fp8 \
  --all2all-backend allgather_reducescatter --no-enable-prefix-caching \
  --attention-backend FLASHINFER_MLA_SPARSE_SM120 --disable-flashinfer-autotune \
  --language-model-only --mamba-cache-mode none --cudagraph-metrics \
  --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice \
  --kv-transfer-config "$transfer" --speculative-config "$speculate" --attention-config "$attention"
```

Once startup completes, check the service from another shell on the GPU host.

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
```

The repository provides equivalent [DP8](examples/serve_glm53_flash_nvfp4_dp8.sh)
and [TP8 + DCP8](examples/serve_glm53_flash_nvfp4_dcp8.sh) launch scripts for
an installed wheel. Pass the in-container model path and policy JSON path to
either script. Use a writable directory for the policy so workers can create
lease locks next to it.

### Opt in to MTP3 on TP8 + DCP8

The DCP8 script defaults to MTP0. On the GLM-5.3-Flash NVFP4 profile,
set `KITEPOND_DCP8_MTP=3` to use the checkpoint's native MTP layer.
The script uses a 16 GiB HiSparse host pool and 92% GPU memory utilization
for this profile. Create a writable policy directory and run the exact
profile with the following commands.

```bash
policy_dir=$(mktemp -d)
printf '%s\n' '{"rank_bytes":2147483648,"total_bytes":17179869184}' > "$policy_dir/policy.json"
KITEPOND_DCP8_MTP=3 bash examples/serve_glm53_flash_nvfp4_dcp8.sh \
  /models/GLM-5.3-FLASH-NVFP4 "$policy_dir/policy.json"
```

For a custom DCP8 command, pass
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.
Kitepond supplies draft TP8 and draft DCP8. Keep source blocks at 256,
indexer blocks at 2048, and graph mode at `FULL_AND_PIECEWISE`.
This opt-in is limited to 131,072 context tokens, 16 sequences, and 2,048
batched tokens. The capacity figures above use MTP0.

### TP8 + DCP1

For TP8 + DCP1, start the container with `--shm-size 416g` for the native
shared pool. In the inline DP8 command, set tensor parallel size to 8 and
both data parallel sizes to 1. Keep decode context parallel size at 1.
Use `attention='{"hisparse_config":{"indexer_block_size":2048}}'` and
`transfer='{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":384}}'`.
This profile uses the native shared pool and needs no external lease policy.

## Build the wheel from source

Run from a clean Kitepond checkout on a Linux host with Git and Python 3.12.
The builder packages the current Git commit with the pinned vLLM source and
uses stock compatibility fingerprints captured from the official image listed
above. Building the wheel does not need Docker or a GPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install setuptools==80.9.0
git clone --depth 1 --branch v0.34.0 https://github.com/vllm-project/vllm.git ../vllm-v034
test "$(git -C ../vllm-v034 rev-parse HEAD)" = 62d4bc3c7e33b1b6c906e233e6af2d3506c482ca
bash tools/build_release.sh ../vllm-v034 dist
(cd dist && sha256sum --check SHA256SUMS)
```

The release workflow builds from the release tag. The tag must be `v`
followed by the package version in `pyproject.toml`. Publishing the release
attaches the wheel, source archive and checksums. A manual Actions run builds
the same files and stores them in the run's artifacts.

## vLLM compatibility patches

The source build applies the patches in `compat/vllm-v034/` to vLLM
0.34.0 and packages the resulting Python files inside the wheel. They cover
HiSparse host memory leases, SM120 DCP8 and native MTP3, scheduler and worker
lifecycle, and Mooncake handoff and host-page retention. The wheel checks the pinned
official vLLM image when `kitepond serve` is invoked with `--enable` and loads
the compatibility files for that server process. This is the installation path
until the required extension interfaces land upstream. We aim to bring this
integration to vLLM main so users can enable Kitepond through the native CLI.
Use one of the verified launch configurations above for the model and topology.

## Repository layout

| Path | Content |
| --- | --- |
| `src/kitepond/core/` | Cache lifecycle, lookup, and readiness |
| `src/kitepond/models/` | Model-specific state adapters |
| `src/kitepond/backends/` and `src/kitepond/integrations/` | Storage and connector integrations |
| `compat/vllm-v034/` and `tools/` | vLLM compatibility patches and wheel build tools |
| `examples/` and `tests/` | Launch scripts and regression tests |

## Upstream integration and contributions

We are preparing a vLLM contribution for external HiSparse host memory
leases and the scheduler/worker interfaces needed to use them, together
with rank-local SM120 HiSparse execution under DCP. We are checking the
DCP changes against the open resident sparse MLA work. The branch below
is the review branch for the proposed vLLM changes.
[vLLM fork branch](https://github.com/deepagent-ltd/vllm/tree/kitepond/hisparse-integration) ·
[Related vLLM PR #66909](https://github.com/vllm-project/vllm/pull/66909)

Next work includes long-context recall benchmarks and P/D handoff across
DP4 + EP4 and TP4 + DCP4 + EP4 layouts.

Contributions are welcome. For a compatibility report or benchmark, include
the vLLM build, model revision and quantization, GPU model and count,
parallel settings, and the exact launch and workload commands.
[Issues](https://github.com/deepagent-ltd/Kitepond/issues)

[Apache License 2.0](LICENSE)
