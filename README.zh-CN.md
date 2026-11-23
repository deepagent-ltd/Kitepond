# Kitepond 鸢池

**[中文](README.zh-CN.md)** · **[English](README.md)**

鸢池（Kitepond）是面向 vLLM 的分层 KV Cache 管理组件，按冷热统一调度 KV 在 GPU 显存、主存与
NVMe SSD 之间的驻留、复用与回收。它为 vLLM 0.34.0 补齐了 SM120 上 GLM 稀疏 MLA 的
HiSparse + DCP8 组合支持，让两种扩容方式在同一请求上叠加。HiSparse 在解码时从主存
按需召回历史 KV，压低单条请求的 GPU 常驻占用。DCP 沿序列维度将同一请求的 KV 分到多张卡，
扩大该请求可用的 KV 空间。冷会话的 KV 通过 LMCache FSL2 写入 SSD，恢复时读回主存。
P/D 分离部署时，鸢池使用 Mooncake 传输 KV，并承担缓存交接与 GLM 会话状态恢复。
当前适配与验证聚焦 SM120。

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

## 为什么选择鸢池（Kitepond）

- **让两种扩容方式在同一条请求上叠加。** DCP 把请求的历史 KV 分到多张卡，HiSparse 再缩小各卡
  的 GPU 常驻工作集。在 8 张 RTX 6000D 上以 90% 显存利用率运行 GLM-5.3-Flash NVFP4，
  TP8 + EP8 + DCP8 构型开启 Kitepond 后，单条 1 Mi token 请求的整机 KV 显存占用从
  **6.304 GiB 降至 0.976 GiB，下降 84.5%**。在相同 GPU KV 预算、主存充足时，该构型的
  整机 KV 容量从 **58.40 增至 375.05 Mi token，达到原来的 6.42 倍**。
- **补齐 SM120 上的 HiSparse + DCP8。** Kitepond 为 vLLM 0.34.0 的 GLM 稀疏 MLA 路径
  补充了按 rank 分片、主存召回和跨卡注意力结果归约，并接入外部主存池的分配与归还。
  [vLLM 0.34.0 配置检查](https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/config/vllm.py#L1810-L1822)
- **按冷热管理三级缓存。** HiSparse 管理活跃会话的 GPU 工作集和主存召回。LMCache FSL2
  将冷 KV 写到 NVMe SSD，恢复时读回主存。Kitepond 统一管理缓存的保留、复用与回收。
- **在 P/D 交接时复用历史 KV 和模型状态。** Mooncake 承担 KV 传输。Kitepond 协调两端的
  缓存所有权、增量复用与 GLM 会话状态交接。

## 容量与性能

在一台配备 8 张 RTX 6000D、GPU 显存共 672 GB 的服务器上，`--gpu-memory-utilization` 设为
`0.90`，GLM-5.3-Flash NVFP4 以 DP8 + EP8 运行。开启 Kitepond 后，单条 1 Mi token 请求的
整机 KV 显存占用从 **7.359 GiB 降至 1.387 GiB**。在相同的 GPU KV 预算下，整机容量从
24.35 Mi tokens 增至 129.21 Mi tokens，达到原来的 **5.3 倍**。

| GLM-5.3-Flash NVFP4 构型 | 显存计量范围 | 单条 1 Mi token 请求的 KV 显存（关闭 → 开启） | 相同 GPU KV 预算下的整机容量（关闭 → 开启） |
| --- | --- | ---: | ---: |
| DP8 + EP8 | 请求所属的单卡 | 7.359 → 1.387 GiB | 24.35 → 129.21 Mi tokens |
| TP8 + EP8 + DCP1 | 单卡，限制整个 TP 组的准入 | 6.685 → 0.579 GiB | 6.42 → 36.42 Mi tokens |
| TP8 + EP8 + DCP8 | 八卡分片合计 | 6.304 → 0.976 GiB | 58.40 → 375.05 Mi tokens |

DP8 的整机容量为各独立 DP rank 的容量之和，单条请求使用所属 rank 的池。DCP1 的八个
TP rank 对同一上下文共同准入，受最紧张的 rank 约束，整机 KVpool 按 TP 组计一次。
DCP8 将同一请求的历史 KV 分片到八张卡。末列假设主存充足，给出相同 GPU KV 预算下的
整机容量，对 TP 上的重复 KV 按逻辑容量对齐。配置 2 TB 主存的节点可以为表中 DP8 + EP8
容量提供足够的 KV 后备空间。1 Mi 等于 1,048,576。

下方示例使用 384 GiB 主存 KV 池和 131,072 token 的上下文上限，便于复现服务验证。
部署时可评估将 `--max-model-len` 调至 `1048576` 或将 `--max-num-seqs` 调至 `128`。
DP8 的 `--max-num-seqs` 按 rank 计。扩大服务上限前需在目标节点单独验证容量和延迟。

同一轮研究还在 vLLM 0.34.0 兼容构建上覆盖了 Flash FP8 和完整 GLM-5.3 ModelOpt NVFP4。
下面列出当前 WHL 的已验证配置。长上下文召回下的持续 Prefill 吞吐、稳态 Decode 吞吐、TTFT 和
TPOT 仍在测试。

## 已验证的环境

以下启动方法使用官方 Linux amd64 vLLM 0.34.0 镜像、8 张 NVIDIA RTX 6000D 和
GLM-5.3-Flash NVFP4 权重。表中各构型均在 GPU 显存利用率 90%、上下文上限 131,072 token
的条件下通过请求生成验证。

| 单机构型 | MTP 草稿 token 数 | HiSparse 主存 |
| --- | ---: | --- |
| DP8 + EP8 | 3 | 每个 DP rank 48 GiB，共 384 GiB |
| TP8 + EP8 + DCP1 | 3 | 共享 384 GiB |
| TP8 + EP8 + DCP8 | 0 | 每个 rank 48 GiB，共 384 GiB |

为模型加载、工作进程和操作系统另留主存。DP8 的请求及其 KV 都留在被分配的 DP rank。
本文覆盖单机服务。其他权重、GPU 和 P/D 部署需单独验证。

## 安装和运行 WHL

从 v0.2.0 起可在对应的 [GitHub Release](https://github.com/deepagent-ltd/Kitepond/releases)
下载 WHL 和 `SHA256SUMS`。

在 GPU 宿主机上准备 `kitepond-0.2.0-py3-none-any.whl` 和完整的模型权重。
替换示例中的两个本地路径后启动容器。

```bash
VLLM_IMAGE=vllm/vllm-openai@sha256:cf5b2fdd087eb4f6d261736967cc5cde974bd22ff4df50365f8fa419b6b4c484
MODEL_DIR=/absolute/path/to/GLM-5.3-FLASH-NVFP4
WHEEL_DIR=/absolute/path/to/wheel-directory

docker run --rm -it --gpus all --network host --shm-size 16g \
  --mount "type=bind,source=$MODEL_DIR,target=/models/GLM-5.3-FLASH-NVFP4,readonly" \
  --mount "type=bind,source=$WHEEL_DIR,target=/artifacts,readonly" \
  --entrypoint /bin/bash "$VLLM_IMAGE"
```

进入容器后安装 WHL，再启动 DP8 + EP8。
`kitepond serve MODEL [vLLM 参数] --enable` 为服务进程加载对应的兼容代码，调用固定基础镜像中的
CUDA 扩展。张量并行、数据并行、专家并行和解码上下文并行仍使用 vLLM 原生参数。
其他命令以及未加 `--enable` 的 `serve` 都走原生 vLLM CLI。加上 `--enable` 后，Kitepond 在加载权重前检查
本地模型、GPU 和并行构型，为所选构型补齐 HiSparse 设置，遇到冲突的配置会在启动时报错。

```bash
uv pip install --system --offline --no-index --no-deps \
  /artifacts/kitepond-0.2.0-py3-none-any.whl
kitepond serve --enable --help=all
```

已有启动命令时，更换命令前缀并添加 `--enable` 即可按原配置调用。
下面的 `vllm_args` 是由原启动参数组成的 Bash 数组，按原配置保留 `-dp`、`-tp` 和
`--decode-context-parallel-size` 等参数。

```bash
# 原启动命令: vllm serve "$MODEL" "${vllm_args[@]}"
kitepond serve "$MODEL" "${vllm_args[@]}" --enable
```

例如用 `kitepond serve "$MODEL" -dp 8 --enable` 启动 DP8，用
`kitepond serve "$MODEL" -tp 8 --enable` 启动 TP8，需要 DCP8 时再加
`--decode-context-parallel-size 8`。已验收的构型都使用 EP，未指定时 Kitepond 会开启 EP。
TP8/DCP1 使用默认主存池时，容器需配置 `--shm-size 416g`。

在容器中运行以下完整命令可启动 DP8 + EP8。

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

启动完成后，在 GPU 宿主机的另一个终端验证服务。

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
```

仓库提供 WHL 安装后的 [DP8](examples/serve_glm53_flash_nvfp4_dp8.sh) 和
[TP8 + DCP8](examples/serve_glm53_flash_nvfp4_dcp8.sh) 启动脚本。
给脚本传入容器内的模型路径和 policy JSON 路径。policy 文件所在目录须可写，
工作进程会在其旁边创建租约锁文件。

### 在 TP8 + DCP8 上启用 MTP3

DCP8 脚本默认使用 MTP0。使用 GLM-5.3-Flash NVFP4 时，可设置
`KITEPOND_DCP8_MTP=3` 启用模型自带的 MTP 层。此配置使用 16 GiB HiSparse
主存池和 92% 的 GPU 显存利用率。用以下命令创建可写的 policy 目录并启动。

```bash
policy_dir=$(mktemp -d)
printf '%s\n' '{"rank_bytes":2147483648,"total_bytes":17179869184}' > "$policy_dir/policy.json"
KITEPOND_DCP8_MTP=3 bash examples/serve_glm53_flash_nvfp4_dcp8.sh \
  /models/GLM-5.3-FLASH-NVFP4 "$policy_dir/policy.json"
```

自定义 DCP8 启动命令可加上
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`。
Kitepond 会设置 draft TP8 和 draft DCP8。源块大小仍为 256，索引块大小为 2048，
图模式为 `FULL_AND_PIECEWISE`。此选项的上限为 131,072 个上下文 token、
16 条序列和 2,048 个批处理 token。上文公布的容量数字使用 MTP0。

### TP8 + DCP1

使用 TP8 + DCP1 时，在启动容器时设置 `--shm-size 416g` 供原生共享池使用。
在上面的 DP8 命令中，把 tensor parallel size 改为 8，两个 data parallel size 都改为 1，
decode context parallel size 保持为 1。设置
`attention='{"hisparse_config":{"indexer_block_size":2048}}'` 和
`transfer='{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":384}}'`。
这个构型使用原生共享池，无需外部租约 policy。

## 从源码构建 WHL

在安装了 Git 和 Python 3.12 的 Linux 宿主机上，从干净的 Kitepond 源码目录执行以下命令。
构建读取当前 Git 提交和固定版本的 vLLM 源码，并使用从上文官方镜像采集的兼容性指纹。
构建过程无需 Docker 或 GPU。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install setuptools==80.9.0
git clone --depth 1 --branch v0.34.0 https://github.com/vllm-project/vllm.git ../vllm-v034
test "$(git -C ../vllm-v034 rev-parse HEAD)" = 62d4bc3c7e33b1b6c906e233e6af2d3506c482ca
bash tools/build_release.sh ../vllm-v034 dist
(cd dist && sha256sum --check SHA256SUMS)
```

发布流程从 Release tag 构建，tag 必须与 `pyproject.toml` 中的版本对应，格式为
`v<版本号>`。对于可变 Release，发布后工作流会附上 WHL、源码包和校验文件。
对于不可变 Release，请先为现有 tag 创建草稿。手动运行工作流并使用默认输入，
可在该次运行的 Artifacts 中下载 WHL、源码包和校验文件。独立核验后，针对该 tag
再次运行 `release-wheel.yml`，设置 `stage_draft=true`，在
`expected_source_commit` 中填写完整的 Git commit，在
`expected_manifest_sha256` 中填写经核验的 `SHA256SUMS` 文件的 SHA-256。
等三个附件上传并校验后再发布草稿。

## vLLM 兼容补丁

源码构建时将 `compat/vllm-v034/` 下的补丁应用到 vLLM 0.34.0，并把生成的 Python
文件打入 WHL。补丁涉及 HiSparse 主存租约、SM120 DCP8 和原生 MTP3、调度器与工作进程生命周期、
Mooncake 交接和主存页保留。`kitepond serve` 带 `--enable` 启动时校验固定的官方 vLLM 镜像，
只为该服务进程加载兼容文件。这是上游接口合入前的过渡方案。我们希望尽快将所需接口合入
vLLM 主线，让用户能用 vLLM 原生命令启用 Kitepond。请从上文验证过的模型与构型中选择启动配置。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `src/kitepond/core/` | 缓存生命周期、查找和就绪状态 |
| `src/kitepond/models/` | 各模型的状态适配器 |
| `src/kitepond/backends/` 和 `src/kitepond/integrations/` | 存储后端与连接器集成 |
| `compat/vllm-v034/` 和 `tools/` | vLLM 兼容补丁和 WHL 构建工具 |
| `examples/` 和 `tests/` | 启动脚本和回归测试 |

## 上游集成与贡献

我们正在准备向 vLLM 提交 HiSparse 外部主存租约、对应的调度器与工作进程接口，以及
SM120 在 DCP 下按 rank 执行 HiSparse 的改动。DCP 改动正在与已有的驻留稀疏 MLA 工作
核对。以下是拟提交改动的审阅分支。 [vLLM fork 分支](https://github.com/deepagent-ltd/vllm/tree/kitepond/hisparse-integration) ·
[相关 vLLM PR #66909](https://github.com/vllm-project/vllm/pull/66909)

下一步工作包括长上下文召回性能测试，以及 DP4 + EP4 与 TP4 + DCP4 + EP4 构型下的 P/D 交接。

欢迎贡献。提交兼容性反馈或性能结果时，请附上 vLLM 构建版本、模型版本与量化方式、GPU 型号和数量、
并行配置、完整启动命令和测试命令。 [Issues](https://github.com/deepagent-ltd/Kitepond/issues)

[Apache License 2.0](LICENSE)
