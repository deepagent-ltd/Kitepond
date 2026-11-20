"""Build a native GLM-5.3-Flash DP8+EP8 launch plan."""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .checkpoints import FLASH_ARCHITECTURE as REFERENCE_ARCHITECTURE
from .checkpoints import INDEX_TOPK as REFERENCE_TOPK
from .checkpoints import CheckpointKind, inspect_checkpoint, model_json as model_json
from .hisparse_host import DP_SIZE as DP_RANKS
from .hisparse_host import POLICY_ENV, read_policy

VLLM_VERSION = "0.34.0"
REFERENCE_IMAGE = (
    "vllm/vllm-openai@sha256:"
    "cf5b2fdd087eb4f6d261736967cc5cde974bd22ff4df50365f8fa419b6b4c484"
)
CapacityComparison = Literal["stock-resident", "indexer2048"]
CAPACITY_CAPTURE_SIZES = (
    *range(1, 17), *range(20, 65, 4), 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
)


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    context_limit: int
    shard_count: int
    index_topk: int

    @classmethod
    def inspect(cls, model_dir: Path) -> Checkpoint:
        metadata = inspect_checkpoint(model_dir)
        if metadata.kind not in (CheckpointKind.FLASH_NVFP4, CheckpointKind.FLASH_FP8):
            raise ValueError("The DP8 launch profile requires GLM-5.3-Flash NVFP4 or FP8")
        return cls(
            metadata.path, metadata.context_limit, metadata.shard_count, metadata.index_topk
        )


@dataclass(frozen=True)
class LaunchProfile:
    checkpoint: Checkpoint
    host_pool_gib_per_rank: int
    host_pool_budget_gib: int
    served_model_name: str
    host: str
    port: int
    devices: tuple[int, ...]
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    mtp: int
    prefix_caching: bool
    stock_topk_override: bool
    external_host_policy: Path | None = None
    capacity_comparison: CapacityComparison | None = None

    def __post_init__(self) -> None:
        if self.capacity_comparison not in (None, "stock-resident", "indexer2048"):
            raise ValueError("Unsupported capacity comparison arm")
        if self.capacity_comparison == "stock-resident":
            if self.external_host_policy is not None:
                raise ValueError("The stock-resident arm does not use an external host policy")
            if self.host_pool_gib_per_rank or self.host_pool_budget_gib:
                raise ValueError("The stock-resident arm has zero host pool")
        elif self.host_pool_gib_per_rank <= 0 or self.host_pool_budget_gib <= 0:
            raise ValueError("Host-pool limits must be positive GiB")
        if self.capacity_comparison == "indexer2048" and self.external_host_policy is None:
            raise ValueError("The indexer2048 arm requires an external host policy")
        if self.total_host_pool_gib > self.host_pool_budget_gib:
            raise ValueError(
                f"Requested {self.total_host_pool_gib} GiB host pool across eight DP "
                f"ranks exceeds the {self.host_pool_budget_gib} GiB budget"
            )
        if self.external_host_policy is not None:
            if self.stock_topk_override:
                raise ValueError("External host policy requires the vLLM fork")
            _, rank_bytes, total_bytes = read_policy(self.external_host_policy)
            if (
                rank_bytes != self.host_pool_gib_per_rank << 30
                or total_bytes != self.host_pool_budget_gib << 30
            ):
                raise ValueError("External host policy and launch budgets must match")
        if (
            len(self.devices) != DP_RANKS
            or len(set(self.devices)) != DP_RANKS
            or min(self.devices) < 0
        ):
            raise ValueError("Specify exactly eight distinct non-negative GPU indices")
        if not self.host.strip() or not self.served_model_name.strip():
            raise ValueError("Host and served model name must be non-empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("Port must be in 1..65535")
        if not 0 < self.max_model_len <= self.checkpoint.context_limit:
            raise ValueError(
                f"Model length must be in 1..{self.checkpoint.context_limit}"
            )
        if not 0 < self.max_num_seqs <= self.max_num_batched_tokens:
            raise ValueError("Batch token budget must cover at least one token per sequence")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("GPU memory utilization must be in (0, 1)")
        if self.mtp not in (0, 3):
            raise ValueError("This profile supports MTP disabled or three draft tokens")
        if self.mtp and self.max_num_seqs * (self.mtp + 1) > self.max_num_batched_tokens:
            raise ValueError("Batch token budget must cover the MTP verification width")

    @property
    def total_host_pool_gib(self) -> int:
        return self.host_pool_gib_per_rank * DP_RANKS

    def environment(self) -> dict[str, str]:
        environment = {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, self.devices)),
            "HF_HUB_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "VLLM_USE_RUST_FRONTEND": "0",
        }
        if self.external_host_policy is not None:
            environment[POLICY_ENV] = str(self.external_host_policy.resolve())
        return environment

    def argv(self) -> list[str]:
        compilation: dict[str, str | tuple[int, ...]] = {
            "cudagraph_mode": "FULL_AND_PIECEWISE",
        }
        if self.capacity_comparison is not None:
            compilation["cudagraph_capture_sizes"] = CAPACITY_CAPTURE_SIZES
        argv = [
            "vllm", "serve", str(self.checkpoint.path),
            "--served-model-name", self.served_model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--api-server-count", "1",
            "--tensor-parallel-size", "1",
            "--data-parallel-size", str(DP_RANKS),
            "--data-parallel-size-local", str(DP_RANKS),
            "--data-parallel-backend", "mp",
            "--pipeline-parallel-size", "1",
            "--prefill-context-parallel-size", "1",
            "--decode-context-parallel-size", "1",
            "--enable-expert-parallel",
            "--kv-cache-dtype", "fp8",
            "--enable-chunked-prefill",
            "--scheduler-reserve-full-isl",
            "--max-model-len", str(self.max_model_len),
            "--max-num-seqs", str(self.max_num_seqs),
            "--max-num-batched-tokens", str(self.max_num_batched_tokens),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--compilation-config", json.dumps(compilation, separators=(",", ":")),
            "--tool-call-parser", "glm47",
            "--reasoning-parser", "glm47",
            "--enable-auto-tool-choice",
        ]
        if self.capacity_comparison != "stock-resident":
            connector_options: dict[str, int | str] = {"host_pool_gib": self.host_pool_gib_per_rank}
            if self.external_host_policy is not None:
                connector_options["host_pool_allocator"] = "kitepond.hisparse_host.acquire"
            connector = {
                "kv_connector": "HiSparseConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": connector_options,
            }
            argv.extend(("--kv-transfer-config", json.dumps(connector, separators=(",", ":"))))
        if self.capacity_comparison is not None:
            argv.extend((
                "--attention-backend", "FLASHINFER_MLA_SPARSE_SM120",
                "--all2all-backend", "allgather_reducescatter",
                "--disable-flashinfer-autotune",
                "--language-model-only",
                "--mamba-cache-mode", "none",
                "--cudagraph-metrics",
            ))
        if self.capacity_comparison == "indexer2048":
            attention = {"hisparse_config": {"indexer_block_size": 2048}}
            argv.extend(("--attention-config", json.dumps(attention, separators=(",", ":"))))
        argv.append(
            "--enable-prefix-caching"
            if self.prefix_caching
            else "--no-enable-prefix-caching"
        )
        if self.mtp:
            speculative = {"method": "mtp", "num_speculative_tokens": self.mtp}
            argv.extend(
                ("--speculative-config", json.dumps(speculative, separators=(",", ":")))
            )
        if self.stock_topk_override:
            override = {"index_topk": self.checkpoint.index_topk}
            argv.extend(("--hf-overrides", json.dumps(override, separators=(",", ":"))))
        return argv

    def as_dict(self, cli_checked: bool) -> dict:
        plan = {
            "scope": "native_configuration_only",
            "reference_image": REFERENCE_IMAGE,
            "vllm_version": VLLM_VERSION,
            "vllm_cli_checked": cli_checked,
            "topology": {
                "tp": 1, "dp": DP_RANKS, "ep": DP_RANKS, "pp": 1, "pcp": 1, "dcp": 1,
            },
            "checkpoint_shards": self.checkpoint.shard_count,
            "host_pool": {
                "per_dp_rank_gib": self.host_pool_gib_per_rank,
                "requested_total_gib": self.total_host_pool_gib,
                "budget_total_gib": self.host_pool_budget_gib,
            },
            "environment": self.environment(),
            "argv": self.argv(),
        }
        if self.capacity_comparison is not None:
            plan["capacity_comparison"] = self.capacity_comparison
        return plan

    def shell_command(self) -> str:
        assignments = [f"{key}={value}" for key, value in self.environment().items()]
        return shlex.join(["env", *assignments, *self.argv()])


def check_vllm_cli(profile: LaunchProfile) -> argparse.Namespace:
    import vllm

    from .vllm_cli import validate_serve_argv

    if profile.external_host_policy is not None:
        from vllm.v1.hisparse import runtime

        if not hasattr(runtime, "HiSparseHostMemoryLease"):
            raise ValueError("External host policy requires the vLLM fork")

    version = vllm.__version__
    if version.split("+", maxsplit=1)[0] != VLLM_VERSION:
        raise ValueError(f"Expected vLLM {VLLM_VERSION}, found {version}")
    return validate_serve_argv(profile.argv())


def gpu_indices(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(index) for index in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated GPU indices") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", type=Path, required=True, help="Local checkpoint directory")
    parser.add_argument("--host-pool-gib-per-rank", type=int)
    parser.add_argument("--host-pool-budget-gib", type=int)
    parser.add_argument(
        "--capacity-comparison", choices=("stock-resident", "indexer2048"),
        help="Use the paired 128Ki context, MTP3 and 92 percent SM120 capacity defaults",
    )
    parser.add_argument("--served-model-name", default="glm-5.3-flash")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18120)
    parser.add_argument("--devices", type=gpu_indices, default=tuple(range(DP_RANKS)))
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int, default=16, help="Per DP replica")
    parser.add_argument(
        "--max-num-batched-tokens", type=int, help="Per DP replica"
    )
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--mtp", type=int, choices=(0, 3))
    parser.add_argument("--prefix-caching", action="store_true")
    parser.add_argument(
        "--stock-topk-override", action="store_true",
        help="Mirror the checkpoint's nested index_topk at the root for stock vLLM v0.34",
    )
    parser.add_argument(
        "--external-host-policy", type=Path,
        help="Use an explicit Kitepond RAM budget policy with the vLLM fork",
    )
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    parser.add_argument(
        "--check-vllm-cli", action="store_true",
        help="Check argv against installed vLLM v0.34 without constructing an engine",
    )
    options = vars(parser.parse_args(argv))
    output_format = options.pop("format")
    cli_checked = options.pop("check_vllm_cli")
    model_dir = options.pop("model")
    comparison = options["capacity_comparison"]
    if comparison is None:
        defaults = {
            "max_model_len": 32768, "max_num_batched_tokens": 1024,
            "gpu_memory_utilization": 0.90, "mtp": 0,
        }
        for key in ("host_pool_gib_per_rank", "host_pool_budget_gib"):
            if options[key] is None:
                parser.error(f"--{key.replace('_', '-')} is required without a capacity comparison")
    else:
        pool_per_rank, pool_total = (0, 0) if comparison == "stock-resident" else (8, 64)
        defaults = {
            "max_model_len": 131072, "max_num_batched_tokens": 2048,
            "gpu_memory_utilization": 0.92, "mtp": 3,
            "host_pool_gib_per_rank": pool_per_rank, "host_pool_budget_gib": pool_total,
        }
    for key, value in defaults.items():
        if options[key] is None:
            options[key] = value
    try:
        profile = LaunchProfile(Checkpoint.inspect(model_dir), **options)
        if cli_checked:
            check_vllm_cli(profile)
    except (ValueError, ImportError, OSError) as exc:
        parser.error(str(exc))
    if output_format == "shell":
        print(profile.shell_command())
    else:
        print(json.dumps(profile.as_dict(cli_checked), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
