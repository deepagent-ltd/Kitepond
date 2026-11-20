"""Generate a GLM-5.3-Flash launch plan on an upstream HiSparse backend."""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Sequence
from pathlib import Path

from .checkpoints import CheckpointKind, inspect_checkpoint

UPSTREAM_REFERENCE = "https://github.com/vllm-project/vllm/pull/61243"

# These are the devices and KV formats in the upstream Flash TP4 tests.
# Leave automatic backend selection in place where the reference did so.
PROFILES = {
    "h100-bf16": ("bfloat16", "FLASHMLA_SPARSE"),
    "h200-bf16": ("bfloat16", None),
    "b200-bf16": ("bfloat16", None),
    "b200-fp8": ("fp8", None),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--host-pool-gib", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--served-model-name", default="glm-5.3-flash")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--check-vllm-cli", action="store_true")
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    args = parser.parse_args(argv)
    try:
        checkpoint = inspect_checkpoint(args.model)
        if checkpoint.kind != CheckpointKind.FLASH_FP8:
            raise ValueError("The upstream device recipes require GLM-5.3-Flash FP8 weights")
        if args.host_pool_gib <= 0:
            raise ValueError("The host pool must be positive GiB")
        if not 0 < args.max_model_len <= checkpoint.context_limit:
            raise ValueError(f"Model length must be in 1..{checkpoint.context_limit}")
        if not 1 <= args.port <= 65535 or not args.served_model_name.strip():
            raise ValueError("Specify a non-empty served model name and a port in 1..65535")

        dtype, backend = PROFILES[args.profile]
        connector = {
            "kv_connector": "HiSparseConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"host_pool_gib": args.host_pool_gib},
        }
        command = [
            "vllm", "serve", str(checkpoint.path),
            "--served-model-name", args.served_model_name,
            "--host", "127.0.0.1", "--port", str(args.port),
            "--language-model-only", "--tensor-parallel-size", "4",
            "--decode-context-parallel-size", "1",
            "--kv-cache-dtype", dtype,
            "--max-model-len", str(args.max_model_len),
            "--max-num-batched-tokens", "4096", "--max-num-seqs", "32",
            "--no-enable-prefix-caching",
            "--attention-config", '{"hisparse_config":{}}',
            "--kv-transfer-config", json.dumps(connector, separators=(",", ":")),
        ]
        if backend is not None:
            command.extend(("--attention-backend", backend))
        if args.check_vllm_cli:
            from .vllm_cli import validate_serve_argv

            validate_serve_argv(command)
    except (ImportError, ValueError) as exc:
        parser.error(str(exc))

    if args.format == "shell":
        print(shlex.join(command))
    else:
        print(json.dumps({
            "profile": args.profile,
            "qualification": "upstream_reference_only",
            "reference": UPSTREAM_REFERENCE,
            "checkpoint": checkpoint.kind.value,
            "host_allocator": "vllm_native",
            "host_pool_gib_for_tp_group": args.host_pool_gib,
            "vllm_cli_checked": args.check_vllm_cli,
            "argv": command,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
