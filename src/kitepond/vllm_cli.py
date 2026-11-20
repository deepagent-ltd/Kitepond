"""Use vLLM's native parser and activate Kitepond on explicit opt-in."""

import argparse
import json
import math
import mmap
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from kitepond.checkpoints import CheckpointKind, inspect_checkpoint


def _require(args: argparse.Namespace, **settings) -> None:
    for name, required in settings.items():
        value = getattr(args, name)
        if value is None or (name == "kv_cache_dtype" and value == "auto"):
            setattr(args, name, required)
        elif value != required:
            option = "--" + name.replace("_", "-")
            raise ValueError(f"{option}={value!r} conflicts with Kitepond; expected {required!r}")


def _check_gpus(args: argparse.Namespace) -> None:
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability

    if not current_platform.is_cuda():
        raise ValueError("Kitepond needs eight CUDA GPUs for this topology")
    sm120 = DeviceCapability(12, 0)
    if args.device_ids:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.utils.import_utils import import_pynvml

        selected = EngineArgs.from_cli_args(args)._resolve_device_ids()
        if selected is None or len(selected) != 8:
            raise ValueError("--device-ids must select eight GPUs for this topology")
        nvml = import_pynvml()
        nvml.nvmlInit()
        try:
            capabilities = [
                DeviceCapability(*nvml.nvmlDeviceGetCudaComputeCapability(
                    nvml.nvmlDeviceGetHandleByIndex(device)
                ))
                for device in selected
            ]
        except nvml.NVMLError as error:
            raise ValueError(f"Cannot check the selected --device-ids: {error}") from error
        finally:
            nvml.nvmlShutdown()
    else:
        if current_platform.device_count() < 8:
            raise ValueError("Kitepond needs eight visible CUDA GPUs for this topology")
        capabilities = [current_platform.get_device_capability(rank) for rank in range(8)]
    if any(capability != sm120 for capability in capabilities):
        raise ValueError("Kitepond needs eight SM120 GPUs for this topology")


def _configure_host_pool(args: argparse.Namespace, dp: bool, dcp: int, runtime: Path) -> None:
    from vllm.config.kv_transfer import KVTransferConfig

    from kitepond.hisparse_host import POLICY_ENV, read_policy
    from kitepond.hisparse_host_dcp8 import POLICY_ENV as DCP_POLICY_ENV

    transfer = args.kv_transfer_config
    if transfer is None:
        transfer = KVTransferConfig(kv_connector="HiSparseConnector", kv_role="kv_both")
    if (
        transfer.kv_connector != "HiSparseConnector"
        or transfer.kv_role != "kv_both"
        or transfer.kv_connector_module_path is not None
    ):
        raise ValueError("--kv-transfer-config must use the built-in HiSparseConnector with kv_both")

    options = dict(transfer.kv_connector_extra_config)
    pool = options.setdefault("host_pool_gib", 48 if dp else 384)
    if type(pool) not in (int, float) or not math.isfinite(pool) or pool <= 0:
        raise ValueError("HiSparse host_pool_gib must be a positive finite number")

    if dp:
        allocator = "kitepond.hisparse_host.acquire"
        if options.setdefault("host_pool_allocator", allocator) != allocator:
            raise ValueError("DP8 requires the Kitepond HiSparse host allocator")
    elif options.get("host_pool_allocator") is not None:
        raise ValueError("host_pool_allocator is supported only for DP8")

    if dcp == 8:
        provider = "kitepond.hisparse_host_dcp8.acquire"
        extra = dict(args.additional_config)
        if extra.setdefault("hisparse_host_buffer_provider", provider) != provider:
            raise ValueError("DCP8 requires the Kitepond HiSparse host buffer provider")
        args.additional_config = extra

    if dp or dcp == 8:
        env = POLICY_ENV if dp else DCP_POLICY_ENV
        per_rank = pool if dp else pool / 8
        rank_bytes = math.ceil(per_rank * (1 << 30) / mmap.PAGESIZE) * mmap.PAGESIZE
        if os.environ.get(env):
            policy = Path(os.environ[env])
            if not policy.is_absolute():
                raise ValueError(f"{env} must name an absolute policy file")
            _, budget, _ = read_policy(policy)
            if budget < rank_bytes:
                raise ValueError(f"HiSparse host_pool_gib exceeds {env}'s per-rank budget")
        else:
            policy = runtime / "host-pool-policy.json"
            policy.write_text(
                json.dumps({"rank_bytes": rank_bytes, "total_bytes": 8 * rank_bytes}),
                encoding="utf-8",
            )
            os.environ[env] = str(policy)

    args.kv_transfer_config = replace(transfer, kv_connector_extra_config=options)


def _prepare_serve(args: argparse.Namespace) -> None:
    from vllm.config.attention import AttentionConfig, HiSparseConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    mode = (args.tensor_parallel_size, args.data_parallel_size, args.decode_context_parallel_size)
    if mode not in ((1, 8, 1), (8, 1, 1), (8, 1, 8)):
        raise ValueError("supported topologies are DP8/DCP1, TP8/DCP1 and TP8/DCP8")
    if (
        args.pipeline_parallel_size != 1
        or args.prefill_context_parallel_size != 1
        or args.nnodes != 1
        or args.data_parallel_backend != "mp"
        or args.distributed_executor_backend == "ray"
        or args.data_parallel_rank is not None
        or args.data_parallel_start_rank is not None
        or args.data_parallel_external_lb
        or args.data_parallel_hybrid_lb
        or args.data_parallel_multi_port_external_lb
    ):
        raise ValueError("Kitepond requires a single-node topology, PP1 and PCP1")

    _, dp, dcp = mode
    _require(args, enable_expert_parallel=True, data_parallel_size_local=dp)
    _require(args, kv_cache_dtype="fp8_ds_mla" if dcp == 8 else "fp8")
    if dcp == 8:
        if args.speculative_config or args.spec_method or args.spec_model:
            raise ValueError("Kitepond's DCP8 profile requires speculative decoding to be disabled")
        _require(
            args,
            block_size=256,
            cp_kv_cache_interleave_size=4,
            dcp_comm_backend="ag_rs",
            distributed_executor_backend="mp",
            enable_prefix_caching=False,
            enable_extensible_kv_cache=False,
            mamba_cache_mode="none",
        )

    # Match ServeSubcommand.cmd before EngineArgs is built for device selection.
    args.model = args.model_tag or args.model
    model = inspect_checkpoint(Path(args.model))
    if model.kind != CheckpointKind.FLASH_NVFP4:
        raise ValueError("this wheel supports GLM-5.3-Flash NVFP4")
    attention = args.attention_config
    backend = attention.backend_per_kind.get("mla_attention")
    if backend is None:
        backend = args.attention_backend or attention.backend
    required_backend = AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120
    if backend is None:
        args.attention_backend = required_backend
    elif AttentionConfig.validate_backend_before(backend) != required_backend:
        raise ValueError("Kitepond requires FLASHINFER_MLA_SPARSE_SM120 for MLA")

    sparse = attention.hisparse_config or HiSparseConfig()
    changes = {}
    if sparse.indexer_block_size is None:
        changes["indexer_block_size"] = 8192 if dp == 8 else 2048
    if dcp == 8:
        changes["enable_dcp8"] = True
        if sparse.pack_hot_groups is None:
            changes["pack_hot_groups"] = True
    args.attention_config = replace(attention, hisparse_config=replace(sparse, **changes))
    _configure_host_pool(args, dp == 8, dcp, Path(os.environ["KITEPOND_RUNTIME_DIR"]))
    _check_gpus(args)


def run_enabled_serve() -> None:
    from vllm.entrypoints.cli import serve
    from vllm.entrypoints.cli.main import main

    option_end = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    native = sys.argv[:option_end]
    if "--omni" in native:
        raise SystemExit("Kitepond preflight: --enable cannot be combined with --omni")
    # Keep the model and options in their original order for vLLM's parser.
    sys.argv[:] = [arg for arg in native if arg != "--enable"] + sys.argv[option_end:]

    class KitepondServe(serve.ServeSubcommand):
        def subparser_init(self, subparsers):
            parser = super().subparser_init(subparsers)
            parser.add_argument("--enable", action="store_true", help="Enable Kitepond KV management")
            parser.set_defaults(enable_expert_parallel=None)
            self.parser = parser
            return parser

        def validate(self, args):
            try:
                _prepare_serve(args)
            except (OSError, ValueError) as error:
                self.parser.error(f"Kitepond preflight: {error}")
            return super().validate(args)

    # Only the isolated, opt-in vLLM subprocess uses this serve registration.
    serve.cmd_init = lambda: [KitepondServe()]
    main()


def validate_serve_argv(argv: Sequence[str]) -> argparse.Namespace:
    from vllm.entrypoints.common.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = make_arg_parser(FlexibleArgumentParser(prog="vllm serve"))
    args = parser.parse_args(argv[2:])
    validate_parsed_serve_args(args)
    return args
