# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect the installed vLLM source without importing the engine."""

import ast
from importlib.metadata import distribution


def _source_nodes(root, relative):
    path = root / relative
    if not path.is_file():
        return []
    return ast.parse(path.read_text(encoding="utf-8")).body


def _methods(root, relative, class_name):
    for node in _source_nodes(root, relative):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                member.name: member
                for member in node.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return {}


def _arguments(method):
    if method is None:
        return ()
    return tuple(
        arg.arg
        for arg in method.args.posonlyargs + method.args.args + method.args.kwonlyargs
    )


def _references(method, name):
    return method is not None and any(
        (isinstance(node, ast.Constant) and node.value == name)
        or (isinstance(node, ast.Attribute) and node.attr == name)
        or (isinstance(node, ast.Name) and node.id == name)
        for node in ast.walk(method)
    )


def inspect_engine() -> dict:
    """Report source-level capabilities for the current wheel."""
    package = distribution("vllm")
    root = package.locate_file("vllm")
    connector_root = "distributed/kv_transfer/kv_connector"
    factory = _methods(root, f"{connector_root}/factory.py", "KVConnectorFactory")
    connector = _methods(root, f"{connector_root}/v1/base.py", "KVConnectorBase_V1")
    scheduler = _methods(root, "v1/core/sched/scheduler.py", "Scheduler")
    wrapper = _methods(root, "v1/worker/worker_base.py", "WorkerWrapperBase")
    manager = _methods(root, "v1/core/kv_cache_manager.py", "KVCacheManager")

    host = _methods(root, "v1/hisparse/runtime.py", "HiSparseHostPool")
    host_manager = _methods(
        root, "v1/core/single_type_kv_cache_manager.py", "HiSparseSourceManager"
    )
    spec_registry = _methods(
        root, "v1/kv_cache_spec_registry.py", "KVCacheSpecRegistry"
    )
    engine = _methods(root, "v1/engine/core.py", "EngineCore")
    worker_connector = _methods(
        root, "v1/worker/gpu/kv_connector.py", "ActiveKVConnector"
    )
    coordinator = next(
        (
            node
            for node in _source_nodes(root, "v1/hisparse/coordinator.py")
            if isinstance(node, ast.FunctionDef)
            and node.name == "get_hisparse_coordinator"
        ),
        None,
    )
    worker = _methods(root, "v1/worker/gpu_worker.py", "Worker")
    runner = _methods(root, "v1/worker/gpu/model_runner.py", "GPUModelRunner")
    config = _methods(root, "config/vllm.py", "VllmConfig")
    platform = _methods(root, "platforms/interface.py", "Platform")
    engine_process = _methods(root, "v1/engine/core.py", "EngineCoreProc")
    scheduler_interface = _methods(
        root, "v1/core/sched/interface.py", "SchedulerInterface"
    )
    residency = _methods(root, "v1/hisparse/coordinator.py", "HiSparseCoordinator")
    mooncake = _methods(
        root,
        f"{connector_root}/v1/mooncake/mooncake_connector.py",
        "MooncakeConnectorWorker",
    )
    warmup = {
        node.name: node
        for node in _source_nodes(root, "v1/worker/gpu/warmup.py")
        if isinstance(node, ast.FunctionDef)
    }
    late_output = worker_connector.get("finalize_step")
    extension_init = _references(wrapper.get("init_device"), "init_worker_extension")
    extension_destroy = _references(wrapper.get("shutdown"), "destroy_worker_extension")
    return {
        "distribution_version": package.version,
        "scope": "source_signatures",
        "connector_module_path": _references(
            factory.get("get_connector_class"), "kv_connector_module_path"
        ),
        "connector_init_args": _arguments(connector.get("__init__")),
        "scheduler_cache_manager_binding": _references(
            scheduler.get("__init__"), "bind_kv_cache_manager"
        ),
        "worker_extension_init_callback": extension_init,
        "worker_extension_destroy_callback": extension_destroy,
        "post_fence_metadata_hook": (
            "output" in _arguments(late_output)
            and _references(late_output, "build_connector_worker_meta")
        ),
        "cache_spec_registration": "register" in spec_registry,
        "exact_snapshot_reservation": "plan_native_mamba_snapshot" in manager,
        "shared_host_allocator_binding": "bind_borrowed_authority" in host_manager,
        "shared_host_authority_binding": "bind_authority_region" in host,
        "engine_connector_control": "kv_connector_control" in engine,
        "exact_boundary_offers": "take_boundary_state_offloads" in manager,
        "scheduler_host_pool_provider": (
            "host_pool_provider" in _arguments(coordinator)
        ),
        "worker_host_buffer_provider": _references(
            host.get("allocate"), "hisparse_host_buffer_provider"
        ),
        "worker_model_runner_factory": "_make_model_runner" in worker,
        "worker_speculator_factory": (
            "_make_speculator" in runner
            and _references(runner.get("__init__"), "_make_speculator")
        ),
        "hisparse_dcp_guard_in_config": any(
            _references(method, "HiSparse does not support decode context parallelism.")
            for method in config.values()
        ),
        "hisparse_dcp_provider_gate": any(
            _references(method, "validate_hisparse_dcp_launch")
            for method in config.values()
        ),
        "hisparse_dcp_independent_hybrid_pages": _references(
            platform.get("_align_hybrid_block_size"), "separate_hybrid_state_pages"
        ),
        "native_synchronous_step_callback": (
            _references(engine.get("__init__"), "_configure_synchronous_step_callback")
            and _references(engine.get("step"), "_synchronous_step_callback")
        ),
        "native_worker_startup_callback": (
            "get_worker_startup_callback" in scheduler_interface
            and _references(engine.get("__init__"), "_run_worker_startup_callback")
        ),
        "native_engine_idle_callback": (
            "get_quiescent_callback" in scheduler_interface
            and any(
                _references(method, "get_quiescent_callback")
                for method in engine_process.values()
            )
        ),
        "native_terminal_host_fence": {
            "plan_terminal_host_prefix",
            "has_durable_host_prefix",
        }
        <= residency.keys(),
        "mooncake_named_control": {
            "bind_named_control_handler",
            "resolve_named_control_peer",
        }
        <= mooncake.keys(),
        "scheduler_owned_hisparse_warmup": all(
            _references(warmup.get(step), "_requires_scheduler_cache_ownership")
            for step in ("run_mixed_prefill_decode_warmup", "warmup_kernels")
        ),
        "full_cache_mode": "unavailable",
    }
