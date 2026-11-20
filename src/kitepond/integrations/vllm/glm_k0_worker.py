# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit GPU worker for GLM K0 P using the stock MTP implementation."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from kitepond.core.readiness import _digest
from kitepond.models.glm5next.fingerprint import config_fingerprint

from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm.v1.worker.gpu_worker import Worker

from .glm_k0_routes import K0_CONFIG
from .glm_k0_runtime import GlmK0PrefillRuntime
from .glm_prefill_bootstrap import MooncakeConnector, bind_native_glm_prefill_source
from .prefill_return_receive import prefill_return_boot_id
from .ready_transport import ReadyPublisher


class GlmK0MTP(MTPSpeculator):
    """Bind admitted inputs and observe the native draft before finalizing saves."""

    def __init__(self, config, device, runtime: GlmK0PrefillRuntime):
        super().__init__(config, device)
        self._kitepond_k0 = runtime

    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        *,
        dp_sync_state=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
        num_speculative_tokens=None,
    ):
        active = not (dummy_run or is_profile) and has_kv_transfer_group()
        if active:
            if num_speculative_tokens != 0 or mm_inputs is not None:
                raise ValueError(
                    "Named GLM P requires causal text K0 from its scheduler"
                )
            runtime = self._kitepond_k0
            chunks, rows, last_sampled, next_prefill_tokens = runtime.prepare(
                input_batch, last_sampled, next_prefill_tokens
            )
        result = super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync_state=dp_sync_state,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
            num_speculative_tokens=num_speculative_tokens,
        )
        if active:
            runtime.finish(
                chunks,
                rows,
                input_batch,
                attn_metadata,
                self.input_buffers,
                self.last_token_indices,
                num_sampled,
                num_rejected,
            )
        return result


class GlmK0PrefillRunner(GPUModelRunner):
    def __init__(self, config, device):
        p = config.parallel_config
        speculative = config.speculative_config
        options = config.additional_config
        settings = options.get(K0_CONFIG) if isinstance(options, dict) else None
        host = options.get("kitepond_glm_host") if isinstance(options, dict) else None
        if (
            not isinstance(settings, dict)
            or set(settings) != {"ready_directory"}
            or type(settings["ready_directory"]) is not str
            or not Path(settings["ready_directory"]).is_absolute()
            or not isinstance(host, dict)
            or host.get("role") != "producer"
            or not _digest(host.get("model_sha256"))
            or config.scheduler_config.async_scheduling is not False
            or config.max_concurrent_batches != 1
            or (p.tensor_parallel_size, p.pipeline_parallel_size, p.data_parallel_size)
            != (4, 1, 1)
            or p.prefill_context_parallel_size != 1
            or (p.decode_context_parallel_size, p.cp_kv_cache_interleave_size) != (4, 4)
            or speculative is None
            or speculative.method != "mtp"
            or speculative.num_speculative_tokens != 3
        ):
            raise ValueError(
                "GLM K0 P requires its exact synchronous native TP4/DCP4 profile"
            )
        self.kitepond_k0 = GlmK0PrefillRuntime(
            host["model_sha256"],
            config_fingerprint(config.model_config.hf_text_config.to_dict()),
        )
        self.kitepond_ready_directory = Path(settings["ready_directory"])
        super().__init__(config, device)

    def _make_speculator(self):
        return GlmK0MTP(self.vllm_config, self.device, self.kitepond_k0)


class GlmK0PrefillWorker(Worker):
    """Select through vLLM's native Worker class setting on P only."""

    def _make_model_runner(self):
        if not self.use_v2_model_runner:
            raise ValueError("GLM K0 P requires the native v2 worker")
        return GlmK0PrefillRunner(self.vllm_config, self.device)

    def initialize_from_config(self, kv_cache_config):
        super().initialize_from_config(kv_cache_config)
        runner = self.model_runner
        runtime = runner.kitepond_k0
        publisher = ReadyPublisher(runner.kitepond_ready_directory)
        try:
            views = bind_native_glm_prefill_source(
                runner,
                publisher,
                runtime.current_grant,
                runtime,
                collect_writes=runtime.collect_writes,
                export=runtime.export,
            )
            views.bind_tail_writers()
            runtime.bind_region(views.descriptor)
            _, mooncake = get_kv_transfer_group().sub_connectors
            if type(mooncake) is not MooncakeConnector:
                raise ValueError("GLM K0 P has another native Mooncake child")
            mooncake.bind_k0_runtime(runtime)
            self._kitepond_ready_views = views
            self._kitepond_warm_imports = None
            if prefill_return_boot_id(self.vllm_config.additional_config) is not None:
                from .glm_warm_prefill_worker import GlmPrefillWarmImports

                imports = GlmPrefillWarmImports(
                    views,
                    torch.cuda.Stream(device=self.device),
                    capacity=self.vllm_config.scheduler_config.max_num_seqs,
                )
                runtime.bind_warm_import_verifier(imports.require_loaded)
                self._kitepond_warm_imports = imports
        except BaseException:
            publisher.close()
            raise

    def kitepond_prefill_ready_identity(self) -> dict[str, int | str]:
        """Read the real initialized worker identity for the local READY collector."""
        views = getattr(self, "_kitepond_ready_views", None)
        if views is None:
            raise ValueError(
                "GLM P READY identity is unavailable before native startup"
            )
        return {
            "rank": views.rank,
            "pid": os.getpid(),
            "region_sha256": views.descriptor.fingerprint(),
        }

    def _require_warm_imports(self):
        imports = getattr(self, "_kitepond_warm_imports", None)
        if imports is None:
            raise ValueError("Native warm P worker is not enabled")
        return imports

    def kitepond_prefill_submit_terminal_return(self, transfer, timeout) -> None:
        self._require_warm_imports().submit(transfer, timeout)

    def kitepond_prefill_terminal_return_result(self, transfer):
        return self._require_warm_imports().result(transfer)

    def kitepond_prefill_retire_terminal_return(self, transfer) -> None:
        self._require_warm_imports().retire(transfer)

    def shutdown(self) -> None:
        imports = getattr(self, "_kitepond_warm_imports", None)
        if imports is not None:
            imports.close()
            self._kitepond_warm_imports = None
        super().shutdown()
