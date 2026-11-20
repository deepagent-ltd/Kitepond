# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind exact GLM terminal sampling to an explicitly selected v2 worker."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch
from kitepond.models.glm5next.frontier import (
    resolve_native_gdn_row,
    validate_native_terminal_frontier,
)
from kitepond.models.glm5next.frontier_gpu import NativeAcceptedFrontier

from vllm.distributed import get_dcp_group
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm.v1.worker.gpu_worker import Worker

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from .glm_native_views import GlmNativeStateViews
    from .glm_terminal_bootstrap import TerminalRankEnrollment, TerminalStartupIdentity
    from .glm_terminal_manifest import (
        FencedTerminalSource,
        GlmTerminalManifest,
        SignedTerminalRank,
        TerminalFrameReceipt,
        TerminalRankSigner,
    )
    from .terminal_checkpoint import TerminalPinLease, TerminalReadPermit


@dataclass(frozen=True)
class _SampleFrame:
    slot: int
    serial: int
    event: torch.cuda.Event
    batch_index: int
    gdn_selectors: dict


@dataclass(frozen=True)
class NativeTerminalRows:
    target: dict
    draft: dict
    kda_selectors: dict[str, dict[str, list]]


class GlmTerminalMTP(MTPSpeculator):
    """Record only draft step zero, while its native query is in scope."""

    def __init__(self, config, device, observe_query):
        super().__init__(config, device)
        self._observe_query = observe_query

    def on_prefill_begin(self, num_reqs: int) -> None:
        super().on_prefill_begin(num_reqs)
        self._observe_query(self)


class GlmTerminalRunner(GPUModelRunner):
    def __init__(self, config, device):
        if not callable(getattr(GPUModelRunner, "_make_speculator", None)):
            raise RuntimeError("vLLM has no supported early speculator factory")
        parallel = config.parallel_config
        spec = config.speculative_config
        if (
            config.scheduler_config.async_scheduling is not False
            or config.max_concurrent_batches != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.tensor_parallel_size != 4
            or parallel.decode_context_parallel_size != 4
            or spec is None
            or spec.method != "mtp"
            or spec.num_speculative_tokens != 3
        ):
            raise ValueError(
                "Exact GLM terminal capture requires synchronous TP4 DCP4 MTP3"
            )
        self._frames: dict[str, _SampleFrame] = {}
        self._active_batch = False
        self._observed_draft = False
        super().__init__(config, device)
        self.frontier = NativeAcceptedFrontier(
            self.device, self.max_num_reqs, self.main_stream
        )

    def _make_speculator(self):
        return GlmTerminalMTP(self.vllm_config, self.device, self._observe_draft)

    def _observe_draft(self, speculator) -> None:
        if not self._active_batch:
            return
        if self._observed_draft:
            raise ValueError(
                "GLM native sample observed two draft queries for one step"
            )
        self.frontier.draft(speculator)
        self._observed_draft = True

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        self.frontier.warmup(self)
        return result

    def start_imported_frontier(self, request_id, imported_tokens, native_slot):
        """Call only after a typed complete native incremental or cold import."""
        self.frontier.start(request_id, imported_tokens, native_slot)

    def drop_imported_frontier(self, request_id):
        """Called before native request cleanup may recycle the GPU slot."""
        self.frontier.clear(request_id)
        self._frames.pop(request_id, None)

    def sample_tokens(self, grammar_output):
        step = self.execute_model_state
        if step is None or not self.frontier.requests:
            return super().sample_tokens(grammar_output)
        batch = step.input_batch
        tracked = {
            request_id: (i, int(batch.idx_mapping_np[i]))
            for i, request_id in enumerate(batch.req_ids)
            if request_id in self.frontier.requests
        }
        if not tracked:
            return super().sample_tokens(grammar_output)
        if any(
            self.frontier.requests[key] != slot for key, (_, slot) in tracked.items()
        ):
            raise ValueError("Tracked GLM request left its native GPU slot")
        # Keep only KDA state selectors. Every tracked request uses the same
        # native attention metadata for this one synchronous sample.
        gdn = {
            name: meta
            for name, meta in step.attn_metadata.items()
            if hasattr(meta, "spec_state_indices_tensor")
        }
        # Invalidate the preceding step's proof before the model can write.
        for request_id in tracked:
            self._frames.pop(request_id, None)
        with torch.cuda.stream(self.main_stream):
            self.frontier.target(batch)
            self._active_batch = True
            self._observed_draft = False
            try:
                result = super().sample_tokens(grammar_output)
                sampling = getattr(result, "sampler_output", None)
                if sampling is None or not self._observed_draft:
                    return result
                self.frontier.advance(self, sampling.num_rejected)
                fence = torch.cuda.Event()
                fence.record(self.main_stream)
                for request_id, (batch_index, slot) in tracked.items():
                    self._frames[request_id] = _SampleFrame(
                        slot, self.frontier.serial, fence, batch_index, gdn
                    )
                return result
            finally:
                self._active_batch = False

    def terminal_rows(self, request_id, intent, imported_tokens) -> NativeTerminalRows:
        """Consume the captured step before cleanup or reuse of its native slot."""
        observed = self._frames.pop(request_id, None)
        if observed is None:
            raise ValueError("No completed GLM sample frame")
        observed.event.synchronize()
        slot = observed.slot
        packed = self.frontier.frame[slot].cpu().tolist()
        query_len = packed[18]
        if (
            packed[24] != observed.serial
            or packed[25] != observed.serial
            or not 1 <= query_len <= 4
            or packed[19] != slot
        ):
            raise ValueError("GLM frame is stale or outside its captured native query")
        tokens = self.req_states.all_token_ids.gpu[slot]
        frame = {
            "target_positions": packed[:query_len],
            "target_input_ids": packed[4 : 4 + query_len],
            "native_req_index": slot,
            "draft_prepared": {
                "positions": packed[8 : 8 + query_len],
                "input_ids": packed[12 : 12 + query_len],
                "last_token_indices": [packed[16]],
                "query_start_loc": packed[20:22],
            },
            "after_finalize": {
                "raw_gpu_computed": self.req_states.num_computed_tokens.gpu[slot].item()
            },
        }
        if not observed.gdn_selectors:
            raise ValueError("GLM terminal proof lacks native KDA selectors")
        target, draft = validate_native_terminal_frontier(
            self.frontier.state[slot].cpu().tolist(),
            frame,
            intent,
            imported_tokens,
            tokens[: intent.requested_common_tokens + 1].cpu().tolist(),
        )
        selectors = {
            name: {
                key: value.cpu().tolist()
                for key, value in resolve_native_gdn_row(
                    metadata, observed.batch_index
                ).items()
            }
            for name, metadata in observed.gdn_selectors.items()
        }
        return NativeTerminalRows(target, draft, selectors)


@dataclass
class _SavedTerminalFrame:
    lease: "TerminalPinLease"
    observed: NativeTerminalRows
    receipt: "TerminalFrameReceipt"
    checkpoint: "SignedTerminalRank | None" = None
    source: "FencedTerminalSource | None" = None


class GlmTerminalWorker(Worker):
    """Use via the native Worker selector on an engine with the early factory."""

    def _make_model_runner(self):
        if not self.use_v2_model_runner:
            raise ValueError("Kitepond GLM terminal capture requires the v2 Worker")
        return GlmTerminalRunner(self.vllm_config, self.device)

    def initialize_from_config(self, kv_cache_config: "KVCacheConfig") -> None:
        super().initialize_from_config(kv_cache_config)
        options = self.vllm_config.additional_config
        enabled = (
            options.get("kitepond_glm_native_terminal", False)
            if isinstance(options, dict)
            else False
        )
        if type(enabled) is not bool:
            raise ValueError("GLM terminal native views require an explicit boolean")
        if enabled:
            from .glm_native_views import GlmNativeStateViews

            views = GlmNativeStateViews(
                self.model_runner, get_dcp_group().rank_in_group
            )
            views.bind_tail_writers()
            self._kitepond_native_state_views = views
        if isinstance(options, dict) and "kitepond_glm_cold_import" in options:
            from .glm_return_bootstrap import bind_native_glm_cold_import

            self._kitepond_decode_import = bind_native_glm_cold_import(
                self.model_runner, self.kitepond_native_state_views()
            )

    def kitepond_prepare_shared_mla(self, offer):
        runtime = getattr(self, "_kitepond_decode_import", None)
        if runtime is None:
            raise ValueError("Shared MLA preflight needs the initialized native worker")
        return runtime.prepare_shared_mla(offer)

    def kitepond_native_state_views(self) -> "GlmNativeStateViews":
        views = getattr(self, "_kitepond_native_state_views", None)
        if views is None:
            raise RuntimeError("Worker has no enrolled native GLM terminal views")
        return views

    def kitepond_enroll_terminal_worker(
        self, identity: "TerminalStartupIdentity", challenge: str
    ) -> "TerminalRankEnrollment":
        from .glm_terminal_bootstrap import enroll_native_worker

        return enroll_native_worker(self, identity, challenge)

    def bind_kitepond_native_terminal_source(
        self, signer: "TerminalRankSigner", representation_sha256: str
    ) -> None:
        views = self.kitepond_native_state_views()
        self.bind_kitepond_terminal_source(
            signer,
            partial(views.terminal_source, representation_sha256=representation_sha256),
        )

    def bind_kitepond_terminal_source(
        self,
        signer: "TerminalRankSigner",
        wait_native_source: Callable[
            ["TerminalPinLease", NativeTerminalRows], "FencedTerminalSource"
        ],
    ) -> None:
        """Bind the local native source and rank credential once at worker startup."""
        if (
            hasattr(self, "_kitepond_terminal_source")
            or not callable(wait_native_source)
            or signer.rank != get_dcp_group().rank_in_group
        ):
            raise ValueError(
                "GLM terminal source is already bound or on a foreign rank"
            )
        self._kitepond_terminal_source = (signer, wait_native_source)
        self._kitepond_terminal_frames: dict[str, _SavedTerminalFrame] = {}

    def _require_terminal_source(self):
        source = getattr(self, "_kitepond_terminal_source", None)
        if source is None or not isinstance(self.model_runner, GlmTerminalRunner):
            raise RuntimeError("GLM terminal worker has no bound native source")
        return source

    def kitepond_control_cold_rank(self, command, signatures):
        """Run one authenticated lease command on the owning worker."""
        from .glm_cold_rank_control import GlmWorkerColdLeases
        from .glm_cold_restore_control import ColdRestoreControl, GlmWorkerColdRestore

        memory = getattr(self, "_kitepond_native_cold_memory", None)
        if memory is None:
            raise RuntimeError("Worker has no authenticated native GLM cold extent")
        if type(command) is ColdRestoreControl:
            from .glm_cold_scheduler import native_cold_restore_enabled

            runtime = getattr(self, "_kitepond_decode_import", None)
            if runtime is None or not native_cold_restore_enabled(self.vllm_config):
                raise RuntimeError("Worker detached cold read is disabled")
            restore = getattr(self, "_kitepond_cold_restore", None)
            if restore is None:
                signer, _ = self._require_terminal_source()
                restore = GlmWorkerColdRestore(
                    memory, self.kitepond_native_state_views(), signer
                )
                runtime.bind_detached_restore(restore)
                self._kitepond_cold_restore = restore
            return restore.handle(command, signatures)
        state = getattr(self, "_kitepond_native_cold_leases", None)
        if state is None:
            state = GlmWorkerColdLeases(self, memory)
            self._kitepond_native_cold_leases = state
        return state.handle(command, signatures)

    def kitepond_take_terminal_frame(
        self, lease: "TerminalPinLease", imported_tokens: int
    ) -> "TerminalFrameReceipt":
        """Save the accepted GPU frame during the synchronous engine callback."""
        from .glm_terminal_manifest import take_terminal_frame

        signer, _ = self._require_terminal_source()
        frames = self._kitepond_terminal_frames
        now = time.monotonic_ns()
        if lease.identity.deadline_monotonic_ns <= now:
            raise ValueError("GLM terminal worker frame is expired")
        for key, entry in tuple(frames.items()):
            if entry.lease.identity.deadline_monotonic_ns <= now:
                frames.pop(key)
        key = lease.identity.request_id
        previous = frames.get(key)
        if previous is not None:
            if (
                previous.lease != lease
                or previous.receipt.proof.imported_tokens != imported_tokens
            ):
                raise ValueError("GLM terminal worker frame belongs to another attempt")
            return previous.receipt
        if len(frames) >= self.vllm_config.scheduler_config.max_num_seqs:
            raise BufferError("GLM terminal worker frames are at capacity")
        observed, receipt = take_terminal_frame(
            self.model_runner, lease, imported_tokens, signer
        )
        frames[key] = _SavedTerminalFrame(lease, observed, receipt)
        return receipt

    def kitepond_terminal_checkpoint(
        self, lease: "TerminalPinLease", imported_tokens: int
    ) -> "SignedTerminalRank":
        """Finalize later from the saved frame after native source durability."""
        from .glm_terminal_manifest import finalize_terminal_rank

        signer, wait_native_source = self._require_terminal_source()
        entry = self._kitepond_terminal_frames.get(lease.identity.request_id)
        if (
            entry is None
            or entry.lease != lease
            or entry.receipt.proof.imported_tokens != imported_tokens
            or lease.identity.deadline_monotonic_ns <= time.monotonic_ns()
        ):
            raise ValueError("No matching captured GLM terminal worker frame")
        if entry.checkpoint is None:
            resolved = []

            def resolve(pin, observed):
                source = wait_native_source(pin, observed)
                resolved.append(source)
                return source

            entry.checkpoint = finalize_terminal_rank(
                lease, entry.observed, entry.receipt, signer, resolve
            )
            (entry.source,) = resolved
        return entry.checkpoint

    def terminal_window_source(
        self,
        lease: "TerminalPinLease",
        permit: "TerminalReadPermit",
        manifest: "GlmTerminalManifest",
    ) -> "FencedTerminalSource":
        """Resolve the already verified local rank under the trusted read permit."""
        signer, _ = self._require_terminal_source()
        entry = self._kitepond_terminal_frames.get(lease.identity.request_id)
        if (
            entry is None
            or entry.lease != lease
            or entry.source is None
            or entry.checkpoint is None
            or permit.lease_sha256 != lease.sha256
            or manifest.lease_sha256 != lease.sha256
            or permit.manifest_sha256 != manifest.sha256
            or len(manifest.ranks) != 4
            or manifest.ranks[signer.rank] != entry.checkpoint
            or lease.identity.deadline_monotonic_ns <= time.monotonic_ns()
        ):
            raise ValueError("GLM terminal window has no verified native rank source")
        return entry.source

    def kitepond_register_terminal_return(self, transfer) -> None:
        """Register this native D rank's exact read permit before starting P fetch."""
        from .glm_return_bootstrap import current_decode_terminal_sources

        owner = current_decode_terminal_sources()
        owner.call(owner.register(transfer, self))

    def kitepond_terminal_return_result(self, transfer):
        """Read the authenticated P fence from this worker's native sender loop."""
        from .glm_return_bootstrap import current_decode_terminal_sources

        owner = current_decode_terminal_sources()
        return owner.call(owner.result(transfer.attempt(owner.scope.rank)))

    def kitepond_retire_terminal_return(self, transfer) -> None:
        """Remove the route after the owner has consumed its fenced receipt."""
        from .glm_return_bootstrap import current_decode_terminal_sources

        owner = current_decode_terminal_sources()

        owner.call(owner.retire_attempt(transfer.attempt(owner.scope.rank)))

    def kitepond_discard_terminal_frame(self, lease: "TerminalPinLease") -> bool:
        self._require_terminal_source()
        frames = self._kitepond_terminal_frames
        key = lease.identity.request_id
        entry = frames.get(key)
        if entry is None:
            return False
        if entry.lease != lease:
            raise ValueError("GLM terminal cancellation changed the native pin")
        frames.pop(key)
        return True
