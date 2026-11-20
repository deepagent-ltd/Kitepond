# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Take the native D terminal frame before a later model step can reuse it."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

from kitepond.integrations.vllm.checkpoint import CheckpointReadReceipt
from kitepond.integrations.vllm.glm_terminal_manifest import (
    GlmTerminalCheckpointVerifier,
    TerminalFrameReceipt,
)
from kitepond.integrations.vllm.terminal_checkpoint import (
    StockTerminalCheckpoints,
    StockTerminalCheckpointScheduler,
    TerminalPinLease,
    TerminalReadPermit,
)
from kitepond.integrations.vllm.terminal_host_prefix import NativeTerminalHostPrefix

if TYPE_CHECKING:
    from vllm.v1.core.sched.interface import (
        SynchronousStepCallback,
        SynchronousStepRPC,
    )
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

    from .glm_terminal_bootstrap import GlmTerminalStartup


@dataclass(frozen=True)
class _CaptureBinding:
    verifier: GlmTerminalCheckpointVerifier
    imported_boundary: Callable[[TerminalPinLease], int]
    rpc_timeout: float
    clock: Callable[[], int]
    terminal_host_prefix: NativeTerminalHostPrefix | None


class GlmTerminalCheckpointScheduler(StockTerminalCheckpointScheduler):
    """Explicit D-side scheduler with native committed-step capture."""

    def __init__(self, *args, **kwargs):
        if not callable(
            getattr(
                StockTerminalCheckpointScheduler, "get_synchronous_step_callback", None
            )
        ):
            raise RuntimeError("vLLM has no native synchronous terminal step callback")
        super().__init__(*args, **kwargs)
        self._terminal_capture: _CaptureBinding | None = None
        self._pending_host_writes: set[str] = set()
        from .glm_terminal_bootstrap import GlmTerminalStartup, deployment_enabled

        self._terminal_startup: GlmTerminalStartup | None = None
        if deployment_enabled(self.vllm_config):
            if not callable(
                getattr(
                    StockTerminalCheckpointScheduler,
                    "get_worker_startup_callback",
                    None,
                )
            ):
                raise RuntimeError("vLLM has no native worker startup callback")
            self._terminal_startup = GlmTerminalStartup(self)
        self._terminal_frames: dict[
            str, tuple[str, tuple[TerminalFrameReceipt, ...]]
        ] = {}

    def get_worker_startup_callback(
        self,
    ) -> Callable[[SynchronousStepRPC], None] | None:
        startup = self._terminal_startup
        return None if startup is None else startup.bind

    def bind_terminal_transport_settlement(
        self, verify: Callable[[TerminalReadPermit, CheckpointReadReceipt], None]
    ) -> None:
        startup = self._terminal_startup
        if startup is None:
            raise RuntimeError("GLM terminal worker startup is disabled")
        startup.bind_transport_settlement(verify)

    def get_synchronous_step_callback(self) -> SynchronousStepCallback:
        return self._capture_committed_terminal_frames

    def bind_glm_terminal_capture(
        self,
        *,
        source_boot_id: str,
        verifier: GlmTerminalCheckpointVerifier,
        verify_settlement: Callable[[TerminalReadPermit, CheckpointReadReceipt], None],
        imported_boundary: Callable[[TerminalPinLease], int],
        rpc_timeout: float,
        enable_native_host_fence: bool = False,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> StockTerminalCheckpoints:
        if (
            self._terminal_capture is not None
            or type(enable_native_host_fence) is not bool
            or not callable(imported_boundary)
            or not callable(verify_settlement)
            or type(rpc_timeout) not in (int, float)
            or not isfinite(rpc_timeout)
            or rpc_timeout <= 0
        ):
            raise ValueError(
                "GLM terminal capture requires an explicit bounded D binding"
            )
        terminal_host_prefix = (
            NativeTerminalHostPrefix(self.kv_cache_manager)
            if enable_native_host_fence
            else None
        )
        owner = self.bind_terminal_checkpoints(
            source_boot_id=source_boot_id,
            verify_checkpoint=self._verify_committed_terminal_manifest,
            verify_settlement=verify_settlement,
            clock=clock,
        )
        self._terminal_capture = _CaptureBinding(
            verifier, imported_boundary, rpc_timeout, clock, terminal_host_prefix
        )
        return owner

    def _capture_committed_terminal_frames(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
        worker_rpc: SynchronousStepRPC,
    ) -> None:
        owner, capture = self._terminal_checkpoints, self._terminal_capture
        if owner is None:
            return
        if capture is None:
            raise RuntimeError("GLM terminal scheduler has no bound capture verifier")
        deadline = None
        for req_id in scheduler_output.num_scheduled_tokens:
            lease = owner.lease_for(req_id)
            if lease is None or lease.step_index != self.current_step:
                continue
            if (
                req_id not in model_runner_output.req_id_to_index
                or req_id in self._terminal_frames
            ):
                raise ValueError("GLM terminal frame lacks its exact native completion")
            now = capture.clock()
            if lease.identity.deadline_monotonic_ns <= now:
                owner.expire(lease)
                continue
            if deadline is None:
                deadline = now + int(capture.rpc_timeout * 1e9)
            imported = capture.imported_boundary(lease)
            if (
                type(imported) is not int
                or not 0 <= imported < lease.requested_common_tokens
            ):
                raise ValueError("GLM terminal pin has no verified imported frontier")

            # Only ephemeral frame metadata is read here. Host durability can
            # require a subsequent step and must be handled outside this RPC.
            wait_seconds = (
                min(deadline, lease.identity.deadline_monotonic_ns) - capture.clock()
            ) / 1e9
            if wait_seconds <= 0:
                raise TimeoutError("GLM terminal capture exceeded the step deadline")
            receipts = worker_rpc(
                "kitepond_take_terminal_frame",
                timeout=wait_seconds,
                args=(lease, imported),
            )
            frames = capture.verifier.verify_frames(lease, tuple(receipts))
            if any(frame.proof.imported_tokens != imported for frame in frames):
                raise ValueError(
                    "GLM terminal worker frame changed the imported frontier"
                )
            self._terminal_frames[req_id] = (lease.sha256, frames)
            if (
                capture.terminal_host_prefix is not None
                and not capture.terminal_host_prefix.plan(lease)
            ):
                self._pending_host_writes.add(req_id)

    def schedule(self, *args, **kwargs):
        capture, owner = self._terminal_capture, self._terminal_checkpoints
        if capture is not None and capture.terminal_host_prefix is not None:
            assert owner is not None
            for req_id in tuple(self._pending_host_writes):
                lease = owner.lease_for(req_id)
                if (
                    lease is None
                    or capture.clock() >= lease.identity.deadline_monotonic_ns
                    or capture.terminal_host_prefix.plan(lease)
                ):
                    self._pending_host_writes.discard(req_id)
        return super().schedule(*args, **kwargs)

    def _verify_committed_terminal_manifest(
        self, lease: TerminalPinLease, proof: object
    ) -> str:
        capture = self._terminal_capture
        frames = self._terminal_frames.get(lease.identity.request_id)
        if capture is None or frames is None or frames[0] != lease.sha256:
            raise ValueError(
                "GLM terminal manifest has no committed worker-frame quorum"
            )
        if self._terminal_startup is not None:
            self._terminal_startup.require_transport_settlement()
        if capture.terminal_host_prefix is not None:
            capture.terminal_host_prefix.require_durable(lease)
        return capture.verifier.verify_captured_manifest(lease, proof, frames[1])

    def _release_terminal_request(self, request) -> None:
        self._terminal_frames.pop(request.request_id, None)
        self._pending_host_writes.discard(request.request_id)
        super()._release_terminal_request(request)
