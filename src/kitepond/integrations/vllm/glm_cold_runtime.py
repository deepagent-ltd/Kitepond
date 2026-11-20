# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assemble the cold lane on the engine's native scheduler owner."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import get_ident
from time import monotonic_ns
from typing import Protocol

from kitepond.core.contracts import Wait
from kitepond.core.policy import ColdJobScheduler, IdlePolicy, JobView, LaneBudget
from kitepond.models.glm5next.cold_lifecycle import GlmColdCheckpoint
from kitepond.models.glm5next.idle_owner_loop import GlmIdleColdLoop, GlmIdleRamOwner
from kitepond.models.glm5next.layout import Glm5NextPDLayout

from .glm_terminal_manifest import GlmTerminalManifest
from .terminal_checkpoint import (
    StockTerminalCheckpoints,
    TerminalPinLease,
    TerminalReadPermit,
)


@dataclass(frozen=True)
class ColdDirectoryScope:
    namespace: str
    generation: int


@dataclass(frozen=True)
class GlmColdDirectoryImport:
    """One accepted descriptor with its live native ownership evidence."""

    checkpoint: GlmColdCheckpoint
    lease: TerminalPinLease
    permit: TerminalReadPermit
    manifest: GlmTerminalManifest

    def require_exact_checkpoint(self) -> None:
        checkpoint = self.checkpoint
        _validate_terminal_layout(
            self.lease, self.permit, self.manifest, checkpoint.layout
        )
        if (
            checkpoint.terminal_manifest_sha256 != self.permit.manifest_sha256
            or checkpoint.plans
            != tuple(rank.checkpoint.plan for rank in self.manifest.ranks)
        ):
            raise ValueError("Cold directory import changed its accepted rank plan")


def _validate_terminal_layout(lease, permit, manifest, layout) -> None:
    identity = lease.identity
    layout.require_decode_ready()
    if (
        manifest.lease_sha256 != lease.sha256
        or manifest.sha256 != permit.manifest_sha256
        or layout.request_id != identity.request_id
        or layout.model_fingerprint != identity.model_fingerprint
        or layout.config_fingerprint != identity.config_fingerprint
        or layout.epoch != identity.epoch
        or layout.target_valid_tokens != lease.requested_common_tokens
        or any(
            state.accepted_row != layout.committed_kda_slot
            for rank in manifest.ranks
            for state in rank.checkpoint.kda
        )
    ):
        raise ValueError("Cold directory import changed its accepted native state")


class GlmColdDirectory(GlmIdleRamOwner, Protocol):
    """Port to the engine's existing live RAM entry directory."""

    engine_id: str
    boot_id: str

    def cold_scope(self, lease: TerminalPinLease) -> ColdDirectoryScope | Wait:
        """Resolve namespace and current generation from native business state."""

    def import_accepted_terminal(self, entry: GlmColdDirectoryImport) -> Wait | None:
        """Atomically retain independent live RAM refs and register the entry.

        Same-generation repeat imports are idempotent. Verify the entry and
        allocator refs against current directory ownership before publishing.
        Wait retains no new references. This callback performs no NVMe I/O.
        """


class QuiescentRegistrar(Protocol):
    def __call__(
        self, *, interval_ns: int, callback: Callable[[], tuple[JobView, ...]]
    ) -> None: ...


class GlmColdRuntime:
    """Bind the trusted directory to a cadence-limited native owner callback."""

    def __init__(
        self,
        root: Path,
        directory: GlmColdDirectory,
        terminal: StockTerminalCheckpoints,
        *,
        register_quiescent: QuiescentRegistrar,
        cadence_ns: int,
        idle: IdlePolicy,
        foreground: LaneBudget,
        background: LaneBudget,
        max_jobs: int = 128,
        scan_limit: int = 32,
        clock: Callable[[], int] = monotonic_ns,
    ) -> None:
        if type(cadence_ns) is not int or cadence_ns <= 0:
            raise ValueError("Cold runtime needs an explicit positive owner cadence")
        if not isinstance(terminal, StockTerminalCheckpoints):
            raise TypeError("Cold runtime requires the native terminal pin owner")
        self.directory, self.terminal = directory, terminal
        self._scope = (directory.engine_id, directory.boot_id)
        self._thread = get_ident()
        self._clock = clock
        self._cadence_ns = cadence_ns
        self._last_callback_ns: int | None = None
        self._idle = GlmIdleColdLoop(
            root,
            directory,
            idle=idle,
            foreground=foreground,
            background=background,
            max_jobs=max_jobs,
            scan_limit=scan_limit,
            clock=clock,
        )
        register_quiescent(interval_ns=cadence_ns, callback=self._advance_quiescent)

    @property
    def scheduler(self) -> ColdJobScheduler:
        return self._idle.scheduler

    def _require_owner(self) -> None:
        if get_ident() != self._thread:
            raise RuntimeError("Cold runtime belongs to the native scheduler owner")

    def _advance_quiescent(self) -> tuple[JobView, ...]:
        self._require_owner()
        now = self._clock()
        last = self._last_callback_ns
        if type(now) is not int or now < 0 or (last is not None and now < last):
            raise ValueError("Cold runtime clock must remain monotonic")
        if last is not None and now - last < self._cadence_ns:
            return ()
        self._last_callback_ns = now
        return self._idle.tick()

    def demote_idle(self, scope: ColdDirectoryScope) -> int | Wait:
        """Submit explicit idle demotion using the current real business clock."""
        self._require_owner()
        select = getattr(self.directory, "resident_checkpoint", None)
        if not callable(select):
            return Wait("GLM_COLD_OPERATOR_UNAVAILABLE")
        checkpoint = select(scope)
        if isinstance(checkpoint, Wait):
            return checkpoint
        if type(checkpoint) is not GlmColdCheckpoint:
            raise ValueError(
                "Cold operator did not receive a native accepted checkpoint"
            )
        if self.scheduler.has_job(checkpoint):
            return Wait("GLM_COLD_GENERATION_IN_USE")
        try:
            return self.scheduler.submit("STORE", checkpoint, operator_idle=True)
        except BufferError:
            return Wait("COLD_IO_ADMISSION_WAIT")

    def import_terminal(
        self,
        lease: TerminalPinLease,
        permit: TerminalReadPermit,
        manifest: GlmTerminalManifest,
        layout: Glm5NextPDLayout,
    ) -> GlmColdCheckpoint | Wait:
        """Enroll an accepted terminal in RAM while the original pin is live."""
        self._require_owner()
        if (
            type(lease) is not TerminalPinLease
            or type(permit) is not TerminalReadPermit
            or type(manifest) is not GlmTerminalManifest
            or type(layout) is not Glm5NextPDLayout
        ):
            raise TypeError("Cold directory requires exact native terminal evidence")
        # Object identity here is the authorization, rather than a copied digest.
        self.terminal.require_read_open(lease, permit)
        identity = lease.identity
        if (identity.source_engine_id, identity.source_boot_id) != self._scope:
            raise ValueError("Cold directory import changed its accepted native state")
        _validate_terminal_layout(lease, permit, manifest, layout)
        scope = self.directory.cold_scope(lease)
        if isinstance(scope, Wait):
            return scope
        if type(scope) is not ColdDirectoryScope:
            raise TypeError("Cold scope must come from the native directory")
        checkpoint = GlmColdCheckpoint(
            scope.namespace,
            scope.generation,
            layout,
            permit.manifest_sha256,
            tuple(rank.checkpoint.plan for rank in manifest.ranks),
        )
        accepted = self.directory.import_accepted_terminal(
            GlmColdDirectoryImport(checkpoint, lease, permit, manifest)
        )
        if isinstance(accepted, Wait):
            return accepted
        if accepted is not None:
            raise RuntimeError("Native directory did not settle its import contract")
        return checkpoint
