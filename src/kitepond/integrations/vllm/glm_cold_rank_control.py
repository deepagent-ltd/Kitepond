# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Authenticated native D rank claims for idle FSL2 writes."""

from __future__ import annotations

from dataclasses import dataclass
from hmac import compare_digest
from typing import TYPE_CHECKING, Literal

from kitepond.backends.fsl2 import BundleIdentity
from kitepond.core.contracts import Wait
from kitepond.models.glm5next.cold_lifecycle import GlmColdCheckpoint
from kitepond.models.glm5next.import_recipe import ByteSpan, ColdBinding

from .glm_cold_runtime import GlmColdDirectoryImport
from .glm_terminal_manifest import TerminalRankSigner, _is_digest
from .native_glm_cold_provider import (
    NativeColdRankBytes,
    NativeColdScratch,
    NativeGlmRankMemory,
)

if TYPE_CHECKING:
    from .glm_terminal_worker import GlmTerminalWorker

RPC_NAME = "kitepond_control_cold_rank"
_Verb = Literal["retain", "stage", "unstage", "release"]


@dataclass(frozen=True)
class ColdRankCommand:
    """Operate only on one authorized logical generation in one D boot."""

    verb: _Verb
    handle: str
    boot_id: str
    identity: BundleIdentity
    entry: GlmColdDirectoryImport | None = None
    stage_id: str | None = None

    def signed_message(self) -> dict:
        return {
            "schema": "kitepond.glm-native-cold-rank.v1",
            "verb": self.verb,
            "handle": self.handle,
            "stage_id": self.stage_id,
            "boot_id": self.boot_id,
            "identity": {
                "namespace": self.identity.namespace,
                "generation": self.identity.generation,
                "model": self.identity.model_fingerprint,
                "semantic": self.identity.semantic_fingerprint,
                "boundary": self.identity.boundary_tokens,
            },
            "permit": self.entry.permit.sha256 if self.entry is not None else None,
        }


@dataclass(frozen=True)
class ColdRankClaim:
    rank: int
    region_sha256: str
    handle: str
    identity: BundleIdentity
    bindings: tuple[ColdBinding, ...]
    stage: ByteSpan | None = None
    stage_id: str | None = None


@dataclass
class _Resident:
    checkpoint: GlmColdCheckpoint
    data: NativeColdRankBytes
    stage: NativeColdScratch | None = None
    stage_id: str | None = None


class GlmWorkerColdLeases:
    """Keep worker-local native tail leases while the directory owns MLA pages."""

    def __init__(self, worker: GlmTerminalWorker, memory: NativeGlmRankMemory) -> None:
        self.worker, self.memory = worker, memory
        self._retained: dict[str, _Resident] = {}

    def _require_native(self) -> None:
        views = self.worker.kitepond_native_state_views()
        if (
            views.rank != self.memory.rank
            or views.descriptor != self.memory.region.descriptor
        ):
            raise ValueError("Cold worker lost its initialized native cache views")

    def handle(self, command: ColdRankCommand, signatures: tuple[str, ...]):
        signer, _ = self.worker._require_terminal_source()
        if (
            type(command) is not ColdRankCommand
            or type(command.identity) is not BundleIdentity
            or command.boot_id != self.memory.region.descriptor.authority_boot_id
            or command.verb not in ("retain", "stage", "unstage", "release")
            or not _is_digest(command.handle)
            or (command.verb == "retain")
            != (type(command.entry) is GlmColdDirectoryImport)
            or (command.verb != "retain" and command.entry is not None)
            or (
                not _is_digest(command.stage_id)
                if command.verb in ("stage", "unstage")
                else command.stage_id is not None
            )
            or type(signatures) is not tuple
            or len(signatures) != 4
            or not _is_digest(signatures[signer.rank])
            or not compare_digest(
                signatures[signer.rank], signer._sign(command.signed_message())
            )
        ):
            raise ValueError("Cold worker command does not authenticate its D owner")
        self._require_native()
        existing = self._retained.get(command.handle)
        if existing is not None and existing.checkpoint.identity != command.identity:
            raise ValueError("Cold worker generation changed its existing lease")
        if command.verb == "retain":
            return self._retain(command, existing)
        if command.verb == "stage":
            return self._stage(command, existing)
        if existing is not None:
            self._settle(command, existing)
        return self._claim(command)

    def _claim(self, command, *, data=None, stage=None) -> ColdRankClaim:
        return ColdRankClaim(
            self.memory.rank,
            self.memory.region_sha256,
            command.handle,
            command.identity,
            data.bindings if data is not None else (),
            stage.span if stage is not None else None,
            command.stage_id,
        )

    def _retain(self, command: ColdRankCommand, existing: _Resident | None):
        entry = command.entry
        assert entry is not None
        checkpoint = entry.checkpoint
        if checkpoint.identity != command.identity:
            raise ValueError("Cold worker command changed its accepted checkpoint")
        if existing is not None:
            return self._claim(command, data=existing.data)
        aux = self.memory.reserve_aux(
            checkpoint,
            "STORE",
            evidence=entry,
            require_native=self._require_native,
        )
        if isinstance(aux, Wait):
            return aux
        data = self.memory.prepare_from_worker(self.worker, entry, aux)
        if isinstance(data, Wait):
            return data
        self._retained[command.handle] = _Resident(checkpoint, data)
        return self._claim(command, data=data)

    def _stage(self, command: ColdRankCommand, existing: _Resident | None):
        if existing is None:
            return Wait("NATIVE_COLD_RANK_SOURCE_WAIT")
        existing.data.require_live()
        if existing.stage is None:
            scratch = self.memory.reserve_scratch(existing.checkpoint)
            if isinstance(scratch, Wait):
                return scratch
            existing.stage = scratch
            existing.stage_id = command.stage_id
        elif existing.stage_id != command.stage_id:
            return Wait("NATIVE_COLD_RANK_SCRATCH_BUSY")
        return self._claim(command, data=existing.data, stage=existing.stage)

    def _settle(self, command: ColdRankCommand, existing: _Resident) -> None:
        if command.verb == "release" and existing.stage is not None:
            raise ValueError("Cannot release native cold bytes while FSL2 owns scratch")
        if command.verb == "release":
            existing.data.settle("RELEASED")
            del self._retained[command.handle]
        elif existing.stage is not None and existing.stage_id == command.stage_id:
            existing.stage.release()
            existing.stage = None
            existing.stage_id = None


def sign_cold_rank_command(
    command: ColdRankCommand, signers: tuple[TerminalRankSigner, ...]
) -> tuple[str, ...]:
    if tuple(signer.rank for signer in signers) != (0, 1, 2, 3):
        raise ValueError("Cold rank control requires exact native signer order")
    message = command.signed_message()
    return tuple(signer._sign(message) for signer in signers)
