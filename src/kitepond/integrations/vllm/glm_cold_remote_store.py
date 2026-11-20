# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the existing FSL2 store from EngineCore's borrowed D regions."""

from __future__ import annotations

from secrets import token_hex
from typing import TYPE_CHECKING

from kitepond.core.contracts import Wait
from kitepond.models.glm5next.cold_authority import NativeRegionColdAuthority
from kitepond.models.glm5next.cold_lifecycle import GlmColdCheckpoint, GlmColdRamGrant

from .glm_cold_enrollment import NativeGlmColdRegions
from .glm_cold_rank_control import (
    RPC_NAME,
    ColdRankClaim,
    ColdRankCommand,
    sign_cold_rank_command,
)
from .glm_cold_runtime import GlmColdDirectoryImport
from .glm_terminal_manifest import TerminalRankSigner
from .native_glm_cold_directory import RetainedColdRam

if TYPE_CHECKING:
    from vllm.v1.core.sched.interface import SynchronousStepRPC


class NativeGlmColdStore:
    """Claim worker ranges through the trusted owner before queuing FSL2 work."""

    def __init__(
        self,
        regions: NativeGlmColdRegions,
        worker_control: SynchronousStepRPC,
        signers: tuple[TerminalRankSigner, ...],
    ) -> None:
        if (
            type(regions) is not NativeGlmColdRegions
            or tuple(signer.rank for signer in signers) != (0, 1, 2, 3)
            or not callable(worker_control)
        ):
            raise ValueError("Cold store needs authenticated native control")
        self.regions = regions
        self.region_fingerprints = tuple(
            region.descriptor.fingerprint() for region in regions.regions
        )
        self._worker_control = worker_control
        self._signers = signers
        self._boot = regions.regions[0].descriptor.authority_boot_id
        self._sources: dict[int, tuple[str, tuple[ColdRankClaim, ...]]] = {}

    def _call(self, command: ColdRankCommand) -> tuple[ColdRankClaim, ...] | Wait:
        raw = self._worker_control(
            RPC_NAME,
            timeout=5.0,
            args=(command, sign_cold_rank_command(command, self._signers)),
        )
        if len(raw) != 4:
            raise ValueError("Cold worker control has no four-rank receipt quorum")
        for rank, claim in enumerate(raw):
            if isinstance(claim, Wait):
                continue
            if (
                type(claim) is not ColdRankClaim
                or claim.rank != rank
                or claim.region_sha256 != self.region_fingerprints[rank]
                or claim.handle != command.handle
                or claim.identity != command.identity
                or claim.stage_id != command.stage_id
            ):
                raise ValueError("Cold worker control changed its native rank receipt")
        return next((claim for claim in raw if isinstance(claim, Wait)), tuple(raw))

    def _release(self, command: ColdRankCommand) -> None:
        if isinstance(self._call(command), Wait):
            raise RuntimeError("Cold worker did not attest release of its native lease")

    def retain_terminal(self, entry: GlmColdDirectoryImport) -> RetainedColdRam | Wait:
        checkpoint = entry.checkpoint
        handle = token_hex(32)
        retain = ColdRankCommand(
            "retain", handle, self._boot, checkpoint.identity, entry
        )
        release = ColdRankCommand("release", handle, self._boot, checkpoint.identity)
        try:
            claims = self._call(retain)
            if not isinstance(claims, Wait) and any(
                tuple(binding.part for binding in claim.bindings) != plan.parts
                or claim.stage is not None
                for claim, plan in zip(claims, checkpoint.plans, strict=True)
            ):
                raise ValueError(
                    "Cold worker did not claim the accepted terminal parts"
                )
        except Exception:
            self._release(release)
            raise
        if isinstance(claims, Wait):
            self._release(release)
            return claims

        def live():
            if id(source) not in self._sources:
                raise ValueError("Cold terminal source is no longer retained")

        def drop():
            live()
            self._release(release)
            del self._sources[id(source)]

        source = RetainedColdRam(checkpoint.identity, self.regions.regions, live, drop)
        self._sources[id(source)] = (handle, claims)
        return source

    def snapshot(
        self,
        checkpoint: GlmColdCheckpoint,
        source: RetainedColdRam,
        *,
        max_inflight_bytes: int,
    ) -> GlmColdRamGrant | Wait:
        held = self._sources.get(id(source))
        if held is None or source.identity != checkpoint.identity:
            raise ValueError("Cold store borrowed a foreign retained source")
        per_rank = tuple(
            max(part.nbytes for part in plan.parts) for plan in checkpoint.plans
        )
        if sum(per_rank) + min(max(per_rank), 65536) > max_inflight_bytes:
            return Wait("NATIVE_COLD_STAGING_BUDGET_WAIT")
        handle, original = held
        stage_id = token_hex(32)
        stage = ColdRankCommand(
            "stage",
            handle,
            self._boot,
            checkpoint.identity,
            stage_id=stage_id,
        )
        unstage = ColdRankCommand(
            "unstage",
            handle,
            self._boot,
            checkpoint.identity,
            stage_id=stage_id,
        )
        staged = True

        def live():
            source.require_live()
            if not staged:
                raise ValueError("Cold snapshot lost its worker staging lease")

        def settle(_state):
            nonlocal staged
            live()
            self._release(unstage)
            staged = False

        try:
            claims = self._call(stage)
            if not isinstance(claims, Wait):
                authorities = []
                for rank, (claim, saved) in enumerate(
                    zip(claims, original, strict=True)
                ):
                    if (
                        claim.bindings != saved.bindings
                        or claim.stage is None
                        or claim.stage.nbytes != per_rank[rank]
                    ):
                        raise ValueError("Cold store changed its prepared native bytes")
                    authorities.append(
                        NativeRegionColdAuthority(
                            region=self.regions.regions[rank],
                            identity=checkpoint.identity,
                            plan=checkpoint.plans[rank],
                            bindings=claim.bindings,
                            stage=claim.stage,
                            verify_reference=live,
                            mode="STORE",
                        )
                    )
                return GlmColdRamGrant(
                    checkpoint, tuple(authorities), self.regions.layout_sha256, settle
                )
        except Exception:
            settle("FAILED")
            raise
        settle("FAILED")
        return claims

    def reserve_restore(self, *_args, **_kwargs) -> Wait:
        return Wait("NATIVE_COLD_RESTORE_FENCE_UNAVAILABLE")
