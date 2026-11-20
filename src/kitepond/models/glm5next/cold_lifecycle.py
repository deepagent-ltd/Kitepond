# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive GLM idle checkpoints through native RAM leases and four FSL2 ranks."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from kitepond.backends.fsl2 import (
    BundleIdentity,
    BundlePlan,
    ColdStorageAdapter,
    digest,
)
from kitepond.backends.fsl2_aggregate import FourRankManifestStore
from kitepond.core.contracts import Wait
from kitepond.core.policy import Mode, NativeModelColdAdapter, NativeModelColdJob

from .cold_authority import NativeRegionColdAuthority
from .layout import Glm5NextPDLayout

VERIFY_BYTES = 65536


@dataclass(frozen=True)
class GlmColdCheckpoint:
    """Exact accepted boundary, independent of the RAM page IDs on restore."""

    namespace: str
    generation: int
    layout: Glm5NextPDLayout
    terminal_manifest_sha256: str
    plans: tuple[BundlePlan, ...]
    identity: BundleIdentity = field(init=False)

    def __post_init__(self) -> None:
        self.layout.require_decode_ready()
        if (
            type(self.terminal_manifest_sha256) is not str
            or len(self.terminal_manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.terminal_manifest_sha256)
            or type(self.plans) is not tuple
            or len(self.plans) != 4
            or any(
                not isinstance(plan, BundlePlan)
                or any(part.rank != rank for part in plan.parts)
                for rank, plan in enumerate(self.plans)
            )
            or len({plan.representation_fingerprint for plan in self.plans}) != 1
        ):
            raise ValueError("GLM cold checkpoint needs its accepted four-rank layout")
        # Validate the complete persisted identity before an owner can acquire RAM.
        object.__setattr__(
            self,
            "identity",
            BundleIdentity(
                self.namespace,
                self.layout.model_fingerprint,
                digest(self.metadata),
                self.generation,
                self.layout.target_valid_tokens,
            ),
        )

    @property
    def metadata(self) -> dict:
        return {
            "schema": "glm-native-cold-checkpoint-v1",
            "model_fingerprint": self.layout.model_fingerprint,
            "config_fingerprint": self.layout.config_fingerprint,
            "epoch": self.layout.epoch,
            "layout_sha256": digest(self.layout.to_metadata()),
            "boundary_tokens": self.layout.target_valid_tokens,
            "terminal_manifest_sha256": self.terminal_manifest_sha256,
        }


@dataclass(frozen=True)
class GlmColdRamGrant:
    """Four live references reserved by the native owner for one operation.

    Settle is called on the owner loop only after every rank task and aggregate
    operation has drained. The owner releases temporary refs on every outcome.
    Only COMMITTED allows it to demote resident RAM after a final live-use check.
    Only VERIFIED allows it to keep the restore reservation for native import.
    """

    checkpoint: GlmColdCheckpoint
    ranks: tuple[NativeRegionColdAuthority, ...]
    ram_layout_sha256: str
    settle: Callable[[str], None]
    on_durable_publication: Callable[[dict], None] | None = None
    expected_aggregate_sha256: str | None = None


class GlmColdRamOwner(Protocol):
    """The native allocation owner implements idle/ref checks and RAM admission."""

    def last_business_use_ns(self, checkpoint: GlmColdCheckpoint) -> int: ...

    def acquire_snapshot(
        self,
        checkpoint: GlmColdCheckpoint,
        *,
        observed_last_use_ns: int,
        idle_cutoff_ns: int,
        max_inflight_bytes: int,
    ) -> GlmColdRamGrant | Wait:
        """Atomically recheck current epoch/idle time and acquire inactive refs."""

    def reserve_restore(
        self, checkpoint: GlmColdCheckpoint, *, max_inflight_bytes: int
    ) -> GlmColdRamGrant | Wait:
        """Reserve the exact current native destination and accepted checksums."""


class GlmFSL2ColdAdapter(NativeModelColdAdapter):
    """Connect the idle policy to real rank storage using model-owned RAM."""

    def __init__(self, root: Path, owner: GlmColdRamOwner) -> None:
        self.root, self.owner = Path(root), owner

    @staticmethod
    def _checkpoint(handle: object) -> GlmColdCheckpoint:
        if type(handle) is not GlmColdCheckpoint:
            raise TypeError("GLM cold jobs need an exact model checkpoint")
        return handle

    def last_business_use_ns(self, handle: object) -> int:
        return self.owner.last_business_use_ns(self._checkpoint(handle))

    def _prepare(
        self,
        mode: Mode,
        checkpoint: GlmColdCheckpoint,
        grant: GlmColdRamGrant | Wait,
        budget: int,
    ) -> NativeModelColdJob | Wait:
        if isinstance(grant, Wait):
            return grant
        if type(grant) is not GlmColdRamGrant or not callable(grant.settle):
            raise ValueError("GLM cold admission needs a model-owned RAM grant")
        try:
            return _RankJob(self.root, mode, checkpoint, grant, budget)
        except Exception:
            grant.settle("FAILED" if mode == "STORE" else "MISS")
            raise

    def begin_store(
        self,
        handle: object,
        *,
        observed_last_use_ns: int,
        idle_cutoff_ns: int,
        max_inflight_bytes: int,
    ) -> NativeModelColdJob | Wait:
        checkpoint = self._checkpoint(handle)
        grant = self.owner.acquire_snapshot(
            checkpoint,
            observed_last_use_ns=observed_last_use_ns,
            idle_cutoff_ns=idle_cutoff_ns,
            max_inflight_bytes=max_inflight_bytes,
        )
        return self._prepare("STORE", checkpoint, grant, max_inflight_bytes)

    def begin_restore(
        self, handle: object, *, max_inflight_bytes: int
    ) -> NativeModelColdJob | Wait:
        checkpoint = self._checkpoint(handle)
        grant = self.owner.reserve_restore(
            checkpoint, max_inflight_bytes=max_inflight_bytes
        )
        return self._prepare("RESTORE", checkpoint, grant, max_inflight_bytes)


class _RankJob(NativeModelColdJob):
    """At most one live FSL2 part. Full manifest verification runs off the loop."""

    def __init__(self, root, mode, checkpoint, grant, budget) -> None:
        super().__init__(mode, budget)
        identity = checkpoint.identity
        ranks = grant.ranks
        if (
            grant.checkpoint is not checkpoint
            or type(ranks) is not tuple
            or len(ranks) != 4
            or any(
                type(authority) is not NativeRegionColdAuthority
                or authority.mode != mode
                or authority.plan != plan
                or authority.identity != identity
                or authority.region.descriptor.layout_sha256
                != grant.ram_layout_sha256
                for authority, plan in zip(ranks, checkpoint.plans, strict=True)
            )
        ):
            raise ValueError("GLM native RAM grant changed epoch, rank or layout")
        owners = {
            (region.authority_engine_id, region.authority_boot_id)
            for region in (authority.region.descriptor for authority in ranks)
        }
        if len(owners) != 1:
            raise ValueError("GLM RAM grant does not share one native owner epoch")
        largest = max(part.nbytes for plan in checkpoint.plans for part in plan.parts)
        scratch = min(largest, VERIFY_BYTES)
        if sum(authority.stage.nbytes for authority in ranks) + scratch > budget:
            raise ValueError("GLM native staging exceeds the whole cold-job grant")
        adapters = []
        try:
            for rank, authority in enumerate(ranks):
                adapters.append(
                    ColdStorageAdapter(
                        root / f"rank-{rank}",
                        authority,
                        max_inflight_bytes=authority.stage.nbytes + scratch,
                        verify_chunk_bytes=scratch,
                    )
                )
            self.aggregate = FourRankManifestStore(root, tuple(adapters))
        except Exception:
            for adapter in adapters:
                adapter.close()
            raise
        self.checkpoint, self.identity, self.grant = checkpoint, identity, grant
        self.adapters = tuple(adapters)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._aggregate: Future | None = None
        self._rank_manifests: tuple[str, ...] | None = None
        self._rank = 0
        self._ticket = None
        self._cancelled = False
        self._state = "PENDING"

    def cancel(self) -> None:
        self._cancelled = True
        if self._ticket is not None:
            self._ticket.cancel()

    def _finish(self, state: str) -> str:
        self.grant.settle(state)
        self._state = state
        return state

    def _publish(self) -> dict | Wait | None:
        result = self.aggregate.publish(
            self.identity, self.checkpoint.plans, self.checkpoint.metadata
        )
        writer = self.grant.on_durable_publication
        if isinstance(result, dict) and writer is not None:
            writer(result)
        return result

    def _poll_aggregate(self) -> str:
        restoring = self.mode == "RESTORE"
        if self._aggregate is None:
            if self._cancelled:
                return self._finish("CANCELLED")
            args = (self.identity, self.checkpoint.plans)
            if restoring:
                self._aggregate = self._executor.submit(self.aggregate.discover, *args)
            else:
                self._aggregate = self._executor.submit(self._publish)
            return "PENDING"
        if not self._aggregate.done():
            return "PENDING"
        future, self._aggregate = self._aggregate, None
        if self._cancelled:
            return self._finish("CANCELLED")
        try:
            result = future.result()
        except (OSError, RuntimeError, ValueError):
            return self._finish("MISS" if restoring else "FAILED")
        if isinstance(result, Wait):
            return "PENDING"
        if result is None:
            return self._finish("MISS" if restoring else "FAILED")
        if not restoring:
            return self._finish("COMMITTED")
        expected = self.grant.expected_aggregate_sha256
        if expected is not None and result["digest"] != expected:
            return self._finish("MISS")
        self._rank_manifests = tuple(
            rank["manifest_sha256"] for rank in result["ranks"]
        )
        return "PENDING"

    def poll(self) -> str:
        if self._state != "PENDING":
            return self._state
        if (
            self.mode == "RESTORE" and self._rank_manifests is None
            or self.mode == "STORE" and self._rank == len(self.adapters)
        ):
            return self._poll_aggregate()
        if self._ticket is None and self._cancelled:
            return self._finish("CANCELLED")
        adapter = self.adapters[self._rank]
        if self._ticket is None:
            authority = self.grant.ranks[self._rank]
            try:
                if self.mode == "STORE":
                    ticket = adapter.begin_store(
                        self.identity, authority.plan, authority.lease
                    )
                else:
                    assert self._rank_manifests is not None
                    ticket = adapter.begin_restore(
                        self.identity,
                        authority.plan,
                        authority.lease,
                        expected_manifest_digest=self._rank_manifests[self._rank],
                    )
            except FileExistsError:
                # Resume only an immutable exact rank. Publish hashes every
                # payload again before declaring the aggregate durable.
                if (
                    self.mode == "STORE"
                    and adapter.discover(self.identity, authority.plan) is not None
                ):
                    self._rank += 1
                    return "PENDING"
                return self._finish("FAILED" if self.mode == "STORE" else "MISS")
            except (OSError, ValueError, RuntimeError):
                if adapter.ticket is not None:
                    # Keep a real ticket, even if submission reported an error.
                    self._ticket = adapter.ticket
                    self.cancel()
                    return "PENDING"
                return self._finish("FAILED" if self.mode == "STORE" else "MISS")
            if isinstance(ticket, Wait):
                if ticket.reason == "COLD_MISS":
                    return self._finish("MISS")
                return "PENDING"
            self._ticket = ticket
        state = adapter.poll(self._ticket)
        if state == "PENDING":
            return state
        self._ticket = None
        if self._cancelled:
            return self._finish("CANCELLED")
        expected = "COMMITTED" if self.mode == "STORE" else "VERIFIED"
        if state != expected:
            return self._finish(state)
        self._rank += 1
        if self._rank == len(self.adapters) and self.mode == "RESTORE":
            return self._finish("VERIFIED")
        return "PENDING"

    def reap(self) -> None:
        if self._state == "PENDING":
            raise RuntimeError("Native cold job still holds a RAM reference")
        self._executor.shutdown()
        for adapter in self.adapters:
            if not adapter.close():
                raise RuntimeError("Native cold adapter did not drain")
