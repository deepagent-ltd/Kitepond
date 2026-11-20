# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use Mooncake rank windows for one checked native GLM D-to-P import."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from threading import get_ident
from typing import TYPE_CHECKING, Literal

import torch
from kitepond.backends.fsl2 import digest
from kitepond.integrations.mooncake.live_receive import (
    CommittedRankWindows,
    NativeRankWindowWriter,
    RankWindowAttempt,
    ReverseRankWindowPeer,
    mooncake_receive_pool,
    receive_rank_windows,
)
from kitepond.integrations.mooncake.window import manifest_digest, plan_windows
from kitepond.models.glm5next.cold_restore import ColdRestoreLayout
from kitepond.models.glm5next.frontier import digest as digest_tokens
from kitepond.models.glm5next.layout import CacheRole
from kitepond.models.glm5next.live_windows import GlmRankWindowPlan, rank_window_records

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.v1.kv_cache_interface import HiSparseHotSpec, HiSparseResidentSpec
from vllm.v1.request import RequestStatus

from .borrowed_host import BorrowedHostMetadata
from .checkpoint import CheckpointReadReceipt
from .glm_live_receive import _NativeGlmSink
from .glm_terminal_manifest import GlmTerminalManifest
from .hisparse_attachment import HiSparseConnector
from .terminal_checkpoint import (
    StockTerminalCheckpoints,
    TerminalCacheGroup,
    TerminalPinLease,
    TerminalReadPermit,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from kitepond.integrations.mooncake.live_receive import RankWindowPeer
    from kitepond.integrations.mooncake.window import RegisteredWindowPool

    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

    from .glm_native_views import GlmNativeStateViews
    from .glm_return_bootstrap import DecodeTerminalRankResult
    from .glm_terminal_manifest import FencedTerminalSource
    from .glm_terminal_worker import GlmTerminalWorker
    from .host_authority import NativeHostGrant


@dataclass(frozen=True)
class PrefillTerminalReservation:
    request_id: str
    engine_id: str
    boot_id: str
    grant: NativeHostGrant
    groups: tuple[TerminalCacheGroup, ...]

    @property
    def sha256(self) -> str:
        return digest(asdict(self))

    @classmethod
    def capture(
        cls,
        scheduler: Scheduler,
        request: Request,
        lease: TerminalPinLease,
        grant: NativeHostGrant,
        *,
        boot_id: str,
    ) -> PrefillTerminalReservation:
        """Read the actual waiting P request's native writable allocation."""
        manager = scheduler.kv_cache_manager
        transfer = scheduler.vllm_config.kv_transfer_config
        boundary = lease.requested_common_tokens
        identity = lease.identity
        if (
            transfer is None
            or not transfer.engine_id
            or transfer.engine_id == identity.source_engine_id
            or not boot_id
            or scheduler.requests.get(request.request_id) is not request
            or request.status is not RequestStatus.WAITING_FOR_REMOTE_KVS
            or request.num_prompt_tokens <= boundary
            or digest_tokens(list(request.all_token_ids[:boundary]))
            != lease.prefix_sha256
            or request.all_token_ids[boundary] != lease.next_token_id
            or (grant.request_id, grant.epoch) != (request.request_id, identity.epoch)
            or (grant.engine_id, grant.boot_id)
            != (identity.source_engine_id, identity.source_boot_id)
            or grant.reservation_tokens < boundary
            or grant.prefix_tokens
            or grant.readonly_pages
            or grant.copy_tail is not None
        ):
            raise ValueError(
                "Native P terminal return lacks an exact writable waiting grant"
            )

        groups = []
        for group_id, (group, blocks) in enumerate(
            zip(
                manager.kv_cache_config.kv_cache_groups,
                manager.get_blocks(request.request_id).blocks,
                strict=True,
            )
        ):
            if isinstance(group.kv_cache_spec, (HiSparseResidentSpec, HiSparseHotSpec)):
                continue
            ids = tuple(block.block_id for block in blocks)
            if not ids or not any(
                block.ref_cnt > 0 and not block.is_null for block in blocks
            ):
                raise ValueError(
                    "Native P terminal return cache group has no reservation"
                )
            if group.host_resident:
                pool = blocks[0].pool
                if (
                    not isinstance(pool, BorrowedHostMetadata)
                    or (pool.engine_id, pool.boot_id)
                    != (grant.engine_id, grant.boot_id)
                    or ids != grant.writable_pages
                    or any(not pool.is_block_writable(block) for block in blocks)
                ):
                    raise ValueError(
                        "Native P terminal return host reservation is not writable"
                    )
                pinned = {
                    page
                    for source in lease.groups
                    if source.location == "host"
                    for page in source.block_ids
                }
                if not pinned.isdisjoint(ids):
                    raise ValueError(
                        "Native P terminal return would overwrite its D source"
                    )
            groups.append(
                TerminalCacheGroup(
                    group_id,
                    "host" if group.host_resident else "device",
                    tuple(group.layer_names),
                    ids,
                )
            )
        if sum(group.location == "host" for group in groups) != 1:
            raise ValueError(
                "Native P terminal return requires a unique host authority"
            )
        return cls(
            request.request_id, transfer.engine_id, boot_id, grant, tuple(groups)
        )


@dataclass(frozen=True)
class GlmTerminalReturn:
    lease: TerminalPinLease
    permit: TerminalReadPermit
    manifest: GlmTerminalManifest
    destination: PrefillTerminalReservation

    def records(self, rank: int) -> tuple[dict, ...]:
        if type(rank) is not int or not 0 <= rank < 4:
            raise ValueError("Terminal return requires one qualified native rank")
        accepted = self.manifest.ranks[rank].checkpoint.plan.parts
        return rank_window_records(tuple(("final", part) for part in accepted))

    def attempt(self, rank: int) -> RankWindowAttempt:
        source = self.lease.identity
        records = self.records(rank)
        return RankWindowAttempt(
            self.destination.request_id,
            source.transfer_id,
            source.epoch,
            rank,
            source.source_boot_id,
            self.permit.sha256,
            self.destination.sha256,
            manifest_digest(records),
        )


@dataclass
class _ActiveReturn:
    transfer: GlmTerminalReturn
    commits: dict[int, CommittedRankWindows] = field(default_factory=dict)


class TerminalReturnSettlements:
    """D scheduler owner for authenticated P post-fence completions."""

    def __init__(self, owner: StockTerminalCheckpoints):
        self._owner = owner
        self._active: dict[str, _ActiveReturn] = {}
        self._thread = get_ident()
        self._observed: tuple[TerminalReadPermit, CheckpointReadReceipt] | None = None

    def _require_owner(self) -> None:
        if get_ident() != self._thread:
            raise RuntimeError(
                "Native terminal return settlement belongs to its scheduler"
            )

    def prepare(
        self,
        lease: TerminalPinLease,
        permit: TerminalReadPermit,
        manifest: GlmTerminalManifest,
        destination: PrefillTerminalReservation,
    ) -> GlmTerminalReturn:
        self._require_owner()
        self._owner.require_read_open(lease, permit)
        if (
            type(manifest) is not GlmTerminalManifest
            or manifest.lease_sha256 != lease.sha256
            or manifest.sha256 != permit.manifest_sha256
            or type(destination) is not PrefillTerminalReservation
            or (destination.grant.engine_id, destination.grant.boot_id)
            != (lease.identity.source_engine_id, lease.identity.source_boot_id)
            or destination.grant.epoch != lease.identity.epoch
        ):
            raise ValueError("Native terminal return changed its approved transfer")
        transfer = GlmTerminalReturn(lease, permit, manifest, destination)
        existing = self._active.get(permit.sha256)
        if existing is not None:
            if existing.transfer != transfer:
                raise ValueError(
                    "Native terminal return already has another destination"
                )
            return existing.transfer
        self._active[permit.sha256] = _ActiveReturn(transfer)
        return transfer

    def record_source_commit(
        self, transfer: GlmTerminalReturn, commitment: CommittedRankWindows
    ) -> None:
        """Accept the actual D writer's completed commit on its local control path."""
        self._require_owner()
        active = self._active.get(transfer.permit.sha256)
        if (
            active is None
            or active.transfer is not transfer
            or type(commitment) is not CommittedRankWindows
        ):
            raise ValueError("Native terminal source commit has no active transfer")
        rank = commitment.attempt.rank
        if (
            commitment.attempt != transfer.attempt(rank)
            or commitment.consumed_sequence
            != len(plan_windows(transfer.records(rank))) - 1
            or active.commits.setdefault(rank, commitment) != commitment
        ):
            raise ValueError(
                "Native terminal rank commit changed its last consumed window"
            )

    def verify(
        self, permit: TerminalReadPermit, receipt: CheckpointReadReceipt
    ) -> None:
        """Pass to the native owner as its exact reverse completion verifier."""
        self._require_owner()
        observed = self._observed
        if observed is None or observed[0] is not permit or observed[1] is not receipt:
            raise ValueError("Unattested native P terminal readback settlement")

    def accept_prefill_fence(
        self,
        transfer: GlmTerminalReturn,
        attempt: RankWindowAttempt,
        receipt: CheckpointReadReceipt,
        *,
        peer_engine_id: str,
        peer_boot_id: str,
    ) -> bool:
        """Use only with the authenticated native P peer's settled result."""
        self._require_owner()
        active = self._active.get(transfer.permit.sha256)
        if (
            active is None
            or active.transfer is not transfer
            or self._observed is not None
            or receipt.status != "acked"
            or receipt.local_rank not in active.commits
            or (peer_engine_id, peer_boot_id)
            != (transfer.destination.engine_id, transfer.destination.boot_id)
            or attempt != transfer.attempt(receipt.local_rank)
            or receipt.lease_sha256 != transfer.permit.sha256
        ):
            raise ValueError(
                "Native P readback settlement changed its boot or destination"
            )
        return self._settle(transfer, receipt)

    def accept_native_worker_result(
        self, transfer: GlmTerminalReturn, result: DecodeTerminalRankResult
    ) -> bool:
        """Consume only a receipt obtained from the same native D rank via executor."""
        from .glm_return_bootstrap import DecodeTerminalRankResult

        self._require_owner()
        active = self._active.get(transfer.permit.sha256)
        if (
            type(result) is not DecodeTerminalRankResult
            or active is None
            or active.transfer is not transfer
            or self._observed is not None
            or result.proof.attempt != transfer.attempt(result.proof.attempt.rank)
        ):
            raise ValueError("Native D reverse result changed the admitted attempt")
        receipt = CheckpointReadReceipt(
            transfer.permit.sha256,
            result.proof.attempt.rank,
            result.proof.outcome,
            0,
        )
        if result.committed is not None:
            self.record_source_commit(transfer, result.committed)
        else:
            self._owner.cancel(transfer.lease)
        return self._settle(transfer, receipt)

    def _settle(
        self, transfer: GlmTerminalReturn, receipt: CheckpointReadReceipt
    ) -> bool:
        self._observed = (transfer.permit, receipt)
        try:
            complete = self._owner.settle(transfer.lease, transfer.permit, receipt)
            if complete:
                self._active.pop(transfer.permit.sha256)
            return complete
        finally:
            self._observed = None


def open_terminal_rank_writer(
    transfer: GlmTerminalReturn,
    worker: GlmTerminalWorker,
    rank: int,
    engine,
    pool: RegisteredWindowPool,
    *,
    destination: tuple[str, int, str],
    release_source: Callable[[], None],
    timeout: float,
) -> NativeRankWindowWriter:
    """Bind a signed terminal source to the existing rank window service."""
    source = worker.terminal_window_source(
        transfer.lease, transfer.permit, transfer.manifest
    )
    return open_fenced_terminal_rank_writer(
        transfer,
        source,
        rank,
        engine,
        pool,
        destination=destination,
        release_source=release_source,
        timeout=timeout,
    )


def open_fenced_terminal_rank_writer(
    transfer: GlmTerminalReturn,
    source: FencedTerminalSource,
    rank: int,
    engine,
    pool: RegisteredWindowPool,
    *,
    destination: tuple[str, int, str],
    release_source: Callable[[], None],
    timeout: float,
) -> NativeRankWindowWriter:
    """Use the already enrolled native source under the exact read permit."""
    attempt = transfer.attempt(rank)
    if rank != source.layout.rank:
        raise ValueError("Native terminal worker cannot read another rank")
    plan = GlmRankWindowPlan(source.layout, 0)

    def pack(window, payload):
        if time.monotonic_ns() >= transfer.lease.identity.deadline_monotonic_ns:
            raise TimeoutError("Native terminal read permit expired")
        return plan.pack(window, payload, read_final=source.read_chunk)

    return NativeRankWindowWriter(
        attempt,
        plan.manifest(),
        engine,
        pool,
        destination=destination,
        pack=pack,
        release_source=release_source,
        timeout=timeout,
    )


async def receive_terminal_prefill_rank(
    transfer: GlmTerminalReturn,
    cache: ColdRestoreLayout,
    layer_groups: Mapping[str, int],
    pool: RegisteredWindowPool,
    peer: RankWindowPeer,
    observe: Callable[[RankWindowAttempt, CheckpointReadReceipt], None],
    *,
    seed_imported_tail: Callable[[], None],
    reverse_peer: ReverseRankWindowPeer | None = None,
    stream: torch.cuda.Stream | None = None,
    timeout: float,
) -> None:
    """Restore and check exact admitted P bytes before acknowledging its fence."""
    layout = cache.layout
    identity = transfer.lease.identity
    pinned = {group.group_id: group for group in transfer.destination.groups}
    if (
        type(cache) is not ColdRestoreLayout
        or layout.request_id != transfer.destination.request_id
        or (layout.epoch, layout.model_fingerprint, layout.config_fingerprint)
        != (identity.epoch, identity.model_fingerprint, identity.config_fingerprint)
        or layout.target_valid_tokens != transfer.lease.requested_common_tokens
        or layout.committed_kda_slot != 0
        or set(layer_groups) != set(cache.layers)
        or any(
            (group := pinned.get(layer_groups[name])) is None
            or group.block_ids != layer.block_ids
            or layer.identity.layer_name not in group.layer_names
            or group.location
            != ("host" if layer.identity.cache_role is CacheRole.MLA else "device")
            for name, layer in cache.layers.items()
        )
    ):
        raise ValueError("Native P cache is not the admitted terminal destination")

    plan = GlmRankWindowPlan(cache, 0)
    attempt = transfer.attempt(cache.rank)
    sink = _NativeGlmSink(plan, stream, None, seed_imported_tail, needs_cow=False)

    def completed(outcome: Literal["loaded", "abort_settled"]) -> None:
        if outcome == "loaded" or reverse_peer is not None:
            observe(
                attempt,
                CheckpointReadReceipt(
                    transfer.permit.sha256,
                    cache.rank,
                    "acked" if outcome == "loaded" else "abort_settled",
                    0,
                ),
            )

    await receive_rank_windows(
        attempt,
        plan.manifest(),
        pool,
        peer,
        sink,
        completed,
        timeout=timeout,
        reverse_peer=reverse_peer,
    )


class GlmNativeTerminalReturnWindows:
    """Use the actual bidirectional Mooncake worker and its registered scratch."""

    def __init__(self, connector: MultiConnector, *, slots: int) -> None:
        if (
            not isinstance(connector, MultiConnector)
            or connector.role is not KVConnectorRole.WORKER
            or len(connector.sub_connectors) != 2
            or type(slots) is not int
            or slots < 2
        ):
            raise ValueError("Terminal reverse windows need the native worker pair")
        owner, mooncake = connector.sub_connectors
        if (
            not isinstance(owner, HiSparseConnector)
            or not isinstance(mooncake, MooncakeConnector)
            or mooncake.connector_worker is None
            or mooncake.connector_worker.is_kv_producer
            or mooncake.connector_worker.is_kv_consumer
        ):
            raise ValueError("Terminal reverse windows require Mooncake kv_both")
        self.stage = owner.stage
        self._owner = owner
        self._engine = mooncake.connector_worker.engine
        self._pool = mooncake_receive_pool(self._engine, slots)

    @property
    def pool_id(self) -> str:
        return self._pool.pool_id

    def open_decode_writer(
        self,
        transfer: GlmTerminalReturn,
        worker: GlmTerminalWorker,
        rank: int,
        *,
        destination: tuple[str, int, str],
        release_source: Callable[[], None],
        timeout: float,
    ) -> NativeRankWindowWriter:
        if self.stage != "decode":
            raise ValueError("Terminal reverse source is not the D worker")
        return open_terminal_rank_writer(
            transfer,
            worker,
            rank,
            self._engine,
            self._pool,
            destination=destination,
            release_source=release_source,
            timeout=timeout,
        )

    async def receive_at_prefill(
        self,
        transfer: GlmTerminalReturn,
        cache: ColdRestoreLayout,
        layer_groups: Mapping[str, int],
        peer: RankWindowPeer,
        observe: Callable[[RankWindowAttempt, CheckpointReadReceipt], None],
        *,
        native_views: GlmNativeStateViews,
        stream: torch.cuda.Stream | None,
        timeout: float,
        reverse_peer: ReverseRankWindowPeer | None = None,
        before_route: Callable[[RankWindowAttempt, CheckpointReadReceipt], None]
        | None = None,
    ) -> None:
        self._owner.require_prefill_return_worker(transfer, cache.rank)
        descriptor = native_views.descriptor
        grant = transfer.destination.grant
        if (
            descriptor.authority_engine_id,
            descriptor.authority_boot_id,
            descriptor.source_rank,
        ) != (
            grant.engine_id,
            grant.boot_id,
            cache.rank,
        ) or descriptor.fingerprint() != transfer.manifest.ranks[
            cache.rank
        ].checkpoint.authority_sha256:
            raise ValueError("Native P terminal return has another D host authority")

        seed_tail = native_views.prepare_prefill_return_tail(cache, layer_groups)

        def complete(attempt, receipt):
            if before_route is not None:
                before_route(attempt, receipt)
            self._owner.report_prefill_return_fence(transfer, attempt, receipt)
            observe(attempt, receipt)

        await receive_terminal_prefill_rank(
            transfer,
            cache,
            layer_groups,
            self._pool,
            peer,
            complete,
            seed_imported_tail=seed_tail,
            reverse_peer=reverse_peer,
            stream=stream,
            timeout=timeout,
        )

    def close(self) -> None:
        self._pool.close()
