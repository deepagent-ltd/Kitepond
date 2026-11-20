# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-owned native D terminal references for a later P transfer."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from kitepond.integrations.vllm.checkpoint import CheckpointReadReceipt

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheGroupRole,
)
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

_TERMINAL = {RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED}
_SOURCE_RANKS = frozenset(range(4))


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _is_digest(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


@dataclass(frozen=True)
class TerminalTransferIdentity:
    request_id: str
    transfer_id: str
    epoch: int
    source_engine_id: str
    source_boot_id: str
    model_fingerprint: str
    config_fingerprint: str
    deadline_monotonic_ns: int

    def __post_init__(self) -> None:
        if (
            any(
                not isinstance(value, str) or not value
                for value in (
                    self.request_id,
                    self.transfer_id,
                    self.source_engine_id,
                    self.source_boot_id,
                )
            )
            or type(self.epoch) is not int
            or self.epoch < 0
            or not _is_digest(self.model_fingerprint)
            or not _is_digest(self.config_fingerprint)
            or type(self.deadline_monotonic_ns) is not int
            or self.deadline_monotonic_ns <= 0
        ):
            raise ValueError("Invalid native terminal transfer identity")


@dataclass(frozen=True)
class TerminalCacheGroup:
    group_id: int
    location: str
    layer_names: tuple[str, ...]
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class TerminalPinLease:
    """Exact native pin, pending authoritative target/draft and byte evidence."""

    identity: TerminalTransferIdentity
    step_index: int
    requested_common_tokens: int
    visible_tokens: int
    prefix_sha256: str
    next_token_id: int
    groups: tuple[TerminalCacheGroup, ...]
    nonce: str = field(default_factory=lambda: secrets.token_hex(16))

    @property
    def sha256(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class TerminalReadPermit:
    lease_sha256: str
    manifest_sha256: str

    @property
    def sha256(self) -> str:
        return _sha256(asdict(self))


@dataclass
class _Hold:
    request: Request
    lease: TerminalPinLease
    refs: tuple[tuple[BlockPool, tuple[KVCacheBlock, ...]], ...]
    permit: TerminalReadPermit | None = None
    closing: bool = False
    settlements: dict[int, CheckpointReadReceipt] = field(default_factory=dict)


class StockTerminalCheckpoints:
    """Retain the D scheduler's real host and device allocation references."""

    def __init__(
        self,
        manager: KVCacheManager,
        max_leases: int,
        *,
        source_engine_id: str,
        source_boot_id: str,
        on_release: Callable[[Request], None],
        verify_checkpoint: Callable[[TerminalPinLease, object], str] | None = None,
        verify_settlement: Callable[[TerminalReadPermit, CheckpointReadReceipt], None]
        | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if type(max_leases) is not int or max_leases <= 0:
            raise ValueError("Terminal pin capacity must be a positive integer")
        if not all(
            isinstance(v, str) and v for v in (source_engine_id, source_boot_id)
        ):
            raise ValueError("Native terminal ownership needs a local engine and boot")
        self._identity = (source_engine_id, source_boot_id)
        self._manager = manager
        self._max_leases = max_leases
        self._on_release = on_release
        self._verify_checkpoint = verify_checkpoint
        self._verify_settlement = verify_settlement
        self._clock = clock
        self._owner = threading.get_ident()
        self._armed: dict[str, tuple[Request, TerminalTransferIdentity]] = {}
        self._active: dict[str, _Hold] = {}
        self._terminal_receipts: deque[
            tuple[TerminalReadPermit, dict[int, CheckpointReadReceipt]]
        ] = deque(maxlen=max_leases)
        self._scheduled: tuple[SchedulerOutput, int] | None = None
        self._completion: ModelRunnerOutput | None = None
        self._read_observer = None

    def bind_read_observer(self, observer) -> None:
        """Bind one trusted feature while retaining the native permit owner."""
        self._require_owner()
        if self._read_observer is not None or not callable(observer):
            raise ValueError("Terminal read observer is invalid or already bound")
        self._read_observer = observer

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError("Native terminal checkpoints belong to the scheduler")

    def arm(self, request: Request, identity: TerminalTransferIdentity) -> None:
        self._require_owner()
        req_id = request.request_id
        live = (entry[1] for entry in self._armed.values())
        active = (hold.lease.identity for hold in self._active.values())
        if (
            type(identity) is not TerminalTransferIdentity
            or identity.request_id != req_id
            or (identity.source_engine_id, identity.source_boot_id) != self._identity
            or any(
                (old.transfer_id, old.epoch) == (identity.transfer_id, identity.epoch)
                for old in (*live, *active)
            )
            or request.is_finished()
            or req_id in self._armed
            or req_id in self._active
            or len(self._armed) + len(self._active) >= self._max_leases
            or identity.deadline_monotonic_ns <= self._clock()
        ):
            raise ValueError("Native terminal pin cannot arm this request")
        self._armed[req_id] = (request, identity)

    def require_available_schedule(self) -> None:
        self._require_owner()
        if self._scheduled is not None:
            raise RuntimeError("Native terminal scheduler already has a tracked step")

    def note_schedule(self, output: SchedulerOutput, step: int) -> None:
        self.require_available_schedule()
        if self._armed.keys() & output.num_scheduled_tokens.keys():
            self._scheduled = (output, step)

    def enter_completion(
        self, output: SchedulerOutput, result: ModelRunnerOutput
    ) -> None:
        self._require_owner()
        if self._scheduled is not None:
            if self._scheduled[0] is not output or self._completion is not None:
                raise ValueError("Native terminal completion is not the tracked step")
            self._completion = result

    def leave_completion(self) -> None:
        self._require_owner()
        self._completion = None
        self._scheduled = None

    def lease_for(self, request_id: str) -> TerminalPinLease | None:
        self._require_owner()
        hold = self._active.get(request_id)
        return None if hold is None else hold.lease

    def capture(self, request: Request) -> TerminalPinLease | None:
        """Capture native request tables before the scheduler can release them."""
        self._require_owner()
        req_id = request.request_id
        armed = self._armed.get(req_id)
        if armed is None:
            return None
        if request is not armed[0]:
            raise ValueError("Native terminal completion changed the request object")
        identity = armed[1]
        if (
            request.status not in _TERMINAL
            or identity.deadline_monotonic_ns <= self._clock()
        ):
            del self._armed[req_id]
            return None

        scheduled, model_output = self._scheduled, self._completion
        common, visible = request.num_computed_tokens, request.num_tokens
        if (
            scheduled is None
            or model_output is None
            or scheduled[1] <= 0
            or scheduled[0].num_scheduled_tokens.get(req_id, 0) <= 0
            or req_id not in model_output.req_id_to_index
            or request.num_in_flight_tokens
            or request.num_stale_output_tokens
            or not 0 < common < visible
        ):
            raise ValueError("Terminal request lacks its actual finished model step")

        groups, refs = self._native_groups(req_id)
        lease = TerminalPinLease(
            identity=identity,
            step_index=scheduled[1],
            requested_common_tokens=common,
            visible_tokens=visible,
            prefix_sha256=_sha256(list(request.all_token_ids[:common])),
            next_token_id=request.all_token_ids[common],
            groups=groups,
        )
        touched: list[tuple[BlockPool, KVCacheBlock]] = []
        try:
            for pool, blocks in refs:
                for block in blocks:
                    pool.touch((block,))
                    touched.append((pool, block))
        except BaseException:
            for pool, block in reversed(touched):
                pool.free_blocks((block,))
            raise
        self._active[req_id] = _Hold(request, lease, refs)
        del self._armed[req_id]
        return lease

    def _native_groups(self, request_id):
        groups = []
        pools: dict[str, tuple[BlockPool, dict[int, KVCacheBlock]]] = {}
        for group_id, (group, manager, blocks) in enumerate(
            zip(
                self._manager.kv_cache_config.kv_cache_groups,
                self._manager.coordinator.single_type_managers,
                self._manager.get_blocks(request_id).blocks,
                strict=True,
            )
        ):
            if isinstance(group.kv_cache_spec, (HiSparseResidentSpec, HiSparseHotSpec)):
                continue
            kind = "host" if group.host_resident else "device"
            if group.host_resident != (group.role is KVCacheGroupRole.HISPARSE_SOURCE):
                raise ValueError(
                    "Native terminal host authority is not the HiSparse source"
                )
            pool, held = pools.setdefault(kind, (manager.block_pool, {}))
            if pool is not manager.block_pool:
                raise ValueError("Native terminal cache changed its authoritative pool")
            for block in blocks:
                if block.is_null:
                    continue
                if (
                    not 0 < block.block_id < len(pool.blocks)
                    or pool.blocks[block.block_id] is not block
                    or block.ref_cnt <= 0
                ):
                    raise ValueError("Native terminal table contains a dead block")
                held[block.block_id] = block
            groups.append(
                TerminalCacheGroup(
                    group_id,
                    kind,
                    tuple(group.layer_names),
                    tuple(b.block_id for b in blocks),
                )
            )
        if set(pools) != {"host", "device"} or any(
            not blocks for _, blocks in pools.values()
        ):
            raise ValueError(
                "Terminal pin needs both native HiSparse and device owners"
            )
        refs = tuple(
            (pool, tuple(blocks[key] for key in sorted(blocks)))
            for pool, blocks in pools.values()
        )
        return tuple(groups), refs

    def _require_live(self, lease: TerminalPinLease) -> _Hold:
        self._require_owner()
        hold = self._active.get(lease.identity.request_id)
        if hold is None or hold.lease is not lease:
            raise ValueError(
                "Native terminal lease is stale or belongs to another attempt"
            )
        return hold

    def begin_reads(
        self, lease: TerminalPinLease, checkpoint_proof: object
    ) -> TerminalReadPermit:
        hold = self._require_live(lease)
        if self._verify_checkpoint is None or self._verify_settlement is None:
            raise RuntimeError(
                "Native terminal checkpoint and receipt verifiers are unavailable"
            )
        if (
            hold.permit is not None
            or lease.identity.deadline_monotonic_ns <= self._clock()
        ):
            raise ValueError("Native terminal transport is already open or expired")
        manifest = self._verify_checkpoint(lease, checkpoint_proof)
        if not _is_digest(manifest):
            raise ValueError("Exact accepted terminal checkpoint manifest is invalid")
        permit = TerminalReadPermit(lease.sha256, manifest)
        hold.permit = permit
        if self._read_observer is not None:
            self._read_observer(lease, permit, checkpoint_proof)
        return permit

    def require_read_open(
        self, lease: TerminalPinLease, permit: TerminalReadPermit
    ) -> None:
        hold = self._require_live(lease)
        if (
            hold.permit is not permit
            or hold.closing
            or lease.identity.deadline_monotonic_ns <= self._clock()
        ):
            raise ValueError("Native terminal transport cannot open new source work")

    def cancel(self, lease: TerminalPinLease) -> bool:
        hold = self._require_live(lease)
        if hold.permit is not None:
            hold.closing = True
            # The transport must stop issuing work and settle all source ranks.
            return False
        self._release(hold)
        return True

    def expire(self, lease: TerminalPinLease) -> bool:
        self._require_live(lease)
        if lease.identity.deadline_monotonic_ns > self._clock():
            return False
        return self.cancel(lease)

    def settle(
        self,
        lease: TerminalPinLease,
        permit: TerminalReadPermit,
        receipt: CheckpointReadReceipt,
    ) -> bool:
        self._require_owner()
        if (
            type(lease) is not TerminalPinLease
            or type(permit) is not TerminalReadPermit
            or type(receipt) is not CheckpointReadReceipt
            or receipt.lease_sha256 != permit.sha256
            or permit.lease_sha256 != lease.sha256
        ):
            raise ValueError("Native terminal receipt differs from its active transfer")
        for closed, settled in self._terminal_receipts:
            if closed is permit:
                if settled.get(receipt.local_rank) != receipt:
                    raise ValueError(
                        "Conflicting already settled terminal rank receipt"
                    )
                return True
        hold = self._require_live(lease)
        if hold.permit is not permit or self._verify_settlement is None:
            raise ValueError("Native terminal transfer did not authorize reads")
        previous = hold.settlements.get(receipt.local_rank)
        if previous is not None:
            if previous != receipt:
                raise ValueError("Conflicting native terminal rank receipt")
            return False
        self._verify_settlement(permit, receipt)
        hold.settlements[receipt.local_rank] = receipt
        if hold.settlements.keys() != _SOURCE_RANKS:
            return False
        self._release(hold)
        self._terminal_receipts.append((permit, hold.settlements))
        return True

    def _release(self, hold: _Hold) -> None:
        for pool, blocks in hold.refs:
            pool.free_blocks(reversed(blocks))
        del self._active[hold.request.request_id]
        self._on_release(hold.request)


class StockTerminalCheckpointScheduler(Scheduler):
    """Explicit native pre-free capture with independent connector ownership."""

    _terminal_checkpoints: StockTerminalCheckpoints | None = None

    def bind_terminal_checkpoints(
        self, *, source_boot_id: str, **verifiers
    ) -> StockTerminalCheckpoints:
        transfer = self.vllm_config.kv_transfer_config
        if (
            transfer is None
            or self._terminal_checkpoints is not None
            or self.vllm_config.scheduler_config.async_scheduling is not False
            or self.vllm_config.max_concurrent_batches != 1
            or self.ec_connector is not None
        ):
            raise ValueError(
                "Terminal pin needs one synchronous native scheduler owner"
            )
        self._terminal_native_can_release: dict[str, bool] = {}
        owner = StockTerminalCheckpoints(
            self.kv_cache_manager,
            self.vllm_config.scheduler_config.max_num_seqs,
            source_engine_id=transfer.engine_id,
            source_boot_id=source_boot_id,
            on_release=self._release_terminal_request,
            **verifiers,
        )
        self._terminal_checkpoints = owner
        return owner

    def schedule(self, *args, **kwargs):
        owner = self._terminal_checkpoints
        if owner is not None:
            owner.require_available_schedule()
        output = super().schedule(*args, **kwargs)
        if owner is not None:
            owner.note_schedule(output, self.current_step)
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        owner = self._terminal_checkpoints
        if owner is None:
            return super().update_from_output(scheduler_output, model_runner_output)
        owner.enter_completion(scheduler_output, model_runner_output)
        try:
            return super().update_from_output(scheduler_output, model_runner_output)
        finally:
            owner.leave_completion()

    def _connector_finished(self, request):
        delayed, payload = super()._connector_finished(request)
        owner = self._terminal_checkpoints
        if owner is not None and owner.capture(request) is not None:
            self._terminal_native_can_release[request.request_id] = not delayed
            return True, payload
        return delayed, payload

    def _free_blocks(self, request):
        if (
            self._terminal_checkpoints is not None
            and request.request_id in self._terminal_native_can_release
        ):
            self._terminal_native_can_release[request.request_id] = True
            return
        super()._free_blocks(request)

    def _release_terminal_request(self, request):
        if self._terminal_native_can_release.pop(request.request_id):
            super()._free_blocks(request)
