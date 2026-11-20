# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Receive canonical windows around the native Mooncake registration owner."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from kitepond.core.readiness import _digest

from .window import (
    WINDOW_BYTES,
    RegisteredWindowPool,
    TransferWindow,
    WindowLease,
    WindowSendSession,
    manifest_digest,
    plan_windows,
)


class _MooncakeMemory(Protocol):
    def batch_register_memory(self, pointers: list[int], lengths: list[int]) -> int: ...
    def batch_unregister_memory(self, pointers: list[int]) -> int: ...


def mooncake_receive_pool(
    engine: _MooncakeMemory, slots: int, *, window_bytes: int = WINDOW_BYTES
) -> RegisteredWindowPool:
    """Register receiver scratch on the worker's existing TransferEngine."""

    def register(tensor):
        if engine.batch_register_memory([tensor.data_ptr()], [tensor.numel()]) != 0:
            raise RuntimeError("Mooncake receive scratch registration failed")

    def unregister(tensor):
        if engine.batch_unregister_memory([tensor.data_ptr()]) != 0:
            raise RuntimeError("Mooncake receive scratch unregistration failed")

    return RegisteredWindowPool(slots, register, unregister, window_bytes=window_bytes)


@dataclass(frozen=True)
class RankWindowAttempt:
    request_id: str
    transfer_id: str
    epoch: int
    rank: int
    source_boot_id: str
    grant_sha256: str
    destination_sha256: str
    manifest_sha256: str
    mla_rank_digests: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if (
            any(
                type(key) is not str or not key
                for key in (self.request_id, self.transfer_id, self.source_boot_id)
            )
            or type(self.epoch) is not int
            or not 0 <= self.epoch < 2**63
            or type(self.rank) is not int
            or not 0 <= self.rank < 4
            or (
                self.mla_rank_digests is not None
                and (
                    type(self.mla_rank_digests) is not tuple
                    or len(self.mla_rank_digests) != 4
                    or not all(_digest(value) for value in self.mla_rank_digests)
                )
            )
            or not all(
                _digest(value)
                for value in (
                    self.grant_sha256,
                    self.destination_sha256,
                    self.manifest_sha256,
                )
            )
        ):
            raise ValueError("Invalid qualified Mooncake rank rendezvous")


@dataclass(frozen=True)
class WrittenRankWindow:
    """Returned only by the bound Mooncake peer after a native sync write."""

    attempt: RankWindowAttempt
    window_sha256: str
    descriptor: dict


@dataclass(frozen=True)
class CommittedRankWindows:
    attempt: RankWindowAttempt
    consumed_sequence: int


@dataclass(frozen=True)
class AbortedRankWindows:
    attempt: RankWindowAttempt


@dataclass(frozen=True)
class FencedRankWindows:
    attempt: RankWindowAttempt
    outcome: Literal["acked", "abort_settled"]

    def __post_init__(self) -> None:
        if type(self.attempt) is not RankWindowAttempt or self.outcome not in (
            "acked",
            "abort_settled",
        ):
            raise ValueError("Mooncake rank settled with an invalid read disposition")


class RankWindowPeer(Protocol):
    """The trusted native GLM control peer, on an already bound transport."""

    async def fetch(
        self,
        attempt: RankWindowAttempt,
        recipe: dict,
        descriptor: dict,
        previous_consumed: int,
    ) -> WrittenRankWindow: ...

    async def commit(
        self, attempt: RankWindowAttempt, last_sequence: int
    ) -> CommittedRankWindows: ...


class ReverseRankWindowPeer(RankWindowPeer, Protocol):
    async def abort(
        self, attempt: RankWindowAttempt, pool_id: str
    ) -> AbortedRankWindows: ...

    async def settle(self, proof: FencedRankWindows) -> None: ...


class RankWindowSink(Protocol):
    """The local model's admitted destination and real native stream fence."""

    def scatter(self, window: TransferWindow, payload: torch.Tensor) -> int: ...
    def fence(self) -> None: ...
    def finish(self) -> None: ...


async def _finish_before_cancel(operation: Awaitable):
    """Retain the local lease until even a cancelled caller's work has drained."""
    task = asyncio.ensure_future(operation)
    cancellation = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as error:
            if task.cancelled():
                raise
            cancellation = error
    if cancellation is not None:
        raise cancellation
    return result


async def receive_rank_windows(
    attempt: RankWindowAttempt,
    manifest: Sequence[dict],
    pool: RegisteredWindowPool,
    peer: RankWindowPeer,
    sink: RankWindowSink,
    observe: Callable[[Literal["loaded", "abort_settled"]], None],
    *,
    timeout: float,
    reverse_peer: ReverseRankWindowPeer | None = None,
    forward_abort_peer: ReverseRankWindowPeer | None = None,
) -> None:
    """Publish only after remote writes, local reads and native COW settle."""
    if attempt.manifest_sha256 != manifest_digest(manifest):
        raise ValueError("Mooncake receive changed its local manifest")
    windows = plan_windows(manifest, window_bytes=pool.window_bytes)
    can_settle_abort = True
    uncertain: WindowLease | None = None
    abort_peer = reverse_peer or forward_abort_peer
    if (reverse_peer is not None and forward_abort_peer is not None) or (
        abort_peer is not None and abort_peer is not peer
    ):
        raise ValueError("Mooncake source settlement must use the original peer")
    source_committed = False

    async def consume(window):
        nonlocal can_settle_abort, uncertain
        lease = await pool.acquire(timeout)
        remote_fenced, local_fenced = False, True

        def scatter_and_fence():
            nonlocal local_fenced
            try:
                payload = lease.tensor[: window.nbytes]
                if sink.scatter(window, payload) != window.nbytes:
                    raise ValueError("Native GLM scatter changed the byte count")
            finally:
                sink.fence()
                local_fenced = True

        try:
            recipe, descriptor = window.to_metadata(), lease.descriptor()
            recipe_sha256 = manifest_digest(recipe)
            # Treat a missing or mismatched reply as uncertain DMA.
            reply = await asyncio.wait_for(
                peer.fetch(attempt, recipe, descriptor.copy(), window.sequence - 1),
                timeout,
            )
            if (
                type(reply) is not WrittenRankWindow
                or reply.attempt != attempt
                or reply.window_sha256 != recipe_sha256
                or reply.descriptor != descriptor
            ):
                raise ValueError("Mooncake write proof differs from its live window")
            remote_fenced = True
            local_fenced = False
            await _finish_before_cancel(asyncio.to_thread(scatter_and_fence))
        finally:
            can_reuse = remote_fenced and local_fenced
            released = False

            async def release():
                nonlocal released
                await pool.release(lease, writes_settled=can_reuse)
                released = True

            if not can_reuse and abort_peer is not None:
                uncertain = lease
            else:
                try:
                    await _finish_before_cancel(release())
                finally:
                    can_settle_abort = released and can_reuse

    async def settle_peer(outcome: Literal["acked", "abort_settled"]) -> None:
        assert abort_peer is not None
        proof = FencedRankWindows(attempt, outcome)
        try:
            await asyncio.wait_for(abort_peer.settle(proof), timeout)
        except TimeoutError:
            # The first response may have been lost after the source persisted
            # the proof. Retry the same disposition once on the fresh socket.
            await asyncio.wait_for(abort_peer.settle(proof), timeout)

    async def recover_abort() -> None:
        assert abort_peer is not None
        safe = False
        try:
            stopped = await asyncio.wait_for(
                abort_peer.abort(attempt, pool.pool_id), timeout
            )
            if type(stopped) is not AbortedRankWindows or stopped.attempt != attempt:
                raise ValueError("Mooncake abort changed its source attempt")
            await asyncio.to_thread(sink.fence)
            safe = True
            await settle_peer("abort_settled")
            observe("abort_settled")
        finally:
            if uncertain is not None:
                await pool.release(uncertain, writes_settled=safe)

    try:
        for window in windows:
            await consume(window)
        last = len(windows) - 1
        commit = await asyncio.wait_for(peer.commit(attempt, last), timeout)
        if type(commit) is not CommittedRankWindows or commit != CommittedRankWindows(
            attempt, last
        ):
            raise ValueError("Mooncake source rank did not commit all windows")
        source_committed = True
        await _finish_before_cancel(asyncio.to_thread(sink.finish))
        await _finish_before_cancel(asyncio.to_thread(sink.fence))
    except BaseException:
        if reverse_peer is not None or (
            abort_peer is not None and not source_committed
        ):
            await _finish_before_cancel(recover_abort())
        elif can_settle_abort:
            await _finish_before_cancel(asyncio.to_thread(sink.fence))
            observe("abort_settled")
        raise

    if reverse_peer is None:
        observe("loaded")
    else:

        async def publish_loaded() -> None:
            await settle_peer("acked")
            observe("loaded")

        await _finish_before_cancel(publish_loaded())


class _MooncakeTransfer(_MooncakeMemory, Protocol):
    def batch_transfer_sync_write(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> int: ...


class NativeRankWindowWriter:
    """Serve a pinned GLM model source through the existing Mooncake engine."""

    def __init__(
        self,
        attempt: RankWindowAttempt,
        manifest: Sequence[dict],
        engine: _MooncakeTransfer,
        pool: RegisteredWindowPool,
        *,
        destination: tuple[str, int, str],
        pack: Callable[[TransferWindow, torch.Tensor], int],
        release_source: Callable[[], None],
        timeout: float,
    ) -> None:
        self.attempt, self.engine, self.pool = attempt, engine, pool
        self.session = WindowSendSession(
            manifest, manifest_sha256=attempt.manifest_sha256, destination=destination
        )
        self.remote_session = f"{destination[0]}:{destination[1]}"
        self.pack, self.release_source = pack, release_source
        self.timeout = timeout
        self._released = False
        self._dma_uncertain = False
        self._abandoned = False
        self._fenced_settlement: FencedRankWindows | None = None

    async def fetch(
        self,
        attempt: RankWindowAttempt,
        recipe: dict,
        descriptor: dict,
        previous_consumed: int,
    ) -> WrittenRankWindow:
        if attempt != self.attempt:
            raise ValueError("Mooncake fetch changed the native source attempt")

        async with self.session.lock:
            if self._released or self._dma_uncertain:
                raise ValueError("Mooncake source rank is closed or unsafe")
            window, duplicate = self.session.validate_fetch(
                recipe, descriptor, previous_consumed, self.session.destination
            )
            if not duplicate:
                lease = await self.pool.acquire(self.timeout)
                settled = True

                def write() -> None:
                    nonlocal settled
                    payload = lease.tensor[: window.nbytes]
                    if self.pack(window, payload) != window.nbytes:
                        raise ValueError("Native GLM source changed the byte count")
                    settled = False
                    status = self.engine.batch_transfer_sync_write(
                        self.remote_session,
                        [payload.data_ptr()],
                        [descriptor["address"]],
                        [window.nbytes],
                    )
                    if status == 0:
                        settled = True
                    self.session.mark_copied(
                        window, descriptor, native_write_status=status
                    )

                try:
                    await _finish_before_cancel(asyncio.to_thread(write))
                finally:
                    self._dma_uncertain = not settled
                    await _finish_before_cancel(
                        self.pool.release(lease, writes_settled=settled)
                    )
            return WrittenRankWindow(
                attempt, manifest_digest(recipe), descriptor.copy()
            )

    async def commit(
        self, attempt: RankWindowAttempt, last_sequence: int
    ) -> CommittedRankWindows:
        if attempt != self.attempt:
            raise ValueError("Mooncake commit changed the native source attempt")
        async with self.session.lock:
            if self._dma_uncertain or (self._released and not self.session.committed):
                raise ValueError("Mooncake source cannot commit an unsafe attempt")
            self.session.commit(last_sequence)
            self._release()
            return CommittedRankWindows(attempt, last_sequence)

    async def abandon(self) -> None:
        """Stop further windows after all native writes have settled."""
        async with self.session.lock:
            if self._dma_uncertain or (
                self._fenced_settlement is not None
                and self._fenced_settlement.outcome == "acked"
            ):
                raise RuntimeError("Mooncake source cannot abandon its current result")
            self._abandoned = True
            self._release()

    async def fenced_settlement(
        self, attempt: RankWindowAttempt, outcome: Literal["acked", "abort_settled"]
    ) -> FencedRankWindows:
        result = FencedRankWindows(attempt, outcome)
        async with self.session.lock:
            if (
                attempt != self.attempt
                or not self._released
                or self._dma_uncertain
                or (
                    outcome == "acked"
                    and (not self.session.committed or self._abandoned)
                )
                or (outcome == "abort_settled" and not self._abandoned)
                or (
                    self._fenced_settlement is not None
                    and self._fenced_settlement != result
                )
            ):
                raise ValueError("Mooncake source has no matching fenced settlement")
            self._fenced_settlement = result
            return result

    def _release(self) -> None:
        if not self._released:
            self.release_source()
            self._released = True
