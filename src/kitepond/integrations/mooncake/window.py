# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical manifests and fixed host windows for native Mooncake transfers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
from collections import deque
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

# Derived from the GLM window-v2 contract at 19e62b54.
WINDOW_PROTOCOL = "glm53f-window-v2"
MAX_CHUNK_BYTES = 4 * 1024**2
WINDOW_BYTES = 32 * 1024**2
WINDOW_CHUNKS = 64
MAX_RECEIVE_WINDOWS = 16
MAX_MANIFEST_CHUNKS = 16384


def manifest_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class TransferWindow:
    sequence: int
    first_chunk: int
    entries: tuple[dict[str, Any], ...]
    nbytes: int

    @property
    def end_chunk(self) -> int:
        return self.first_chunk + len(self.entries)

    def to_metadata(self) -> dict[str, Any]:
        return dict(
            sequence=self.sequence,
            first_chunk=self.first_chunk,
            entries=deepcopy(list(self.entries)),
            window_bytes=self.nbytes,
        )


def plan_windows(
    manifest: Sequence[dict[str, Any]],
    *,
    window_bytes: int = WINDOW_BYTES,
    max_chunks: int = WINDOW_CHUNKS,
) -> tuple[TransferWindow, ...]:
    """Retain canonical offsets; window offsets are derived independently."""
    if (
        type(window_bytes) is not int
        or not 0 < window_bytes <= WINDOW_BYTES
        or type(max_chunks) is not int
        or not 0 < max_chunks <= WINDOW_CHUNKS
    ):
        raise ValueError("GLM window budget exceeds its qualified bounds")
    if not 0 < len(manifest) <= MAX_MANIFEST_CHUNKS:
        raise ValueError("GLM window manifest has an invalid chunk count")
    windows: list[TransferWindow] = []
    entries: list[dict[str, Any]] = []
    canonical_offset = used = first = 0
    for index, entry in enumerate(manifest):
        count = entry.get("nbytes")
        if (
            type(count) is not int
            or not 0 < count <= min(MAX_CHUNK_BYTES, window_bytes)
            or type(entry.get("offset")) is not int
            or entry["offset"] != canonical_offset
        ):
            raise ValueError("GLM window manifest is not a bounded canonical recipe")
        if entries and (used + count > window_bytes or len(entries) == max_chunks):
            windows.append(TransferWindow(len(windows), first, tuple(entries), used))
            first, entries, used = index, [], 0
        entries.append(deepcopy(entry))
        used += count
        canonical_offset += count
    windows.append(TransferWindow(len(windows), first, tuple(entries), used))
    return tuple(windows)


@dataclass
class RegisteredWindow:
    slot: int
    tensor: torch.Tensor
    generation: int = 0
    leased: bool = False
    quarantined: bool = False


@dataclass(frozen=True)
class WindowLease:
    pool_id: str
    slot: int
    generation: int
    tensor: torch.Tensor
    slot_count: int = 2

    def descriptor(self) -> dict[str, Any]:
        descriptor = dict(
            pool_id=self.pool_id,
            slot=self.slot,
            generation=self.generation,
            address=self.tensor.data_ptr(),
            capacity=self.tensor.numel(),
        )
        if self.slot_count != 2:
            descriptor["slot_count"] = self.slot_count
        return descriptor


class RegisteredWindowPool:
    """A fixed FIFO pool; uncertain peer writes permanently consume a slot."""

    def __init__(
        self,
        slots: int,
        register: Callable[[torch.Tensor], None],
        unregister: Callable[[torch.Tensor], None],
        *,
        window_bytes: int = WINDOW_BYTES,
    ):
        if (
            type(slots) is not int
            or not 1 <= slots <= MAX_RECEIVE_WINDOWS
            or type(window_bytes) is not int
            or not 0 < window_bytes <= WINDOW_BYTES
        ):
            raise ValueError("GLM registered window pool exceeds its fixed budget")
        self.pool_id = secrets.token_hex(16)
        self.window_bytes = window_bytes
        self._register, self._unregister = register, unregister
        self._lock = asyncio.Lock()
        self._waiters: deque[asyncio.Future[WindowLease]] = deque()
        self._closed = False
        self._slots: list[RegisteredWindow] = []
        try:
            for index in range(slots):
                tensor = torch.empty(window_bytes, dtype=torch.uint8, device="cpu")
                register(tensor)
                self._slots.append(RegisteredWindow(index, tensor))
        except BaseException:
            for slot in self._slots:
                unregister(slot.tensor)
            raise
        self.peak_registered_bytes = len(self._slots) * self.window_bytes

    @property
    def registered_bytes(self) -> int:
        return len(self._slots) * self.window_bytes

    @property
    def quarantined_slots(self) -> int:
        return sum(slot.quarantined for slot in self._slots)

    @property
    def leased_slots(self) -> int:
        return sum(slot.leased for slot in self._slots)

    @property
    def waiting_requests(self) -> int:
        return sum(not waiter.done() for waiter in self._waiters)

    def _grant_locked(self) -> None:
        while self._waiters:
            if self._waiters[0].done():
                self._waiters.popleft()
                continue
            available = next(
                (
                    slot
                    for slot in self._slots
                    if not slot.leased and not slot.quarantined
                ),
                None,
            )
            if available is None:
                if all(slot.quarantined for slot in self._slots):
                    while self._waiters:
                        waiter = self._waiters.popleft()
                        if not waiter.done():
                            waiter.set_exception(
                                RuntimeError(
                                    "GLM registered window pool is quarantined"
                                )
                            )
                return
            waiter = self._waiters.popleft()
            available.generation += 1
            available.leased = True
            waiter.set_result(
                WindowLease(
                    self.pool_id,
                    available.slot,
                    available.generation,
                    available.tensor,
                    len(self._slots),
                )
            )

    def _return_locked(self, lease: WindowLease, writes_settled: bool) -> None:
        if lease.pool_id != self.pool_id or not 0 <= lease.slot < len(self._slots):
            raise ValueError("GLM window lease belongs to another pool")
        slot = self._slots[lease.slot]
        if (
            not slot.leased
            or slot.generation != lease.generation
            or slot.tensor is not lease.tensor
        ):
            raise ValueError("GLM window lease is stale or already released")
        slot.leased = False
        slot.quarantined = not writes_settled

    async def acquire(self, timeout: float) -> WindowLease:
        async with self._lock:
            if self._closed:
                raise RuntimeError("GLM registered window pool is closed")
            waiter: asyncio.Future[WindowLease] = (
                asyncio.get_running_loop().create_future()
            )
            self._waiters.append(waiter)
            self._grant_locked()
        try:
            return await asyncio.wait_for(asyncio.shield(waiter), timeout)
        except BaseException:
            async with self._lock:
                if waiter.done() and not waiter.cancelled():
                    if waiter.exception() is None:
                        self._return_locked(waiter.result(), writes_settled=True)
                else:
                    waiter.cancel()
                    self._waiters.remove(waiter)
                self._grant_locked()
            raise

    async def release(self, lease: WindowLease, *, writes_settled: bool) -> None:
        if type(writes_settled) is not bool:
            raise ValueError("GLM window release needs an explicit write disposition")
        async with self._lock:
            self._return_locked(lease, writes_settled)
            self._grant_locked()

    def close(self) -> None:
        if self.leased_slots or self.quarantined_slots or self.waiting_requests:
            raise RuntimeError("GLM window pool still owns live or uncertain writes")
        self._closed = True
        while self._slots:
            self._unregister(self._slots[-1].tensor)
            self._slots.pop()


class WindowSendSession:
    """Validate byte windows and consumption before a separate native import fence."""

    def __init__(
        self,
        manifest: Sequence[dict[str, Any]],
        *,
        manifest_sha256: str,
        destination: tuple[str, int, str],
    ):
        if manifest_digest(manifest) != manifest_sha256:
            raise ValueError("GLM window session manifest digest mismatch")
        self.windows = plan_windows(manifest)
        self.manifest_sha256 = manifest_sha256
        self.destination = destination
        self.next_sequence = 0
        self.consumed_sequence = -1
        self.last_descriptor: dict[str, Any] | None = None
        self.committed = False
        self._slot_generations: dict[int, int] = {}
        self._slot_addresses: dict[int, int] = {}
        self._destination_slot_count: int | None = None
        self.lock = asyncio.Lock()

    def validate_fetch(
        self,
        recipe: dict[str, Any],
        descriptor: dict[str, Any],
        previous_consumed: int,
        destination: tuple[str, int, str],
    ) -> tuple[TransferWindow, bool]:
        if self.committed or destination != self.destination:
            raise ValueError("GLM window request changed its peer or completed epoch")
        sequence = recipe.get("sequence")
        if type(sequence) is not int or not 0 <= sequence < len(self.windows):
            raise ValueError("GLM window sequence is invalid")
        expected = self.windows[sequence]
        if recipe != expected.to_metadata():
            raise ValueError("GLM window chunk recipe differs from canonical manifest")
        slot_count = descriptor.get("slot_count", 2)
        legacy_fields = {"pool_id", "slot", "generation", "address", "capacity"}
        if (
            set(descriptor) not in (legacy_fields, legacy_fields | {"slot_count"})
            or type(slot_count) is not int
            or not 2 <= slot_count <= MAX_RECEIVE_WINDOWS
            or (
                self._destination_slot_count is not None
                and slot_count != self._destination_slot_count
            )
            or descriptor["pool_id"] != destination[2]
            or any(
                type(descriptor[key]) is not int
                for key in ("slot", "generation", "address", "capacity")
            )
            or not 0 <= descriptor["slot"] < slot_count
            or descriptor["generation"] <= 0
            or descriptor["address"] <= 0
            or descriptor["capacity"] != WINDOW_BYTES
            or expected.nbytes > descriptor["capacity"]
        ):
            raise ValueError("GLM destination window descriptor is invalid")
        if type(previous_consumed) is not int or previous_consumed != sequence - 1:
            raise ValueError("GLM window lacks its previous consumption receipt")
        if sequence == self.next_sequence - 1:
            if sequence <= self.consumed_sequence or descriptor != self.last_descriptor:
                raise ValueError("GLM cannot replay a consumed or changed window")
            return expected, True
        if sequence != self.next_sequence:
            raise ValueError("GLM window skipped or reordered its canonical chunks")
        if descriptor["generation"] <= self._slot_generations.get(
            descriptor["slot"], 0
        ):
            raise ValueError("GLM window reused a stale slot generation")
        if (
            self._slot_addresses.get(descriptor["slot"], descriptor["address"])
            != descriptor["address"]
        ):
            raise ValueError("GLM registered slot changed its fixed address")
        self._destination_slot_count = slot_count
        self.consumed_sequence = previous_consumed
        return expected, False

    def mark_copied(
        self,
        window: TransferWindow,
        descriptor: dict[str, Any],
        *,
        native_write_status: int,
    ) -> None:
        """Record the status returned by Mooncake's synchronous byte write."""
        if type(native_write_status) is not int or native_write_status != 0:
            raise ValueError("Mooncake did not settle the native byte write")
        expected, replay = self.validate_fetch(
            window.to_metadata(), descriptor, window.sequence - 1, self.destination
        )
        if replay or window != expected:
            raise ValueError("GLM window copy completed out of order")
        self.last_descriptor = dict(descriptor)
        self._slot_generations[descriptor["slot"]] = descriptor["generation"]
        self._slot_addresses[descriptor["slot"]] = descriptor["address"]
        self.next_sequence += 1

    def commit(self, consumed_sequence: int) -> bool:
        """Record peer consumption of all bytes, without asserting device readiness."""
        if (
            type(consumed_sequence) is not int
            or consumed_sequence != len(self.windows) - 1
            or self.next_sequence != len(self.windows)
        ):
            raise ValueError("GLM rank cannot commit before every window is consumed")
        first = not self.committed
        self.consumed_sequence = consumed_sequence
        self.committed = True
        return first
