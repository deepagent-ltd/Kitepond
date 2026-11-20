# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deliver an optional post-fence P receipt between trusted local workers."""

from __future__ import annotations

import json
import os
import secrets
import socket
import stat
import struct
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kitepond.core.readiness import ReadyAttempt, ReadyReceipt, _digest, _uint

_PACKET_LIMIT = 192 * 1024
_CREDENTIALS = struct.Struct("3i")
_MAX_PENDING = 128
_TTL_SECONDS = 30


@dataclass(frozen=True)
class ReadyTicket:
    attempt: ReadyAttempt
    nonce: str


def _start_ticks(pid: int) -> int | None:
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat_line[stat_line.rindex(")") + 2 :].split()
        if fields[0] not in {"X", "Z"}:
            return int(fields[19])
    except (IndexError, OSError, ValueError):
        pass
    return None


@dataclass(frozen=True)
class ReadyWorker:
    """A rank's PID and process birth time supplied by its trusted launcher."""

    pid: int
    start_ticks: int

    @classmethod
    def from_pid(cls, pid: int) -> ReadyWorker:
        if not _uint(pid) or not pid:
            raise ValueError("Invalid READY worker PID")
        started = _start_ticks(pid)
        if started is None:
            raise ValueError("READY worker process is absent")
        return cls(pid, started)

    def __post_init__(self) -> None:
        if not _uint(self.pid) or not self.pid or not _uint(self.start_ticks):
            raise ValueError("Invalid READY worker identity")

    def is_live(self) -> bool:
        return _start_ticks(self.pid) == self.start_ticks


def _endpoint(directory: Path) -> Path:
    info = directory.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("READY socket needs a private owner-only directory")
    return directory / "p-ready.sock"


class ReadyPublisher:
    """Publish after the connector save fence without blocking on delivery."""

    def __init__(self, directory: Path):
        self._path = _endpoint(directory)
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._socket.setblocking(False)

    def after_save_fence(
        self,
        ticket: ReadyTicket,
        save_and_verify: Callable[[], ReadyReceipt | None],
    ) -> bool:
        def save_batch():
            receipt = save_and_verify()
            return () if receipt is None else ((ticket, receipt),)

        return bool(self.after_batch_save_fence(save_batch))

    def after_batch_save_fence(
        self,
        save_and_verify: Callable[[], tuple[tuple[ReadyTicket, ReadyReceipt], ...]],
    ) -> int:
        # Run the real adapter fence before writing one datagram for this rank.
        completed = save_and_verify()
        if not 0 < len(completed) <= _MAX_PENDING:
            return 0
        records = [
            {"nonce": ticket.nonce, "receipt": asdict(receipt)}
            for ticket, receipt in completed
            if receipt.attempt == ticket.attempt
        ]
        if not records:
            return 0
        encoded = json.dumps(
            {"v": 1, "records": records}, separators=(",", ":")
        ).encode()
        if len(encoded) > _PACKET_LIMIT:
            return 0
        try:
            self._socket.sendto(encoded, str(self._path))
        except OSError:
            return 0
        return len(records)

    def close(self) -> None:
        self._socket.close()


@dataclass
class _LiveAttempt:
    ticket: ReadyTicket
    regions: dict[int, str]
    tokens: int
    opened_ns: int
    deadline_ns: int
    receipts: dict[int, ReadyReceipt] = field(default_factory=dict)


class ReadyCollector:
    """Collect single-use receipts on the owning scheduler thread."""

    def __init__(
        self,
        directory: Path,
        workers: Mapping[int, ReadyWorker],
        *,
        max_pending: int = _MAX_PENDING,
        ttl_seconds: float = _TTL_SECONDS,
    ):
        if (
            not workers
            or any(
                not _uint(rank) or not peer.is_live() for rank, peer in workers.items()
            )
            or len({peer.pid for peer in workers.values()}) != len(workers)
            or not 0 < max_pending <= _MAX_PENDING
            or not 0 < ttl_seconds <= _TTL_SECONDS
        ):
            raise ValueError("Invalid READY receiver limits or worker enrollment")
        self._workers = dict(workers)
        self._peer_pids = {peer.pid for peer in self._workers.values()}
        self._max_pending = max_pending
        self._ttl_ns = int(ttl_seconds * 1_000_000_000)
        self._pending: dict[str, _LiveAttempt] = {}
        self._path = _endpoint(directory)
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            self._socket.setblocking(False)
            self._socket.bind(str(self._path))
        except OSError:
            self._socket.close()
            raise
        self._inode = self._path.lstat().st_ino
        self._closed = False

    def arm(
        self, attempt: ReadyAttempt, region_by_rank: Mapping[int, str], tokens: int
    ) -> ReadyTicket:
        if self._closed:
            raise RuntimeError("READY receiver is closed")
        if (
            region_by_rank.keys() != self._workers.keys()
            or any(not _digest(region) for region in region_by_rank.values())
            or not _uint(tokens)
            or not tokens
        ):
            raise ValueError("READY receipt requires the admitted regions and tokens")
        now = time.monotonic_ns()
        for nonce, pending in tuple(self._pending.items()):
            if pending.deadline_ns <= now:
                self._pending.pop(nonce)
            elif (
                pending.ticket.attempt.request_id == attempt.request_id
                and pending.ticket.attempt.transfer_id == attempt.transfer_id
            ):
                raise ValueError("Cancel the current READY attempt before replacing it")
        if len(self._pending) >= self._max_pending:
            raise BufferError("READY receiver is at capacity")
        ticket = ReadyTicket(attempt, secrets.token_hex(16))
        self._pending[ticket.nonce] = _LiveAttempt(
            ticket, dict(region_by_rank), tokens, now, now + self._ttl_ns
        )
        return ticket

    def _live(self, ticket: ReadyTicket) -> _LiveAttempt | None:
        pending = self._pending.get(ticket.nonce)
        return pending if pending is not None and pending.ticket == ticket else None

    def cancel(self, ticket: ReadyTicket) -> None:
        if self._live(ticket) is not None:
            self._pending.pop(ticket.nonce)

    def take(self, ticket: ReadyTicket) -> tuple[ReadyReceipt, ...] | None:
        pending = self._live(ticket)
        if pending is None:
            return None
        self._drain()
        if self._live(ticket) is None:
            return None
        if pending.deadline_ns <= time.monotonic_ns() or any(
            not worker.is_live() for worker in self._workers.values()
        ):
            self.cancel(ticket)
            return None
        if pending.regions.keys() != pending.receipts.keys():
            return None
        self.cancel(ticket)
        return tuple(pending.receipts[rank] for rank in sorted(pending.regions))

    def _drain(self) -> None:
        for _ in range(_MAX_PENDING):
            try:
                data, ancillary, flags, _ = self._socket.recvmsg(
                    _PACKET_LIMIT, socket.CMSG_SPACE(_CREDENTIALS.size)
                )
            except BlockingIOError:
                return
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                continue
            for level, kind, credentials in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS:
                    pid, uid, _ = _CREDENTIALS.unpack(credentials)
                    if uid == os.getuid() and pid in self._peer_pids:
                        self._record(data, pid)

    def _record(self, data: bytes, pid: int) -> None:
        try:
            envelope = json.loads(data)
            version = envelope["v"]
            records = envelope["records"]
        except (KeyError, TypeError, ValueError, RecursionError):
            return
        if (
            type(version) is not int
            or version != 1
            or not isinstance(records, list)
            or not 0 < len(records) <= _MAX_PENDING
        ):
            return
        for record in records:
            self._accept(record, pid)

    def _accept(self, record: dict, pid: int) -> None:
        try:
            nonce = record["nonce"]
            if not _digest(nonce, 32):
                return
            pending = self._pending.get(nonce)
            if pending is None:
                return
            row = record["receipt"]
            attempt = ReadyAttempt(**row.pop("attempt"))
            receipt = ReadyReceipt(attempt, **row)
        except (AttributeError, KeyError, TypeError, ValueError):
            return
        peer = self._workers.get(receipt.rank)
        now = time.monotonic_ns()
        if (
            peer is None
            or pid != peer.pid
            or not peer.is_live()
            or receipt.attempt != pending.ticket.attempt
            or receipt.region_sha256 != pending.regions[receipt.rank]
            or receipt.target_valid_tokens != pending.tokens
            or receipt.draft_valid_tokens != pending.tokens
            or not pending.opened_ns <= receipt.completed_monotonic_ns <= now
            or now >= pending.deadline_ns
        ):
            return
        previous = pending.receipts.setdefault(receipt.rank, receipt)
        if previous != receipt:
            self.cancel(pending.ticket)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending.clear()
        self._socket.close()
        try:
            if self._path.lstat().st_ino == self._inode:
                self._path.unlink()
        except FileNotFoundError:
            pass
