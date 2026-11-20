# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded foreground metadata interests over opaque model-owned handles."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from kitepond.core.contracts import Wait


@dataclass(eq=False)
class ForegroundInterest:
    issuer: ForegroundInterestTable
    token: str
    nonce: str
    handle: object
    deadline_ns: int


class ForegroundInterestTable:
    """Retain metadata only; expiration never settles model reads or I/O."""

    def __init__(self, *, limit: int, lease_ns: int):
        if (
            type(limit) is not int
            or not 1 <= limit <= 4096
            or type(lease_ns) is not int
            or lease_ns <= 0
        ):
            raise ValueError("Foreground interests need a finite count and lease")
        self.limit, self.lease_ns = limit, lease_ns
        self._last_ns = 0
        self._tokens: dict[str, ForegroundInterest] = {}
        self._nonces: dict[str, str] = {}

    def _clock(self, now_ns: int) -> None:
        if type(now_ns) is not int or now_ns < self._last_ns:
            raise ValueError("Foreground lease clock must be monotonic")
        self._last_ns = now_ns

    @staticmethod
    def _nonce(value: str) -> None:
        if type(value) is not str or not 1 <= len(value) <= 256:
            raise ValueError("Foreground interest needs a bounded client nonce")

    def expire(self, *, now_ns: int) -> int:
        self._clock(now_ns)
        expired = [r for r in self._tokens.values() if r.deadline_ns <= now_ns]
        for record in expired:
            self._tokens.pop(record.token)
            self._nonces.pop(record.nonce)
        return len(expired)

    def begin(
        self, handle: object, nonce: str, *, now_ns: int
    ) -> ForegroundInterest | Wait:
        self._nonce(nonce)
        if handle is None:
            raise ValueError("Foreground handle must be model-owned")
        self.expire(now_ns=now_ns)
        previous = self._nonces.get(nonce)
        if previous is not None:
            record = self._tokens[previous]
            if record.handle is not handle:
                raise ValueError("Foreground nonce changed its model handle")
            record.deadline_ns = now_ns + self.lease_ns
            return record
        if len(self._tokens) >= self.limit:
            return Wait("FOREGROUND_LOOKUP_ADMISSION_WAIT")
        token = secrets.token_hex(32)
        record = ForegroundInterest(self, token, nonce, handle, now_ns + self.lease_ns)
        self._tokens[token], self._nonces[nonce] = record, token
        return record

    def renew(self, token: str, *, now_ns: int) -> ForegroundInterest | None:
        self.expire(now_ns=now_ns)
        record = self._tokens.get(token)
        if record is not None:
            record.deadline_ns = now_ns + self.lease_ns
        return record

    def release(self, nonce: str, *, now_ns: int) -> bool:
        self._nonce(nonce)
        self.expire(now_ns=now_ns)
        token = self._nonces.pop(nonce, None)
        if token is None:
            return False
        self._tokens.pop(token)
        return True

    def active(self, handle: object, *, now_ns: int) -> int:
        self.expire(now_ns=now_ns)
        return sum(r.handle is handle for r in self._tokens.values())

    def count(self, *, now_ns: int) -> int:
        self.expire(now_ns=now_ns)
        return len(self._tokens)
