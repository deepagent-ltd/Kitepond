# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the GLM P READY receipt after the actual host and draft write fences."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from kitepond.core.readiness import ReadyAttempt, ReadyReceipt

from .shared_region import NativeHiSparseRegionDescriptor


class _PrefillStep(Protocol):
    @property
    def epoch(self) -> int: ...


class _PrefillCheckpoint(Protocol):
    def wait_ready(self, request_id: str) -> _PrefillStep: ...

    def require_completed_boundary(self, request_id: str) -> int: ...


@dataclass(frozen=True)
class PrefillWriteGrant:
    attempt: ReadyAttempt
    region: NativeHiSparseRegionDescriptor
    tokens: int

    def __post_init__(self) -> None:
        if type(self.tokens) is not int or self.tokens <= 0:
            raise ValueError("Prefill write grant needs a positive token boundary")
        self.region.validate()


def complete_prefill_write(
    grant: PrefillWriteGrant,
    *,
    local_rank: int,
    current_grant: Callable[[str], PrefillWriteGrant | None],
    wait_host_store: Callable[[str], None],
    checkpoint: _PrefillCheckpoint,
) -> ReadyReceipt:
    """Wait for both native writes while the producer still holds its grant.

    The native producer owns the grant journal and records this receipt under
    the same journal lock. Both fence callbacks must raise if writes cannot be
    drained. The connector can then send the recorded receipt to the scheduler.
    """
    request_id = grant.attempt.request_id
    if local_rank != grant.region.source_rank or current_grant(request_id) is not grant:
        raise ValueError("GLM prefiller has no matching live rank grant")

    wait_host_store(request_id)
    step = checkpoint.wait_ready(request_id)
    boundary = checkpoint.require_completed_boundary(request_id)
    if (
        step.epoch != grant.attempt.epoch
        or boundary != grant.tokens
        or current_grant(request_id) is not grant
    ):
        raise ValueError("Settled GLM prefill does not match the admitted lease")

    return ReadyReceipt(
        attempt=grant.attempt,
        rank=local_rank,
        region_sha256=grant.region.fingerprint(),
        target_valid_tokens=boundary,
        draft_valid_tokens=boundary,
        completed_monotonic_ns=time.monotonic_ns(),
    )
