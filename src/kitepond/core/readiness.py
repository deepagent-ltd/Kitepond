# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Identity and settled host-write receipt for P/D handoff."""

from dataclasses import dataclass


def _uint(value: int) -> bool:
    return type(value) is int and value >= 0


def _digest(value: str, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


@dataclass(frozen=True)
class ReadyAttempt:
    request_id: str
    transfer_id: str
    epoch: int
    lease_sha256: str

    def __post_init__(self) -> None:
        if (
            any(
                not isinstance(value, str) or not 0 < len(value) <= 128
                for value in (self.request_id, self.transfer_id)
            )
            or not _uint(self.epoch)
            or not _digest(self.lease_sha256)
        ):
            raise ValueError("Invalid READY attempt identity")


@dataclass(frozen=True)
class ReadyReceipt:
    """The native adapter has verified success and drained both IO directions."""

    attempt: ReadyAttempt
    rank: int
    region_sha256: str
    target_valid_tokens: int
    draft_valid_tokens: int
    completed_monotonic_ns: int

    def __post_init__(self) -> None:
        if (
            not _uint(self.rank)
            or not _digest(self.region_sha256)
            or not _uint(self.target_valid_tokens)
            or not _uint(self.draft_valid_tokens)
            or self.draft_valid_tokens > self.target_valid_tokens
            or not _uint(self.completed_monotonic_ns)
            or not self.completed_monotonic_ns
        ):
            raise ValueError("Invalid settled READY receipt")
