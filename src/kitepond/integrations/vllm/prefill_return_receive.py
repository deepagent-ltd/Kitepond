# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Carry fenced native P return reads in existing HiSparse worker metadata."""

from collections.abc import Mapping
from dataclasses import dataclass

from kitepond.integrations.mooncake.live_receive import RankWindowAttempt

from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnectorWorkerMetadata,
)

from .checkpoint import CheckpointReadReceipt

PREFILL_RETURN_BOOT_CONFIG = "kitepond_glm_prefill_return_boot_id"


def prefill_return_boot_id(additional) -> str | None:
    if (
        not isinstance(additional, Mapping)
        or PREFILL_RETURN_BOOT_CONFIG not in additional
    ):
        return None
    boot = additional[PREFILL_RETURN_BOOT_CONFIG]
    if type(boot) is not str or not 0 < len(boot) <= 256 or boot != boot.strip():
        raise ValueError("Native P return needs its current explicit boot identity")
    return boot


@dataclass(frozen=True)
class PrefillReturnFence:
    attempt: RankWindowAttempt
    receipt: CheckpointReadReceipt

    def __post_init__(self) -> None:
        if (
            type(self.attempt) is not RankWindowAttempt
            or type(self.receipt) is not CheckpointReadReceipt
            or self.receipt.local_rank != self.attempt.rank
            or self.receipt.lease_sha256 != self.attempt.grant_sha256
        ):
            raise ValueError("Native P return fence differs from its rank or permit")


@dataclass
class PrefillReturnProofs(HiSparseConnectorWorkerMetadata):
    fences: tuple[PrefillReturnFence, ...] = ()

    def aggregate(self, other):
        base = super().aggregate(other)
        incoming = other.fences if isinstance(other, PrefillReturnProofs) else ()
        return PrefillReturnProofs(
            base.enqueued_transfer_counts,
            base.completed_transfer_counts,
            base.completed_host_copy_dst_ids,
            self.fences + incoming,
        )
