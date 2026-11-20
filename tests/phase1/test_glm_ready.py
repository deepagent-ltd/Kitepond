# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
from kitepond.core.readiness import ReadyAttempt
from kitepond.models.glm5next.ready import PrefillWriteGrant, complete_prefill_write
from kitepond.models.glm5next.shared_region import NativeHiSparseRegionDescriptor


@pytest.fixture
def grant():
    attempt = ReadyAttempt("request", "transfer", 4, "a" * 64)
    region = NativeHiSparseRegionDescriptor(
        "glm53f-hisparse-authority-region-v1",
        "decode",
        "boot",
        1,
        10,
        16,
        0,
        32,
        4096,
        "b" * 64,
    )
    return PrefillWriteGrant(attempt, region, 128)


class Checkpoint:
    def __init__(self, trace, *, epoch=4, boundary=128):
        self.trace = trace
        self.epoch = epoch
        self.boundary = boundary

    def wait_ready(self, request_id):
        self.trace.append(("draft_fence", request_id))
        return SimpleNamespace(epoch=self.epoch)

    def require_completed_boundary(self, request_id):
        self.trace.append(("boundary", request_id))
        return self.boundary


def test_settled_receipt_follows_both_writes(grant):
    trace = []
    receipt = complete_prefill_write(
        grant,
        local_rank=1,
        current_grant=lambda key: grant,
        wait_host_store=lambda key: trace.append(("host_fence", key)),
        checkpoint=Checkpoint(trace),
    )
    assert trace == [
        (step, "request") for step in ("host_fence", "draft_fence", "boundary")
    ]
    assert receipt.attempt == grant.attempt
    assert receipt.rank == 1
    assert receipt.region_sha256 == grant.region.fingerprint()
    assert receipt.target_valid_tokens == receipt.draft_valid_tokens == 128


@pytest.mark.parametrize("wrong", ["rank", "stale", "epoch", "boundary", "reallocated"])
def test_stale_or_unsettled_write_cannot_publish(grant, wrong):
    trace = []
    active = iter(
        [
            replace(grant) if wrong == "stale" else grant,
            replace(grant) if wrong == "reallocated" else grant,
        ]
    )
    checkpoint = Checkpoint(
        trace,
        epoch=5 if wrong == "epoch" else 4,
        boundary=127 if wrong == "boundary" else 128,
    )
    with pytest.raises(ValueError, match="live rank grant|admitted lease"):
        complete_prefill_write(
            grant,
            local_rank=2 if wrong == "rank" else 1,
            current_grant=lambda key: next(active),
            wait_host_store=lambda key: trace.append(("host_fence", key)),
            checkpoint=checkpoint,
        )
    if wrong in {"rank", "stale"}:
        assert trace == []


def test_host_fence_failure_does_not_reach_draft(grant):
    trace = []

    def wait_host_store(key):
        raise OSError("host copy failed")

    with pytest.raises(OSError, match="host copy failed"):
        complete_prefill_write(
            grant,
            local_rank=1,
            current_grant=lambda key: grant,
            wait_host_store=wait_host_store,
            checkpoint=Checkpoint(trace),
        )
    assert trace == []
