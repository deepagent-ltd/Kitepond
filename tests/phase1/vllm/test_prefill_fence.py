# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest
from kitepond.core.readiness import ReadyAttempt
from kitepond.integrations.vllm.prefill_fence import (
    NativePrefillSaveFence,
    PrefillWrite,
)
from kitepond.integrations.vllm.ready_transport import (
    ReadyCollector,
    ReadyPublisher,
    ReadyWorker,
)
from kitepond.models.glm5next.ready import PrefillWriteGrant
from kitepond.models.glm5next.shared_region import NativeHiSparseRegionDescriptor

from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
    HiSparseConnectorWorker,
)


@pytest.mark.parametrize("issued", [True, False])
def test_stock_draft_mirror_must_be_submitted_and_fenced_before_ready(tmp_path, issued):
    traces = []
    worker = HiSparseConnectorWorker.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = True
    worker._draft_mirror_pending = True
    worker._draft_layers = (3,)
    worker._post_forward_transfers = []
    worker._dma_submitted = False

    def enqueue(layers):
        traces.append(("dma", layers))
        worker._dma_submitted = issued

    worker._enqueue_row_dma = enqueue
    worker._submit_transfers = lambda transfers: traces.append(("submit", transfers))
    worker.host_write_event = SimpleNamespace(
        synchronize=lambda: traces.append(("host_fence",))
    )
    checkpoint = SimpleNamespace(
        wait_ready=lambda request: (
            traces.append(("draft_fence", request)) or SimpleNamespace(epoch=4)
        ),
        require_completed_boundary=lambda _: 128,
    )
    identity = ReadyAttempt("request", "transfer", 4, "a" * 64)
    region = NativeHiSparseRegionDescriptor(
        "glm53f-hisparse-authority-region-v1",
        "D",
        "epoch",
        0,
        20,
        100,
        0,
        19,
        4096,
        "b" * 64,
    )
    grant = PrefillWriteGrant(identity, region, 128)
    directory = tmp_path / "ready"
    directory.mkdir(mode=0o700)
    collector = ReadyCollector(directory, {0: ReadyWorker.from_pid(os.getpid())})
    publisher = ReadyPublisher(directory)
    try:
        ticket = collector.arm(identity, {0: region.fingerprint()}, 128)
        wrapper = NativePrefillSaveFence(
            worker, publisher, lambda _: grant, checkpoint, local_rank=0
        )
        meta = HiSparseConnectorMetadata(
            None, (), (), {"request": (object(),)}, True, {}
        )
        writes = (PrefillWrite(ticket, grant),)
        if not issued:
            with pytest.raises(ValueError, match="did not submit"):
                wrapper.publish(meta, writes)
            assert collector.take(ticket) is None
            assert traces == [("dma", (3,)), ("submit", [])]
        else:
            assert wrapper.publish(meta, writes) == 1
            (receipt,) = collector.take(ticket)
            assert receipt.target_valid_tokens == receipt.draft_valid_tokens == 128
            assert traces == [
                ("dma", (3,)),
                ("submit", []),
                ("host_fence",),
                ("draft_fence", "request"),
            ]
    finally:
        publisher.close()
        collector.close()
