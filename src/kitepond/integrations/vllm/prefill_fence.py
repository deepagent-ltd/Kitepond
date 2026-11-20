# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fence native HiSparse rows in the deferred connector save callback."""

from collections.abc import Callable
from dataclasses import dataclass

from kitepond.core.readiness import ReadyReceipt
from kitepond.models.glm5next.ready import PrefillWriteGrant, complete_prefill_write

from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
    HiSparseConnectorWorker,
)

from .ready_transport import ReadyPublisher, ReadyTicket


@dataclass(frozen=True)
class PrefillWrite:
    ticket: ReadyTicket
    grant: PrefillWriteGrant


class NativePrefillSaveFence:
    """Called by a P connector only after its drafter finished the step."""

    def __init__(
        self,
        worker: HiSparseConnectorWorker,
        publisher: ReadyPublisher,
        current_grant: Callable[[str], PrefillWriteGrant | None],
        checkpoint,
        *,
        local_rank: int,
        record_source: Callable[[tuple[tuple[PrefillWrite, ReadyReceipt], ...]], None]
        | None = None,
    ) -> None:
        if not isinstance(worker, HiSparseConnectorWorker):
            raise TypeError("GLM prefill needs the actual HiSparse transfer worker")
        self._worker = worker
        self._publisher = publisher
        self._current_grant = current_grant
        self._checkpoint = checkpoint
        self._rank = local_rank
        self._record_source = record_source

    def publish(
        self, metadata: HiSparseConnectorMetadata, writes: tuple[PrefillWrite, ...]
    ) -> int:
        if not writes:
            return 0
        worker = self._worker
        if not worker.is_host_writer or not isinstance(
            metadata, HiSparseConnectorMetadata
        ):
            raise ValueError("GLM prefill requires native rank-local host writes")
        if not worker._draft_mirror_pending:
            raise ValueError("Native HiSparse has no draft rows from this step")
        if any(
            item.ticket.attempt != item.grant.attempt
            or item.grant.region.source_rank != self._rank
            or self._current_grant(item.grant.attempt.request_id) is not item.grant
            or not metadata.row_mirrors.get(item.grant.attempt.request_id)
            for item in writes
        ):
            raise ValueError("Prefill write proof differs from admitted native rows")

        def collect():
            # Stock performs this work before the next forward. The explicit
            # connector can run it now since the native draft already returned.
            worker._finish_previous_step()
            if not worker._dma_submitted or worker._draft_mirror_pending:
                raise ValueError("Native HiSparse did not submit this draft mirror")
            worker.host_write_event.synchronize()
            records = tuple(
                (
                    item,
                    complete_prefill_write(
                        item.grant,
                        local_rank=self._rank,
                        current_grant=self._current_grant,
                        # Both target and draft DMAs were fenced as a batch.
                        wait_host_store=_already_fenced,
                        checkpoint=self._checkpoint,
                    ),
                )
                for item in writes
            )
            if self._record_source is not None:
                self._record_source(records)
            return tuple((item.ticket, receipt) for item, receipt in records)

        return self._publisher.after_batch_save_fence(collect)


def _already_fenced(request_id: str) -> None:
    pass
