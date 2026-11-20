# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Own opt-in reverse imports on the existing native GLM prefill worker."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

from kitepond.models.glm5next.layout import Glm5NextPDLayout

from .glm_return_bootstrap import GlmPrefillTerminalDestination
from .glm_terminal_return import GlmTerminalReturn
from .prefill_return_receive import PrefillReturnFence

if TYPE_CHECKING:
    import torch

    from .glm_k0_routes import GlmK0Route
    from .glm_native_views import GlmNativeStateViews


@dataclass
class _WarmImport:
    transfer: GlmTerminalReturn
    future: Future | None = None
    fence: PrefillReturnFence | None = None


class GlmPrefillWarmImports:
    """Keep one shared receive owner and bound rank proofs for K0 continuation."""

    def __init__(
        self,
        views: GlmNativeStateViews,
        stream: torch.cuda.Stream | None,
        *,
        capacity: int,
    ) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Native warm P imports need a bounded capacity")
        self._views, self._stream = views, stream
        self._capacity = capacity
        self._destination = GlmPrefillTerminalDestination()
        self._reads: dict[str, _WarmImport] = {}
        self._lock = Lock()

    def submit(self, transfer: GlmTerminalReturn, timeout: float) -> None:
        if type(transfer) is not GlmTerminalReturn:
            raise ValueError("Native warm P needs the exact terminal transfer")
        req_id = transfer.destination.request_id
        with self._lock:
            previous = self._reads.get(req_id)
            if previous is not None:
                if previous.transfer != transfer:
                    raise ValueError("Native warm P request already has another permit")
                return
            if len(self._reads) >= self._capacity:
                raise BufferError("Native warm P receive capacity is exhausted")
            lease = transfer.lease
            identity = lease.identity
            boundary = lease.requested_common_tokens
            groups = {
                group.group_id: group.block_ids for group in transfer.destination.groups
            }
            if (
                len(groups) != len(transfer.destination.groups)
                or transfer.permit.lease_sha256 != lease.sha256
                or transfer.permit.manifest_sha256 != transfer.manifest.sha256
            ):
                raise ValueError("Native warm P permit names another allocation")
            layout = Glm5NextPDLayout(
                req_id,
                identity.model_fingerprint,
                identity.config_fingerprint,
                0,
                identity.epoch,
                boundary,
                boundary,
                0,
                "decode_ready",
            )
            cache, layer_groups = self._views.for_terminal_return(layout, groups)
            read = _WarmImport(transfer)

            def observe(attempt, receipt) -> None:
                fence = PrefillReturnFence(attempt, receipt)
                if attempt != transfer.attempt(self._views.rank):
                    raise ValueError("Native warm P receipt changed its rank or permit")
                with self._lock:
                    if self._reads.get(req_id) is not read or (
                        read.fence is not None and read.fence != fence
                    ):
                        raise ValueError("Native warm P receipt has no active owner")
                    read.fence = fence

            read.future = self._destination.receive(
                transfer,
                self._views,
                cache,
                layer_groups,
                self._stream,
                observe,
                timeout=timeout,
            )
            self._reads[req_id] = read

    def result(self, transfer: GlmTerminalReturn) -> PrefillReturnFence | None:
        with self._lock:
            read = self._reads.get(transfer.destination.request_id)
            if read is None or read.transfer != transfer:
                raise ValueError("Native warm P transfer is not registered")
            return read.fence

    def require_loaded(self, route: GlmK0Route) -> None:
        with self._lock:
            read = self._reads.get(route.grant.request_id)
            if (
                read is None
                or read.transfer.destination.grant != route.grant
                or read.transfer.lease.requested_common_tokens != route.imported_tokens
                or read.fence is None
                or read.fence.receipt.status != "acked"
            ):
                raise ValueError("Native P K0 has no verified local warm prefix")

    def retire(self, transfer: GlmTerminalReturn) -> None:
        request_id = transfer.destination.request_id
        with self._lock:
            read = self._reads.get(request_id)
            if read is None:
                return
            if (
                read.transfer != transfer
                or read.fence is None
                or read.future is None
                or not read.future.done()
            ):
                raise ValueError("Native warm P transfer is not fenced and settled")
            self._reads.pop(request_id)

    def close(self) -> None:
        with self._lock:
            if self._reads:
                raise RuntimeError("Native warm P worker retains active import owners")
        self._destination.close()
