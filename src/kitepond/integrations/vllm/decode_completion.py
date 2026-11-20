# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route D receive completions through the active HiSparse grant."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorWorkerMetadata,
)
from vllm.v1.outputs import KVConnectorOutput

from .decode_receive import DecodeImportRankReceipt, DecodeReceiveProofs
from .hisparse_attachment import DecodeImportBinding, HiSparseConnector
from .terminal_checkpoint import StockTerminalCheckpointScheduler

if TYPE_CHECKING:
    from vllm.v1.request import Request

_RANKS = frozenset(range(4))


DecodeImportReceipts = Callable[
    [DecodeImportBinding], tuple[DecodeImportRankReceipt, ...] | None
]


def _settled_failure(
    binding: DecodeImportBinding, receipts: tuple[DecodeImportRankReceipt, ...]
) -> bool | None:
    grant, destination = binding.grant.sha256, binding.destination_sha256
    ranks = {}
    for receipt in receipts:
        if type(receipt) is not DecodeImportRankReceipt:
            raise ValueError("D import produced an invalid rank receipt")
        if (receipt.grant_sha256, receipt.destination_sha256) != (grant, destination):
            continue
        if ranks.setdefault(receipt.rank, receipt) != receipt:
            raise ValueError("Conflicting receipts for the current D import")
    if ranks.keys() != _RANKS:
        return None
    return any(receipt.outcome == "abort_settled" for receipt in ranks.values())


class GrantRoutedScheduler(StockTerminalCheckpointScheduler):
    """Keep scheduler receive and failure signals tied to D grant completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.scheduler_config.async_scheduling is not False:
            raise ValueError("GrantRoutedScheduler requires --no-async-scheduling")
        connector = self.connector
        children = (
            connector.sub_connectors
            if isinstance(connector, MultiConnector)
            else [connector]
        )
        owners = [child for child in children if isinstance(child, HiSparseConnector)]
        if len(owners) > 1:
            raise ValueError("Multiple Kitepond HiSparse owners were configured")
        self._grant_completion_owner = owners[0] if owners else None

        if self._grant_completion_owner is not None:
            if children[0] is not self._grant_completion_owner:
                raise ValueError("The Kitepond HiSparse grant owner must load first")
            self._grant_completion_owner.enable_grant_completion_routing()
        self._grant_completion_source: DecodeImportReceipts | None = None
        self._native_decode_proofs: dict[str, dict[int, DecodeImportRankReceipt]] = {}
        self._settled_decode_imports: dict[str, DecodeImportBinding] = {}

    def bind_verified_decode_imports(self, source: DecodeImportReceipts) -> None:
        """Bind the authenticated post-fence reader on the scheduler owner thread."""
        owner = self._grant_completion_owner
        if (
            owner is None
            or owner.stage != "decode"
            or self._grant_completion_source is not None
            or not callable(source)
        ):
            raise ValueError("D grant completion source is unavailable or bound")
        owner.decode_grant_bindings()  # Checks scheduler ownership.
        self._grant_completion_source = source

    def verified_decode_import(self, request: Request) -> DecodeImportBinding | None:
        """Look up the same live native Request after a successful D quorum."""
        owner = self._grant_completion_owner
        if owner is None or owner.stage != "decode":
            return None
        owner.decode_grant_bindings()  # Checks scheduler ownership.
        binding = self._settled_decode_imports.get(request.request_id)
        if (
            binding is not None
            and binding.request is request
            and self.requests.get(request.request_id) is request
        ):
            return binding
        return None

    def _forget_retired_import(self, request: Request) -> None:
        request_id = request.request_id
        binding = self._settled_decode_imports.get(request_id)
        if (
            binding is not None
            and binding.request is request
            and self.requests.get(request_id) is not request
        ):
            self._settled_decode_imports.pop(request_id)

    def _free_blocks(self, request: Request) -> None:
        super()._free_blocks(request)
        self._forget_retired_import(request)

    def _release_terminal_request(self, request: Request) -> None:
        super()._release_terminal_request(request)
        self._forget_retired_import(request)

    def _observe_native_decode_proofs(self, selected, pending) -> None:
        by_grant = {binding.grant.sha256: binding for binding in pending.values()}
        proofs = self._native_decode_proofs
        for grant in proofs.keys() - by_grant.keys():
            del proofs[grant]
        metadata = selected.kv_connector_worker_meta
        if isinstance(self.connector, MultiConnector):
            if not isinstance(metadata, MultiKVConnectorWorkerMetadata):
                return
            metadata = metadata.metadata[0]
        if not isinstance(metadata, DecodeReceiveProofs):
            return
        for receipt in metadata.receipts:
            if type(receipt) is not DecodeImportRankReceipt:
                raise ValueError("Invalid native D receive receipt")
            binding = by_grant.get(receipt.grant_sha256)
            if binding is None:
                continue
            if binding.destination_sha256 != receipt.destination_sha256:
                raise ValueError("Native D receipt differs from its active allocation")
            ranks = proofs.setdefault(receipt.grant_sha256, {})
            if ranks.setdefault(receipt.rank, receipt) != receipt:
                raise ValueError("Native D rank changed its settled outcome")

    def update_from_output(self, scheduler_output, model_runner_output):
        owner = self._grant_completion_owner
        if owner is None:
            return super().update_from_output(scheduler_output, model_runner_output)

        protected = owner.decode_grant_bindings()
        pending = {
            request_id: binding
            for request_id, binding in protected.items()
            if binding is not None
        }
        original = model_runner_output.kv_connector_output
        selected = original or KVConnectorOutput()
        self._observe_native_decode_proofs(selected, pending)
        # A flat failure cannot disambiguate overlapping HMA block IDs.
        if selected.invalid_block_ids and any(
            block in selected.invalid_block_ids
            for binding in pending.values()
            for group in binding.block_ids
            for block in group
        ):
            raise ValueError("D grant requires request-scoped transfer failures")
        accepted = {}
        settled_failures = set()
        source = self._grant_completion_source
        for request_id, binding in pending.items():
            if self.requests.get(request_id) is not binding.request:
                raise ValueError("D import changed its active scheduler request")
            native = self._native_decode_proofs.get(binding.grant.sha256, {})
            external = () if source is None else (source(binding) or ())
            failed = _settled_failure(binding, tuple(native.values()))
            external_failed = _settled_failure(binding, external)
            if failed is None:
                failed = external_failed
            elif external_failed is not None and external_failed != failed:
                raise ValueError("D proof sources disagree on a settled grant")
            if failed is None or (binding.request.is_finished() and not failed):
                continue
            accepted[request_id] = binding
            if failed:
                settled_failures.add(request_id)

        received = set(selected.finished_recving or ()) - protected.keys()
        sent = set(selected.finished_sending or ()) - pending.keys()
        failures = selected.failed_recving - protected.keys()
        received.update(accepted)
        failures.update(settled_failures)
        if accepted or (original is not None and protected):
            routed = replace(
                selected,
                finished_recving=received or None,
                finished_sending=sent or None,
                failed_recving=failures,
            )
            model_runner_output = replace(
                model_runner_output, kv_connector_output=routed
            )
        with owner.routed_decode_completions(accepted):
            output = super().update_from_output(scheduler_output, model_runner_output)
        for request_id, binding in accepted.items():
            if (
                request_id not in settled_failures
                and self.requests.get(request_id) is binding.request
            ):
                self._settled_decode_imports[request_id] = binding
        return output
