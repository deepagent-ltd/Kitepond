# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve the original prompt frontier in the stock K0 MTP prefill pass."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from kitepond.models.glm5next.layout import Glm5NextPDLayout
from kitepond.models.glm5next.ready import PrefillWriteGrant
from kitepond.models.glm5next.shared_region import NativeHiSparseRegionDescriptor

from .glm_k0_routes import GlmK0Chunk, GlmK0Metadata, GlmK0Route
from .glm_prefill_sources import GlmPrefillExport
from .prefill_fence import PrefillWrite

if TYPE_CHECKING:
    from kitepond.core.readiness import ReadyReceipt


def _record_event(device):
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))
    return event


@dataclass(frozen=True)
class _Final:
    chunk: GlmK0Chunk
    event: object
    observed: torch.Tensor
    state_blocks: tuple[tuple[str, torch.Tensor], ...]
    last_index: int


class GlmK0PrefillRuntime:
    """Prove the native target and draft obey the same admitted frontier."""

    def __init__(
        self, model_sha256: str, config_sha256: str, *, record_event=_record_event
    ):
        self.model_sha256 = model_sha256
        self.config_sha256 = config_sha256
        self._record_event = record_event
        self._region: NativeHiSparseRegionDescriptor | None = None
        self._metadata: GlmK0Metadata | None = None
        self._grants: dict[str, PrefillWriteGrant] = {}
        self._chunk_ends: dict[str, int] = {}
        self._completed: dict[str, _Final] = {}
        self._confirmed: dict[str, tuple[tuple[str, int], ...]] = {}
        self._writes: tuple[PrefillWrite, ...] = ()
        self._verify_warm_import: Callable[[GlmK0Route], None] | None = None

    def bind_warm_import_verifier(self, verify: Callable[[GlmK0Route], None]) -> None:
        if self._verify_warm_import is not None or not callable(verify):
            raise ValueError("Native P K0 already has a warm import verifier")
        self._verify_warm_import = verify

    def bind_region(self, region: NativeHiSparseRegionDescriptor) -> None:
        if (
            self._region is not None
            or type(region) is not NativeHiSparseRegionDescriptor
        ):
            raise ValueError("GLM K0 P requires one native rank-local backing")
        region.validate()
        self._region = region

    def bind_step(self, metadata: GlmK0Metadata) -> None:
        if type(metadata) is not GlmK0Metadata or self._region is None or self._writes:
            raise ValueError("GLM K0 P native metadata changed before its save fence")
        for request_id in metadata.retired:
            self._grants.pop(request_id, None)
            self._chunk_ends.pop(request_id, None)
            self._completed.pop(request_id, None)
            self._confirmed.pop(request_id, None)
        self._metadata = metadata

    def prepare(self, batch, last_sampled, next_prefill_tokens):
        metadata, region = self._metadata, self._region
        if metadata is None or region is None or batch.num_draft_tokens:
            raise ValueError("GLM K0 P requires the actual admitted prefill batch")
        by_request = {chunk.request_id: chunk for chunk in metadata.chunks}
        if (
            batch.num_reqs != len(by_request)
            or len(set(batch.req_ids)) != batch.num_reqs
            or set(batch.req_ids) != by_request.keys()
            or len(set(batch.idx_mapping_np.tolist())) != batch.num_reqs
        ):
            raise ValueError("GLM K0 P request batch differs from native scheduling")

        final_rows, slots, tokens = [], [], []
        chunks = tuple(by_request[request_id] for request_id in batch.req_ids)
        for i, chunk in enumerate(chunks):
            route = chunk.route
            grant = route.grant
            slot = int(batch.idx_mapping_np[i])
            if (
                (grant.engine_id, grant.boot_id)
                != (region.authority_engine_id, region.authority_boot_id)
                or chunk.start
                != self._chunk_ends.get(chunk.request_id, route.imported_tokens)
                or int(batch.num_computed_tokens_np[i]) != chunk.start
                or int(batch.num_scheduled_tokens[i]) != chunk.end - chunk.start
                or int(batch.prefill_len_np[i]) != route.valid_tokens
                or not 0 <= slot < last_sampled.numel()
                or not bool(batch.is_prefilling_np[i])
            ):
                raise ValueError("GLM K0 P input changed its admitted causal prefix")
            old = self._grants.get(chunk.request_id)
            if old is None:
                if route.imported_tokens:
                    verify = self._verify_warm_import
                    if verify is None:
                        raise ValueError(
                            "Native P K0 cannot trust an unverified warm prefix"
                        )
                    verify(route)
                self._grants[chunk.request_id] = PrefillWriteGrant(
                    route.ticket.attempt, region, route.valid_tokens
                )
            elif old.attempt != route.ticket.attempt:
                raise ValueError("GLM K0 P reused a request under another grant")
            if chunk.final:
                final_rows.append(i)
                slots.append(slot)
                tokens.append(route.next_token)

        if not final_rows:
            return chunks, (), last_sampled, next_prefill_tokens
        if (
            last_sampled.ndim != 2
            or last_sampled.shape[1] != 1
            or next_prefill_tokens.ndim != 2
            or next_prefill_tokens.shape[0] < 1
            or next_prefill_tokens.shape[1] != last_sampled.shape[0]
            or next_prefill_tokens.device != last_sampled.device
            or min(slots) < 0
            or max(slots) >= last_sampled.shape[0]
        ):
            raise ValueError("GLM K0 token arrays differ from the native request slots")
        index = torch.tensor(slots, dtype=torch.long, device=last_sampled.device)
        # Native RequestState stores samples as [slots, 1] and prefill
        # lookahead as [rows, slots]. The K0 input consumes only row zero.
        fixed_last = last_sampled.clone()
        fixed_next = next_prefill_tokens.clone()
        fixed_last[:, 0].index_copy_(
            0, index, torch.tensor(tokens, dtype=fixed_last.dtype, device=index.device)
        )
        fixed_next[0].index_copy_(
            0, index, torch.tensor(tokens, dtype=fixed_next.dtype, device=index.device)
        )
        return chunks, tuple(final_rows), fixed_last, fixed_next

    def finish(
        self,
        chunks,
        final_rows,
        batch,
        metadata,
        draft_buffers,
        last_indices,
        num_sampled,
        num_rejected,
    ) -> None:
        if final_rows:
            device = draft_buffers.input_ids.device
            rows = torch.tensor(final_rows, dtype=torch.long, device=device)
            last = last_indices.index_select(0, rows).long()
            observed = torch.stack(
                (
                    last,
                    draft_buffers.input_ids.index_select(0, last).long(),
                    draft_buffers.positions.index_select(0, last).long(),
                    batch.positions.index_select(0, last).long(),
                    num_sampled.index_select(0, rows).long(),
                    num_rejected.index_select(0, rows).long(),
                ),
                dim=1,
            )
            blocks = {}
            for layer, meta in metadata.items():
                if not hasattr(meta, "non_spec_state_indices_tensor"):
                    continue
                state = meta.non_spec_state_indices_tensor
                if (
                    state is None
                    or state.ndim != 1
                    or state.shape[0] < batch.num_reqs
                    or meta.spec_sequence_masks is not None
                ):
                    raise ValueError(
                        "GLM K0 P did not select native non-spec KDA slots"
                    )
                blocks[layer] = state.index_select(0, rows)
            if not blocks:
                raise ValueError("GLM K0 P has no native GDN state selection witness")
            event = self._record_event(device)
            for i, row in enumerate(final_rows):
                chunk = chunks[row]
                self._completed[chunk.request_id] = _Final(
                    chunk,
                    event,
                    observed[i],
                    tuple((name, value[i]) for name, value in sorted(blocks.items())),
                    int(batch.query_start_loc_np[row + 1]) - 1,
                )
            self._writes = tuple(
                PrefillWrite(
                    chunks[row].route.ticket, self._grants[chunks[row].request_id]
                )
                for row in final_rows
            )
        for chunk in chunks:
            self._chunk_ends[chunk.request_id] = chunk.end
        self._metadata = None

    def current_grant(self, request_id: str) -> PrefillWriteGrant | None:
        return self._grants.get(request_id)

    def collect_writes(self) -> tuple[PrefillWrite, ...]:
        writes, self._writes = self._writes, ()
        return writes

    def require_completed_boundary(self, request_id: str) -> int:
        final = self._completed.get(request_id)
        if final is None or self._chunk_ends.get(request_id) != final.chunk.end:
            raise ValueError("GLM K0 P has no completed final frontier")
        return final.chunk.end

    def wait_ready(self, request_id: str) -> GlmK0Chunk:
        boundary = self.require_completed_boundary(request_id)
        final = self._completed[request_id]
        final.event.synchronize()
        expected = [
            final.last_index,
            final.chunk.route.next_token,
            boundary - 1,
            boundary - 1,
            1,
            0,
        ]
        if final.observed.cpu().tolist() != expected:
            raise ValueError(
                "GLM native draft did not use the exact final prompt frontier"
            )
        blocks = tuple((name, int(value.item())) for name, value in final.state_blocks)
        if any(block <= 0 for _, block in blocks):
            raise ValueError("GLM K0 P selected a null KDA state slot")
        self._confirmed[request_id] = blocks
        return final.chunk

    def export(self, write: PrefillWrite, receipt: ReadyReceipt) -> GlmPrefillExport:
        request_id = write.grant.attempt.request_id
        final = self._completed[request_id]
        if (
            request_id not in self._confirmed
            or final.chunk.route.ticket != write.ticket
        ):
            raise ValueError("GLM K0 P export has no actual fenced draft frontier")
        boundary = final.chunk.end
        return GlmPrefillExport(
            Glm5NextPDLayout(
                request_id,
                self.model_sha256,
                self.config_sha256,
                0,
                receipt.attempt.epoch,
                boundary,
                boundary,
                0,
                "decode_ready",
            ),
            0,
            native_kda_blocks=self._confirmed[request_id],
        )
