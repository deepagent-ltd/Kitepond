# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable, native-admitted GLM exporter metadata for K=0 prefill."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kitepond.core.readiness import ReadyAttempt

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorMetadata,
)
from vllm.v1.request import Request

from .host_authority import NativeHostGrant
from .ready_transport import ReadyTicket

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

K0_CONFIG = "kitepond_glm_prefill_k0"


@dataclass(frozen=True)
class GlmK0Route:
    grant: NativeHostGrant
    ticket: ReadyTicket
    original_tokens: int
    next_token: int
    prompt_sha256: str
    imported_tokens: int = 0

    @property
    def valid_tokens(self) -> int:
        return self.original_tokens - 1

    @classmethod
    def from_native_request(
        cls,
        request: Request,
        grant: NativeHostGrant,
        ticket: ReadyTicket,
        *,
        imported_tokens: int = 0,
    ) -> GlmK0Route:
        if (
            type(request) is not Request
            or type(grant) is not NativeHostGrant
            or type(ticket) is not ReadyTicket
        ):
            raise ValueError("GLM K0 requires native admission objects")
        params = request.kv_transfer_params or {}
        tokens = request.prompt_token_ids
        expected = ReadyAttempt(
            grant.request_id, grant.transfer_id, grant.epoch, grant.sha256
        )
        if (
            request.request_id != grant.request_id
            or request.max_tokens != 1
            or request.num_computed_tokens
            or request.get_skip_reading_prefix_cache()
            or not tokens
            or not 2 <= len(tokens) == request.num_prompt_tokens <= 1048576
            or type(imported_tokens) is not int
            or not 0 <= imported_tokens < len(tokens) - 1
            or any(type(token) is not int or token < 0 for token in tokens)
            or params.get("do_remote_decode") is not True
            or params.get("transfer_id") != grant.transfer_id
            or grant.prefix_tokens
            or grant.reservation_tokens < len(tokens) - 1
            or ticket.attempt != expected
        ):
            raise ValueError("GLM K0 request has no exact native P admission")
        return cls(
            grant,
            ticket,
            len(tokens),
            tokens[-1],
            _prompt_digest(tokens),
            imported_tokens,
        )

    def verify_truncation(self, request: Request) -> None:
        tokens = request.prompt_token_ids
        if (
            not tokens
            or len(tokens) != self.valid_tokens
            or request.num_prompt_tokens != self.valid_tokens
            or _prompt_digest([*tokens, self.next_token]) != self.prompt_sha256
        ):
            raise ValueError("Native P did not leave exactly the withheld prompt token")


def _prompt_digest(tokens) -> str:
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class GlmK0Chunk:
    route: GlmK0Route
    start: int
    end: int

    @property
    def request_id(self) -> str:
        return self.route.grant.request_id

    @property
    def epoch(self) -> int:
        return self.route.grant.epoch

    @property
    def final(self) -> bool:
        return self.end == self.route.valid_tokens


class GlmK0Metadata(MooncakeConnectorMetadata):
    def __init__(self, native: MooncakeConnectorMetadata, chunks, retired) -> None:
        self.reqs_to_recv = native.reqs_to_recv
        self.reqs_to_send = native.reqs_to_send
        self.reqs_not_processed = native.reqs_not_processed
        self.chunks: tuple[GlmK0Chunk, ...] = chunks
        self.retired: frozenset[str] = retired


class GlmK0SchedulerRoutes:
    """Attach producer state only to native scheduling results for the same request."""

    def __init__(self) -> None:
        self._requests: dict[str, tuple[Request, GlmK0Route]] = {}

    def stage(self, request: Request, route: GlmK0Route) -> None:
        key = request.request_id
        if key in self._requests or key != route.grant.request_id:
            raise ValueError("GLM K0 request is already staged or foreign")
        self._requests[key] = (request, route)

    def require(self, request: Request) -> GlmK0Route:
        staged = self._requests.get(request.request_id)
        if staged is None or staged[0] is not request:
            raise ValueError("GLM K0 request was not admitted by the native owner")
        return staged[1]

    def retire(self, request: Request) -> None:
        self.require(request)
        self._requests.pop(request.request_id)

    def build(
        self, output: SchedulerOutput, native: MooncakeConnectorMetadata
    ) -> GlmK0Metadata:
        if output.scheduled_spec_decode_tokens:
            raise ValueError("GLM P K0 cannot execute previously proposed drafts")
        positions = {
            request.req_id: request.num_computed_tokens
            for request in output.scheduled_new_reqs
        }
        positions.update(
            zip(
                output.scheduled_cached_reqs.req_ids,
                output.scheduled_cached_reqs.num_computed_tokens,
                strict=True,
            )
        )
        chunks = []
        for request_id, count in output.num_scheduled_tokens.items():
            if request_id not in self._requests or request_id not in positions:
                raise ValueError("GLM K0 scheduling lacks the native admitted route")
            request, route = self._requests[request_id]
            begin = positions[request_id]
            if not 0 <= begin < begin + count <= route.valid_tokens:
                raise ValueError("GLM native P exceeded its withheld-token frontier")
            if request.num_prompt_tokens != route.valid_tokens:
                raise ValueError(
                    "GLM native P prompt boundary changed during scheduling"
                )
            chunks.append(GlmK0Chunk(route, begin, begin + count))
        output.num_spec_tokens_to_schedule = 0
        output.skip_drafts = True
        return GlmK0Metadata(native, tuple(chunks), frozenset(output.finished_req_ids))
