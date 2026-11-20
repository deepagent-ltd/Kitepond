# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise explicit native HiSparse host attachment on the scheduler owner."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from kitepond.models.glm5next.shared_mla import valid_rank_digests

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnector as NativeHiSparseConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnectorWorkerMetadata,
)
from vllm.distributed.parallel_state import get_dcp_group
from vllm.v1.hisparse.coordinator import HiSparseCoordinator
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from .decode_receive import (
    DecodeReceiveMailbox,
    DecodeReceiveTicket,
    DecodeReceiveWork,
    destination_digest,
)
from .hisparse_binding import (
    attach_decode_grant,
    attach_prefill_grant,
    bind_decode_host,
    bind_prefill_host,
)
from .host_authority import NativeHostGrant
from .prefill_return_receive import (
    PrefillReturnFence,
    PrefillReturnProofs,
    prefill_return_boot_id,
)
from .profile import UnsupportedKitepondProfile

if TYPE_CHECKING:
    from kitepond.integrations.mooncake.live_receive import RankWindowAttempt

    from .checkpoint import CheckpointReadReceipt
    from .glm_terminal_return import GlmTerminalReturn

ATTACHMENT_MODE = "hisparse-attachment-probe"


@dataclass(frozen=True)
class DecodeImportBinding:
    """Actual native destination assigned by admission for a staged host grant."""

    request: Request
    grant: NativeHostGrant
    imported_tokens: int
    block_ids: tuple[tuple[int, ...], ...]
    detached_cold: bool = False
    mla_rank_digests: tuple[str, ...] | None = None

    @property
    def destination_sha256(self) -> str:
        return destination_digest(
            self.imported_tokens, self.block_ids, self.mla_rank_digests
        )


@dataclass
class _StagedGrant:
    request: Request
    grant: NativeHostGrant
    imported_tokens: int
    attached: bool = False
    finished: bool = False
    decoded: bool = False
    binding: DecodeImportBinding | None = None
    detached_cold: bool = False
    mla_rank_digests: tuple[str, ...] | None = None


class HiSparseConnector(NativeHiSparseConnector):
    """Native HiSparse owner with explicit scheduler-side attachment binding."""

    # MultiConnector identifies its stats by the real class name. Match the
    # native configuration name as well as the inherited native stats schema.

    def __init__(self, vllm_config, role, kv_cache_config):
        options = vllm_config.additional_config
        settings = options.get("kitepond") if isinstance(options, Mapping) else None
        if (
            not isinstance(settings, Mapping)
            or set(settings) != {"mode", "stage", "authority_engine_id", "boot_id"}
            or settings["mode"] != ATTACHMENT_MODE
            or not isinstance(settings["stage"], str)
            or settings["stage"] not in {"prefill", "decode"}
            or vllm_config.cache_config.enable_prefix_caching
            or any(
                not isinstance(settings[key], str) or not settings[key]
                for key in ("authority_engine_id", "boot_id")
            )
            or (
                settings["stage"] == "decode"
                and (
                    vllm_config.kv_transfer_config is None
                    or vllm_config.kv_transfer_config.engine_id
                    != settings["authority_engine_id"]
                )
            )
        ):
            raise UnsupportedKitepondProfile(
                "Explicit verified HiSparse binding required"
            )
        if not callable(
            getattr(HiSparseCoordinator, "attach_external_host_blocks", None)
        ):
            raise UnsupportedKitepondProfile(
                "Native HiSparse attachment is unavailable"
            )
        super().__init__(vllm_config, role, kv_cache_config)
        self.stage = settings["stage"]
        self.prefill_return_boot = prefill_return_boot_id(options)
        if self.prefill_return_boot is not None and self.stage != "prefill":
            raise UnsupportedKitepondProfile("Terminal return boot belongs to P only")
        self._engine_id = settings["authority_engine_id"]
        self._boot_id = settings["boot_id"]
        self._owner = threading.get_ident()
        self._bound_control: dict[str, Callable[[str, dict], dict]] = {}
        self._manager = None
        self.authority = None
        self._borrowed = None
        self._pending: dict[str, _StagedGrant] = {}
        self._routed: dict[str, DecodeImportBinding] | None = None
        self._routed_prefill: dict[str, DecodeImportBinding] | None = None
        self._prefill_return_fences: dict[str, PrefillReturnFence] = {}
        self._prefill_return_lock = threading.Lock()
        self._completion_routing_enabled = False
        self._decode_mailbox: DecodeReceiveMailbox | None = None
        self._verify_decode_fence: (
            Callable[[DecodeImportBinding, KVConnectorOutput], None] | None
        ) = None

    def _require_scheduler_owner(self) -> None:
        if (
            self.role is not KVConnectorRole.SCHEDULER
            or threading.get_ident() != self._owner
        ):
            raise RuntimeError("HiSparse attachment belongs to the scheduler owner")

    def bind_control(
        self, namespace: str, handler: Callable[[str, dict], dict]
    ) -> None:
        self._require_scheduler_owner()
        if not namespace or namespace in self._bound_control or not callable(handler):
            raise ValueError("Native HiSparse control must bind an unused namespace")
        self._bound_control[namespace] = handler

    def get_control_handler(self, namespace: str):
        handler = self._bound_control.get(namespace)
        if handler is not None:
            self._require_scheduler_owner()
            return handler
        inherited = getattr(super(), "get_control_handler", None)
        return inherited(namespace) if inherited else None

    def bind_kv_cache_manager(self, manager):
        self._require_scheduler_owner()
        if manager is self._manager and manager is not None:
            return
        if manager is None or self._manager is not None:
            raise ValueError("HiSparse attachment requires one native cache manager")
        if self.stage == "prefill":
            self._borrowed = bind_prefill_host(
                manager, engine_id=self._engine_id, boot_id=self._boot_id
            )
        else:
            self.authority = bind_decode_host(
                manager, engine_id=self._engine_id, boot_id=self._boot_id
            )
        super().bind_kv_cache_manager(manager)
        self._manager = manager

    def bind_decode_completion_verifier(
        self, verify: Callable[[DecodeImportBinding, KVConnectorOutput], None]
    ) -> None:
        """Install the actual model transfer verifier from the scheduler owner."""
        self._require_scheduler_owner()
        if (
            self.stage != "decode"
            or self._verify_decode_fence is not None
            or not callable(verify)
        ):
            raise ValueError(
                "Decode settlement verifier is unavailable or already bound"
            )
        self._verify_decode_fence = verify

    def stage_host_grant(
        self,
        request: Request,
        grant: NativeHostGrant,
        *,
        imported_tokens: int = 0,
        detached_cold: bool = False,
        mla_rank_digests: tuple[str, ...] | None = None,
    ) -> None:
        """Queue one trusted lease against the exact native Request object."""
        self._require_scheduler_owner()
        if (
            self._manager is None
            or type(request) is not Request
            or type(grant) is not NativeHostGrant
            or request.status != RequestStatus.WAITING
            or request.num_computed_tokens
            or request.request_id in self._pending
            or request.request_id != grant.request_id
            or (grant.engine_id, grant.boot_id) != (self._engine_id, self._boot_id)
            or type(imported_tokens) is not int
            or type(detached_cold) is not bool
            or (detached_cold and (self.stage != "decode" or grant.prefix_tokens))
            or (
                mla_rank_digests is not None
                and (
                    self.stage != "decode"
                    or detached_cold
                    or grant.prefix_tokens
                    or grant.copy_tail is not None
                    or not valid_rank_digests(mla_rank_digests)
                )
            )
        ):
            raise ValueError("Invalid HiSparse request/lease binding")
        if self.stage == "prefill":
            if imported_tokens or grant.prefix_tokens or request.max_tokens != 1:
                raise ValueError("Prefill attachment probe requires a cold grant")
        elif request.skip_reading_prefix_cache or not (
            0 < imported_tokens < request.num_prompt_tokens
            and imported_tokens <= grant.reservation_tokens
        ):
            raise ValueError("Decode attachment needs a bounded imported prefix")
        self._pending[request.request_id] = _StagedGrant(
            request,
            grant,
            imported_tokens,
            detached_cold=detached_cold,
            mla_rank_digests=mla_rank_digests,
        )

    def stage_prefill_terminal_return(
        self, request: Request, grant: NativeHostGrant, *, imported_tokens: int
    ) -> None:
        """Stage the P allocation after its scheduler verifies the terminal offer."""
        self._require_scheduler_owner()
        if (
            self.stage != "prefill"
            or self.prefill_return_boot is None
            or type(imported_tokens) is not int
            or not 0 < imported_tokens < request.num_prompt_tokens
            or request.skip_reading_prefix_cache
            or grant.reservation_tokens < request.num_tokens
            or grant.readonly_pages
            or grant.copy_tail is not None
        ):
            raise ValueError(
                "Native P terminal return needs an independent writable grant"
            )
        self.stage_host_grant(request, grant)
        self._pending[request.request_id].imported_tokens = imported_tokens

    def pending_prefill_terminal_return(
        self, request: Request
    ) -> DecodeImportBinding | None:
        self._require_scheduler_owner()
        state = self._state(request)
        if (
            self.stage != "prefill"
            or self.prefill_return_boot is None
            or state is None
            or not state.imported_tokens
            or state.decoded
        ):
            return None
        return state.binding

    def has_pending_push_work(self) -> bool:
        if (
            self.stage == "prefill"
            and self.prefill_return_boot is not None
            and any(
                state.finished and state.binding is not None and not state.decoded
                for state in self._pending.values()
            )
        ):
            return True
        return super().has_pending_push_work()

    def _state(self, request: Request) -> _StagedGrant | None:
        state = self._pending.get(request.request_id)
        if state is not None and state.request is not request:
            raise ValueError("HiSparse request ID changed before lease retirement")
        return state

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self._require_scheduler_owner()
        state = self._state(request)
        if state is None:
            params = request.kv_transfer_params or {}
            if (
                self._completion_routing_enabled
                and self.stage == "decode"
                and params.get("do_remote_prefill") is True
            ):
                return 0, False
            return None, False
        if state.finished:
            return None, False
        if num_computed_tokens:
            raise ValueError("Attachment probe needs a fresh native prefix lookup")
        if state.decoded:
            return 0, False
        if not state.attached:
            if self.stage == "prefill":
                attach_prefill_grant(self._manager, self._borrowed, state.grant)
            else:
                attach_decode_grant(self._manager, self.authority, state.grant)
            state.attached = True
        return state.imported_tokens, bool(state.imported_tokens)

    def get_loaded_kv_cache_group_ids(self, request: Request) -> tuple[int, ...]:
        self._require_scheduler_owner()
        state = self._state(request)
        if (
            state is not None
            and state.imported_tokens
            and not state.finished
            and not state.decoded
        ):
            # A model-state import includes non-APC groups such as KDA and
            # the draft tail. Native hot/resident accelerator replicas stay local.
            return self._kv_cache_config.transfer_group_ids
        return super().get_loaded_kv_cache_group_ids(request)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self._require_scheduler_owner()
        state = self._state(request)
        if (
            state is None
            and self._completion_routing_enabled
            and self.stage == "decode"
        ):
            return super().update_state_after_alloc(
                request, blocks, num_external_tokens
            )
        if state is None or not state.attached or state.finished:
            raise ValueError("Native allocation has no staged HiSparse grant")
        if (
            state.imported_tokens
            and state.binding is None
            and num_external_tokens != state.imported_tokens
        ):
            raise ValueError("Native allocation skipped its staged import")
        if num_external_tokens:
            if (
                state.binding is not None
                or num_external_tokens != state.imported_tokens
            ):
                raise ValueError("Native allocation differs from its staged import")
            state.binding = DecodeImportBinding(
                request,
                state.grant,
                num_external_tokens,
                tuple(tuple(group) for group in blocks.get_block_ids()),
                state.detached_cold,
                state.mla_rank_digests,
            )
        super().update_state_after_alloc(request, blocks, num_external_tokens)

    def pending_decode_import(self, request: Request) -> DecodeImportBinding | None:
        self._require_scheduler_owner()
        state = self._state(request)
        return state.binding if state is not None and not state.decoded else None

    def enable_grant_completion_routing(self) -> None:
        self._require_scheduler_owner()
        self._completion_routing_enabled = True

    def decode_grant_bindings(self) -> dict[str, DecodeImportBinding | None]:
        """Keep receive ownership while the current request has an attached grant."""
        self._require_scheduler_owner()
        if self.stage != "decode":
            return {}
        return {
            request_id: None if state.decoded else state.binding
            for request_id, state in self._pending.items()
        }

    def build_connector_meta(self, scheduler_output):
        native = super().build_connector_meta(scheduler_output)
        if not self._completion_routing_enabled or self.stage != "decode":
            return native
        tickets = tuple(
            DecodeReceiveTicket.from_binding(binding)
            for binding in self.decode_grant_bindings().values()
            if binding is not None
        )
        if not tickets:
            return native
        return DecodeReceiveWork(
            **{field.name: getattr(native, field.name) for field in fields(native)},
            tickets=tickets,
        )

    def start_load_kv(self, forward_context, **kwargs):
        super().start_load_kv(forward_context, **kwargs)
        metadata = self._get_connector_metadata()
        if self.stage != "decode":
            return
        if isinstance(metadata, DecodeReceiveWork):
            group = get_dcp_group()
            if group.world_size != 4:
                raise ValueError("D import proofs require the actual four DCP ranks")
            if self._decode_mailbox is None:
                self._decode_mailbox = DecodeReceiveMailbox(
                    group.rank_in_group, self._boot_id
                )
            self._decode_mailbox.bind(metadata.tickets)
        elif self._decode_mailbox is not None:
            self._decode_mailbox.bind(())

    def decode_receive_mailbox(self) -> DecodeReceiveMailbox:
        """Return the process-local proof channel to the native D receive path."""
        if self.role is not KVConnectorRole.WORKER or self._decode_mailbox is None:
            raise ValueError("Native D receive mailbox is unavailable")
        return self._decode_mailbox

    def require_prefill_return_worker(
        self, transfer: GlmTerminalReturn, rank: int
    ) -> None:
        if (
            self.role is not KVConnectorRole.WORKER
            or self.stage != "prefill"
            or self.prefill_return_boot is None
        ):
            raise ValueError("Native P terminal fence has no selected worker")
        dcp = get_dcp_group()
        destination = transfer.destination
        if (
            dcp.world_size != 4
            or dcp.rank_in_group != rank
            or destination.boot_id != self.prefill_return_boot
            or destination.engine_id != self._vllm_config.kv_transfer_config.engine_id
            or (destination.grant.engine_id, destination.grant.boot_id)
            != (self._engine_id, self._boot_id)
        ):
            raise ValueError("Native P terminal fence names a foreign worker or boot")

    def report_prefill_return_fence(
        self,
        transfer: GlmTerminalReturn,
        attempt: RankWindowAttempt,
        receipt: CheckpointReadReceipt,
    ) -> None:
        """Record the local worker's native P readback after its actual fence."""
        binding = PrefillReturnFence(attempt, receipt)
        self.require_prefill_return_worker(transfer, receipt.local_rank)
        if attempt != transfer.attempt(receipt.local_rank):
            raise ValueError("Native P terminal fence changed its transfer")
        with self._prefill_return_lock:
            fences = self._prefill_return_fences
            limit = self._vllm_config.scheduler_config.max_num_seqs
            if attempt.request_id not in fences and len(fences) >= limit:
                raise BufferError("Native P terminal worker completion queue is full")
            if fences.setdefault(attempt.request_id, binding) != binding:
                raise ValueError("Native P terminal worker changed its fence receipt")

    def build_connector_worker_meta(self):
        native = super().build_connector_worker_meta()
        if self.stage == "prefill" and self.prefill_return_boot is not None:
            with self._prefill_return_lock:
                fences = tuple(self._prefill_return_fences.values())
                self._prefill_return_fences.clear()
            base = native or HiSparseConnectorWorkerMetadata({}, {})
            return PrefillReturnProofs(
                base.enqueued_transfer_counts,
                base.completed_transfer_counts,
                base.completed_host_copy_dst_ids,
                fences,
            )
        if self._decode_mailbox is None:
            return native
        return self._decode_mailbox.drain(native)

    @contextmanager
    def routed_decode_completions(
        self, accepted: dict[str, DecodeImportBinding]
    ) -> Iterator[None]:
        """Expose preverified receives to this child during native output routing."""
        self._require_scheduler_owner()
        if self._routed is not None:
            raise RuntimeError("Decode completions are already being routed")
        self._routed = accepted
        try:
            yield
        finally:
            self._routed = None

    @contextmanager
    def routed_prefill_return_completions(
        self, accepted: dict[str, DecodeImportBinding]
    ) -> Iterator[None]:
        self._require_scheduler_owner()
        if (
            self.stage != "prefill"
            or self.prefill_return_boot is None
            or self._routed_prefill is not None
        ):
            raise ValueError("Native P terminal readiness route is unavailable")
        self._routed_prefill = accepted
        try:
            yield
        finally:
            self._routed_prefill = None

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        self._require_scheduler_owner()
        if self._routed_prefill is not None:
            by_id = self._routed_prefill
            discarded = {
                request_id for request_id in by_id if self._pending[request_id].finished
            }
            if (
                not by_id.keys() <= set(connector_output.finished_recving or ())
                or connector_output.failed_recving & by_id.keys() != discarded
            ):
                raise ValueError(
                    "Native P terminal output lacks its complete routed quorum"
                )
            for request_id, binding in by_id.items():
                if self.pending_prefill_terminal_return(binding.request) is not binding:
                    raise ValueError("Native P completion changed its live allocation")
            super().update_connector_output(connector_output)
            for request_id in by_id:
                self._pending[request_id].decoded = True
            return
        settled = []
        routed = self._routed
        received = connector_output.finished_recving or ()
        for request_id in received if routed is None else routed:
            state = self._pending.get(request_id)
            if (
                self.stage != "decode"
                or state is None
                or state.binding is None
                or state.decoded
                or (
                    state.finished and request_id not in connector_output.failed_recving
                )
            ):
                raise ValueError("HiSparse completion lacks a matching verified import")
            if routed is None:
                if self._verify_decode_fence is None:
                    raise ValueError(
                        "HiSparse completion lacks a matching verified import"
                    )
                self._verify_decode_fence(state.binding, connector_output)
            elif routed[request_id] is not state.binding or request_id not in received:
                raise ValueError(
                    "HiSparse routed completion changed its import binding"
                )
            settled.append(state)
        super().update_connector_output(connector_output)
        for state in settled:
            state.decoded = True
            if state.finished:
                self._pending.pop(state.request.request_id)

    def request_finished_all_groups(self, request, block_ids):
        self._require_scheduler_owner()
        state = self._state(request)
        if state is not None:
            state.finished = True
            if self.stage == "decode":
                if not state.attached:
                    self.authority.cancel(state.grant)
                if state.binding is None or state.decoded:
                    self._pending.pop(request.request_id)
        return super().request_finished_all_groups(request, block_ids)

    def retire_prefill_grant(self, request: Request) -> NativeHostGrant:
        """Return the transport lease after native request refs have drained."""
        self._require_scheduler_owner()
        state = self._state(request)
        if self.stage != "prefill" or state is None or not state.finished:
            raise ValueError("Prefill lease is not eligible for retirement")
        if state.attached:
            self._borrowed.discard(state.grant)
        self._pending.pop(request.request_id)
        return state.grant


# Preserve the standalone attachment-probe import path.
KitepondHiSparseAttachmentConnector = HiSparseConnector
