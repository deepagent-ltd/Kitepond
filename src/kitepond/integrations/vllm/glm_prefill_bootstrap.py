# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind model-owned GLM P READY to native Mooncake at worker startup."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from kitepond.core.readiness import ReadyReceipt, _digest
from kitepond.integrations.mooncake.named_control import (
    GlmWindowControlHandler,
    GlmWindowControlScope,
)
from kitepond.models.glm5next.shared_mla import SHARED_MLA_CONFIG

from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnector as NativeMooncakeConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.distributed.parallel_state import get_dcp_group, get_tp_group

from .glm_k0_routes import K0_CONFIG, GlmK0Route, GlmK0SchedulerRoutes
from .glm_native_views import GlmNativeStateViews
from .glm_prefill_sources import GlmPrefillExport, GlmPrefillReadySources
from .hisparse_attachment import ATTACHMENT_MODE, HiSparseConnector
from .host_authority import NativeHostGrant
from .prefill_fence import NativePrefillSaveFence, PrefillWrite
from .ready_transport import ReadyPublisher, ReadyTicket

_CONFIG = "kitepond_glm_window"
_KEYS_FILE = "KITEPOND_GLM_WINDOW_KEYS_FILE"
_KEYS_SCHEMA = "kitepond.glm-window-keys.v1"
_MAX_KEY_BYTES = 4096
_RANKS = frozenset(map(str, range(4)))


@dataclass(frozen=True)
class _Profile:
    stage: str
    p_engine_id: str
    p_boot_id: str
    d_engine_id: str
    d_boot_id: str
    decode_host: str

    @classmethod
    def from_config(cls, config) -> _Profile:
        options = config.additional_config
        window = options.get(_CONFIG) if isinstance(options, Mapping) else None
        lease = options.get("kitepond") if isinstance(options, Mapping) else None
        if (
            not isinstance(window, Mapping)
            or set(window)
            != {"p_engine_id", "p_boot_id", "d_engine_id", "d_boot_id", "decode_host"}
            or any(
                type(value) is not str or not 0 < len(value) <= 128
                for value in window.values()
            )
            or not isinstance(lease, Mapping)
            or lease.get("mode") != ATTACHMENT_MODE
            or lease.get("stage") not in ("prefill", "decode")
            or (lease.get("authority_engine_id"), lease.get("boot_id"))
            != (window["d_engine_id"], window["d_boot_id"])
            or window["p_engine_id"] == window["d_engine_id"]
            or window["p_boot_id"] == window["d_boot_id"]
            or config.kv_transfer_config is None
        ):
            raise ValueError("GLM named control requires an exact native role profile")
        profile = cls(stage=lease["stage"], **window)
        if not isinstance(
            ipaddress.ip_address(profile.decode_host), ipaddress.IPv4Address
        ):
            raise ValueError("GLM single-node named control requires its IPv4 D host")
        transfer = config.kv_transfer_config
        if profile.stage == "prefill":
            allowed = ("kv_producer", "kv_both")
            engine_id = profile.p_engine_id
        else:
            allowed = ("kv_consumer", "kv_both")
            engine_id = profile.d_engine_id
        if transfer.kv_role not in allowed or transfer.engine_id != engine_id:
            raise ValueError(
                "GLM named control belongs to another native role or engine"
            )
        return profile

    def scope_for_rank(self, rank: int) -> GlmWindowControlScope:
        path = os.environ.get(_KEYS_FILE)
        if not path or not os.path.isabs(path) or str(rank) not in _RANKS:
            raise ValueError("GLM named control needs a private rank key file")
        try:
            fd = os.open(
                path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
            )
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(info.st_mode)
                    or stat.S_IMODE(info.st_mode) not in (0o400, 0o600)
                    or info.st_uid != os.geteuid()
                    or not 0 < info.st_size <= _MAX_KEY_BYTES
                ):
                    raise ValueError
                raw = stream.read(_MAX_KEY_BYTES + 1)
            if len(raw) > _MAX_KEY_BYTES:
                raise ValueError
            document = json.loads(raw)
            expected = {
                "schema": _KEYS_SCHEMA,
                "p_engine_id": self.p_engine_id,
                "p_boot_id": self.p_boot_id,
                "d_engine_id": self.d_engine_id,
                "d_boot_id": self.d_boot_id,
            }
            if type(document) is not dict or set(document) != {*expected, "rank_keys"}:
                raise ValueError
            keys = document.pop("rank_keys")
            if (
                document != expected
                or type(keys) is not dict
                or set(keys) != _RANKS
                or any(not _digest(value) for value in keys.values())
                or len(set(keys.values())) != 4
            ):
                raise ValueError
            key = bytes.fromhex(keys[str(rank)])
        except (UnicodeError, ValueError, OSError):
            raise ValueError("Invalid private GLM named control credentials") from None
        return GlmWindowControlScope(
            self.p_boot_id, self.d_boot_id, rank, key, forward_abort=True
        )


class MooncakeConnector(NativeMooncakeConnector):
    """Select this explicit external module for the named GLM profile."""

    def __init__(self, vllm_config, role, kv_cache_config):
        self._profile = _Profile.from_config(vllm_config)
        self._scope = None
        self._sources = None
        self._source_handler = None
        self._native_fence = None
        self._fence_input = None
        self._publisher = None
        self._k0_runtime = None
        self._k0_routes = None
        mla_preflight = vllm_config.additional_config.get(SHARED_MLA_CONFIG, False)
        if type(mla_preflight) is not bool:
            raise ValueError("GLM MLA preflight needs an explicit boolean")
        settings = vllm_config.additional_config.get(K0_CONFIG)
        self._k0_enabled = self._profile.stage == "prefill" and settings is not None
        if self._k0_enabled and role is KVConnectorRole.SCHEDULER:
            self._k0_routes = GlmK0SchedulerRoutes()
        if role is KVConnectorRole.WORKER:
            tp, dcp = get_tp_group(), get_dcp_group()
            if (
                tp.world_size != 4
                or dcp.world_size != 4
                or tp.rank_in_group != dcp.rank_in_group
            ):
                raise ValueError(
                    "GLM named control needs aligned TP4/DCP4 rank identity"
                )
            self._scope = self._profile.scope_for_rank(tp.rank_in_group)
        super().__init__(vllm_config, role, kv_cache_config)
        if self._scope is not None and self._profile.stage == "prefill":
            worker = self.connector_worker
            sources = GlmPrefillReadySources(
                worker, self._profile.decode_host, mla_preflight=mla_preflight
            )
            try:
                handler = GlmWindowControlHandler(
                    worker,
                    self._scope,
                    sources.lookup,
                    admit_source=sources.open_writer,
                    prepare_mla=sources.prepare_shared_mla if mla_preflight else None,
                    cancel_unopened=sources.cancel_unopened,
                    reject_native_requests=True,
                )
            except BaseException:
                sources.close()
                raise
            self._sources, self._source_handler = sources, handler

    def on_new_request(self, request) -> None:
        route = (
            self._k0_routes.require(request) if self._k0_routes is not None else None
        )
        super().on_new_request(request)
        if route is not None:
            route.verify_truncation(request)

    def build_connector_meta(self, scheduler_output):
        native = super().build_connector_meta(scheduler_output)
        if self._k0_routes is None:
            return native
        return self._k0_routes.build(scheduler_output, native)

    def request_finished_all_groups(self, request, block_ids):
        result = super().request_finished_all_groups(request, block_ids)
        if self._k0_routes is not None:
            self._k0_routes.retire(request)
        return result

    def bind_k0_runtime(self, runtime) -> None:
        if not self._k0_enabled or self._k0_runtime is not None or self._scope is None:
            raise ValueError("GLM K0 P has no qualified native worker role")
        self._k0_runtime = runtime

    def start_load_kv(self, forward_context, **kwargs):
        if self._k0_enabled:
            if self._k0_runtime is None:
                raise ValueError("GLM K0 P has no admitted native model runtime")
            self._k0_runtime.bind_step(self._get_connector_metadata())
        super().start_load_kv(forward_context, **kwargs)

    def window_scope(self) -> GlmWindowControlScope:
        if self._scope is None:
            raise ValueError("GLM named control has no native worker rank")
        return self._scope

    def bind_prefill_save_fence(
        self,
        hisparse: HiSparseConnector,
        views: GlmNativeStateViews,
        publisher: ReadyPublisher,
        current_grant,
        checkpoint,
        *,
        collect_writes: Callable[[], tuple[PrefillWrite, ...]],
        export: Callable[[PrefillWrite, ReadyReceipt], GlmPrefillExport],
    ) -> None:
        sources = self._sources
        if (
            sources is None
            or self._native_fence is not None
            or type(hisparse) is not HiSparseConnector
            or hisparse.stage != "prefill"
            or hisparse.role is not KVConnectorRole.WORKER
            or not callable(collect_writes)
            or (
                views.descriptor.authority_engine_id,
                views.descriptor.authority_boot_id,
            )
            != (self._profile.d_engine_id, self._profile.d_boot_id)
        ):
            raise ValueError("GLM P fence requires the qualified native prefill owner")
        fence = NativePrefillSaveFence(
            hisparse.connector_worker,
            publisher,
            current_grant,
            checkpoint,
            local_rank=self.window_scope().rank,
            record_source=sources.record_fenced,
        )
        sources.bind_export(views, export)
        self._native_fence, self._fence_input = fence, (hisparse, collect_writes)
        self._publisher = publisher

    def wait_for_save(self):
        super().wait_for_save()
        if self._native_fence is not None:
            hisparse, collect = self._fence_input
            writes = collect()
            if writes:
                self._native_fence.publish(hisparse._get_connector_metadata(), writes)

    def get_transfer_results(self, finished_req_ids):
        if self._sources is not None and self._sources.has_fenced_sources():
            worker = self.connector_worker
            asyncio.run_coroutine_threadsafe(
                self._sources.expire(), worker.sender_loop
            ).result()
        return super().get_transfer_results(finished_req_ids)

    def shutdown(self):
        if self._sources is not None:
            sources = self._sources
            if self._publisher is not None:
                self._publisher.close()
                self._publisher = None
            sources.close()
            self._sources = None
        if self.connector_worker is not None:
            self.connector_worker.shutdown()
        super().shutdown()


def stage_native_glm_k0_request(
    connector: MultiConnector,
    request,
    grant: NativeHostGrant,
    ticket: ReadyTicket,
    *,
    imported_tokens: int = 0,
    stage_warm: Callable[[], None] | None = None,
) -> None:
    """Use the trusted P control owner before the normal native add_request."""
    if type(connector) is not MultiConnector or len(connector.sub_connectors) != 2:
        raise ValueError("GLM P K0 admission needs native HiSparse and Mooncake")
    hisparse, mooncake = connector.sub_connectors
    if (
        type(hisparse) is not HiSparseConnector
        or type(mooncake) is not MooncakeConnector
    ):
        raise ValueError("GLM P K0 admission has another native child")
    routes = mooncake._k0_routes
    if routes is None:
        raise ValueError("GLM P K0 scheduler role is not enabled")
    if bool(imported_tokens) != callable(stage_warm):
        raise ValueError("Native warm K0 needs its verified allocation owner")
    route = GlmK0Route.from_native_request(
        request, grant, ticket, imported_tokens=imported_tokens
    )
    routes.stage(request, route)
    try:
        if stage_warm is not None:
            stage_warm()
        else:
            hisparse.stage_host_grant(request, grant)
    except BaseException:
        routes.retire(request)
        raise


def bind_native_glm_prefill_source(
    runner,
    publisher: ReadyPublisher,
    current_grant,
    checkpoint,
    *,
    collect_writes: Callable[[], tuple[PrefillWrite, ...]],
    export: Callable[[PrefillWrite, ReadyReceipt], GlmPrefillExport],
) -> GlmNativeStateViews:
    """Enroll the initialized P rank's actual model views and admission owner."""
    connector = get_kv_transfer_group()
    if type(connector) is not MultiConnector or len(connector.sub_connectors) != 2:
        raise ValueError("GLM prefill requires native HiSparse then named Mooncake")
    hisparse, mooncake = connector.sub_connectors
    if type(mooncake) is not MooncakeConnector:
        raise ValueError("GLM prefill has no explicit named Mooncake child")
    views = GlmNativeStateViews(runner, mooncake.window_scope().rank)
    mooncake.bind_prefill_save_fence(
        hisparse,
        views,
        publisher,
        current_grant,
        checkpoint,
        collect_writes=collect_writes,
        export=export,
    )
    return views
