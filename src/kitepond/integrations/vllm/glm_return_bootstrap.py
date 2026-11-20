# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Mooncake rank ownership for GLM terminal return."""

from __future__ import annotations

import asyncio
import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import torch
from kitepond.integrations.mooncake.live_receive import (
    CommittedRankWindows,
    FencedRankWindows,
    NativeRankWindowWriter,
    RankWindowAttempt,
    mooncake_receive_pool,
)
from kitepond.integrations.mooncake.named_control import (
    GlmWindowControlHandler,
    connect_glm_window_peer,
)
from kitepond.models.glm5next.fingerprint import config_fingerprint

from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    get_mooncake_bootstrap_addr,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from .glm_decode_bootstrap import GlmColdImportProfile, GlmDecodeImportRuntime
from .glm_prefill_bootstrap import MooncakeConnector as GlmMooncakeConnector
from .glm_terminal_return import (
    GlmNativeTerminalReturnWindows,
    GlmTerminalReturn,
    open_fenced_terminal_rank_writer,
)

if TYPE_CHECKING:
    from .glm_terminal_manifest import FencedTerminalSource

_CONFIG = "kitepond_glm_terminal_return"
_MAX_RANK_ATTEMPTS = 128


@dataclass(frozen=True)
class DecodeTerminalRankResult:
    """One authenticated P read fence and its local D writer disposition."""

    proof: FencedRankWindows
    committed: CommittedRankWindows | None

    def __post_init__(self) -> None:
        if (
            type(self.proof) is not FencedRankWindows
            or (
                self.proof.outcome == "acked"
                and (
                    type(self.committed) is not CommittedRankWindows
                    or self.committed.attempt != self.proof.attempt
                )
            )
            or (self.proof.outcome == "abort_settled" and self.committed is not None)
        ):
            raise ValueError("GLM terminal result differs from its native source")


@dataclass
class _RankSource:
    transfer: GlmTerminalReturn
    terminal_worker: object
    source: FencedTerminalSource
    writer: NativeRankWindowWriter | None = None
    result: DecodeTerminalRankResult | None = None


class GlmDecodeTerminalSources:
    """Keep bounded native source routes and authenticate the P receiver fence."""

    def __init__(self, worker, scope, prefill_host: str, *, capacity: int):
        if (
            not scope.reverse
            or worker.is_kv_consumer
            or getattr(worker, "sender_loop", None) is None
            or worker.tp_rank != scope.rank
            or type(capacity) is not int
            or not 0 < capacity <= _MAX_RANK_ATTEMPTS
            or not isinstance(ipaddress.ip_address(prefill_host), ipaddress.IPv4Address)
        ):
            raise ValueError("GLM return needs its bounded native D sender route")
        self.worker, self.scope = worker, scope
        self._prefill_host, self._capacity = prefill_host, capacity
        self._routes: dict[RankWindowAttempt, _RankSource] = {}
        self._pool = mooncake_receive_pool(worker.engine, 1)
        try:
            self._handler = GlmWindowControlHandler(
                worker,
                scope,
                self._lookup,
                admit_source=self._open,
                settled=self._settled,
            )
        except BaseException:
            self._pool.close()
            raise

    def call(self, command):
        future = asyncio.run_coroutine_threadsafe(command, self.worker.sender_loop)
        try:
            return future.result(timeout=5)
        except TimeoutError:
            future.cancel()
            raise

    def _require_sender(self) -> None:
        if asyncio.get_running_loop() is not self.worker.sender_loop:
            raise RuntimeError("GLM reverse source belongs to its native sender loop")

    async def register(self, transfer: GlmTerminalReturn, terminal_worker) -> None:
        self._require_sender()
        if (
            type(transfer) is not GlmTerminalReturn
            or transfer.destination.boot_id != self.scope.source_boot_id
            or transfer.lease.identity.source_boot_id != self.scope.decode_boot_id
            or transfer.destination.grant.engine_id != self.worker.engine_id
        ):
            raise ValueError("GLM reverse registration belongs to another deployment")
        attempt = transfer.attempt(self.scope.rank)
        existing = self._routes.get(attempt)
        if existing is not None:
            if (
                existing.transfer != transfer
                or existing.terminal_worker is not terminal_worker
            ):
                raise ValueError("GLM reverse source changed its native terminal owner")
            return
        if len(self._routes) >= self._capacity:
            raise BufferError("GLM native reverse source registry is full")
        source = terminal_worker.terminal_window_source(
            transfer.lease, transfer.permit, transfer.manifest
        )
        self._routes[attempt] = _RankSource(transfer, terminal_worker, source)

    def _lookup(self, attempt: RankWindowAttempt) -> NativeRankWindowWriter | None:
        self._require_sender()
        route = self._routes.get(attempt)
        return route.writer if route is not None else None

    async def _open(self, attempt, destination) -> NativeRankWindowWriter | None:
        self._require_sender()
        route = self._routes.get(attempt)
        if route is None or destination[0] != self._prefill_host:
            return None
        if route.writer is None:
            route.writer = open_fenced_terminal_rank_writer(
                route.transfer,
                route.source,
                self.scope.rank,
                self.worker.engine,
                self._pool,
                destination=destination,
                # Native scheduler remains the authority until a fenced rank receipt.
                release_source=lambda: None,
                timeout=5.0,
            )
        return route.writer

    async def _settled(self, proof: FencedRankWindows) -> None:
        self._require_sender()
        route = self._routes[proof.attempt]
        writer = route.writer
        assert writer is not None
        committed = (
            CommittedRankWindows(proof.attempt, writer.session.consumed_sequence)
            if proof.outcome == "acked"
            else None
        )
        result = DecodeTerminalRankResult(proof, committed)
        if route.result is not None and route.result != result:
            raise ValueError("GLM terminal rank changed its settled receipt")
        route.result = result

    async def result(
        self, attempt: RankWindowAttempt
    ) -> DecodeTerminalRankResult | None:
        self._require_sender()
        route = self._routes.get(attempt)
        if route is None:
            raise ValueError("GLM reverse read has no admitted native source")
        return route.result

    async def retire(self, result: DecodeTerminalRankResult) -> None:
        self._require_sender()
        route = self._routes.get(result.proof.attempt)
        if route is None or route.result != result:
            raise ValueError("GLM reverse source lacks the same fenced receipt")
        del self._routes[result.proof.attempt]

    async def retire_attempt(self, attempt: RankWindowAttempt) -> None:
        self._require_sender()
        route = self._routes.get(attempt)
        if route is not None:
            if route.result is None:
                raise ValueError("GLM reverse route still awaits its native peer fence")
            await self.retire(route.result)

    def close(self) -> None:
        if self._routes:
            raise RuntimeError("GLM reverse source still owns admitted attempts")
        self._pool.close()


@dataclass(frozen=True)
class _ReturnRoute:
    prefill_host: str
    decode_bootstrap: str

    @classmethod
    def from_config(cls, config) -> _ReturnRoute:
        additional = config.additional_config
        options = additional.get(_CONFIG) if isinstance(additional, Mapping) else None
        peer = (
            additional.get("kitepond_glm_window")
            if isinstance(additional, Mapping)
            else None
        )
        if (
            not isinstance(options, Mapping)
            or set(options) != {"prefill_host", "decode_bootstrap"}
            or not isinstance(peer, Mapping)
            or any(
                type(value) is not str or not 0 < len(value) <= 128
                for value in options.values()
            )
        ):
            raise ValueError("GLM return requires an explicit native route")
        try:
            prefill = ipaddress.ip_address(options["prefill_host"])
            bootstrap = urlsplit(options["decode_bootstrap"])
            if (
                not isinstance(prefill, ipaddress.IPv4Address)
                or bootstrap.scheme != "http"
                or bootstrap.hostname != peer.get("decode_host")
                or not bootstrap.port
                or bootstrap.path
                or bootstrap.query
                or bootstrap.fragment
                or bootstrap.username is not None
            ):
                raise ValueError
        except ValueError:
            raise ValueError(
                "GLM return requires its configured native IPv4 peers"
            ) from None
        return cls(str(prefill), options["decode_bootstrap"])


class MooncakeConnector(GlmMooncakeConnector):
    """Select on D to register the reverse namespace before the native listener."""

    def __init__(self, vllm_config, role, kv_cache_config):
        self._return_sources = None
        self._cold_import = None
        self._cold_import_profile = GlmColdImportProfile.from_config(vllm_config)
        options = vllm_config.additional_config
        route = (
            _ReturnRoute.from_config(vllm_config)
            if isinstance(options, Mapping) and _CONFIG in options
            else None
        )
        lease = options.get("kitepond", {}) if isinstance(options, Mapping) else {}
        transfer = vllm_config.kv_transfer_config
        if (
            not isinstance(lease, Mapping)
            or lease.get("stage") != "decode"
            or transfer is None
            or transfer.kv_role != "kv_both"
        ):
            raise ValueError("Native GLM D windows require the explicit kv_both role")
        if route is not None:
            peer = urlsplit(route.decode_bootstrap)
            if get_mooncake_bootstrap_addr(vllm_config) != (peer.hostname, peer.port):
                raise ValueError(
                    "GLM reverse source requires the registered native D route"
                )
        super().__init__(vllm_config, role, kv_cache_config)
        worker = self.connector_worker
        if role is KVConnectorRole.WORKER and route is not None:
            self._return_sources = GlmDecodeTerminalSources(
                worker,
                replace(self.window_scope(), reverse=True),
                route.prefill_host,
                capacity=min(
                    vllm_config.scheduler_config.max_num_seqs, _MAX_RANK_ATTEMPTS
                ),
            )

    def _named_prefill(self, request) -> bool:
        return self._cold_import_profile is not None and bool(
            (request.kv_transfer_params or {}).get("do_remote_prefill")
        )

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if self._named_prefill(request):
            return 0, False
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if not self._named_prefill(request):
            super().update_state_after_alloc(request, blocks, num_external_tokens)

    def request_finished_all_groups(self, request, block_ids):
        if self._named_prefill(request):
            return False, None
        return super().request_finished_all_groups(request, block_ids)

    def bind_cold_import(self, multi, runner, views) -> None:
        profile = self._cold_import_profile
        config = self.connector_worker.vllm_config
        host = config.additional_config.get("kitepond_glm_host")
        if (
            profile is None
            or self._cold_import is not None
            or not isinstance(host, Mapping)
            or host.get("role") != "authority"
            or runner.device.type != "cuda"
        ):
            raise ValueError("Native D cold import requires its initialized GPU owner")
        self._cold_import = GlmDecodeImportRuntime(
            multi,
            runner,
            views,
            self.window_scope(),
            profile,
            p_engine_id=self._profile.p_engine_id,
            model_sha256=host["model_sha256"],
            config_sha256=config_fingerprint(
                config.model_config.hf_text_config.to_dict()
            ),
            main_stream=runner.main_stream,
            receive_stream=torch.cuda.Stream(device=runner.device),
            capacity=config.scheduler_config.max_num_seqs,
        )

    def start_load_kv(self, forward_context, **kwargs):
        if self._cold_import_profile is not None:
            if self._cold_import is None:
                raise ValueError("Native D cold import is not bound to its worker")
            native = self._get_connector_metadata()
            if any(native.reqs_to_recv.values()):
                raise ValueError("Named D import cannot run a raw Mooncake pull")
            self._cold_import.bind_step(
                self._cold_import.hisparse_metadata(),
                kwargs.get("request_ids", ()),
                getattr(kwargs.get("scheduler_output"), "finished_req_ids", ()),
            )
        super().start_load_kv(forward_context, **kwargs)

    def get_transfer_results(self, finished_req_ids):
        if self._cold_import is not None:
            self._cold_import.retire_finished(finished_req_ids)
        return super().get_transfer_results(finished_req_ids)

    def return_sources(self) -> GlmDecodeTerminalSources:
        if self._return_sources is None:
            raise RuntimeError("GLM reverse worker source is not initialized")
        return self._return_sources

    def shutdown(self):
        if self._cold_import is not None:
            self._cold_import.close()
            self._cold_import = None
        sources = self._return_sources
        if sources is not None:
            sources.close()
            self._return_sources = None
        super().shutdown()


def _native_pair() -> tuple[MultiConnector, GlmMooncakeConnector]:
    connector = get_kv_transfer_group()
    if (
        type(connector) is not MultiConnector
        or len(connector.sub_connectors) != 2
        or not isinstance(connector.sub_connectors[1], GlmMooncakeConnector)
    ):
        raise ValueError(
            "GLM reverse control needs the configured native Mooncake child"
        )
    return connector, connector.sub_connectors[1]


def bind_native_glm_cold_import(runner, views) -> GlmDecodeImportRuntime:
    connector, mooncake = _native_pair()
    if type(mooncake) is not MooncakeConnector:
        raise ValueError(
            "GLM D cold import requires its explicit native Mooncake child"
        )
    mooncake.bind_cold_import(connector, runner, views)
    assert mooncake._cold_import is not None
    return mooncake._cold_import


def current_decode_terminal_sources() -> GlmDecodeTerminalSources:
    _, mooncake = _native_pair()
    if type(mooncake) is not MooncakeConnector:
        raise ValueError("GLM reverse control D endpoint is not configured")
    return mooncake.return_sources()


class GlmPrefillTerminalDestination:
    """Reuse bounded rank scratch on the installed native P receiver."""

    def __init__(self) -> None:
        connector, mooncake = _native_pair()
        worker = mooncake.connector_worker
        config = worker.vllm_config
        route = _ReturnRoute.from_config(config)
        if (
            mooncake._profile.stage != "prefill"
            or config.kv_transfer_config.kv_role != "kv_both"
            or worker.hostname != route.prefill_host
        ):
            raise ValueError("GLM reverse destination is another native P role")
        self._worker, self._route = worker, route
        self._scope = replace(mooncake.window_scope(), reverse=True)
        self._windows = GlmNativeTerminalReturnWindows(connector, slots=2)
        self._max_active = config.scheduler_config.max_num_seqs
        self._active: set[RankWindowAttempt] = set()

    def receive(
        self,
        transfer: GlmTerminalReturn,
        views,
        cache,
        layer_groups,
        stream,
        observe,
        *,
        timeout: float,
    ):
        """Submit an already native-admitted P import to its current receiver loop."""
        if type(timeout) not in (int, float) or not 0 < timeout <= 60:
            raise ValueError("GLM reverse read requires a bounded timeout")
        attempt = transfer.attempt(self._scope.rank)
        self._scope.check_attempt(attempt)

        async def receive_bound():
            if attempt in self._active or len(self._active) >= self._max_active:
                raise BufferError("GLM native reverse receiver already active or full")
            self._active.add(attempt)
            try:
                async with connect_glm_window_peer(
                    self._worker,
                    transfer.lease.identity.source_engine_id,
                    self._route.decode_bootstrap,
                    self._scope,
                    discovery_timeout=timeout,
                ) as peer:
                    await self._windows.receive_at_prefill(
                        transfer,
                        cache,
                        layer_groups,
                        peer,
                        lambda *_: None,
                        native_views=views,
                        stream=stream,
                        timeout=timeout,
                        reverse_peer=peer,
                        before_route=observe,
                    )
            finally:
                self._active.remove(attempt)

        return asyncio.run_coroutine_threadsafe(
            receive_bound(), self._worker.receiver_loop
        )

    def close(self) -> None:
        async def retire():
            if self._active:
                raise RuntimeError("GLM reverse receiver still owns active routes")
            self._windows.close()

        asyncio.run_coroutine_threadsafe(retire(), self._worker.receiver_loop).result(
            timeout=5
        )
