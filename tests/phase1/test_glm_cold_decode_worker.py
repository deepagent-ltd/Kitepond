# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the native D worker's cold ticket, real named route and readback."""

import multiprocessing
import secrets
from dataclasses import replace
from multiprocessing.reduction import DupFd
from types import SimpleNamespace

import pytest
import torch
from kitepond.integrations.mooncake.live_receive import RankWindowAttempt
from kitepond.integrations.mooncake.named_control import GlmWindowControlScope
from kitepond.integrations.mooncake.window import manifest_digest
from kitepond.integrations.vllm.decode_receive import (
    DecodeReceiveMailbox,
    DecodeReceiveTicket,
    DecodeReceiveWork,
)
from kitepond.integrations.vllm.glm_decode_bootstrap import (
    GlmColdImportProfile,
    GlmDecodeImportRuntime,
)
from kitepond.integrations.vllm.glm_native_views import (
    GlmNativeStateViews,
    _NativeLayer,
)
from kitepond.integrations.vllm.glm_return_bootstrap import MooncakeConnector
from kitepond.integrations.vllm.hisparse_attachment import HiSparseConnector
from kitepond.models.glm5next.cold_restore import ColdRestoreLayout
from kitepond.models.glm5next.import_recipe import part_identity
from kitepond.models.glm5next.layout import CacheRole
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion
from test_glm_cold_restore import _live_model_plan
from test_glm_named_control import _worker
from test_glm_reverse_tcp import _native, _source

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorMetadata,
    MooncakeConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector


@pytest.mark.parametrize("flag", [True, 1])
def test_raw_native_pull_only_for_unselected_default_profile(flag):
    def create(config, cache):
        return MooncakeConnectorScheduler(config, "D", cache)

    alias = object.__new__(MooncakeConnector)
    alias.connector_scheduler = _worker("kv_both", 2, create=create)
    params = {
        "do_remote_prefill": flag,
        "remote_engine_id": "P",
        "remote_bootstrap_addr": "http://127.0.0.1:8151",
        "transfer_id": "cold",
    }
    request = SimpleNamespace(
        kv_transfer_params=params,
        request_id="native",
        prompt_token_ids=list(range(12)),
        status=None,
    )
    alias._cold_import_profile = GlmColdImportProfile(
        params["remote_bootstrap_addr"], 4
    )
    alias._cold_import = None
    assert alias.get_num_new_matched_tokens(request, 0) == (0, False)
    alias.update_state_after_alloc(request, None, 0)
    assert alias.request_finished_all_groups(request, ([],)) == (False, None)
    assert request.kv_transfer_params["do_remote_prefill"]
    assert not alias.connector_scheduler._reqs_need_recv
    with pytest.raises(ValueError, match="not bound"):
        alias.start_load_kv(None)
    alias._cold_import_profile = None
    assert alias.get_num_new_matched_tokens(request, 0)[0] > 0
    alias.update_state_after_alloc(request, None, 0)
    assert request.request_id in alias.connector_scheduler._reqs_need_recv


def test_cold_worker_imports_over_native_mooncake_to_real_shared_pages(monkeypatch):
    pytest.importorskip("mooncake.engine", exc_type=ImportError)
    pytest.importorskip(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.named_control"
    )
    cache = _live_model_plan(120).final
    layout = replace(cache.layout, committed_kda_slot=0)
    rank = cache.rank
    scope = GlmWindowControlScope("P-boot", "D-boot", rank, secrets.token_bytes(32))
    mla = next(
        layer
        for layer in cache.layers.values()
        if layer.identity.cache_role is CacheRole.MLA
    )
    scope_args = dict(
        authority_engine_id="D",
        authority_boot_id=scope.decode_boot_id,
        source_rank=rank,
        layout_sha256="a" * 64,
    )
    shared = NativeHiSparseMemfdRegion.create(mla.tensor.numel(), **scope_args)
    tensor = shared.tensor().view(mla.tensor.shape)
    source = _live_model_plan(40).final
    for layer in source.layers.values():
        if layer.identity.cache_role is CacheRole.MLA:
            tensor.copy_(layer.tensor)
    target = ColdRestoreLayout(
        layout,
        rank,
        [
            replace(layer, tensor=tensor)
            if layer.identity.cache_role is CacheRole.MLA
            else layer
            for layer in cache.layers.values()
        ],
    )
    views = object.__new__(GlmNativeStateViews)
    views.rank = rank
    views.descriptor = shared.descriptor
    views._groups = len(target.layers)
    views._layers = {
        name: _NativeLayer(group_id, layer.identity, layer.tensor)
        for group_id, (name, layer) in enumerate(target.layers.items())
    }
    tail = next(
        layer
        for layer in target.layers.values()
        if layer.identity.cache_role is CacheRole.KPOOL_TAIL
    )
    views._tail_positions = {
        tail.identity.layer_name: torch.full((4, 8), -99, dtype=torch.int32)
    }
    views._tail_attempts = {}
    ticket = DecodeReceiveTicket(
        layout.request_id,
        "cold-rank",
        "D",
        "D-boot",
        layout.epoch,
        "a" * 64,
        layout.target_valid_tokens,
        tuple(layer.block_ids for layer in target.layers.values()),
        0,
        None,
    )
    mailbox = DecodeReceiveMailbox(rank, scope.decode_boot_id)
    metadata = DecodeReceiveWork(None, (), (), {}, False, {}, (ticket,))
    owner = object.__new__(HiSparseConnector)
    owner._role = KVConnectorRole.WORKER
    owner.stage = "decode"
    owner._decode_mailbox = mailbox
    owner.bind_connector_metadata(metadata)
    mailbox.bind((ticket,))

    alias = object.__new__(MooncakeConnector)
    alias._role = KVConnectorRole.WORKER
    alias._k0_enabled = False
    alias.connector_worker = _native(scope, "D")
    alias.bind_connector_metadata(MooncakeConnectorMetadata())
    native = alias.connector_worker
    pair = object.__new__(MultiConnector)
    pair._role = KVConnectorRole.WORKER
    pair._connectors = [owner, alias]
    received_state, retired_state, written = [], [], []
    runner = SimpleNamespace(
        req_states=SimpleNamespace(req_id_to_index={ticket.request_id: 3}),
        start_imported_frontier=lambda *args: received_state.append(args),
        drop_imported_frontier=retired_state.append,
    )

    original_scatter = ColdRestoreLayout.scatter_window

    def observe_scatter(native_cache, parts, payload):
        roles = {part_identity(part.role).cache_role for part in parts}
        assert CacheRole.MLA not in roles
        written.extend(roles)
        return original_scatter(native_cache, parts, payload)

    monkeypatch.setattr(ColdRestoreLayout, "scatter_window", observe_scatter)
    ctx = multiprocessing.get_context("spawn")
    pipe, source_pipe = ctx.Pipe()
    child = ctx.Process(
        target=_source,
        args=(source_pipe, scope, False, (shared.descriptor, DupFd(shared.fd))),
    )
    child.start()
    source_pipe.close()

    def recv():
        assert pipe.poll(35), "Native cold worker did not respond"
        return pipe.recv()

    try:
        profile = GlmColdImportProfile(recv(), 5)
        runtime = GlmDecodeImportRuntime(
            pair,
            runner,
            views,
            scope,
            profile,
            p_engine_id="P",
            model_sha256=layout.model_fingerprint,
            config_sha256=layout.config_fingerprint,
            main_stream=None,
            receive_stream=None,
            capacity=2,
        )
        alias._cold_import_profile = profile
        alias._cold_import = runtime
        plan, _ = runtime._plan(ticket)
        attempt = RankWindowAttempt(
            ticket.request_id,
            ticket.transfer_id,
            ticket.epoch,
            rank,
            scope.source_boot_id,
            ticket.grant_sha256,
            ticket.destination_sha256,
            manifest_digest(plan.manifest()),
        )
        pipe.send(
            (
                attempt,
                (native.hostname, native.rpc_port, runtime._receiver._pool.pool_id),
            )
        )
        assert recv() == "registered"
        alias.start_load_kv(None)
        alias.start_load_kv(None)
        runtime._imports[ticket.request_id].future.result(timeout=20)
        (proof,) = mailbox.drain(None).receipts
        assert (proof.rank, proof.outcome) == (rank, "loaded")
        assert proof.destination_sha256 == ticket.destination_sha256
        assert not received_state
        assert set(written) == {
            CacheRole.INDEXER,
            CacheRole.KDA_CONV,
            CacheRole.KDA_RECURRENT,
            CacheRole.KPOOL_TAIL,
        }
        assert native._remote_agents["P"][rank][0].startswith("tcp://")
        assert torch.equal(
            tensor,
            next(
                layer.tensor
                for layer in source.layers.values()
                if layer.identity.cache_role is CacheRole.MLA
            ),
        )
        stamps = [-1] * 8
        for position in range(ticket.imported_tokens // 4 * 4, ticket.imported_tokens):
            stamps[position % 8] = position
        assert (
            views._tail_positions[tail.identity.layer_name][tail.block_ids[0]].tolist()
            == stamps
        )
        mailbox.bind(())
        owner.bind_connector_metadata(replace(metadata, tickets=()))
        alias.start_load_kv(None, request_ids=[ticket.request_id])
        assert received_state == [(ticket.request_id, ticket.imported_tokens, 3)]
        runtime.retire_finished({ticket.request_id})
        assert retired_state == [ticket.request_id]
        runtime.close()
        pipe.send("done")
        settled, closed, transferred, cuda = recv()
        assert settled == [] and closed == [rank] and sum(transferred) > 0 and not cuda
        assert not torch.cuda.is_initialized()
        child.join(timeout=10)
        assert child.exitcode == 0
    finally:
        native.shutdown()
        native.receiver_loop.close()
        native.sender_loop.close()
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
        pipe.close()
        shared.close()
