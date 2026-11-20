# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the actual native MTP K0 entry and connector save ordering."""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from kitepond.core.readiness import ReadyAttempt
from kitepond.integrations.vllm.glm_k0_routes import (
    GlmK0Route,
    GlmK0SchedulerRoutes,
)
from kitepond.integrations.vllm.glm_k0_runtime import GlmK0PrefillRuntime
from kitepond.integrations.vllm.glm_k0_worker import GlmK0MTP, GlmK0PrefillWorker
from kitepond.integrations.vllm.host_authority import NativeHostGrant
from kitepond.integrations.vllm.prefill_fence import NativePrefillSaveFence
from kitepond.integrations.vllm.ready_transport import (
    ReadyCollector,
    ReadyPublisher,
    ReadyWorker,
)
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion
from test_glm_cold_restore import _cache

from vllm import SamplingParams
from vllm.config import CUDAGraphMode
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
    HiSparseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorMetadata,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar import speculator as stock
from vllm.v1.worker.gpu.states import RequestState


def _request():
    return Request(
        request_id="prefill",
        prompt_token_ids=[11, 22, 33, 44, 55, 777],
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={
                "kv_transfer_params": {
                    "do_remote_decode": True,
                    "transfer_id": "handoff",
                }
            },
        ),
        pooling_params=None,
    )


def _cpu_batch(start, target):
    count = len(target)
    return SimpleNamespace(
        req_ids=["prefill"],
        num_reqs=1,
        num_draft_tokens=0,
        idx_mapping=torch.tensor([2], dtype=torch.int32),
        idx_mapping_np=np.array([2], dtype=np.int64),
        num_computed_tokens_np=np.array([start], dtype=np.int32),
        num_scheduled_tokens=np.array([count], dtype=np.int32),
        prefill_len_np=np.array([5], dtype=np.int32),
        is_prefilling_np=np.array([True]),
        num_tokens=count,
        num_tokens_after_padding=count,
        decode_graph_eligible=False,
        seq_lens_cpu_upper_bound=torch.tensor([start + count]),
        positions=torch.arange(start, start + count, dtype=torch.int64),
        input_ids=torch.tensor(target, dtype=torch.int32),
        query_start_loc_np=np.array([0, count], dtype=np.int32),
    )


@pytest.mark.parametrize("imported_tokens", [0, 3])
def test_native_k0_shift_then_fenced_ready(tmp_path, imported_tokens):
    traces = []
    request = _request()
    grant = NativeHostGrant(
        "decode", "boot", "prefill", "handoff", 4, 0, 5, (), (2,), None
    )
    identity = ReadyAttempt(
        request.request_id, grant.transfer_id, grant.epoch, grant.sha256
    )
    region = NativeHiSparseMemfdRegion.create(
        4096,
        authority_engine_id="decode",
        authority_boot_id="boot",
        source_rank=0,
        layout_sha256="f" * 64,
    )
    directory = tmp_path / "ready"
    directory.mkdir(mode=0o700)
    collector = ReadyCollector(directory, {0: ReadyWorker.from_pid(os.getpid())})
    publisher = ReadyPublisher(directory)
    try:
        ticket = collector.arm(identity, {0: region.descriptor.fingerprint()}, 5)
        route = GlmK0Route.from_native_request(
            request, grant, ticket, imported_tokens=imported_tokens
        )
        assert route.next_token == 777

        # Use stock one-token withholding and the native SchedulerOutput DTO.
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
            mooncake_connector,
        )

        native_owner = mooncake_connector.MooncakeConnectorScheduler.__new__(
            mooncake_connector.MooncakeConnectorScheduler
        )
        native_owner._truncate_mamba_request_for_prefill(request)
        route.verify_truncation(request)
        assert request.prompt_token_ids == [11, 22, 33, 44, 55]

        routes = GlmK0SchedulerRoutes()
        routes.stage(request, route)
        runtime = GlmK0PrefillRuntime(
            "a" * 64,
            "b" * 64,
            record_event=lambda _: SimpleNamespace(
                synchronize=lambda: traces.append("draft_fence")
            ),
        )
        runtime.bind_region(region.descriptor)
        verified_imports = []
        if imported_tokens:
            runtime.bind_warm_import_verifier(verified_imports.append)
        worker = GlmK0PrefillWorker.__new__(GlmK0PrefillWorker)
        with pytest.raises(ValueError, match="unavailable before native startup"):
            worker.kitepond_prefill_ready_identity()
        worker._kitepond_ready_views = SimpleNamespace(
            rank=region.descriptor.source_rank, descriptor=region.descriptor
        )
        assert worker.kitepond_prefill_ready_identity() == {
            "rank": 0,
            "pid": os.getpid(),
            "region_sha256": region.descriptor.fingerprint(),
        }
        native = GlmK0MTP.__new__(GlmK0MTP)
        native._kitepond_k0 = runtime
        native.num_speculative_steps = 3
        native.max_model_len = 64
        native.max_num_reqs = 4
        native.pcp_manager = None
        native.dp_size = 1
        native.dp_rank = 0
        native.prefill_cudagraph_manager = None
        native.input_buffers = SimpleNamespace(
            input_ids=torch.zeros(8, dtype=torch.int32),
            positions=torch.zeros(8, dtype=torch.int64),
        )
        native.hidden_states = torch.zeros(8, 2)
        native.last_token_indices = torch.zeros(4, dtype=torch.int32)
        native.current_draft_step = torch.zeros(1, dtype=torch.int32)
        native.draft_tokens = torch.zeros(4, 3, dtype=torch.int32)
        native._copy_request_inputs = lambda *args, **kwargs: None
        native._prepare_eplb_forward = lambda _: None
        native.on_prefill_begin = lambda _: None
        native.on_prefill_end = lambda _: None
        native._prefill = lambda *_, skip_sample, **kw: traces.append(
            ("propose", skip_sample)
        )

        def cpu_input_kernel(
            indices, _, buffers, batch, sampled, rejected, last, nexts, limit
        ):
            count = len(batch.input_ids)
            indices[0] = count - 1
            buffers.positions[:count] = batch.positions
            buffers.input_ids[: count - 1] = batch.input_ids[1:]
            slot = batch.idx_mapping_np[0]
            assert last.shape == (4, 1) and nexts.shape == (2, 4)
            assert last.dtype == torch.int64 and nexts.dtype == torch.int32
            assert nexts[1].tolist() == [10, 20, 30, 40]
            if sampled[0]:
                assert last[slot, 0] == nexts[0, slot] == 777
            token_view = last if sampled[0] else nexts
            buffers.input_ids[count - 1] = token_view.flatten()[slot]

        host = HiSparseConnectorWorker.__new__(HiSparseConnectorWorker)
        host.is_host_writer = True
        host._draft_mirror_pending = True
        host._draft_layers = (3,)
        host._post_forward_transfers = []
        host._dma_submitted = False
        host._enqueue_row_dma = lambda _: setattr(host, "_dma_submitted", True)
        host._submit_transfers = lambda _: None
        host.host_write_event = SimpleNamespace(
            synchronize=lambda: traces.append("host_fence")
        )
        saved = []
        fence = NativePrefillSaveFence(
            host,
            publisher,
            runtime.current_grant,
            runtime,
            local_rank=0,
            record_source=lambda records: saved.extend(
                runtime.export(write, receipt) for write, receipt in records
            ),
        )
        native_meta = HiSparseConnectorMetadata(
            None, (), (), {request.request_id: (object(),)}, True, {}
        )
        active = ActiveKVConnector.__new__(ActiveKVConnector)
        active._disabled = False
        active.kv_connector = SimpleNamespace(
            wait_for_save=lambda: fence.publish(native_meta, runtime.collect_writes()),
            clear_connector_metadata=lambda: traces.append("native_finalize"),
        )
        # Instantiate the installed runtime's allocation path. Only the
        # unrelated CUDA-only UVA storage is replaced on this CPU fixture.
        with (
            patch("vllm.v1.worker.gpu.states.UvaBackedTensor"),
            patch("vllm.v1.worker.gpu.states.StagedWriteTensor"),
        ):
            states = RequestState(4, 64, 8, 3, 8192, torch.device("cpu"), 2)
        last_sampled = states.last_sampled_tokens
        next_prefill = states.next_prefill_tokens
        last_sampled[2, 0] = 5000
        next_prefill[0, 2] = 44
        next_prefill[1] = torch.tensor([10, 20, 30, 40])

        with (
            patch(
                "kitepond.integrations.vllm.glm_k0_worker.has_kv_transfer_group",
                return_value=True,
            ),
            patch.object(stock, "prepare_prefill_inputs", cpu_input_kernel),
            patch.object(
                stock,
                "dispatch_cg_and_sync_dp",
                side_effect=lambda *args, **kw: (
                    SimpleNamespace(cg_mode=CUDAGraphMode.NONE, num_tokens=args[2]),
                    None,
                ),
            ),
        ):
            steps = ((3, [44, 55]),)
            if not imported_tokens:
                steps = ((0, [11, 22, 33]), *steps)
            for start, values in steps:
                step = SchedulerOutput.make_empty()
                step.scheduled_new_reqs = [
                    SimpleNamespace(
                        req_id=request.request_id, num_computed_tokens=start
                    )
                ]
                step.num_scheduled_tokens = {request.request_id: len(values)}
                runtime.bind_step(routes.build(step, MooncakeConnectorMetadata()))
                assert step.num_spec_tokens_to_schedule == 0 and step.skip_drafts
                batch = _cpu_batch(start, values)
                result = native.propose(
                    batch,
                    {
                        "target.kda": SimpleNamespace(
                            non_spec_state_indices_tensor=torch.tensor([7]),
                            spec_sequence_masks=None,
                        )
                    },
                    {},
                    torch.zeros(len(values), 2),
                    None,
                    torch.tensor([int(start > 0)]),
                    torch.tensor([0]),
                    last_sampled,
                    next_prefill,
                    torch.zeros(4),
                    torch.zeros(4, dtype=torch.long),
                    num_speculative_tokens=step.num_spec_tokens_to_schedule,
                )
                assert result.tolist() == [[-1, -1, -1]]
                assert last_sampled[2, 0] == 5000 and next_prefill[0, 2] == 44
                assert next_prefill[1].tolist() == [10, 20, 30, 40]
                before = len(saved)
                if start == 0:
                    assert native.input_buffers.input_ids[: len(values)].tolist() == [
                        22,
                        33,
                        44,
                    ]
                    with pytest.raises(ValueError, match="no completed final"):
                        runtime.wait_ready(request.request_id)
                else:
                    assert native.input_buffers.input_ids[: len(values)].tolist() == [
                        55,
                        777,
                    ]
                assert len(saved) == before and collector.take(ticket) is None
                active.finalize_step()

        (receipt,) = collector.take(ticket)
        assert receipt.target_valid_tokens == receipt.draft_valid_tokens == 5
        (export,) = saved
        assert export.layout.target_valid_tokens == 5
        assert export.native_kda_blocks == (("target.kda", 7),)
        assert verified_imports == ([route] if imported_tokens else [])
        expected = [("propose", True), "host_fence", "draft_fence", "native_finalize"]
        if not imported_tokens:
            expected = [("propose", True), "native_finalize", *expected]
        assert traces == expected
    finally:
        publisher.close()
        collector.close()
        region.close()


def test_native_k0_export_tail_scrubs_only_unused_scratch_rows():
    from kitepond.integrations.mooncake.window import plan_windows
    from kitepond.integrations.vllm.glm_native_views import (
        GlmNativeStateViews,
        _NativeLayer,
    )
    from kitepond.integrations.vllm.glm_prefill_sources import _canonicalize_window_tail
    from kitepond.models.glm5next.import_recipe import part_identity
    from kitepond.models.glm5next.layout import CacheRole
    from kitepond.models.glm5next.live_windows import GlmRankWindowPlan

    base = _cache()
    cache = type(base)(
        replace(base.layout, committed_kda_slot=0),
        base.rank,
        tuple(base.layers.values()),
    )
    views = GlmNativeStateViews.__new__(GlmNativeStateViews)
    views.rank = cache.rank
    views._layers = {
        name: _NativeLayer(i, layer.identity, layer.tensor)
        for i, (name, layer) in enumerate(cache.layers.items())
    }
    views._tail_positions = {}
    selected = {}
    for layer in cache.layers.values():
        identity = layer.identity
        if identity.cache_role in (CacheRole.KDA_CONV, CacheRole.KDA_RECURRENT):
            selected[identity.layer_name] = layer.block_ids[0]
        if identity.cache_role is CacheRole.KPOOL_TAIL:
            layer.tensor.view(torch.uint8).fill_(0xA5)
            rows = torch.full((layer.tensor.shape[0], 8), -1, dtype=torch.int32)
            for pos in range(1056, 1059):
                rows[layer.block_ids[0], pos % 8] = pos
            views._tail_positions[identity.layer_name] = rows
    views.verify_k0_prefill_export(cache, tuple(selected.items()))
    canonical = views.verify_k0_prefill_export(
        cache, tuple((name, 2) for name in selected)
    )
    for name, layer in canonical.layers.items():
        if layer.identity.cache_role in (CacheRole.KDA_CONV, CacheRole.KDA_RECURRENT):
            assert layer.block_ids == (2,) and layer.tensor is cache.layers[name].tensor
    with pytest.raises(ValueError, match="selected native KDA block"):
        views.verify_k0_prefill_export(
            cache, tuple((key, val + 1) for key, val in selected.items())
        )
    plan = GlmRankWindowPlan(cache, 0)
    for window in plan_windows(plan.manifest()):
        slab = torch.empty(window.nbytes, dtype=torch.uint8)
        plan.pack(window, slab)
        _canonicalize_window_tail(window, slab, cache.layout.target_valid_tokens)
        offset = 0
        for entry in window.entries:
            end = offset + entry["nbytes"]
            if part_identity(entry["role"]).cache_role is CacheRole.KPOOL_TAIL:
                actual = slab[offset:end].view(8, -1)
                assert torch.all(actual[:3] == 0xA5)
                assert torch.all(actual[3:] == 0)
            offset = end
    for stamps in views._tail_positions.values():
        stamps.fill_(-1)
    with pytest.raises(ValueError, match="were not written"):
        views.verify_k0_prefill_export(cache, tuple(selected.items()))
