# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import (
    worker as worker_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnector,
    HiSparseConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
    HiSparseConnectorWorker,
)
from vllm.v1.hisparse import runtime as runtime_module
from vllm.v1.hisparse.coordinator import HiSparseCoordinator
from vllm.v1.hisparse.runtime import HiSparseCacheHandle
from vllm.v1.hisparse.types import (
    SparseKVOffloadCommand,
    SparseKVPageTransfer,
)
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as autoregressive
from vllm.v1.worker.gpu.spec_decode.mtp import speculator as mtp
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar import speculator as target_ar



class _FakeEvent:
    log: list[tuple[str, object]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def record(self, stream=None) -> None:
        self.log.append(("event", self))

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("accepted", range(4))
def test_dcp8_draft_mirror_covers_draft_owned_rows(monkeypatch, rank, accepted):
    """Mirror each layer's written rows from the allocated scheduler envelope."""
    block_size, global_page = 256, 2048
    start_token = global_page - 3
    target_positions = range(start_token, start_token + 4)
    lookahead_positions = range(start_token, start_token + 7)
    resident_pages = [(1, 2), (3, 4)]
    host_pages = (2, 5)
    log: list[tuple[str, object]] = []
    monkeypatch.setattr(_FakeEvent, "log", log)

    def swap_blocks_batch(src, dst, sizes):
        for src_ptr, dst_ptr, size in zip(src.tolist(), dst.tolist(), sizes.tolist()):
            ctypes.memmove(dst_ptr, src_ptr, size)
            log.append(("dma", dst_ptr))

    compute_stream = MagicMock()
    monkeypatch.setattr(torch, "Event", _FakeEvent)
    monkeypatch.setattr(torch, "Stream", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(torch.accelerator, "current_stream", lambda: compute_stream)
    empty = torch.empty
    monkeypatch.setattr(
        torch, "empty", lambda *args, pin_memory=False, **kwargs: empty(*args, **kwargs)
    )
    monkeypatch.setattr(
        runtime_module,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL),
    )

    def slot(position, pages):
        page, offset = divmod(position, global_page)
        stripe, within_stripe = divmod(offset, 32)
        owner, within_rank = divmod(within_stripe, 4)
        return (
            pages[page] * block_size + stripe * 4 + within_rank if owner == rank else -1
        )

    request_state_indices = torch.full((1,), -1, dtype=torch.int32)
    handles = []
    for group, pages in enumerate(resident_pages):
        runtime = SimpleNamespace(
            is_group_leader=True,
            eager_host_mirror=True,
            resident_source_index=group,
            host_cache=torch.full((6 * block_size, 1), -1, dtype=torch.int32),
            request_state_indices=request_state_indices,
            begin_forward=MagicMock(),
            max_swap_rows=64,
        )
        handle = HiSparseCacheHandle(runtime)
        handle.view = SimpleNamespace(
            block_size=block_size,
            cache=torch.full((6, block_size, 1), -2, dtype=torch.int32),
        )
        handle.slot_mapping = torch.tensor(
            [slot(pos, pages) for pos in target_positions]
        )
        handle.draft_layer = group == 1
        handles.append(handle)
    target, draft = handles
    for handle in handles:
        handle.mirror_slot_mapping = target.slot_mapping

    backend = SimpleNamespace(swap_blocks_batch=swap_blocks_batch)
    monkeypatch.setattr(worker_module, "get_hisparse_backend", lambda: backend)
    worker = HiSparseConnectorWorker(
        SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
            num_lookahead_tokens=3,
        ),
        MagicMock(),
    )
    worker._dcp_mirror_geometry = (8, rank, 4)
    worker.initialize(handles, ["target", "draft"], 1, 6, torch.device("cpu"), [])
    worker._invalidate_written_slots = MagicMock()
    # An unrelated aged full page can be handed back this step. The pages in
    # the mirror envelope are still held by the active request's tail.
    transfer = SparseKVPageTransfer(
        transfer_id=7, host_block_id=1, resident_block_ids=(5, 5)
    )
    coordinator = object.__new__(HiSparseCoordinator)
    coordinator.resident_managers = [
        SimpleNamespace(
            block_size=global_page,
            req_to_blocks={"req": pages},
            get_resident_page=lambda req_id, page, pages=pages: SimpleNamespace(
                block_id=pages[page]
            ),
        )
        for pages in resident_pages
    ]
    coordinator.host_manager = SimpleNamespace(
        req_to_blocks={
            "req": [
                SimpleNamespace(block_id=page, is_null=False) for page in host_pages
            ]
        }
    )
    envelope = coordinator.build_row_mirrors(
        (("req", start_token, len(lookahead_positions)),)
    )
    connector = object.__new__(HiSparseConnector)
    connector.connector_worker = worker
    connector._get_connector_metadata = MagicMock(
        return_value=HiSparseConnectorMetadata(
            SparseKVOffloadCommand([transfer]),
            (),
            (),
            {"req": tuple(envelope)},
            True,
            {},
        )
    )
    attn_metadata = SimpleNamespace(
        num_actual_tokens=4,
        num_decode_tokens=4,
        num_reqs=1,
        max_query_len=4,
        req_id_per_token=torch.zeros(4, dtype=torch.int32),
    )
    connector.start_load_kv(
        SimpleNamespace(),
        request_ids=["req"],
        num_tokens=4,
        attn_metadata={"target": attn_metadata},
    )

    def write(handle, positions, value, expected):
        for pos, row in zip(positions, handle.slot_mapping.tolist()):
            if row >= 0:
                handle.view.cache.flatten()[row] = pos + value
                expected[slot(pos, host_pages)] = pos + value

    expected_target = torch.full_like(target.runtime.host_cache, -1)
    expected_draft = torch.full_like(draft.runtime.host_cache, -1)
    write(target, target_positions, 10000, expected_target)
    connector.finish_forward()
    assert worker.take_transfer_updates() == ([], [])
    assert torch.all(draft.runtime.host_cache == -1)

    # Exercise the native MTP proposal through its FULL replay dispatch.
    # CPU substitutes the GPU kernels, graph bodies and graph dispatch only.
    speculator = object.__new__(mtp.MTPSpeculator)
    positions = torch.tensor([start_token + accepted + 1])
    speculator.__dict__.update(
        vllm_config=SimpleNamespace(
            attention_config=SimpleNamespace(
                hisparse_config=SimpleNamespace(enable_dcp8=True)
            ),
            parallel_config=SimpleNamespace(),
            model_config=SimpleNamespace(max_model_len=1 << 20),
        ),
        dp_size=1,
        dp_rank=0,
        max_num_reqs=1,
        num_speculative_steps=3,
        pcp_manager=None,
        supports_decode_metadata_update=False,
        hidden_states=torch.empty(4, 1),
        last_token_indices=torch.zeros(1, dtype=torch.int32),
        current_draft_step=torch.tensor(0),
        draft_tokens=torch.zeros(1, 3, dtype=torch.int64),
        input_buffers=SimpleNamespace(
            positions=positions,
            query_start_loc=torch.arange(2),
        ),
        idx_mapping=torch.tensor([0]),
        kv_cache_config=None,
    )

    def compute_draft_slots(idx_mapping, query_start_loc, positions, num_tokens):
        position = int(positions[0])
        for handle, pages in zip(handles, resident_pages, strict=True):
            handle.slot_mapping.fill_(-1)
            handle.slot_mapping[0] = slot(position, pages)
        return None

    def replay_draft_prefill(batch_desc):
        write(draft, target_positions, 20000, expected_draft)

    def replay_draft_decode(batch_desc):
        step = int(speculator.current_draft_step)
        write(draft, (int(positions[0]),), 20000 + step, expected_draft)
        positions.add_(1)

    speculator.block_tables = SimpleNamespace(
        compute_slot_mappings=compute_draft_slots,
    )
    speculator.prefill_cudagraph_manager = SimpleNamespace(
        run_fullgraph=replay_draft_prefill,
    )
    speculator.decode_cudagraph_manager = SimpleNamespace(
        specialize_spec_tokens=lambda batch_desc, steps: batch_desc,
        run_fullgraph=replay_draft_decode,
    )
    for name in (
        "_copy_request_inputs",
        "_prepare_eplb_forward",
        "_prepare_decode_inputs",
        "_build_uniform_attn_metadata",
    ):
        monkeypatch.setattr(speculator, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        speculator, "_build_drafter_attn_metadata", lambda *args, **kwargs: ({}, {})
    )

    def dispatch(manager, num_reqs, num_tokens, *args, **kwargs):
        return BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens, num_reqs), None

    monkeypatch.setattr(target_ar, "prepare_prefill_inputs", lambda *args: None)
    monkeypatch.setattr(target_ar, "dispatch_cg_and_sync_dp", dispatch)
    monkeypatch.setattr(autoregressive, "dispatch_cg_and_sync_dp", dispatch)
    monkeypatch.setattr(
        autoregressive, "build_slot_mappings_by_layer", lambda *args: {}
    )
    # The old source has no graph staging hook. Preserve its native replay
    # in the same worker test so the only red path is the missing host row.
    monkeypatch.setattr(mtp, "has_kv_transfer_group", lambda: True, raising=False)
    monkeypatch.setattr(mtp, "get_kv_transfer_group", lambda: connector, raising=False)
    batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=4,
        num_tokens_after_padding=4,
        num_scheduled_tokens=torch.tensor([4]).numpy(),
        seq_lens_cpu_upper_bound=torch.tensor([start_token + 4]),
        idx_mapping=torch.tensor([0]),
        decode_graph_eligible=True,
    )
    speculator.propose(
        batch,
        {},
        {},
        torch.zeros_like(speculator.hidden_states),
        None,
        num_sampled=torch.tensor([accepted + 1]),
        num_rejected=torch.tensor([3 - accepted]),
        last_sampled=None,
        next_prefill_tokens=None,
        temperature=None,
        seeds=None,
    )

    compute_stream.wait_event.reset_mock()
    connector._get_connector_metadata.return_value = HiSparseConnectorMetadata(
        None, (), (), {}, True, {}
    )
    connector.start_load_kv(SimpleNamespace(), request_ids=[], num_tokens=0)
    ((completion_event, _),) = worker._pending_transfer_events
    worker_meta = connector.build_connector_worker_meta()

    torch.testing.assert_close(target.runtime.host_cache, expected_target)
    torch.testing.assert_close(draft.runtime.host_cache, expected_draft)
    draft_host = draft.runtime.host_cache
    draft_host_range = range(
        draft_host.data_ptr(),
        draft_host.data_ptr() + draft_host.numel() * draft_host.element_size(),
    )
    draft_copies = [
        index
        for index, (kind, ptr) in enumerate(log)
        if kind == "dma" and ptr in draft_host_range
    ]
    if draft_copies:
        assert max(draft_copies) < log.index(("event", completion_event))
        draft_copy_event = next(
            event for kind, event in log[max(draft_copies) :] if kind == "event"
        )
        assert compute_stream.wait_event.call_args_list[0].args == (draft_copy_event,)
    assert worker_meta is not None
    assert worker_meta.enqueued_transfer_counts == {7: 1}
    assert worker_meta.completed_transfer_counts == {7: 1}

    # A second metadata handoff can reuse resident storage. It must not
    # replay the previous mapping or acknowledge the previous page again.
    connector.finish_forward()
    draft.view.cache.fill_(-3)
    connector.start_load_kv(SimpleNamespace(), request_ids=[], num_tokens=0)
    assert worker.take_transfer_updates() == ([], [])
    torch.testing.assert_close(draft.runtime.host_cache, expected_draft)
