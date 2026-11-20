# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import test_stock_terminal_checkpoint as stock_terminal
import torch
from kitepond.integrations.vllm.terminal_host_prefix import NativeTerminalHostPrefix
from test_stock_terminal_checkpoint import identity

from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
    HiSparseConnectorWorker,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.hisparse.types import SparseKVPageTransfer
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request

capture = stock_terminal.capture


def test_terminal_host_tail_remains_unreadable_until_every_native_write_settles(
    capture,
):
    scheduler, owner, _, _ = capture()
    request = Request(
        request_id="partial",
        prompt_token_ids=list(range(1, 18)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    scheduler.add_request(request)
    owner.arm(request, identity(request.request_id))
    step = scheduler.schedule()
    scheduler.update_from_output(
        step,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[33]],
        ),
    )
    lease = owner.lease_for(request.request_id)
    assert lease is not None and lease.requested_common_tokens == 17
    witness = NativeTerminalHostPrefix(scheduler.kv_cache_manager)
    host = next(group for group in lease.groups if group.location == "host")
    fake = replace(host, block_ids=(host.block_ids[0], 0))
    with pytest.raises(ValueError, match="active pin"):
        witness.plan(replace(lease, groups=(fake, *lease.groups[1:])))
    with pytest.raises(BlockingIOError, match="not yet settled"):
        witness.require_durable(lease)
    assert not witness.plan(lease)
    assert not witness.plan(lease)
    assert scheduler.connector.has_pending_push_work()
    next_step = scheduler.schedule()
    assert next_step.total_num_scheduled_tokens == 0
    command = next_step.kv_connector_metadata.command
    assert command is not None
    transfers = command.page_transfers
    final = next(
        transfer
        for transfer in transfers
        if transfer.host_block_id == host.block_ids[1]
    )
    assert not final.after_forward and final.materialize_host and not final.restore
    assert (
        sum(transfer.host_block_id == host.block_ids[1] for transfer in transfers) == 1
    )

    pending = witness.coordinator.pending_spills
    spill_ids = tuple(pending)
    enqueued = {transfer_id: 4 for transfer_id in spill_ids}
    completed = {transfer_id: 4 for transfer_id in spill_ids}
    completed[final.transfer_id] = 3
    scheduler.connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=HiSparseConnectorWorkerMetadata(
                enqueued, completed
            )
        )
    )
    with pytest.raises(BlockingIOError, match="not yet settled"):
        witness.require_durable(lease)
    assert final.transfer_id in pending
    scheduler.connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=HiSparseConnectorWorkerMetadata(
                {}, {final.transfer_id: 1}
            )
        )
    )
    witness.require_durable(lease)
    assert not pending and witness.plan(lease)
    assert scheduler.schedule().kv_connector_metadata.command.page_transfers == []
    assert owner.cancel(lease)
    with pytest.raises(ValueError, match="not live"):
        witness.require_durable(lease)


@pytest.mark.parametrize("eager", [False, True])
def test_terminal_pre_forward_uses_real_copy_path_even_with_eager_mirror(eager):
    worker = object.__new__(HiSparseConnectorWorker)
    worker.cache_handles = [
        SimpleNamespace(runtime=SimpleNamespace(eager_host_mirror=eager))
    ]
    worker._record_transfer_completion = Mock()
    worker._enqueue_transfers = Mock()
    normal = SparseKVPageTransfer(1, 2, (3,), True)
    terminal = SparseKVPageTransfer(4, 5, (6,), False, materialize_host=True)
    worker._submit_transfers([normal, terminal])
    if eager:
        worker._record_transfer_completion.assert_called_once_with([normal])
        worker._enqueue_transfers.assert_called_once_with([terminal])
    else:
        worker._record_transfer_completion.assert_not_called()
        worker._enqueue_transfers.assert_called_once_with([normal, terminal])


def test_terminal_kpool_readback_preserves_only_the_physically_accepted_rows(capture):
    from test_glm_terminal_manifest import fenced_source

    scheduler, owner, _, _ = capture(
        host_layers=("mla.target", "mla.draft"),
        device_layers=(
            "idx.target",
            "idx.draft",
            "tail.target",
            "tail.draft",
            "kda",
        ),
    )
    # Capture a real 66-token pin with the same configured scheduler.
    large = Request(
        request_id="large",
        prompt_token_ids=list(range(66)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    scheduler.add_request(large)
    owner.arm(large, identity(large.request_id))
    step = scheduler.schedule()
    assert large.request_id in step.num_scheduled_tokens
    scheduler.update_from_output(
        step,
        ModelRunnerOutput(
            req_ids=tuple(step.num_scheduled_tokens),
            req_id_to_index={
                key: index for index, key in enumerate(step.num_scheduled_tokens)
            },
            sampled_token_ids=[[33] for _ in step.num_scheduled_tokens],
        ),
    )
    lease = owner.lease_for(large.request_id)
    assert lease is not None and lease.requested_common_tokens == 66
    source = fenced_source(lease, 0, "d" * 64)
    for part in source.layout.manifest():
        if not part.role.endswith(":kpool_tail"):
            continue
        layer = source.layout.layers[part.role]
        for slot in range(8):
            layer.tensor[layer.block_ids[0], :, slot, :].fill_(slot + 1)
        expected = source.read_chunk(part).view(torch.bfloat16).view(8, 2, 128)
        assert expected[0].unique().tolist() == [1.0]
        assert expected[1].unique().tolist() == [2.0]
        assert expected[2:].count_nonzero() == 0
        assert layer.tensor[layer.block_ids[0], :, 7, :].unique().tolist() == [8.0]
        with pytest.raises(ValueError, match="exact accepted KPool tail"):
            replace(source, tail_positions=()).read_chunk(part)
