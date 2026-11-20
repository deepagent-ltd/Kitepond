# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from kitepond.integrations.vllm.glm_terminal_worker import (
    GlmTerminalMTP,
    GlmTerminalRunner,
    GlmTerminalWorker,
)

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm.v1.worker.gpu_worker import Worker


def test_stock_runner_refuses_to_claim_capture_without_the_factory():
    assert issubclass(GlmTerminalWorker, Worker)
    assert issubclass(GlmTerminalRunner, GPUModelRunner)
    assert issubclass(GlmTerminalMTP, MTPSpeculator)
    if not hasattr(GPUModelRunner, "_make_speculator"):
        with pytest.raises(RuntimeError, match="early speculator factory"):
            GlmTerminalRunner(None, None)
    else:
        assert GlmTerminalRunner._make_speculator is not GPUModelRunner._make_speculator


def test_draft_query_observer_runs_after_native_mtp_resets_topk_state():
    recorded = []
    speculator = GlmTerminalMTP.__new__(GlmTerminalMTP)
    speculator.share_mtp_topk_indices = True
    speculator.model = SimpleNamespace(
        model=SimpleNamespace(
            set_skip_topk=lambda value: recorded.append(("stock_topk", value))
        )
    )
    speculator._observe_query = lambda current: recorded.append(("draft", current))
    speculator.on_prefill_begin(num_reqs=1)
    assert recorded == [("stock_topk", False), ("draft", speculator)]
