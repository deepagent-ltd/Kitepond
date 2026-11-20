# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from kitepond.integrations.vllm.glm_scheduler import GlmKitepondScheduler
from kitepond.integrations.vllm.glm_terminal_manifest import (
    GlmTerminalCheckpointVerifier,
    GlmTerminalManifest,
    TerminalRankSigner,
)
from kitepond.integrations.vllm.glm_terminal_scheduler import (
    GlmTerminalCheckpointScheduler,
)
from kitepond.integrations.vllm.glm_terminal_worker import GlmTerminalWorker
from kitepond.models.glm5next.layout import CacheRole
from test_glm_terminal_manifest import (
    BOUNDARY,
    FORMAT,
    ROLES,
    Fence,
    fenced_source,
    make_runner,
)
from test_stock_terminal_checkpoint import (
    CpuFixtureHiSparseConnector,
    identity,
    receipt,
)

from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.config.scheduler import SchedulerConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

BLOCK = 16
KEYS = {rank: bytes([rank + 1]) * 32 for rank in range(4)}
SOURCES = {rank: str(rank + 1) * 64 for rank in range(4)}
TOKENS = [index % 63 + 1 for index in range(BOUNDARY)] + [33]


class CpuTerminalExecutor(Executor):
    supports_timed_collective_rpc = True

    def _init_executor(self):
        self.calls = []
        self.durable = False
        self.ranks = []
        for rank in range(4):
            worker = object.__new__(GlmTerminalWorker)
            worker.vllm_config = self.vllm_config
            worker._kitepond_terminal_source = (
                TerminalRankSigner(rank, KEYS[rank]),
                self._late_source(rank),
            )
            worker._kitepond_terminal_frames = {}
            self.ranks.append(worker)

    def _late_source(self, rank):
        def resolve(lease, observed):
            self.calls.append(("durability", lease.identity.request_id, rank))
            if not self.durable:
                raise ValueError("Native terminal spill requires a later step")
            return fenced_source(lease, rank, SOURCES[rank])

        return resolve

    def collective_rpc(
        self,
        method,
        timeout=None,
        args=(),
        kwargs=None,
        non_block=False,
    ):
        if method == "get_kv_connector_handshake_metadata":
            return [{}]
        assert not non_block and method in {
            "kitepond_take_terminal_frame",
            "kitepond_terminal_checkpoint",
            "kitepond_discard_terminal_frame",
        }
        req_id = args[0].identity.request_id
        self.calls.append((method, req_id))
        if method == "kitepond_take_terminal_frame":
            assert 0 < timeout <= 2
            assert req_id in self.current.num_scheduled_tokens
        return [
            getattr(worker, method)(*args, **(kwargs or {})) for worker in self.ranks
        ]

    def execute_model(self, scheduler_output, non_block=False):
        assert non_block
        self.calls.append(("forward", tuple(scheduler_output.num_scheduled_tokens)))
        self.current = scheduler_output
        for worker in self.ranks:
            if hasattr(worker, "model_runner"):
                worker.model_runner._frames.clear()
        pending = Future()
        pending.set_result(None)
        return pending

    def sample_tokens(self, grammar_output, non_block=False):
        req_ids = list(self.current.num_scheduled_tokens)
        self.calls.append(("sample", tuple(req_ids)))
        for request in self.current.scheduled_new_reqs:
            if request.req_id == "terminal":
                intent = SimpleNamespace(
                    identity=SimpleNamespace(request_id=request.req_id)
                )
                kda_blocks = request.block_ids[1][:4]
                for worker in self.ranks:
                    worker.model_runner = make_runner(
                        intent, TOKENS, Fence(), kda_blocks
                    )
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[33] for _ in req_ids],
        )

    def check_health(self):
        pass

    def shutdown(self):
        self.ranks.clear()


class CpuTerminalCore(EngineCore):
    def _initialize_kv_caches(self, vllm_config):
        # CPU supplies the model allocation topology. Native Scheduler and
        # HiSparse own the actual request pages and reference counts.
        register_all_kvcache_specs(vllm_config)
        spec = FullAttentionSpec(
            block_size=BLOCK,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        )
        groups = [
            KVCacheGroupSpec(
                [r.layer_name for r in ROLES if r.cache_role is CacheRole.MLA],
                spec,
                role=KVCacheGroupRole.HISPARSE_SOURCE,
                host_resident=True,
            ),
            KVCacheGroupSpec(
                sorted(
                    {r.layer_name for r in ROLES if r.cache_role is not CacheRole.MLA}
                ),
                spec,
                role=KVCacheGroupRole.HISPARSE_INDEXER,
            ),
            KVCacheGroupSpec(
                ["resident"],
                HiSparseResidentSpec(block_size=BLOCK, page_size=4 * BLOCK),
                enable_kv_transfer=False,
            ),
            KVCacheGroupSpec(
                ["hot"],
                HiSparseHotSpec(
                    block_size=BLOCK, page_size=4 * BLOCK, blocks_per_request=2
                ),
                enable_kv_transfer=False,
            ),
        ]
        vllm_config.cache_config.num_gpu_blocks = 32
        return KVCacheConfig(
            num_blocks=32,
            hisparse_host_num_blocks=16,
            kv_cache_tensors=[],
            kv_cache_groups=groups,
        )


@pytest.fixture(params=[GlmTerminalCheckpointScheduler, GlmKitepondScheduler])
def terminal_engine(tmp_path, request):
    if not hasattr(Scheduler, "get_synchronous_step_callback"):
        pytest.fail("Terminal replay requires the native 5ce89f2e hook")
    (tmp_path / "config.json").write_text(
        """{
            "model_type": "opt", "architectures": ["OPTForCausalLM"],
            "vocab_size": 64, "hidden_size": 16, "word_embed_proj_dim": 16,
            "ffn_dim": 32, "num_attention_heads": 2, "num_hidden_layers": 1,
            "max_position_embeddings": 128, "pad_token_id": 0
        }""",
        encoding="utf-8",
    )
    cfg = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="float32",
            max_model_len=128,
            enforce_eager=True,
        ),
        cache_config=CacheConfig(block_size=BLOCK, enable_prefix_caching=False),
        parallel_config=ParallelConfig(
            distributed_executor_backend=CpuTerminalExecutor
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=128,
            max_num_batched_tokens=128,
            max_num_seqs=2,
            async_scheduling=False,
            is_encoder_decoder=False,
            scheduler_cls=request.param,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector=CpuFixtureHiSparseConnector.__name__,
            kv_connector_module_path=CpuFixtureHiSparseConnector.__module__,
            engine_id="D",
            kv_role="kv_both",
            kv_connector_extra_config={"host_pool_gib": 1},
        ),
    )
    engine = CpuTerminalCore(cfg, CpuTerminalExecutor, log_stats=False)
    executor = engine.model_executor
    attested = set()

    def verify_host(_lease):
        if not executor.durable:
            raise ValueError("Native terminal host prefix is not durable")

    def verify_settlement(_permit, settled):
        if settled not in attested:
            raise ValueError("Unknown fixture transfer receipt")

    verifier = GlmTerminalCheckpointVerifier(
        KEYS,
        SOURCES,
        verify_host,
        ROLES,
        representation_sha256=FORMAT,
    )
    owner = engine.scheduler.bind_glm_terminal_capture(
        source_boot_id="D-boot",
        verifier=verifier,
        verify_settlement=verify_settlement,
        imported_boundary=lambda _: 32,
        rpc_timeout=2,
    )
    try:
        yield engine, owner, attested
    finally:
        engine.shutdown()


def _add_request(engine, request_id, prompt):
    request = Request(
        request_id=request_id,
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    engine.add_request(request)
    return request


def test_native_committed_step_captures_worker_frame_before_a_future_host_spill(
    terminal_engine,
):
    engine, owner, attested = terminal_engine
    executor = engine.model_executor
    request = _add_request(engine, "terminal", TOKENS[:-1])
    owner.arm(
        request,
        replace(
            identity("terminal"),
            deadline_monotonic_ns=time.monotonic_ns() + 30_000_000_000,
        ),
    )
    outputs, executed = engine.step()
    assert executed and outputs[0].outputs[0].new_token_ids == [33]
    assert executor.calls == [
        ("forward", ("terminal",)),
        ("sample", ("terminal",)),
        ("kitepond_take_terminal_frame", "terminal"),
    ]
    lease = owner.lease_for("terminal")
    assert lease is not None
    assert all(
        "terminal" not in worker.model_runner._frames
        and "terminal" in worker._kitepond_terminal_frames
        for worker in executor.ranks
    )
    with pytest.raises(ValueError, match="requires a later step"):
        engine.collective_rpc("kitepond_terminal_checkpoint", args=(lease, 32))

    _add_request(engine, "next", [1, 2, 3])
    assert engine.step()[1]
    executor.durable = True
    signed = tuple(
        engine.collective_rpc("kitepond_terminal_checkpoint", args=(lease, 32))
    )
    manifest = GlmTerminalManifest(lease.sha256, signed)
    assert [rank.checkpoint.rank for rank in signed] == [0, 1, 2, 3]
    assert all(not worker.model_runner._frames for worker in executor.ranks)
    first, *rest = signed
    wrong = replace(first, checkpoint=replace(first.checkpoint, frame_sha256="a" * 64))
    with pytest.raises(ValueError, match="differs from the committed worker frames"):
        owner.begin_reads(lease, replace(manifest, ranks=(wrong, *rest)))
    permit = owner.begin_reads(lease, manifest)
    assert permit.manifest_sha256 == manifest.sha256
    for rank in (3, 1, 0, 2):
        settled = receipt(permit, rank)
        attested.add(settled)
        assert owner.settle(lease, permit, settled) == (rank == 2)
    assert "terminal" not in engine.scheduler.requests
    assert (
        engine.collective_rpc("kitepond_discard_terminal_frame", args=(lease,))
        == [True] * 4
    )
