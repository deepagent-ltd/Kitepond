# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from threading import Event
from time import monotonic_ns

import pytest
from kitepond.core.contracts import Wait
from kitepond.core.policy import IdlePolicy, LaneBudget
from kitepond.integrations.vllm.glm_cold_scheduler import (
    COLD_CONFIG,
    GlmColdRuntimeMixin,
)
from kitepond.integrations.vllm.glm_terminal_manifest import (
    GlmTerminalCheckpointVerifier,
    GlmTerminalManifest,
)
from kitepond.integrations.vllm.glm_terminal_scheduler import (
    GlmTerminalCheckpointScheduler,
)
from terminal_post_step_probe import (
    KEYS,
    SOURCES,
    TOKENS,
    CpuTerminalCore,
    CpuTerminalExecutor,
    _add_request,
)
from test_glm_cold_lifecycle import BUDGET
from test_glm_cold_runtime import ServiceDirectory
from test_glm_terminal_manifest import FORMAT, ROLES, fenced_source
from test_stock_terminal_checkpoint import (
    CpuFixtureHiSparseConnector,
    identity,
    receipt,
)

from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.config.scheduler import SchedulerConfig
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnectorMetadata,
    HiSparseConnectorWorkerMetadata,
)
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.utils import EngineZmqAddresses
from vllm.v1.outputs import KVConnectorOutput

CADENCE_NS = 20_000_000
IDLE = IdlePolicy(15)
LANE = LaneBudget(1, BUDGET, BUDGET)


class CpuColdCoreProc(EngineCoreProc, CpuTerminalCore):
    """Run native waits with a CPU model fixture and no frontend sockets."""

    @contextmanager
    def _perform_handshakes(self, *args):
        yield EngineZmqAddresses(inputs=[], outputs=[])

    def process_input_sockets(self, _, __, ___, ready_event):
        ready_event.set()

    def process_output_sockets(self, *args):
        pass


class CpuColdExecutor(CpuTerminalExecutor):
    def sample_tokens(self, grammar_output, non_block=False):
        output = super().sample_tokens(grammar_output, non_block)
        meta = self.current.kv_connector_metadata
        assert isinstance(meta, HiSparseConnectorMetadata)
        transfers = meta.command.page_transfers if meta.command is not None else ()
        if not transfers:
            return output
        counts = {transfer.transfer_id: len(self.ranks) for transfer in transfers}
        return replace(
            output,
            kv_connector_output=KVConnectorOutput(
                kv_connector_worker_meta=HiSparseConnectorWorkerMetadata(counts, counts)
            ),
        )


class StartupDirectory:
    engine_id = "D"
    boot_id = "D-boot"

    def __init__(self):
        self.native = None
        self.committed = Event()

    def cold_scope(self, lease):
        return self.native.cold_scope(lease)

    def import_accepted_terminal(self, entry):
        return self.native.import_accepted_terminal(entry)

    def expired_verified_checkpoints(self, **kwargs):
        if self.native is not None:
            yield from self.native.expired_verified_checkpoints(**kwargs)

    def last_business_use_ns(self, checkpoint):
        return self.native.last_business_use_ns(checkpoint)

    def acquire_snapshot(self, checkpoint, **kwargs):
        grant = self.native.acquire_snapshot(checkpoint, **kwargs)
        if isinstance(grant, Wait):
            return grant

        def settle(state):
            grant.settle(state)
            if state == "COMMITTED":
                self.committed.set()

        return replace(grant, settle=settle)

    def reserve_restore(self, checkpoint, **kwargs):
        return self.native.reserve_restore(checkpoint, **kwargs)


def _configuration(tmp_path, scheduler_cls, *, enabled):
    (tmp_path / "config.json").write_text(
        """{
            "model_type": "opt", "architectures": ["OPTForCausalLM"],
            "vocab_size": 64, "hidden_size": 16, "word_embed_proj_dim": 16,
            "ffn_dim": 32, "num_attention_heads": 2, "num_hidden_layers": 1,
            "max_position_embeddings": 128, "pad_token_id": 0
        }""",
        encoding="utf-8",
    )
    cold = {
        "mode": "idle-fsl2",
        "root": str(tmp_path / "cold"),
        "engine_id": "D",
        "boot_id": "D-boot",
        "cadence_ns": CADENCE_NS,
        "idle_minutes": IDLE.idle_minutes,
        "foreground": asdict(LANE),
        "background": asdict(LANE),
    }
    return VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="float32",
            max_model_len=128,
            enforce_eager=True,
        ),
        cache_config=CacheConfig(block_size=16, enable_prefix_caching=False),
        parallel_config=ParallelConfig(distributed_executor_backend=CpuColdExecutor),
        scheduler_config=SchedulerConfig(
            max_model_len=128,
            max_num_batched_tokens=128,
            max_num_seqs=2,
            async_scheduling=False,
            is_encoder_decoder=False,
            scheduler_cls=scheduler_cls,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector=CpuFixtureHiSparseConnector.__name__,
            kv_connector_module_path=CpuFixtureHiSparseConnector.__module__,
            engine_id="D",
            kv_role="kv_both",
            kv_connector_extra_config={"host_pool_gib": 1},
        ),
        additional_config={COLD_CONFIG: cold} if enabled else {},
    )


def test_disabled_profile_does_not_bind_an_idle_runtime(tmp_path):
    class Disabled(GlmColdRuntimeMixin, GlmTerminalCheckpointScheduler):
        pass

    config = _configuration(tmp_path, Disabled, enabled=False)
    engine = CpuColdCoreProc(config, True, "", CpuColdExecutor, False)
    try:
        assert engine.scheduler.get_quiescent_callback() is None
        with pytest.raises(ValueError, match="explicit idle-fsl2 deployment"):
            engine.scheduler.bind_glm_cold_runtime(StartupDirectory())
        assert not (tmp_path / "cold").exists()
    finally:
        engine.shutdown()


def test_glm_owner_commits_the_idle_checkpoint_without_frontend_input(tmp_path):
    ready = Future()
    directory = StartupDirectory()
    attested = set()
    durable = Event()
    offset = 0

    def clock():
        return monotonic_ns() + offset

    def verify_host(_lease):
        if not durable.is_set():
            raise ValueError("Native terminal host bytes still pending")

    def verify_settlement(_permit, result):
        if result not in attested:
            raise ValueError("Unknown native rank settlement")

    class AssemblyScheduler(GlmColdRuntimeMixin, GlmTerminalCheckpointScheduler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            with pytest.raises(RuntimeError, match="matching native bind"):
                self.get_quiescent_callback()
            verifier = GlmTerminalCheckpointVerifier(
                KEYS, SOURCES, verify_host, ROLES, representation_sha256=FORMAT
            )
            self.terminal = self.bind_glm_terminal_capture(
                source_boot_id=directory.boot_id,
                verifier=verifier,
                verify_settlement=verify_settlement,
                imported_boundary=lambda _: 32,
                rpc_timeout=2,
            )
            foreign = StartupDirectory()
            foreign.boot_id = "other-boot"
            with pytest.raises(ValueError, match="identity mismatch"):
                self.bind_glm_cold_runtime(foreign)
            self.cold = self.bind_glm_cold_runtime(directory, clock=clock)
            with pytest.raises(RuntimeError, match="already bound"):
                self.bind_glm_cold_runtime(directory)

    def run_engine():
        nonlocal offset
        config = _configuration(tmp_path, AssemblyScheduler, enabled=True)
        engine = CpuColdCoreProc(config, True, "", CpuColdExecutor, False)
        try:
            with ExitStack() as cleanup:
                owner = engine.scheduler.terminal
                executor = engine.model_executor
                request = _add_request(engine, "terminal", TOKENS[:-1])
                owner.arm(
                    request,
                    replace(
                        identity("terminal"),
                        deadline_monotonic_ns=monotonic_ns() + 30_000_000_000,
                    ),
                )
                assert engine.step()[1]
                lease = owner.lease_for(request.request_id)
                assert lease is not None
                _add_request(engine, "native-cleanup", [1, 2, 3])
                assert engine.step()[1]
                for _ in range(4):
                    if not engine.scheduler.connector.has_pending_push_work():
                        break
                    engine.step()
                assert not engine.scheduler.connector.has_pending_push_work()
                executor.durable = True
                durable.set()
                signed = tuple(
                    engine.collective_rpc(
                        "kitepond_terminal_checkpoint", args=(lease, 32)
                    )
                )
                manifest = GlmTerminalManifest(lease.sha256, signed)
                permit = owner.begin_reads(lease, manifest)
                layout = fenced_source(lease, 0, SOURCES[0]).layout.layout
                directory.native = ServiceDirectory(
                    cleanup, engine.scheduler, lease, manifest, layout, [clock()]
                )
                imported = engine.scheduler.cold.import_terminal(
                    lease, permit, manifest, layout
                )
                assert not isinstance(imported, Wait)
                for rank in range(4):
                    settled = receipt(permit, rank)
                    attested.add(settled)
                    assert owner.settle(lease, permit, settled) == (rank == 3)
                assert owner.lease_for(request.request_id) is None
                for _ in range(4):
                    if not engine.has_work():
                        break
                    engine.step()
                assert not engine.has_work()
                assert engine.input_queue.empty() and directory.native.pins
                assert not (tmp_path / "cold").exists()
                before = tuple(executor.calls)
                held = tuple(block for _, block in directory.native.pins)
                offset += IDLE.idle_ns
                ready.set_result(engine)
                with pytest.raises(SystemExit):
                    engine.run_busy_loop()
                assert tuple(executor.calls) == before
                assert all(block.ref_cnt == 0 for block in held)
                lane = engine.scheduler.cold.scheduler.observe_lane("STORE")
                assert lane.metadata_entries == 0
        finally:
            engine.shutdown()

    def owner_thread():
        try:
            run_engine()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise

    with ThreadPoolExecutor(max_workers=1) as threads:
        completion = threads.submit(owner_thread)
        engine = ready.result(timeout=15)
        try:
            assert directory.committed.wait(10)
            assert len(list((tmp_path / "cold/aggregates").glob("*.json"))) == 1
            assert not directory.native.resident and not directory.native.pins
            assert directory.native.acquired == 1
        finally:
            engine.shutdown_state = EngineShutdownState.REQUESTED
            engine.input_queue.put((EngineCoreRequestType.WAKEUP, None))
        completion.result(timeout=10)
