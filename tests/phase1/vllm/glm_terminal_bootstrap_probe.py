# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from kitepond.core.contracts import Wait
from kitepond.integrations.vllm import glm_terminal_worker
from kitepond.integrations.vllm.checkpoint import CheckpointReadReceipt
from kitepond.integrations.vllm.glm_cold_catalog import GlmColdCatalog
from kitepond.integrations.vllm.glm_cold_scheduler import (
    COLD_CONFIG,
    GlmColdRuntimeMixin,
    NativeGlmColdStorePolicy,
)
from kitepond.integrations.vllm.glm_host_backing import _Launch, _open_extended_backing
from kitepond.integrations.vllm.glm_terminal_bootstrap import (
    BOOTSTRAP_CONFIG,
    TerminalStartupIdentity,
)
from kitepond.integrations.vllm.glm_terminal_scheduler import (
    GlmTerminalCheckpointScheduler,
)
from kitepond.integrations.vllm.glm_terminal_worker import GlmTerminalWorker
from kitepond.integrations.vllm.terminal_checkpoint import (
    TerminalPinLease,
    TerminalReadPermit,
)
from kitepond.models.glm5next.import_recipe import part_role
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion
from terminal_post_step_probe import (
    BLOCK,
    CpuTerminalCore,
    CpuTerminalExecutor,
)
from test_glm_terminal_manifest import FORMAT, ROLES
from test_stock_terminal_checkpoint import CpuFixtureHiSparseConnector, identity
from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.config.scheduler import SchedulerConfig


class TerminalStartupScheduler(GlmTerminalCheckpointScheduler):
    def verified_decode_import(self, request):
        return self.verified if self.verified.request is request else None

    def get_worker_startup_callback(self):
        startup = super().get_worker_startup_callback()
        if startup is None:
            return None

        def bind(rpc):
            self.startup_rpc = rpc
            startup(rpc)

        return bind

    def get_owner_worker_control_registration(self):
        if not self.vllm_config.additional_config.get("test_owner_controls"):
            return None

        def bind(rpc):
            assert self._terminal_startup.require_cold_regions().layout.extent == 16384
            self.owner_rpc = rpc

        return (("kitepond_probe_owner_control",), bind)


class TerminalStartupExecutor(CpuTerminalExecutor):
    scenario: ClassVar[str]
    resource_owner: ClassVar[ExitStack]
    native_rank: ClassVar[int]
    latest_workers: ClassVar[list[GlmTerminalWorker]]
    latest_pools: ClassVar[list[SimpleNamespace]]

    def _init_executor(self):
        self.calls = []
        self.ranks = []
        self.__class__.latest_workers = self.ranks
        self.__class__.latest_pools = []
        for _ in range(4):
            worker = object.__new__(GlmTerminalWorker)
            worker.vllm_config = self.vllm_config
            self.ranks.append(worker)

    def register_test_caches(self):
        for rank, worker in enumerate(self.ranks):
            boot = (
                "foreign" if self.scenario == "wrong_boot" and rank == 2 else "D-boot"
            )
            scope = dict(
                authority_engine_id="D",
                authority_boot_id=boot,
                source_rank=rank,
                layout_sha256="a" * 64,
            )
            if COLD_CONFIG in worker.vllm_config.additional_config:
                launcher = _Launch.from_config(worker.vllm_config)
                assert launcher.cold_extent is not None
                backing, drop = _open_extended_backing(
                    launcher, scope, launcher.cold_extent.layout(4096)
                )
                descriptor = backing._kitepond_glm_host_descriptor
                self.resource_owner.callback(drop)
                self.latest_pools.append(SimpleNamespace(registered=backing))
                worker.model_runner = SimpleNamespace(
                    kv_cache_config=SimpleNamespace(
                        hisparse_host_num_blocks=1,
                        hisparse_host_block_stride=4096,
                    )
                )
            else:
                region = NativeHiSparseMemfdRegion.create(4096, **scope)
                descriptor = region.descriptor
                self.resource_owner.callback(region.close)
            worker._kitepond_native_state_views = SimpleNamespace(
                rank=rank,
                descriptor=descriptor,
                role_catalog=lambda: tuple(sorted(ROLES, key=part_role)),
                terminal_source=lambda *_args, **_kwargs: None,
            )
        self.calls.append("registered")

    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block=False
    ):
        if method == "kitepond_probe_owner_control":
            assert self.calls[-1] == "terminal-enrollment"
            assert timeout == 1.0 and not kwargs and not non_block
            return [args[0] + rank for rank in range(4)]
        if method == "get_kv_connector_handshake_metadata":
            assert self.calls == ["registered"]
            self.calls.append("native-handshake")
            return [{}]
        assert self.calls[-1] == "native-handshake"
        assert method == "kitepond_enroll_terminal_worker"
        assert timeout == 5.0 and not kwargs and not non_block
        assert len(args) == 2 and type(args[0]) is TerminalStartupIdentity
        assert type(args[1]) is str and len(args[1]) == 64
        self.calls.append("terminal-enrollment")
        results = []
        for rank, worker in enumerate(self.ranks):
            self.__class__.native_rank = (
                0 if self.scenario == "wrong_rank" and rank == 2 else rank
            )
            results.append(worker.kitepond_enroll_terminal_worker(*args))
        if self.scenario == "wrong_signature":
            results[-1] = replace(results[-1], signature=results[0].signature)
        elif self.scenario == "cold_tampered":
            layout = results[-1].cold_layout
            results[-1] = replace(
                results[-1], cold_layout=replace(layout, aux_end=layout.aux_end + 4096)
            )
        self.enrollment = tuple(results)
        return results


class TerminalStartupStoreScheduler(GlmColdRuntimeMixin, TerminalStartupScheduler):
    pass


def _store_policy(scheduler):
    cold = scheduler.vllm_config.additional_config[COLD_CONFIG]
    return NativeGlmColdStorePolicy(
        lambda _: Wait("CPU_SCOPE_NOT_REQUIRED"),
        GlmColdCatalog(
            Path(cold["root"]),
            engine_id="D",
            boot_id="D-boot",
            model_fingerprint="a" * 64,
            config_fingerprint="b" * 64,
            signing_key=b"q" * 32,
        ),
        4,
    )


class TerminalStartupCore(CpuTerminalCore):
    def _initialize_kv_caches(self, config):
        cache = super()._initialize_kv_caches(config)
        self.model_executor.register_test_caches()
        return cache


@pytest.fixture
def launch(tmp_path, monkeypatch):
    with ExitStack() as resources:
        monkeypatch.setattr(
            glm_terminal_worker,
            "get_dcp_group",
            lambda: SimpleNamespace(rank_in_group=TerminalStartupExecutor.native_rank),
        )
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "opt",
                    "architectures": ["OPTForCausalLM"],
                    "vocab_size": 64,
                    "hidden_size": 16,
                    "word_embed_proj_dim": 16,
                    "ffn_dim": 32,
                    "num_attention_heads": 2,
                    "num_hidden_layers": 1,
                    "max_position_embeddings": 128,
                    "pad_token_id": 0,
                }
            )
        )
        secret = tmp_path / "credential-redacted-before-output"
        keys = {str(rank): bytes([rank + 1]).hex() * 32 for rank in range(4)}
        secret.write_text(
            json.dumps(
                {
                    "schema": "kitepond.glm-terminal-keys.v1",
                    "engine_id": "D",
                    "boot_id": "D-boot",
                    "representation_sha256": FORMAT,
                    "rank_keys": keys,
                }
            )
        )
        secret.chmod(0o600)
        monkeypatch.setenv("KITEPOND_GLM_TERMINAL_KEYS_FILE", str(secret))

        def make(*, scenario="normal", enabled=True, cold=False):
            TerminalStartupExecutor.resource_owner = resources
            TerminalStartupExecutor.scenario = scenario
            if scenario == "public_file":
                secret.chmod(0o644)
            if scenario == "symlink_file":
                link = tmp_path / "credential-link"
                link.symlink_to(secret)
                monkeypatch.setenv("KITEPOND_GLM_TERMINAL_KEYS_FILE", str(link))
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
                    distributed_executor_backend=TerminalStartupExecutor
                ),
                scheduler_config=SchedulerConfig(
                    max_model_len=128,
                    max_num_batched_tokens=128,
                    max_num_seqs=2,
                    async_scheduling=False,
                    is_encoder_decoder=False,
                    scheduler_cls=(
                        TerminalStartupStoreScheduler
                        if scenario == "cold_store"
                        else TerminalStartupScheduler
                    ),
                ),
                kv_transfer_config=KVTransferConfig(
                    kv_connector=CpuFixtureHiSparseConnector.__name__,
                    kv_connector_module_path=CpuFixtureHiSparseConnector.__module__,
                    engine_id="D",
                    kv_role="kv_both",
                    kv_connector_extra_config={"host_pool_gib": 1},
                ),
                additional_config={
                    "kitepond": {
                        "authority_engine_id": "D",
                        "boot_id": "D-boot",
                        "stage": "decode",
                    },
                    "kitepond_glm_host": {
                        "role": "producer" if scenario == "wrong_role" else "authority",
                        "authority_engine_id": "D",
                        "authority_boot_id": "D-boot",
                    },
                    "kitepond_glm_native_terminal": enabled,
                    BOOTSTRAP_CONFIG: enabled,
                },
            )
            if cold:
                from kitepond.integrations.vllm import glm_cold_enrollment

                monkeypatch.setattr(
                    glm_cold_enrollment,
                    "get_hisparse_host_pool",
                    lambda *_: TerminalStartupExecutor.latest_pools[
                        TerminalStartupExecutor.native_rank
                    ],
                )
                socket_directory = tmp_path / "s"
                socket_directory.mkdir(mode=0o700)
                cfg.additional_config["kitepond_glm_host"].update(
                    socket_directory=str(socket_directory),
                    model_sha256="a" * 64,
                    cold_extent={
                        "aux_bytes_per_rank": 8192,
                        "scratch_bytes_per_rank": 4096,
                        "max_total_bytes": 4 * 16384,
                    },
                )
                lane = {
                    "max_active_jobs": 1,
                    "max_inflight_bytes": 8192,
                    "per_job_bytes": 8192,
                }
                cfg.additional_config[COLD_CONFIG] = {
                    "mode": "idle-fsl2",
                    "root": str(tmp_path / "cold"),
                    "engine_id": "D",
                    "boot_id": "D-boot",
                    "cadence_ns": 20_000_000,
                    "idle_minutes": 15,
                    "foreground": lane,
                    "background": lane,
                }
                if scenario == "owner_control":
                    cfg.additional_config["test_owner_controls"] = True
                elif scenario == "cold_store":
                    cfg.additional_config["kitepond_glm_cold_store_policy"] = {
                        "mode": "store-only-lab",
                        "factory": __name__ + "._store_policy",
                    }
            engine = TerminalStartupCore(cfg, TerminalStartupExecutor, log_stats=False)
            resources.callback(engine.shutdown)
            if cold:
                regions = engine.scheduler._terminal_startup.require_cold_regions()
                resources.callback(regions.close)
            return engine

        yield make, secret, keys


def test_engine_owner_enrolls_exact_rank_keys_after_native_cache_bind(launch, caplog):
    make, secret, keys = launch
    engine = make()
    executor, scheduler = engine.model_executor, engine.scheduler
    assert executor.calls == ["registered", "native-handshake", "terminal-enrollment"]
    assert not scheduler.has_requests()
    assert scheduler._terminal_checkpoints is not None
    for rank, worker in enumerate(executor.ranks):
        signer, _ = worker._kitepond_terminal_source
        assert (signer.rank, signer.key) == (rank, bytes.fromhex(keys[str(rank)]))
    for secret_value in keys.values():
        assert secret_value not in repr(executor.enrollment)
        assert secret_value not in caplog.text
    assert str(secret) not in caplog.text
    with pytest.raises(RuntimeError, match="outside its synchronous step callback"):
        scheduler.startup_rpc("kitepond_enroll_terminal_worker", timeout=1)

    lease = TerminalPinLease(identity("terminal"), 1, 66, 67, "e" * 64, 33, ())
    request = object()
    scheduler.requests["terminal"] = request
    scheduler.verified = SimpleNamespace(
        request=request,
        imported_tokens=63,
        grant=SimpleNamespace(
            engine_id="D", boot_id="D-boot", epoch=lease.identity.epoch
        ),
    )
    assert scheduler._terminal_capture.imported_boundary(lease) == 63
    scheduler.verified.grant.epoch += 1
    with pytest.raises(ValueError, match="differ in boot or epoch"):
        scheduler._terminal_capture.imported_boundary(lease)

    permit = TerminalReadPermit(lease.sha256, "b" * 64)
    receipt = CheckpointReadReceipt(permit.sha256, 0, "acked", 0)
    with pytest.raises(RuntimeError, match="no native settlement verifier"):
        scheduler._terminal_startup._verify_settlement(permit, receipt)
    attested = []

    def settle(actual_permit, actual_receipt):
        attested.append((actual_permit, actual_receipt))

    scheduler.bind_terminal_transport_settlement(settle)
    scheduler._terminal_startup._verify_settlement(permit, receipt)
    assert attested == [(permit, receipt)]


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("wrong_role", "enabled native D role"),
        ("wrong_boot", "another rank or boot"),
        ("wrong_rank", "foreign rank"),
        ("wrong_signature", "cannot authenticate rank or boot"),
        ("public_file", "Invalid private GLM terminal credential file"),
        ("symlink_file", "Invalid private GLM terminal credential file"),
    ],
)
def test_startup_rejects_foreign_rank_or_insecure_credentials(
    launch, scenario, message
):
    make, secret, keys = launch
    with pytest.raises(ValueError, match=message) as failure:
        make(scenario=scenario)
    assert str(secret) not in str(failure.value)
    assert all(value not in str(failure.value) for value in keys.values())


def test_disabled_startup_and_early_worker_enrollment_do_not_read_credentials(launch):
    make, secret, _ = launch
    secret.unlink()
    engine = make(enabled=False)
    assert engine.model_executor.calls == ["registered", "native-handshake"]
    assert engine.scheduler._terminal_checkpoints is None
    early = object.__new__(GlmTerminalWorker)
    scope = TerminalStartupIdentity("D", "D-boot", FORMAT)
    with pytest.raises(RuntimeError, match="no enrolled native GLM terminal views"):
        early.kitepond_enroll_terminal_worker(scope, "f" * 64)


def test_engine_owner_attaches_signed_native_cold_bytes_on_startup(launch):
    make, _, _ = launch
    engine = make(cold=True)
    cold = engine.scheduler._terminal_startup.require_cold_regions()
    assert cold.layout.extent == 16384
    for rank, (worker, region) in enumerate(
        zip(engine.model_executor.ranks, cold.regions, strict=True)
    ):
        memory = worker._kitepond_native_cold_memory
        assert region.descriptor.source_rank == rank
        assert region.descriptor == memory.region.descriptor
        assert memory.region.owner and not region.owner
        assert memory.mla_bytes == memory._registered.nbytes == 4096
        memory.region.mapping[4096:4100] = b"cold"
        assert region.mapping[4096:4100] == b"cold"


def test_engine_authenticates_cold_geometry_with_terminal_credentials(launch):
    make, _, _ = launch
    with pytest.raises(ValueError, match="cannot authenticate rank or boot"):
        make(cold=True, scenario="cold_tampered")


def test_engine_owner_worker_controls_are_bounded_and_independent_of_startup(launch):
    make, _, _ = launch
    engine = make(cold=True, scenario="owner_control")
    scheduler = engine.scheduler
    rpc = scheduler.owner_rpc
    method = "kitepond_probe_owner_control"
    assert rpc(method, timeout=1.0, args=(3,)) == [3, 4, 5, 6]
    with pytest.raises(ValueError, match="not registered"):
        rpc("kitepond_unrelated", timeout=1.0)
    with pytest.raises(ValueError, match="finite timeout"):
        rpc(method, timeout=float("nan"))
    with ThreadPoolExecutor(max_workers=1) as executor:
        remote = executor.submit(rpc, method, timeout=1.0, args=(3,))
        with pytest.raises(RuntimeError, match="EngineCore owner"):
            remote.result(timeout=3)
    with pytest.raises(RuntimeError, match="outside its synchronous step callback"):
        scheduler.startup_rpc(method, timeout=1.0, args=(3,))


def test_native_policy_assembles_the_cold_directory_before_quiescence(launch):
    make, _, _ = launch
    engine = make(cold=True, scenario="cold_store")
    scheduler = engine.scheduler
    runtime = scheduler.native_glm_cold_store
    directory = runtime.directory
    assert directory.terminal is scheduler._terminal_checkpoints
    assert (
        directory.provider.regions is scheduler._terminal_startup.require_cold_regions()
    )
    interval, callback = scheduler.get_quiescent_callback()
    assert interval == 20_000_000
    assert callback() == ()
    assert not directory.catalog.root.exists()
