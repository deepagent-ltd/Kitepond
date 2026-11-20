# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in scheduler binding for the GLM idle FSL2 runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import TIMEOUT_MAX
from time import monotonic_ns
from typing import TYPE_CHECKING

from kitepond.core.policy import IdlePolicy, LaneBudget

if TYPE_CHECKING:
    from kitepond.core.contracts import Wait

    from vllm.v1.request import Request

    from .glm_cold_catalog import GlmColdCatalog, VerifiedColdCheckpoint
    from .glm_cold_decode_bridge import GlmColdDecodeBridge
    from .glm_cold_runtime import ColdDirectoryScope, GlmColdDirectory, GlmColdRuntime
    from .glm_nvme_control import GlmNvmeOperator

COLD_CONFIG = "kitepond_glm_cold"
_STORE_POLICY_CONFIG = "kitepond_glm_cold_store_policy"


def native_cold_restore_enabled(config) -> bool:
    options = config.additional_config
    raw = options.get(_STORE_POLICY_CONFIG) if isinstance(options, Mapping) else None
    return isinstance(raw, Mapping) and raw.get("mode") == "store-restore-lab"


_DEPLOYMENT_FIELDS = {
    "mode",
    "root",
    "engine_id",
    "boot_id",
    "cadence_ns",
    "idle_minutes",
    "foreground",
    "background",
}
_LANE_FIELDS = {"max_active_jobs", "max_inflight_bytes", "per_job_bytes"}
_DIRECTORY_METHODS = (
    "cold_scope",
    "import_accepted_terminal",
    "expired_verified_checkpoints",
    "last_business_use_ns",
    "acquire_snapshot",
    "reserve_restore",
)


def _lane(raw) -> LaneBudget:
    if not isinstance(raw, Mapping) or raw.keys() != _LANE_FIELDS:
        raise ValueError("GLM cold lane needs explicit job and byte budgets")
    return LaneBudget(**raw)


@dataclass(frozen=True)
class _Deployment:
    root: Path
    engine_id: str
    boot_id: str
    cadence_ns: int
    idle: IdlePolicy
    foreground: LaneBudget
    background: LaneBudget

    @classmethod
    def load(cls, additional) -> _Deployment | None:
        if not isinstance(additional, Mapping) or COLD_CONFIG not in additional:
            return None
        raw = additional[COLD_CONFIG]
        if (
            not isinstance(raw, Mapping)
            or raw.keys() != _DEPLOYMENT_FIELDS
            or raw["mode"] != "idle-fsl2"
            or type(raw["root"]) is not str
            or not Path(raw["root"]).is_absolute()
            or any(
                type(raw[name]) is not str
                or not raw[name]
                or raw[name].strip() != raw[name]
                for name in ("engine_id", "boot_id")
            )
            or type(raw["cadence_ns"]) is not int
            or not 1_000_000 <= raw["cadence_ns"] <= TIMEOUT_MAX * 1_000_000_000
        ):
            raise ValueError("Invalid kitepond_glm_cold idle-fsl2 deployment")
        return cls(
            Path(raw["root"]),
            raw["engine_id"],
            raw["boot_id"],
            raw["cadence_ns"],
            IdlePolicy(raw["idle_minutes"]),
            _lane(raw["foreground"]),
            _lane(raw["background"]),
        )


@dataclass(frozen=True)
class _Binding:
    deployment: _Deployment
    interval_ns: int
    callback: Callable[[], object]


@dataclass(frozen=True)
class NativeGlmColdStorePolicy:
    """Service-owned scope resolution and already-authorized cold catalog."""

    resolve_scope: Callable[[Request], ColdDirectoryScope | Wait]
    catalog: GlmColdCatalog
    max_entries: int
    authorize_prefix: Callable[[Request, VerifiedColdCheckpoint], str] | None = None


class GlmColdRuntimeMixin:
    """Bind one concrete native directory before EngineCoreProc starts."""

    _glm_cold_binding: _Binding | None = None
    _glm_cold_registration_read = False
    _native_glm_cold_store: GlmColdRuntime | None = None
    _native_glm_cold_bridge: GlmColdDecodeBridge | None = None
    _native_glm_nvme_operator: GlmNvmeOperator | None = None

    def get_owner_worker_control_registration(self):
        additional = self.vllm_config.additional_config
        raw = (
            additional.get(_STORE_POLICY_CONFIG)
            if isinstance(additional, Mapping)
            else None
        )
        if raw is None:
            if (
                isinstance(additional, Mapping)
                and "kitepond_glm_nvme_control" in additional
            ):
                raise ValueError("Native NVMe control needs the detached cold policy")
            parent = getattr(super(), "get_owner_worker_control_registration", None)
            return parent() if callable(parent) else None
        deployment = _Deployment.load(additional)
        restore = native_cold_restore_enabled(self.vllm_config)
        if (
            not isinstance(raw, Mapping)
            or raw.keys() != {"mode", "factory"}
            or raw["mode"] not in {"store-only-lab", "store-restore-lab"}
            or type(raw["factory"]) is not str
            or not raw["factory"]
            or deployment is None
            or self._terminal_startup is None
            or (restore and "kitepond_glm_cold_import" not in additional)
            or (not restore and "kitepond_glm_nvme_control" in additional)
        ):
            raise ValueError("Native GLM FSL2 store needs an explicit lab-owned policy")

        from .glm_cold_rank_control import RPC_NAME

        def bind(worker_rpc):
            from vllm.utils.import_utils import resolve_obj_by_qualname

            from .glm_cold_catalog import GlmColdCatalog
            from .native_glm_cold_directory import NativeGlmColdDirectory

            startup = self._terminal_startup
            assert startup is not None
            try:
                policy = resolve_obj_by_qualname(raw["factory"])(self)
                if (
                    type(policy) is not NativeGlmColdStorePolicy
                    or not callable(policy.resolve_scope)
                    or type(policy.catalog) is not GlmColdCatalog
                    or policy.catalog.root != deployment.root
                    or (restore and not callable(policy.authorize_prefix))
                ):
                    raise ValueError("Native cold policy needs its scoped catalog")
                provider = startup.bind_native_cold_store(worker_rpc, restore=restore)
                operator = None
                if "kitepond_glm_nvme_control" in additional:
                    from .glm_nvme_control import GlmNvmeOperator

                    operator = GlmNvmeOperator.from_config(self, deployment, policy)
                directory = NativeGlmColdDirectory(
                    self._terminal_checkpoints,
                    provider,
                    policy.resolve_scope if operator is None else operator.scope,
                    max_entries=policy.max_entries,
                    catalog=policy.catalog,
                )
                runtime = self.bind_glm_cold_runtime(directory)
                if restore:
                    from .glm_cold_decode_bridge import GlmColdDecodeBridge

                    self._native_glm_cold_bridge = GlmColdDecodeBridge(
                        self,
                        runtime,
                        authorize_prefix=(
                            policy.authorize_prefix
                            if operator is None
                            else operator.authorize
                        ),
                        register_grant=provider.register_grant,
                        release_grant=provider.release_grant,
                        confirm_rank_ownership=provider.confirm_rank_ownership,
                        transfer_to_worker=provider.detach_to_worker,
                    )
                self._native_glm_cold_store = runtime
                if operator is not None:
                    from .glm_nvme_control import NAMESPACE

                    self._cold_decode_owner.bind_control(NAMESPACE, operator.handle)
                    self._terminal_checkpoints.bind_read_observer(
                        operator.on_terminal_read
                    )
                    self._native_glm_nvme_operator = operator
            except BaseException:
                startup.close_cold_regions()
                raise

        return ((RPC_NAME,), bind)

    @property
    def native_glm_cold_store(self) -> GlmColdRuntime:
        runtime = self._native_glm_cold_store
        if runtime is None:
            raise RuntimeError("Native GLM FSL2 store was not configured")
        return runtime

    @property
    def native_glm_cold_decode_bridge(self) -> GlmColdDecodeBridge:
        bridge = self._native_glm_cold_bridge
        if bridge is None:
            raise RuntimeError("Native GLM detached cold readback is disabled")
        return bridge

    def add_request(self, request):
        operator = self._native_glm_nvme_operator
        if operator is not None and operator.receive(request):
            return None
        return super().add_request(request)

    def finish_requests(self, request_ids, finished_status):
        operator = self._native_glm_nvme_operator
        if operator is not None:
            if request_ids is not None and not isinstance(request_ids, str):
                request_ids = tuple(request_ids)
            operator.abort_requests(request_ids)
        return super().finish_requests(request_ids, finished_status)

    def bind_glm_cold_runtime(
        self, directory: GlmColdDirectory, *, clock: Callable[[], int] = monotonic_ns
    ) -> GlmColdRuntime:
        deployment = _Deployment.load(self.vllm_config.additional_config)
        if deployment is None:
            raise ValueError("GLM cold runtime needs an explicit idle-fsl2 deployment")
        if self._glm_cold_binding is not None:
            raise RuntimeError("GLM cold runtime is already bound")
        if self._glm_cold_registration_read:
            raise RuntimeError("Bind the GLM cold runtime before native startup")
        if not callable(getattr(super(), "get_quiescent_callback", None)):
            raise RuntimeError("vLLM has no native quiescent scheduler callback")
        terminal = self._terminal_checkpoints
        transfer = self.vllm_config.kv_transfer_config
        scope = (deployment.engine_id, deployment.boot_id)
        directory_scope = (
            getattr(directory, "engine_id", None),
            getattr(directory, "boot_id", None),
        )
        if (
            terminal is None
            or self._terminal_capture is None
            or terminal._manager is not self.kv_cache_manager
            or terminal._identity != scope
            or transfer is None
            or transfer.engine_id != deployment.engine_id
            or directory_scope != scope
        ):
            raise ValueError("GLM cold runtime directory or owner identity mismatch")
        if any(
            not callable(getattr(directory, name, None)) for name in _DIRECTORY_METHODS
        ):
            raise TypeError("GLM cold runtime requires a complete native directory")
        terminal._require_owner()

        from .glm_cold_runtime import GlmColdRuntime

        binding = None

        def register(*, interval_ns, callback):
            nonlocal binding
            binding = _Binding(deployment, interval_ns, callback)

        runtime = GlmColdRuntime(
            deployment.root,
            directory,
            terminal,
            register_quiescent=register,
            cadence_ns=deployment.cadence_ns,
            idle=deployment.idle,
            foreground=deployment.foreground,
            background=deployment.background,
            clock=clock,
        )
        assert binding is not None
        self._glm_cold_binding = binding
        return runtime

    def get_quiescent_callback(self) -> tuple[int, Callable[[], object]] | None:
        deployment = _Deployment.load(self.vllm_config.additional_config)
        binding = self._glm_cold_binding
        if deployment is None and binding is None:
            self._glm_cold_registration_read = True
            inherited = getattr(super(), "get_quiescent_callback", None)
            return inherited() if callable(inherited) else None
        if binding is None or binding.deployment != deployment:
            raise RuntimeError("Configured cold runtime needs a matching native bind")
        self._glm_cold_registration_read = True
        operator = self._native_glm_nvme_operator
        if operator is None:
            return binding.interval_ns, binding.callback

        def advance():
            jobs = binding.callback()
            operator.after_tick()
            return jobs

        return binding.interval_ns, advance
