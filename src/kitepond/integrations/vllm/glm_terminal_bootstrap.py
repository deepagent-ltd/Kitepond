# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Enroll terminal rank signers on the EngineCore owner before serving."""

from __future__ import annotations

import hmac
import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from secrets import token_hex
from threading import get_ident
from typing import TYPE_CHECKING

from kitepond.models.glm5next.host_cold_extent import ColdExtentLayout
from kitepond.models.glm5next.import_recipe import part_role
from kitepond.models.glm5next.layout import CacheIdentity

from vllm.v1.kv_cache_interface import HiSparseHotSpec, HiSparseResidentSpec

from .glm_terminal_manifest import (
    GlmTerminalCheckpointVerifier,
    TerminalRankSigner,
    _is_digest,
)
from .terminal_host_prefix import NativeTerminalHostPrefix

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.interface import SynchronousStepRPC

    from .checkpoint import CheckpointReadReceipt
    from .glm_cold_enrollment import NativeGlmColdRegions
    from .glm_terminal_scheduler import GlmTerminalCheckpointScheduler
    from .glm_terminal_worker import GlmTerminalWorker
    from .terminal_checkpoint import TerminalPinLease, TerminalReadPermit

BOOTSTRAP_CONFIG = "kitepond_glm_terminal_bootstrap"
_KEY_FILE_ENV = "KITEPOND_GLM_TERMINAL_KEYS_FILE"
_KEYS_SCHEMA = "kitepond.glm-terminal-keys.v1"
_ENROLLMENT_SCHEMA = "kitepond.glm-terminal-enrollment.v1"
_RANKS = frozenset(range(4))
_MAX_KEY_FILE_BYTES = 4096
_RPC_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class TerminalStartupIdentity:
    engine_id: str
    boot_id: str
    representation_sha256: str


@dataclass(frozen=True)
class _Credentials:
    identity: TerminalStartupIdentity
    rank_keys: tuple[bytes, ...] = field(repr=False)


@dataclass(frozen=True)
class TerminalRankEnrollment:
    identity: TerminalStartupIdentity
    challenge: str
    rank: int
    region_sha256: str
    host_layout_sha256: str
    roles: tuple[CacheIdentity, ...]
    signature: str
    cold_layout: ColdExtentLayout | None = None

    def signed_message(self) -> dict:
        message = {
            "schema": _ENROLLMENT_SCHEMA,
            "identity": asdict(self.identity),
            "challenge": self.challenge,
            "rank": self.rank,
            "region_sha256": self.region_sha256,
            "host_layout_sha256": self.host_layout_sha256,
            "roles": tuple(part_role(role) for role in self.roles),
        }
        if self.cold_layout is not None:
            message["cold_layout"] = asdict(self.cold_layout)
        return message


def deployment_enabled(config: VllmConfig) -> bool:
    additional = config.additional_config
    enabled = (
        additional.get(BOOTSTRAP_CONFIG, False)
        if isinstance(additional, Mapping)
        else False
    )
    if type(enabled) is not bool:
        raise ValueError("GLM terminal startup requires an explicit boolean")
    return enabled


def _load_credentials(config: VllmConfig) -> _Credentials:
    path = os.environ.get(_KEY_FILE_ENV)
    additional = config.additional_config
    scope = additional.get("kitepond") if isinstance(additional, Mapping) else None
    host = (
        additional.get("kitepond_glm_host") if isinstance(additional, Mapping) else None
    )
    if (
        not deployment_enabled(config)
        or not isinstance(scope, Mapping)
        or scope.get("stage") != "decode"
        or not isinstance(host, Mapping)
        or host.get("role") != "authority"
        or (host.get("authority_engine_id"), host.get("authority_boot_id"))
        != (scope.get("authority_engine_id"), scope.get("boot_id"))
        or additional.get("kitepond_glm_native_terminal") is not True
        or config.kv_transfer_config is None
        or not path
        or not os.path.isabs(path)
    ):
        raise ValueError("GLM terminal startup requires the enabled native D role")

    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as file:
            info = os.fstat(file.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) not in {0o400, 0o600}
                or info.st_uid != os.geteuid()
                or not 0 < info.st_size <= _MAX_KEY_FILE_BYTES
            ):
                raise ValueError
            body = file.read(_MAX_KEY_FILE_BYTES + 1)
        if len(body) > _MAX_KEY_FILE_BYTES:
            raise ValueError
        raw = json.loads(body)
        if (
            type(raw) is not dict
            or raw.keys()
            != {"schema", "engine_id", "boot_id", "representation_sha256", "rank_keys"}
            or raw["schema"] != _KEYS_SCHEMA
            or any(
                type(raw[field]) is not str
                or not 0 < len(raw[field]) <= 256
                or raw[field].strip() != raw[field]
                for field in ("engine_id", "boot_id")
            )
            or not _is_digest(raw["representation_sha256"])
            or type(raw["rank_keys"]) is not dict
            or raw["rank_keys"].keys() != {str(rank) for rank in _RANKS}
            or any(not _is_digest(value) for value in raw["rank_keys"].values())
            or len(set(raw["rank_keys"].values())) != 4
        ):
            raise ValueError
        identity = TerminalStartupIdentity(
            raw["engine_id"], raw["boot_id"], raw["representation_sha256"]
        )
        keys = tuple(bytes.fromhex(raw["rank_keys"][str(rank)]) for rank in range(4))
    except (UnicodeError, ValueError, OSError):
        raise ValueError("Invalid private GLM terminal credential file") from None

    if (identity.engine_id, identity.boot_id) != (
        config.kv_transfer_config.engine_id,
        scope.get("boot_id"),
    ) or identity.engine_id != scope.get("authority_engine_id"):
        raise ValueError("GLM terminal credential file belongs to another engine boot")
    return _Credentials(identity, keys)


def enroll_native_worker(
    worker: GlmTerminalWorker,
    identity: TerminalStartupIdentity,
    challenge: str,
) -> TerminalRankEnrollment:
    """Authenticate this actual cache owner without sending rank credentials."""
    views = worker.kitepond_native_state_views()
    credentials = _load_credentials(worker.vllm_config)
    descriptor = views.descriptor
    if (
        type(identity) is not TerminalStartupIdentity
        or identity != credentials.identity
        or not _is_digest(challenge)
        or (descriptor.authority_engine_id, descriptor.authority_boot_id)
        != (identity.engine_id, identity.boot_id)
        or descriptor.source_rank != views.rank
        or views.rank not in _RANKS
    ):
        raise ValueError("GLM terminal worker belongs to another rank or boot")

    signer = TerminalRankSigner(views.rank, credentials.rank_keys[views.rank])
    from .glm_cold_scheduler import _Deployment

    cold_layout = None
    if _Deployment.load(worker.vllm_config.additional_config) is not None:
        from .glm_cold_enrollment import register_worker_cold_memory

        cold_layout = register_worker_cold_memory(worker)
    evidence = TerminalRankEnrollment(
        identity,
        challenge,
        signer.rank,
        descriptor.fingerprint(),
        descriptor.layout_sha256,
        views.role_catalog(),
        "",
        cold_layout,
    )
    receipt = replace(evidence, signature=signer._sign(evidence.signed_message()))
    worker.bind_kitepond_native_terminal_source(signer, identity.representation_sha256)
    return receipt


class GlmTerminalStartup:
    """Bind the exact four local cache owners once on the scheduler thread."""

    def __init__(self, scheduler: GlmTerminalCheckpointScheduler) -> None:
        self._scheduler = scheduler
        self._thread = get_ident()
        self._verify_transport = None
        self._cold_regions: NativeGlmColdRegions | None = None
        self._cold_signers: tuple[TerminalRankSigner, ...] = ()

    def require_cold_regions(self) -> NativeGlmColdRegions:
        self._require_owner()
        if self._cold_regions is None:
            raise RuntimeError("Cold runtime has no authenticated native regions")
        return self._cold_regions

    def close_cold_regions(self) -> None:
        """Release bootstrap's borrow before any directory owns it."""
        self._require_owner()
        if self._cold_regions is not None:
            self._cold_regions.close()
            self._cold_regions = None
            self._cold_signers = ()

    def bind_native_cold_store(
        self, worker_rpc: SynchronousStepRPC, *, restore: bool = False
    ):
        from .glm_cold_remote_store import NativeGlmColdStore

        if restore:
            from .glm_cold_remote_restore import NativeGlmColdRestore

            provider = NativeGlmColdRestore
        else:
            provider = NativeGlmColdStore
        return provider(
            self.require_cold_regions(),
            worker_rpc,
            self._cold_signers,
        )

    def _require_owner(self) -> None:
        if self._thread != get_ident():
            raise RuntimeError("GLM terminal startup belongs to the EngineCore owner")

    def bind_transport_settlement(
        self, verify: Callable[[TerminalReadPermit, CheckpointReadReceipt], None]
    ) -> None:
        self._require_owner()
        if self._verify_transport is not None or not callable(verify):
            raise ValueError(
                "GLM terminal settlement verifier is invalid or already bound"
            )
        self._verify_transport = verify

    def require_transport_settlement(self) -> None:
        if self._verify_transport is None:
            raise RuntimeError("GLM reverse transfer has no native settlement verifier")

    def _verify_settlement(
        self, permit: TerminalReadPermit, receipt: CheckpointReadReceipt
    ) -> None:
        self._require_owner()
        self.require_transport_settlement()
        assert self._verify_transport is not None
        self._verify_transport(permit, receipt)

    def require_imported_boundary(self, lease: TerminalPinLease) -> int:
        self._require_owner()
        identity = lease.identity
        request = self._scheduler.requests.get(identity.request_id)
        lookup = getattr(self._scheduler, "verified_decode_import", None)
        if request is None or not callable(lookup):
            raise ValueError("GLM terminal source has no verified current D import")
        binding = lookup(request)
        if (
            binding is None
            or binding.request is not request
            or (binding.grant.engine_id, binding.grant.boot_id, binding.grant.epoch)
            != (identity.source_engine_id, identity.source_boot_id, identity.epoch)
        ):
            raise ValueError(
                "GLM terminal source and its D import differ in boot or epoch"
            )
        return binding.imported_tokens

    def bind(self, worker_rpc: SynchronousStepRPC) -> None:
        self._require_owner()
        if not callable(getattr(self._scheduler, "verified_decode_import", None)):
            raise RuntimeError("GLM terminal startup needs verified native D admission")
        credentials = _load_credentials(self._scheduler.vllm_config)
        identity = credentials.identity
        fence = NativeTerminalHostPrefix(self._scheduler.kv_cache_manager)
        challenge = token_hex(32)
        reports = tuple(
            worker_rpc(
                "kitepond_enroll_terminal_worker",
                timeout=_RPC_TIMEOUT_SECONDS,
                args=(identity, challenge),
            )
        )
        if (
            len(reports) != 4
            or any(type(report) is not TerminalRankEnrollment for report in reports)
            or {report.rank for report in reports} != _RANKS
            or len({report.region_sha256 for report in reports}) != 4
            or len({report.host_layout_sha256 for report in reports}) != 1
            or len({report.roles for report in reports}) != 1
        ):
            raise ValueError(
                "GLM terminal startup requires four matching native workers"
            )
        cache = self._scheduler.kv_cache_manager.kv_cache_config
        native_layers = {
            name
            for group in cache.kv_cache_groups
            if not isinstance(
                group.kv_cache_spec, (HiSparseResidentSpec, HiSparseHotSpec)
            )
            for name in group.layer_names
        }
        roles = reports[0].roles
        if (
            roles != tuple(sorted(roles, key=part_role))
            or {role.layer_name for role in roles} != native_layers
        ):
            raise ValueError(
                "GLM terminal startup catalogue differs from native scheduling"
            )
        for report in reports:
            signer = TerminalRankSigner(report.rank, credentials.rank_keys[report.rank])
            if (
                report.identity != identity
                or report.challenge != challenge
                or (
                    report.cold_layout is not None
                    and type(report.cold_layout) is not ColdExtentLayout
                )
                or not _is_digest(report.region_sha256)
                or not _is_digest(report.host_layout_sha256)
                or not _is_digest(report.signature)
                or not hmac.compare_digest(
                    report.signature, signer._sign(report.signed_message())
                )
            ):
                raise ValueError(
                    "GLM terminal worker enrollment cannot authenticate rank or boot"
                )
        verifier = GlmTerminalCheckpointVerifier(
            dict(enumerate(credentials.rank_keys)),
            {report.rank: report.region_sha256 for report in reports},
            fence.require_durable,
            roles,
            representation_sha256=identity.representation_sha256,
        )
        from .glm_cold_scheduler import _Deployment

        deployment = _Deployment.load(self._scheduler.vllm_config.additional_config)
        cold_regions = None
        if deployment is not None:
            from .glm_cold_enrollment import NativeGlmColdRegions

            if (deployment.engine_id, deployment.boot_id) != (
                identity.engine_id,
                identity.boot_id,
            ):
                raise ValueError("GLM cold deployment belongs to another engine boot")
            cold_regions = NativeGlmColdRegions.open(
                self._scheduler.vllm_config, reports
            )
        try:
            self._scheduler.bind_glm_terminal_capture(
                source_boot_id=identity.boot_id,
                verifier=verifier,
                verify_settlement=self._verify_settlement,
                imported_boundary=self.require_imported_boundary,
                rpc_timeout=_RPC_TIMEOUT_SECONDS,
                enable_native_host_fence=True,
            )
        except BaseException:
            if cold_regions is not None:
                cold_regions.close()
            raise
        self._cold_regions = cold_regions
        if cold_regions is not None:
            self._cold_signers = tuple(
                TerminalRankSigner(rank, key)
                for rank, key in enumerate(credentials.rank_keys)
            )
