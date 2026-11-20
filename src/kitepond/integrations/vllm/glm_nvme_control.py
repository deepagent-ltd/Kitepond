# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private D operator control for an accepted GLM NVMe generation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hmac import compare_digest
from pathlib import Path
from secrets import token_hex
from time import monotonic_ns

from kitepond.core.contracts import Wait
from kitepond.core.readiness import _digest
from kitepond.models.glm5next.frontier import digest
from kitepond.models.glm5next.layout import Glm5NextPDLayout

from vllm.v1.request import Request

from .glm_cold_runtime import ColdDirectoryScope
from .glm_nvme_protocol import NAMESPACE as NAMESPACE
from .glm_nvme_protocol import RESUME_PARAM, private_key, seal_nvme_operation
from .glm_terminal_manifest import GlmTerminalManifest

CONFIG = "kitepond_glm_nvme_control"
_KEY_ENV = "KITEPOND_GLM_NVME_OPERATOR_KEY_FILE"
_IDENTITY = {"engine_id", "boot_id", "request_id"}
_SCOPE = {"namespace", "generation", "transfer_id", "epoch", "deadline_ns"}
_RESUME = {"boundary", "prefix_sha256", "prompt_sha256"}
_MAX_TOKENS = 1048576


def _label(value: object) -> bool:
    return type(value) is str and 0 < len(value) <= 128 and value.strip() == value


@dataclass
class _Operation:
    mode: str
    body: dict
    scope: ColdDirectoryScope
    request: Request | None = None
    token: str | None = None
    stage: str = "armed"
    prefix: str | None = None
    boundary: int = 0
    job_id: int | None = None
    cold_admission: object = None
    reason: str | None = None
    cancel: bool = False


class GlmNvmeOperator:
    """Authenticate control intents, then bind only the actual owner Request."""

    def __init__(self, scheduler, deployment, policy, key, *, max_pending):
        if type(max_pending) is not int or not 0 < max_pending <= 128:
            raise ValueError("NVMe operator metadata budget is out of range")
        self.scheduler, self.deployment, self.policy = scheduler, deployment, policy
        self._key, self._limit = key, max_pending
        self._operations: dict[str, _Operation] = {}
        self._admitting: Request | None = None

    @classmethod
    def from_config(cls, scheduler, deployment, policy):
        raw = scheduler.vllm_config.additional_config.get(CONFIG)
        if raw is None:
            return None
        if not isinstance(raw, Mapping) or raw.keys() != {"max_pending"}:
            raise ValueError("D NVMe control needs one explicit metadata budget")
        key_path = os.environ.get(_KEY_ENV)
        if not key_path:
            raise ValueError("D NVMe control credential is unavailable")
        return cls(scheduler, deployment, policy, private_key(Path(key_path)), **raw)

    def _require_owner(self):
        self.scheduler._cold_decode_owner._require_scheduler_owner()

    def scope(self, request):
        operation = self._operations.get(request.request_id)
        if operation is not None and operation.request is request:
            return operation.scope
        return self.policy.resolve_scope(request)

    def authorize(self, request, verified):
        operation = self._operations.get(request.request_id)
        if (
            operation is not None
            and operation.request is request
            and operation.mode == "resume_arm"
        ):
            checkpoint = verified.checkpoint
            expected = operation.scope
            if (
                (checkpoint.namespace, checkpoint.generation)
                != (expected.namespace, expected.generation)
                or checkpoint.layout.target_valid_tokens != operation.boundary
                or verified.accepted_prefix_sha256 != operation.prefix
            ):
                raise ValueError(
                    "NVMe catalog changed its authorized generation or boundary"
                )
            return operation.prefix
        return self.policy.authorize_prefix(request, verified)

    def _open(self, operation, envelope):
        if (
            type(envelope) is not dict
            or envelope.keys() != {"body", "mac"}
            or type(envelope["body"]) is not dict
            or not _digest(envelope["mac"])
        ):
            raise ValueError("D NVMe control envelope is malformed")
        body = envelope["body"]
        fields = _IDENTITY
        if operation in {"archive_arm", "resume_arm"}:
            fields = fields | _SCOPE
        if operation == "resume_arm":
            fields = fields | _RESUME
        if (
            body.keys() != fields
            or (body["engine_id"], body["boot_id"])
            != (self.deployment.engine_id, self.deployment.boot_id)
            or not _label(body["request_id"])
            or not compare_digest(
                envelope["mac"], seal_nvme_operation(operation, body, self._key)["mac"]
            )
        ):
            raise ValueError("D NVMe command is not authorized for the current boot")
        return body

    def handle(self, operation, envelope):
        self._require_owner()
        if operation not in {
            "archive_arm",
            "resume_arm",
            "migrate",
            "status",
            "cancel",
            "forget",
        }:
            raise ValueError("Unsupported D NVMe operator operation")
        body = self._open(operation, envelope)
        request_id = body["request_id"]
        current = self._operations.get(request_id)
        if operation in {"archive_arm", "resume_arm"}:
            return self._arm(operation, body, current)
        if current is None:
            return {"state": "unknown"}
        if operation == "migrate":
            return self._migrate(current)
        if operation == "cancel":
            return self._cancel(current)
        if operation == "forget":
            if current.stage not in {
                "admitted",
                "cancelled",
                "capture_failed",
                "capture_wait",
                "committed",
                "failed",
                "miss",
            }:
                return {"state": "wait", "reason": "GLM_NVME_OPERATION_LIVE"}
            del self._operations[request_id]
            return {"state": "forgotten"}
        return self._status(current)

    def _arm(self, mode, body, current):
        if (
            not _label(body["namespace"])
            or not _label(body["transfer_id"])
            or any(
                type(body[k]) is not int or not 0 <= body[k] < 2**63
                for k in ("generation", "epoch", "deadline_ns")
            )
            or not monotonic_ns()
            < body["deadline_ns"]
            <= monotonic_ns() + 900_000_000_000
        ):
            raise ValueError("D NVMe intent has a foreign scope, sequence or deadline")
        if current is not None:
            if current.mode != mode or current.body != body:
                raise ValueError("D NVMe intent cannot replace another attempt")
            return {
                **self._status(current),
                **({"resume_token": current.token} if current.token else {}),
            }
        if len(self._operations) >= self._limit:
            return {"state": "wait", "reason": "GLM_NVME_OPERATOR_FULL"}
        scope = ColdDirectoryScope(body["namespace"], body["generation"])
        request = self.scheduler.requests.get(body["request_id"])
        if mode == "archive_arm":
            if request is not None and type(request) is not Request:
                raise ValueError("NVMe archival needs a typed native source Request")
            self._operations[body["request_id"]] = _Operation(
                mode, dict(body), scope, request
            )
            return {"state": "armed"}
        if (
            request is not None
            or type(body["boundary"]) is not int
            or not 0 < body["boundary"] < _MAX_TOKENS
            or not all(_digest(body[k]) for k in ("prefix_sha256", "prompt_sha256"))
        ):
            raise ValueError("D NVMe resume needs a fresh exact native prefix")
        token = token_hex(32)
        self._operations[body["request_id"]] = _Operation(
            mode,
            dict(body),
            scope,
            token=token,
            prefix=body["prefix_sha256"],
            boundary=body["boundary"],
        )
        return {"state": "armed", "resume_token": token}

    def on_terminal_read(self, lease, permit, manifest):
        current = self._operations.get(lease.identity.request_id)
        if current is None or current.mode != "archive_arm" or current.stage != "armed":
            return
        runtime = self.scheduler.native_glm_cold_store
        identity = lease.identity
        try:
            runtime.terminal.require_read_open(lease, permit)
            if (
                type(manifest) is not GlmTerminalManifest
                or (identity.transfer_id, identity.epoch)
                != (current.body["transfer_id"], current.body["epoch"])
                or monotonic_ns() > current.body["deadline_ns"]
                or runtime.terminal._require_live(lease).request is not current.request
            ):
                raise ValueError("Native archive changed its armed terminal")
            boundary = lease.requested_common_tokens
            accepted = {
                row.accepted_row
                for rank in manifest.ranks
                for row in rank.checkpoint.kda
            }
            if len(accepted) != 1 or not 0 < boundary <= current.request.num_tokens:
                raise ValueError("Native terminal boundary or KDA row is unavailable")
            [slot] = accepted
            layout = Glm5NextPDLayout(
                identity.request_id,
                identity.model_fingerprint,
                identity.config_fingerprint,
                0,
                identity.epoch,
                boundary,
                boundary,
                slot,
                "decode_ready",
            )
            captured = runtime.import_terminal(lease, permit, manifest, layout)
            if isinstance(captured, Wait):
                current.stage, current.reason = "capture_wait", captured.reason
                return
            current.prefix = digest(list(current.request.all_token_ids[:boundary]))
            runtime.directory.bind_resident_prefix(captured, current.prefix)
            current.boundary, current.stage = boundary, "captured"
        except Exception as error:
            current.stage, current.reason = "capture_failed", type(error).__name__
        finally:
            current.request = None

    def _migrate(self, current):
        if (
            current.mode != "archive_arm"
            or current.cancel
            or current.stage not in {"captured", "demoting"}
        ):
            raise ValueError("NVMe migration requires an accepted native archive")
        runtime = self.scheduler.native_glm_cold_store
        if runtime.terminal.lease_for(current.body["request_id"]) is not None:
            return {"state": "wait", "reason": "GLM_LIVE_REFERENCE_WAIT"}
        if current.job_id is None:
            job = runtime.demote_idle(current.scope)
            if isinstance(job, Wait):
                return {"state": "wait", "reason": job.reason}
            current.job_id, current.stage = job, "demoting"
        return self._status(current)

    def _status(self, current):
        view = None
        if current.job_id is not None:
            view = asdict(
                self.scheduler.native_glm_cold_store.scheduler.view(current.job_id)
            )
        return {
            "state": current.stage,
            "reason": current.reason,
            "job": view,
            "boundary": current.boundary or None,
            "prefix_sha256": current.prefix,
        }

    def _cancel(self, current):
        if current.stage == "admitted":
            return {"state": "admitted"}
        current.cancel = True
        if current.job_id is not None:
            self.scheduler.native_glm_cold_store.scheduler.cancel(current.job_id)
        elif current.cold_admission is None:
            current.stage = "cancelled"
        return {"state": "cancel_requested"}

    def receive(self, request):
        if self._admitting is request:
            return False
        params = request.kv_transfer_params or {}
        current = self._operations.get(request.request_id)
        if RESUME_PARAM not in params:
            if (
                current is not None
                and current.mode == "archive_arm"
                and current.stage == "armed"
                and current.request is None
                and not request.abort_immediately
                and monotonic_ns() <= current.body["deadline_ns"]
            ):
                current.request = request
            return False
        if (
            current is None
            or current.mode != "resume_arm"
            or current.request is not None
            or current.cancel
            or request.abort_immediately
            or monotonic_ns() > current.body["deadline_ns"]
            or type(params[RESUME_PARAM]) is not str
            or not compare_digest(current.token, params[RESUME_PARAM])
            or type(request.prompt_token_ids) is not list
            or not current.boundary < request.num_prompt_tokens <= _MAX_TOKENS
            or digest(request.prompt_token_ids) != current.body["prompt_sha256"]
        ):
            # A rejected native ADD must not reach the ordinary scheduler or
            # raise out of the EngineCore client dispatch loop.
            return True
        current.request, current.stage = request, "awaiting_ram"
        return True

    def abort_requests(self, request_ids) -> None:
        targets = self._operations if request_ids is None else request_ids
        if isinstance(targets, str):
            targets = (targets,)
        for request_id in targets:
            current = self._operations.get(request_id)
            if (
                current is not None
                and current.mode == "resume_arm"
                and current.stage
                in {
                    "awaiting_ram",
                    "reading",
                }
            ):
                current.cancel = True

    def after_tick(self) -> None:
        self._require_owner()
        if self.scheduler.has_requests():
            return
        runtime = self.scheduler.native_glm_cold_store
        bridge = self.scheduler.native_glm_cold_decode_bridge
        for current in self._operations.values():
            if current.job_id is not None:
                view = runtime.scheduler.view(current.job_id)
                if view.state in {"COMMITTED", "CANCELLED", "FAILED"}:
                    runtime.scheduler.reap(current.job_id)
                    current.job_id, current.stage = None, view.state.lower()
            if (
                current.stage == "armed"
                and monotonic_ns() > current.body["deadline_ns"]
            ):
                current.stage = "cancelled"
            if current.mode != "resume_arm" or current.stage not in {
                "awaiting_ram",
                "reading",
            }:
                continue
            expired = monotonic_ns() > current.body["deadline_ns"]
            try:
                if current.cancel or expired:
                    if current.cold_admission is not None and isinstance(
                        result := bridge.cancel(current.cold_admission), Wait
                    ):
                        current.reason = result.reason
                        continue
                    current.stage = "cancelled"
                    current.request = current.cold_admission = None
                    continue
                if current.cold_admission is None:
                    verified = runtime.directory.catalog.reopen(current.scope)
                    if verified is None:
                        current.stage, current.reason = (
                            "miss",
                            "GLM_CHECKPOINT_UNAVAILABLE",
                        )
                        current.request = None
                        continue
                    if verified.accepted_prefix_sha256 != current.prefix:
                        current.stage, current.reason = (
                            "miss",
                            "GLM_NVME_PREFIX_UNAVAILABLE",
                        )
                        current.request = None
                        continue
                    result = bridge.queue(
                        current.request,
                        verified,
                        transfer_id=current.body["transfer_id"],
                        epoch=current.body["epoch"],
                    )
                    if isinstance(result, Wait):
                        current.reason = result.reason
                        continue
                    current.cold_admission, current.stage = result, "reading"
                    continue
                self._admitting = current.request
                try:
                    result = bridge.stage(current.cold_admission)
                finally:
                    self._admitting = None
                if result is True:
                    current.stage, current.reason = "admitted", None
                    current.request = current.cold_admission = None
                    return
                current.reason = result.reason
                if result.reason in {
                    "GLM_COLD_READBACK_MISS",
                    "GLM_COLD_READBACK_FAILED",
                    "GLM_COLD_READBACK_CANCELLED",
                }:
                    current.stage = "miss"
                    current.request = current.cold_admission = None
            except (ValueError, OSError) as error:
                current.reason = type(error).__name__
                if current.cold_admission is None:
                    current.stage = "failed"
                    current.request = None
                else:
                    current.cancel = True


def lab_nvme_policy(scheduler):
    """Build the private one-model policy with a catalog key retained across boots."""
    from kitepond.models.glm5next.fingerprint import config_fingerprint

    from .glm_cold_catalog import GlmColdCatalog
    from .glm_cold_scheduler import COLD_CONFIG, NativeGlmColdStorePolicy

    options = scheduler.vllm_config.additional_config
    deployment = options[COLD_CONFIG]
    host = options["kitepond_glm_host"]
    catalog_path = os.environ.get("KITEPOND_GLM_NVME_CATALOG_KEY_FILE")
    if not catalog_path:
        raise ValueError("D NVMe durable catalog key is unavailable")
    catalog = GlmColdCatalog(
        Path(deployment["root"]),
        engine_id=deployment["engine_id"],
        boot_id=deployment["boot_id"],
        model_fingerprint=host["model_sha256"],
        config_fingerprint=config_fingerprint(
            scheduler.vllm_config.model_config.hf_text_config.to_dict()
        ),
        signing_key=private_key(Path(catalog_path)),
    )

    def trusted_only(_request, _receipt):
        raise ValueError("D NVMe admission requires its signed operator intent")

    return NativeGlmColdStorePolicy(
        lambda _: Wait("GLM_NVME_OPERATOR_INTENT_REQUIRED"),
        catalog,
        options[CONFIG]["max_pending"],
        authorize_prefix=trusted_only,
    )
