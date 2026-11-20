# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Authenticate cold GLM metadata without opening FSL2 payloads."""

from __future__ import annotations

import hmac
import os
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

from kitepond.backends._fsl2_retirement import sync_directory
from kitepond.backends.fsl2 import (
    BundleIdentity,
    BundlePlan,
    ColdStorageAdapter,
    PartSpec,
    _hex,
    canonical,
    digest,
    rank_manifest_name,
    read_manifest_document,
)
from kitepond.backends.fsl2_aggregate import FourRankManifestStore
from kitepond.models.glm5next.cold_lifecycle import GlmColdCheckpoint
from kitepond.models.glm5next.layout import VERSION, Glm5NextPDLayout

from .glm_cold_runtime import ColdDirectoryScope

_SCHEMA = "kitepond-glm-cold-catalog-v1"
_DOMAIN = b"kitepond/glm-cold-catalog/v1\x00"
_RECEIPT_DOMAIN = b"kitepond/glm-cold-reopen/v1\x00"
_CATALOG_BYTES = 64 * 1024


def _text(value) -> bool:
    return type(value) is str and 0 < len(value) <= 256


@dataclass(frozen=True)
class VerifiedColdCheckpoint:
    """Authenticated metadata selected for one fresh D boot."""

    checkpoint: GlmColdCheckpoint
    source_boot_id: str
    target_boot_id: str
    aggregate_sha256: str
    hashes_by_rank: tuple[tuple[str, ...], ...]
    mac: str
    accepted_prefix_sha256: str | None = None

    def checksums(self, rank: int) -> dict[PartSpec, str]:
        return dict(
            zip(
                self.checkpoint.plans[rank].parts,
                self.hashes_by_rank[rank],
                strict=True,
            )
        )


class GlmColdCatalogIdentityError(ValueError):
    """The accepted terminal is scoped to another model or configuration."""


class GlmColdCatalog:
    """Exact-scope metadata lookup for trusted control and startup callers."""

    def __init__(
        self,
        root: Path,
        *,
        engine_id: str,
        boot_id: str,
        model_fingerprint: str,
        config_fingerprint: str,
        signing_key: bytes,
    ) -> None:
        if (
            not Path(root).is_absolute()
            or not _text(engine_id)
            or not _text(boot_id)
            or not _hex(model_fingerprint)
            or not _hex(config_fingerprint)
            or type(signing_key) is not bytes
            or len(signing_key) != 32
        ):
            raise ValueError("Cold catalog needs a scoped deployment and signing key")
        self.root = Path(root)
        self.engine_id, self.boot_id = engine_id, boot_id
        self.model_fingerprint, self.config_fingerprint = (
            model_fingerprint,
            config_fingerprint,
        )
        self._key = signing_key

    def _mac(self, domain: bytes, body: dict) -> str:
        return hmac.digest(self._key, domain + canonical(body), "sha256").hex()

    def _target(self, namespace: str, generation: int) -> Path:
        if not _text(namespace) or type(generation) is not int or generation < 0:
            raise ValueError("Cold catalog needs an exact authorized generation")
        key = digest(
            {
                "engine_id": self.engine_id,
                "model_fingerprint": self.model_fingerprint,
                "config_fingerprint": self.config_fingerprint,
                "layout_version": VERSION,
                "namespace": namespace,
                "generation": generation,
            }
        )
        return self.root / "glm-catalog-v1" / (key + ".json")

    def _aggregate(self, identity: BundleIdentity) -> dict | None:
        path = self.root / "aggregates"
        if path.is_symlink():
            return None
        return read_manifest_document(path / (digest(asdict(identity)) + ".json"))

    def _authorized(self, path: Path) -> dict | None:
        if path.parent.is_symlink():
            return None
        body = read_manifest_document(path, _CATALOG_BYTES)
        if body is None:
            return None
        tag = body.pop("mac", None)
        if not _hex(tag) or not hmac.compare_digest(tag, self._mac(_DOMAIN, body)):
            return None
        return body

    def require_layout_identity(self, layout: Glm5NextPDLayout) -> None:
        if type(layout) is not Glm5NextPDLayout or (
            layout.model_fingerprint,
            layout.config_fingerprint,
        ) != (self.model_fingerprint, self.config_fingerprint):
            raise GlmColdCatalogIdentityError(
                "Accepted terminal model or configuration differs from the cold catalog"
            )

    def publish(
        self,
        checkpoint: GlmColdCheckpoint,
        aggregate: dict,
        *,
        accepted_prefix_sha256: str | None = None,
    ) -> None:
        """Record a catalog witness only after FSL2 aggregate publication."""
        self.require_layout_identity(checkpoint.layout)
        if (
            aggregate.get("identity") != asdict(checkpoint.identity)
            or aggregate.get("metadata") != checkpoint.metadata
            or not _hex(aggregate.get("digest"))
            or self._aggregate(checkpoint.identity) != aggregate
            or (accepted_prefix_sha256 is not None and not _hex(accepted_prefix_sha256))
        ):
            raise ValueError("Cold catalog witness differs from durable FSL2 metadata")
        path = self._target(checkpoint.namespace, checkpoint.generation)
        path.parent.mkdir(mode=0o700, exist_ok=True)
        if path.parent.is_symlink():
            raise ValueError("Cold catalog directory must be owned locally")
        body = {
            "schema": _SCHEMA,
            "engine_id": self.engine_id,
            "source_boot_id": self.boot_id,
            "identity": asdict(checkpoint.identity),
            "layout": checkpoint.layout.to_metadata(),
            "terminal_manifest_sha256": checkpoint.terminal_manifest_sha256,
            "aggregate_sha256": aggregate["digest"],
            "rank_files": [
                rank_manifest_name(checkpoint.identity, plan)
                for plan in checkpoint.plans
            ],
        }
        if accepted_prefix_sha256 is not None:
            body["accepted_prefix_sha256"] = accepted_prefix_sha256
        payload = canonical(dict(body, mac=self._mac(_DOMAIN, body)))
        if len(payload) > _CATALOG_BYTES:
            raise ValueError("Cold catalog metadata exceeds its envelope")

        temp = path.parent / ("." + secrets.token_hex(16) + ".tmp")
        inserted = False
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            try:
                os.link(temp, path)
                inserted = True
            except FileExistsError:
                existing = self._authorized(path)
                if (
                    existing is None
                    or dict(existing, source_boot_id=self.boot_id) != body
                ):
                    raise
        finally:
            temp.unlink(missing_ok=True)
        try:
            sync_directory(path.parent)
            sync_directory(self.root)
        except OSError:
            if inserted:
                path.unlink(missing_ok=True)
            raise

    def _receipt_mac(
        self, checkpoint, source, aggregate, hashes, accepted_prefix=None
    ) -> str:
        body = {
            "engine_id": self.engine_id,
            "target_boot_id": self.boot_id,
            "source_boot_id": source,
            "identity": asdict(checkpoint.identity),
            "plans": [digest(plan.document()) for plan in checkpoint.plans],
            "hashes": hashes,
            "aggregate_sha256": aggregate,
        }
        if accepted_prefix is not None:
            body["accepted_prefix_sha256"] = accepted_prefix
        return self._mac(_RECEIPT_DOMAIN, body)

    def verify(self, receipt: VerifiedColdCheckpoint) -> bool:
        """Verify an issued current-boot receipt without touching disk."""
        if type(receipt) is not VerifiedColdCheckpoint:
            return False
        checkpoint = receipt.checkpoint
        try:
            return (
                type(checkpoint) is GlmColdCheckpoint
                and checkpoint.layout.model_fingerprint == self.model_fingerprint
                and checkpoint.layout.config_fingerprint == self.config_fingerprint
                and receipt.target_boot_id == self.boot_id
                and _hex(receipt.mac)
                and (
                    receipt.accepted_prefix_sha256 is None
                    or _hex(receipt.accepted_prefix_sha256)
                )
                and hmac.compare_digest(
                    receipt.mac,
                    self._receipt_mac(
                        checkpoint,
                        receipt.source_boot_id,
                        receipt.aggregate_sha256,
                        receipt.hashes_by_rank,
                        receipt.accepted_prefix_sha256,
                    ),
                )
            )
        except (TypeError, ValueError, RecursionError):
            return False

    def reopen(self, scope: ColdDirectoryScope) -> VerifiedColdCheckpoint | None:
        """Read only metadata, for a previously authorized session scope."""
        if type(scope) is not ColdDirectoryScope:
            raise TypeError("Reopen requires an authorized cold directory scope")
        try:
            path = self._target(scope.namespace, scope.generation)
            body = self._authorized(path)
            if (
                body is None
                or body.get("schema") != _SCHEMA
                or body.get("engine_id") != self.engine_id
                or not _text(body.get("source_boot_id"))
                or not _hex(body.get("aggregate_sha256"))
            ):
                return None
            identity = BundleIdentity(**body["identity"])
            prefix = body.get("accepted_prefix_sha256")
            if prefix is not None and not _hex(prefix):
                return None
            recorded_layout = body["layout"]
            if (
                type(recorded_layout) is not dict
                or recorded_layout.get("version") != VERSION
            ):
                return None
            layout = Glm5NextPDLayout(**recorded_layout)
            if (
                identity.namespace != scope.namespace
                or identity.generation != scope.generation
                or identity.model_fingerprint != self.model_fingerprint
                or layout.model_fingerprint != self.model_fingerprint
                or layout.config_fingerprint != self.config_fingerprint
            ):
                return None
            names = body["rank_files"]
            if type(names) is not list or len(names) != 4:
                return None
            receiver = []
            plans = []
            for rank, file in enumerate(names):
                if (
                    type(file) is not str
                    or not file.endswith(".json")
                    or not _hex(file[:-5])
                ):
                    return None
                parent = self.root / f"rank-{rank}" / "manifests"
                if parent.is_symlink() or parent.parent.is_symlink():
                    return None
                doc = read_manifest_document(parent / file)
                if doc is None:
                    return None
                plan = BundlePlan.from_document(doc["plan"])
                verified = ColdStorageAdapter.validate_rank_document(
                    doc, identity, plan
                )
                if file != rank_manifest_name(identity, plan) or verified is None:
                    return None
                receiver.append(doc)
                plans.append(plan)
            checkpoint = GlmColdCheckpoint(
                scope.namespace,
                scope.generation,
                layout,
                body["terminal_manifest_sha256"],
                tuple(plans),
            )
            if checkpoint.identity != identity:
                return None
            aggregate = self._aggregate(identity)
            expected = FourRankManifestStore._document(
                identity, checkpoint.metadata, checkpoint.plans, receiver
            )
            if aggregate != expected or expected["digest"] != body["aggregate_sha256"]:
                return None
            hashes = tuple(
                tuple(part["sha256"] for part in doc["parts"]) for doc in receiver
            )
            source = body["source_boot_id"]
            auth = self._receipt_mac(
                checkpoint, source, expected["digest"], hashes, prefix
            )
            return VerifiedColdCheckpoint(
                checkpoint,
                source,
                self.boot_id,
                expected["digest"],
                hashes,
                auth,
                prefix,
            )
        except (OSError, KeyError, TypeError, ValueError, RecursionError):
            return None
