# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish a durable candidate for four verified native FSL2 rank manifests."""

import fcntl
import os
import secrets
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path

from kitepond.backends._fsl2_retirement import sync_directory
from kitepond.backends.fsl2 import (
    MAX_MANIFEST_BYTES,
    BundleIdentity,
    BundlePlan,
    ColdStorageAdapter,
    canonical,
    digest,
    read_manifest_document,
)
from kitepond.core.contracts import Wait

SCHEMA = "kitepond-four-rank-fsl2-v1"


def _sync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_plans(identity: BundleIdentity, plans: tuple[BundlePlan, ...]) -> None:
    if (
        not isinstance(identity, BundleIdentity)
        or type(plans) is not tuple
        or len(plans) != 4
        or any(
            not isinstance(plan, BundlePlan)
            or any(spec.rank != rank for spec in plan.parts)
            for rank, plan in enumerate(plans)
        )
    ):
        raise ValueError(
            "Aggregate requires four exact rank plans and a native identity"
        )


def _validate_metadata(identity: BundleIdentity, metadata: dict) -> None:
    if (
        type(metadata) is not dict
        or metadata.get("model_fingerprint") != identity.model_fingerprint
        or type(metadata.get("boundary_tokens")) is not int
        or metadata["boundary_tokens"] != identity.boundary_tokens
        or digest(metadata) != identity.semantic_fingerprint
    ):
        raise ValueError("Aggregate metadata differs from the exact native identity")


class FourRankManifestStore:
    """Coordinate metadata only. The existing rank adapters own physical FSL2 I/O."""

    def __init__(self, root: Path, adapters: tuple[ColdStorageAdapter, ...]) -> None:
        self.root = Path(root).resolve(strict=True)
        if (
            type(adapters) is not tuple
            or len(adapters) != 4
            or any(
                not isinstance(adapter, ColdStorageAdapter)
                or adapter.root.resolve() != self.root / f"rank-{rank}"
                for rank, adapter in enumerate(adapters)
            )
        ):
            raise ValueError(
                "Aggregate requires the four existing native rank adapters"
            )
        self.adapters = adapters
        self.manifests = self.root / "aggregates"
        self.manifests.mkdir(mode=0o700, exist_ok=True)
        if self.manifests.is_symlink():
            raise ValueError("Aggregate manifest directory must be owned locally")

    def _path(self, identity: BundleIdentity) -> Path:
        return self.manifests / (digest(asdict(identity)) + ".json")

    @contextmanager
    def _owned(self):
        fd = os.open(
            self.root / ".full-io.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
        busy = Wait("COLD_IO_ADMISSION_WAIT")
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield busy
                return
            with ExitStack() as held:
                for adapter in self.adapters:
                    if not adapter.lock.acquire(blocking=False):
                        yield busy
                        return
                    held.callback(adapter.lock.release)
                    if adapter.closed or adapter.closing:
                        raise RuntimeError("Aggregate requires open rank storage")
                    if (
                        adapter.ticket is not None
                        or adapter.retirement.active is not None
                        or adapter._namespace_held
                        or not adapter._try_namespace()
                    ):
                        yield busy
                        return
                    held.callback(adapter._release_namespace)
                    if adapter.retirement.pending_journals():
                        yield Wait("COLD_RETIREMENT_RECOVERY_REQUIRED")
                        return
                yield None
        finally:
            os.close(fd)

    def _ranks(self, identity, plans):
        documents = tuple(
            adapter.discover(identity, plan)
            for adapter, plan in zip(self.adapters, plans, strict=True)
        )
        return None if None in documents else documents

    @staticmethod
    def _document(identity, metadata, plans, documents):
        body = {
            "schema": SCHEMA,
            "identity": asdict(identity),
            "metadata": metadata,
            "ranks": [
                {
                    "rank": rank,
                    "plan_sha256": digest(plan.document()),
                    "manifest_sha256": document["digest"],
                }
                for rank, (plan, document) in enumerate(
                    zip(plans, documents, strict=True)
                )
            ],
        }
        return dict(body, digest=digest(body))

    def _sync_ranks(self, identity, plans, documents) -> None:
        for adapter, plan, document in zip(
            self.adapters, plans, documents, strict=True
        ):
            for spec, part in zip(plan.parts, document["parts"], strict=True):
                key = adapter._key(part["key"])
                adapter._verify_file(key, spec, part["sha256"])
                _sync_file(adapter._file(key))
            _sync_file(adapter._path(identity, plan))
            sync_directory(adapter.data)
            sync_directory(adapter.manifests)
            sync_directory(adapter.root)
        sync_directory(self.root)

    def publish(
        self, identity: BundleIdentity, plans: tuple[BundlePlan, ...], metadata: dict
    ) -> dict | Wait | None:
        """Publish once all ranks exist, verify and sync their bytes, then link."""
        _validate_plans(identity, plans)
        _validate_metadata(identity, metadata)
        with self._owned() as wait:
            if wait is not None:
                return wait
            ranks = self._ranks(identity, plans)
            if ranks is None:
                return None
            document = self._document(identity, metadata, plans, ranks)
            payload = canonical(document)
            if len(payload) > MAX_MANIFEST_BYTES:
                raise ValueError("Aggregate manifest exceeds its bounded envelope")
            self._sync_ranks(identity, plans, ranks)
            path = self._path(identity)
            temporary = self.manifests / ("." + secrets.token_hex(16) + ".tmp")
            created = False
            try:
                with temporary.open("xb") as file:
                    temporary.chmod(0o600)
                    file.write(payload)
                    file.flush()
                    os.fsync(file.fileno())
                try:
                    os.link(temporary, path)
                    created = True
                except FileExistsError:
                    prior = read_manifest_document(path)
                    if prior is None or canonical(prior) != payload:
                        raise
            finally:
                temporary.unlink(missing_ok=True)
            try:
                sync_directory(self.manifests)
            except OSError:
                if created:
                    path.unlink(missing_ok=True)
                raise
            return document

    def discover(
        self, identity: BundleIdentity, plans: tuple[BundlePlan, ...]
    ) -> dict | Wait | None:
        """Reopen one exact manifest candidate. The native restore verifies bytes."""
        _validate_plans(identity, plans)
        with self._owned() as wait:
            if wait is not None:
                return wait
            document = read_manifest_document(self._path(identity))
            if document is None:
                return None
            try:
                metadata = document["metadata"]
                _validate_metadata(identity, metadata)
                ranks = self._ranks(identity, plans)
                if ranks is None:
                    return None
                expected = self._document(identity, metadata, plans, ranks)
                return expected if canonical(document) == canonical(expected) else None
            except (ValueError, TypeError, KeyError):
                return None
