# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded inactive FSL2 retirement; manifest withdrawal precedes real deletion."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import select
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kitepond.backends.fsl2 import (
    MAX_MANIFEST_BYTES,
    SCHEMA,
    BundleIdentity,
    BundlePlan,
    ColdStorageAdapter,
    _hex,
    canonical,
    digest,
)
from kitepond.core.contracts import Wait

MAX_BUNDLES = 128
DELETE_BATCH = 64
FSL2_DELETE_SOURCE_SHA256 = (
    "6ce3f48c70b1de1edc671dbec5d4fbaf9ba1a08429a2cf30bb3ef3bc7181b7c3"
)


def sync_directory(path: Path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_binding(adapter: ColdStorageAdapter, path: Path):
    if path.is_symlink() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Untrusted cold manifest path/size")
    document = json.loads(path.read_bytes())
    if type(document) is not dict or document.get("schema") != SCHEMA:
        return None
    identity = BundleIdentity(**document["identity"])
    spec = document["plan"]
    plan = BundlePlan.from_document(spec)
    verified = adapter._discover_path(path, identity, plan)
    if verified is None or path.name != adapter._path(identity, plan).name:
        raise ValueError("Corrupt native cold manifest remains quarantined")
    return identity, plan, verified


@dataclass(eq=False)
class RetireTicket:
    issuer: ColdBundleRetirement
    identity: BundleIdentity
    plan: BundlePlan
    manifest_sha256: str
    original: Path
    journal: Path
    state_path: Path
    document: dict
    keys: list[Any]
    owns_l2_inventory: bool = True
    state: str = "PENDING"
    cancelled: bool = False
    started: bool = False
    position: int = 0
    future: Future | None = None
    batch: list[Any] = field(default_factory=list)
    deleted_bytes: int = 0
    last_error: str | None = None
    retry_after_ns: int = 0

    def cancel(self):
        self.cancelled = True


class ColdBundleRetirement:
    def __init__(self, adapter: ColdStorageAdapter):
        if type(adapter) is not ColdStorageAdapter:
            raise TypeError("Cold retirement requires the real adapter")
        # Version-pinned completion seam: public delete() swallows timeout/errors.
        if (
            not callable(getattr(adapter.backend, "_execute_delete", None))
            or not callable(getattr(adapter.backend, "_notify_keys_deleted", None))
            or not getattr(adapter.backend, "_loop", None)
        ):
            raise RuntimeError("Native FSL2 exact delete completion is unavailable")
        actual_delete = type(adapter.backend)._execute_delete
        if (
            not inspect.iscoroutinefunction(actual_delete)
            or hashlib.sha256(inspect.getsource(actual_delete).encode()).hexdigest()
            != FSL2_DELETE_SOURCE_SHA256
        ):
            raise RuntimeError(
                "Unsupported native FSL2 delete completion implementation"
            )
        self.adapter = adapter
        self.root = adapter.manifests / ".retiring"
        self.active: RetireTicket | None = None

    def pending_journals(self):
        return self.root.exists() and any(self.root.glob("*.json"))

    def _check(self, ticket):
        if type(ticket) is not RetireTicket or ticket.issuer is not self:
            raise ValueError("Foreign cold retirement ticket")
        if self.active is not ticket:
            raise ValueError("Retirement ticket is not currently issued")

    def _write_state(self, ticket, state):
        body = {
            "schema": "kitepond-cold-retirement-v1",
            "manifest_sha256": ticket.manifest_sha256,
            "state": state,
        }
        body["digest"] = digest(body)
        temp = ticket.state_path.with_suffix(".tmp")
        with temp.open("wb") as stream:
            os.chmod(temp, 0o600)
            stream.write(canonical(body))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, ticket.state_path)
        sync_directory(self.root)

    def begin(
        self, identity, plan, *, expected_manifest_sha256, owns_l2_inventory=True
    ):
        adapter = self.adapter
        if type(owns_l2_inventory) is not bool:
            raise ValueError("L2 inventory ownership requires an explicit boolean")
        if not _hex(expected_manifest_sha256):
            raise ValueError("Exact native manifest digest is required")
        with adapter.lock:
            current = self.active
            if current is not None:
                if (
                    current.identity == identity
                    and current.plan == plan
                    and current.manifest_sha256 == expected_manifest_sha256
                ):
                    if current.owns_l2_inventory != owns_l2_inventory:
                        raise ValueError(
                            "Retirement changed its L2 inventory ownership"
                        )
                    return current
                return Wait("COLD_RETIREMENT_ADMISSION_WAIT")
            if adapter.closed or adapter.closing:
                raise RuntimeError("Cold adapter is closed or draining")
            if adapter.ticket is not None or not adapter._try_namespace():
                return Wait("COLD_IO_ADMISSION_WAIT")
            try:
                adapter.authority.check_retire(identity, plan)
                if any(
                    path.is_symlink()
                    for path in (
                        adapter.root,
                        adapter.data,
                        adapter.manifests,
                        self.root,
                    )
                ):
                    raise ValueError("Native cold directories cannot be symlinks")
                original = adapter._path(identity, plan)
                self.root.mkdir(exist_ok=True, mode=0o700)
                journal = self.root / original.name
                state_path = self.root / (original.stem + ".state")
                journals = list(self.root.glob("*.json"))
                if len(journals) > 1 or journals and journal not in journals:
                    adapter._release_namespace()
                    return Wait("COLD_RETIREMENT_RECOVERY_REQUIRED")
                if journal.exists():
                    if original.exists():
                        raise RuntimeError("Withdrawn manifest was republished")
                    document = adapter._discover_path(journal, identity, plan)
                else:
                    document = adapter.discover(identity, plan)
                if document is None:
                    adapter._release_namespace()
                    return Wait("COLD_MISS")
                if document["digest"] != expected_manifest_sha256:
                    raise ValueError("Stale retirement changed its manifest digest")
                keys = [adapter._key(row["key"]) for row in document["parts"]]
                if not journal.exists():
                    if original.is_symlink():
                        raise ValueError("Manifest cannot be a symlink")
                    os.replace(original, journal)
                    sync_directory(adapter.manifests)
                    sync_directory(self.root)
                ticket = RetireTicket(
                    self,
                    identity,
                    plan,
                    document["digest"],
                    original,
                    journal,
                    state_path,
                    document,
                    keys,
                    owns_l2_inventory=owns_l2_inventory,
                )
                if state_path.exists():
                    if state_path.is_symlink() or state_path.stat().st_size > 4096:
                        raise ValueError("Untrusted native retirement state file")
                    saved = json.loads(state_path.read_bytes())
                    if (
                        type(saved) is not dict
                        or set(saved)
                        != {"schema", "manifest_sha256", "state", "digest"}
                        or saved["digest"]
                        != digest({k: v for k, v in saved.items() if k != "digest"})
                        or saved.get("schema") != "kitepond-cold-retirement-v1"
                        or saved.get("manifest_sha256") != ticket.manifest_sha256
                        or saved.get("state")
                        not in {"WITHDRAWN", "DELETING", "RETIRED"}
                    ):
                        raise ValueError("Invalid native retirement recovery state")
                    ticket.started = saved["state"] != "WITHDRAWN"
                    if saved["state"] == "RETIRED":
                        ticket.state = "RETIRED"
                    elif ticket.started:
                        ticket.keys = self._unshared(ticket)
                self.active = ticket
                if not state_path.exists():
                    try:
                        self._write_state(ticket, "WITHDRAWN")
                    except Exception as error:
                        ticket.last_error = type(error).__name__
                return ticket
            except BaseException:
                if self.active is None:
                    adapter._release_namespace()
                raise

    def _unshared(self, ticket):
        adapter = self.adapter
        targets = {adapter._file(key).name for key in ticket.keys}
        protected = set()
        count = 0
        for path in adapter.manifests.glob("*.json"):
            count += 1
            if count > MAX_BUNDLES:
                raise RuntimeError("Cold manifest inventory exceeds bounded policy")
            binding = read_binding(adapter, path)
            if binding is None:
                continue
            _, _, doc = binding
            for row in doc["parts"]:
                name = adapter._file(adapter._key(row["key"])).name
                if name in targets:
                    protected.add(name)
        return [key for key in ticket.keys if adapter._file(key).name not in protected]

    def poll(self, ticket):
        adapter = self.adapter
        with adapter.lock:
            self._check(ticket)
            if ticket.state != "PENDING":
                return ticket.state
            if ticket.future is not None:
                if not ticket.future.done():
                    return "PENDING"
                if ticket.future.cancelled():
                    ticket.last_error = "CANCELLED_IO_UNKNOWN"
                    return "PENDING"
                try:
                    removed, sizes = ticket.future.result()
                except Exception:
                    # A cancelled/failed completion is not a native I/O fence.
                    ticket.last_error = "DELETE_IO_UNKNOWN"
                    return "PENDING"
                # A new delete-only adapter never accounted persisted keys.
                # Native directory quota still consumes the actual removed bytes.
                if removed and ticket.owns_l2_inventory:
                    adapter.backend._notify_keys_deleted(removed, sizes)
                incomplete = any(adapter._file(key).exists() for key in ticket.batch)
                if not incomplete:
                    ticket.deleted_bytes += sum(sizes)
                    ticket.position += len(ticket.batch)
                ticket.future, ticket.batch = None, []
                ticket.last_error = "DELETE_INCOMPLETE" if incomplete else None
                if incomplete:
                    ticket.retry_after_ns = time.monotonic_ns() + 1_000_000_000
                    return "PENDING"
            if time.monotonic_ns() < ticket.retry_after_ns:
                return "PENDING"
            try:
                adapter.authority.check_retire(ticket.identity, ticket.plan)
                if ticket.cancelled and not ticket.started:
                    if ticket.original.exists():
                        raise RuntimeError("Cancellation cannot overwrite a manifest")
                    for key, spec, row in zip(
                        ticket.keys,
                        ticket.plan.parts,
                        ticket.document["parts"],
                        strict=True,
                    ):
                        adapter._verify_file(key, spec, row["sha256"])
                    os.link(ticket.journal, ticket.original)
                    sync_directory(adapter.manifests)
                    ticket.state = "CANCELLED"
                    return ticket.state
                if not ticket.started:
                    ticket.keys = self._unshared(ticket)
                    self._write_state(ticket, "DELETING")
                    ticket.started = True
                if ticket.position >= len(ticket.keys):
                    self._write_state(ticket, "RETIRED")
                    ticket.state = "RETIRED"
                    return ticket.state
                ticket.batch = ticket.keys[
                    ticket.position : ticket.position + DELETE_BATCH
                ]
                # The actual native coroutine/future is retained. No timeout or
                # caller completion flag can authorize reference/lock release.
                ticket.future = asyncio.run_coroutine_threadsafe(
                    adapter.backend._execute_delete(ticket.batch),
                    adapter.backend._loop,
                )
            except Exception as error:
                ticket.last_error = type(error).__name__
            return "PENDING"

    def wait(self, ticket, timeout):
        deadline = time.monotonic() + timeout
        while self.poll(ticket) == "PENDING":
            if time.monotonic() >= deadline:
                return "PENDING"
            # CPU/control wait only; a real backend future determines completion.
            select.select([], [], [], min(0.01, max(0.0, deadline - time.monotonic())))
        return ticket.state

    def reap(self, ticket):
        with self.adapter.lock:
            self._check(ticket)
            if (
                ticket.state not in {"RETIRED", "CANCELLED"}
                or ticket.future is not None
            ):
                raise RuntimeError("Native deletion remains unconfirmed")
            ticket.journal.unlink(missing_ok=True)
            ticket.state_path.unlink(missing_ok=True)
            sync_directory(self.root)
            self.active = None
            self.adapter._release_namespace()
            return ticket.state
