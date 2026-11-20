# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Store cold bundles through LMCache FSL2 using caller-owned host buffers.

The LMCache MemoryObj and filename ABI is confined to this optional backend.
FSL2 uses CPU buffers and LMCache loads its own platform-specific dependencies.
A successful restore verifies the RAM contents. The caller still owns the
native reservation and decides when execution may use the restored state.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import select
import stat
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
    _object_key_to_filename,
)
from lmcache.v1.memory_management import MemoryObj

from kitepond.core.contracts import Wait

SCHEMA = "kitepond-cold-bundle-v1"
MAX_MANIFEST_BYTES = 4 * 1024**2
MAX_PARTS = 4096


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _hex(value):
    return (
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def read_manifest_document(path: Path, limit: int = MAX_MANIFEST_BYTES) -> dict | None:
    """Read only a bounded, regular metadata file without following its link."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as file:
            info = os.fstat(file.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                return None
            data = file.read(limit + 1)
        if len(data) > limit:
            return None
        document = json.loads(data)
        return document if type(document) is dict else None
    except (OSError, ValueError, RecursionError):
        return None


@dataclass(frozen=True)
class BundleIdentity:
    namespace: str
    model_fingerprint: str
    semantic_fingerprint: str
    generation: int
    boundary_tokens: int

    def __post_init__(self):
        if (
            type(self.namespace) is not str
            or not 0 < len(self.namespace) <= 256
            or not _hex(self.model_fingerprint)
            or not _hex(self.semantic_fingerprint)
            or type(self.generation) is not int
            or self.generation < 0
            or type(self.boundary_tokens) is not int
            or self.boundary_tokens <= 0
        ):
            raise ValueError("Invalid logical bundle identity")


@dataclass(frozen=True)
class PartSpec:
    role: str
    rank: int
    index: int
    shape: tuple[int, ...]
    dtype: str
    nbytes: int

    def __post_init__(self):
        if (
            type(self.role) is not str
            or not 0 < len(self.role) <= 128
            or type(self.rank) is not int
            or not 0 <= self.rank < 2**32
            or type(self.index) is not int
            or self.index < 0
            or type(self.shape) is not tuple
            or not 1 <= len(self.shape) <= 8
            or any(type(x) is not int or x <= 0 for x in self.shape)
            or type(self.nbytes) is not int
            or self.nbytes <= 0
            or type(self.dtype) is not str
            or not self.dtype.startswith("torch.")
        ):
            raise ValueError("Invalid logical part")
        dtype = getattr(torch, self.dtype[6:], None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError("Unsupported native dtype")
        elements = 1
        for size in self.shape:
            elements *= size
        if elements * dtype.itemsize != self.nbytes:
            raise ValueError("Part shape/dtype/byte count differ")


@dataclass(frozen=True)
class BundlePlan:
    representation_schema: str
    representation_fingerprint: str
    required_roles: tuple[str, ...]
    parts: tuple[PartSpec, ...]

    def __post_init__(self):
        if (
            type(self.representation_schema) is not str
            or not self.representation_schema
            or len(self.representation_schema) > 128
            or type(self.required_roles) is not tuple
            or type(self.parts) is not tuple
            or any(not isinstance(p, PartSpec) for p in self.parts)
            or not _hex(self.representation_fingerprint)
            or not 0 < len(self.parts) <= MAX_PARTS
            or any(type(role) is not str or not role for role in self.required_roles)
            or len(set(self.required_roles)) != len(self.required_roles)
            or set(self.required_roles) != {p.role for p in self.parts}
            or len({(p.role, p.rank, p.index) for p in self.parts}) != len(self.parts)
        ):
            raise ValueError("Incomplete or duplicate required bundle roles/parts")

    def document(self):
        return json.loads(canonical(asdict(self)))

    @classmethod
    def from_document(cls, raw: dict) -> BundlePlan:
        return cls(
            raw["representation_schema"],
            raw["representation_fingerprint"],
            tuple(raw["required_roles"]),
            tuple(
                PartSpec(**(part | {"shape": tuple(part["shape"])}))
                for part in raw["parts"]
            ),
        )


def rank_manifest_name(identity: BundleIdentity, plan: BundlePlan) -> str:
    return digest({"identity": asdict(identity), "plan": plan.document()}) + ".json"


@dataclass(frozen=True)
class ColdSnapshotLease:
    authority: NativeColdAuthority
    token: object


@dataclass(frozen=True)
class RamRestoreLease:
    authority: NativeColdAuthority
    token: object


class MemoryBorrow:
    """Real native ref, retained until actual backend I/O completion."""

    def __init__(self, authority, memory: MemoryObj, released=None):
        if not isinstance(memory, MemoryObj):
            raise TypeError("Borrow requires native MemoryObj")
        self.authority, self.memory = authority, memory
        self.released = released
        self.closed = False
        memory.ref_count_up()

    def close_after_completion(self):
        if self.closed:
            raise RuntimeError("Native memory borrow already closed")
        self.closed = True
        self.memory.ref_count_down()
        if self.released is not None:
            self.released()


class NativeColdAuthority(ABC):
    """Trusted engine seam; implement actual native eligibility, never a flag."""

    def check_retire(self, identity: BundleIdentity, plan: BundlePlan):
        raise NotImplementedError(
            "Native inactive retirement qualification is required"
        )

    @abstractmethod
    def check_store(
        self, lease: ColdSnapshotLease, identity: BundleIdentity, plan: BundlePlan
    ): ...
    @abstractmethod
    def borrow_source(
        self, lease: ColdSnapshotLease, part: PartSpec
    ) -> MemoryBorrow: ...
    def checked_store_borrow(
        self,
        lease: ColdSnapshotLease,
        identity: BundleIdentity,
        plan: BundlePlan,
        part: PartSpec,
    ) -> MemoryBorrow:
        """Validate this part's ticket before borrowing its actual source bytes."""
        self.check_store(lease, identity, plan)
        return self.borrow_source(lease, part)

    @abstractmethod
    def finish_store(self, lease: ColdSnapshotLease): ...

    def abort_store(self, lease: ColdSnapshotLease):
        """Retire a drained failed snapshot; legacy authorities keep their callback."""
        self.finish_store(lease)

    @abstractmethod
    def check_restore(
        self, lease: RamRestoreLease, identity: BundleIdentity, plan: BundlePlan
    ): ...
    @abstractmethod
    def borrow_destination(
        self, lease: RamRestoreLease, part: PartSpec
    ) -> MemoryBorrow: ...
    @abstractmethod
    def finish_restore(self, lease: RamRestoreLease, verified: bool): ...


@dataclass
class Ticket:
    mode: str
    identity: BundleIdentity
    plan: BundlePlan
    lease: ColdSnapshotLease | RamRestoreLease
    manifest: dict
    state: str = "PENDING"
    cancelled: bool = False
    position: int = 0
    task_id: Any = None
    phase: str = ""
    locked_key: ObjectKey | None = None
    borrow: MemoryBorrow | None = None
    charged_bytes: int = 0
    keys: list[ObjectKey] = field(default_factory=list)
    error: str | None = None

    def cancel(self):
        self.cancelled = True


class ColdStorageAdapter:
    """One bounded caller job, streamed parts, one native FSL2 backend."""

    def __init__(
        self,
        root,
        authority: NativeColdAuthority,
        *,
        max_inflight_bytes: int,
        verify_chunk_bytes=65536,
    ):
        if not isinstance(authority, NativeColdAuthority):
            raise TypeError("Storage requires a native authority interface")
        if type(max_inflight_bytes) is not int or max_inflight_bytes <= 0:
            raise ValueError("Invalid cold I/O byte budget")
        if (
            type(verify_chunk_bytes) is not int
            or not 0 < verify_chunk_bytes <= max_inflight_bytes
        ):
            raise ValueError("Invalid verification scratch budget")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.data = self.root / "payloads"
        self.manifests = self.root / "manifests"
        self.manifests.mkdir(exist_ok=True, mode=0o700)
        self.authority = authority
        self.backend = FSL2Adapter(
            FSL2AdapterConfig(
                str(self.data), relative_tmp_dir=".tmp", use_odirect=False
            )
        )
        self.max_inflight_bytes = max_inflight_bytes
        self.verify_chunk_bytes = verify_chunk_bytes
        self.inflight_bytes = self.peak_inflight_bytes = 0
        self.ticket: Ticket | None = None
        self.closed = False
        self.closing = False
        self.lock = threading.RLock()
        self._namespace_fd = os.open(
            self.root / ".owner.lock", os.O_RDWR | os.O_CREAT, 0o600
        )
        self._namespace_held = False
        from kitepond.backends._fsl2_retirement import ColdBundleRetirement

        try:
            self.retirement = ColdBundleRetirement(self)
        except BaseException:
            self.backend.close()
            os.close(self._namespace_fd)
            raise

    def _try_namespace(self):
        if self._namespace_held:
            return True
        try:
            fcntl.flock(self._namespace_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        self._namespace_held = True
        return True

    def _release_namespace(self):
        if self._namespace_held:
            fcntl.flock(self._namespace_fd, fcntl.LOCK_UN)
            self._namespace_held = False

    def _path(self, identity, plan):
        return self.manifests / rank_manifest_name(identity, plan)

    def _check_lease(self, ticket: Ticket):
        lease = ticket.lease
        if not isinstance(lease, (ColdSnapshotLease, RamRestoreLease)):
            raise TypeError("Untyped native lease")
        if lease.authority is not self.authority:
            raise TypeError("Foreign native lease")
        if ticket.mode == "STORE" and isinstance(lease, ColdSnapshotLease):
            self.authority.check_store(lease, ticket.identity, ticket.plan)
        elif ticket.mode == "RESTORE" and isinstance(lease, RamRestoreLease):
            self.authority.check_restore(lease, ticket.identity, ticket.plan)
        else:
            raise TypeError("Untyped native lease or invalid cold operation")

    def _borrow_store_part(self, ticket: Ticket, part: PartSpec) -> MemoryBorrow:
        lease = ticket.lease
        if not isinstance(lease, (ColdSnapshotLease, RamRestoreLease)):
            raise TypeError("Untyped native lease")
        if lease.authority is not self.authority:
            raise TypeError("Foreign native lease")
        if ticket.mode != "STORE" or not isinstance(lease, ColdSnapshotLease):
            raise TypeError("Untyped native lease or invalid cold operation")
        return self.authority.checked_store_borrow(
            lease, ticket.identity, ticket.plan, part
        )

    def discover(self, identity: BundleIdentity, plan: BundlePlan):
        """Manifest candidate only; never an execution-ready hit."""
        try:
            path = self._path(identity, plan)
        except (ValueError, TypeError, AttributeError):
            return None
        return self._discover_path(path, identity, plan)

    def _discover_path(self, path, identity, plan):
        return self.validate_rank_document(read_manifest_document(path), identity, plan)

    @staticmethod
    def validate_rank_document(doc, identity, plan):
        """Validate an immutable rank manifest without a live L2 adapter."""
        try:
            if (
                type(doc) is not dict
                or set(doc) != {"schema", "identity", "plan", "parts", "digest"}
                or doc["schema"] != SCHEMA
                or canonical(doc["identity"]) != canonical(asdict(identity))
                or canonical(doc["plan"]) != canonical(plan.document())
                or len(doc["parts"]) != len(plan.parts)
            ):
                return None
            body = {k: v for k, v in doc.items() if k != "digest"}
            if doc["digest"] != digest(body):
                return None
            for position, (spec, part) in enumerate(
                zip(plan.parts, doc["parts"], strict=True)
            ):
                if (
                    set(part) != {"spec", "key", "sha256"}
                    or canonical(part["spec"]) != canonical(asdict(spec))
                    or not _hex(part["sha256"])
                ):
                    return None
                key = ColdStorageAdapter._key(part["key"])
                if (
                    key.kv_rank != spec.rank
                    or key.object_group_id != position
                    or key.model_name != "kitepond-" + identity.model_fingerprint[:16]
                    or key.cache_salt
                    != hashlib.sha256(identity.namespace.encode()).hexdigest()[:32]
                ):
                    return None
            return doc
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RecursionError,
        ):
            return None

    def begin_store(self, identity, plan, lease):
        with self.lock:
            if self.closed or self.closing:
                raise RuntimeError("Cold adapter is closed or draining")
            if self.ticket is not None or self.retirement.active is not None:
                return Wait("COLD_IO_ADMISSION_WAIT")
            if not self._try_namespace():
                return Wait("COLD_IO_ADMISSION_WAIT")
            try:
                if self.retirement.pending_journals():
                    self._release_namespace()
                    return Wait("COLD_RETIREMENT_RECOVERY_REQUIRED")
                ticket = Ticket(
                    "STORE",
                    identity,
                    plan,
                    lease,
                    {
                        "schema": SCHEMA,
                        "identity": asdict(identity),
                        "plan": plan.document(),
                        "parts": [],
                    },
                )
                self._check_lease(ticket)
                if self._path(identity, plan).exists():
                    raise FileExistsError("Immutable cold bundle already exists")
                self.ticket = ticket
                self._advance(ticket)
                return ticket
            except BaseException:
                if self.ticket is None:
                    self._release_namespace()
                raise

    def begin_restore(
        self, identity, plan, reservation, *, expected_manifest_digest=None
    ):
        """Authorize data I/O only for the selected exact rank manifest."""
        if expected_manifest_digest is not None and not _hex(expected_manifest_digest):
            raise ValueError("Invalid expected cold rank manifest digest")
        with self.lock:
            if self.closed or self.closing:
                raise RuntimeError("Cold adapter is closed or draining")
            if self.ticket is not None or self.retirement.active is not None:
                return Wait("COLD_IO_ADMISSION_WAIT")
            if not self._try_namespace():
                return Wait("COLD_IO_ADMISSION_WAIT")
            try:
                if self.retirement.pending_journals():
                    self._release_namespace()
                    return Wait("COLD_RETIREMENT_RECOVERY_REQUIRED")
                manifest = self.discover(identity, plan)
                if manifest is None or (
                    expected_manifest_digest is not None
                    and manifest["digest"] != expected_manifest_digest
                ):
                    self._release_namespace()
                    return Wait("COLD_MISS")
                if reservation is None:
                    self._release_namespace()
                    return Wait("RAM_ADMISSION_WAIT")
                ticket = Ticket("RESTORE", identity, plan, reservation, manifest)
                self._check_lease(ticket)
                ticket.keys = [self._key(part["key"]) for part in manifest["parts"]]
                self.ticket = ticket
                self._advance(ticket)
                return ticket
            except BaseException:
                if self.ticket is None:
                    self._release_namespace()
                raise

    @staticmethod
    def _key(doc):
        if set(doc) != {
            "chunk_hash",
            "model_name",
            "kv_rank",
            "object_group_id",
            "cache_salt",
        }:
            raise ValueError("Invalid stored object key")
        if (
            not _hex(doc["chunk_hash"])
            or type(doc["model_name"]) is not str
            or type(doc["cache_salt"]) is not str
            or type(doc["kv_rank"]) is not int
            or not 0 <= doc["kv_rank"] < 2**32
            or type(doc["object_group_id"]) is not int
            or not 0 <= doc["object_group_id"] < MAX_PARTS
        ):
            raise ValueError("Invalid stored object key types")
        return ObjectKey(
            chunk_hash=bytes.fromhex(doc["chunk_hash"]),
            model_name=doc["model_name"],
            kv_rank=doc["kv_rank"],
            object_group_id=doc["object_group_id"],
            cache_salt=doc["cache_salt"],
        )

    @staticmethod
    def _key_doc(key):
        return {
            "chunk_hash": key.chunk_hash.hex(),
            "model_name": key.model_name,
            "kv_rank": key.kv_rank,
            "object_group_id": key.object_group_id,
            "cache_salt": key.cache_salt,
        }

    def _file(self, key):
        # Version-pinned native FSL2 filename seam, isolated to this backend.
        return self.data / _object_key_to_filename(key)

    def _verify_file(self, key, spec, checksum):
        path = self._file(key)
        if path.stat().st_size != spec.nbytes:
            raise ValueError("Cold payload length mismatch")
        h = hashlib.sha256()
        with path.open("rb", buffering=0) as stream:
            remaining = spec.nbytes
            while remaining:
                piece = stream.read(min(remaining, self.verify_chunk_bytes))
                if not piece:
                    raise ValueError("Cold payload became short")
                h.update(piece)
                remaining -= len(piece)
                del piece  # Do not overlap two verification chunks.
            if stream.read(1):
                raise ValueError("Cold payload has trailing bytes")
        if h.hexdigest() != checksum:
            raise ValueError("Cold payload checksum mismatch")

    def _memory(self, borrow, spec):
        if (
            not isinstance(borrow, MemoryBorrow)
            or borrow.authority is not self.authority
            or borrow.closed
        ):
            raise TypeError("Native part borrow is invalid")
        obj = borrow.memory
        tensor = obj.raw_tensor
        if tensor is not None and tensor.device.type != "cpu":
            raise ValueError("FSL2 needs caller-owned host bytes")
        if (
            obj.get_size() != spec.nbytes
            or len(obj.byte_array) != spec.nbytes
            or tuple(obj.get_shape()) != spec.shape
            or str(obj.get_dtype()) != spec.dtype
        ):
            raise ValueError("Native part representation mismatch")
        return obj

    def _release_part(self, ticket):
        if ticket.borrow is not None:
            ticket.borrow.close_after_completion()
            ticket.borrow = None
        self.inflight_bytes -= ticket.charged_bytes
        ticket.charged_bytes = 0
        ticket.task_id = None

    def _submit(self, ticket, phase, submit, *args):
        # Without a task receipt, a raised SDK call cannot prove that I/O was
        # never admitted. Preserve the key/borrow and expose a pending ticket.
        ticket.phase = phase + "_SUBMIT_UNKNOWN"
        try:
            task_id = submit(*args)
        except Exception as error:
            ticket.error = type(error).__name__
        else:
            if task_id is None:
                ticket.error = "MissingTaskReceipt"
            else:
                ticket.task_id = task_id
                ticket.phase = phase

    def _settle_restore_part(self, ticket, state=None):
        self._release_part(ticket)
        key = ticket.locked_key
        assert key is not None
        try:
            self.backend.submit_unlock([key])
        except Exception as error:
            # Do not repeat an unlock that may already have been accepted.
            ticket.phase = "UNLOCK_UNKNOWN"
            ticket.error = type(error).__name__
            return "PENDING"
        ticket.locked_key = None
        if state is not None:
            return self._finish(ticket, state)
        ticket.position += 1
        self._advance(ticket)
        return ticket.state

    def _advance(self, ticket):
        try:
            if ticket.cancelled:
                return self._finish(ticket, "CANCELLED")
            if ticket.mode != "STORE" or ticket.position >= len(ticket.plan.parts):
                self._check_lease(ticket)
            if ticket.position == len(ticket.plan.parts):
                if ticket.mode == "STORE":
                    body = dict(ticket.manifest)
                    body["digest"] = digest(ticket.manifest)
                    data = canonical(body)
                    if len(data) > MAX_MANIFEST_BYTES:
                        raise ValueError("Cold manifest exceeds bounded size")
                    target = self._path(ticket.identity, ticket.plan)
                    temp = target.with_name(
                        target.name + "." + secrets.token_hex(16) + ".tmp"
                    )
                    try:
                        with temp.open("xb") as stream:
                            os.chmod(temp, 0o600)
                            stream.write(data)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.link(
                            temp, target
                        )  # atomic publish, never overwrite immutable generation
                    finally:
                        temp.unlink(missing_ok=True)
                    return self._finish(ticket, "COMMITTED")
                return self._finish(ticket, "VERIFIED")
            spec = ticket.plan.parts[ticket.position]
            charge = spec.nbytes + min(spec.nbytes, self.verify_chunk_bytes)
            if charge > self.max_inflight_bytes:
                raise ValueError("Split caller part to fit cold I/O byte budget")
            self.inflight_bytes = ticket.charged_bytes = charge
            self.peak_inflight_bytes = max(self.peak_inflight_bytes, charge)
            if ticket.mode == "STORE":
                ticket.borrow = self._borrow_store_part(ticket, spec)
                obj = self._memory(ticket.borrow, spec)
                checksum = hashlib.sha256(obj.byte_array).hexdigest()
                key = ObjectKey(
                    chunk_hash=hashlib.sha256(
                        canonical(
                            {
                                "identity": asdict(ticket.identity),
                                "plan": ticket.plan.document(),
                                "part": asdict(spec),
                                "attempt": secrets.token_hex(16),
                            }
                        )
                    ).digest(),
                    model_name="kitepond-" + ticket.identity.model_fingerprint[:16],
                    kv_rank=spec.rank,
                    object_group_id=ticket.position,
                    cache_salt=hashlib.sha256(
                        ticket.identity.namespace.encode()
                    ).hexdigest()[:32],
                )
                ticket.keys.append(key)
                ticket.manifest["parts"].append(
                    {
                        "spec": json.loads(canonical(asdict(spec))),
                        "key": self._key_doc(key),
                        "sha256": checksum,
                    }
                )
                self._submit(
                    ticket, "STORE", self.backend.submit_store_task, [key], [obj]
                )
            else:
                key = ticket.keys[ticket.position]
                dtype = getattr(torch, spec.dtype[6:])
                layout = {
                    key.object_group_id: MemoryLayoutDesc(
                        shapes=[torch.Size(spec.shape)], dtypes=[dtype]
                    )
                }
                self._submit(
                    ticket,
                    "LOOKUP",
                    self.backend.submit_lookup_and_lock_task,
                    [key],
                    layout,
                )
        except Exception as error:
            ticket.error = type(error).__name__
            self._release_part(ticket)
            self._finish(ticket, "FAILED" if ticket.mode == "STORE" else "MISS")

    def poll(self, ticket: Ticket):
        with self.lock:
            if ticket.state != "PENDING":
                return ticket.state
            if self.ticket is not ticket:
                raise ValueError("Foreign cold I/O ticket")
            if ticket.phase == "STORE":
                results = self.backend.pop_completed_store_tasks()
                result = results.get(ticket.task_id)
                if result is None:
                    return "PENDING"
                spec = ticket.plan.parts[ticket.position]
                try:
                    if (
                        not result.is_successful()
                        or result.bytes_transferred() != spec.nbytes
                    ):
                        raise OSError("Native store did not complete exact bytes")
                    if not ticket.cancelled:
                        self._verify_file(
                            ticket.keys[-1],
                            spec,
                            ticket.manifest["parts"][-1]["sha256"],
                        )
                except Exception as error:
                    ticket.error = type(error).__name__
                    self._release_part(ticket)
                    return self._finish(ticket, "FAILED")
                self._release_part(ticket)
                ticket.position += 1
                self._advance(ticket)
            elif ticket.phase == "LOOKUP":
                result = self.backend.query_lookup_and_lock_result(ticket.task_id)
                if result is None:
                    return "PENDING"
                if not result.test(0):
                    self._release_part(ticket)
                    return self._finish(
                        ticket, "CANCELLED" if ticket.cancelled else "MISS"
                    )
                key = ticket.keys[ticket.position]
                ticket.task_id = None
                ticket.locked_key = key
                if ticket.cancelled:
                    return self._settle_restore_part(ticket, "CANCELLED")
                try:
                    self._check_lease(ticket)
                    spec = ticket.plan.parts[ticket.position]
                    if self._file(key).stat().st_size != spec.nbytes:
                        raise ValueError("Cold payload length mismatch")
                    lease = ticket.lease
                    if not isinstance(lease, RamRestoreLease):
                        raise TypeError("Load requires its native restore lease")
                    ticket.borrow = self.authority.borrow_destination(lease, spec)
                    obj = self._memory(ticket.borrow, spec)
                except Exception as error:
                    ticket.error = type(error).__name__
                    return self._settle_restore_part(ticket, "MISS")
                self._submit(
                    ticket, "LOAD", self.backend.submit_load_task, [key], [obj]
                )
            elif ticket.phase == "LOAD":
                result = self.backend.query_load_result(ticket.task_id)
                if result is None:
                    return "PENDING"
                try:
                    spec = ticket.plan.parts[ticket.position]
                    if (
                        self._file(ticket.keys[ticket.position]).stat().st_size
                        != spec.nbytes
                    ):
                        raise ValueError("Cold payload length changed during load")
                    borrow = ticket.borrow
                    if borrow is None:
                        raise RuntimeError(
                            "Completed load lost its native memory borrow"
                        )
                    if (
                        not result.test(0)
                        or hashlib.sha256(borrow.memory.byte_array).hexdigest()
                        != ticket.manifest["parts"][ticket.position]["sha256"]
                    ):
                        raise ValueError("Native load or checksum validation failed")
                except Exception as error:
                    ticket.error = type(error).__name__
                    return self._settle_restore_part(ticket, "MISS")
                return self._settle_restore_part(ticket)
            return ticket.state

    def _finish(self, ticket, state):
        assert ticket.task_id is None and ticket.borrow is None
        assert ticket.locked_key is None
        if ticket.mode == "STORE":
            if state == "COMMITTED":
                self.authority.finish_store(ticket.lease)
            else:
                self.authority.abort_store(ticket.lease)
            if state != "COMMITTED" and ticket.keys:
                self.backend.delete(ticket.keys)
        else:
            self.authority.finish_restore(ticket.lease, state == "VERIFIED")
        self.ticket = None
        self._release_namespace()
        ticket.state = state
        return state

    def wait(self, ticket, timeout):
        deadline = time.monotonic() + timeout
        while True:
            state = self.poll(ticket)
            if state != "PENDING":
                return state
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "PENDING"  # never cancel I/O or drop native refs on timeout
            fds = [
                self.backend.get_store_event_fd(),
                self.backend.get_lookup_and_lock_event_fd(),
                self.backend.get_load_event_fd(),
            ]
            ready, _, _ = select.select(fds, [], [], remaining)
            for fd in ready:
                os.read(fd, 8)

    def close(self, timeout=0):
        with self.lock:
            if self.closed:
                return True
            self.closing = True
            ticket = self.ticket
        if ticket is not None and self.wait(ticket, timeout) == "PENDING":
            return False
        retirement = self.retirement.active
        if (
            retirement is not None
            and self.retirement.wait(retirement, timeout) == "PENDING"
        ):
            return False
        self.backend.close()  # only after all borrowed I/O and delete futures completed
        self._release_namespace()
        os.close(self._namespace_fd)
        self.closed = True
        return True
