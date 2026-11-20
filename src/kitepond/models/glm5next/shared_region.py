# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Linux shared memory backing for GLM HiSparse.

A memfd mapping can back the existing native HiSparse tensor without consuming
/dev/shm mount capacity. Joining never creates, truncates or prefaults a region.
Native page allocation, refs, publication and GPU registration remain separate.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import mmap
import os
from dataclasses import asdict, dataclass
from typing import Any


def _start_ticks(pid: int) -> int:
    # /proc/<pid>/stat comm may contain spaces and parentheses.
    with open(f"/proc/{pid}/stat", encoding="utf-8") as stream:
        stat = stream.read()
    return int(stat[stat.rindex(")") + 2 :].split()[19])


def _hex64(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class NativeHiSparseRegionDescriptor:
    schema: str
    authority_engine_id: str
    authority_boot_id: str
    source_rank: int
    owner_pid: int
    owner_start_ticks: int
    device_id: int
    inode: int
    nbytes: int
    layout_sha256: str

    def validate(self) -> None:
        if self.schema != "glm53f-hisparse-authority-region-v1":
            raise ValueError("Wrong authority-region schema")
        for name in ("authority_engine_id", "authority_boot_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"Missing {name}")
        for name in (
            "source_rank",
            "owner_pid",
            "owner_start_ticks",
            "device_id",
            "inode",
            "nbytes",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.source_rank < 4:
            raise ValueError("The first authority candidate requires DCP4")
        if min(self.owner_pid, self.owner_start_ticks, self.inode, self.nbytes) <= 0:
            raise ValueError("Invalid authority-region identity")
        if self.device_id < 0 or self.nbytes % mmap.PAGESIZE:
            raise ValueError("Invalid authority-region extent")
        if not _hex64(self.layout_sha256):
            raise ValueError("Missing layout fingerprint")

    def fingerprint(self) -> str:
        self.validate()
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class NativeHiSparseMemfdRegion:
    """Joinable allocation backend only; never a second KV block pool."""

    def __init__(
        self,
        fd: int,
        mapping: mmap.mmap,
        descriptor: NativeHiSparseRegionDescriptor,
        *,
        owner: bool,
    ) -> None:
        self.fd = fd
        self.mapping = mapping
        self.descriptor = descriptor
        self.owner = owner

    @classmethod
    def create(
        cls,
        nbytes: int,
        *,
        authority_engine_id: str,
        authority_boot_id: str,
        source_rank: int,
        layout_sha256: str,
    ) -> NativeHiSparseMemfdRegion:
        if type(nbytes) is not int or nbytes <= 0 or nbytes % mmap.PAGESIZE:
            raise ValueError("Native host extent must be positive and page aligned")
        # The production caller must run native check_hisparse_host_memory first.
        fd = os.memfd_create(
            f"glm-hisparse-r{source_rank}", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
        )
        mapping = None
        try:
            os.ftruncate(fd, nbytes)
            fcntl.fcntl(
                fd,
                fcntl.F_ADD_SEALS,
                fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SEAL,
            )
            stat = os.fstat(fd)
            descriptor = NativeHiSparseRegionDescriptor(
                schema="glm53f-hisparse-authority-region-v1",
                authority_engine_id=authority_engine_id,
                authority_boot_id=authority_boot_id,
                source_rank=source_rank,
                owner_pid=os.getpid(),
                owner_start_ticks=_start_ticks(os.getpid()),
                device_id=stat.st_dev,
                inode=stat.st_ino,
                nbytes=nbytes,
                layout_sha256=layout_sha256,
            )
            descriptor.validate()
            mapping = mmap.mmap(fd, nbytes, flags=mmap.MAP_SHARED)
            return cls(fd, mapping, descriptor, owner=True)
        except BaseException:
            if mapping is not None:
                mapping.close()
            os.close(fd)
            raise

    @classmethod
    def join_received_fd(
        cls,
        descriptor: NativeHiSparseRegionDescriptor,
        received_fd: int,
        *,
        authority_engine_id: str,
        authority_boot_id: str,
        source_rank: int,
        layout_sha256: str,
        map_bytes: int | None = None,
    ) -> NativeHiSparseMemfdRegion:
        # Own the received fd on entry, including all validation failures.
        fd = received_fd
        mapping = None
        try:
            descriptor.validate()
            if (
                descriptor.authority_engine_id != authority_engine_id
                or descriptor.authority_boot_id != authority_boot_id
                or descriptor.source_rank != source_rank
                or descriptor.layout_sha256 != layout_sha256
            ):
                raise ValueError("Authority region is from a different owner or layout")
            if _start_ticks(descriptor.owner_pid) != descriptor.owner_start_ticks:
                raise ValueError("Authority PID was reused")
            stat = os.fstat(fd)
            if (stat.st_dev, stat.st_ino, stat.st_size) != (
                descriptor.device_id,
                descriptor.inode,
                descriptor.nbytes,
            ):
                raise ValueError("Authority fd identity or extent changed")
            if _start_ticks(descriptor.owner_pid) != descriptor.owner_start_ticks:
                raise ValueError("Authority PID changed while joining")
            size = descriptor.nbytes if map_bytes is None else map_bytes
            if (
                type(size) is not int
                or not 0 < size <= descriptor.nbytes
                or size % mmap.PAGESIZE
            ):
                raise ValueError("Invalid native region mapping prefix")
            mapping = mmap.mmap(fd, size, flags=mmap.MAP_SHARED)
            return cls(fd, mapping, descriptor, owner=False)
        except BaseException:
            if mapping is not None:
                mapping.close()
            if fd >= 0:
                os.close(fd)
            raise

    def tensor(self) -> Any:
        import torch

        return torch.frombuffer(self.mapping, dtype=torch.uint8)

    def close(self) -> None:
        # Caller must release all tensor views/registered ranges and settle reads.
        self.mapping.close()
        os.close(self.fd)
