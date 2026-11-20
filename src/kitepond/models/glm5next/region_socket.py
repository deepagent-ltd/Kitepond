# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pass the existing D region to local GLM P workers without copying pages."""

from __future__ import annotations

import array
import json
import os
import socket
import stat
import struct
from dataclasses import asdict
from pathlib import Path

from .shared_region import NativeHiSparseMemfdRegion, NativeHiSparseRegionDescriptor

_MAX_PACKET = 16 * 1024
_CREDENTIALS = struct.Struct("iII")
_FD_SIZE = array.array("i").itemsize


def _scope(engine_id, boot_id, rank, layout_sha256) -> dict:
    return {
        "schema": "kitepond.glm-host-region-request.v1",
        "engine_id": engine_id,
        "boot_id": boot_id,
        "rank": rank,
        "layout_sha256": layout_sha256,
    }


def _identities(conn: socket.socket) -> tuple[int, int]:
    pid, uid, _ = _CREDENTIALS.unpack(
        conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _CREDENTIALS.size)
    )
    return pid, uid


def _endpoint(directory: Path, rank: int) -> Path:
    if type(rank) is not int or not 0 <= rank < 4:
        raise ValueError("GLM rank must be in [0, 4)")
    metadata = directory.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise ValueError("GLM region socket needs a private owner-only directory")
    return directory / f"host-region-{rank}.sock"


class RegionPublisher:
    """One owner and rank per endpoint. Launcher manages the accept loop."""

    def __init__(self, directory: Path, region: NativeHiSparseMemfdRegion):
        if not region.owner or region.descriptor.owner_pid != os.getpid():
            raise ValueError("Only the native region owner can publish its FD")
        self.region = region
        self.path = _endpoint(directory, region.descriptor.source_rank)
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            self.listener.bind(str(self.path))
            self.listener.listen(4)
        except BaseException:
            self.listener.close()
            raise
        self.inode = self.path.lstat().st_ino

    def serve_one(self, *, accept_timeout: float) -> bool:
        self.listener.settimeout(accept_timeout)
        conn, _ = self.listener.accept()
        desc = self.region.descriptor
        with conn:
            conn.settimeout(2)
            _, uid = _identities(conn)
            try:
                packet = conn.recv(_MAX_PACKET + 1)
                if uid != os.geteuid() or len(packet) > _MAX_PACKET:
                    return False
                request = json.loads(packet)
                if request != _scope(
                    desc.authority_engine_id,
                    desc.authority_boot_id,
                    desc.source_rank,
                    desc.layout_sha256,
                ):
                    return False
                socket.send_fds(
                    conn, [json.dumps(asdict(desc)).encode()], [self.region.fd]
                )
            except (OSError, ValueError, RecursionError):
                return False
        return True

    def close(self) -> None:
        self.listener.close()
        try:
            if self.path.lstat().st_ino == self.inode:
                self.path.unlink()
        except FileNotFoundError:
            pass


def join_region(
    directory: Path,
    *,
    authority_engine_id: str,
    authority_boot_id: str,
    source_rank: int,
    layout_sha256: str,
    map_bytes: int | None = None,
) -> NativeHiSparseMemfdRegion:
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as conn:
        conn.settimeout(5)
        conn.connect(str(_endpoint(directory, source_rank)))
        owner_pid, uid = _identities(conn)
        if uid != os.geteuid():
            raise PermissionError("Wrong GLM region owner UID")
        conn.sendall(
            json.dumps(
                _scope(
                    authority_engine_id, authority_boot_id, source_rank, layout_sha256
                )
            ).encode()
        )
        packet, ancdata, flags, _ = conn.recvmsg(
            _MAX_PACKET, socket.CMSG_SPACE(_FD_SIZE), socket.MSG_CMSG_CLOEXEC
        )
    fds = array.array("i")
    for level, kind, data in ancdata:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            fds.frombytes(data[: len(data) // _FD_SIZE * _FD_SIZE])
    try:
        if len(fds) != 1 or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise ValueError("Expected exactly one native region FD")
        descriptor = NativeHiSparseRegionDescriptor(**json.loads(packet))
        if descriptor.owner_pid != owner_pid:
            raise ValueError("Wrong GLM region owner PID")
        return NativeHiSparseMemfdRegion.join_received_fd(
            descriptor,
            fds.pop(),
            authority_engine_id=authority_engine_id,
            authority_boot_id=authority_boot_id,
            source_rank=source_rank,
            layout_sha256=layout_sha256,
            map_bytes=map_bytes,
        )
    finally:
        for fd in fds:
            os.close(fd)
