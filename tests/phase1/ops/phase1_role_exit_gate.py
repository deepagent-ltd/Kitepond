# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private process synchronization for the native paired-role exit trials."""

from __future__ import annotations

import asyncio
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns

GATE_ENV = "KITEPOND_GLM_ROLE_EXIT_GATE"
POSITION = "first_remote_window_verified_and_local_fenced_before_commit"


def private_directory(directory: Path) -> Path:
    mode = directory.lstat()
    if (
        not directory.is_absolute()
        or directory.resolve(strict=True) != directory
        or not stat.S_ISDIR(mode.st_mode)
        or mode.st_uid != os.geteuid()
        or mode.st_mode & 0o077
    ):
        raise ValueError("Role exit gate needs its sealed 0700 local directory")
    return directory


def emit(path: Path, value: dict) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    with path.with_suffix(".part").open("xb") as file:
        file.write(encoded)
        file.flush()
        os.fsync(file.fileno())
    # Hardlink publishes the complete file once and refuses to overwrite evidence.
    pending = path.with_suffix(".part")
    os.link(pending, path, follow_symlinks=False)
    pending.unlink()


def load(path: Path) -> dict:
    data = json.loads(path.read_bytes())
    if type(data) is not dict:
        raise ValueError("Role exit event is not an object")
    return data


@dataclass(frozen=True)
class LinuxProcess:
    pid: int
    group: int
    session: int
    start_ticks: int
    state: str

    @classmethod
    def read(cls, pid):
        try:
            row = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        except (FileNotFoundError, ProcessLookupError):
            return None
        fields = row[row.rindex(")") + 2 :].split()
        return cls(pid, int(fields[2]), int(fields[3]), int(fields[19]), fields[0])

    def live(self):
        current = self.read(self.pid)
        return (
            current is not None
            and current.start_ticks == self.start_ticks
            and current.group == self.group
            and current.session == self.session
            and current.state not in {"X", "Z"}
        )


ATTEMPT_FIELDS = (
    "request_id",
    "transfer_id",
    "epoch",
    "grant_sha256",
    "source_boot_id",
)


def selected(attempt):
    raw = os.environ.get(GATE_ENV)
    if not raw:
        return None
    directory = private_directory(Path(raw))
    target_file = directory / "target.json"
    if not target_file.is_file():
        return None
    target = load(target_file)
    if any(getattr(attempt, name) != target[name] for name in ATTEMPT_FIELDS):
        return None
    return directory, target


def source_event(writer, window=None):
    found = selected(writer.attempt)
    if found is None or (window is not None and window.sequence != 0):
        return
    directory, target = found
    process = LinuxProcess.read(os.getpid())
    if process is None:
        raise RuntimeError("Cannot bind P window to its real local process")
    attempt = writer.attempt
    event = {
        **{name: target[name] for name in ATTEMPT_FIELDS},
        "recovery_case": target["recovery_case"],
        "case_id": target["case_id"],
        "decode_boot_id": target["decode_boot_id"],
        "rank": attempt.rank,
        "worker_pid": process.pid,
        "worker_start_ticks": process.start_ticks,
        "manifest_sha256": attempt.manifest_sha256,
        "observed_monotonic_ns": monotonic_ns(),
        "writer_released": writer._released,
        "dma_uncertain": writer._dma_uncertain,
    }
    if window is not None:
        if writer._released or writer._dma_uncertain:
            raise ValueError(
                "Expected an active P writer after its verified sync write"
            )
        event["written_nbytes"] = window.nbytes
    prefix = "source-released" if window is None else "source-fenced"
    emit(directory / f"{prefix}-{attempt.rank}.json", event)


async def observe_fenced_window(attempt, window, remote_fenced, local_fenced):
    """Called only by the sealed D instrument after actual receipt and native fence."""
    found = None if window.sequence != 0 else selected(attempt)
    if found is None:
        return
    directory, target = found
    if (
        not remote_fenced
        or not local_fenced
        or attempt.rank not in range(4)
        or not 0 < window.nbytes <= 32 * 1024**2
    ):
        raise ValueError("Exit gate needs a verified and locally fenced real window")
    process = LinuxProcess.read(os.getpid())
    if process is None:
        raise RuntimeError("Cannot bind D window to its real local process")
    event = {
        **{name: target[name] for name in ATTEMPT_FIELDS},
        "recovery_case": target["recovery_case"],
        "case_id": target["case_id"],
        "decode_boot_id": target["decode_boot_id"],
        "rank": attempt.rank,
        "position": POSITION,
        "received_nbytes": window.nbytes,
        "window_sequence": 0,
        "remote_write_fenced": True,
        "local_cuda_fenced": True,
        "source_commit_pending": True,
        "destination_sha256": attempt.destination_sha256,
        "manifest_sha256": attempt.manifest_sha256,
        "worker_pid": process.pid,
        "worker_start_ticks": process.start_ticks,
        "observed_monotonic_ns": monotonic_ns(),
    }
    emit(directory / f"rank-{attempt.rank}.json", event)
    release = directory / "release.json"
    while monotonic_ns() < target["deadline_ns"]:
        if release.is_file():
            if load(release)["case_id"] != target["case_id"]:
                raise ValueError("Foreign release of the native role exit gate")
            return
        await asyncio.sleep(0.01)
    raise TimeoutError("Sealed role exit gate was not released by the controller")
