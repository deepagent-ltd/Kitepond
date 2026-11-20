# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only scoped fault at the stock FSL2 final file replace."""

import argparse
import asyncio
import errno
import fcntl
import hashlib
import json
import os
import stat
from pathlib import Path
from time import monotonic, monotonic_ns, sleep

SCHEMA = "kitepond.native-fsl2-partial-write.v1"
ARM_ENV = "KITEPOND_FSL2_PARTIAL_WRITE_ARM"


def read(path):
    return json.loads(path.read_bytes())


def emit(path, record):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())


def device_identity(path):
    s = path.stat()
    return [s.st_dev, s.st_ino]


def archive_counts(root):
    return {
        "payloads": sum(
            1 for item in root.glob("rank-*/payloads/*.data") if item.is_file()
        ),
        "temporary": sum(
            1 for item in root.glob("rank-*/payloads/.tmp/*.data") if item.is_file()
        ),
        "rank_manifests": sum(
            1 for item in root.glob("rank-*/manifests/*.json") if item.is_file()
        ),
        "aggregates": sum(
            1 for item in root.glob("aggregates/*.json") if item.is_file()
        ),
        "catalog": sum(
            1 for item in root.glob("glm-catalog-v1/*.json") if item.is_file()
        ),
    }


def identity_file(path):
    with path.open("rb") as stream:
        st = os.fstat(stream.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size <= 0:
            raise ValueError("Fault witness needs a nonempty real file")
        return {
            "name": path.name,
            "bytes": st.st_size,
            "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
        }


def create_arm(root, evidence, boot):
    root, evidence = root.resolve(strict=True), evidence.resolve(strict=True)
    if (
        root == evidence
        or evidence.is_relative_to(root)
        or not evidence.is_relative_to(root.parent)
        or any(root.iterdir())
        or root.stat().st_uid != os.geteuid()
        or stat.S_IMODE(root.stat().st_mode) != 0o700
    ):
        raise ValueError("Partial store needs an empty, dedicated writable root")
    value = {
        "schema": SCHEMA,
        "root": str(root),
        "root_identity": device_identity(root),
        "boot": boot,
    }
    arm = evidence / "arm.json"
    emit(arm, value)
    return arm


def _publish(path, record):
    temporary = path.with_suffix(".writing")
    emit(temporary, record)
    os.replace(temporary, path)


def _wait_json(path, timeout):
    until = monotonic() + timeout
    while not path.is_file():
        if monotonic() >= until:
            raise TimeoutError("Independent partial-store witness did not arrive")
        sleep(0.02)
    return read(path)


def observe(arm, timeout):
    """A separate process checks both on-disk files while the writer pauses."""
    config = read(arm)
    root = Path(config["root"])
    proposal = _wait_json(arm.parent / "pending.json", timeout)
    payloads = root / "rank-0/payloads"
    first = proposal["first"]["name"]
    second = proposal["second_temporary"]["name"]
    if (
        first != Path(first).name
        or second != Path(second).name
        or config["boot"] != proposal["boot"]
        or proposal["pid"] == os.getpid()
        or config["root_identity"] != device_identity(root)
        or identity_file(payloads / first) != proposal["first"]
        or identity_file(payloads / ".tmp" / second) != proposal["second_temporary"]
        or archive_counts(root) != proposal["observed_counts"]
    ):
        raise ValueError(
            "Independent process did not observe the proposed partial disk state"
        )
    _publish(
        arm.parent / "witness.json",
        {
            "boot": config["boot"],
            "pid": os.getpid(),
            "writer_pid": proposal["pid"],
            "observed_counts": archive_counts(root),
            "observed_ns": monotonic_ns(),
            "first_sha256": proposal["first"]["sha256"],
            "second_sha256": proposal["second_temporary"]["sha256"],
        },
    )


def _replace_once(arm, src, dst):
    config = read(arm)
    root = Path(config["root"])
    evidence = arm.parent
    if (
        config["schema"] != SCHEMA
        or not config["boot"]
        or device_identity(root) != config["root_identity"]
        or evidence.is_relative_to(root)
    ):
        raise ValueError("Fault test root changed")
    with (evidence / "replace.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        first_path = evidence / "first.json"
        event_path = evidence / "injected.json"
        if event_path.exists():
            return os.replace(src, dst)
        if not first_path.exists():
            # First finalize is an actual stock file rename of completed bytes.
            os.replace(src, dst)
            emit(
                first_path,
                {
                    "schema": SCHEMA,
                    "boot": config["boot"],
                    "pid": os.getpid(),
                    "first": identity_file(dst),
                    "observed_ns": monotonic_ns(),
                },
            )
            return None
        first = read(first_path)
        if first["boot"] != config["boot"] or first["pid"] != os.getpid():
            raise ValueError("Partial store may only affect one actual rank owner")
        first_file = dst.parent / first["first"]["name"]
        if (
            first_file == dst
            or identity_file(first_file) != first["first"]
            or dst.exists()
        ):
            raise ValueError("Missing first finalized or second staged FSL2 part")
        counts = archive_counts(root)
        if counts != {
            "payloads": 1,
            "temporary": 1,
            "rank_manifests": 0,
            "aggregates": 0,
            "catalog": 0,
        }:
            raise ValueError("Store had already escaped its isolated unpublished phase")
        proposal = {
            "schema": SCHEMA,
            "boot": config["boot"],
            "pid": os.getpid(),
            "rank": 0,
            "errno": errno.ENOSPC,
            "observed_counts": counts,
            "first": first["first"],
            "second_temporary": identity_file(src),
            "observed_ns": monotonic_ns(),
        }
        _publish(evidence / "pending.json", proposal)
        witness = _wait_json(evidence / "witness.json", 10)
        if (
            witness["writer_pid"] != os.getpid()
            or witness["pid"] == os.getpid()
            or witness["boot"] != config["boot"]
            or witness["observed_counts"] != counts
            or witness["first_sha256"] != proposal["first"]["sha256"]
            or witness["second_sha256"] != proposal["second_temporary"]["sha256"]
        ):
            raise ValueError("Partial-write witness did not match the physical files")
        emit(event_path, proposal | {"independent_witness": witness})
        raise OSError(
            errno.ENOSPC, "Test-only partial store publication fault", str(dst)
        )


def install(arm):
    """Install only in the test D subprocess environment, and return an undo."""
    from aiofiles import os as aio_os

    arm = arm.resolve(strict=True)
    config = read(arm)
    selected = Path(config["root"]) / "rank-0/payloads"
    original = aio_os.replace

    async def replace(src, dst, *args, **kwargs):
        if (
            not args
            and not kwargs
            and isinstance(src, (str, Path))
            and isinstance(dst, (str, Path))
            and Path(src).parent == selected / ".tmp"
            and Path(dst).parent == selected
            and Path(src).name == Path(dst).name
            and Path(dst).suffix == ".data"
        ):
            return await asyncio.to_thread(_replace_once, arm, Path(src), Path(dst))
        return await original(src, dst, *args, **kwargs)

    aio_os.replace = replace

    def undo():
        aio_os.replace = original

    return undo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observe", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    observe(args.observe, args.timeout)
