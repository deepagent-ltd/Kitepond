# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect a Kitepond wheel without extracting or importing it."""

import binascii
import csv
import hashlib
import io
import stat
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

DIST_NAME = "kitepond"


def is_hex(value, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and set(value) <= set("0123456789abcdef")
    )


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _member_paths(archive: zipfile.ZipFile) -> tuple[set[str], str]:
    members = set()
    metadata_roots = set()
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        if (
            not path.parts
            or info.orig_filename != name
            or str(path) != name
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or info.is_dir()
            or stat.S_ISLNK(info.external_attr >> 16)
            or any(ord(char) < 32 for char in name)
            or name in members
        ):
            raise ValueError(f"Ambiguous or unsafe wheel member {name!r}")
        root = path.parts[0]
        if root.startswith(f"{DIST_NAME}-") and root.endswith(".dist-info"):
            metadata_roots.add(root)
        elif root != DIST_NAME:
            raise ValueError(f"Unexpected wheel namespace {root!r}")
        if path.suffix in {".pth", ".pyc", ".pyo"}:
            raise ValueError(f"Unexpected startup or bytecode member {name!r}")
        members.add(name)
    if len(metadata_roots) != 1:
        raise ValueError("Exactly one Kitepond .dist-info directory is required")
    return members, metadata_roots.pop()


def _verify_record(archive: zipfile.ZipFile, members: set[str], path: str) -> None:
    reader = csv.reader(io.StringIO(archive.read(path).decode("utf-8"), newline=""))
    seen = set()
    for row in reader:
        if len(row) != 3:
            raise ValueError("Wheel RECORD must have three columns")
        name, digest, size = row
        if name not in members or name in seen:
            raise ValueError(f"Unknown or duplicate RECORD member {name!r}")
        seen.add(name)
        if name == path:
            if digest or size:
                raise ValueError(
                    "Wheel RECORD self-entry must have empty hash and size"
                )
            continue
        with archive.open(name) as stream:
            raw_digest = hashlib.file_digest(stream, "sha256").digest()
        encoded = binascii.b2a_base64(raw_digest, newline=False).decode("ascii")
        expected = encoded.rstrip("=").replace("+", "-").replace("/", "_")
        if digest != f"sha256={expected}" or size != str(
            archive.getinfo(name).file_size
        ):
            raise ValueError(f"Wheel RECORD does not match member {name!r}")
    if seen != members:
        raise ValueError("Wheel RECORD does not cover every archived file")


def inspect_wheel(wheel: Path) -> dict:
    """Return the audited release identity and sorted member paths."""
    if wheel.suffix != ".whl":
        raise ValueError("Expected a .whl artifact")
    with zipfile.ZipFile(wheel) as archive:
        members, root = _member_paths(archive)
        metadata = BytesParser().parsebytes(archive.read(f"{root}/METADATA"))
        wheel_metadata = BytesParser().parsebytes(archive.read(f"{root}/WHEEL"))
        version = metadata["Version"]
        if (
            metadata["Name"] != DIST_NAME
            or not version
            or root != f"{DIST_NAME}-{version}.dist-info"
        ):
            raise ValueError("Wheel distribution identity does not match Kitepond")
        filename_fields = wheel.stem.split("-")
        tags = wheel_metadata.get_all("Tag", [])
        if (
            len(filename_fields) != 5
            or filename_fields[:2] != [DIST_NAME, version]
            or tags != ["-".join(filename_fields[-3:])]
        ):
            raise ValueError("Wheel filename and metadata tags do not match")
        if (
            any(name.endswith((".so", ".pyd")) for name in members)
            and filename_fields[-1] == "any"
        ):
            raise ValueError("A wheel with native members needs a platform tag")
        _verify_record(archive, members, f"{root}/RECORD")
    return {
        "name": DIST_NAME,
        "version": version,
        "filename": wheel.name,
        "size_bytes": wheel.stat().st_size,
        "sha256": sha256_file(wheel),
        "tag": tags[0],
        "members": sorted(members),
    }
