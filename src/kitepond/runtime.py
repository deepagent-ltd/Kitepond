# SPDX-License-Identifier: Apache-2.0
"""Run vLLM with Kitepond's version-pinned compatibility files."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

BUNDLE = Path(__file__).with_name("v034-runtime.zip")
CLI = "from kitepond.vllm_cli import run_enabled_serve; run_enabled_serve()"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _package_path(name: str) -> Path:
    parts = PurePosixPath(name).parts
    if len(parts) < 2 or parts[0] != "vllm" or ".." in parts or "\\" in name:
        raise ValueError(f"Invalid runtime bundle path: {name!r}")
    return Path(*parts[1:])


def verify_base(manifest: dict) -> Path:
    if manifest["python"] != list(sys.version_info[:2]):
        raise ValueError(f"Python must match the runtime bundle: {manifest['python']}")
    if manifest["system"] != platform.system() or manifest["machine"] != platform.machine():
        raise ValueError("This runtime bundle was built for a different operating system or architecture")
    for name, expected in manifest["distributions"].items():
        observed = importlib.metadata.version(name)
        if observed != expected:
            raise ValueError(f"Expected {name} {expected}; found {observed}")
    distribution = importlib.metadata.distribution("vllm")
    package = Path(distribution.locate_file("vllm")).resolve(strict=True)
    for name, expected in manifest["native"].items():
        path = package / _package_path(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Native vLLM file differs from the qualified image: {name}")
    return package


def assemble_tree(archive: ZipFile, manifest: dict, source: Path, target: Path) -> None:
    replacements = {}
    for name, record in manifest["python_files"].items():
        path = _package_path(name)
        if path.suffix != ".py":
            raise ValueError(f"Only Python sources may be overlaid: {name}")
        original = source / path
        expected = record["base"]
        matches = (
            not os.path.lexists(original)
            if expected is None
            else original.is_file() and file_sha256(original) == expected
        )
        if not matches:
            raise ValueError(f"Base vLLM source differs from the qualified image: {name}")
        payload = archive.read(f"overlay/{name}")
        if hashlib.sha256(payload).hexdigest() != record["overlay"]:
            raise ValueError(f"Runtime bundle source failed checksum verification: {name}")
        replacements[path] = payload

    # An ordinary Python package resolves through one directory. Preserve the
    # complete base package shape, including its compiled native extensions.
    shutil.copytree(
        source,
        target,
        copy_function=os.symlink,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for path, payload in replacements.items():
        destination = target / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            destination.unlink()
        with destination.open("xb") as stream:
            stream.write(payload)


def _run_vllm(args: list[str], overlay: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(overlay), env.get("PYTHONPATH")))
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["KITEPOND_RUNTIME_DIR"] = str(overlay)
    with subprocess.Popen([sys.executable, "-P", "-c", CLI, "serve", *args], env=env) as child:
        def forward(signum: int, _frame) -> None:
            if child.poll() is None:
                child.send_signal(signum)

        signals = (signal.SIGINT, signal.SIGTERM)
        previous = {sig: signal.signal(sig, forward) for sig in signals}
        try:
            result = child.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return result if result >= 0 else 128 - result


def serve(args: list[str]) -> int:
    try:
        with ZipFile(BUNDLE) as archive, TemporaryDirectory(prefix="kitepond-vllm-") as tmp:
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("schema") != 1:
                raise ValueError("Unsupported version of the runtime bundle")
            package = verify_base(manifest)
            overlay = Path(tmp)
            assemble_tree(archive, manifest, package, overlay / "vllm")
            return _run_vllm(args, overlay)
    except (OSError, ValueError, KeyError, BadZipFile, importlib.metadata.PackageNotFoundError) as exc:
        print(f"Kitepond runtime preflight failed: {exc}", file=sys.stderr)
        return 2
