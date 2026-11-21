# SPDX-License-Identifier: Apache-2.0
"""Seal the selected vLLM Python sources against the official v0.34.0 install."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

BASE = "62d4bc3c7e33b1b6c906e233e6af2d3506c482ca"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--sources", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--target", required=True, help="Git revision of the overlay or patch set")
    args = cli.parse_args()
    names = ("vllm", "torch", "flashinfer-python")
    versions = {name: importlib.metadata.version(name) for name in names}
    if versions["vllm"] != "0.34.0":
        raise ValueError("Use the pinned official vLLM v0.34.0 install")
    distribution = importlib.metadata.distribution("vllm")
    stock = Path(distribution.locate_file("")).resolve()
    payloads = {}
    for path in sorted(args.sources.rglob("*.py")):
        name = path.relative_to(args.sources).as_posix()
        if not name.startswith("vllm/"):
            raise ValueError(f"Overlay must contain vllm Python sources: {name}")
        original = stock / name
        payloads[name] = {
            "base": sha(original) if original.is_file() else None,
            "overlay": sha(path),
        }
    if not payloads:
        raise ValueError("No overlay sources were selected")
    native = {
        p.relative_to(stock).as_posix(): sha(p)
        for p in sorted((stock / "vllm").rglob("*.so"))
    }
    if not native:
        raise ValueError("vLLM native extension inventory is empty")
    manifest = {
        "schema": 1,
        "python": list(sys.version_info[:2]),
        "system": platform.system(),
        "machine": platform.machine(),
        "stock_revision": BASE,
        "target_revision": args.target,
        "distributions": versions,
        "native": native,
        "python_files": payloads,
    }
    with ZipFile(args.output, "w") as archive:
        archive.writestr(
            ZipInfo("manifest.json"), json.dumps(manifest, sort_keys=True), ZIP_DEFLATED
        )
        for name in payloads:
            archive.writestr(
                ZipInfo(f"overlay/{name}"), (args.sources / name).read_bytes(), ZIP_DEFLATED
            )


if __name__ == "__main__":
    main()
