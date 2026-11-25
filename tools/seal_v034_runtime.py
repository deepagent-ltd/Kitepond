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


def stock_install() -> tuple[dict, Path]:
    names = ("vllm", "torch", "flashinfer-python")
    versions = {name: importlib.metadata.version(name) for name in names}
    if versions["vllm"] != "0.34.0":
        raise ValueError("Use the pinned official vLLM v0.34.0 install")
    distribution = importlib.metadata.distribution("vllm")
    stock = Path(distribution.locate_file("")).resolve()
    native = {
        p.relative_to(stock).as_posix(): sha(p)
        for p in sorted((stock / "vllm").rglob("*.so"))
    }
    if not native:
        raise ValueError("vLLM native extension inventory is empty")
    return {
        "schema": 1,
        "python": list(sys.version_info[:2]),
        "system": platform.system(),
        "machine": platform.machine(),
        "stock_revision": BASE,
        "distributions": versions,
        "native": native,
    }, stock


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--sources", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--target", help="Git revision of the overlay or patch set")
    stock_input = cli.add_mutually_exclusive_group()
    stock_input.add_argument("--stock-manifest", type=Path)
    stock_input.add_argument("--capture-stock", action="store_true")
    args = cli.parse_args()

    if args.capture_stock:
        if args.sources is not None or args.target is not None:
            cli.error("--capture-stock takes only --output")
        manifest, stock = stock_install()
        manifest["python_sources"] = {
            p.relative_to(stock).as_posix(): sha(p)
            for p in sorted((stock / "vllm").rglob("*.py"))
        }
        args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return
    if args.sources is None or args.target is None:
        cli.error("--sources and --target are required when sealing an overlay")

    if args.stock_manifest is None:
        manifest, stock = stock_install()
    else:
        manifest = json.loads(args.stock_manifest.read_text())
        stock_hashes = manifest.pop("python_sources")
        if (
            manifest["schema"] != 1
            or manifest["stock_revision"] != BASE
            or manifest["distributions"]["vllm"] != "0.34.0"
            or not manifest["native"]
            or not stock_hashes
        ):
            raise ValueError("Invalid pinned vLLM v0.34.0 stock manifest")

    payloads = {}
    for path in sorted(args.sources.rglob("*.py")):
        name = path.relative_to(args.sources).as_posix()
        if not name.startswith("vllm/"):
            raise ValueError(f"Overlay must contain vllm Python sources: {name}")
        if args.stock_manifest is None:
            original = stock / name
            base_hash = sha(original) if original.is_file() else None
        else:
            base_hash = stock_hashes.get(name)
        payloads[name] = {"base": base_hash, "overlay": sha(path)}
    if not payloads:
        raise ValueError("No overlay sources were selected")
    manifest.update(target_revision=args.target, python_files=payloads)
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
