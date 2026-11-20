# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline release verification and read-only preflight commands."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import zipfile
from itertools import combinations
from pathlib import Path

from kitepond.ops.acceptance import evaluate_release
from kitepond.ops.capture import capture_snapshot
from kitepond.ops.diagnose import diagnostic_report
from kitepond.ops.independent import evaluate_independent_glm
from kitepond.ops.report import operational_report
from kitepond.ops.wheel import inspect_wheel, is_hex, sha256_file

SCHEMA = "kitepond.release-manifest.v1"


def _pinned_image(value) -> bool:
    if not isinstance(value, str):
        return False
    name, _, digest = value.rpartition("@sha256:")
    return (
        bool(name)
        and is_hex(digest, 64)
        and name.isascii()
        and all(char.isalnum() or char in "./_:-" for char in name)
    )


def _python_version(value) -> bool:
    if not isinstance(value, str):
        return False
    major, dot, minor = value.partition(".")
    return bool(dot) and major.isdecimal() and minor.isdecimal() and value.isascii()


def _validate_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported release manifest schema")
    base = manifest.get("base")
    if (
        not isinstance(base, dict)
        or not _pinned_image(base.get("image"))
        or not isinstance(base.get("vllm"), str)
        or not base["vllm"]
        or not _python_version(base.get("python"))
    ):
        raise ValueError("Release base must pin an image digest, vLLM and Python")
    source = manifest.get("source_commit")
    if not is_hex(source, 40):
        raise ValueError("Release source must be a full Git commit ID")
    wheel = manifest.get("wheel")
    if (
        not isinstance(wheel, dict)
        or not isinstance(wheel.get("version"), str)
        or not wheel["version"]
        or not is_hex(wheel.get("sha256"), 64)
    ):
        raise ValueError("Release wheel must provide a version and SHA-256 digest")
    return manifest


def _read_manifest(path: Path) -> dict:
    return _validate_manifest(json.loads(path.read_text(encoding="utf-8")))


def _seal(args) -> dict:
    return _validate_manifest(
        {
            "schema": SCHEMA,
            "source_commit": args.source_commit,
            "base": {
                "image": args.base_image,
                "vllm": args.base_vllm_version,
                "python": args.base_python_version,
            },
            "wheel": inspect_wheel(args.wheel),
        }
    )


def _verify(args) -> dict:
    manifest = _read_manifest(args.manifest)
    if inspect_wheel(args.wheel) != manifest["wheel"]:
        raise ValueError("Wheel does not match the sealed manifest")
    return {"wheel_verified": True, "manifest_sha256": sha256_file(args.manifest)}


def _installed_distribution(name: str) -> importlib.metadata.Distribution | None:
    try:
        return importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _engine_files(vllm: importlib.metadata.Distribution, origin: str) -> dict:
    members = sorted(str(path) for path in vllm.files or () if path.parts[0] == "vllm")
    if "vllm/__init__.py" not in members or len(members) != len(set(members)):
        raise ValueError("The vLLM distribution file inventory is unavailable")
    fingerprint = hashlib.sha256()
    for member in members:
        with vllm.locate_file(member).open("rb") as stream:
            content = hashlib.file_digest(stream, "sha256").digest()
        fingerprint.update(member.encode("utf-8"))
        fingerprint.update(b"\x00")
        fingerprint.update(content)
    return {
        "version": vllm.version,
        "origin": origin,
        "vllm_file_count": len(members),
        "vllm_files_sha256": fingerprint.hexdigest(),
    }


def _doctor(args) -> dict:
    manifest = _read_manifest(args.manifest)
    base = manifest["base"]
    engine = importlib.util.find_spec("vllm")
    vllm = _installed_distribution("vllm")
    kitepond = _installed_distribution("kitepond")
    installed_vllm_origin = (
        Path(vllm.locate_file("vllm/__init__.py")).resolve() if vllm else None
    )
    observed = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "vllm": vllm.version if vllm else None,
        "kitepond": kitepond.version if kitepond else None,
        "vllm_origin": None if engine is None else engine.origin,
        "vllm_distribution_origin": (
            str(installed_vllm_origin) if installed_vllm_origin is not None else None
        ),
    }
    matches = {
        "python": observed["python"] == base["python"],
        "vllm": observed["vllm"] == base["vllm"],
        "vllm_origin": (
            engine is not None
            and engine.origin is not None
            and Path(engine.origin).resolve() == installed_vllm_origin
        ),
        "kitepond": (
            kitepond is not None
            and kitepond.version == manifest["wheel"]["version"]
            and Path(kitepond.locate_file("kitepond/ops/cli.py")).resolve()
            == Path(__file__).resolve()
        ),
    }
    result = {
        "scope": "installed Python metadata and module location",
        "manifest_sha256": sha256_file(args.manifest),
        "observed": observed,
        "matches": matches,
        "compatible_metadata": all(matches.values()),
    }
    if args.engine_files:
        result["engine_files"] = (
            _engine_files(vllm, engine.origin)
            if matches["vllm_origin"] and vllm and engine and engine.origin
            else None
        )
    if args.engine_hooks:
        result["engine_source_capabilities"] = None
        if matches["vllm_origin"]:
            from kitepond.integrations.vllm.compat import inspect_engine

            result["engine_source_capabilities"] = inspect_engine()
    return result


def _plan(args) -> dict:
    manifest = _read_manifest(args.manifest)
    paths = {
        "model": args.model_dir.resolve(),
        "state": args.state_root.resolve(),
        "cold": args.cold_root.resolve(),
    }
    for path in paths.values():
        if path.exists() and not path.is_dir():
            raise ValueError(f"Deployment root is not a directory: {path}")
    for left, right in combinations(paths.values(), 2):
        if left.is_relative_to(right) or right.is_relative_to(left):
            raise ValueError("Model, state and cold roots must be disjoint")
    model_files = ["config.json", "model.safetensors.index.json"]
    model_files.extend(
        name
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        if (paths["model"] / name).is_file()
    )
    if (
        len(set(args.prefill_gpus + args.decode_gpus)) != 8
        or min(args.prefill_gpus + args.decode_gpus) < 0
    ):
        raise ValueError("Prefill and decode require eight distinct GPU indices")
    return {
        "scope": "read-only deployment plan",
        "manifest_sha256": sha256_file(args.manifest),
        "source_commit": manifest["source_commit"],
        "base": manifest["base"],
        "wheel_sha256": manifest["wheel"]["sha256"],
        "paths": {role: str(path) for role, path in paths.items()},
        "model_files": {
            name: sha256_file(paths["model"] / name) for name in model_files
        },
        "roles": {"prefill": args.prefill_gpus, "decode": args.decode_gpus},
    }


def _capture(args) -> dict:
    manifest = args.manifest.read_bytes()
    _validate_manifest(json.loads(manifest))
    return capture_snapshot(args.sources, hashlib.sha256(manifest).hexdigest())


def _report(args) -> dict:
    current = args.current.read_bytes()
    previous = args.previous.read_bytes() if args.previous else None
    report = operational_report(
        json.loads(current),
        None if previous is None else json.loads(previous),
        sha256_file(args.manifest),
    )
    report["snapshot_sha256"] = {
        "current": hashlib.sha256(current).hexdigest(),
        "previous": None if previous is None else hashlib.sha256(previous).hexdigest(),
    }
    return report


def _accept(args) -> dict:
    return evaluate_release(args.index, args.manifest, args.wheel, _read_manifest)


def _accept_glm(args) -> dict:
    return evaluate_independent_glm(args.index)


def _diagnose(args) -> dict:
    return diagnostic_report(
        _read_manifest(args.manifest),
        sha256_file(args.manifest),
        current=args.current,
        previous=args.previous,
        doctor=args.doctor,
        acceptance=args.acceptance,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kitepond")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="Serve with vLLM. Add --enable to activate Kitepond")
    seal = commands.add_parser("seal", help="Generate a release artifact manifest")
    seal.add_argument("--wheel", type=Path, required=True)
    seal.add_argument("--source-commit", required=True)
    seal.add_argument("--base-image", required=True)
    seal.add_argument("--base-vllm-version", required=True)
    seal.add_argument("--base-python-version", required=True)

    verify = commands.add_parser(
        "verify", help="Check wheel contents against a manifest"
    )
    verify.add_argument("--wheel", type=Path, required=True)
    doctor = commands.add_parser("doctor", help="Check installed Python metadata")
    doctor.add_argument("--engine-hooks", action="store_true")
    doctor.add_argument(
        "--engine-files",
        action="store_true",
        help="Hash vLLM distribution files for comparison with the base image",
    )
    plan = commands.add_parser("plan", help="Print a deployment plan without changes")
    plan.add_argument("--model-dir", type=Path, required=True)
    plan.add_argument("--state-root", type=Path, required=True)
    plan.add_argument("--cold-root", type=Path, required=True)
    plan.add_argument("--prefill-gpus", nargs=4, type=int, required=True)
    plan.add_argument("--decode-gpus", nargs=4, type=int, required=True)
    report = commands.add_parser("report", help="Report captured operational evidence")
    report.add_argument("--current", type=Path, required=True)
    report.add_argument("--previous", type=Path)
    capture = commands.add_parser(
        "capture", help="Capture same-host operational evidence"
    )
    capture.add_argument("--sources", type=Path, required=True)
    accept = commands.add_parser(
        "accept", help="Evaluate captured Phase 1 release evidence"
    )
    accept.add_argument("--index", type=Path, required=True)
    accept.add_argument("--wheel", type=Path, required=True)
    standalone = commands.add_parser(
        "accept-glm", help="Evaluate captured independent GLM evidence"
    )
    standalone.add_argument("--index", type=Path, required=True)
    diagnose = commands.add_parser(
        "diagnose", help="Summarize captured Phase 1 operations without private paths"
    )
    diagnose.add_argument("--current", type=Path)
    diagnose.add_argument("--previous", type=Path)
    diagnose.add_argument("--doctor", type=Path)
    diagnose.add_argument("--acceptance", type=Path)
    for command in (verify, doctor, plan, report, capture, accept, diagnose):
        command.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    option_end = argv.index("--") if "--" in argv else len(argv)
    if argv[:1] == ["serve"] and "--enable" in argv[:option_end]:
        from kitepond.runtime import serve

        return serve(argv[1:])
    handlers = {
        "seal": _seal,
        "verify": _verify,
        "doctor": _doctor,
        "plan": _plan,
        "report": _report,
        "capture": _capture,
        "accept": _accept,
        "accept-glm": _accept_glm,
        "diagnose": _diagnose,
    }
    if argv and argv[0] not in handlers and argv[0] not in ("-h", "--help"):
        try:
            return os.execvp("vllm", ["vllm", *argv])
        except FileNotFoundError:
            print("Kitepond could not find vllm on PATH.", file=sys.stderr)
            return 127
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = handlers[args.command](args)
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as issue:
        parser.exit(1, f"kitepond {args.command}: {issue}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command in ("accept", "accept-glm"):
        return 0 if result["complete"] else 2
    return int(
        not (
            result.get("compatible_metadata", True)
            and result.get("artifact_matches", True)
        )
    )
