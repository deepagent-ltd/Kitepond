# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate one Kitepond wheel inside a disposable stock vLLM image."""

import argparse
import contextlib
import hashlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _snapshot_vllm() -> dict:
    dist = metadata.distribution("vllm")
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None or not dist.files:
        raise RuntimeError("Stock vLLM import path or installation RECORD is missing")
    members = sorted(str(file) for file in dist.files if file.parts[0] == "vllm")
    if "vllm/__init__.py" not in members:
        raise RuntimeError("Stock vLLM distribution does not own its import entry")
    if Path(spec.origin).resolve() != dist.locate_file("vllm/__init__.py").resolve():
        raise RuntimeError("Stock vLLM import path has been shadowed")
    fingerprint = hashlib.sha256()
    for member in members:
        with dist.locate_file(member).open("rb") as stream:
            content = hashlib.file_digest(stream, "sha256").digest()
        fingerprint.update(member.encode("utf-8"))
        fingerprint.update(b"\x00")
        fingerprint.update(content)
    return {
        "version": dist.version,
        "origin": spec.origin,
        "vllm_file_count": len(members),
        "vllm_files_sha256": fingerprint.hexdigest(),
    }


_GUARD = "HiSparse does not support decode context parallelism."
_STOCK_GUARD_SCHEMA = "kitepond.stock-hisparse-dcp-gate.v1"


def _guard_blocked(phase, error, **observations):
    return {
        "state": "blocked",
        "phase": phase,
        "error_type": type(error).__name__,
        "detail": str(error).splitlines()[0][:240] if str(error) else "",
        **observations,
    }


def _check_stock_hisparse_dcp(target: Path):
    """Exercise only VllmConfig in this separate process."""
    profile = {
        "tp": 4,
        "dcp_control": 1,
        "dcp_candidate": 4,
        "interleave": 4,
        "cuda_platform_predicate_simulated": True,
        "serving_runtime_evaluated": False,
    }
    try:
        sys.path.insert(0, str(target))
        import kitepond

        if not Path(kitepond.__file__).resolve().is_relative_to(target.resolve()):
            raise RuntimeError("Kitepond child import differs from the wheel target")
        from unittest.mock import patch

        from vllm.config import (
            AttentionConfig,
            HiSparseConfig,
            ParallelConfig,
            VllmConfig,
        )
        from vllm.platforms import current_platform
    except Exception as error:
        return profile | _guard_blocked("native_import", error)

    def configure(dcp):
        parallel = ParallelConfig(
            tensor_parallel_size=4,
            decode_context_parallel_size=dcp,
            cp_kv_cache_interleave_size=4,
        )
        # This traverses the genuine configuration validator. Only its device
        # predicate is simulated. No engine, device or model is initialized.
        with patch.object(current_platform, "is_cuda", return_value=True):
            VllmConfig(
                parallel_config=parallel,
                attention_config=AttentionConfig(hisparse_config=HiSparseConfig()),
            )

    try:
        configure(1)
    except Exception as error:
        return profile | _guard_blocked("dcp1_control", error)
    profile["dcp1_configuration"] = "accepted"
    try:
        configure(4)
    except Exception as error:
        if isinstance(error, ValueError) and _GUARD in str(error):
            return profile | {"state": "rejected_by_stock_validator", "detail": _GUARD}
        return profile | _guard_blocked("dcp4_configuration", error)
    return profile | {"state": "configuration_accepted", "detail": ""}


def _run_guard_process(target: Path) -> dict:
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                "--native-guard-worker",
                str(target),
            ],
            check=True,
            timeout=90,
            stdout=subprocess.PIPE,
            text=True,
        )
        scope = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        scope = _guard_blocked("probe_process", error)
    return {"schema": _STOCK_GUARD_SCHEMA, **scope}


def main() -> int:
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or Path("/dev/nvidia0").exists()
        or Path("/dev/nvidiactl").exists()
    ):
        raise RuntimeError("This smoke test requires CPU-only isolation")
    if len(sys.argv) == 3 and sys.argv[1] == "--native-guard-worker":
        with contextlib.redirect_stdout(sys.stderr):
            guard = _check_stock_hisparse_dcp(Path(sys.argv[2]))
        print(json.dumps(guard))
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check-hisparse-dcp", action="store_true")
    args = parser.parse_args()
    wheel, manifest = args.wheel, args.manifest
    guard = None
    if args.check_hisparse_dcp and os.environ.get("VLLM_TARGET_DEVICE") != "cpu":
        parser.error("Set VLLM_TARGET_DEVICE=cpu for the isolated native guard probe")
    if importlib.util.find_spec("kitepond") is not None:
        raise RuntimeError("Clean stock environment already contains Kitepond")
    before = _snapshot_vllm()
    with tempfile.TemporaryDirectory(prefix="kitepond-stock-") as temporary:
        target = Path(temporary) / "site"
        pip = [sys.executable, "-I", "-m", "pip"]
        subprocess.run(
            [
                *pip,
                "--disable-pip-version-check",
                "install",
                "--no-index",
                "--no-deps",
                "--no-cache-dir",
                "--target",
                str(target),
                str(wheel),
            ],
            check=True,
            stdout=sys.stderr,
        )
        sys.path.insert(0, str(target))
        spec = importlib.util.find_spec("kitepond.ops")
        if (
            spec is None
            or spec.origin is None
            or not Path(spec.origin).is_relative_to(target)
        ):
            raise RuntimeError("Kitepond was not imported from the clean wheel install")
        from kitepond.ops.cli import main as cli

        checks = {}
        for command in (
            ["verify", "--wheel", str(wheel), "--manifest", str(manifest)],
            ["doctor", "--manifest", str(manifest), "--engine-files", "--engine-hooks"],
        ):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = cli(command)
            checks[command[0]] = json.loads(output.getvalue())
            if status:
                raise RuntimeError(f"Kitepond {command[0]} failed: {output.getvalue()}")
        if checks["doctor"]["engine_files"] != before:
            raise RuntimeError("vLLM digest differs from the independent stock digest")
        after = _snapshot_vllm()
        if before != after:
            raise RuntimeError("Kitepond installation changed stock vLLM bytes or path")
        if "vllm" in sys.modules or "torch" in sys.modules:
            raise RuntimeError("CPU-only metadata check imported the serving engine")
        if args.check_hisparse_dcp:
            guard = _run_guard_process(target)
        sys.path.pop(0)
        shutil.rmtree(target)
        if _snapshot_vllm() != before:
            raise RuntimeError("Stock engine snapshot differed after removing target")
    observation = {
        "scope": "stock image wheel installation and file integrity",
        "stock": before,
        "checks": checks,
        "vllm_preserved": True,
        "engine_imported": False,
        "gpu_qualified": False,
    }
    if guard is not None:
        observation["hisparse_dcp_guard"] = guard
    print(json.dumps(observation, indent=2))
    if guard is not None and guard["state"] == "blocked":
        return 2
    if guard is not None and guard["state"] == "configuration_accepted":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
