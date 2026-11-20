# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise offline wheel verification and the read-only deployment CLI."""

import binascii
import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kitepond.ops.cli import main
from kitepond.ops.wheel import inspect_wheel

DIST_ROOT = "kitepond-0.1.0.dev0.dist-info"


def write_fixture(path: Path, *, extra: str | None = None, corrupt: bool = False):
    entries = {
        f"{DIST_ROOT}/METADATA": b"Name: kitepond\nVersion: 0.1.0.dev0\n",
        f"{DIST_ROOT}/WHEEL": b"Tag: py3-none-any\n",
        "kitepond/ops/cli.py": b"pass\n",
    }
    if extra:
        entries[extra] = b"pass\n"
    with zipfile.ZipFile(path, "w") as archive:
        record = io.StringIO(newline="")
        writer = csv.writer(record)
        for name, raw in entries.items():
            hashed = binascii.b2a_base64(hashlib.sha256(raw).digest(), newline=False)
            digest = (
                hashed.decode("ascii").rstrip("=").replace("+", "-").replace("/", "_")
            )
            writer.writerow((name, f"sha256={digest}", len(raw)))
            archive.writestr(
                name, raw + b"tamper" if corrupt and name.endswith(".py") else raw
            )
        record_name = f"{DIST_ROOT}/RECORD"
        writer.writerow((record_name, "", ""))
        archive.writestr(record_name, record.getvalue())


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.wheel = self.directory / "kitepond-0.1.0.dev0-py3-none-any.whl"
        self.manifest = self.directory / "release.json"

    def _invoke(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(list(args))
        return status, json.loads(output.getvalue())

    def _seal(self):
        status, manifest = self._invoke(
            "seal",
            "--wheel",
            str(self.wheel),
            "--source-commit",
            "1" * 40,
            "--base-image",
            f"vllm/vllm-openai@sha256:{'2' * 64}",
            "--base-vllm-version",
            "0.32.1rc1.dev361+g83c078d95",
            "--base-python-version",
            "3.12",
        )
        self.assertEqual(status, 0)
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    def test_seal_and_verify_bind_artifact_bytes(self):
        write_fixture(self.wheel)
        self._seal()
        args = ("verify", "--manifest", str(self.manifest), "--wheel", str(self.wheel))
        status, result = self._invoke(*args)
        self.assertEqual(status, 0)
        self.assertTrue(result["wheel_verified"])
        write_fixture(self.wheel, extra="kitepond/ops/another.py")
        with (
            contextlib.redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit),
        ):
            self._invoke(*args)
        self.assertIn("does not match the sealed manifest", stderr.getvalue())

    def test_record_rejects_rewritten_payload(self):
        write_fixture(self.wheel, corrupt=True)
        with self.assertRaisesRegex(ValueError, "does not match member"):
            inspect_wheel(self.wheel)

    def test_disallows_other_python_namespaces_and_path_escape(self):
        for path in (
            "vllm/__init__.py",
            "kitepond/bootstrap.pth",
            "kitepond/../outside.py",
        ):
            with self.subTest(path=path):
                write_fixture(self.wheel, extra=path)
                with self.assertRaises(ValueError):
                    inspect_wheel(self.wheel)

    def test_plan_hashes_model_metadata_without_writing_directories(self):
        write_fixture(self.wheel)
        self._seal()
        model = self.directory / "model"
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
        (model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
        state = self.directory / "state"
        cold = self.directory / "cold"
        args = (
            "plan",
            "--manifest",
            str(self.manifest),
            "--model-dir",
            str(model),
            "--state-root",
            str(state),
            "--cold-root",
            str(cold),
            "--prefill-gpus",
            "4",
            "5",
            "6",
            "7",
            "--decode-gpus",
            "0",
            "1",
            "2",
            "3",
        )
        status, result = self._invoke(*args)
        self.assertEqual(status, 0)
        self.assertEqual(result["roles"]["decode"], [0, 1, 2, 3])
        self.assertFalse(state.exists())
        self.assertFalse(cold.exists())
        with (
            contextlib.redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit),
        ):
            self._invoke(*args[:-1], "4")
        self.assertIn("require eight distinct GPU indices", stderr.getvalue())

    def test_doctor_rejects_vllm_module_shadowing_pinned_distribution(self):
        write_fixture(self.wheel)
        self._seal()
        distributions = {
            "vllm": SimpleNamespace(
                version="0.32.1rc1.dev361+g83c078d95",
                locate_file=lambda name: self.directory / "base" / name,
            ),
            "kitepond": SimpleNamespace(
                version="0.1.0.dev0",
                locate_file=lambda name: Path(main.__code__.co_filename),
            ),
        }
        with (
            patch(
                "kitepond.ops.cli._installed_distribution",
                side_effect=distributions.__getitem__,
            ),
            patch(
                "kitepond.ops.cli.importlib.util.find_spec",
                return_value=SimpleNamespace(
                    origin=str(self.directory / "overlay/vllm/__init__.py")
                ),
            ),
        ):
            status, report = self._invoke("doctor", "--manifest", str(self.manifest))
        self.assertEqual(status, 1)
        self.assertTrue(report["matches"]["vllm"])
        self.assertFalse(report["matches"]["vllm_origin"])
