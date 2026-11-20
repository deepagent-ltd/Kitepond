import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from zipfile import ZipFile

from kitepond.runtime import assemble_tree
from kitepond.ops.cli import main


class RuntimeDispatchTests(TestCase):
    def test_only_serve_with_an_explicit_enable_uses_the_overlay(self):
        cases = (
            ["serve", "/model", "-tp", "8", "--config", "launch.yaml"],
            ["bench", "serve", "--num-prompts", "64"],
            ["serve", "--", "--enable"],
        )
        for argv in cases:
            with self.subTest(argv=argv), patch(
                "kitepond.ops.cli.os.execvp", return_value=0
            ) as execute:
                self.assertEqual(main(argv), 0)
                execute.assert_called_once_with("vllm", ["vllm", *argv])

    def test_enabled_serve_passes_native_arguments_to_the_parser(self):
        argv = ["serve", "/model", "-dp", "8", "--enable", "--port=18120"]
        with patch("kitepond.runtime.serve", return_value=0) as serve:
            self.assertEqual(main(argv), 0)
            serve.assert_called_once_with(argv[1:])


class RuntimeOverlayTests(TestCase):
    def test_isolated_overlay_replaces_python_and_keeps_native_library(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            stock = root / "stock"
            stock.mkdir()
            original = stock / "__init__.py"
            original.write_bytes(b"stock")
            library = stock / "_native.so"
            library.write_bytes(b"native")
            replacement = b"overlay"
            manifest = {"python_files": {"vllm/__init__.py": {
                "base": hashlib.sha256(b"stock").hexdigest(),
                "overlay": hashlib.sha256(replacement).hexdigest(),
            }}}
            with ZipFile(root / "compat.zip", "w") as archive:
                archive.writestr("overlay/vllm/__init__.py", replacement)
                candidate = root / "vllm"
                assemble_tree(archive, manifest, stock, candidate)
                self.assertEqual(original.read_bytes(), b"stock")
                self.assertEqual((candidate / "__init__.py").read_bytes(), replacement)
                self.assertEqual((candidate / "_native.so").resolve(), library.resolve())
                with self.assertRaisesRegex(ValueError, "Base vLLM source differs"):
                    original.write_bytes(b"different")
                    assemble_tree(archive, manifest, stock, root / "rejected")
                self.assertFalse((root / "rejected").exists())
