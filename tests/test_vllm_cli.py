# SPDX-License-Identifier: Apache-2.0
from argparse import Namespace
from dataclasses import dataclass
from unittest import TestCase

from kitepond.vllm_cli import _configure_dcp8_speculation


@dataclass
class AttentionOptions:
    enable_glm_flash_dcp8_mtp3: bool = False


def arguments(spec=None, **overrides):
    args = vars(
        Namespace(
            speculative_config=spec,
            spec_method=None,
            spec_model=None,
            spec_tokens=None,
            attention_config=AttentionOptions(),
        )
    )
    return Namespace(**(args | overrides))


class DCP8SpeculationTests(TestCase):
    def test_no_request_keeps_speculation_disabled(self):
        args = arguments()
        _configure_dcp8_speculation(args)
        self.assertIsNone(args.speculative_config)
        self.assertFalse(args.attention_config.enable_glm_flash_dcp8_mtp3)

    def test_opt_in_resolves_native_draft_geometry(self):
        base = {"method": "mtp", "num_speculative_tokens": 3}
        expected = base | {
            "draft_tensor_parallel_size": 8,
            "draft_decode_context_parallel_size": 8,
        }
        for request in (base, expected):
            with self.subTest(request=request):
                args = arguments(request)
                _configure_dcp8_speculation(args)
                self.assertTrue(args.attention_config.enable_glm_flash_dcp8_mtp3)
                self.assertEqual(args.speculative_config, expected)

    def test_rejects_draft_layout_or_method_overrides(self):
        base = {"method": "mtp", "num_speculative_tokens": 3}
        invalid = (
            [],
            {},
            base | {"method": "dspark"},
            base | {"num_speculative_tokens": 1},
            base | {"num_speculative_tokens": 3.0},
            base | {"draft_tensor_parallel_size": 1},
            base | {"draft_decode_context_parallel_size": 1},
            base | {"model": "other-weights"},
            base | {"enable_adaptive_verification": True},
        )
        for request in invalid:
            with self.subTest(request=request), self.assertRaisesRegex(
                ValueError, "native MTP3"
            ):
                _configure_dcp8_speculation(arguments(request))

    def test_split_flags_cannot_bypass_the_dcp8_gate(self):
        for key, value in (
            ("spec_method", "mtp"),
            ("spec_model", "other-weights"),
            ("spec_tokens", 3),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "requires --speculative-config"
            ):
                _configure_dcp8_speculation(arguments(**{key: value}))
