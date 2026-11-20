# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical GLM text configuration identity for native runtime and CLI."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def config_fingerprint(text_config: Mapping[str, object]) -> str:
    """Bind model behavior while allowing engine-local Transformers metadata."""
    canonical = {
        key: value
        for key, value in text_config.items()
        if key not in {"_name_or_path", "transformers_version"}
        and not key.startswith("_attn")
    }
    return hashlib.sha256(
        json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()
