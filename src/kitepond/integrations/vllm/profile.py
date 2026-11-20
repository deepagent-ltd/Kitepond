# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select the scope of an explicitly enabled Kitepond adapter."""

from collections.abc import Mapping

PROBE_MODE = "registration-probe"


class UnsupportedKitepondProfile(ValueError):
    """The requested profile has no qualified adapter."""


def require_probe_mode(config) -> None:
    options = config.additional_config
    settings = options.get("kitepond") if isinstance(options, Mapping) else None
    if not isinstance(settings, Mapping) or settings != {"mode": PROBE_MODE}:
        raise UnsupportedKitepondProfile(
            "Kitepond requires additional_config.kitepond.mode=registration-probe "
            "with no other Kitepond options. Full cache mode is unavailable."
        )
