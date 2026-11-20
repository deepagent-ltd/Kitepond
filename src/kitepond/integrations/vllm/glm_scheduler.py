# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native scheduler profile for a bidirectional GLM Kitepond decode role."""

from .decode_completion import GrantRoutedScheduler
from .glm_cold_prefill_scheduler import GlmColdDecodeAdmissionMixin
from .glm_cold_scheduler import GlmColdRuntimeMixin
from .glm_terminal_scheduler import GlmTerminalCheckpointScheduler
from .glm_warm_decode_scheduler import GlmWarmDecodeMixin


class GlmKitepondScheduler(
    GlmWarmDecodeMixin,
    GlmColdDecodeAdmissionMixin,
    GlmColdRuntimeMixin,
    GrantRoutedScheduler,
    GlmTerminalCheckpointScheduler,
):
    """Own GLM receive, terminal capture and idle cold transitions."""
