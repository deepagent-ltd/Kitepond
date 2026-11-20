# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit worker classes for a private one-shot Mooncake fault experiment."""

import os
from pathlib import Path

from kitepond.integrations.vllm.glm_k0_worker import GlmK0PrefillWorker
from kitepond.integrations.vllm.glm_terminal_worker import GlmTerminalWorker
from lost_write_ack_injector import (
    install_missing_write_ack,
    install_target_observation,
    require_private,
)

from vllm.distributed.kv_transfer import get_kv_transfer_group

LAB_DIR_ENV = "KITEPOND_LAB_LOST_WRITE_ACK_DIRECTORY"


def _directory():
    directory = Path(os.environ[LAB_DIR_ENV])
    require_private(directory)
    return directory


class LostWriteAckPrefillWorker(GlmK0PrefillWorker):
    def initialize_from_config(self, kv_cache_config):
        super().initialize_from_config(kv_cache_config)
        _, alias = get_kv_transfer_group().sub_connectors
        install_missing_write_ack(
            alias.connector_worker,
            alias.window_scope(),
            alias._sources.lookup,
            _directory(),
        )


class LostWriteAckDecodeWorker(GlmTerminalWorker):
    def initialize_from_config(self, kv_cache_config):
        super().initialize_from_config(kv_cache_config)
        install_target_observation(self._kitepond_decode_import, _directory())
