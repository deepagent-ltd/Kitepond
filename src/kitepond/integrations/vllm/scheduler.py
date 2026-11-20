# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External scheduler entry point using vLLM's native cache manager."""

from vllm.v1.core.sched.scheduler import Scheduler

from .connector import KitepondConnector
from .profile import require_probe_mode


class KitepondScheduler(Scheduler):
    """Carry the registration probe through the native scheduler."""

    def __init__(self, vllm_config, *args, **kwargs):
        require_probe_mode(vllm_config)
        super().__init__(vllm_config, *args, **kwargs)
        connector = self.get_kv_connector()
        if not isinstance(connector, KitepondConnector):
            raise ValueError("KitepondScheduler requires KitepondConnector")
        connector.bind_kv_cache_manager(self.kv_cache_manager)
