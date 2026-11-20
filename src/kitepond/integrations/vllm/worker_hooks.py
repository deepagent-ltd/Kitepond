# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep registration state local to its worker instance."""

from .profile import require_probe_mode


def _activate(worker):
    require_probe_mode(worker.vllm_config)
    if hasattr(worker, "_kitepond_registration_state"):
        raise RuntimeError("Kitepond worker lifecycle already started")
    worker._kitepond_registration_state = "initialized"


def _deactivate(worker):
    if getattr(worker, "_kitepond_registration_state", None) == "initialized":
        worker._kitepond_registration_state = "closed"


class KitepondWorkerExtension:
    """Use the stock lifecycle and collective RPC extension hooks."""

    def init_worker_extension(self):
        _activate(self)

    def destroy_worker_extension(self):
        _deactivate(self)

    def kitepond_worker_registration(self):
        return getattr(self, "_kitepond_registration_state", "not_initialized")


class WorkerLifecycleBridge:
    """Provide the same lifecycle on stock builds without extension callbacks."""

    def init_device(self):
        require_probe_mode(self.vllm_config)
        super().init_device()
        _activate(self)

    def shutdown(self):
        _deactivate(self)
        super().shutdown()

    def kitepond_worker_registration(self):
        return getattr(self, "_kitepond_registration_state", "not_initialized")
