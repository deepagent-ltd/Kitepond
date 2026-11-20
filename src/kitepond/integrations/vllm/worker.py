# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit worker class for stock versions with manual lifecycle bridging."""

from vllm.v1.worker.gpu_worker import Worker

from .worker_hooks import WorkerLifecycleBridge


class KitepondWorker(WorkerLifecycleBridge, Worker):
    """Delegate device ownership and teardown to the stock worker."""
