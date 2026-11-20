# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register a passive connector on the stock vLLM worker and scheduler paths."""

from collections import Counter

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)

from .lifecycle_control import LIFECYCLE_NAMESPACE, LifecycleControl
from .profile import require_probe_mode


class RegistrationMetadata(KVConnectorMetadata):
    """A stock step envelope without any transfer or readiness claims."""


class KitepondConnector(KVConnectorBase_V1, SupportsHMA):
    """Exercise stock connector callbacks without owning any KV blocks."""

    def __init__(self, vllm_config, role, kv_cache_config):
        require_probe_mode(vllm_config)
        super().__init__(vllm_config, role, kv_cache_config)
        self._lifecycle_control: LifecycleControl | None = None
        self._probe_manager = None
        self._probe_counts: Counter[str] = Counter()

    def _record(self, expected_role, operation):
        if self.role is not expected_role:
            raise RuntimeError(f"Invalid connector role for {operation}")
        self._probe_counts[operation] += 1

    def bind_kv_cache_manager(self, manager):
        if self._probe_manager is manager and manager is not None:
            return
        if manager is None or self._probe_manager is not None:
            raise RuntimeError("Invalid Kitepond probe cache manager binding")
        self._record(KVConnectorRole.SCHEDULER, "bind_manager")
        self._probe_manager = manager
        super().bind_kv_cache_manager(manager)

    def bind_lifecycle_control(self, control: LifecycleControl) -> None:
        """Bind the scheduler owner's explicit, read-only control capability."""
        if self.role is not KVConnectorRole.SCHEDULER:
            raise RuntimeError("Lifecycle control belongs to the scheduler connector")
        if not isinstance(control, LifecycleControl):
            raise TypeError("Explicit Kitepond lifecycle control is required")
        control.require_owner()
        if control.engine_id != self._kv_transfer_config.engine_id:
            raise ValueError(
                "Lifecycle control belongs to a different connector engine"
            )
        if (
            self._lifecycle_control is not None
            and self._lifecycle_control is not control
        ):
            raise RuntimeError("Lifecycle control is already bound")
        self._lifecycle_control = control

    def get_control_handler(self, namespace: str):
        if self._lifecycle_control is None or namespace != LIFECYCLE_NAMESPACE:
            return None
        return self._lifecycle_control.handle

    def registration_report(self) -> dict:
        return {
            "role": self.role.name,
            "manager_bound": self._probe_manager is not None,
            "callbacks": dict(self._probe_counts),
        }

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self._record(KVConnectorRole.SCHEDULER, "lookup")
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens:
            raise RuntimeError("Kitepond probe cannot import external tokens")
        self._record(KVConnectorRole.SCHEDULER, "allocate")

    def build_connector_meta(self, scheduler_output):
        return RegistrationMetadata()

    def request_finished_all_groups(self, request, block_ids):
        self._record(KVConnectorRole.SCHEDULER, "finish")
        return False, None

    def start_load_kv(self, forward_context, **kwargs):
        self._record(KVConnectorRole.WORKER, "start_load")

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        self._record(KVConnectorRole.WORKER, "wait_for_save")
