# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-owned access to the native HiSparse host durability fence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.hisparse.coordinator import get_hisparse_coordinator

if TYPE_CHECKING:
    from kitepond.integrations.vllm.terminal_checkpoint import TerminalPinLease

    from vllm.v1.core.kv_cache_manager import KVCacheManager


class NativeTerminalHostPrefix:
    """Use the retained native host table and its existing transfer events."""

    def __init__(self, manager: KVCacheManager) -> None:
        coordinator = get_hisparse_coordinator(manager)
        if (
            coordinator.host_group_id is None
            or not coordinator.resident_managers
            or not callable(getattr(coordinator, "plan_terminal_host_prefix", None))
            or not callable(getattr(coordinator, "has_durable_host_prefix", None))
        ):
            raise RuntimeError("Engine has no native HiSparse terminal host fence")
        self.coordinator = coordinator
        self.group_id = coordinator.host_group_id

    def _blocks(self, lease: TerminalPinLease) -> tuple[int, ...]:
        groups = tuple(group for group in lease.groups if group.location == "host")
        if len(groups) != 1 or groups[0].group_id != self.group_id:
            raise ValueError(
                "Terminal lease does not hold the native HiSparse host group"
            )
        return groups[0].block_ids

    def plan(self, lease: TerminalPinLease) -> bool:
        """Submit asynchronous native transfers for missing accepted pages."""
        return self.coordinator.plan_terminal_host_prefix(
            lease.identity.request_id,
            lease.requested_common_tokens,
            block_ids=self._blocks(lease),
        )

    def require_durable(self, lease: TerminalPinLease) -> None:
        """Authorize read access only after actual connector completion."""
        if not self.coordinator.has_durable_host_prefix(
            lease.identity.request_id,
            lease.requested_common_tokens,
            block_ids=self._blocks(lease),
        ):
            raise BlockingIOError("Native terminal host writes have not yet settled")
