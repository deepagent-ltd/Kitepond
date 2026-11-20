# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed client for a single engine's opt-in connector-control namespace."""

from typing import Any, Protocol


class ConnectorControlClient(Protocol):
    async def kv_connector_control_async(
        self, namespace: str, operation: str, payload: dict[str, Any]
    ) -> Any: ...


class ConnectorControlUnavailable(RuntimeError):
    """This installed engine client does not expose explicit connector control."""


class ConnectorControlTransport:
    """Forward calls through the native connector control entry point."""

    def __init__(self, client: ConnectorControlClient, *, namespace: str) -> None:
        if not callable(getattr(client, "kv_connector_control_async", None)):
            raise ConnectorControlUnavailable(
                "The installed EngineCore client lacks typed connector control"
            )
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("Connector control namespace is required")
        self._client = client
        self._namespace = namespace

    async def call(self, operation: str, payload: dict[str, Any]) -> Any:
        if (
            not isinstance(operation, str)
            or not operation.strip()
            or not isinstance(payload, dict)
            or any(not isinstance(key, str) for key in payload)
        ):
            raise ValueError("Invalid connector control operation or payload")
        return await self._client.kv_connector_control_async(
            self._namespace, operation, payload
        )
