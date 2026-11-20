# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cover typed connector routing and stock client capability rejection."""

import unittest
from unittest.mock import AsyncMock

from kitepond.integrations.vllm.control_transport import (
    ConnectorControlTransport,
    ConnectorControlUnavailable,
)


class _GenericClient:
    call_utility_async = AsyncMock()


class _TypedClient(_GenericClient):
    def __init__(self):
        self.kv_connector_control_async = AsyncMock(return_value={"closed": False})


class ControlTransportTest(unittest.IsolatedAsyncioTestCase):
    def test_generic_utility_is_never_used(self):
        with self.assertRaises(ConnectorControlUnavailable):
            ConnectorControlTransport(_GenericClient(), namespace="fixture.control.v1")
        _GenericClient.call_utility_async.assert_not_awaited()

    async def test_payload_uses_the_bound_typed_namespace(self):
        client = _TypedClient()
        transport = ConnectorControlTransport(client, namespace="fixture.control.v1")
        payload = {"transfer": {"id": "unit", "epoch": 4}}
        self.assertEqual(await transport.call("lookup", payload), {"closed": False})
        client.kv_connector_control_async.assert_awaited_once_with(
            "fixture.control.v1", "lookup", payload
        )
        with self.assertRaises(ValueError):
            await transport.call("", payload)
        with self.assertRaises(ValueError):
            await transport.call("lookup", {1: "ambiguous"})
        client.kv_connector_control_async.assert_awaited_once()
        client.call_utility_async.assert_not_awaited()

    async def test_failed_typed_call_propagates_without_retry(self):
        client = _TypedClient()
        client.kv_connector_control_async.side_effect = TimeoutError("result unknown")
        transport = ConnectorControlTransport(client, namespace="fixture.control.v1")
        with self.assertRaisesRegex(TimeoutError, "result unknown"):
            await transport.call("release", {"lease": "unit"})
        client.kv_connector_control_async.assert_awaited_once()
