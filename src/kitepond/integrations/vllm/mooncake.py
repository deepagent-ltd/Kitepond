# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compose the stock forward transfer with the external Kitepond probe."""

from vllm.config import KVTransferConfig

_ROLES = {"kv_producer", "kv_consumer"}


def mooncake_forward_config(
    role: str, *, engine_id: str, mooncake_options: dict | None = None
) -> KVTransferConfig:
    """Select the native Mooncake connector and a passive Kitepond child."""
    if not isinstance(role, str) or role not in _ROLES:
        raise ValueError("Mooncake forward requires a producer or a consumer")
    if not isinstance(engine_id, str) or not engine_id.strip():
        raise ValueError("Mooncake forward requires an explicit engine identity")
    return KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role=role,
        engine_id=engine_id,
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "KitepondConnector",
                    "kv_connector_module_path": "kitepond.integrations.vllm.connector",
                    "kv_role": role,
                },
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": role,
                    "kv_connector_extra_config": dict(mooncake_options or {}),
                },
            ]
        },
    )
