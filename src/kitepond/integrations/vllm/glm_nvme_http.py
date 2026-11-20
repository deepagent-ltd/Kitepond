# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate loopback-only operator transport for the GLM NVMe lifecycle."""

from .cold_prefill_http import parse_control_envelope, private_control_ingress
from .glm_nvme_protocol import CONTROL_PATH, NAMESPACE, TOKEN_ENV

_OPERATIONS = {"archive_arm", "resume_arm", "migrate", "status", "cancel", "forget"}


def _parse(raw):
    operation, payload = parse_control_envelope(raw)
    if operation not in _OPERATIONS or payload.keys() != {"body", "mac"}:
        raise ValueError("Invalid D NVMe control operation")
    return operation, payload


async def nvme_ingress(request, call_next):
    return await private_control_ingress(
        request,
        call_next,
        path=CONTROL_PATH,
        namespace=NAMESPACE,
        token_env=TOKEN_ENV,
        parse_body=_parse,
    )
