# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire constants and signing for the private GLM NVMe protocol."""

from __future__ import annotations

import os
import stat
from hashlib import sha256
from hmac import digest as hmac_digest
from json import dumps
from pathlib import Path

NAMESPACE = "kitepond.glm-nvme.v1"
RESUME_PARAM = "kitepond_nvme_resume"
CONTROL_PATH = "/internal/kitepond/glm-nvme/v1"
TOKEN_ENV = "KITEPOND_GLM_NVME_HTTP_TOKEN"


def private_key(path: Path) -> bytes:
    if not path.is_absolute():
        raise ValueError("NVMe operator key must name an absolute private file")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    with os.fdopen(os.open(path, flags), "rb") as file:
        info = os.fstat(file.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) not in {0o400, 0o600}
            or info.st_size != 32
        ):
            raise ValueError("NVMe operator key must be a private 32-byte file")
        key = file.read(33)
    if len(key) != 32:
        raise ValueError("NVMe operator key changed while reading")
    return key


def seal_nvme_operation(operation: str, body: dict, key: bytes) -> dict:
    message = dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    signed = NAMESPACE.encode() + b"\0" + operation.encode() + b"\0" + message.encode()
    return {"body": body, "mac": hmac_digest(key, signed, sha256).hex()}
