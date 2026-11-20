# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in registration of a GLM backend in the current vLLM process."""

GLM_HISPARSE_BACKEND = "KITEPOND_GLM_HISPARSE_SM120"
GLM_HISPARSE_BACKEND_PATH = (
    "kitepond.integrations.vllm.glm_sparse_backend.GlmHiSparseDCPBackend"
)


def register_glm_hisparse_backend():
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    known = AttentionBackendEnum.__members__.get(GLM_HISPARSE_BACKEND)
    if known is None:
        register_backend(GLM_HISPARSE_BACKEND, GLM_HISPARSE_BACKEND_PATH)
        known = AttentionBackendEnum[GLM_HISPARSE_BACKEND]
    elif known.get_path() != GLM_HISPARSE_BACKEND_PATH:
        raise RuntimeError("The GLM HiSparse attention alias is already in use")
    return known
