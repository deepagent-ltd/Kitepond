# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use the engine's canonical pure-text GLM request cache semantics."""

from kitepond.core.lookup import match_prefix_candidates
from kitepond.models.glm5next.frontier import digest

from vllm.v1.request import Request


def match_glm_native_request(request, candidates, *, max_tokens):
    if (
        type(request) is not Request
        or request.prompt_token_ids is None
        or request.prompt_embeds is not None
        or request.prompt_is_token_ids is not None
        or request.mm_features
        or request.lora_request is not None
        or (request.cache_salt is not None and type(request.cache_salt) is not str)
    ):
        raise ValueError("GLM cache lookup requires an exact native pure-text request")
    return match_prefix_candidates(
        request.all_token_ids,
        candidates,
        namespace_sha256=digest({"cache_salt": request.cache_salt}),
        prompt_tokens=request.num_prompt_tokens,
        max_tokens=max_tokens,
    )
