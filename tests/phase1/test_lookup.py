# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json

from kitepond.core.lookup import (
    NativeLookupCandidate,
    match_prefix_candidates,
    prefix_token_sha256_many,
)


def _hash(tokens):
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


def test_many_exact_prefix_hashes_share_a_single_bounded_pass():
    tokens = list(range(14_000))
    boundaries = [13, 4097, 5000, 9810, 13]
    result = prefix_token_sha256_many(tokens, boundaries, max_tokens=14_000)
    assert result.processed_tokens == max(boundaries)
    assert result.largest_chunk == 4096
    assert result.digests == {i: _hash(tokens[:i]) for i in set(boundaries)}


def test_ns_boundary_and_next_token_required_even_when_cache_looks_resident():
    tokens = list(range(20))
    ns = "a" * 64
    candidates = [
        NativeLookupCandidate("0" * 64, 11, _hash(tokens[:11]), 11, ns, False),
        NativeLookupCandidate("1" * 64, 11, _hash(tokens[:11]), 11, ns, True),
        NativeLookupCandidate("2" * 64, 14, _hash(tokens[:14]), 14, "b" * 64, True),
        NativeLookupCandidate("3" * 64, 15, _hash(tokens[:15]), 16, ns, True),
        NativeLookupCandidate("4" * 64, 18, _hash(tokens[:18]), 18, ns, False),
        NativeLookupCandidate("5" * 64, 12, _hash(tokens[:11]), 12, ns, True),
    ]
    result = match_prefix_candidates(
        tokens, candidates, namespace_sha256=ns, prompt_tokens=18, max_tokens=18
    )
    assert result.candidates == (candidates[1], candidates[0])
    assert result.hashes.processed_tokens == 12
