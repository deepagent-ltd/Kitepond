# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check bounded exact cache prefix and next-token identities."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

MAX_CANDIDATES = 256
HASH_CHUNK_TOKENS = 4096


def _hex(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


@dataclass(frozen=True)
class NativeLookupCandidate:
    entry_id: str
    boundary_tokens: int
    prefix_token_sha256: str
    lookahead_token_id: int
    cache_namespace_sha256: str
    ram_observed: bool

    def __post_init__(self):
        if (
            not _hex(self.entry_id)
            or type(self.boundary_tokens) is not int
            or self.boundary_tokens <= 0
            or not _hex(self.prefix_token_sha256)
            or type(self.lookahead_token_id) is not int
            or self.lookahead_token_id < 0
            or not _hex(self.cache_namespace_sha256)
            or type(self.ram_observed) is not bool
        ):
            raise ValueError("Invalid bounded native lookup metadata")


@dataclass(frozen=True)
class PrefixHashResult:
    digests: dict[int, str]
    processed_tokens: int
    chunks: int
    largest_chunk: int


def prefix_token_sha256_many(
    token_ids: Sequence[int],
    boundaries: Sequence[int],
    *,
    max_tokens: int,
    chunk_tokens: int = HASH_CHUNK_TOKENS,
) -> PrefixHashResult:
    """Exact native JSON-list hashes in one bounded streaming pass.

    Each prefix digest closes a copied SHA state with ']'. Encoding holds at
    most one bounded token chunk, never a whole prompt copy per candidate.
    """
    if (
        len(boundaries) > MAX_CANDIDATES
        or type(max_tokens) is not int
        or max_tokens <= 0
        or type(chunk_tokens) is not int
        or not 0 < chunk_tokens <= HASH_CHUNK_TOKENS
        or any(
            type(boundary) is not int
            or not 0 <= boundary <= min(len(token_ids), max_tokens)
            for boundary in boundaries
        )
    ):
        raise ValueError("Invalid bounded native prefix-hash request")
    digest = hashlib.sha256(b"[")
    position = chunks = largest = 0
    result = {}
    for boundary in sorted(set(boundaries)):
        while position < boundary:
            stop = min(boundary, position + chunk_tokens)
            part = list(token_ids[position:stop])
            if len(part) != stop - position or any(
                type(token) is not int or token < 0 for token in part
            ):
                raise ValueError("Prompt fingerprint requires exact native token IDs")
            encoded = json.dumps(part, separators=(",", ":")).encode()
            if position:
                digest.update(b",")
            digest.update(memoryview(encoded)[1:-1])
            chunks += 1
            largest = max(largest, len(part))
            position = stop
        snapshot = digest.copy()
        snapshot.update(b"]")
        result[boundary] = snapshot.hexdigest()
    return PrefixHashResult(result, position, chunks, largest)


@dataclass(frozen=True)
class NativeLookupMatch:
    candidates: tuple[NativeLookupCandidate, ...]
    hashes: PrefixHashResult


def match_prefix_candidates(
    tokens: Sequence[int],
    candidates: Sequence[NativeLookupCandidate],
    *,
    namespace_sha256: str,
    prompt_tokens: int,
    max_tokens: int,
) -> NativeLookupMatch:
    """Match cache identity only. A matching record cannot grant execution."""
    if (
        not _hex(namespace_sha256)
        or type(prompt_tokens) is not int
        or not 0 < prompt_tokens <= len(tokens)
        or type(max_tokens) is not int
        or max_tokens <= 0
        or len(candidates) > MAX_CANDIDATES
        or any(type(c) is not NativeLookupCandidate for c in candidates)
    ):
        raise ValueError("Lookup needs a bounded token sequence and cache namespace")
    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.cache_namespace_sha256 == namespace_sha256
        and candidate.boundary_tokens < prompt_tokens
        and candidate.boundary_tokens <= max_tokens
        and type(tokens[candidate.boundary_tokens]) is int
        and tokens[candidate.boundary_tokens] == candidate.lookahead_token_id
    )
    hashes = prefix_token_sha256_many(
        tokens, [c.boundary_tokens for c in eligible], max_tokens=max_tokens
    )
    matched = tuple(
        sorted(
            (
                candidate
                for candidate in eligible
                if hashes.digests[candidate.boundary_tokens]
                == candidate.prefix_token_sha256
            ),
            key=lambda c: (-c.boundary_tokens, not c.ram_observed, c.entry_id),
        )
    )
    return NativeLookupMatch(matched, hashes)
