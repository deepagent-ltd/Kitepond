# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proof for omitting only already-shared GLM MLA rows from a rank transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from hmac import compare_digest
from json import dumps
from time import monotonic_ns
from typing import TYPE_CHECKING

from kitepond.core.readiness import ReadyAttempt, ReadyReceipt, _digest

from .cold_restore import ColdRestoreLayout, native_rank_rows
from .layout import CacheRole
from .shared_region import NativeHiSparseRegionDescriptor

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kitepond.integrations.vllm.glm_cold_prefill_admission import (
        ColdPrefillControl,
        ColdPrefillOffer,
    )
    from kitepond.integrations.vllm.host_authority import NativeHostAuthority


SHARED_MLA_CONFIG = "kitepond_glm_shared_mla_preflight"
_RANKS = (0, 1, 2, 3)
_ROW_BYTES = 656
_ROWS_PER_PAGE = 256
_TOKENS_PER_PAGE = len(_RANKS) * _ROWS_PER_PAGE


@dataclass(frozen=True)
class SharedMlaLayerPages:
    """One native layer's byte-addressing contract for the admitted prefix."""

    layer: str
    storage_offset: int
    strides: tuple[int, ...]
    block_ids: tuple[int, ...]
    rows: int


@dataclass(frozen=True)
class SharedMlaRankPages:
    """Observation from a rank owner after its native view/backing is enrolled."""

    attempt: ReadyAttempt
    descriptor: NativeHiSparseRegionDescriptor
    valid_tokens: int
    model_fingerprint: str
    config_fingerprint: str
    layers: tuple[SharedMlaLayerPages, ...]

    @property
    def sha256(self) -> str:
        payload = dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return sha256(b"kitepond-glm-mla-pages-v1\x00" + payload).hexdigest()

    @classmethod
    def observe_enrolled(
        cls,
        cache: ColdRestoreLayout,
        descriptor: NativeHiSparseRegionDescriptor,
        attempt: ReadyAttempt,
    ) -> SharedMlaRankPages:
        """Call only after verifying the cache is backed by this native region."""
        descriptor.validate()
        if (
            cache.rank != descriptor.source_rank
            or cache.layout.request_id != attempt.request_id
            or cache.layout.epoch != attempt.epoch
        ):
            raise ValueError("Shared MLA view belongs to another native attempt")
        tokens = cache.layout.target_valid_tokens
        rows = native_rank_rows(tokens, CacheRole.MLA, cache.rank)
        pages = (rows + _ROWS_PER_PAGE - 1) // _ROWS_PER_PAGE
        layers = []
        for name, layer in sorted(cache.layers.items()):
            if layer.identity.cache_role is not CacheRole.MLA:
                continue
            view = layer.tensor
            block_ids = layer.block_ids[:pages]
            if (
                view.device.type != "cpu"
                or view.shape[1:] != (1, _ROWS_PER_PAGE, _ROW_BYTES)
                or view.stride(-1) != 1
                or view.stride(-2) != _ROW_BYTES
                or view.stride(0) < _ROWS_PER_PAGE * _ROW_BYTES
                or len(block_ids) != pages
                or len(set(block_ids)) != pages
                or any(not 0 < block < view.shape[0] for block in block_ids)
                or view.storage_offset()
                + max(block_ids, default=0) * view.stride(0)
                + _ROWS_PER_PAGE * _ROW_BYTES
                > descriptor.nbytes
            ):
                raise ValueError("Shared MLA view changed its native B656 pages")
            layers.append(
                SharedMlaLayerPages(
                    name, view.storage_offset(), tuple(view.stride()), block_ids, rows
                )
            )
        if not layers:
            raise ValueError("Shared MLA view has no native source layers")
        return cls(
            attempt,
            descriptor,
            tokens,
            cache.layout.model_fingerprint,
            cache.layout.config_fingerprint,
            tuple(layers),
        )


@dataclass(frozen=True)
class VerifiedSharedMlaElision:
    """Four exact physical aliases authenticated before opening a reduced plan."""

    attempt: ReadyAttempt
    rank_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.attempt) is not ReadyAttempt or not valid_rank_digests(
            self.rank_digests
        ):
            raise ValueError(
                "Shared MLA decision needs one exact attempt and all ranks"
            )

    def validate_local(
        self,
        cache: ColdRestoreLayout,
        region: NativeHiSparseRegionDescriptor,
        attempt: ReadyAttempt,
    ) -> None:
        expected = self.rank_digests[cache.rank]
        if (
            self.attempt != attempt
            or SharedMlaRankPages.observe_enrolled(cache, region, attempt).sha256
            != expected
        ):
            raise ValueError("Shared MLA rank changed after proof admission")


def valid_rank_digests(value: object) -> bool:
    return (
        type(value) is tuple
        and len(value) == len(_RANKS)
        and all(_digest(item) for item in value)
    )


def confirm_shared_mla_elision(
    control: ColdPrefillControl,
    authority: NativeHostAuthority,
    offer: ColdPrefillOffer,
    receipts: tuple[ReadyReceipt, ...],
    ready_mac: str,
    producer: Mapping[int, SharedMlaRankPages],
    consumer: Mapping[int, SharedMlaRankPages],
) -> VerifiedSharedMlaElision | None:
    """Use full transfer unless both trusted owners prove the same four mappings.

    The caller supplies freshly observed producer and consumer maps from the
    owning native workers. This local candidate deliberately has no wire parser.
    The deployment must carry observations over authenticated scoped control
    before enabling a reduced plan. An invalid grant or READY fails closed.
    """
    authority._require_owner()
    if (
        not authority.ready_for_adoption(offer.grant)
        or monotonic_ns() >= offer.arm_deadline_ns
        or not _digest(ready_mac)
        or not compare_digest(control.ready_mac(offer, receipts), ready_mac)
    ):
        raise ValueError("Shared MLA elision requires the active signed READY lease")
    if set(producer) != set(_RANKS) or set(consumer) != set(_RANKS):
        return None
    if authority.block_tokens != _TOKENS_PER_PAGE:
        return None
    grant, attempt = offer.grant, offer.attempt
    first = producer[0]
    for receipt in receipts:
        rank = receipt.rank
        rows = native_rank_rows(offer.imported_tokens, CacheRole.MLA, rank)
        pages = (rows + _ROWS_PER_PAGE - 1) // _ROWS_PER_PAGE
        expected_pages = grant.writable_pages[:pages]
        if len(expected_pages) != pages:
            return None
        observed = producer[rank]
        if (
            observed != consumer[rank]
            or observed.attempt != attempt
            or observed.valid_tokens != offer.imported_tokens
            or (observed.model_fingerprint, observed.config_fingerprint)
            != (first.model_fingerprint, first.config_fingerprint)
            or observed.descriptor.source_rank != rank
            or (
                observed.descriptor.authority_engine_id,
                observed.descriptor.authority_boot_id,
            )
            != (grant.engine_id, grant.boot_id)
            or observed.descriptor.fingerprint() != receipt.region_sha256
            or not observed.layers
            or any(layer.block_ids != expected_pages for layer in observed.layers)
        ):
            return None
    return VerifiedSharedMlaElision(
        attempt, tuple(producer[rank].sha256 for rank in _RANKS)
    )
