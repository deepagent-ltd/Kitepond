# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Authenticate the lease for P's first write into a live D-owned host pool."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from hmac import compare_digest
from hmac import digest as hmac_digest
from json import dumps
from time import monotonic_ns

from kitepond.core.contracts import Wait
from kitepond.core.readiness import ReadyAttempt, ReadyReceipt, _digest
from kitepond.models.glm5next.frontier import digest as digest_tokens

from .host_authority import HostRankSettlement, NativeHostAuthority, NativeHostGrant
from .ready_transport import ReadyCollector, ReadyTicket

_COMPONENT = "kitepond.glm-cold-prefill.v1"
_DIRECTIONS = {"offer": b"offer\x00", "ready": b"ready\x00"}


def _identifier(value: str) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and value.isascii()
        and (value.replace("-", "").replace("_", "").replace(".", "").isalnum())
    )


@dataclass(frozen=True)
class ColdPrefillPrompt:
    request_id: str
    transfer_id: str
    epoch: int
    tokens: int
    prompt_sha256: str

    def __post_init__(self) -> None:
        if (
            not _identifier(self.request_id)
            or not _identifier(self.transfer_id)
            or type(self.epoch) is not int
            or not 0 <= self.epoch < 2**63
            or type(self.tokens) is not int
            or not 2 <= self.tokens < 2**31
            or not _digest(self.prompt_sha256)
        ):
            raise ValueError("Invalid first-prefill request identity or prompt")

    @classmethod
    def from_tokens(
        cls, request_id: str, transfer_id: str, epoch: int, tokens: Sequence[int]
    ) -> ColdPrefillPrompt:
        if any(type(token) is not int or not 0 <= token < 2**31 for token in tokens):
            raise ValueError("Cold prefill requires the actual raw prompt token IDs")
        return cls(
            request_id, transfer_id, epoch, len(tokens), digest_tokens(list(tokens))
        )


@dataclass(frozen=True)
class ColdPrefillOffer:
    prompt: ColdPrefillPrompt
    grant: NativeHostGrant
    prefill_boot_id: str
    arm_deadline_ns: int

    def __post_init__(self) -> None:
        if (
            type(self.prompt) is not ColdPrefillPrompt
            or type(self.grant) is not NativeHostGrant
            or not _identifier(self.prefill_boot_id)
            or type(self.arm_deadline_ns) is not int
            or not 0 < self.arm_deadline_ns < 2**63
            or self.grant.request_id != self.prompt.request_id
            or self.grant.transfer_id != self.prompt.transfer_id
            or self.grant.epoch != self.prompt.epoch
            or self.grant.reservation_tokens != self.prompt.tokens
            or self.grant.prefix_tokens
            or self.grant.readonly_pages
            or self.grant.copy_tail is not None
            or not _digest(self.grant.nonce, 32)
            or not self.grant.writable_pages
            or any(
                type(page) is not int or page < 1 for page in self.grant.writable_pages
            )
            or len(set(self.grant.writable_pages)) != len(self.grant.writable_pages)
        ):
            raise ValueError("Cold offer does not describe the current native lease")

    @property
    def attempt(self) -> ReadyAttempt:
        grant = self.grant
        return ReadyAttempt(
            grant.request_id, grant.transfer_id, grant.epoch, grant.sha256
        )

    @property
    def imported_tokens(self) -> int:
        # Native Mooncake reserves the original last prompt token for D.
        return self.prompt.tokens - 1


class ColdPrefillControl:
    """A private boot-scoped signer for the D offer and the drained P READY quorum."""

    def __init__(
        self,
        *,
        d_engine_id: str,
        d_boot_id: str,
        p_boot_id: str,
        key: bytes,
    ) -> None:
        if (
            any(not _identifier(value) for value in (d_engine_id, d_boot_id, p_boot_id))
            or type(key) is not bytes
            or len(key) != 32
        ):
            raise ValueError(
                "Cold prefill needs a private boot-scoped control identity"
            )
        self.d_identity = (d_engine_id, d_boot_id)
        self.p_boot_id = p_boot_id
        self._key = key

    def _mac(self, kind: str, value: dict) -> str:
        message = dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hmac_digest(
            self._key,
            _COMPONENT.encode() + b"\x00" + _DIRECTIONS[kind] + message.encode(),
            sha256,
        ).hex()

    def _require_scope(self, offer: ColdPrefillOffer) -> None:
        if (
            offer.grant.engine_id,
            offer.grant.boot_id,
        ) != self.d_identity or offer.prefill_boot_id != self.p_boot_id:
            raise ValueError("Cold prefill lease belongs to a different boot")

    def export_offer(self, offer: ColdPrefillOffer) -> dict:
        self._require_scope(offer)
        body = asdict(offer)
        return {"body": body, "mac": self._mac("offer", body)}

    def open_offer(
        self,
        envelope: dict,
        prompt: ColdPrefillPrompt,
        *,
        clock: Callable[[], int] = monotonic_ns,
    ) -> ColdPrefillOffer:
        if (
            not isinstance(envelope, dict)
            or envelope.keys() != {"body", "mac"}
            or not isinstance(envelope["body"], dict)
            or not _digest(envelope["mac"])
            or not compare_digest(self._mac("offer", envelope["body"]), envelope["mac"])
        ):
            raise ValueError("Cold prefill offer is not authenticated")

        body = envelope["body"]
        if body.keys() != {item.name for item in fields(ColdPrefillOffer)}:
            raise ValueError("Cold offer fields differ from the active version")
        raw = dict(body["grant"])
        raw["readonly_pages"] = tuple(raw["readonly_pages"])
        raw["writable_pages"] = tuple(raw["writable_pages"])
        offer = ColdPrefillOffer(
            ColdPrefillPrompt(**body["prompt"]),
            NativeHostGrant(**raw),
            body["prefill_boot_id"],
            body["arm_deadline_ns"],
        )
        self._require_scope(offer)
        if offer.prompt != prompt or clock() >= offer.arm_deadline_ns:
            raise ValueError(
                "Cold prefill offer is stale or belongs to a different prompt"
            )
        return offer

    def arm(
        self,
        offer: ColdPrefillOffer,
        collector: ReadyCollector,
        regions: Mapping[int, str],
    ) -> ReadyTicket:
        self._require_scope(offer)
        if regions.keys() != {0, 1, 2, 3} or monotonic_ns() >= offer.arm_deadline_ns:
            raise ValueError(
                "Cold prefill needs four live enrolled ranks before its deadline"
            )
        return collector.arm(offer.attempt, regions, offer.imported_tokens)

    def ready_mac(
        self, offer: ColdPrefillOffer, receipts: tuple[ReadyReceipt, ...]
    ) -> str:
        self._require_scope(offer)
        if (
            len(receipts) != 4
            or {item.rank for item in receipts} != {0, 1, 2, 3}
            or any(
                item.attempt != offer.attempt
                or item.target_valid_tokens != offer.imported_tokens
                or item.draft_valid_tokens != offer.imported_tokens
                for item in receipts
            )
        ):
            raise ValueError("Cold prefill needs the actual four-rank N-1 READY quorum")
        return self._mac(
            "ready",
            {
                "offer": asdict(offer),
                "ranks": [
                    asdict(item)
                    for item in sorted(receipts, key=lambda item: item.rank)
                ],
            },
        )


class ColdPrefillAdmission:
    """Create only D-native grants and retain dispatched pages until real fences."""

    def __init__(
        self,
        authority: NativeHostAuthority,
        control: ColdPrefillControl,
        *,
        context_limit: int,
        max_pending: int,
        arm_timeout_seconds: float,
        clock: Callable[[], int] = monotonic_ns,
    ) -> None:
        if (
            (authority.engine_id, authority.boot_id) != control.d_identity
            or type(context_limit) is not int
            or not 2 <= context_limit < 2**31
            or type(max_pending) is not int
            or not 1 <= max_pending <= 128
            or type(arm_timeout_seconds) not in (int, float)
            or not math.isfinite(arm_timeout_seconds)
            or not 0 < arm_timeout_seconds <= 60
            or not callable(clock)
        ):
            raise ValueError(
                "Cold prefill needs the active D owner and bounded ingress"
            )
        self._authority = authority
        self._control = control
        self._limit = context_limit
        self._pending = max_pending
        self._arm_ns = int(arm_timeout_seconds * 1_000_000_000)
        if self._arm_ns < 1:
            raise ValueError("Cold prefill needs a positive arm deadline")
        self._clock = clock
        self._offers: dict[str, ColdPrefillOffer] = {}
        self._aborts: dict[str, dict[int, HostRankSettlement]] = {}

    def offer(self, prompt: ColdPrefillPrompt) -> ColdPrefillOffer | Wait:
        self._authority._require_owner()
        if type(prompt) is not ColdPrefillPrompt or prompt.tokens > self._limit:
            raise ValueError("Cold prefill exceeds the native context limit")
        issued = self._offers.get(prompt.request_id)
        if issued is not None:
            if (
                issued.prompt != prompt
                or prompt.request_id in self._aborts
                or self._clock() >= issued.arm_deadline_ns
            ):
                raise ValueError(
                    "Cold prefill request already has a different or expired lease"
                )
            return issued
        if len(self._offers) >= self._pending:
            return Wait("RAM_ADMISSION_WAIT")

        grant = self._authority.reserve(
            prompt.request_id,
            prompt.transfer_id,
            prompt.epoch,
            prefix_tokens=0,
            reservation_tokens=prompt.tokens,
            source=(),
        )
        if isinstance(grant, Wait):
            return grant

        try:
            result = ColdPrefillOffer(
                prompt, grant, self._control.p_boot_id, self._clock() + self._arm_ns
            )
            self._authority.dispatch(grant)
        except Exception:
            self._authority.cancel(grant)
            raise
        self._offers[prompt.request_id] = result
        return result

    def verify_ready(
        self, offer: ColdPrefillOffer, receipts: tuple[ReadyReceipt, ...], mac: str
    ) -> NativeHostGrant:
        self._require_offer(offer)
        if (
            offer.prompt.request_id in self._aborts
            or not _digest(mac)
            or not compare_digest(self._control.ready_mac(offer, receipts), mac)
        ):
            raise ValueError("Cold prefill READY quorum is not authenticated")
        for receipt in receipts:
            self._authority.settle(
                offer.grant,
                HostRankSettlement(offer.grant.sha256, receipt.rank, "ready", 0, 0),
            )
        return offer.grant

    def _require_offer(self, offer: ColdPrefillOffer) -> None:
        self._authority._require_owner()
        if self._offers.get(offer.prompt.request_id) is not offer:
            raise ValueError("Cold prefill offer is no longer owned by this D boot")

    def detach_adopted(self, offer: ColdPrefillOffer) -> None:
        self._require_offer(offer)
        if self._authority.retains(offer.grant):
            raise ValueError("Cold prefill lease is still owned by the RAM authority")
        self._offers.pop(offer.prompt.request_id)
        self._aborts.pop(offer.prompt.request_id, None)

    def settle_abort(
        self, offer: ColdPrefillOffer, settlements: tuple[HostRankSettlement, ...]
    ) -> bool:
        """Consume native abort fences supplied by the authenticated transport owner."""
        self._require_offer(offer)
        if any(
            item.result != "abort_settled" or item.lease_sha256 != offer.grant.sha256
            for item in settlements
        ):
            raise ValueError("Cold prefill abort fences must match this native offer")
        request_id = offer.prompt.request_id
        pending = self._aborts.setdefault(request_id, {})
        for item in settlements:
            if pending.setdefault(item.rank, item) != item:
                raise ValueError("Conflicting cold prefill abort fence")
        if pending.keys() != {0, 1, 2, 3}:
            return False
        # A prior READY quorum proves the save fence, not a subsequently
        # opened Mooncake source. Require all four abort fences before releasing.
        if not self._authority.cancel(offer.grant):
            for item in pending.values():
                self._authority.settle(offer.grant, item)
        self._offers.pop(request_id)
        self._aborts.pop(request_id)
        return True
