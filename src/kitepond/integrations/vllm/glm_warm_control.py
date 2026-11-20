# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Authenticate the native GLM handoff messages across API processes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from hmac import compare_digest
from hmac import digest as hmac_digest
from json import dumps
from time import monotonic_ns

from kitepond.backends.fsl2 import BundlePlan
from kitepond.core.readiness import _digest

from .glm_terminal_manifest import (
    GlmTerminalManifest,
    SignedTerminalRank,
    TerminalKDASelection,
    TerminalRankCheckpoint,
)
from .glm_terminal_return import GlmTerminalReturn, PrefillTerminalReservation
from .glm_warm_decode_scheduler import GlmPublishedTerminalSource
from .host_authority import NativeHostGrant
from .terminal_checkpoint import (
    TerminalCacheGroup,
    TerminalPinLease,
    TerminalReadPermit,
    TerminalTransferIdentity,
)

NAMESPACE = "kitepond.glm-warm-return.v1"
OPERATIONS = frozenset(
    {
        "warm_arm",
        "warm_source",
        "warm_authorize",
        "warm_destination",
        "warm_prepare",
        "warm_submit",
        "warm_poll",
        "warm_release",
        "warm_cancel",
    }
)


def _groups(values) -> tuple[TerminalCacheGroup, ...]:
    return tuple(
        TerminalCacheGroup(
            **(
                group
                | {
                    "layer_names": tuple(group["layer_names"]),
                    "block_ids": tuple(group["block_ids"]),
                }
            )
        )
        for group in values
    )


def _lease(body: dict) -> TerminalPinLease:
    return TerminalPinLease(
        **(
            body
            | {
                "identity": TerminalTransferIdentity(**body["identity"]),
                "groups": _groups(body["groups"]),
            }
        )
    )


def _signed_rank(body: dict) -> SignedTerminalRank:
    raw = body["checkpoint"]
    checkpoint = TerminalRankCheckpoint(
        **(
            raw
            | {
                "role_catalog": tuple(raw["role_catalog"]),
                "kda": tuple(TerminalKDASelection(**item) for item in raw["kda"]),
                "tail_blocks": tuple(tuple(item) for item in raw["tail_blocks"]),
                "tail_positions": tuple(
                    (name, tuple(positions))
                    for name, positions in raw["tail_positions"]
                ),
                "plan": BundlePlan.from_document(raw["plan"]),
            }
        )
    )
    return SignedTerminalRank(**(body | {"checkpoint": checkpoint}))


def _manifest(body: dict) -> GlmTerminalManifest:
    result = GlmTerminalManifest(
        **(body | {"ranks": tuple(_signed_rank(item) for item in body["ranks"])})
    )
    if len(result.ranks) != 4 or tuple(
        item.checkpoint.rank for item in result.ranks
    ) != (0, 1, 2, 3):
        raise ValueError("Warm source needs four ordered signed rank manifests")
    return result


def _destination(body: dict) -> PrefillTerminalReservation:
    raw = body["grant"]
    grant = NativeHostGrant(
        **(
            raw
            | {
                "readonly_pages": tuple(raw["readonly_pages"]),
                "writable_pages": tuple(raw["writable_pages"]),
                "copy_tail": None
                if raw["copy_tail"] is None
                else tuple(raw["copy_tail"]),
            }
        )
    )
    return PrefillTerminalReservation(
        **(body | {"grant": grant, "groups": _groups(body["groups"])})
    )


def _source(body: dict) -> GlmPublishedTerminalSource:
    return GlmPublishedTerminalSource(
        **(
            body
            | {"lease": _lease(body["lease"]), "manifest": _manifest(body["manifest"])}
        )
    )


def _transfer(body: dict) -> GlmTerminalReturn:
    return GlmTerminalReturn(
        **(
            body
            | {
                "lease": _lease(body["lease"]),
                "permit": TerminalReadPermit(**body["permit"]),
                "manifest": _manifest(body["manifest"]),
                "destination": _destination(body["destination"]),
            }
        )
    )


_DECODERS = {
    "source": (GlmPublishedTerminalSource, _source),
    "destination": (PrefillTerminalReservation, _destination),
    "transfer": (GlmTerminalReturn, _transfer),
}


class GlmWarmControl:
    """Use the existing private deployment key, bound to both native boots."""

    def __init__(self, cold_control) -> None:
        self._cold = cold_control

    def _mac(self, kind: str, body: dict) -> str:
        canonical = dumps(
            (NAMESPACE, kind, self._cold.d_identity, self._cold.p_boot_id, body),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hmac_digest(self._cold._key, canonical.encode(), "sha256").hex()

    def seal(self, kind, data) -> dict:
        expected, _ = _DECODERS[kind]
        if type(data) is not expected:
            raise ValueError("Native warm control changed its outbound message type")
        body = asdict(data)
        return {"body": body, "mac": self._mac(kind, body)}

    def open(
        self,
        kind,
        envelope,
        *,
        allow_expired=False,
        clock: Callable[[], int] = monotonic_ns,
    ):
        _, decode = _DECODERS[kind]
        if (
            type(envelope) is not dict
            or envelope.keys() != {"body", "mac"}
            or type(envelope["body"]) is not dict
            or not _digest(envelope["mac"])
            or not compare_digest(self._mac(kind, envelope["body"]), envelope["mac"])
        ):
            raise ValueError("Native warm control message is not authenticated")
        data = decode(envelope["body"])
        if kind in {"source", "transfer"}:
            lease, manifest = data.lease, data.manifest
            identity = lease.identity
            if (
                (identity.source_engine_id, identity.source_boot_id)
                != self._cold.d_identity
                or (not allow_expired and identity.deadline_monotonic_ns <= clock())
                or manifest.lease_sha256 != lease.sha256
            ):
                raise ValueError("Native warm source does not match its live identity")
        if kind in {"destination", "transfer"}:
            destination = data if kind == "destination" else data.destination
            if (
                (destination.grant.engine_id, destination.grant.boot_id)
                != self._cold.d_identity
                or destination.boot_id != self._cold.p_boot_id
                or destination.request_id != destination.grant.request_id
            ):
                raise ValueError("Native warm P destination belongs to another boot")
        if kind == "transfer" and (
            data.permit.lease_sha256 != data.lease.sha256
            or data.permit.manifest_sha256 != data.manifest.sha256
            or data.lease.identity.epoch != data.destination.grant.epoch
        ):
            raise ValueError("Native warm permit changed its destination")
        return data
