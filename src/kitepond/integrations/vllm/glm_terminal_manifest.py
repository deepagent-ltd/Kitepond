# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify local GLM worker evidence before opening D terminal reads."""

from __future__ import annotations

import hmac
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import torch
from kitepond.backends.fsl2 import BundlePlan, canonical, digest
from kitepond.integrations.vllm.terminal_checkpoint import TerminalPinLease
from kitepond.models.glm5next.frontier import digest as frontier_digest
from kitepond.models.glm5next.import_recipe import (
    ROW_WIDTHS,
    part_identity,
    part_role,
    validate_restore_coverage,
)
from kitepond.models.glm5next.layout import CacheIdentity, CacheRole

if TYPE_CHECKING:
    from kitepond.backends.fsl2 import PartSpec
    from kitepond.integrations.vllm.glm_terminal_worker import (
        GlmTerminalRunner,
        NativeTerminalRows,
    )
    from kitepond.models.glm5next.cold_restore import ColdRestoreLayout

_SCHEMA = "kitepond-glm-terminal-checkpoint-v1"
_FRAME_SCHEMA = "kitepond-glm-terminal-frame-v1"
_SIGNATURE_DOMAIN = b"kitepond-glm-terminal-rank-v1\x00"
_SOURCE_RANKS = frozenset(range(4))


def _is_digest(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


@dataclass(frozen=True)
class TerminalRankSigner:
    rank: int
    key: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank not in _SOURCE_RANKS:
            raise ValueError("Invalid terminal worker rank")
        if type(self.key) is not bytes or len(self.key) != 32:
            raise ValueError("Terminal worker signing key must contain 32 bytes")

    def _sign(self, value: object) -> str:
        return hmac.digest(
            self.key, _SIGNATURE_DOMAIN + canonical(value), "sha256"
        ).hex()


@dataclass(frozen=True)
class TerminalFrameProof:
    rank: int
    lease_sha256: str
    boundary_tokens: int
    imported_tokens: int
    target_sha256: str
    draft_sha256: str
    selectors_sha256: str
    captured_monotonic_ns: int


@dataclass(frozen=True)
class TerminalFrameReceipt:
    proof: TerminalFrameProof
    signature: str

    @property
    def sha256(self) -> str:
        return digest({"schema": _FRAME_SCHEMA, **asdict(self)})


def _frame_message(proof: TerminalFrameProof) -> dict:
    return {"schema": _FRAME_SCHEMA, **asdict(proof)}


@dataclass(frozen=True)
class FencedTerminalSource:
    """A model-owned rank source described after native cache writes settle."""

    layout: ColdRestoreLayout
    authority_sha256: str
    representation_sha256: str
    tail_positions: tuple[tuple[str, tuple[int, ...]], ...]

    def read_chunk(self, part: PartSpec) -> torch.Tensor:
        """Read accepted history from native cache and normalize wire scratch."""
        identity = part_identity(part.role)
        if identity.cache_role is CacheRole.KDA_CONV:
            raw = self.layout.read_chunk(part).view(torch.bfloat16).view(part.shape)
            accepted = self.layout.layout.committed_kda_slot
            history_rows = 3
            canonical = torch.zeros_like(raw)
            canonical[:history_rows].copy_(raw[accepted : accepted + history_rows])
            return canonical.view(torch.uint8).flatten()
        if identity.cache_role is not CacheRole.KPOOL_TAIL:
            return self.layout.read_chunk(part)

        tails = dict(self.tail_positions)
        boundary = self.layout.layout.valid_tokens(identity)
        if len(tails) != len(self.tail_positions) or tails.get(part.role) != tuple(
            range(boundary - boundary % 4, boundary)
        ):
            raise ValueError(
                "Native terminal export lacks the exact accepted KPool tail"
            )
        payload = self.layout.read_chunk(part).clone()
        ring = payload.view(torch.bfloat16).view(8, 2, 128)
        keep = {position % ring.shape[0] for position in tails[part.role]}
        for slot in range(ring.shape[0]):
            if slot not in keep:
                ring[slot].zero_()
        return payload


@dataclass(frozen=True)
class TerminalKDASelection:
    layer: str
    conv_block: int
    recurrent_block: int
    accepted_row: int


@dataclass(frozen=True)
class TerminalRankCheckpoint:
    rank: int
    lease_sha256: str
    boundary_tokens: int
    target_sha256: str
    draft_sha256: str
    frame_sha256: str
    authority_sha256: str
    role_catalog: tuple[str, ...]
    kda: tuple[TerminalKDASelection, ...]
    tail_blocks: tuple[tuple[str, int], ...]
    tail_positions: tuple[tuple[str, tuple[int, ...]], ...]
    plan: BundlePlan
    completed_monotonic_ns: int


@dataclass(frozen=True)
class SignedTerminalRank:
    checkpoint: TerminalRankCheckpoint
    signature: str


@dataclass(frozen=True)
class GlmTerminalManifest:
    lease_sha256: str
    ranks: tuple[SignedTerminalRank, ...]

    @property
    def sha256(self) -> str:
        return digest({"schema": _SCHEMA, **asdict(self)})


def _pinned_layers(lease: TerminalPinLease) -> dict[str, tuple[str, tuple[int, ...]]]:
    result = {}
    for group in lease.groups:
        for layer in group.layer_names:
            if layer in result:
                raise ValueError("Native terminal layer has multiple owning groups")
            result[layer] = (group.location, group.block_ids)
    return result


def _expected_tail(
    roles: tuple[CacheIdentity, ...], boundary: int
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    positions = tuple(range(boundary - boundary % 4, boundary))
    return tuple(
        sorted(
            (part_role(identity), positions)
            for identity in roles
            if identity.cache_role is CacheRole.KPOOL_TAIL
        )
    )


def _accepted_row(lease: TerminalPinLease, observed: NativeTerminalRows) -> int:
    target = observed.target
    draft = observed.draft
    frontier = target["frontier"]
    b = lease.requested_common_tokens
    start = frontier["last_frame_start"]
    accepted_row = b - 1 - start
    if (
        frontier != draft["frontier"]
        or frontier["native_Request_prefix_sha256"] != lease.prefix_sha256
        or frontier["terminal_common_B"] != b
        or not 0 <= accepted_row < 4
        or any(
            [item[0] for item in rows] != list(range(start, b))
            or any(not _is_digest(item[1]) for item in rows)
            for rows in (target["latest_rows"], draft["latest_shifted_rows"])
        )
    ):
        raise ValueError("GLM worker's accepted terminal frontier differs from its pin")
    return accepted_row


def _kda_selections(
    observed: NativeTerminalRows, native_source: FencedTerminalSource, row: int
) -> tuple[TerminalKDASelection, ...]:
    layers = native_source.layout.layers
    conv = {
        layer.identity.layer_name: layer
        for layer in layers.values()
        if layer.identity.cache_role is CacheRole.KDA_CONV
    }
    recurrent = {
        layer.identity.layer_name: layer
        for layer in layers.values()
        if layer.identity.cache_role is CacheRole.KDA_RECURRENT
    }
    if (
        not conv
        or conv.keys() != recurrent.keys()
        or conv.keys() != observed.kda_selectors.keys()
    ):
        raise ValueError("GLM terminal worker lost an authoritative KDA layer")

    result = []
    for layer in sorted(conv):
        selector = observed.kda_selectors[layer]
        if selector.keys() == {"spec_state_indices_tensor"}:
            [states] = selector["spec_state_indices_tensor"]
            if len(states) != 4:
                raise ValueError(
                    "GLM exact terminal capture requires native MTP3 selectors"
                )
        elif selector.keys() == {"non_spec_state_indices_tensor"} and row == 0:
            states = selector["non_spec_state_indices_tensor"]
            if len(states) != 1:
                raise ValueError("GLM terminal non-spec KDA selector is ambiguous")
        else:
            raise ValueError("No valid native KDA selector for the accepted row")
        conv_block, recurrent_block = states[0], states[row]
        if (
            type(conv_block) is not int
            or type(recurrent_block) is not int
            or min(conv_block, recurrent_block) <= 0
            or conv_block != conv[layer].block_ids[0]
            or row >= len(recurrent[layer].block_ids)
            or recurrent_block != recurrent[layer].block_ids[row]
        ):
            raise ValueError(
                "GLM accepted KDA selector differs from the native source block"
            )
        result.append(TerminalKDASelection(layer, conv_block, recurrent_block, row))
    return tuple(result)


def take_terminal_frame(
    runner: GlmTerminalRunner,
    lease: TerminalPinLease,
    imported_tokens: int,
    signer: TerminalRankSigner,
    *,
    clock: Callable[[], int] = time.monotonic_ns,
) -> tuple[NativeTerminalRows, TerminalFrameReceipt]:
    """Save the one accepted frame without opening late host/source work."""
    if (
        type(imported_tokens) is not int
        or not 0 <= imported_tokens < lease.requested_common_tokens
        or lease.identity.deadline_monotonic_ns <= clock()
    ):
        raise ValueError("GLM terminal worker frame is expired or lacks its import")
    observed = runner.terminal_rows(lease.identity.request_id, lease, imported_tokens)
    _accepted_row(lease, observed)
    when = clock()
    if not 0 < when < lease.identity.deadline_monotonic_ns:
        raise ValueError("GLM terminal worker frame completed too late")
    proof = TerminalFrameProof(
        signer.rank,
        lease.sha256,
        lease.requested_common_tokens,
        imported_tokens,
        frontier_digest(observed.target),
        frontier_digest(observed.draft),
        frontier_digest(observed.kda_selectors),
        when,
    )
    return observed, TerminalFrameReceipt(proof, signer._sign(_frame_message(proof)))


def finalize_terminal_rank(
    lease: TerminalPinLease,
    observed: NativeTerminalRows,
    frame: TerminalFrameReceipt,
    signer: TerminalRankSigner,
    wait_native_source: Callable[
        [TerminalPinLease, NativeTerminalRows], FencedTerminalSource
    ],
    *,
    clock: Callable[[], int] = time.monotonic_ns,
) -> SignedTerminalRank:
    """Bind the saved frame to the independently settled native source."""
    proof = frame.proof
    if (
        proof.rank != signer.rank
        or proof.lease_sha256 != lease.sha256
        or proof.boundary_tokens != lease.requested_common_tokens
        or not hmac.compare_digest(frame.signature, signer._sign(_frame_message(proof)))
        or not 0
        < proof.captured_monotonic_ns
        <= clock()
        < lease.identity.deadline_monotonic_ns
    ):
        raise ValueError("GLM terminal checkpoint has no matching captured frame")
    row = _accepted_row(lease, observed)
    source = wait_native_source(lease, observed)
    if (
        proof.target_sha256 != frontier_digest(observed.target)
        or proof.draft_sha256 != frontier_digest(observed.draft)
        or proof.selectors_sha256 != frontier_digest(observed.kda_selectors)
    ):
        raise ValueError("GLM terminal source altered its captured frame")
    if type(source) is not FencedTerminalSource:
        raise TypeError("GLM terminal worker needs its fenced native source view")
    layout = source.layout.layout
    b = lease.requested_common_tokens
    identity = lease.identity
    if (
        source.layout.rank != signer.rank
        or layout.request_id != identity.request_id
        or layout.epoch != identity.epoch
        or layout.model_fingerprint != identity.model_fingerprint
        or layout.config_fingerprint != identity.config_fingerprint
        or layout.target_valid_tokens != b
        or layout.draft_valid_tokens != b
        or layout.committed_kda_slot != row
        or not _is_digest(source.authority_sha256)
        or not _is_digest(source.representation_sha256)
    ):
        raise ValueError("Fenced GLM source changed the terminal layout or identity")

    pinned = _pinned_layers(lease)
    roles = tuple(layer.identity for layer in source.layout.layers.values())
    for layer in source.layout.layers.values():
        cache_role = layer.identity.cache_role
        name = layer.identity.layer_name
        expected_location = "host" if cache_role is CacheRole.MLA else "device"
        location, owned = pinned.get(name, (None, ()))
        if location != expected_location:
            raise ValueError("GLM terminal source is outside its native pinned layer")
        if cache_role in ROW_WIDTHS:
            exact = layer.block_ids == owned
        else:
            slot = row if cache_role is CacheRole.KDA_RECURRENT else 0
            exact = (
                slot < len(layer.block_ids)
                and 0 < layer.block_ids[slot] < layer.tensor.shape[0]
                and layer.block_ids[slot] in owned
            )
        if not exact:
            raise ValueError(
                "GLM terminal source selected a different native block table"
            )
    if source.tail_positions != _expected_tail(roles, b):
        raise ValueError("GLM KPool tail lacks exact accepted write evidence")

    parts = source.layout.manifest()
    plan = BundlePlan(
        _SCHEMA,
        source.representation_sha256,
        tuple(sorted({part.role for part in parts})),
        parts,
    )
    completed = clock()
    if not (
        proof.captured_monotonic_ns <= completed < lease.identity.deadline_monotonic_ns
    ):
        raise ValueError("Fenced GLM terminal proof was captured too late")
    checkpoint = TerminalRankCheckpoint(
        signer.rank,
        lease.sha256,
        b,
        proof.target_sha256,
        proof.draft_sha256,
        frame.sha256,
        source.authority_sha256,
        tuple(sorted(part_role(identity) for identity in roles)),
        _kda_selections(observed, source, row),
        tuple(
            sorted(
                (part_role(layer.identity), layer.block_ids[0])
                for layer in source.layout.layers.values()
                if layer.identity.cache_role is CacheRole.KPOOL_TAIL
            )
        ),
        source.tail_positions,
        plan,
        completed,
    )
    return SignedTerminalRank(checkpoint, signer._sign(asdict(checkpoint)))


def capture_terminal_rank(
    runner: GlmTerminalRunner,
    lease: TerminalPinLease,
    imported_tokens: int,
    signer: TerminalRankSigner,
    wait_native_source: Callable[
        [TerminalPinLease, NativeTerminalRows], FencedTerminalSource
    ],
    *,
    clock: Callable[[], int] = time.monotonic_ns,
) -> SignedTerminalRank:
    """Capture and finalize together when the native source is already durable."""
    observed, frame = take_terminal_frame(
        runner, lease, imported_tokens, signer, clock=clock
    )
    return finalize_terminal_rank(
        lease, observed, frame, signer, wait_native_source, clock=clock
    )


def _row_count(boundary: int, rank: int, role: CacheRole) -> int | None:
    if role is CacheRole.INDEXER:
        return max(0, (boundary // 4 + 3 - rank) // 4)
    if role is CacheRole.MLA:
        full, rest = divmod(boundary, 16)
        return full * 4 + min(4, max(0, rest - rank * 4))
    return None


class GlmTerminalCheckpointVerifier:
    """Scheduler-side verifier for StockTerminalCheckpoints.begin_reads."""

    def __init__(
        self,
        rank_keys: Mapping[int, bytes],
        authority_by_rank: Mapping[int, str],
        verify_host_prefix: Callable[[TerminalPinLease], None],
        roles: tuple[CacheIdentity, ...],
        *,
        representation_sha256: str,
        mla_row_bytes: int = 656,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if (
            rank_keys.keys() != _SOURCE_RANKS
            or authority_by_rank.keys() != _SOURCE_RANKS
            or len(set(rank_keys.values())) != 4
            or len(set(authority_by_rank.values())) != 4
            or any(not _is_digest(value) for value in authority_by_rank.values())
            or not callable(verify_host_prefix)
            or not _is_digest(representation_sha256)
            or mla_row_bytes not in ROW_WIDTHS[CacheRole.MLA]
        ):
            raise ValueError(
                "GLM terminal verifier requires four distinct enrolled ranks"
            )
        self._signers = {
            rank: TerminalRankSigner(rank, key) for rank, key in rank_keys.items()
        }
        self._authority = dict(authority_by_rank)
        self._representation_sha256 = representation_sha256
        self._mla_row_bytes = mla_row_bytes

        model_roles = {(identity.model_role, identity.cache_role) for identity in roles}
        required = {
            (model_role, role)
            for model_role in ("target", "draft")
            for role in (CacheRole.MLA, CacheRole.INDEXER, CacheRole.KPOOL_TAIL)
        } | {("target", CacheRole.KDA_CONV), ("target", CacheRole.KDA_RECURRENT)}
        if len(set(roles)) != len(roles) or model_roles != required:
            raise ValueError(
                "GLM terminal verifier requires the complete native role catalog"
            )
        self._roles = tuple(sorted(part_role(identity) for identity in roles))
        self._verify_host_prefix = verify_host_prefix
        self._clock = clock

    def verify_frames(
        self, lease: TerminalPinLease, frames: tuple[TerminalFrameReceipt, ...]
    ) -> tuple[TerminalFrameReceipt, ...]:
        if (
            len(frames) != 4
            or any(type(frame) is not TerminalFrameReceipt for frame in frames)
            or {frame.proof.rank for frame in frames} != _SOURCE_RANKS
        ):
            raise ValueError("GLM terminal step did not capture all four worker frames")
        now = self._clock()
        frontiers = set()
        for frame in frames:
            proof = frame.proof
            if (
                proof.lease_sha256 != lease.sha256
                or proof.boundary_tokens != lease.requested_common_tokens
                or not 0 <= proof.imported_tokens < proof.boundary_tokens
                or not 0
                < proof.captured_monotonic_ns
                <= now
                < lease.identity.deadline_monotonic_ns
                or not all(
                    _is_digest(v)
                    for v in (
                        proof.target_sha256,
                        proof.draft_sha256,
                        proof.selectors_sha256,
                        frame.signature,
                    )
                )
                or not hmac.compare_digest(
                    frame.signature,
                    self._signers[proof.rank]._sign(_frame_message(proof)),
                )
            ):
                raise ValueError(
                    "GLM terminal worker frame changed its lease or signature"
                )
            frontiers.add(
                (proof.imported_tokens, proof.target_sha256, proof.draft_sha256)
            )
        if len(frontiers) != 1:
            raise ValueError(
                "GLM terminal worker frames disagree on the accepted state"
            )
        return tuple(sorted(frames, key=lambda frame: frame.proof.rank))

    def verify_captured_manifest(
        self,
        lease: TerminalPinLease,
        proof: object,
        frames: tuple[TerminalFrameReceipt, ...],
    ) -> str:
        if (
            type(proof) is not GlmTerminalManifest
            or type(proof.ranks) is not tuple
            or len(proof.ranks) != 4
            or any(type(rank) is not SignedTerminalRank for rank in proof.ranks)
        ):
            raise ValueError("GLM terminal reads require a signed native manifest")
        frames = self.verify_frames(lease, frames)
        expected = tuple(
            (f.proof.rank, f.sha256, f.proof.target_sha256, f.proof.draft_sha256)
            for f in frames
        )
        actual = tuple(
            (c.rank, c.frame_sha256, c.target_sha256, c.draft_sha256)
            for c in (r.checkpoint for r in proof.ranks)
        )
        if actual != expected or any(
            rank.checkpoint.completed_monotonic_ns < frame.proof.captured_monotonic_ns
            for rank, frame in zip(proof.ranks, frames, strict=True)
        ):
            raise ValueError(
                "GLM terminal manifest differs from the committed worker frames"
            )
        return self.verify(lease, proof)

    def verify(
        self,
        lease: TerminalPinLease,
        proof: object,
    ) -> str:
        if type(proof) is not GlmTerminalManifest or proof.lease_sha256 != lease.sha256:
            raise ValueError("GLM terminal manifest is not bound to the active lease")
        if (
            type(proof.ranks) is not tuple
            or len(proof.ranks) != 4
            or any(type(item) is not SignedTerminalRank for item in proof.ranks)
            or tuple(item.checkpoint.rank for item in proof.ranks) != tuple(range(4))
        ):
            raise ValueError(
                "GLM terminal manifest needs four signed ranks in canonical order"
            )
        pinned = _pinned_layers(lease)
        now = self._clock()
        frontiers = set()
        first = None
        for receipt in proof.ranks:
            row = receipt.checkpoint
            rank = row.rank
            if (
                row.lease_sha256 != lease.sha256
                or row.boundary_tokens != lease.requested_common_tokens
                or row.authority_sha256 != self._authority[rank]
                or not _is_digest(row.target_sha256)
                or not _is_digest(row.draft_sha256)
                or not _is_digest(row.frame_sha256)
                or not 0
                < row.completed_monotonic_ns
                <= now
                < lease.identity.deadline_monotonic_ns
                or not _is_digest(receipt.signature)
                or not hmac.compare_digest(
                    self._signers[rank]._sign(asdict(row)), receipt.signature
                )
            ):
                raise ValueError(
                    "GLM terminal rank proof failed its signature or source identity"
                )
            self._validate_source_plan(lease, row, pinned)
            frontiers.add((row.target_sha256, row.draft_sha256))
            common = (
                row.plan.representation_fingerprint,
                row.kda,
                row.tail_blocks,
                row.tail_positions,
            )
            if first is not None and first != common:
                raise ValueError("GLM terminal ranks disagree on exact state selectors")
            first = common

        if len(frontiers) != 1:
            raise ValueError(
                "GLM terminal ranks disagree on accepted target/draft tokens"
            )
        if self._verify_host_prefix(lease) is not None:
            raise ValueError("Native terminal host fence did not confirm durability")
        return proof.sha256

    def _validate_source_plan(
        self,
        lease: TerminalPinLease,
        row: TerminalRankCheckpoint,
        pinned: Mapping[str, tuple[str, tuple[int, ...]]],
    ) -> None:
        parts = row.plan.parts
        if row.plan.representation_schema != _SCHEMA or any(
            part.rank != row.rank for part in parts
        ):
            raise ValueError(
                "GLM terminal plan named a different native rank or schema"
            )
        if row.plan.representation_fingerprint != self._representation_sha256:
            raise ValueError("GLM terminal plan changed the configured representation")
        if row.role_catalog != self._roles:
            raise ValueError(
                "GLM terminal plan lost the configured native role catalog"
            )
        groups = validate_restore_coverage(parts, parts)
        roles = tuple(part_identity(role) for role in self._roles)
        boundary = lease.requested_common_tokens
        if row.tail_positions != _expected_tail(roles, boundary):
            raise ValueError("GLM terminal plan lost its exact KPool tail frontier")
        if tuple(name for name, _ in row.tail_blocks) != tuple(
            name for name, _ in row.tail_positions
        ) or any(
            type(block) is not int
            or block <= 0
            or block not in pinned.get(part_identity(name).layer_name, (None, ()))[1]
            for name, block in row.tail_blocks
        ):
            raise ValueError("GLM terminal KPool tail block lost its native ownership")
        expected_parts = set()
        for role in roles:
            name = part_role(role)
            location, blocks = pinned.get(role.layer_name, (None, ()))
            expected_location = "host" if role.cache_role is CacheRole.MLA else "device"
            if location != expected_location:
                raise ValueError("GLM terminal rank plan refers to an unpinned layer")
            required = _row_count(boundary, row.rank, role.cache_role)
            if required != 0:
                expected_parts.add((name, row.rank))
                if not any(block > 0 for block in blocks):
                    raise ValueError(
                        "GLM terminal rank plan has no owned physical blocks"
                    )
            role_parts = groups.get((name, row.rank), ())
            if role.cache_role is CacheRole.MLA and any(
                part.shape[1] != self._mla_row_bytes for part in role_parts
            ):
                raise ValueError("GLM terminal plan changed the configured MLA format")
            if (
                required is not None
                and sum(part.shape[0] for part in role_parts) != required
            ):
                raise ValueError(
                    "GLM terminal byte plan differs from its exact boundary"
                )
        if groups.keys() != expected_parts:
            raise ValueError(
                "GLM terminal rank plan has missing or foreign cache parts"
            )

        state_layers = {
            identity.layer_name
            for identity in roles
            if identity.cache_role is CacheRole.KDA_CONV
        }
        recurrent_layers = {
            identity.layer_name
            for identity in roles
            if identity.cache_role is CacheRole.KDA_RECURRENT
        }
        if (
            not state_layers
            or state_layers != recurrent_layers
            or len(row.kda) != len(state_layers)
            or {selection.layer for selection in row.kda} != state_layers
        ):
            raise ValueError("GLM terminal rank lacks the exact KDA state pairs")
        for selection in row.kda:
            blocks = pinned[selection.layer][1]
            slot = selection.accepted_row
            if (
                type(slot) is not int
                or not 0 <= slot < 4
                or selection.conv_block not in blocks
                or selection.recurrent_block not in blocks
                or min(selection.conv_block, selection.recurrent_block) <= 0
            ):
                raise ValueError("GLM terminal KDA selector is outside its native pin")
