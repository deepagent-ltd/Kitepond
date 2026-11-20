# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KV layout and exact state byte selection."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal

import torch

SCHEMA = "glm5next-pd-layout"
VERSION = 2


class CacheRole(str, Enum):
    MLA = "mla"
    INDEXER = "indexer"
    KDA_CONV = "kda_conv"
    KDA_RECURRENT = "kda_recurrent"
    KPOOL_TAIL = "kpool_tail"


@dataclass(frozen=True)
class CacheIdentity:
    layer_name: str
    cache_role: CacheRole
    model_role: Literal["target", "draft"]

    def __post_init__(self) -> None:
        if not self.layer_name or self.layer_name.strip() != self.layer_name:
            raise ValueError("cache identity requires an exact nonempty layer name")
        if not isinstance(self.cache_role, CacheRole):
            raise ValueError("cache identity requires a CacheRole")
        if self.model_role not in {"target", "draft"}:
            raise ValueError("model_role must be target or draft")


@dataclass(frozen=True)
class Glm5NextPDLayout:
    request_id: str
    model_fingerprint: str
    config_fingerprint: str
    producer_owner: int
    epoch: int
    target_valid_tokens: int
    draft_valid_tokens: int
    committed_kda_slot: int
    boundary_status: Literal["decode_ready", "needs_draft_bridge"]
    conv_base_slot: int = 0
    dcp_size: int = 4
    interleave_size: int = 4
    schema: str = SCHEMA
    version: int = VERSION

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or self.version != VERSION:
            raise ValueError("unsupported GLM P/D layout schema/version")
        if not self.request_id:
            raise ValueError("request_id is required")
        for fingerprint in (self.model_fingerprint, self.config_fingerprint):
            if len(fingerprint) != 64 or any(
                char not in "0123456789abcdef" for char in fingerprint
            ):
                raise ValueError("model/config fingerprints must be SHA256 hex")
        if self.dcp_size != 4 or self.interleave_size != 4:
            raise ValueError("GLM P/D layout requires DCP4/interleave4")
        if type(self.producer_owner) is not int or not 0 <= self.producer_owner < 4:
            raise ValueError("producer_owner must identify one of four P owners")
        for value in (self.epoch, self.target_valid_tokens, self.draft_valid_tokens):
            if type(value) is not int or value < 0:
                raise ValueError("epoch and valid token boundaries must be nonnegative")
        if (
            type(self.committed_kda_slot) is not int
            or not 0 <= self.committed_kda_slot < 4
        ):
            raise ValueError("committed_kda_slot must be a native MTP3 slot")
        if type(self.conv_base_slot) is not int or self.conv_base_slot != 0:
            raise ValueError("native GLM conv base slot must be zero")
        if self.boundary_status not in {"decode_ready", "needs_draft_bridge"}:
            raise ValueError("explicit target/draft boundary status is required")
        if self.draft_valid_tokens > self.target_valid_tokens:
            raise ValueError("draft valid boundary cannot exceed the target boundary")

    def valid_tokens(self, identity: CacheIdentity) -> int:
        return (
            self.target_valid_tokens
            if identity.model_role == "target"
            else self.draft_valid_tokens
        )

    def require_decode_ready(self) -> None:
        if self.boundary_status != "decode_ready":
            raise ValueError("P/D handoff needs_draft_bridge; D decode is not ready")
        if self.draft_valid_tokens != self.target_valid_tokens:
            raise ValueError(
                "decode-ready K0 handoff requires equal target/draft boundaries"
            )

    def to_metadata(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TensorLayout:
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int


def describe_tensor(tensor: torch.Tensor) -> TensorLayout:
    return TensorLayout(
        str(tensor.dtype),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
    )


@dataclass(frozen=True)
class PackedRows:
    identity: CacheIdentity
    dcp_rank: int
    local_row_ids: torch.Tensor
    payload: torch.Tensor
    source_layout: TensorLayout


def _check_rank(dcp_rank: int) -> None:
    if type(dcp_rank) is not int or not 0 <= dcp_rank < 4:
        raise ValueError("DCP rank must be in [0, 4)")


def dcp_row_mapping(
    global_row_ids: torch.Tensor,
    *,
    cache_role: CacheRole,
    dcp_size: int = 4,
    interleave_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dcp_size != 4 or interleave_size != 4:
        raise ValueError("GLM P/D layout requires DCP4/interleave4")
    if global_row_ids.ndim != 1 or global_row_ids.dtype != torch.int64:
        raise ValueError("global row ids must be a one-dimensional int64 tensor")
    if bool(torch.any(global_row_ids < 0)):
        raise ValueError("global row ids must be nonnegative")
    if cache_role is CacheRole.MLA:
        return (global_row_ids // 4) % 4, (
            global_row_ids // 16
        ) * 4 + global_row_ids % 4
    if cache_role is CacheRole.INDEXER:
        return global_row_ids % 4, global_row_ids // 4
    raise ValueError("DCP row mapping requires MLA or INDEXER cache role")


def pack_kv_rows(
    source: torch.Tensor,
    global_row_ids: torch.Tensor,
    *,
    identity: CacheIdentity,
    layout: Glm5NextPDLayout,
    dcp_rank: int,
) -> PackedRows:
    """Pack native rows without changing their representation or logical position."""
    _check_rank(dcp_rank)
    role = identity.cache_role
    if role not in {CacheRole.MLA, CacheRole.INDEXER}:
        raise ValueError("row packing requires MLA or INDEXER identity")
    widths = (528, 656) if role is CacheRole.MLA else (132,)
    if source.ndim != 2 or source.dtype != torch.uint8 or source.shape[1] not in widths:
        raise ValueError("GLM KV requires uint8 MLA rows528/656 or indexer rows132")
    if (
        source.device != global_row_ids.device
        or source.shape[0] != global_row_ids.numel()
    ):
        raise ValueError("source and logical row ids must share row count and device")
    owners, local_ids = dcp_row_mapping(
        global_row_ids,
        cache_role=role,
        dcp_size=layout.dcp_size,
        interleave_size=layout.interleave_size,
    )
    if torch.unique(global_row_ids).numel() != global_row_ids.numel():
        raise ValueError("initialized global row ids must be unique")
    valid = layout.valid_tokens(identity)
    if role is CacheRole.INDEXER:
        valid //= 4
    selected = (global_row_ids < valid) & (owners == dcp_rank)
    return PackedRows(
        identity,
        dcp_rank,
        local_ids[selected].contiguous(),
        source[selected].contiguous(),
        describe_tensor(source),
    )


def _check_kda(
    source: torch.Tensor,
    *,
    identity: CacheIdentity,
    layout: Glm5NextPDLayout,
    dcp_rank: int,
    source_committed_slot: int,
    role: CacheRole,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    _check_rank(dcp_rank)
    if identity.cache_role is not role or identity.model_role != "target":
        raise ValueError("KDA transfer requires the exact target state role")
    expected_slot = (
        layout.conv_base_slot
        if role is CacheRole.KDA_CONV
        else layout.committed_kda_slot
    )
    if type(source_committed_slot) is not int or source_committed_slot != expected_slot:
        state_kind = (
            "conv base" if role is CacheRole.KDA_CONV else "committed recurrent"
        )
        raise ValueError(f"selected KDA view is not the {state_kind} native state slot")
    if source.dtype != dtype or tuple(source.shape) != shape:
        raise ValueError(f"GLM {role.value} requires {dtype} shape{shape}")


def slice_kda_conv(
    source: torch.Tensor,
    *,
    identity: CacheIdentity,
    layout: Glm5NextPDLayout,
    dcp_rank: int,
    source_committed_slot: int,
) -> torch.Tensor:
    _check_kda(
        source,
        identity=identity,
        layout=layout,
        dcp_rank=dcp_rank,
        source_committed_slot=source_committed_slot,
        role=CacheRole.KDA_CONV,
        dtype=torch.bfloat16,
        shape=(6, 24576),
    )
    width = 16 * 128
    start = dcp_rank * width
    return torch.cat(
        [
            source[:, segment * 8192 + start : segment * 8192 + start + width]
            for segment in range(3)
        ],
        dim=1,
    )


def slice_kda_recurrent(
    source: torch.Tensor,
    *,
    identity: CacheIdentity,
    layout: Glm5NextPDLayout,
    dcp_rank: int,
    source_committed_slot: int,
) -> torch.Tensor:
    _check_kda(
        source,
        identity=identity,
        layout=layout,
        dcp_rank=dcp_rank,
        source_committed_slot=source_committed_slot,
        role=CacheRole.KDA_RECURRENT,
        dtype=torch.float32,
        shape=(64, 128, 128),
    )
    return source[dcp_rank * 16 : (dcp_rank + 1) * 16].clone(
        memory_format=torch.contiguous_format
    )


def copy_kpool_tail(
    source: torch.Tensor,
    *,
    identity: CacheIdentity,
    layout: Glm5NextPDLayout,
) -> torch.Tensor:
    if identity.cache_role is not CacheRole.KPOOL_TAIL:
        raise ValueError("tail copying requires KPOOL_TAIL identity")
    if source.dtype != torch.bfloat16 or tuple(source.shape) != (8, 2, 128):
        raise ValueError("GLM kpool tail requires BF16 shape(8, 2, 128)")
    layout.valid_tokens(identity)
    return source.clone(memory_format=torch.contiguous_format)
