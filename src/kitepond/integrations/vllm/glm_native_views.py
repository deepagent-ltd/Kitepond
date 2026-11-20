# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Enroll the live GLM rank buffers for import and exact terminal export."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import torch
from kitepond.models.glm5next.cold_restore import ColdRestoreLayout, RestoreLayer
from kitepond.models.glm5next.import_recipe import part_role
from kitepond.models.glm5next.layout import CacheIdentity, CacheRole, Glm5NextPDLayout
from kitepond.models.glm5next.shared_mla import SharedMlaRankPages

from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheGroupRole,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from .glm_host_backing import describe_glm_host_backing
from .glm_terminal_manifest import FencedTerminalSource, _accepted_row, _is_digest

if TYPE_CHECKING:
    from .decode_receive import DecodeReceiveTicket
    from .glm_terminal_worker import GlmTerminalRunner, NativeTerminalRows
    from .terminal_checkpoint import TerminalPinLease

_ROLES = {
    (model, role)
    for model in ("target", "draft")
    for role in (CacheRole.MLA, CacheRole.INDEXER, CacheRole.KPOOL_TAIL)
} | {("target", CacheRole.KDA_CONV), ("target", CacheRole.KDA_RECURRENT)}
_MLA_PAGE_ROWS = 256
_MLA_ROW_BYTES = 656
_INDEXER_PAGE_ROWS = 64
_RING_SIZE = 8
_POOL_SIZE = 4


@dataclass(frozen=True)
class _NativeLayer:
    group_id: int
    identity: CacheIdentity
    tensor: torch.Tensor


def _as_row_cache(tensor: torch.Tensor, rows: int, width: int) -> torch.Tensor:
    if tensor.dtype != torch.uint8 or tensor.shape[-1] != width:
        raise ValueError("Native GLM byte-row width or dtype changed")
    if tensor.ndim == 2:
        tensor = tensor.view(-1, rows, width)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.shape[1:] != (1, rows, width):
        raise ValueError("Native GLM cache rows cannot be addressed by manager page")
    return tensor


class GlmNativeStateViews:
    """References to the one native rank's initialized cache allocation."""

    def __init__(self, runner: GlmTerminalRunner, rank: int) -> None:
        config = runner.kv_cache_config
        if (
            type(rank) is not int
            or not 0 <= rank < 4
            or config.hisparse_shared_host_pool
            or len(config.host_group_ids) != 1
            or runner.block_tables.cp_rank != rank
            or runner.block_tables.cp_size != 4
            or runner.block_tables.cp_interleave != 4
        ):
            raise ValueError("GLM terminal views require the live DCP4 rank")
        self.rank = rank
        self._groups = len(config.kv_cache_groups)
        self._layers: dict[str, _NativeLayer] = {}
        self._tail_owners: dict[str, object] = {}
        self._tail_positions: dict[str, torch.Tensor] = {}
        self._tail_attempts: dict[tuple[str, int], tuple[str, int, int]] = {}
        descriptors = set()
        context = runner.compilation_config.static_forward_context

        for group_id, group in enumerate(config.kv_cache_groups):
            spec = group.kv_cache_spec
            if isinstance(spec, (HiSparseResidentSpec, HiSparseHotSpec)):
                continue
            for name in group.layer_names:
                layer = context[name]
                layer_spec = (
                    spec.kv_cache_specs[name]
                    if isinstance(spec, UniformTypeKVCacheSpecs)
                    else spec
                )
                model_role: Literal["target", "draft"] = (
                    "draft" if layer.is_draft_layer else "target"
                )
                if group.role is KVCacheGroupRole.HISPARSE_SOURCE:
                    if (
                        not group.host_resident
                        or not isinstance(layer_spec, MLAAttentionSpec)
                        or not layer_spec.dcp_sharded
                        or layer_spec.block_size != _MLA_PAGE_ROWS
                    ):
                        raise ValueError(
                            "GLM source spec changed its DCP page geometry"
                        )
                    runtime = layer.hisparse_cache.runtime
                    host = runtime.registered_host_pool
                    descriptors.add(describe_glm_host_backing(host))
                    tensor = runtime.host_cache
                    if (
                        tensor.device.type != "cpu"
                        or tensor.untyped_storage().data_ptr()
                        != host.untyped_storage().data_ptr()
                    ):
                        raise ValueError(
                            "GLM source layer left its registered host pool"
                        )
                    source = _as_row_cache(tensor, _MLA_PAGE_ROWS, _MLA_ROW_BYTES)
                    if source.shape[0] != config.hisparse_host_num_blocks:
                        raise ValueError("GLM source manager and host view disagree")
                    self._add(group_id, name, model_role, CacheRole.MLA, source)
                elif group.role is KVCacheGroupRole.HISPARSE_INDEXER:
                    if (
                        group.host_resident
                        or not isinstance(layer_spec, MLAAttentionSpec)
                        or not layer_spec.dcp_sharded
                        or layer_spec.get_num_kernel_states(layer_spec.block_size)
                        != _INDEXER_PAGE_ROWS
                        or layer_spec.tokens_per_state != _POOL_SIZE
                    ):
                        raise ValueError("GLM indexer cache compression changed")
                    self._add(
                        group_id,
                        name,
                        model_role,
                        CacheRole.INDEXER,
                        _as_row_cache(layer.kv_cache, _INDEXER_PAGE_ROWS, 132),
                    )
                elif isinstance(layer_spec, CircularBufferSpec):
                    if (
                        group.host_resident
                        or layer.get_attn_backend().get_name() != "KPOOL_TAIL"
                        or layer_spec.block_size != _RING_SIZE
                        or layer_spec.dcp_sharded
                    ):
                        raise ValueError(
                            "GLM terminal trace requires native KPool ring"
                        )
                    self._add(
                        group_id, name, model_role, CacheRole.KPOOL_TAIL, layer.kv_cache
                    )
                    self._tail_owners[name] = layer
                elif (
                    isinstance(layer_spec, MambaSpec)
                    and not group.host_resident
                    and layer_spec.mamba_type is MambaAttentionBackendEnum.GDN_ATTN
                    and model_role == "target"
                ):
                    conv, recurrent = layer.kv_cache
                    self._add(group_id, name, model_role, CacheRole.KDA_CONV, conv)
                    self._add(
                        group_id, name, model_role, CacheRole.KDA_RECURRENT, recurrent
                    )
                else:
                    raise ValueError(
                        "GLM worker has an unqualified terminal cache group"
                    )

        roles = {
            (layer.identity.model_role, layer.identity.cache_role)
            for layer in self._layers.values()
        }
        if len(descriptors) != 1 or roles != _ROLES:
            raise ValueError("GLM worker does not own one complete rank-local cache")
        (self.descriptor,) = descriptors
        if self.descriptor.source_rank != rank:
            raise ValueError("GLM worker attached a foreign native authority shard")

    def _add(
        self,
        group_id: int,
        name: str,
        model: Literal["target", "draft"],
        role: CacheRole,
        tensor: torch.Tensor,
    ) -> None:
        identity = CacheIdentity(name, role, model)
        key = part_role(identity)
        if key in self._layers or tensor.device.type != (
            "cpu" if role is CacheRole.MLA else "cuda"
        ):
            raise ValueError("GLM native cache location or identity changed")
        self._layers[key] = _NativeLayer(group_id, identity, tensor)

    def role_catalog(self) -> tuple[CacheIdentity, ...]:
        """Canonical role names from this worker's initialized native caches."""
        return tuple(self._layers[name].identity for name in sorted(self._layers))

    def bind_tail_writers(self) -> None:
        """Enroll the native position stamps before any model graph is captured."""
        if self._tail_positions or any(
            not callable(getattr(layer, "bind_kpool_write_positions", None))
            or layer.kpool_write_positions is not None
            for layer in self._tail_owners.values()
        ):
            raise RuntimeError("The native engine has no unbound KPool writer witness")
        for name, layer in self._tail_owners.items():
            cache = layer.kv_cache
            positions = torch.full(
                (cache.shape[0], cache.shape[2]),
                -1,
                dtype=torch.int32,
                device=cache.device,
            )
            layer.bind_kpool_write_positions(positions)
            self._tail_positions[name] = positions

    def for_import(
        self,
        layout: Glm5NextPDLayout,
        block_ids: tuple[tuple[int, ...], ...],
    ) -> tuple[ColdRestoreLayout, dict[str, int]]:
        """Share registered views with the existing native D restore path."""
        if len(block_ids) != self._groups:
            raise ValueError("GLM worker requires the complete native D block table")
        groups = {role: layer.group_id for role, layer in self._layers.items()}
        layers = [
            RestoreLayer(layer.identity, layer.tensor, block_ids[layer.group_id])
            for layer in self._layers.values()
        ]
        return ColdRestoreLayout(layout, self.rank, layers), groups

    def for_terminal_return(
        self, layout: Glm5NextPDLayout, groups: Mapping[int, tuple[int, ...]]
    ) -> tuple[ColdRestoreLayout, dict[str, int]]:
        if set(groups) != {layer.group_id for layer in self._layers.values()} or any(
            not 0 <= group_id < self._groups for group_id in groups
        ):
            raise ValueError("Native warm P allocation changed its model layer groups")
        return self.for_import(
            layout, tuple(groups.get(group_id, ()) for group_id in range(self._groups))
        )

    def detached_host_layers(
        self, block_ids: tuple[int, ...]
    ) -> tuple[RestoreLayer, ...]:
        """Address only the pre-admission grant's real registered host views."""
        return tuple(
            RestoreLayer(layer.identity, layer.tensor, block_ids)
            for layer in self._layers.values()
            if layer.identity.cache_role is CacheRole.MLA
        )

    def project_shared_mla(
        self, layout: Glm5NextPDLayout, writable_pages: tuple[int, ...], attempt
    ) -> SharedMlaRankPages:
        """Read the projected source from enrolled D host views before adoption."""
        cache = ColdRestoreLayout(
            layout,
            self.rank,
            [
                RestoreLayer(layer.identity, layer.tensor, writable_pages)
                for layer in self._layers.values()
                if layer.identity.cache_role is CacheRole.MLA
            ],
        )
        return SharedMlaRankPages.observe_enrolled(cache, self.descriptor, attempt)

    def require_owned_layout(
        self, cache: ColdRestoreLayout, layer_groups: Mapping[str, int]
    ) -> None:
        """Prove these views are the actual initialized local cache references."""
        if (
            type(cache) is not ColdRestoreLayout
            or cache.rank != self.rank
            or set(cache.layers) != set(self._layers)
            or set(layer_groups) != set(self._layers)
            or any(
                layer.identity != self._layers[name].identity
                or layer.tensor is not self._layers[name].tensor
                or layer_groups[name] != self._layers[name].group_id
                for name, layer in cache.layers.items()
            )
        ):
            raise ValueError(
                "GLM return layout is outside its initialized native views"
            )

    def verify_k0_prefill_export(
        self, cache: ColdRestoreLayout, state_blocks: tuple[tuple[str, int], ...]
    ) -> ColdRestoreLayout:
        """Bind the observed P block to canonical slot zero without copying it."""
        if (
            cache.rank != self.rank
            or set(cache.layers) != set(self._layers)
            or not self._tail_positions
            or cache.layout.committed_kda_slot != 0
        ):
            raise ValueError("GLM K0 export changed its native initialized rank cache")
        selected = dict(state_blocks)
        kda, source_layers = set(), []
        boundary = cache.layout.target_valid_tokens
        positions = range(boundary - boundary % _POOL_SIZE, boundary)
        for role, layer in cache.layers.items():
            identity = layer.identity
            if self._layers[role].tensor is not layer.tensor:
                raise ValueError("GLM K0 export changed native buffer ownership")
            if identity.cache_role in (CacheRole.KDA_CONV, CacheRole.KDA_RECURRENT):
                name = identity.layer_name
                kda.add(name)
                physical = selected.get(name, 0)
                if (
                    physical not in layer.block_ids
                    or not 0 < physical < layer.tensor.shape[0]
                ):
                    raise ValueError("GLM K0 export lost the selected native KDA block")
                layer = replace(layer, block_ids=(physical,))
            elif identity.cache_role is CacheRole.KPOOL_TAIL:
                stamps = self._tail_positions[identity.layer_name]
                block = layer.block_ids[0]
                if not 0 < block < stamps.shape[0]:
                    raise ValueError("GLM K0 P has no retained KPool ring")
                written = stamps[block].cpu().tolist()
                if any(written[pos % _RING_SIZE] != pos for pos in positions):
                    raise ValueError(
                        "GLM P KPool rows were not written at the K0 frontier"
                    )
            source_layers.append(layer)
        if selected.keys() != kda or len(selected) != len(state_blocks):
            raise ValueError("GLM K0 state proof differs from its native layer set")
        return ColdRestoreLayout(cache.layout, cache.rank, source_layers)

    def seed_imported_tail(
        self, cache: ColdRestoreLayout, ticket: DecodeReceiveTicket
    ) -> None:
        """Run after D native scatter and readback, before its loaded receipt."""
        if not self._tail_positions:
            raise RuntimeError("GLM D has no actual KPool writer witness")
        expected, _ = self.for_import(cache.layout, ticket.block_ids)
        if (
            type(cache) is not ColdRestoreLayout
            or cache.rank != self.rank
            or (ticket.request_id, ticket.epoch, ticket.imported_tokens)
            != (
                cache.layout.request_id,
                cache.layout.epoch,
                cache.layout.target_valid_tokens,
            )
            or (ticket.engine_id, ticket.boot_id)
            != (self.descriptor.authority_engine_id, self.descriptor.authority_boot_id)
            or set(cache.layers) != set(expected.layers)
            or any(
                layer.tensor is not expected.layers[role].tensor
                or layer.block_ids != expected.layers[role].block_ids
                for role, layer in cache.layers.items()
            )
        ):
            raise ValueError(
                "GLM imported KPool bytes belong to another native attempt"
            )
        self._prepare_tail_seed(cache)()

    def prepare_prefill_return_tail(
        self, cache: ColdRestoreLayout, layer_groups: Mapping[str, int]
    ) -> Callable[[], None]:
        """Bind the P writer slots before any return bytes are received."""
        self.require_owned_layout(cache, layer_groups)
        return self._prepare_tail_seed(cache)

    def _prepare_tail_seed(self, cache: ColdRestoreLayout) -> Callable[[], None]:
        layout = cache.layout
        boundary = layout.target_valid_tokens
        positions = range(boundary - boundary % _POOL_SIZE, boundary)
        targets = []
        for layer in cache.layers.values():
            if layer.identity.cache_role is not CacheRole.KPOOL_TAIL:
                continue
            block = layer.block_ids[0]
            name = layer.identity.layer_name
            witness = self._tail_positions.get(name)
            if witness is None or not 0 < block < witness.shape[0]:
                raise ValueError("GLM import has no live KPool tail block")
            targets.append(((name, block), witness[block]))
        attempt = (layout.request_id, layout.epoch, boundary)

        def seed() -> None:
            for key, row in targets:
                row.fill_(-1)
                for position in positions:
                    row[position % _RING_SIZE] = position
                self._tail_attempts[key] = attempt

        return seed

    def terminal_source(
        self,
        lease: TerminalPinLease,
        observed: NativeTerminalRows,
        representation_sha256: str,
    ) -> FencedTerminalSource:
        """Resolve exactly the saved frame's pin into the registered native rank."""
        identity = lease.identity
        if (
            (identity.source_engine_id, identity.source_boot_id)
            != (self.descriptor.authority_engine_id, self.descriptor.authority_boot_id)
            or not _is_digest(representation_sha256)
            or not self._tail_positions
        ):
            raise ValueError("GLM terminal source has no active worker authority")
        pinned = {group.group_id: group for group in lease.groups}
        if len(pinned) != len(lease.groups):
            raise ValueError("GLM native pin duplicated an authoritative group")

        boundary = lease.requested_common_tokens
        layout = Glm5NextPDLayout(
            identity.request_id,
            identity.model_fingerprint,
            identity.config_fingerprint,
            0,
            identity.epoch,
            boundary,
            boundary,
            _accepted_row(lease, observed),
            "decode_ready",
        )
        layers, tails = [], []
        expected_tail = tuple(range(boundary - boundary % _POOL_SIZE, boundary))
        for role, layer in self._layers.items():
            group = pinned.get(layer.group_id)
            cache_role = layer.identity.cache_role
            location = "host" if cache_role is CacheRole.MLA else "device"
            if (
                group is None
                or group.location != location
                or layer.identity.layer_name not in group.layer_names
            ):
                raise ValueError("GLM terminal pin changed the native layer authority")
            layers.append(RestoreLayer(layer.identity, layer.tensor, group.block_ids))
            if cache_role is CacheRole.KPOOL_TAIL:
                witness = self._tail_positions[layer.identity.layer_name]
                block = group.block_ids[0]
                if not 0 < block < witness.shape[0]:
                    raise ValueError("GLM terminal pin has no live KPool tail block")
                attempt = self._tail_attempts.get((layer.identity.layer_name, block))
                if (
                    attempt is None
                    or attempt[:2] != (identity.request_id, identity.epoch)
                    or attempt[2] != observed.target["frontier"]["imported_tokens"]
                    or not 0 < attempt[2] < boundary
                ):
                    raise ValueError(
                        "GLM terminal KPool ring has no verified current import"
                    )
                stamps = witness[block].cpu().tolist()
                if any(
                    stamps[position % _RING_SIZE] != position
                    for position in expected_tail
                ):
                    raise ValueError(
                        "GLM terminal KPool rows were not written at accepted positions"
                    )
                tails.append((role, expected_tail))
        return FencedTerminalSource(
            ColdRestoreLayout(layout, self.rank, layers),
            self.descriptor.fingerprint(),
            representation_sha256,
            tuple(sorted(tails)),
        )
