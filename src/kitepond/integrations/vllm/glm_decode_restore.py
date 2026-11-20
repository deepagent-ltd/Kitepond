# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish only after actual, verified rank-local GLM restore writes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
from kitepond.models.glm5next.cold_restore import (
    ColdRestoreLayout,
    verify_cold_import_readback,
)
from kitepond.models.glm5next.import_recipe import (
    ColdBinding,
    compose_import_recipe,
    copy_import_window,
)
from kitepond.models.glm5next.layout import CacheRole
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion

from .decode_receive import DecodeReceiveMailbox, DecodeReceiveTicket

_REQUIRED_ROLES = {
    CacheRole.MLA,
    CacheRole.KDA_CONV,
    CacheRole.KDA_RECURRENT,
    CacheRole.INDEXER,
    CacheRole.KPOOL_TAIL,
}


def validate_native_destination(
    mailbox: DecodeReceiveMailbox,
    ticket: DecodeReceiveTicket,
    cache: ColdRestoreLayout,
    layer_groups: Mapping[str, int],
) -> None:
    if (
        type(cache) is not ColdRestoreLayout
        or (cache.rank, cache.layout.request_id, cache.layout.epoch)
        != (mailbox.rank, ticket.request_id, ticket.epoch)
        or cache.layout.target_valid_tokens != ticket.imported_tokens
        or set(cache.layers) != set(layer_groups)
        or {layer.identity.cache_role for layer in cache.layers.values()}
        != _REQUIRED_ROLES
        or any(
            type(group) is not int
            or not 0 <= group < len(ticket.block_ids)
            or cache.layers[name].block_ids != ticket.block_ids[group]
            for name, group in layer_groups.items()
        )
    ):
        raise ValueError("GLM D import differs from the admitted rank/layer views")


def apply_verified_native_host_import(
    mailbox: DecodeReceiveMailbox,
    ticket: DecodeReceiveTicket,
    cache: ColdRestoreLayout,
    layer_groups: Mapping[str, int],
    region: NativeHiSparseMemfdRegion,
    source_bindings: Sequence[ColdBinding],
    slab: torch.Tensor,
    *,
    source_region_sha256: str,
    stream: torch.cuda.Stream | None = None,
    cow_event: torch.cuda.Event | None = None,
    seed_imported_tail: Callable[[ColdRestoreLayout, DecodeReceiveTicket], None]
    | None = None,
) -> None:
    """Scatter a verified full-prefix rank manifest into admitted native views.

    The caller holds a verified source region and manifest through this call.
    This path reads the shared host region directly and has no remote DMA lease.
    Partial writes become abort-settled only after the local receive fence drains.
    """
    validate_native_destination(mailbox, ticket, cache, layer_groups)
    if ticket.prefix_tokens:
        raise ValueError("Full-prefix restore cannot rewrite a shared native prefix")
    if ticket.copy_tail is not None and cow_event is None:
        raise ValueError("GLM restore requires the native COW fence")
    source = region.descriptor
    if (
        source.source_rank,
        source.authority_engine_id,
        source.authority_boot_id,
        source.fingerprint(),
    ) != (mailbox.rank, ticket.engine_id, ticket.boot_id, source_region_sha256):
        raise ValueError("GLM restore source differs from its verified native region")
    cuda_devices = {
        layer.tensor.device
        for layer in cache.layers.values()
        if layer.tensor.device.type == "cuda"
    }
    if any(
        layer.tensor.device.type not in {"cpu", "cuda"}
        for layer in cache.layers.values()
    ) or (
        (stream is None and (cuda_devices or cow_event is not None))
        or (stream is not None and cuda_devices != {stream.device})
        or (cow_event is not None and not isinstance(cow_event, torch.cuda.Event))
    ):
        raise ValueError("GLM restore requires the actual native receive stream")

    recipe = compose_import_recipe(cache.manifest(), source_bindings)

    def scatter() -> None:
        for bindings in recipe:
            payload = copy_import_window(
                region, bindings, slab, region_sha256=source_region_sha256
            )
            cache.scatter_window(tuple(binding.part for binding in bindings), payload)
        readback = verify_cold_import_readback(
            cache, recipe, region, slab, region_sha256=source_region_sha256
        )
        if readback["status"] != "MATCH":
            raise ValueError("GLM D native destination readback disagrees")
        if seed_imported_tail is not None:
            seed_imported_tail(cache, ticket)

    mailbox._start_import(ticket)
    fence = None
    try:
        if stream is None:
            scatter()
        else:
            fence = torch.cuda.Event()
            with torch.cuda.stream(stream):
                if cow_event is not None:
                    stream.wait_event(cow_event)
                try:
                    scatter()
                finally:
                    fence.record(stream)
            fence.synchronize()
    except BaseException:
        # A failed or unrecorded CUDA fence must never release native refs.
        if stream is None:
            mailbox._post_fenced(ticket, "abort_settled")
        elif fence is not None and fence.device is not None:
            fence.synchronize()
            mailbox._post_fenced(ticket, "abort_settled")
        raise
    mailbox._post_fenced(ticket, "loaded")
