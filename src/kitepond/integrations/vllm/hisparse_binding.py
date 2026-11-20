# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind the P view and D authorization to native HiSparse scheduler owners."""

from inspect import signature

from vllm.v1.hisparse.coordinator import get_hisparse_coordinator

from .borrowed_host import BorrowedHostMetadata
from .host_authority import NativeHostAuthority, NativeHostGrant


def bind_decode_host(manager, *, engine_id: str, boot_id: str) -> NativeHostAuthority:
    """Use the exact native host pool already bound to this D manager."""
    coordinator = get_hisparse_coordinator(manager)
    pool = coordinator.get_host_block_pool()
    if pool is None or pool is manager.block_pool or coordinator.host_manager is None:
        raise ValueError("Decoder has no independent native HiSparse host pool")
    return NativeHostAuthority(
        pool,
        engine_id=engine_id,
        boot_id=boot_id,
        block_tokens=coordinator.host_manager.block_size,
    )


def bind_prefill_host(manager, *, engine_id: str, boot_id: str) -> BorrowedHostMetadata:
    """Select P's metadata pool before any native HiSparse admission."""
    if "host_pool_provider" not in signature(get_hisparse_coordinator).parameters:
        raise RuntimeError("vLLM has no scheduler HiSparse host pool provider")

    def provider(num_blocks, device_event_owner):
        return BorrowedHostMetadata(
            engine_id=engine_id,
            boot_id=boot_id,
            capacity=num_blocks,
            hash_block_size=device_event_owner.hash_block_size,
        )

    coordinator = get_hisparse_coordinator(manager, host_pool_provider=provider)
    borrowed = coordinator.get_host_block_pool()
    if not isinstance(borrowed, BorrowedHostMetadata):
        raise RuntimeError("Native P host manager did not bind the borrowed authority")
    return borrowed


def _native_attacher(manager, pool):
    coordinator = get_hisparse_coordinator(manager)
    if coordinator.get_host_block_pool() is not pool:
        raise ValueError("Native request and grant use different HiSparse allocators")
    attach = getattr(coordinator, "attach_external_host_blocks", None)
    if not callable(attach):
        raise RuntimeError("vLLM has no native HiSparse request attachment")
    return attach


def attach_prefill_grant(manager, view: BorrowedHostMetadata, grant: NativeHostGrant):
    """Stage P metadata before schedule; physical references remain on D."""
    attach = _native_attacher(manager, view)
    blocks = view.install(grant)
    try:
        attach(grant.request_id, blocks, allow_growth=False)
    except BaseException:
        view.free_blocks(blocks)
        view.discard(grant)
        raise


def attach_decode_grant(
    manager, authority: NativeHostAuthority, grant: NativeHostGrant
):
    """Transfer settled grant refs to native D without publishing readiness."""
    attach = _native_attacher(manager, authority.pool)

    def attach_with_growth(request_id, blocks):
        attach(request_id, blocks, allow_growth=True)

    authority.adopt(grant, attach=attach_with_growth)
