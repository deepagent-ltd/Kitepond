# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import time
from dataclasses import replace

import pytest
import torch
from kitepond.integrations.vllm.decode_receive import DecodeReceiveTicket
from kitepond.integrations.vllm.glm_native_views import (
    GlmNativeStateViews,
    _as_row_cache,
    _NativeLayer,
)
from kitepond.integrations.vllm.terminal_checkpoint import (
    TerminalCacheGroup,
    TerminalPinLease,
    TerminalTransferIdentity,
)
from kitepond.models.glm5next.cold_restore import ColdRestoreLayout, RestoreLayer
from kitepond.models.glm5next.layout import CacheRole
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion
from test_glm_terminal_manifest import BOUNDARY, Fence, fenced_source, make_runner


@pytest.fixture
def source_attempt():
    rank, epoch = 0, 7
    tokens = list(range(1, BOUNDARY + 1)) + [33]
    native_ident = TerminalTransferIdentity(
        "request",
        "D-export",
        epoch,
        "D",
        "boot",
        "a" * 64,
        "b" * 64,
        time.monotonic_ns() + 10_000_000_000,
    )
    lease = TerminalPinLease(
        native_ident,
        1,
        BOUNDARY,
        BOUNDARY + 1,
        hashlib.sha256(
            json.dumps(tokens[:BOUNDARY], separators=(",", ":")).encode()
        ).hexdigest(),
        33,
        (
            TerminalCacheGroup(0, "host", ("mla.target", "mla.draft"), (1,)),
            TerminalCacheGroup(1, "device", ("idx.target", "idx.draft"), (2,)),
            TerminalCacheGroup(2, "device", ("tail.target", "tail.draft"), (3,)),
            TerminalCacheGroup(3, "device", ("kda",), (4, 5, 6, 7)),
        ),
    )
    region = NativeHiSparseMemfdRegion.create(
        4096,
        authority_engine_id="D",
        authority_boot_id="boot",
        source_rank=rank,
        layout_sha256="e" * 64,
    )
    try:
        source = fenced_source(lease, rank, region.descriptor.fingerprint())
        owner = GlmNativeStateViews.__new__(GlmNativeStateViews)
        owner.rank = rank
        owner.descriptor = region.descriptor
        owner._groups = len(lease.groups)
        group_lookup = {
            name: group.group_id for group in lease.groups for name in group.layer_names
        }
        owner._layers = {
            role: _NativeLayer(
                group_lookup[layer.identity.layer_name], layer.identity, layer.tensor
            )
            for role, layer in source.layout.layers.items()
        }
        owner._tail_positions = {
            layer.identity.layer_name: torch.full(
                (layer.tensor.shape[0], 8), -1, dtype=torch.int32
            )
            for layer in source.layout.layers.values()
            if layer.identity.cache_role is CacheRole.KPOOL_TAIL
        }
        owner._tail_attempts = {}
        import_layout = replace(
            source.layout.layout,
            target_valid_tokens=63,
            draft_valid_tokens=63,
            committed_kda_slot=0,
        )
        ticket = DecodeReceiveTicket(
            "request",
            "P-import",
            "D",
            "boot",
            epoch,
            "c" * 64,
            63,
            tuple(group.block_ids for group in lease.groups),
            0,
            None,
        )
        import_cache, group_ids = owner.for_import(import_layout, ticket.block_ids)
        assert all(
            layer.tensor is source.layout.layers[role].tensor
            and group_ids[role] == group_lookup[layer.identity.layer_name]
            for role, layer in import_cache.layers.items()
        )
        runner = make_runner(lease, tokens, Fence(), lease.groups[3].block_ids)
        runner.frontier.state[0, 4:6] = torch.tensor([2, 63])
        observed = runner.terminal_rows("request", lease, 63)
        yield owner, lease, observed, import_cache, ticket
    finally:
        region.close()


def test_terminal_source_needs_the_current_import_and_every_layer_write(source_attempt):
    views, lease, observed, cache, ticket = source_attempt
    repr_digest = "d" * 64
    for witness in views._tail_positions.values():
        # A same-position stamp left by an older owner is insufficient.
        witness[3, 0] = 64
        witness[3, 1] = 65
    with pytest.raises(ValueError, match="verified current import"):
        views.terminal_source(lease, observed, repr_digest)
    with pytest.raises(ValueError, match="another native attempt"):
        views.seed_imported_tail(cache, replace(ticket, engine_id="foreign"))
    assert not views._tail_attempts
    first, *rest = cache.layers.values()
    foreign = ColdRestoreLayout(
        cache.layout,
        cache.rank,
        (RestoreLayer(first.identity, first.tensor.clone(), first.block_ids), *rest),
    )
    with pytest.raises(ValueError, match="another native attempt"):
        views.seed_imported_tail(foreign, ticket)

    views.seed_imported_tail(cache, ticket)
    assert all(
        witness[3].tolist() == [-1, -1, -1, -1, 60, 61, 62, -1]
        for witness in views._tail_positions.values()
    )
    with pytest.raises(ValueError, match="were not written"):
        views.terminal_source(lease, observed, repr_digest)
    views._tail_positions["tail.target"][3, 0] = 64
    views._tail_positions["tail.target"][3, 1] = 65
    with pytest.raises(ValueError, match="were not written"):
        views.terminal_source(lease, observed, repr_digest)
    views._tail_positions["tail.draft"][3, 0] = 64
    views._tail_positions["tail.draft"][3, 1] = 65
    actual = views.terminal_source(lease, observed, repr_digest)
    assert actual.authority_sha256 == views.descriptor.fingerprint()
    assert len(actual.tail_positions) == 2
    assert all(positions == (64, 65) for _, positions in actual.tail_positions)
    assert actual.layout.layout.committed_kda_slot == 1
    conv = next(
        part for part in actual.layout.manifest() if part.role.endswith(":kda_conv")
    )
    native = actual.layout.layers[conv.role]
    live = native.tensor[native.block_ids[0]]
    for row in range(6):
        live[row].fill_(row + 1)
    wire = actual.read_chunk(conv).view(torch.bfloat16).view(conv.shape)
    assert wire[:, 0].tolist() == [2, 3, 4, 0, 0, 0]
    assert live[:, 0].tolist() == [1, 2, 3, 4, 5, 6]
    for role, layer in actual.layout.layers.items():
        assert layer.tensor is cache.layers[role].tensor
    with pytest.raises(ValueError, match="verified current import"):
        views.terminal_source(
            replace(lease, identity=replace(lease.identity, epoch=8)),
            observed,
            repr_digest,
        )


def test_glm_native_rows_keep_the_exact_physical_backing():
    contiguous = torch.zeros((2 * 256, 656), dtype=torch.uint8)
    grouped = _as_row_cache(contiguous, 256, 656)
    assert grouped.shape == (2, 1, 256, 656)
    assert (
        grouped.untyped_storage().data_ptr() == contiguous.untyped_storage().data_ptr()
    )
    with pytest.raises(ValueError, match="width or dtype"):
        _as_row_cache(contiguous, 256, 528)


def test_enroll_native_layer_views_and_bind_kpool_ring_before_graphs():
    from types import SimpleNamespace

    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm.models.glm5next.common.attention import Glm5NextTailCache
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec,
        KVCacheConfig,
        KVCacheGroupRole,
        KVCacheGroupSpec,
        MambaSpec,
        MLAAttentionSpec,
    )

    num_blocks = 2
    host_rows = num_blocks * 256
    region = NativeHiSparseMemfdRegion.create(
        2 * host_rows * 656,
        authority_engine_id="D",
        authority_boot_id="boot",
        source_rank=0,
        layout_sha256="a" * 64,
    )
    try:
        memory = region.tensor().view(torch.int8)
        memory._kitepond_glm_host_descriptor = region.descriptor
        host_cache = memory.view(torch.uint8).view(2, host_rows, 656)
        host_layers = {}
        for index, role in enumerate(("target", "draft")):
            runtime = SimpleNamespace(
                registered_host_pool=memory, host_cache=host_cache[index]
            )
            host_layers[f"mla.{role}"] = SimpleNamespace(
                is_draft_layer=role == "draft",
                hisparse_cache=SimpleNamespace(runtime=runtime),
            )

        mla = MLAAttentionSpec(
            block_size=256, num_kv_heads=1, head_size=656, dtype=torch.uint8
        )
        indexer = replace(mla, head_size=132, tokens_per_state=4)
        tail = CircularBufferSpec(
            block_size=8,
            num_kv_heads=2,
            head_size=128,
            head_size_v=0,
            dtype=torch.bfloat16,
            dcp_sharded=False,
        )
        kda = MambaSpec(
            block_size=256,
            shapes=((6, 6144), (16, 128, 128)),
            dtypes=(torch.bfloat16, torch.float32),
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        )
        groups = [
            KVCacheGroupSpec(
                list(host_layers),
                mla,
                host_resident=True,
                role=KVCacheGroupRole.HISPARSE_SOURCE,
            ),
            KVCacheGroupSpec(
                ["idx.target", "idx.draft"],
                indexer,
                role=KVCacheGroupRole.HISPARSE_INDEXER,
            ),
            KVCacheGroupSpec(["tail.target", "tail.draft"], tail),
            KVCacheGroupSpec(["kda"], kda),
        ]
        device_layers = {}
        mode = FakeTensorMode()
        with mode:
            for role in ("target", "draft"):
                device_layers[f"idx.{role}"] = SimpleNamespace(
                    is_draft_layer=role == "draft",
                    kv_cache=torch.empty(
                        (num_blocks, 64, 132), dtype=torch.uint8, device="cuda:0"
                    ),
                )
                native_tail = Glm5NextTailCache.__new__(Glm5NextTailCache)
                torch.nn.Module.__init__(native_tail)
                native_tail.is_draft_layer = role == "draft"
                native_tail.kpool_write_positions = None
                native_tail.kv_cache = torch.empty(
                    (num_blocks, 2, 8, 128), dtype=torch.bfloat16, device="cuda:0"
                )
                device_layers[f"tail.{role}"] = native_tail
            device_layers["kda"] = SimpleNamespace(
                is_draft_layer=False,
                kv_cache=tuple(
                    torch.empty((num_blocks, *shape), dtype=dtype, device="cuda:0")
                    for shape, dtype in zip(kda.shapes, kda.dtypes)
                ),
            )
        runner = SimpleNamespace(
            kv_cache_config=KVCacheConfig(
                num_blocks, [], groups, hisparse_host_num_blocks=num_blocks
            ),
            block_tables=SimpleNamespace(cp_rank=0, cp_size=4, cp_interleave=4),
            compilation_config=SimpleNamespace(
                static_forward_context=host_layers | device_layers
            ),
        )
        views = GlmNativeStateViews(runner, 0)
        assert views.descriptor is region.descriptor
        assert len(views._layers) == 8
        with mode:
            views.bind_tail_writers()
        for name in ("tail.target", "tail.draft"):
            assert (
                views._tail_positions[name] is device_layers[name].kpool_write_positions
            )
            assert views._tail_positions[name].shape == (num_blocks, 8)
        with pytest.raises(RuntimeError, match="unbound KPool"):
            views.bind_tail_writers()
    finally:
        region.close()
