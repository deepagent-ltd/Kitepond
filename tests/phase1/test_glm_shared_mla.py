# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU proof for rank-local MLA aliases and the full-wire fallback."""

from __future__ import annotations

import os
from contextlib import ExitStack, closing
from dataclasses import asdict, replace
from time import monotonic_ns
from types import SimpleNamespace

import pytest
import torch
from kitepond.core.readiness import ReadyReceipt
from kitepond.integrations.mooncake.window import plan_windows
from kitepond.integrations.vllm.glm_cold_prefill_admission import (
    ColdPrefillAdmission,
    ColdPrefillControl,
    ColdPrefillPrompt,
)
from kitepond.integrations.vllm.host_authority import NativeHostAuthority
from kitepond.models.glm5next.cold_restore import ColdRestoreLayout
from kitepond.models.glm5next.import_recipe import part_identity
from kitepond.models.glm5next.layout import CacheRole
from kitepond.models.glm5next.live_windows import GlmRankWindowPlan
from kitepond.models.glm5next.shared_mla import (
    SharedMlaRankPages,
    confirm_shared_mla_elision,
)
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion
from test_glm_cold_restore import _cache

from vllm.v1.core.block_pool import BlockPool

_PAGE_BYTES = 256 * 656
_REGION_BYTES = 4 * _PAGE_BYTES


def _views(rank, backing, grant, fill):
    cache = _cache(rank=rank)
    layout = replace(cache.layout, epoch=grant.epoch, request_id=grant.request_id)
    layers = []
    for layer in cache.layers.values():
        if layer.identity.cache_role is CacheRole.MLA:
            view = torch.frombuffer(backing.mapping, dtype=torch.uint8).view(
                4, 1, 256, 656
            )
            layers.append(replace(layer, tensor=view, block_ids=grant.writable_pages))
        else:
            layer.tensor.view(torch.uint8).fill_(fill)
            layers.append(layer)
    return ColdRestoreLayout(layout, rank, layers)


def _assert_scheduler_choice(
    owner, control, ingress, offer, receipts, ready_mac, candidate
):
    from kitepond.integrations.vllm.glm_cold_prefill_scheduler import (
        GlmColdDecodeAdmissionMixin,
    )

    class NativeSchedulerFixture:
        def get_owner_worker_control_registration(self):
            return None

        def add_request(self, request):
            self.requests[request.request_id] = request

    class TestScheduler(GlmColdDecodeAdmissionMixin, NativeSchedulerFixture):
        pass

    scheduler = object.__new__(TestScheduler)
    scheduler.requests = {}
    scheduler._cold_mla_preflight = True
    scheduler._cold_mla_decisions = {}
    scheduler._cold_decode_config = SimpleNamespace(control=control)
    scheduler._cold_decode_authority = owner
    scheduler._cold_decode_admission = ingress
    scheduler._cold_decode_offered = {offer.prompt.request_id: offer}
    scheduler._cold_decode_staged = {}
    scheduler._cold_decode_verified = set()
    calls = []
    accepted = None

    def rpc(method, timeout, args):
        assert method == "kitepond_prepare_shared_mla"
        assert args == (offer,) and timeout == 4
        assert owner.ready_for_adoption(offer.grant)
        assert not scheduler.requests
        calls.append(method)
        return [candidate.get(rank) for rank in range(4)]

    def stage(request, grant, *, imported_tokens, mla_rank_digests):
        nonlocal accepted
        assert owner.ready_for_adoption(offer.grant) and not scheduler.requests
        assert grant is offer.grant and imported_tokens == offer.imported_tokens
        assert request.request_id == offer.prompt.request_id
        accepted = mla_rank_digests

    scheduler._cold_decode_owner = SimpleNamespace(stage_host_grant=stage)
    commands, bind = scheduler.get_owner_worker_control_registration()
    assert commands == ("kitepond_prepare_shared_mla",)
    bind(rpc)
    payload = {
        "attempt": asdict(offer.attempt),
        "receipts": [asdict(receipt) for receipt in receipts],
        "mac": ready_mac,
    }
    expected = "same_pages" if len(candidate) == 4 else "full"
    assert scheduler._cold_decode_handler("confirm_ready", payload) == {
        "status": "ready",
        "mla_transport": expected,
    }
    # A replay must not obtain a new decision even if a rank has disappeared.
    candidate.clear()
    assert scheduler._cold_decode_handler("confirm_ready", payload) == {
        "status": "ready",
        "mla_transport": expected,
    }
    assert calls == ["kitepond_prepare_shared_mla"]
    request = SimpleNamespace(
        request_id=offer.prompt.request_id,
        resumable=False,
        prompt_token_ids=list(range(offer.prompt.tokens)),
        num_prompt_tokens=offer.prompt.tokens,
        kv_transfer_params={
            "do_remote_prefill": True,
            "transfer_id": offer.prompt.transfer_id,
        },
    )
    scheduler.add_request(request)
    assert scheduler.requests == {request.request_id: request}
    assert bool(accepted) == (expected == "same_pages")


def _trial(owner, control, ingress, offer, regions, borrowed, invalid):
    sources, targets, producer, consumer, receipts = {}, {}, {}, {}, []
    grant = offer.grant
    for rank, region in enumerate(regions):
        src = sources[rank] = _views(rank, borrowed[rank], grant, fill=40)
        dst = targets[rank] = _views(rank, region, grant, fill=99)
        for name, layer in src.layers.items():
            if layer.identity.cache_role is CacheRole.MLA:
                assert layer.tensor.data_ptr() != dst.layers[name].tensor.data_ptr()
                layer.tensor.fill_(rank + 16)
                assert dst.layers[name].tensor.flatten()[0].item() == rank + 16
        descriptor = region.descriptor
        producer[rank] = SharedMlaRankPages.observe_enrolled(
            src, borrowed[rank].descriptor, offer.attempt
        )
        consumer[rank] = SharedMlaRankPages.observe_enrolled(
            dst, descriptor, offer.attempt
        )
        receipts.append(
            ReadyReceipt(
                offer.attempt,
                rank,
                descriptor.fingerprint(),
                offer.imported_tokens,
                offer.imported_tokens,
                monotonic_ns(),
            )
        )

    receipts = tuple(receipts)
    ready_mac = control.ready_mac(offer, receipts)
    if invalid == "cancelled":
        assert owner.cancel(grant) is False
        with pytest.raises(ValueError, match="active signed READY lease"):
            confirm_shared_mla_elision(
                control, owner, offer, receipts, ready_mac, producer, consumer
            )
        return
    ingress.verify_ready(offer, receipts, ready_mac)
    if invalid == "missing":
        producer.pop(3)
    if invalid == "pages":
        view = consumer[3]
        changed = replace(
            view.layers[0], block_ids=tuple(reversed(grant.writable_pages))
        )
        consumer[3] = replace(view, layers=(changed,))
    if invalid == "offset":
        view = consumer[3]
        changed = replace(view.layers[0], storage_offset=32)
        consumer[3] = replace(view, layers=(changed,))
    if invalid == "different_config":
        consumer[3] = replace(consumer[3], config_fingerprint="e" * 64)
    if invalid == "different_inode":
        view = consumer[3]
        with closing(
            NativeHiSparseMemfdRegion.create(
                _REGION_BYTES,
                authority_engine_id=grant.engine_id,
                authority_boot_id=grant.boot_id,
                source_rank=3,
                layout_sha256="a" * 64,
            )
        ) as extra:
            consumer[3] = replace(view, descriptor=extra.descriptor)
    if invalid == "unsigned":
        with pytest.raises(ValueError, match="active signed READY lease"):
            confirm_shared_mla_elision(
                control, owner, offer, receipts, "b" * 64, producer, consumer
            )
        return

    proof = confirm_shared_mla_elision(
        control, owner, offer, receipts, ready_mac, producer, consumer
    )
    assert bool(proof) == (invalid is None)
    if invalid in (None, "missing"):
        _assert_scheduler_choice(
            owner, control, ingress, offer, receipts, ready_mac, producer.copy()
        )
    for rank in range(4):
        region = regions[rank].descriptor
        full_source = GlmRankWindowPlan(sources[rank], 0)
        full_target = GlmRankWindowPlan(targets[rank], 0)
        source = full_source.for_forward_shared_mla(proof, region, offer.attempt)
        target = full_target.for_forward_shared_mla(proof, region, offer.attempt)
        assert source.manifest() == target.manifest()
        if invalid is not None:
            assert source is full_source and target is full_target
            continue

        with pytest.raises(ValueError, match="rank changed"):
            full_source.for_forward_shared_mla(
                proof, region, replace(offer.attempt, transfer_id="replayed")
            )
        retained = source.manifest()
        for record in retained:
            assert part_identity(record["role"]).cache_role is not CacheRole.MLA
        omitted_bytes = sum(
            record["nbytes"]
            for record in full_source.manifest()
            if part_identity(record["role"]).cache_role is CacheRole.MLA
        )
        assert omitted_bytes > 0
        assert (
            sum(row["nbytes"] for row in full_source.manifest())
            - sum(row["nbytes"] for row in retained)
        ) == omitted_bytes
        for window in plan_windows(retained):
            payload = torch.empty(window.nbytes, dtype=torch.uint8)
            assert source.pack(window, payload) == window.nbytes
            assert target.scatter(window, payload) == window.nbytes
            target.verify_readback(window, payload)
        # Check that state outside the optional wire path is still byte-exact.
        for window in plan_windows(full_source.manifest()):
            payload = torch.empty(window.nbytes, dtype=torch.uint8)
            assert full_source.pack(window, payload) == window.nbytes
            full_target.verify_readback(window, payload)


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "missing",
        "pages",
        "offset",
        "different_inode",
        "different_config",
        "unsigned",
        "cancelled",
    ],
)
def test_elision_requires_all_four_physical_pages_then_keeps_exact_other_state(invalid):
    pool = BlockPool(num_gpu_blocks=4, enable_caching=False, hash_block_size=1024)
    owner = NativeHostAuthority(
        pool, engine_id="D", boot_id="D-boot", block_tokens=1024
    )
    control = ColdPrefillControl(
        d_engine_id="D", d_boot_id="D-boot", p_boot_id="P-boot", key=b"a" * 32
    )
    ingress = ColdPrefillAdmission(
        owner, control, context_limit=8192, max_pending=1, arm_timeout_seconds=30
    )
    offer = ingress.offer(
        ColdPrefillPrompt.from_tokens("native-request", "transfer", 19, range(1060))
    )
    with ExitStack() as cleanup:
        regions, borrowed = [], []
        for rank in range(4):
            identity = {
                "authority_engine_id": "D",
                "authority_boot_id": "D-boot",
                "source_rank": rank,
                "layout_sha256": "a" * 64,
            }
            region = cleanup.enter_context(
                closing(NativeHiSparseMemfdRegion.create(_REGION_BYTES, **identity))
            )
            regions.append(region)
            borrowed.append(
                cleanup.enter_context(
                    closing(
                        NativeHiSparseMemfdRegion.join_received_fd(
                            region.descriptor, os.dup(region.fd), **identity
                        )
                    )
                )
            )
        _trial(owner, control, ingress, offer, regions, borrowed, invalid)
    assert owner.retains(offer.grant)
