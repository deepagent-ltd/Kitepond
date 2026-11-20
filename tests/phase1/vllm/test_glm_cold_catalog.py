# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import multiprocessing
import os
import sys
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from hashlib import sha256
from pathlib import Path

import pytest
from kitepond.backends.fsl2 import canonical, digest, rank_manifest_name
from kitepond.core.contracts import Wait
from kitepond.integrations.vllm.glm_cold_catalog import _DOMAIN, GlmColdCatalog
from kitepond.integrations.vllm.glm_cold_runtime import ColdDirectoryScope
from kitepond.integrations.vllm.native_glm_cold_directory import NativeGlmColdDirectory
from kitepond.integrations.vllm.terminal_checkpoint import StockTerminalCheckpoints
from test_glm_cold_lifecycle import BUDGET, Owner, _scheduler, _store, _wait
from test_stock_terminal_checkpoint import capture

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

SCOPE = ColdDirectoryScope("scope", 42)
KEY = bytes(range(32))
OLD_BOOT = "retired-D-boot"
NEW_BOOT = "new-D-boot"


def _catalog(root, *, boot=NEW_BOOT, key=KEY, config="c" * 64):
    return GlmColdCatalog(
        root,
        engine_id="D",
        boot_id=boot,
        model_fingerprint="b" * 64,
        config_fingerprint=config,
        signing_key=key,
    )


class PublishingOwner(Owner):
    def acquire_snapshot(self, checkpoint, **kwargs):
        grant = super().acquire_snapshot(checkpoint, **kwargs)
        assert not isinstance(grant, Wait)
        return replace(
            grant,
            on_durable_publication=partial(
                self.catalog.publish,
                checkpoint,
                accepted_prefix_sha256=self.accepted_prefix_sha256,
            ),
        )


class RestoreProvider:
    def __init__(self, template):
        self.template = template
        self.region_fingerprints = tuple(
            region.descriptor.fingerprint() for region in template.regions
        )
        self.ram_full = True
        self.admitted = False
        self.payload_opens = 0

    def retain_terminal(self, entry):
        raise AssertionError("Reader has no live terminal state")

    def snapshot(self, checkpoint, source, **kwargs):
        raise AssertionError("Reader has no resident checkpoint to snapshot")

    def reserve_restore(self, checkpoint, *, max_inflight_bytes, verified_catalog):
        if self.ram_full:
            return Wait("RAM_ADMISSION_WAIT")
        self.template.checksums = [
            verified_catalog.checksums(rank) for rank in range(4)
        ]
        grant = self.template.reserve_restore(
            checkpoint, max_inflight_bytes=max_inflight_bytes
        )
        self.admitted = True
        return grant


def _write(root: Path, checkpoint=None, accepted_prefix_sha256=None):
    with ExitStack() as cleanup:
        owner = PublishingOwner(cleanup, checkpoint, boot_id=OLD_BOOT)
        owner.catalog = _catalog(root, boot=OLD_BOOT)
        owner.accepted_prefix_sha256 = accepted_prefix_sha256
        scheduler = _scheduler(root, owner)
        now = owner.last_use + scheduler.idle.idle_ns
        assert _store(scheduler, owner.checkpoint, now) == "COMMITTED"
        assert not owner.resident and owner.held is None
        return {
            "pid": os.getpid(),
            "old_sig": owner.catalog.reopen(SCOPE),
            "regions": [region.descriptor.fingerprint() for region in owner.regions],
        }


def _mutate(path: Path, change):
    previous = path.read_bytes()
    update = json.loads(previous)
    change(update)
    path.write_bytes(canonical(update))
    return previous


def _check_metadata_negatives(catalog: GlmColdCatalog, receipt):
    root = catalog.root
    path = next((root / "glm-catalog-v1").glob("*.json"))
    prior = _mutate(path, lambda doc: doc.update(source_boot_id="forged"))
    assert catalog.reopen(SCOPE) is None
    path.write_bytes(prior)
    # Authentication cannot turn old physical-row bytes into the new format.
    legacy = json.loads(prior)
    del legacy["mac"]
    legacy["layout"]["version"] = 1
    path.write_bytes(canonical(dict(legacy, mac=catalog._mac(_DOMAIN, legacy))))
    assert catalog._authorized(path) is not None
    assert catalog.reopen(SCOPE) is None
    # Store the legacy catalog at the actual legacy lookup key. A corrected
    # archive can coexist at the same session and generation.
    legacy_key = digest(
        {
            "engine_id": catalog.engine_id,
            "model_fingerprint": catalog.model_fingerprint,
            "config_fingerprint": catalog.config_fingerprint,
            "namespace": SCOPE.namespace,
            "generation": SCOPE.generation,
        }
    )
    old_path = path.with_name(legacy_key + ".json")
    assert old_path != path
    path.rename(old_path)
    assert catalog._authorized(old_path) is not None
    assert catalog.reopen(SCOPE) is None
    path.write_bytes(prior)
    assert catalog.reopen(SCOPE) is not None
    assert old_path.is_file()
    assert _catalog(root, key=bytes(reversed(KEY))).reopen(SCOPE) is None
    assert _catalog(root, config="f" * 64).reopen(SCOPE) is None
    assert catalog.reopen(replace(SCOPE, generation=99)) is None

    plans = receipt.checkpoint.plans
    name = rank_manifest_name(receipt.checkpoint.identity, plans[3])
    rank = root / "rank-3/manifests" / name
    missing = rank.with_suffix(".withheld")
    rank.rename(missing)
    assert catalog.reopen(SCOPE) is None
    missing.rename(rank)
    prior = _mutate(rank, lambda doc: doc["parts"][0].update(sha256="0" * 64))
    assert catalog.reopen(SCOPE) is None
    rank.write_bytes(prior)


def _replace_after_verification(root: Path, receipt):
    identity = receipt.checkpoint.identity
    plan = receipt.checkpoint.plans[0]
    rank = root / "rank-0/manifests" / rank_manifest_name(identity, plan)

    def rewrite_rank(doc):
        doc["parts"][0]["sha256"] = "0" * 64
        doc["digest"] = digest({k: v for k, v in doc.items() if k != "digest"})

    rank_prior = _mutate(rank, rewrite_rank)
    aggregate = next((root / "aggregates").glob("*.json"))

    def rewrite_aggregate(doc):
        doc["ranks"][0]["manifest_sha256"] = json.loads(rank.read_bytes())["digest"]
        doc["digest"] = digest({k: v for k, v in doc.items() if k != "digest"})

    manifest_prior = _mutate(aggregate, rewrite_aggregate)
    return (rank, rank_prior), (aggregate, manifest_prior)


def _read(root: Path, old_receipt):
    provider = None

    def audit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            name = os.fsdecode(args[0])
            if "/payloads/" in name:
                assert provider is not None and provider.admitted, (
                    "FSL2 opened payload before RAM admission"
                )
                provider.payload_opens += 1

    sys.addaudithook(audit)
    catalog = _catalog(root)
    assert not catalog.verify(old_receipt)
    receipt = catalog.reopen(SCOPE)
    assert receipt is not None and receipt.source_boot_id == OLD_BOOT
    assert receipt.target_boot_id == NEW_BOOT and catalog.verify(receipt)
    _check_metadata_negatives(catalog, receipt)
    checkpoint = receipt.checkpoint

    cfg_path = root.parent / "new-D-native"
    cfg_path.mkdir()
    setup = capture.__wrapped__(cfg_path)
    scheduler, *_ = setup()
    terminal = StockTerminalCheckpoints(
        scheduler.kv_cache_manager,
        4,
        source_engine_id="D",
        source_boot_id=NEW_BOOT,
        on_release=lambda _: None,
    )
    request = Request(
        request_id="new-native-request",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    with ExitStack() as cleanup:
        template = Owner(cleanup, checkpoint, boot_id=NEW_BOOT, seed_source=False)
        template.resident = False
        template.ram_full = False
        provider = RestoreProvider(template)

        def resolve(native):
            return SCOPE if native is request else replace(SCOPE, namespace="other")

        directory = NativeGlmColdDirectory(
            terminal,
            provider,
            resolve,
            max_entries=8,
            clock=lambda: 5,
            catalog=catalog,
        )
        with pytest.raises(ValueError, match="stale"):
            directory.reserve_restore(checkpoint, max_inflight_bytes=BUDGET)
        with pytest.raises(ValueError, match="another boot"):
            directory.register_verified_cold(request, old_receipt)
        wrong_request = Request(
            request_id="wrong-session",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
        )
        with pytest.raises(ValueError, match="authorize this session"):
            directory.register_verified_cold(wrong_request, receipt)
        assert directory.register_verified_cold(request, receipt) is checkpoint
        lane = _scheduler(root, directory)
        job = lane.submit("RESTORE", checkpoint)
        lane.tick(now_ns=5)
        assert lane.view(job).wait_reason == "RAM_ADMISSION_WAIT"
        assert not provider.admitted and provider.payload_opens == 0

        changed = _replace_after_verification(root, receipt)
        provider.ram_full = False
        assert _wait(lane, job, 5) == "MISS"
        assert provider.admitted and provider.payload_opens == 0
        for file, original in changed:
            file.write_bytes(original)
        job = lane.submit("RESTORE", checkpoint)
        assert _wait(lane, job, 5) == "VERIFIED"
        for rank, region in enumerate(template.regions):
            checksums = receipt.checksums(rank)
            for binding in template.destination[rank]:
                (span,) = binding.spans
                value = region.mapping[span.offset : span.offset + span.nbytes]
                assert sha256(value).hexdigest() == checksums[binding.part]
        assert template.held is None
        directory.forget_cold(checkpoint)
        return {
            "pid": os.getpid(),
            "regions": [region.descriptor.fingerprint() for region in template.regions],
        }


def _child(pipe, operation, root, args):
    try:
        pipe.send(operation(root, *args))
    finally:
        pipe.close()


def _run_process(operation, root, *args):
    context = multiprocessing.get_context("spawn")
    inbound, outbound = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(outbound, operation, root, args))
    process.start()
    outbound.close()
    try:
        assert inbound.poll(40), "FSL2 cross-boot child timed out"
        try:
            result = inbound.recv()
        except EOFError:
            pytest.fail("FSL2 cross-boot child exited before its diagnostic receipt")
    finally:
        inbound.close()
        process.join(3)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0
    return result


def test_fsl2_cold_catalog_recovers_after_a_fresh_d_boot(tmp_path):
    root = tmp_path / "cold"
    written = _run_process(_write, root)
    read = _run_process(_read, root, written["old_sig"])
    assert written["pid"] != read["pid"]
    assert set(written["regions"]).isdisjoint(read["regions"])
