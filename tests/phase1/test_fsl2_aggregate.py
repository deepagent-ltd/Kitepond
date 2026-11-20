# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the four-rank manifest on actual CPU FSL2 artifacts."""

import fcntl
import json
import multiprocessing
import os
from contextlib import ExitStack, contextmanager
from dataclasses import replace

import pytest
from kitepond.core.contracts import Wait

torch = pytest.importorskip("torch")
memory = pytest.importorskip("lmcache.v1.memory_management")
fsl2 = pytest.importorskip("kitepond.backends.fsl2")
fsl2_aggregate = pytest.importorskip("kitepond.backends.fsl2_aggregate")

METADATA = {
    "schema": "fixture-native-layout",
    "model_fingerprint": "a" * 64,
    "config_fingerprint": "b" * 64,
    "boundary_tokens": 64,
}
IDENTITY = fsl2.BundleIdentity("owner/decode", "a" * 64, fsl2.digest(METADATA), 12, 64)


class _RankOwner(fsl2.NativeColdAuthority):
    def __init__(self, rank, restore):
        self.part = fsl2.PartSpec("exact-bytes", rank, 0, (64,), "torch.uint8", 64)
        self.plan = fsl2.BundlePlan(
            "fixture-rank-v1", "c" * 64, (self.part.role,), (self.part,)
        )
        self.bytes = (
            torch.zeros(64, dtype=torch.uint8)
            if restore
            else torch.arange(64, dtype=torch.uint8) + rank
        )
        self.memory = memory.TensorMemoryObj(
            self.bytes,
            memory.MemoryObjMetadata(
                shape=self.bytes.shape,
                dtype=self.bytes.dtype,
                address=self.bytes.data_ptr(),
                phy_size=self.bytes.nbytes,
                ref_count=1,
            ),
            parent_allocator=None,
        )
        self.store = fsl2.ColdSnapshotLease(self, object())
        self.restore = fsl2.RamRestoreLease(self, object())

    def check_store(self, lease, identity, plan):
        assert lease is self.store and identity == IDENTITY and plan == self.plan

    def check_restore(self, lease, identity, plan):
        assert lease is self.restore and identity == IDENTITY and plan == self.plan

    def borrow_source(self, lease, part):
        assert lease is self.store and part == self.part
        return fsl2.MemoryBorrow(self, self.memory)

    def borrow_destination(self, lease, part):
        assert lease is self.restore and part == self.part
        return fsl2.MemoryBorrow(self, self.memory)

    def finish_store(self, lease):
        assert lease is self.store and self.memory.get_ref_count() == 1

    def finish_restore(self, lease, verified):
        assert lease is self.restore and verified and self.memory.get_ref_count() == 1


@contextmanager
def _ranks(root, *, restore=False):
    owners = tuple(_RankOwner(rank, restore) for rank in range(4))
    plans = tuple(owner.plan for owner in owners)
    with ExitStack() as teardown:
        adapters = []
        for rank, owner in enumerate(owners):
            adapter = fsl2.ColdStorageAdapter(
                root / f"rank-{rank}", owner, max_inflight_bytes=65536
            )
            adapters.append(adapter)
            teardown.callback(_close, adapter)
        yield tuple(adapters), owners, plans


def _close(adapter):
    assert adapter.close(10)


def _store(adapters, owners, plans, ranks=(0, 1, 2, 3)):
    for rank in ranks:
        writer, owner, plan = adapters[rank], owners[rank], plans[rank]
        ticket = writer.begin_store(IDENTITY, plan, owner.store)
        assert writer.wait(ticket, 10) == "COMMITTED"


def test_partial_store_is_hidden_and_four_ranks_reopen_with_fresh_ram(tmp_path):
    with _ranks(tmp_path) as (adapters, owners, plans):
        aggregate = fsl2_aggregate.FourRankManifestStore(tmp_path, adapters)
        _store(adapters, owners, plans, range(3))
        assert aggregate.publish(IDENTITY, plans, METADATA) is None
        assert aggregate.discover(IDENTITY, plans) is None
        assert not list(aggregate.manifests.iterdir())

        _store(adapters, owners, plans, [3])
        document = aggregate.publish(IDENTITY, plans, METADATA)
        (manifest,) = aggregate.manifests.glob("*.json")
        inode = manifest.stat().st_ino
        assert aggregate.publish(IDENTITY, plans, METADATA) == document
        assert manifest.stat().st_ino == inode
        assert [row["rank"] for row in document["ranks"]] == list(range(4))

    with _ranks(tmp_path, restore=True) as (adapters, owners, plans):
        reopened = fsl2_aggregate.FourRankManifestStore(tmp_path, adapters)
        assert reopened.discover(IDENTITY, plans) == document
        assert reopened.discover(replace(IDENTITY, generation=13), plans) is None
        assert reopened.discover(replace(IDENTITY, namespace="foreign"), plans) is None
        for rank, (reader, owner, plan) in enumerate(
            zip(adapters, owners, plans, strict=True)
        ):
            ticket = reader.begin_restore(IDENTITY, plan, owner.restore)
            assert reader.wait(ticket, 10) == "VERIFIED"
            assert torch.equal(owner.bytes, torch.arange(64, dtype=torch.uint8) + rank)


def _crash_before_publication(root, connection, cutpoint):
    with _ranks(root) as (adapters, owners, plans):
        aggregate = fsl2_aggregate.FourRankManifestStore(root, adapters)
        if cutpoint == "missing-fourth-rank":
            _store(adapters, owners, plans, range(3))
            connection.send("ranks-durable")
            os._exit(71)

        _store(adapters, owners, plans)

        def signal_before_link(*_):
            connection.send("rank-and-staging-files-durable")
            os._exit(72)

        # Only the isolated writer is interrupted. Drop at the atomic
        # aggregate publish link, before readers could see a manifest.
        fsl2_aggregate.os.link = signal_before_link
        aggregate.publish(IDENTITY, plans, METADATA)
        raise AssertionError("The writer never crossed the crash point")


@pytest.mark.parametrize(
    ("cutpoint", "event", "exitcode", "missing"),
    [
        ("missing-fourth-rank", "ranks-durable", 71, [3]),
        ("before-aggregate-link", "rank-and-staging-files-durable", 72, []),
    ],
)
def test_process_exit_during_cold_write_requires_verified_republication(
    tmp_path, cutpoint, event, exitcode, missing
):
    spawn = multiprocessing.get_context("spawn")
    reader, writer = spawn.Pipe(duplex=False)
    child = spawn.Process(
        target=_crash_before_publication,
        args=(tmp_path, writer, cutpoint),
    )
    child.start()
    writer.close()
    try:
        assert reader.poll(30)
        assert reader.recv() == event
        child.join(timeout=15)
        assert child.exitcode == exitcode
    finally:
        reader.close()
        if child.is_alive():
            child.kill()
            child.join(timeout=15)
        child.close()

    with _ranks(tmp_path) as (adapters, owners, plans):
        aggregate = fsl2_aggregate.FourRankManifestStore(tmp_path, adapters)
        assert aggregate.discover(IDENTITY, plans) is None
        assert not list(aggregate.manifests.glob("*.json"))
        _store(adapters, owners, plans, missing)
        committed = aggregate.publish(IDENTITY, plans, METADATA)
        assert committed is not None

    # A new owner, lease and tensors must recover the exact original bytes.
    with _ranks(tmp_path, restore=True) as (adapters, owners, plans):
        aggregate = fsl2_aggregate.FourRankManifestStore(tmp_path, adapters)
        assert aggregate.discover(IDENTITY, plans) == committed
        for rank, (reader, owner, plan) in enumerate(
            zip(adapters, owners, plans, strict=True)
        ):
            ticket = reader.begin_restore(IDENTITY, plan, owner.restore)
            assert reader.wait(ticket, 10) == "VERIFIED"
            assert torch.equal(owner.bytes, torch.arange(64, dtype=torch.uint8) + rank)


def test_corruption_cannot_be_hidden_behind_a_rehashed_rank_or_envelope(tmp_path):
    with _ranks(tmp_path) as (adapters, owners, plans):
        aggregate = fsl2_aggregate.FourRankManifestStore(tmp_path, adapters)
        _store(adapters, owners, plans)
        original = aggregate.publish(IDENTITY, plans, METADATA)
        (manifest,) = aggregate.manifests.glob("*.json")
        rank = adapters[2]
        (rank_manifest,) = rank.manifests.glob("*.json")
        valid = rank_manifest.read_bytes()
        corrupted = json.loads(valid)
        del corrupted["digest"]
        corrupted["parts"][0]["sha256"] = "f" * 64
        rank_manifest.write_bytes(
            fsl2.canonical(dict(corrupted, digest=fsl2.digest(corrupted)))
        )
        assert rank.discover(IDENTITY, plans[2]) is not None
        assert aggregate.discover(IDENTITY, plans) is None
        rank_manifest.write_bytes(valid)

        corrupted = json.loads(manifest.read_bytes())
        del corrupted["digest"]
        corrupted["identity"]["generation"] = 13
        manifest.write_bytes(
            fsl2.canonical(dict(corrupted, digest=fsl2.digest(corrupted)))
        )
        assert aggregate.discover(IDENTITY, plans) is None
        with pytest.raises(FileExistsError):
            aggregate.publish(IDENTITY, plans, METADATA)
        assert json.loads(manifest.read_bytes()) != original


def test_failed_atomic_link_and_bad_verified_rank_bytes_leave_no_candidate(
    tmp_path, monkeypatch
):
    with _ranks(tmp_path) as (adapters, owners, plans):
        aggregate = fsl2_aggregate.FourRankManifestStore(tmp_path, adapters)
        _store(adapters, owners, plans)
        with pytest.raises(ValueError, match="metadata differs"):
            aggregate.publish(IDENTITY, plans, METADATA | {"boundary_tokens": 33})
        with pytest.raises(ValueError, match="four exact rank plans"):
            aggregate.publish(IDENTITY, tuple(reversed(plans)), METADATA)

        def fail_link(*_):
            raise OSError("Injected interruption before link")

        with monkeypatch.context() as patch:
            patch.setattr(fsl2_aggregate.os, "link", fail_link)
            with pytest.raises(OSError, match="interruption"):
                aggregate.publish(IDENTITY, plans, METADATA)
        assert not list(aggregate.manifests.iterdir())
        assert aggregate.discover(IDENTITY, plans) is None

        real_sync = fsl2_aggregate.sync_directory

        def fail_commit_sync(path):
            if path == aggregate.manifests:
                raise OSError("Injected aggregate directory fsync failure")
            return real_sync(path)

        with monkeypatch.context() as patch:
            patch.setattr(fsl2_aggregate, "sync_directory", fail_commit_sync)
            with pytest.raises(OSError, match="fsync failure"):
                aggregate.publish(IDENTITY, plans, METADATA)
        assert not list(aggregate.manifests.iterdir())

        (payload,) = (p for p in adapters[3].data.iterdir() if p.is_file())
        payload.write_bytes(bytes(64))
        with pytest.raises(ValueError, match="checksum mismatch"):
            aggregate.publish(IDENTITY, plans, METADATA)
        assert not list(aggregate.manifests.iterdir())


def test_aggregate_respects_existing_root_and_rank_ownership(tmp_path):
    with _ranks(tmp_path) as (adapters, _, plans):
        l2 = fsl2_aggregate.FourRankManifestStore(tmp_path, adapters)
        for lock in (tmp_path / ".full-io.lock", adapters[2].root / ".owner.lock"):
            with lock.open("a+b") as owner:
                fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                assert l2.publish(IDENTITY, plans, METADATA) == Wait(
                    "COLD_IO_ADMISSION_WAIT"
                )
                assert l2.discover(IDENTITY, plans) == Wait("COLD_IO_ADMISSION_WAIT")
            assert l2.discover(IDENTITY, plans) is None
