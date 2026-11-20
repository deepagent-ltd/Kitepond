# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import multiprocessing
import os
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from threading import Thread
from time import monotonic, monotonic_ns, sleep
from types import SimpleNamespace

import pytest
from kitepond.core.contracts import Wait
from kitepond.integrations.mooncake.window import RegisteredWindowPool
from kitepond.integrations.vllm.decode_receive import DecodeReceiveMailbox
from kitepond.integrations.vllm.glm_cold_enrollment import NativeGlmColdRegions
from kitepond.integrations.vllm.glm_cold_rank_control import RPC_NAME
from kitepond.integrations.vllm.glm_cold_remote_restore import NativeGlmColdRestore
from kitepond.integrations.vllm.glm_cold_scheduler import NativeGlmColdStorePolicy
from kitepond.integrations.vllm.glm_decode_bootstrap import GlmDecodeImportRuntime
from kitepond.integrations.vllm.glm_nvme_control import (
    NAMESPACE,
    RESUME_PARAM,
    seal_nvme_operation,
)
from kitepond.integrations.vllm.glm_scheduler import GlmKitepondScheduler
from kitepond.integrations.vllm.glm_terminal_bootstrap import GlmTerminalStartup
from kitepond.integrations.vllm.glm_terminal_manifest import TerminalRankSigner
from kitepond.integrations.vllm.glm_terminal_worker import GlmTerminalWorker
from kitepond.integrations.vllm.hisparse_attachment import ATTACHMENT_MODE
from kitepond.models.glm5next.frontier import digest
from kitepond.models.glm5next.region_socket import join_region
from stock_hisparse_attachment_probe import _finish, _request, create_scheduler
from test_glm_cold_catalog import (
    NEW_BOOT,
    OLD_BOOT,
    SCOPE,
    _catalog,
    _run_process,
    _write,
)
from test_glm_cold_detached_decode import (
    BOUNDARY,
    LAYOUT,
    FixtureRankPort,
    _checkpoint,
)
from test_glm_cold_lifecycle import BUDGET

from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiKVConnectorWorkerMetadata,
)
from vllm.v1.hisparse.coordinator import HiSparseCoordinator, get_hisparse_coordinator
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus


def _policy(scheduler):
    root = Path(scheduler.vllm_config.additional_config["kitepond_glm_cold"]["root"])
    return NativeGlmColdStorePolicy(
        lambda _: SCOPE,
        _catalog(root),
        4,
        authorize_prefix=lambda *_: digest([3] * BOUNDARY),
    )


def _no_remote_path(*_args):
    raise AssertionError("Detached restore may not contact a P route")


def _worker(pipe, directory, rank, pages):
    with ExitStack() as resources:
        resources.callback(pipe.close)
        port = FixtureRankPort(resources, directory, rank, pages)
        owned = port.memory.region.descriptor
        signer = TerminalRankSigner(rank, bytes([rank + 1]) * 32)
        mailbox = DecodeReceiveMailbox(rank, NEW_BOOT)
        pool = RegisteredWindowPool(1, lambda _: None, lambda _: None)
        loop = asyncio.new_event_loop()
        thread = Thread(target=loop.run_forever)
        thread.start()

        def close():
            loop.call_soon_threadsafe(loop.stop)
            thread.join(5)
            loop.close()

        resources.callback(close)
        runtime = object.__new__(GlmDecodeImportRuntime)
        runtime.views, runtime._detached = port.views, None
        runtime._imports, runtime._current_tickets, runtime._finished = {}, set(), set()
        runtime.worker = SimpleNamespace(receiver_loop=loop)
        runtime.profile = SimpleNamespace(timeout_seconds=5)
        runtime._receiver = SimpleNamespace(_pool=pool)
        runtime._hisparse = SimpleNamespace(decode_receive_mailbox=lambda: mailbox)
        runtime._main_stream = runtime._receive_stream = None
        runtime._plan = _no_remote_path
        worker = object.__new__(GlmTerminalWorker)
        worker._kitepond_native_state_views = port.views
        worker._kitepond_native_cold_memory = port.memory
        worker._kitepond_decode_import = runtime
        worker._require_terminal_source = lambda: (signer, None)
        worker.vllm_config = SimpleNamespace(
            additional_config={
                "kitepond_glm_cold_store_policy": {"mode": "store-restore-lab"},
            }
        )
        pipe.send((os.getpid(), owned, port.memory._extent.layout))
        while True:
            action, arguments = pipe.recv()
            if action == "close":
                if saved := getattr(worker, "_kitepond_cold_restore", None):
                    assert not saved._pending
                pool.close()
                pipe.send("closed")
                return
            if action == "control":
                try:
                    pipe.send(worker.kitepond_control_cold_rank(*arguments))
                except ValueError as error:
                    pipe.send(error)
                continue
            assert action == "ticket"
            (ticket,) = arguments
            foreign = replace(ticket, grant_sha256="0" * 64)
            mailbox.bind((foreign,))
            with pytest.raises(ValueError, match="lacks its exact local restore claim"):
                runtime._submit(foreign)
            mailbox.bind((ticket,))
            runtime._submit(ticket)
            future = runtime._imports[ticket.request_id].future
            future.result(timeout=8)
            proof = mailbox.drain(None)
            runtime._detached.retire_fenced(ticket)
            assert pool.leased_slots == pool.quarantined_slots == 0
            assert not worker._kitepond_cold_restore._pending
            pipe.send(proof)


class Collective:
    def __init__(self, pipes):
        self.pipes = pipes
        self.last_prepare = None
        self.lose_prepare = False
        self.lose_release = False
        self.forge_release = False

    def __call__(self, method, *, timeout, args):
        assert method == RPC_NAME
        command, signatures = args
        if command.verb == "prepare":
            self.last_prepare = args
        missing = command.verb == "release" and self.lose_release
        peers = self.pipes[:3] if missing else self.pipes
        for pipe in peers:
            pipe.send(("control", (command, signatures)))
        results = []
        for pipe in peers:
            assert pipe.poll(timeout)
            results.append(pipe.recv())
        if missing or (command.verb == "prepare" and self.lose_prepare):
            raise TimeoutError("Injected missing native worker acknowledgement")
        if command.verb == "release" and self.forge_release:
            results[-1] = replace(results[-1], signature=results[0].signature)
        return results


def _packet(proof, *, finished=None):
    return ModelRunnerOutput.with_kv_conn_output_only(
        KVConnectorOutput(
            finished_recving=finished,
            kv_connector_worker_meta=MultiKVConnectorWorkerMetadata(
                metadata=(proof, None)
            ),
        )
    )


def _spawn_ranks(cleanup, directory, pages):
    context = multiprocessing.get_context("spawn")
    children = []
    for rank in range(4):
        local, remote = context.Pipe()
        process = context.Process(target=_worker, args=(remote, directory, rank, pages))
        process.start()
        remote.close()
        children.append((process, local))

    def reap():
        for process, pipe in children:
            pipe.close()
            process.join(2)
            if process.is_alive():
                process.terminate()
                process.join(5)

    cleanup.callback(reap)
    fingerprints, regions, pids = [], [], []
    for rank, (_, local) in enumerate(children):
        assert local.poll(55), "Native cold worker did not start"
        pid, descriptor, layout = local.recv()
        pid_region = join_region(
            directory,
            authority_engine_id="D",
            authority_boot_id=NEW_BOOT,
            source_rank=rank,
            layout_sha256=LAYOUT,
        )
        assert pid != os.getpid() and descriptor.owner_pid == pid
        assert pid_region.descriptor == descriptor and descriptor.source_rank == rank
        cleanup.callback(pid_region.close)
        fingerprints.append(descriptor.fingerprint())
        regions.append(pid_region)
        pids.append(pid)
    assert len(set(pids)) == len(set(fingerprints)) == 4
    return children, NativeGlmColdRegions(layout, tuple(regions), LAYOUT), pids


@pytest.mark.skipif(
    not callable(getattr(HiSparseCoordinator, "attach_external_host_blocks", None)),
    reason="this engine cannot attach verified cold grants to native requests",
)
@pytest.mark.parametrize(
    "nvme", (False, True), ids=("trusted-internal", "operator-ingress")
)
def test_signed_restore_across_four_worker_processes_and_release_quorum(
    tmp_path, monkeypatch, nvme
):
    root = tmp_path / "cold"
    prefix_sha256 = digest([3] * BOUNDARY)
    written = _run_process(_write, root, _checkpoint(), prefix_sha256 if nvme else None)
    key = tmp_path / "control.key"
    key.write_bytes(bytes(range(32)))
    key.chmod(0o600)
    monkeypatch.setenv("KITEPOND_GLM_COLD_PREFILL_KEY_FILE", str(key))

    operator_key = bytes(reversed(range(32)))
    if nvme:
        operator_file = tmp_path / "operator.key"
        operator_file.write_bytes(operator_key)
        operator_file.chmod(0o600)
        monkeypatch.setenv("KITEPOND_GLM_NVME_OPERATOR_KEY_FILE", str(operator_file))
    lane = {
        "max_active_jobs": 1,
        "max_inflight_bytes": BUDGET,
        "per_job_bytes": BUDGET,
    }
    scheduler = create_scheduler.__wrapped__(tmp_path)(
        "decode",
        routed=True,
        scheduler_cls=GlmKitepondScheduler,
        extra_config={
            **({"kitepond_glm_nvme_control": {"max_pending": 4}} if nvme else {}),
            "kitepond": {
                "mode": ATTACHMENT_MODE,
                "stage": "decode",
                "authority_engine_id": "D",
                "boot_id": NEW_BOOT,
            },
            "kitepond_glm_cold_import": {
                "prefill_bootstrap": "http://127.0.0.1:18162",
                "timeout_seconds": 5,
            },
            "kitepond_glm_cold": {
                "mode": "idle-fsl2",
                "root": str(root),
                "engine_id": "D",
                "boot_id": NEW_BOOT,
                "cadence_ns": 20_000_000,
                "idle_minutes": 15,
                "foreground": lane,
                "background": lane,
            },
            "kitepond_glm_cold_store_policy": {
                "mode": "store-restore-lab",
                "factory": __name__ + "._policy",
            },
            "kitepond_glm_cold_prefill": {
                "p_boot_id": "P-1",
                "max_pending": 2,
                "arm_timeout_seconds": 15,
            },
        },
    )
    with ExitStack() as cleanup:
        cleanup.callback(scheduler.shutdown)
        host = scheduler._cold_decode_authority
        free = host.pool.get_num_free_blocks()
        scheduler.bind_terminal_checkpoints(source_boot_id=NEW_BOOT)
        sockets = tmp_path / "rank"
        sockets.mkdir(mode=0o700)
        children, regions, pids = _spawn_ranks(cleanup, sockets, len(host.pool.blocks))
        pipes = [local for _, local in children]
        rpc = Collective(pipes)
        startup = GlmTerminalStartup(scheduler)
        scheduler._terminal_startup = startup
        scheduler._terminal_capture = SimpleNamespace(terminal_host_prefix=None)
        startup._cold_regions = regions
        startup._cold_signers = tuple(
            TerminalRankSigner(rank, bytes([rank + 1]) * 32) for rank in range(4)
        )
        methods, bind = scheduler.get_owner_worker_control_registration()
        assert methods == (RPC_NAME,)
        bind(rpc)
        runtime = scheduler.native_glm_cold_store
        assert isinstance(runtime.directory.provider, NativeGlmColdRestore)
        bridge = scheduler.native_glm_cold_decode_bridge
        catalog = runtime.directory.catalog
        verified = catalog.reopen(SCOPE)
        assert verified is not None and verified.source_boot_id == OLD_BOOT
        assert not catalog.verify(written["old_sig"])
        if nvme:
            assert verified.accepted_prefix_sha256 == prefix_sha256
            assert not catalog.verify(
                replace(verified, accepted_prefix_sha256="0" * 64)
            )
        cadence, callback = scheduler.get_quiescent_callback()
        assert cadence == 20_000_000
        opened = []

        def audit(event, args):
            if event == "open" and isinstance(args[0], (str, bytes)):
                path = os.fsdecode(args[0])
                if path.startswith(str(root)) and "/payloads/" in path:
                    assert host.pool.get_num_free_blocks() == free - 2
                    assert not scheduler.has_requests()
                    opened.append(path)

        sys.addaudithook(audit)
        denied = bridge.queue(
            _request(BOUNDARY + 3, "rank-loss"), verified, transfer_id="lost", epoch=7
        )
        assert not isinstance(denied, Wait)
        rpc.lose_prepare = rpc.lose_release = True
        callback()
        assert (
            runtime.scheduler.view(denied.job_id).wait_reason
            == "NATIVE_COLD_RANK_RELEASE_UNPROVEN"
        )
        assert host.pool.get_num_free_blocks() == free - 2 and not opened
        assert not scheduler.has_requests()
        stale_prepare = rpc.last_prepare
        rpc.lose_release = False
        rpc.forge_release = True
        assert bridge.cancel(denied) == Wait("NATIVE_COLD_RANK_RELEASE_UNPROVEN")
        assert host.pool.get_num_free_blocks() == free - 2
        rpc.forge_release = False
        assert bridge.cancel(denied) is True
        assert host.pool.get_num_free_blocks() == free
        rpc.lose_prepare = False
        repeated = rpc(RPC_NAME, timeout=5, args=stale_prepare)
        assert [result.state for result in repeated] == ["released"] * 4

        request = _request(BOUNDARY + 3, "rank-resume")
        deadline = monotonic() + 20
        if nvme:
            control = scheduler.connector.get_control_handler(NAMESPACE)
            assert callable(control)
            common = {
                "engine_id": "D",
                "boot_id": NEW_BOOT,
                "request_id": request.request_id,
            }
            intent = {
                **common,
                "namespace": SCOPE.namespace,
                "generation": SCOPE.generation,
                "transfer_id": "good",
                "epoch": 8,
                "deadline_ns": monotonic_ns() + 20_000_000_000,
                "boundary": BOUNDARY,
                "prefix_sha256": prefix_sha256,
                "prompt_sha256": digest(request.prompt_token_ids),
            }
            old = seal_nvme_operation(
                "resume_arm", dict(intent, boot_id=OLD_BOOT), operator_key
            )
            with pytest.raises(ValueError, match="current boot"):
                control("resume_arm", old)
            armed = control(
                "resume_arm", seal_nvme_operation("resume_arm", intent, operator_key)
            )
            request.kv_transfer_params = {RESUME_PARAM: "0" * 64}
            scheduler.add_request(request)
            assert not scheduler.has_requests() and not opened
            rejected = seal_nvme_operation("status", common, operator_key)
            assert control("status", rejected)["state"] == "armed"
            archive_request = Request(
                request_id="archive-reserved-before-add",
                prompt_token_ids=request.prompt_token_ids,
                sampling_params=request.sampling_params,
                pooling_params=None,
            )
            archive_id = {**common, "request_id": archive_request.request_id}
            archive_intent = {
                **archive_id,
                "namespace": SCOPE.namespace,
                "generation": SCOPE.generation,
                "transfer_id": "future-archive",
                "epoch": 2,
                "deadline_ns": monotonic_ns() + 60_000_000_000,
            }
            assert control(
                "archive_arm",
                seal_nvme_operation("archive_arm", archive_intent, operator_key),
            ) == {"state": "armed"}
            future = scheduler._native_glm_nvme_operator._operations[
                archive_request.request_id
            ]
            assert future.request is None
            scheduler.add_request(archive_request)
            assert future.request is scheduler.requests[archive_request.request_id]
            scheduler.finish_requests(
                archive_request.request_id, RequestStatus.FINISHED_ABORTED
            )
            # The real EngineCore drains the finished notification before it
            # calls a quiescent callback. This fixture drives native steps itself.
            cleanup_step = scheduler.schedule()
            assert archive_request.request_id in cleanup_step.finished_req_ids
            assert not scheduler.has_requests()
            control("cancel", seal_nvme_operation("cancel", archive_id, operator_key))
            assert control(
                "forget", seal_nvme_operation("forget", archive_id, operator_key)
            ) == {"state": "forgotten"}
            request.kv_transfer_params = {RESUME_PARAM: armed["resume_token"]}
            cancelled = Request(
                request_id="cancel-before-ram",
                prompt_token_ids=request.prompt_token_ids,
                sampling_params=request.sampling_params,
                pooling_params=None,
            )
            cancel_intent = {**intent, "request_id": cancelled.request_id}
            cancelled_token = control(
                "resume_arm",
                seal_nvme_operation("resume_arm", cancel_intent, operator_key),
            )["resume_token"]
            cancelled.kv_transfer_params = {RESUME_PARAM: cancelled_token}
            scheduler.add_request(cancelled)
            assert (
                scheduler.finish_requests(
                    [cancelled.request_id], RequestStatus.FINISHED_ABORTED
                )
                == []
            )
            callback()
            cancelled_id = {**common, "request_id": cancelled.request_id}
            assert (
                control(
                    "status", seal_nvme_operation("status", cancelled_id, operator_key)
                )["state"]
                == "cancelled"
            )
            scheduler.add_request(request)
            assert not scheduler.has_requests() and not opened
            status_query = seal_nvme_operation("status", common, operator_key)
            operator = scheduler._native_glm_nvme_operator
            pending = None
            while control("status", status_query)["state"] != "admitted":
                assert monotonic() < deadline
                callback()
                pending = (
                    operator._operations[request.request_id].cold_admission or pending
                )
                sleep(cadence / 1_000_000_000)
            assert pending is not None and pending.request is request
            assert scheduler.requests[request.request_id] is request
        else:
            pending = bridge.queue(request, verified, transfer_id="good", epoch=8)
            assert not isinstance(pending, Wait) and not scheduler.has_requests()
            while runtime.scheduler.view(pending.job_id).state != "VERIFIED":
                assert monotonic() < deadline
                callback()
                sleep(cadence / 1_000_000_000)
            assert bridge.stage(pending) is True
        assert len(opened) == 20
        step = scheduler.schedule()
        (ticket,) = step.kv_connector_metadata.metadata[0].tickets
        assert ticket.grant_sha256 == pending.grant.sha256 and ticket.detached_cold
        assert request.status is RequestStatus.WAITING_FOR_REMOTE_KVS
        for pipe in pipes:
            pipe.send(("ticket", (ticket,)))
        for rank, pipe in enumerate(pipes):
            assert pipe.poll(10)
            proof = pipe.recv()
            (receipt,) = proof.receipts
            assert receipt.rank == rank and receipt.outcome == "loaded"
            scheduler.update_from_output(
                step, _packet(proof, finished={request.request_id})
            )
            if rank < 3:
                assert scheduler.verified_decode_import(request) is None
        assert scheduler.verified_decode_import(request) is not None
        resumed = scheduler.schedule()
        assert resumed.num_scheduled_tokens == {request.request_id: 3}
        _finish(scheduler, request)
        pending_spills = get_hisparse_coordinator(
            scheduler.kv_cache_manager
        ).pending_spills
        if pending_spills:
            counts = dict.fromkeys(pending_spills, 1)
            scheduler.update_from_output(
                resumed, _packet(HiSparseConnectorWorkerMetadata(counts, counts))
            )
        assert host.pool.get_num_free_blocks() == free
        for pipe in pipes:
            pipe.send(("close", ()))
        for process, pipe in children:
            assert pipe.poll(15) and pipe.recv() == "closed"
            process.join(15)
            assert process.exitcode == 0

        (tmp_path / "cold-rank-control-receipt.json").write_text(
            json.dumps(
                {
                    "device": "cpu",
                    "operator_ingress": nvme,
                    "source_pid": written["pid"],
                    "source_boot": OLD_BOOT,
                    "reader_pid": os.getpid(),
                    "reader_boot": NEW_BOOT,
                    "worker_pids": pids,
                    "payload_reads": len(opened),
                    "native_suffix_tokens": 3,
                    "held_on_lost_or_forged_release": True,
                    "released_after_four_signed_replies": True,
                    "retired_prepare_replay_refused": True,
                    "host_pages_final": free,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
