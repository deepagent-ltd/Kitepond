# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise provenance, lifecycle and bounds of the local READY proof route."""

import multiprocessing
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from kitepond.integrations.vllm.ready_transport import (
    ReadyAttempt,
    ReadyCollector,
    ReadyPublisher,
    ReadyReceipt,
    ReadyWorker,
)

_ATTEMPT = ReadyAttempt("request-1", "transfer-9", 7, "1" * 64)
_REGIONS = {rank: str(rank + 2) * 64 for rank in range(4)}
_TOKENS = 128


def _receipt(attempt, rank=0):
    return ReadyReceipt(
        attempt,
        rank,
        _REGIONS[rank],
        _TOKENS,
        _TOKENS,
        time.monotonic_ns(),
    )


def _worker(directory, rank, incoming):
    publisher = ReadyPublisher(directory)
    try:
        while (tickets := incoming.recv()) is not None:
            incoming.send(
                publisher.after_batch_save_fence(
                    lambda: tuple(
                        (ticket, _receipt(ticket.attempt, rank)) for ticket in tickets
                    )
                )
            )
    finally:
        publisher.close()


class ReadyTransportTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def make_collector(self, workers=None, **kwargs):
        if workers is None:
            workers = {
                0: ReadyWorker.from_pid(multiprocessing.current_process().pid),
            }
        collector = ReadyCollector(self.directory, workers, **kwargs)
        self.addCleanup(collector.close)
        return collector

    def make_publisher(self):
        publisher = ReadyPublisher(self.directory)
        self.addCleanup(publisher.close)
        return publisher

    def make_peer(self, rank):
        context = multiprocessing.get_context("spawn")
        pipe, child_pipe = context.Pipe()
        peer = context.Process(target=_worker, args=(self.directory, rank, child_pipe))
        peer.start()
        child_pipe.close()

        def stop():
            try:
                pipe.send(None)
            except BrokenPipeError:
                pass
            finally:
                pipe.close()
            peer.join(timeout=5)
            if peer.is_alive():
                peer.terminate()
                peer.join(timeout=5)

        self.addCleanup(stop)
        return peer, pipe

    def test_full_quorum_requires_enrolled_rank_processes(self):
        rank0, pipe0 = self.make_peer(0)
        rank1, pipe1 = self.make_peer(1)
        collector = self.make_collector(
            {0: ReadyWorker.from_pid(rank0.pid), 1: ReadyWorker.from_pid(rank1.pid)}
        )
        publisher = self.make_publisher()
        ticket = collector.arm(
            _ATTEMPT, {rank: _REGIONS[rank] for rank in range(2)}, _TOKENS
        )
        self.assertTrue(publisher.after_save_fence(ticket, lambda: _receipt(_ATTEMPT)))
        self.assertIsNone(collector.take(ticket))
        pipe0.send((ticket,))
        self.assertTrue(pipe0.poll(5))
        self.assertTrue(pipe0.recv())
        self.assertIsNone(collector.take(ticket))
        pipe1.send((ticket,))
        self.assertTrue(pipe1.poll(5))
        self.assertTrue(pipe1.recv())
        receipts = collector.take(ticket)
        self.assertEqual(tuple(row.rank for row in receipts), (0, 1))
        self.assertIsNone(collector.take(ticket))

    def test_four_rank_batch_at_64_requests(self):
        peers = [self.make_peer(rank) for rank in range(4)]
        collector = self.make_collector(
            {
                rank: ReadyWorker.from_pid(peer.pid)
                for rank, (peer, _) in enumerate(peers)
            }
        )
        tickets = tuple(
            collector.arm(
                replace(
                    _ATTEMPT, request_id=f"request-{idx}", transfer_id=f"transfer-{idx}"
                ),
                _REGIONS,
                _TOKENS,
            )
            for idx in range(64)
        )
        for _, pipe in peers:
            pipe.send(tickets)
        for _, pipe in peers:
            self.assertTrue(pipe.poll(5))
            self.assertEqual(pipe.recv(), 64)
        for ticket in tickets:
            receipts = collector.take(ticket)
            self.assertEqual(tuple(row.rank for row in receipts), (0, 1, 2, 3))
            self.assertEqual({row.attempt for row in receipts}, {ticket.attempt})

    def test_incomplete_fence_wrong_epoch_region_or_watermark(self):
        collector = self.make_collector()
        publisher = self.make_publisher()
        ticket = collector.arm(_ATTEMPT, {0: _REGIONS[0]}, _TOKENS)
        wrong = replace(_ATTEMPT, epoch=8)
        self.assertFalse(publisher.after_save_fence(ticket, lambda: None))
        self.assertIsNone(collector.take(ticket))
        self.assertFalse(publisher.after_save_fence(ticket, lambda: _receipt(wrong)))
        # Authenticate the sender but dispatch a wrong epoch with the admitted nonce.
        forged = replace(ticket, attempt=wrong)
        self.assertTrue(publisher.after_save_fence(forged, lambda: _receipt(wrong)))
        self.assertTrue(
            publisher.after_save_fence(
                ticket, lambda: replace(_receipt(_ATTEMPT), region_sha256=_REGIONS[1])
            )
        )
        self.assertTrue(
            publisher.after_save_fence(
                ticket,
                lambda: replace(_receipt(_ATTEMPT), draft_valid_tokens=_TOKENS - 1),
            )
        )
        self.assertIsNone(collector.take(ticket))
        self.assertTrue(publisher.after_save_fence(ticket, lambda: _receipt(_ATTEMPT)))
        self.assertEqual(len(collector.take(ticket)), 1)

    def test_cancel_late_conflict_and_capacity(self):
        collector = self.make_collector(max_pending=1)
        publisher = self.make_publisher()
        regions = {0: _REGIONS[0]}
        old = collector.arm(_ATTEMPT, regions, _TOKENS)
        with self.assertRaises(ValueError):
            collector.arm(replace(_ATTEMPT, epoch=9), regions, _TOKENS)
        with self.assertRaises(BufferError):
            collector.arm(replace(_ATTEMPT, request_id="different"), regions, _TOKENS)
        collector.cancel(old)
        live = collector.arm(_ATTEMPT, regions, _TOKENS)
        self.assertNotEqual(old.nonce, live.nonce)
        self.assertTrue(publisher.after_save_fence(old, lambda: _receipt(_ATTEMPT)))
        self.assertIsNone(collector.take(old))
        self.assertIsNone(collector.take(live))
        receipt = _receipt(_ATTEMPT)
        self.assertTrue(publisher.after_save_fence(live, lambda: receipt))
        conflicting = replace(
            receipt, completed_monotonic_ns=receipt.completed_monotonic_ns + 1
        )
        self.assertTrue(publisher.after_save_fence(live, lambda: conflicting))
        self.assertIsNone(collector.take(live))
        new = collector.arm(replace(_ATTEMPT, epoch=10), regions, _TOKENS)
        self.assertTrue(publisher.after_save_fence(new, lambda: _receipt(new.attempt)))
        self.assertEqual(len(collector.take(new)), 1)

    def test_expired_or_departed_worker_receipts(self):
        peer, pipe = self.make_peer(0)
        collector = self.make_collector({0: ReadyWorker.from_pid(peer.pid)})
        ticket = collector.arm(_ATTEMPT, {0: _REGIONS[0]}, _TOKENS)
        pipe.send((ticket,))
        self.assertTrue(pipe.poll(5))
        self.assertTrue(pipe.recv())
        peer.terminate()
        peer.join(timeout=5)
        self.assertIsNone(collector.take(ticket))
        collector.close()
        collector = self.make_collector(ttl_seconds=0.01)
        publisher = self.make_publisher()
        ticket = collector.arm(_ATTEMPT, {0: _REGIONS[0]}, _TOKENS)
        time.sleep(0.02)
        self.assertTrue(publisher.after_save_fence(ticket, lambda: _receipt(_ATTEMPT)))
        self.assertIsNone(collector.take(ticket))
