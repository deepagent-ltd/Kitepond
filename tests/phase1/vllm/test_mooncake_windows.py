# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the externalized byte-window contract on CPU memory only."""

import unittest

from kitepond.integrations.mooncake.window import (
    WINDOW_BYTES,
    RegisteredWindowPool,
    WindowSendSession,
    manifest_digest,
    plan_windows,
)


class WindowSendTest(unittest.TestCase):
    def test_canonical_offsets_and_receiver_order_survive_window_split(self):
        manifest = [{"offset": 0, "nbytes": 7}, {"offset": 7, "nbytes": 8}]
        windows = plan_windows(manifest, window_bytes=8)
        self.assertEqual([window.nbytes for window in windows], [7, 8])
        self.assertEqual(windows[1].to_metadata()["entries"], [manifest[1]])
        outgoing = windows[1].to_metadata()
        outgoing["entries"][0]["offset"] = 0
        self.assertEqual(windows[1].to_metadata()["entries"], [manifest[1]])
        with self.assertRaisesRegex(ValueError, "canonical"):
            plan_windows([manifest[1], manifest[0]])

    def test_send_status_and_peer_consumption_are_independent(self):
        # The extra chunk starts a window at the independent chunk-count limit.
        manifest = [{"offset": n, "nbytes": 1} for n in range(65)]
        peer = ("loopback-fixture", 0, "pool-fixture")
        session = WindowSendSession(
            manifest, manifest_sha256=manifest_digest(manifest), destination=peer
        )
        descriptor = {
            "pool_id": peer[2],
            "slot": 0,
            "generation": 1,
            "address": 1024,
            "capacity": WINDOW_BYTES,
        }
        first, second = session.windows
        with self.assertRaisesRegex(ValueError, "out of order|reordered"):
            session.validate_fetch(second.to_metadata(), descriptor, 0, peer)
        with self.assertRaisesRegex(ValueError, "settle"):
            session.mark_copied(first, descriptor, native_write_status=-1)
        with self.assertRaisesRegex(ValueError, "before"):
            session.commit(1)
        session.mark_copied(first, descriptor, native_write_status=0)
        self.assertTrue(
            session.validate_fetch(first.to_metadata(), descriptor, -1, peer)[1]
        )
        with self.assertRaisesRegex(ValueError, "stale"):
            session.validate_fetch(second.to_metadata(), descriptor, 0, peer)
        with self.assertRaisesRegex(ValueError, "previous consumption"):
            session.validate_fetch(
                second.to_metadata(), dict(descriptor, generation=2), -1, peer
            )
        with self.assertRaisesRegex(ValueError, "fixed address"):
            session.validate_fetch(
                second.to_metadata(),
                dict(descriptor, generation=2, address=2048),
                0,
                peer,
            )
        session.mark_copied(
            second, dict(descriptor, generation=2), native_write_status=0
        )
        self.assertTrue(session.commit(1))
        self.assertFalse(session.commit(1))


class WindowPoolTest(unittest.IsolatedAsyncioTestCase):
    async def test_pool_unregistration_requires_settled_lease(self):
        registered = set()
        pool = RegisteredWindowPool(
            1,
            lambda tensor: registered.add(tensor.data_ptr()),
            lambda tensor: registered.remove(tensor.data_ptr()),
            window_bytes=16,
        )
        lease = await pool.acquire(1)
        with self.assertRaisesRegex(RuntimeError, "live or uncertain"):
            pool.close()
        await pool.release(lease, writes_settled=True)
        new_lease = await pool.acquire(1)
        self.assertGreater(new_lease.generation, lease.generation)
        with self.assertRaisesRegex(ValueError, "stale"):
            await pool.release(lease, writes_settled=True)
        await pool.release(new_lease, writes_settled=True)
        pool.close()
        pool.close()
        self.assertEqual(registered, set())

    async def test_partial_unregistration_never_reuses_regions_or_repeats_success(self):
        registered = []
        released = []

        def unregister(tensor):
            if len(released) == 1 and not allow_retry:
                raise RuntimeError("unregister temporarily unavailable")
            released.append(tensor.data_ptr())

        pool = RegisteredWindowPool(
            2,
            lambda tensor: registered.append(tensor.data_ptr()),
            unregister,
            window_bytes=16,
        )
        allow_retry = False
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            pool.close()
        self.assertEqual(pool.registered_bytes, 16)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await pool.acquire(1)
        allow_retry = True
        pool.close()
        self.assertCountEqual(released, registered)

    async def test_uncertain_writes_quarantine_registration(self):
        registered = set()
        pool = RegisteredWindowPool(
            1,
            lambda tensor: registered.add(tensor.data_ptr()),
            lambda tensor: registered.remove(tensor.data_ptr()),
            window_bytes=16,
        )
        lease = await pool.acquire(1)
        await pool.release(lease, writes_settled=False)
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            await pool.acquire(1)
        with self.assertRaisesRegex(RuntimeError, "uncertain"):
            pool.close()
        self.assertEqual(registered, {lease.tensor.data_ptr()})
