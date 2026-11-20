# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One test-only drop of a real authenticated Mooncake write reply."""

from __future__ import annotations

import asyncio
import json
import os
import stat
from dataclasses import asdict, dataclass
from hashlib import sha256
from time import monotonic_ns
from typing import TYPE_CHECKING

from kitepond.core.readiness import ReadyAttempt, _digest
from kitepond.integrations.mooncake.live_receive import NativeRankWindowWriter
from kitepond.integrations.mooncake.window import manifest_digest

if TYPE_CHECKING:
    from pathlib import Path

ARM_FILE = "target.json"
DROP = "source-write-ack-dropped.json"
ABORT = "source-abort-ack.json"
SETTLE = "source-settle-ack.json"
HOLD = "target-before-fence.json"
FENCED = "target-after-fence.json"


@dataclass(frozen=True)
class AckTarget:
    attempt: ReadyAttempt
    rank: int
    p_boot: str
    d_boot: str

    def __post_init__(self):
        if (
            type(self.attempt) is not ReadyAttempt
            or type(self.rank) is not int
            or not 0 <= self.rank < 4
            or any(
                not isinstance(value, str) or not value
                for value in (self.p_boot, self.d_boot)
            )
            or self.p_boot == self.d_boot
        ):
            raise ValueError("Fault target must identify one real rank and boot pair")

    def matches_scope(self, scope) -> bool:
        return not scope.reverse and (
            scope.rank,
            scope.source_boot_id,
            scope.decode_boot_id,
        ) == (self.rank, self.p_boot, self.d_boot)

    def matches(self, attempt) -> bool:
        return (
            self.attempt.request_id,
            self.attempt.transfer_id,
            self.attempt.epoch,
            self.attempt.lease_sha256,
        ) == (
            attempt.request_id,
            attempt.transfer_id,
            attempt.epoch,
            attempt.grant_sha256,
        )


def require_private(directory: Path) -> None:
    info = directory.lstat()
    if (
        not directory.is_absolute()
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("Fault instrumentation requires its private owner directory")


def write_event(directory: Path, name: str, row: dict) -> None:
    data = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    # Readers see the complete evidence only. The final path cannot be replaced.
    pending = directory / f".{name}.{os.getpid()}.pending"
    fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
        os.link(pending, directory / name, follow_symlinks=False)
    finally:
        pending.unlink()


def arm_fault(directory: Path, target: AckTarget) -> None:
    require_private(directory)
    write_event(directory, ARM_FILE, asdict(target))


def read_target(directory: Path) -> AckTarget | None:
    try:
        fd = os.open(directory / ARM_FILE, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or not 0 < info.st_size <= 4096
        ):
            raise ValueError("Invalid private one-shot target")
        body = json.loads(source.read(4097))
    attempt = ReadyAttempt(**body["attempt"])
    return AckTarget(**(body | {"attempt": attempt}))


class _SocketAckObserver:
    def __init__(self, socket, intercept):
        self._socket, self._intercept = socket, intercept

    async def send_multipart(self, frames):
        if not self._intercept(frames):
            await self._socket.send_multipart(frames)


def install_missing_write_ack(native, scope, source, directory: Path) -> None:
    """Drop one real wire ACK after the native handler has proven its write."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.named_control import (
        decode_mooncake_control_message,
    )

    require_private(directory)
    if (
        scope.reverse
        or not scope.forward_abort
        or native.tp_rank != scope.rank
        or native.is_kv_consumer
        or getattr(native, "_kp_ack_loss_installed", False)
    ):
        raise ValueError("Fault needs one native forward producer")
    native._kp_ack_loss_installed = True
    dispatch = native._dispatch_named_control
    dropped: object | None = None

    async def with_one_drop(identity, socket, packet):
        nonlocal dropped
        frame = decode_mooncake_control_message(packet)
        if frame is None or frame[0] != scope.namespace:
            return await dispatch(identity, socket, packet)
        try:
            attempt, operation, sequence, arguments = scope.open(b"fetcher", frame[1])
        except ValueError:
            # Authentication failures keep their native disposition.
            return await dispatch(identity, socket, packet)
        target = read_target(directory)
        if (
            target is None
            or not target.matches_scope(scope)
            or not target.matches(attempt)
            or operation not in ("fetch", "abort", "settle")
        ):
            return await dispatch(identity, socket, packet)

        def watch(frames) -> bool:
            nonlocal dropped
            peer, response = frames
            if peer != identity:
                raise ValueError("Diagnostic ACK has a different native peer")
            seen_attempt, seen_operation, seen_sequence, reply = scope.open(
                b"writer", response
            )
            if (seen_attempt, seen_operation, seen_sequence) != (
                attempt,
                operation,
                sequence,
            ):
                raise ValueError("Diagnostic ACK has different signed request identity")
            proof = {
                "attempt": asdict(attempt),
                "p_boot": scope.source_boot_id,
                "d_boot": scope.decode_boot_id,
                "operation": operation,
                "control_sequence": sequence,
                "signed_reply_sha256": sha256(response).hexdigest(),
                "monotonic_ns": monotonic_ns(),
            }
            if operation == "fetch" and dropped is None:
                recipe = arguments["recipe"]
                descriptor = arguments["descriptor"]
                writer = source(attempt)
                if (
                    type(writer) is not NativeRankWindowWriter
                    or writer.engine is not native.engine
                    or writer._released
                    or writer._dma_uncertain
                    or recipe["sequence"] != 0
                    or arguments["previous_consumed"] != -1
                    or writer.session.next_sequence != 1
                    or writer.session.last_descriptor != descriptor
                    or reply["descriptor"] != descriptor
                    or reply["window_sha256"] != manifest_digest(recipe)
                ):
                    raise ValueError(
                        "Cannot drop an unproven or non-first native write"
                    )
                copied = writer.session.windows[0].nbytes
                if copied <= 0 or not _digest(reply["window_sha256"]):
                    raise ValueError("The native write acknowledged no committed bytes")
                write_event(
                    directory,
                    DROP,
                    proof
                    | {
                        "native_write_completed": True,
                        "source_released": False,
                        "window_sequence": 0,
                        "slot": descriptor["slot"],
                        "generation": descriptor["generation"],
                        "nbytes": copied,
                    },
                )
                dropped = attempt
                return True
            if (
                attempt == dropped
                and operation == "abort"
                and reply == {"closed": True}
            ):
                write_event(directory, ABORT, proof | {"closed": True})
            if (
                attempt == dropped
                and operation == "settle"
                and reply == {"outcome": "abort_settled"}
                and not (directory / SETTLE).exists()
            ):
                write_event(directory, SETTLE, proof | {"outcome": "abort_settled"})
            return False

        return await dispatch(identity, _SocketAckObserver(socket, watch), packet)

    native._dispatch_named_control = with_one_drop


def install_target_observation(runtime, directory: Path) -> None:
    """Record the real D pre-fence hold and post-fence mailbox receipt."""
    require_private(directory)
    if runtime.scope.reverse or getattr(runtime, "_kp_ack_loss_observer", False):
        raise ValueError("Fault evidence needs one forward D receiver instance")
    runtime._kp_ack_loss_observer = True
    receive = runtime._receive

    async def with_observation(ticket, plan, groups, fence):
        target = read_target(directory)
        if (
            target is None
            or not target.matches_scope(runtime.scope)
            or not target.matches(ticket)
        ):
            return await receive(ticket, plan, groups, fence)

        pool = runtime._receiver._pool
        mailbox = runtime._hisparse.decode_receive_mailbox()
        native_post = mailbox._post_fenced
        settled = None

        def observe_fence(observed, outcome):
            nonlocal settled
            native_post(observed, outcome)
            if observed == ticket:
                settled = mailbox.settled_proof(ticket)

        async def observe_hold(task):
            while not task.done() and not (directory / DROP).is_file():
                await asyncio.sleep(0.01)
            if task.done():
                return
            record = json.loads((directory / DROP).read_bytes())
            if record["attempt"]["destination_sha256"] != ticket.destination_sha256:
                return
            slot = pool._slots[record["slot"]]
            if (
                not slot.leased
                or slot.generation != record["generation"]
                or slot.quarantined
                or mailbox.settled_proof(ticket) is not None
            ):
                return
            count = record["nbytes"]
            payload = slot.tensor[:count].numpy().tobytes()
            write_event(
                directory,
                HOLD,
                {
                    "attempt": record["attempt"],
                    "slot": slot.slot,
                    "generation": slot.generation,
                    "leased": True,
                    "native_result_published": False,
                    "payload_sha256": sha256(payload).hexdigest(),
                    "monotonic_ns": monotonic_ns(),
                },
            )

        mailbox._post_fenced = observe_fence
        task = asyncio.create_task(receive(ticket, plan, groups, fence))
        observer = asyncio.create_task(observe_hold(task))
        try:
            return await task
        finally:
            mailbox._post_fenced = native_post
            if not observer.done():
                observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            if settled is not None:
                write_event(
                    directory,
                    FENCED,
                    {
                        "native_receipt": asdict(settled),
                        "request_id": ticket.request_id,
                        "transfer_id": ticket.transfer_id,
                        "epoch": ticket.epoch,
                        "leased_scratch_slots": pool.leased_slots,
                        "quarantined_scratch_slots": pool.quarantined_slots,
                        "monotonic_ns": monotonic_ns(),
                    },
                )

    runtime._receive = with_observation
