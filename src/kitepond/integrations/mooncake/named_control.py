# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Authenticated GLM Window V2 over the native Mooncake peer socket."""

from __future__ import annotations

import asyncio
import hmac
import json
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager, nullcontext
from dataclasses import asdict, dataclass, field

from kitepond.core.readiness import _digest

from .live_receive import (
    AbortedRankWindows,
    CommittedRankWindows,
    FencedRankWindows,
    NativeRankWindowWriter,
    RankWindowAttempt,
    WrittenRankWindow,
)

NAMESPACE = "kitepond.glm-window-v2"
_REVERSE_NAMESPACE = "kitepond.glm-terminal-return-v1"
_AUTH_DOMAIN = b"kitepond/glm-window-v2\x00"
_SIGNATURE_BYTES = 32
_MAX_PACKET_BYTES = 192 * 1024
_MAX_CLOSED_FORWARD_ATTEMPTS = 128


@dataclass(frozen=True)
class GlmWindowControlScope:
    """One local deployment's exact registered P and D worker pair."""

    source_boot_id: str
    decode_boot_id: str
    rank: int
    key: bytes = field(repr=False)
    reverse: bool = False
    forward_abort: bool = False

    def __post_init__(self) -> None:
        if (
            any(
                type(boot) is not str or not 0 < len(boot) <= 128
                for boot in (self.source_boot_id, self.decode_boot_id)
            )
            or self.source_boot_id == self.decode_boot_id
            or type(self.rank) is not int
            or not 0 <= self.rank < 4
            or type(self.key) is not bytes
            or len(self.key) != _SIGNATURE_BYTES
            or type(self.reverse) is not bool
            or type(self.forward_abort) is not bool
        ):
            raise ValueError("Invalid GLM control boot, rank or local key")

    @property
    def namespace(self) -> str:
        return _REVERSE_NAMESPACE if self.reverse else NAMESPACE

    @property
    def supports_abort(self) -> bool:
        return self.reverse or self.forward_abort

    def check_attempt(self, attempt: RankWindowAttempt) -> None:
        if (
            type(attempt) is not RankWindowAttempt
            or attempt.source_boot_id
            != (self.decode_boot_id if self.reverse else self.source_boot_id)
            or attempt.rank != self.rank
            or (self.reverse and attempt.mla_rank_digests is not None)
        ):
            raise ValueError("GLM control changed its qualified source rank")

    @property
    def _domain(self) -> bytes:
        if self.reverse:
            return b"kitepond/glm-terminal-return-v1\x00"
        return _AUTH_DOMAIN

    def _sign(self, direction: bytes, body: bytes) -> bytes:
        return hmac.digest(
            self.key, self._domain + direction + b"\x00" + body, "sha256"
        )

    def seal(self, direction, attempt, operation, sequence, arguments) -> bytes:
        self.check_attempt(attempt)
        identity = asdict(attempt)
        if attempt.mla_rank_digests is None:
            identity.pop("mla_rank_digests")
        body = json.dumps(
            {
                "p_boot": self.source_boot_id,
                "d_boot": self.decode_boot_id,
                "attempt": identity,
                "operation": operation,
                "sequence": sequence,
                "arguments": arguments,
            },
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode()
        packet = self._sign(direction, body) + body
        if len(packet) > _MAX_PACKET_BYTES:
            raise ValueError("GLM Mooncake control packet exceeds its bound")
        return packet

    def _allows_op(self, operation: str) -> bool:
        return (
            operation in ("fetch", "commit")
            or (self.supports_abort and operation in ("abort", "settle"))
            or (not self.reverse and operation == "prepare_mla")
        )

    def open(self, direction, packet):
        if type(packet) is not bytes or not (
            _SIGNATURE_BYTES < len(packet) <= _MAX_PACKET_BYTES
        ):
            raise ValueError("GLM Mooncake control authentication failed")
        body = packet[_SIGNATURE_BYTES:]
        if not hmac.compare_digest(
            packet[:_SIGNATURE_BYTES], self._sign(direction, body)
        ):
            raise ValueError("GLM Mooncake control authentication failed")
        try:
            value = json.loads(body)
            if (
                type(value) is not dict
                or set(value)
                != {"p_boot", "d_boot", "attempt", "operation", "sequence", "arguments"}
                or (value["p_boot"], value["d_boot"])
                != (self.source_boot_id, self.decode_boot_id)
                or type(value["sequence"]) is not int
                or not 0 < value["sequence"] < 2**63
                or not self._allows_op(value["operation"])
                or type(value["arguments"]) is not dict
            ):
                raise ValueError("Invalid GLM Mooncake control envelope")
            identity = value["attempt"]
            if type(identity) is not dict:
                raise ValueError("GLM control attempt is not an object")
            if "mla_rank_digests" in identity:
                ranks = identity["mla_rank_digests"]
                if type(ranks) is not list:
                    raise ValueError("GLM shared MLA proof is not an ordered rank list")
                identity["mla_rank_digests"] = tuple(ranks)
            attempt = RankWindowAttempt(**identity)
        except (TypeError, UnicodeError, ValueError, RecursionError) as error:
            raise ValueError("Invalid GLM Mooncake control envelope") from error
        self.check_attempt(attempt)
        return attempt, value["operation"], value["sequence"], value["arguments"]


@dataclass
class _ForwardActivity:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


@dataclass
class _ClosedForwardAttempt:
    attempt: RankWindowAttempt
    destination: tuple[str, int, str]
    sequence: int
    writer: NativeRankWindowWriter | None
    fenced: bool = False


class GlmWindowControlHandler:
    """Bind an existing writer to a strictly scoped named request handler."""

    def __init__(
        self,
        worker,
        scope: GlmWindowControlScope,
        source: Callable[[RankWindowAttempt], NativeRankWindowWriter | None],
        *,
        admit_source: Callable[
            [RankWindowAttempt, tuple[str, int, str]],
            Awaitable[NativeRankWindowWriter | None],
        ]
        | None = None,
        prepare_mla: Callable[[RankWindowAttempt], Awaitable[str | None]] | None = None,
        reject_native_requests: bool = False,
        settled: Callable[[FencedRankWindows], Awaitable[None]] | None = None,
        cancel_unopened: Callable[
            [RankWindowAttempt, tuple[str, int, str]], Awaitable[bool]
        ]
        | None = None,
    ) -> None:
        binder = getattr(worker, "bind_named_control_handler", None)
        forward_abort = scope.forward_abort and not scope.reverse
        if (
            worker.is_kv_consumer
            or worker.tp_rank != scope.rank
            or not callable(source)
            or not callable(binder)
            or (admit_source is not None and not callable(admit_source))
            or (
                prepare_mla is not None and (scope.reverse or not callable(prepare_mla))
            )
            or type(reject_native_requests) is not bool
            or (scope.reverse and not callable(settled))
            or (not scope.reverse and settled is not None)
            or forward_abort != callable(cancel_unopened)
        ):
            raise ValueError(
                "The native Mooncake sender control extension is unavailable"
            )
        self.worker, self.scope, self.source = worker, scope, source
        self.admit_source = admit_source
        self._prepare_mla = prepare_mla
        self.settled = settled
        self._cancel_unopened = cancel_unopened
        self._forward_activity: dict[str, _ForwardActivity] = {}
        self._closed_forward: dict[str, _ClosedForwardAttempt] = {}
        self._forward_aborts_inflight = 0
        self._sequences: weakref.WeakKeyDictionary[NativeRankWindowWriter, int] = (
            weakref.WeakKeyDictionary()
        )
        if reject_native_requests:
            binder(scope.namespace, self.dispatch, reject_native_requests=True)
        else:
            binder(scope.namespace, self.dispatch)

    async def dispatch(self, packet: bytes) -> bytes:
        attempt, operation, sequence, arguments = self.scope.open(b"fetcher", packet)
        if operation == "prepare_mla":
            if arguments or attempt.mla_rank_digests is not None:
                raise ValueError("Invalid GLM MLA preflight")
            proof = None
            if self._prepare_mla is not None:
                async with self._forward_lock(attempt.transfer_id):
                    proof = await self._prepare_mla(attempt)
                if proof is not None and not _digest(proof):
                    raise ValueError("Invalid GLM MLA source proof")
            return self.scope.seal(
                b"writer", attempt, operation, sequence, {"mla_sha256": proof}
            )
        destination = None
        if operation == "fetch":
            if (
                set(arguments)
                != {"recipe", "descriptor", "previous_consumed", "destination"}
                or type(arguments["recipe"]) is not dict
                or type(arguments["descriptor"]) is not dict
                or type(arguments["previous_consumed"]) is not int
            ):
                raise ValueError("Invalid GLM Mooncake fetch")
            destination = self._destination(arguments.pop("destination"))
            if destination[2] != arguments["descriptor"].get("pool_id"):
                raise ValueError("GLM Mooncake registered destination changed its pool")
        elif operation == "abort":
            if set(arguments) != {"destination"}:
                raise ValueError("Invalid GLM Mooncake abort")
            destination = self._destination(arguments["destination"])
        elif operation == "settle":
            if set(arguments) != {"outcome"} or arguments["outcome"] not in (
                "acked",
                "abort_settled",
            ):
                raise ValueError("Invalid GLM Mooncake settlement")
        elif (
            set(arguments) != {"last_sequence"}
            or type(arguments["last_sequence"]) is not int
        ):
            raise ValueError("Invalid GLM Mooncake commit")

        if self._cancel_unopened is not None:
            async with self._forward_lock(attempt.transfer_id):
                result = await self._dispatch_forward(
                    attempt, operation, sequence, arguments, destination
                )
        else:
            result = await self._execute(
                attempt, operation, sequence, arguments, destination
            )
        return self.scope.seal(b"writer", attempt, operation, sequence, result)

    @asynccontextmanager
    async def _forward_lock(self, transfer_id):
        # Source admission can yield before installing a writer. Serialize the
        # whole attempt so a later abort sees the actual fate of that admission.
        activity = self._forward_activity.get(transfer_id)
        if activity is None:
            activity = self._forward_activity[transfer_id] = _ForwardActivity()
        activity.users += 1
        try:
            async with activity.lock:
                yield
        finally:
            activity.users -= 1
            if not activity.users:
                del self._forward_activity[transfer_id]

    async def _dispatch_forward(
        self, attempt, operation, sequence, arguments, destination
    ) -> dict:
        closed = self._closed_forward.get(attempt.transfer_id)
        if closed is not None:
            if (
                closed.attempt != attempt
                or sequence <= closed.sequence
                or (destination is not None and closed.destination != destination)
            ):
                raise ValueError("GLM forward abort changed its settled attempt")
            if operation == "settle" and arguments["outcome"] == "abort_settled":
                if closed.writer is not None:
                    await closed.writer.fenced_settlement(attempt, "abort_settled")
                closed.writer = None
                closed.fenced = True
                closed.sequence = sequence
                return {"outcome": "abort_settled"}
            if operation == "abort":
                closed.sequence = sequence
                return {"closed": True}
            raise ValueError("GLM forward attempt is closed")
        if operation == "settle":
            raise ValueError("GLM forward has no authenticated source abort")
        if operation != "abort":
            return await self._execute(
                attempt, operation, sequence, arguments, destination
            )
        assert self._cancel_unopened is not None and destination is not None
        if (
            len(self._closed_forward) + self._forward_aborts_inflight
            >= _MAX_CLOSED_FORWARD_ATTEMPTS
        ):
            retired = next(
                (key for key, value in self._closed_forward.items() if value.fenced),
                None,
            )
            if retired is None:
                raise RuntimeError("GLM forward abort settlement backlog is full")
            del self._closed_forward[retired]
        self._forward_aborts_inflight += 1
        try:
            writer = self.source(attempt)
            if writer is None:
                if await self._cancel_unopened(attempt, destination) is not True:
                    raise ValueError("GLM forward has no proof of an unopened source")
            else:
                self._verify_writer(writer, attempt, destination, sequence)
                await writer.abandon()
            self._closed_forward[attempt.transfer_id] = _ClosedForwardAttempt(
                attempt, destination, sequence, writer
            )
            return {"closed": True}
        finally:
            self._forward_aborts_inflight -= 1

    async def _execute(
        self, attempt, operation, sequence, arguments, destination
    ) -> dict:
        writer = self.source(attempt)
        if writer is None and destination is not None and self.admit_source is not None:
            writer = await self.admit_source(attempt, destination)
        self._verify_writer(writer, attempt, destination, sequence)

        if operation == "fetch":
            reply = await writer.fetch(attempt, **arguments)
            result = {
                "window_sha256": reply.window_sha256,
                "descriptor": reply.descriptor,
            }
        elif operation == "commit":
            reply = await writer.commit(attempt, arguments["last_sequence"])
            result = {"consumed_sequence": reply.consumed_sequence}
        elif operation == "abort":
            await writer.abandon()
            result = {"closed": True}
        else:
            proof = await writer.fenced_settlement(attempt, arguments["outcome"])
            assert self.settled is not None
            await self.settled(proof)
            result = {"outcome": proof.outcome}
        return result

    def _verify_writer(self, writer, attempt, destination, sequence) -> None:
        if (
            type(writer) is not NativeRankWindowWriter
            or writer.engine is not self.worker.engine
            or writer.attempt != attempt
            or (destination is not None and writer.session.destination != destination)
        ):
            raise ValueError("GLM Mooncake control has no live source owner")
        if sequence <= self._sequences.get(writer, 0):
            raise ValueError("GLM Mooncake control replay")
        self._sequences[writer] = sequence

    @staticmethod
    def _destination(endpoint) -> tuple[str, int, str]:
        if (
            type(endpoint) is not list
            or len(endpoint) != 3
            or type(endpoint[0]) is not str
            or not endpoint[0]
            or type(endpoint[1]) is not int
            or not 1 <= endpoint[1] <= 65535
            or type(endpoint[2]) is not str
            or not endpoint[2]
        ):
            raise ValueError("Invalid GLM Mooncake registered destination")
        return tuple(endpoint)


class _GlmWindowControlPeer:
    def __init__(
        self,
        socket,
        encode,
        scope: GlmWindowControlScope,
        local_engine: tuple[str, int],
        open_socket=None,
    ) -> None:
        self._socket, self._encode, self._scope = socket, encode, scope
        self._local_engine = local_engine
        self._open_socket = open_socket
        self._attempt: RankWindowAttempt | None = None
        self._sequence = 0
        self._waiting = False

    async def _request(self, attempt, operation, arguments, *, fresh=False):
        self._scope.check_attempt(attempt)
        if self._waiting or (self._attempt is not None and self._attempt != attempt):
            raise ValueError("GLM control socket is already bound to another request")
        self._attempt = attempt
        self._sequence += 1
        sequence = self._sequence
        self._waiting = True
        try:
            packet = self._scope.seal(
                b"fetcher", attempt, operation, sequence, arguments
            )
            route = self._open_socket() if fresh else nullcontext(self._socket)
            with route as socket:
                await socket.send(self._encode(self._scope.namespace, packet))
                response = await socket.recv()
            seen_attempt, seen_op, seen_seq, result = self._scope.open(
                b"writer", response
            )
            if (seen_attempt, seen_op, seen_seq) != (attempt, operation, sequence):
                raise ValueError("GLM control reply changed its live request")
            return result
        finally:
            self._waiting = False

    async def prepare_mla(self, attempt: RankWindowAttempt) -> str | None:
        """Read one fenced source identity without opening a write window."""
        if self._scope.reverse:
            raise ValueError("MLA preflight belongs to the forward producer")
        result = await self._request(attempt, "prepare_mla", {})
        if set(result) != {"mla_sha256"}:
            raise ValueError("GLM MLA preflight reply changed its shape")
        proof = result["mla_sha256"]
        if proof is not None and not _digest(proof):
            raise ValueError("GLM MLA preflight reply is not a source digest")
        return proof

    async def fetch(
        self,
        attempt: RankWindowAttempt,
        recipe: dict,
        descriptor: dict,
        previous_consumed: int,
    ) -> WrittenRankWindow:
        result = await self._request(
            attempt,
            "fetch",
            {
                "recipe": recipe,
                "descriptor": descriptor,
                "previous_consumed": previous_consumed,
                "destination": (*self._local_engine, descriptor.get("pool_id")),
            },
        )
        if set(result) != {"window_sha256", "descriptor"}:
            raise ValueError("GLM Mooncake fetch reply lacks its window descriptor")
        return WrittenRankWindow(attempt, result["window_sha256"], result["descriptor"])

    async def commit(
        self, attempt: RankWindowAttempt, last_sequence: int
    ) -> CommittedRankWindows:
        result = await self._request(
            attempt, "commit", {"last_sequence": last_sequence}
        )
        if (
            set(result) != {"consumed_sequence"}
            or type(result["consumed_sequence"]) is not int
        ):
            raise ValueError("GLM Mooncake commit reply lacks its rank boundary")
        return CommittedRankWindows(attempt, result["consumed_sequence"])

    async def abort(
        self, attempt: RankWindowAttempt, pool_id: str
    ) -> AbortedRankWindows:
        if not self._scope.supports_abort:
            raise ValueError("GLM source abort is not enabled")
        result = await self._request(
            attempt,
            "abort",
            {"destination": (*self._local_engine, pool_id)},
            fresh=True,
        )
        if result != {"closed": True} or result["closed"] is not True:
            raise ValueError("GLM reverse writer did not settle its abort")
        return AbortedRankWindows(attempt)

    async def settle(self, proof: FencedRankWindows) -> None:
        if not self._scope.supports_abort:
            raise ValueError("GLM source settlement is not enabled")
        result = await self._request(
            proof.attempt, "settle", {"outcome": proof.outcome}, fresh=True
        )
        if result != {"outcome": proof.outcome}:
            raise ValueError("GLM reverse writer changed the fenced result")


@contextmanager
def bind_glm_window_peer(
    worker, source_address: str, scope: GlmWindowControlScope
) -> Iterator[_GlmWindowControlPeer]:
    """Open one attempt on the installed native fetcher context and route."""
    import zmq

    from vllm.utils.network_utils import make_zmq_socket

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import named_control
    except ImportError as error:
        raise ValueError(
            "The native Mooncake D control extension is unavailable"
        ) from error

    if (
        worker.is_kv_producer
        or worker.tp_rank != scope.rank
        or type(source_address) is not str
        or not source_address.startswith("tcp://")
    ):
        raise ValueError("GLM control requires its native fetcher and writer route")

    def open_socket():
        return make_zmq_socket(
            worker.async_zmq_ctx, source_address, zmq.DEALER, bind=False, linger=0
        )

    with open_socket() as socket:
        yield _GlmWindowControlPeer(
            socket,
            named_control.encode_mooncake_control_message,
            scope,
            (worker.hostname, worker.rpc_port),
            open_socket,
        )


@asynccontextmanager
async def connect_glm_window_peer(
    worker,
    remote_engine_id: str,
    bootstrap_address: str,
    scope: GlmWindowControlScope,
    *,
    discovery_timeout: float | None = None,
) -> AsyncIterator[_GlmWindowControlPeer]:
    """Discover the admitted source rank through native bootstrap."""
    resolve = getattr(worker, "resolve_named_control_peer", None)
    if (
        worker.is_kv_producer
        or worker.tp_rank != scope.rank
        or not callable(resolve)
        or type(remote_engine_id) is not str
        or not remote_engine_id
        or type(bootstrap_address) is not str
        or not bootstrap_address.startswith(("http://", "https://"))
    ):
        raise ValueError("GLM Mooncake control requires its native bootstrap route")
    source_address = await asyncio.wait_for(
        resolve(remote_engine_id, bootstrap_address), discovery_timeout
    )
    with bind_glm_window_peer(worker, source_address, scope) as peer:
        yield peer
