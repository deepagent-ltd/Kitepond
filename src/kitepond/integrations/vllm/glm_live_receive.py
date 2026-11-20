# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt in to GLM rank windows on the actual native Mooncake worker."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from kitepond.integrations.mooncake.live_receive import (
    RankWindowAttempt,
    RankWindowPeer,
    ReverseRankWindowPeer,
    mooncake_receive_pool,
    receive_rank_windows,
)
from kitepond.integrations.mooncake.window import (
    RegisteredWindowPool,
    manifest_digest,
    plan_windows,
)
from kitepond.models.glm5next.cold_restore import ColdRestoreLayout
from kitepond.models.glm5next.live_windows import GlmRankWindowPlan

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from .decode_receive import DecodeReceiveMailbox, DecodeReceiveTicket
from .glm_decode_restore import validate_native_destination
from .hisparse_attachment import HiSparseConnector


class _NativeGlmSink:
    def __init__(
        self,
        plan: GlmRankWindowPlan,
        stream: torch.cuda.Stream | None,
        cow_event: torch.cuda.Event | None,
        imported: Callable[[], None] | None,
        *,
        needs_cow: bool,
        read_shared_mla: bool = False,
    ):
        self.plan, self.stream, self.cow_event = plan, stream, cow_event
        self.imported = imported
        self.read_shared_mla = read_shared_mla
        devices = {layer.tensor.device for layer in plan.final.layers.values()}
        cuda = {device for device in devices if device.type == "cuda"}
        self.host_writes = any(device.type == "cpu" for device in devices)
        if (
            any(device.type not in {"cpu", "cuda"} for device in devices)
            or (stream is None and (cuda or cow_event is not None))
            or (stream is not None and cuda != {stream.device})
            or (cow_event is not None and not isinstance(cow_event, torch.cuda.Event))
            or (stream is not None and needs_cow and cow_event is None)
        ):
            raise ValueError("GLM D windows require the live rank's native COW stream")

    def scatter(self, window, payload) -> int:
        if self.stream is None:
            return self._scatter_and_check(window, payload)
        if self.host_writes and self.cow_event is not None:
            self.cow_event.synchronize()
        with torch.cuda.stream(self.stream):
            if self.cow_event is not None:
                self.stream.wait_event(self.cow_event)
            return self._scatter_and_check(window, payload)

    def _scatter_and_check(self, window, payload) -> int:
        written = (
            self.plan.scatter(window, payload, read_shared_mla=True)
            if self.read_shared_mla
            else self.plan.scatter(window, payload)
        )
        self.plan.verify_readback(window, payload)
        return written

    def fence(self) -> None:
        if self.stream is None:
            return
        event = torch.cuda.Event()
        if self.cow_event is not None:
            self.stream.wait_event(self.cow_event)
        event.record(self.stream)
        event.synchronize()

    def finish(self) -> None:
        if self.imported is None:
            return
        if self.stream is None:
            self.imported()
        else:
            with torch.cuda.stream(self.stream):
                self.imported()


class GlmNativeWindowReceiver:
    """Shares vLLM's existing native TransferEngine with Mooncake."""

    def __init__(self, connector: MultiConnector, *, slots: int) -> None:
        if (
            type(connector) is not MultiConnector
            or connector.role is not KVConnectorRole.WORKER
            or len(connector.sub_connectors) != 2
        ):
            raise ValueError("GLM rank receive needs its explicit native worker pair")
        owner, mooncake = connector.sub_connectors
        if (
            type(owner) is not HiSparseConnector
            or owner.stage != "decode"
            or not isinstance(mooncake, MooncakeConnector)
            or mooncake.connector_worker is None
            or mooncake.connector_worker.is_kv_producer
        ):
            raise ValueError("GLM rank receive needs native D HiSparse and Mooncake")
        self._owner = owner
        self._pool = mooncake_receive_pool(mooncake.connector_worker.engine, slots)

    async def receive(
        self,
        ticket: DecodeReceiveTicket,
        plan: GlmRankWindowPlan,
        layer_groups: Mapping[str, int],
        peer: RankWindowPeer,
        *,
        source_boot_id: str,
        timeout: float,
        checkpoint_blocks: tuple[tuple[int, ...], ...] | None = None,
        stream: torch.cuda.Stream | None = None,
        cow_event: torch.cuda.Event | None = None,
        seed_imported_tail: Callable[[ColdRestoreLayout, DecodeReceiveTicket], None]
        | None = None,
        read_shared_mla: bool = False,
        forward_abort_peer: ReverseRankWindowPeer | None = None,
    ) -> None:
        """Use the live model's address plan fenced by the native rank owner."""
        await receive_glm_rank_window(
            self._owner.decode_receive_mailbox(),
            self._pool,
            ticket,
            plan,
            layer_groups,
            peer,
            source_boot_id=source_boot_id,
            timeout=timeout,
            checkpoint_blocks=checkpoint_blocks,
            stream=stream,
            cow_event=cow_event,
            seed_imported_tail=seed_imported_tail,
            read_shared_mla=read_shared_mla,
            forward_abort_peer=forward_abort_peer,
        )

    def close(self) -> None:
        """Native worker shutdown closes scratch only after peer leases settle."""
        self._pool.close()


async def receive_glm_rank_window(
    mailbox: DecodeReceiveMailbox,
    pool: RegisteredWindowPool,
    ticket: DecodeReceiveTicket,
    plan: GlmRankWindowPlan,
    layer_groups: Mapping[str, int],
    peer: RankWindowPeer,
    *,
    source_boot_id: str,
    timeout: float,
    checkpoint_blocks: tuple[tuple[int, ...], ...] | None = None,
    stream: torch.cuda.Stream | None = None,
    cow_event: torch.cuda.Event | None = None,
    seed_imported_tail: Callable[[ColdRestoreLayout, DecodeReceiveTicket], None]
    | None = None,
    read_shared_mla: bool = False,
    forward_abort_peer: ReverseRankWindowPeer | None = None,
) -> None:
    """Settle the live rank only after native verified scatter and source commit."""
    validate_native_destination(mailbox, ticket, plan.final, layer_groups)
    if plan.local_tokens < ticket.prefix_tokens // 1024 * 1024:
        raise ValueError("GLM live plan would overwrite a read-only grant page")
    if read_shared_mla and (
        ticket.prefix_tokens or ticket.copy_tail is not None or plan.local_tokens
    ):
        raise ValueError("Shared MLA verification requires one cold native grant")
    checkpoint = plan.checkpoint
    if (checkpoint is None) != (checkpoint_blocks is None) or (
        checkpoint is not None
        and (
            len(checkpoint_blocks) != len(ticket.block_ids)
            or any(
                name not in layer_groups
                or layer.block_ids != checkpoint_blocks[layer_groups[name]]
                for name, layer in checkpoint.layers.items()
            )
        )
    ):
        raise ValueError("GLM receive lacks the exact native checkpoint reservation")
    manifest = plan.manifest()
    attempt = RankWindowAttempt(
        request_id=ticket.request_id,
        transfer_id=ticket.transfer_id,
        epoch=ticket.epoch,
        rank=mailbox.rank,
        source_boot_id=source_boot_id,
        grant_sha256=ticket.grant_sha256,
        destination_sha256=ticket.destination_sha256,
        manifest_sha256=manifest_digest(manifest),
        mla_rank_digests=ticket.mla_rank_digests,
    )

    def finish() -> None:
        if seed_imported_tail is not None:
            seed_imported_tail(plan.final, ticket)

    sink = _NativeGlmSink(
        plan,
        stream,
        cow_event,
        finish,
        needs_cow=bool(ticket.copy_tail or plan.local_tokens),
        read_shared_mla=read_shared_mla,
    )
    plan_windows(manifest, window_bytes=pool.window_bytes)
    mailbox._start_import(ticket)
    await receive_rank_windows(
        attempt,
        manifest,
        pool,
        peer,
        sink,
        lambda outcome: mailbox._post_fenced(ticket, outcome),
        timeout=timeout,
        forward_abort_peer=forward_abort_peer,
    )
