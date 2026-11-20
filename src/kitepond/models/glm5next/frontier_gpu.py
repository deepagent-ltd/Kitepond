# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed proof storage by native request slot, no decode trace or KV store."""

import threading

import torch

from vllm.triton_utils import tl, triton

FRAME_WORDS = 32
STATE_WORDS = 11
MAX_QUERY = 4
MAX_REQUESTS = 64


@triton.jit(
    do_not_specialize=["max_slots", "num_reqs", "padded_tokens", "batch_serial"]
)
def _capture_target(
    ids,
    positions,
    qsl,
    mapping,
    frames,
    states,
    batch_slots,
    max_slots,
    num_reqs,
    padded_tokens,
    batch_serial,
):
    batch = tl.program_id(0)
    req = tl.load(mapping + batch).to(tl.int64)
    tl.store(batch_slots + batch, req)
    if req < 0 or req >= max_slots:
        return
    state = states + req * 11
    if tl.load(state + 10) == 0:
        return
    frame = frames + req * 32
    actual_end = tl.load(qsl + num_reqs).to(tl.int64)
    tl.store(state + 2, tl.load(state + 2) & (actual_end == padded_tokens).to(tl.int64))
    start = tl.load(qsl + batch).to(tl.int64)
    end = tl.load(qsl + batch + 1).to(tl.int64)
    q = end - start
    lane = tl.arange(0, 4)
    active = (lane < q) & (q > 0) & (q <= 4)
    tl.store(frame + lane, tl.load(positions + start + lane, active, -1).to(tl.int64))
    tl.store(frame + 4 + lane, tl.load(ids + start + lane, active, -1).to(tl.int64))
    tl.store(frame + 18, q)
    tl.store(frame + 19, req)
    tl.store(frame + 22, tl.load(state + 4).to(tl.int64) + 1)
    tl.store(frame + 23, tl.full((), -1, tl.int64))
    tl.store(frame + 26, start)
    tl.store(frame + 27, end)
    tl.store(frame + 28, tl.full((), 0, tl.int64))
    tl.store(frame + 24, batch_serial.to(tl.int64))
    tl.store(frame + 25, tl.full((), -1, tl.int64))


@triton.jit(do_not_specialize=["max_slots"])
def _capture_draft(
    ids, positions, qsl, last, draft_step, frames, states, batch_slots, max_slots
):
    batch = tl.program_id(0)
    req = tl.load(batch_slots + batch).to(tl.int64)
    if req < 0 or req >= max_slots:
        return
    state = states + req * 11
    if tl.load(state + 10) == 0:
        return
    frame = frames + req * 32
    start = tl.load(qsl + batch).to(tl.int64)
    end = tl.load(qsl + batch + 1).to(tl.int64)
    q = end - start
    lane = tl.arange(0, 4)
    active = (lane < q) & (q > 0) & (q <= 4)
    tl.store(
        frame + 8 + lane, tl.load(positions + start + lane, active, -1).to(tl.int64)
    )
    tl.store(frame + 12 + lane, tl.load(ids + start + lane, active, -1).to(tl.int64))
    tl.store(frame + 16, tl.load(last + batch).to(tl.int64))
    tl.store(frame + 17, tl.load(draft_step).to(tl.int64))
    tl.store(frame + 20, start)
    tl.store(frame + 21, end)
    tl.store(frame + 28, (tl.load(frame + 23) != -1).to(tl.int64))
    tl.store(frame + 23, tl.load(frame + 22).to(tl.int64))
    tl.store(frame + 25, tl.load(frame + 24).to(tl.int64))


@triton.jit(do_not_specialize=["table_rows", "table_cols", "table_stride", "max_slots"])
def _advance_frontier(
    frames,
    states,
    batch_slots,
    computed,
    rejected,
    tokens,
    table_rows,
    table_cols,
    table_stride,
    max_slots,
):
    batch = tl.program_id(0)
    req = tl.load(batch_slots + batch).to(tl.int64)
    if req < 0 or req >= max_slots or req >= table_rows:
        return
    state = states + req * 11
    if tl.load(state + 10) == 0:
        return
    frame = frames + req * 32
    lane = tl.arange(0, 4)
    q = tl.load(frame + 18).to(tl.int64)
    prev_t = tl.load(state).to(tl.int64)
    prev_d = tl.load(state + 1).to(tl.int64)
    t_valid = tl.load(state + 2).to(tl.int64)
    d_valid = tl.load(state + 3).to(tl.int64)
    steps = tl.load(state + 4).to(tl.int64)
    raw_end = tl.load(computed + req).to(tl.int64)
    reject = tl.load(rejected + batch).to(tl.int64)
    accepted = q - reject
    last = tl.load(frame + 16).to(tl.int64)
    draft_start = tl.load(frame + 20).to(tl.int64)
    last_relative = last - draft_start
    common = (
        (q > 0)
        & (q <= 4)
        & (tl.load(frame + 19) == req)
        & (tl.load(state + 8) == req)
        & (accepted > 0)
        & (accepted <= q)
        & (tl.load(frame + 22) == steps + 1)
        & (tl.load(frame + 23) == steps + 1)
        & (tl.load(frame + 27) - tl.load(frame + 26) == q)
    )
    active = lane < accepted
    t_pos = tl.load(frame + lane).to(tl.int64)
    t_id = tl.load(frame + 4 + lane).to(tl.int64)
    t_bound = (t_pos >= 0) & (t_pos < table_cols)
    t_expect = tl.load(
        tokens + req * table_stride + t_pos, active & t_bound, other=-1
    ).to(tl.int64)
    t_rows_ok = (
        tl.sum(
            tl.where(
                active,
                ((t_pos == prev_t + lane) & t_bound & (t_id == t_expect)).to(tl.int32),
                1,
            )
        )
        == 4
    )
    all_positions_ok = (
        tl.sum(tl.where(lane < q, (t_pos == prev_t + lane).to(tl.int32), 1)) == 4
    )
    target_ok = common & (raw_end == prev_t + accepted) & t_rows_ok & all_positions_ok
    d_pos = tl.load(frame + 8 + lane).to(tl.int64)
    d_id = tl.load(frame + 12 + lane).to(tl.int64)
    d_bound = (d_pos >= 0) & (d_pos + 1 < table_cols)
    d_expect = tl.load(
        tokens + req * table_stride + d_pos + 1, active & d_bound, other=-1
    ).to(tl.int64)
    d_rows_ok = (
        tl.sum(
            tl.where(
                active,
                ((d_pos == prev_d + lane) & d_bound & (d_id == d_expect)).to(tl.int32),
                1,
            )
        )
        == 4
    )
    draft_ok = (
        common
        & (last_relative >= 0)
        & (last_relative < q)
        & (last_relative + 1 == accepted)
        & (tl.load(frame + 17) == 0)
        & (tl.load(frame + 21) - draft_start == q)
        & (tl.load(frame + 28) == 0)
        & (raw_end == prev_d + last_relative + 1)
        & d_rows_ok
    )
    tl.store(state, raw_end)
    tl.store(state + 1, raw_end)
    tl.store(state + 2, t_valid & target_ok.to(tl.int64))
    tl.store(state + 3, d_valid & draft_ok.to(tl.int64))
    tl.store(state + 4, steps + 1)
    tl.store(state + 6, prev_t)
    tl.store(state + 7, raw_end)
    tl.store(state + 9, accepted)
    tl.store(frame + 29, reject)


class NativeAcceptedFrontier:
    """Three small kernels per batch; no CPU tensor reads/sync during decode."""

    def __init__(self, device, max_num_reqs, stream=None):
        if type(max_num_reqs) is not int or not 1 <= max_num_reqs <= MAX_REQUESTS:
            raise ValueError("Native frontier supports at most 64 request slots")
        self.capacity = max_num_reqs
        self.stream = stream
        self._lock = threading.RLock()
        self.state = torch.zeros(
            (max_num_reqs, STATE_WORDS), dtype=torch.int64, device=device
        )
        self.frame = torch.zeros(
            (max_num_reqs, FRAME_WORDS), dtype=torch.int64, device=device
        )
        self.batch_slots = torch.full(
            (max_num_reqs,), -1, dtype=torch.int64, device=device
        )
        self.requests = {}
        self.slot_owners = {}
        self.table_identity = None
        self.num_reqs = 0
        self.serial = 0

    def start(self, request_id, imported_tokens, req_index):
        with self._lock, torch.cuda.stream(self.stream):
            if (
                not request_id
                or type(imported_tokens) is not int
                or imported_tokens < 0
                or (type(req_index) is not int)
                or (not 0 <= req_index < self.capacity)
            ):
                raise ValueError("Invalid native accepted-frontier origin")
            if request_id in self.requests:
                if self.requests[request_id] != req_index:
                    raise ValueError("Native request moved slots")
                return
            if req_index in self.slot_owners:
                raise ValueError("Native request slot still owns an unsettled frontier")
            self.requests[request_id] = req_index
            self.slot_owners[req_index] = request_id
            self.state[req_index].copy_(
                torch.tensor(
                    [
                        imported_tokens,
                        imported_tokens,
                        1,
                        1,
                        0,
                        imported_tokens,
                        imported_tokens,
                        imported_tokens,
                        req_index,
                        0,
                        1,
                    ],
                    dtype=torch.int64,
                    device=self.state.device,
                )
            )
            self.frame[req_index].fill_(-1)

    def target(self, batch, batch_serial=None):
        self.serial = self.serial + 1 if batch_serial is None else batch_serial
        if type(self.serial) is not int or self.serial <= 0:
            raise ValueError("Invalid native sample submission identity")
        if not 1 <= batch.num_reqs <= self.capacity:
            raise ValueError("Native batch exceeds frontier capacity")
        self.num_reqs = batch.num_reqs
        _capture_target[batch.num_reqs,](
            batch.input_ids,
            batch.positions,
            batch.query_start_loc,
            batch.idx_mapping,
            self.frame,
            self.state,
            self.batch_slots,
            self.capacity,
            batch.num_reqs,
            batch.num_tokens_after_padding,
            self.serial,
            num_warps=1,
        )

    def draft(self, speculator):
        _capture_draft[self.num_reqs,](
            speculator.input_buffers.input_ids,
            speculator.input_buffers.positions,
            speculator.input_buffers.query_start_loc,
            speculator.last_token_indices,
            speculator.current_draft_step,
            self.frame,
            self.state,
            self.batch_slots,
            self.capacity,
            num_warps=1,
        )

    def advance(self, runner, rejected):
        table = runner.req_states.all_token_ids.gpu
        identity = (table.data_ptr(), tuple(table.shape), tuple(table.stride()))
        if self.table_identity is None:
            self.table_identity = identity
        if identity != self.table_identity or table.ndim != 2 or table.stride(1) != 1:
            raise ValueError("Native token storage changed during accepted frontier")
        _advance_frontier[self.num_reqs,](
            self.frame,
            self.state,
            self.batch_slots,
            runner.req_states.num_computed_tokens.gpu,
            rejected,
            table,
            table.shape[0],
            table.shape[1],
            table.stride(0),
            self.capacity,
            num_warps=1,
        )

    def clear(self, request_id):
        with self._lock, torch.cuda.stream(self.stream):
            slot = self.requests.pop(request_id, None)
            if slot is not None:
                self.slot_owners.pop(slot)
                self.state[slot, 10].zero_()

    def warmup(self, runner):
        with torch.cuda.stream(runner.main_stream):
            target = runner.input_buffers
            mapping = torch.arange(
                self.capacity, dtype=torch.int64, device=self.state.device
            )
            rejected = torch.zeros(
                self.capacity, dtype=torch.int32, device=self.state.device
            )
            _capture_target[self.capacity,](
                target.input_ids,
                target.positions,
                target.query_start_loc,
                mapping,
                self.frame,
                self.state,
                self.batch_slots,
                self.capacity,
                self.capacity,
                0,
                0,
                num_warps=1,
            )
            spec = runner.speculator
            _capture_draft[self.capacity,](
                spec.input_buffers.input_ids,
                spec.input_buffers.positions,
                spec.input_buffers.query_start_loc,
                spec.last_token_indices,
                spec.current_draft_step,
                self.frame,
                self.state,
                self.batch_slots,
                self.capacity,
                num_warps=1,
            )
            table = runner.req_states.all_token_ids.gpu
            _advance_frontier[self.capacity,](
                self.frame,
                self.state,
                self.batch_slots,
                runner.req_states.num_computed_tokens.gpu,
                rejected,
                table,
                table.shape[0],
                table.shape[1],
                table.stride(0),
                self.capacity,
                num_warps=1,
            )
