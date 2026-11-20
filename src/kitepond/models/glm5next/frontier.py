# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate actual GLM target and draft acceptance at a terminal boundary."""

import hashlib
import json
from dataclasses import dataclass


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_native_terminal_frontier(
    snapshot, frame, intent, imported_tokens, native_tokens
):
    if (
        not isinstance(snapshot, list)
        or len(snapshot) != 11
        or any(type(v) is not int for v in snapshot)
    ):
        raise ValueError("Invalid native GPU accepted-frontier snapshot")
    (
        target_end,
        draft_end,
        t_valid,
        d_valid,
        steps,
        origin,
        last_start,
        last_end,
        slot,
        accepted,
        active,
    ) = snapshot
    b = intent.requested_common_tokens
    if (
        t_valid != 1
        or d_valid != 1
        or active != 1
        or steps < 1
        or origin != imported_tokens
        or slot != frame["native_req_index"]
        or not imported_tokens <= last_start < b <= last_end
        or target_end != last_end
        or draft_end != last_end
        or not 1 <= accepted <= 4
        or last_end - last_start != accepted
    ):
        raise ValueError("Native target/draft accepted frontier did not cover B")
    if (
        len(native_tokens) != b + 1
        or digest(native_tokens[:b]) != intent.prefix_sha256
        or native_tokens[b] != intent.next_token_id
    ):
        raise ValueError(
            "Native GPU token table differs from the actual Request 0:B/extra input"
        )
    positions = frame["target_positions"]
    inputs = frame["target_input_ids"]
    prepared = frame["draft_prepared"]
    last = prepared["last_token_indices"]
    while isinstance(last, list) and len(last) == 1:
        last = last[0]
    last -= prepared["query_start_loc"][0]
    raw = frame["after_finalize"]["raw_gpu_computed"]
    while isinstance(raw, list) and len(raw) == 1:
        raw = raw[0]
    if raw != last_end:
        raise ValueError("Native completion counter changed after frontier capture")
    if (
        not 1 <= len(positions) <= 4
        or len(inputs) != len(positions)
        or positions[:accepted] != list(range(last_start, last_end))
        or last + 1 != accepted
    ):
        raise ValueError("Latest native frame does not match the accepted frontier")
    target_rows = []
    draft_rows = []
    for row, pos in enumerate(positions[:accepted]):
        if pos >= b:
            continue
        if inputs[row] != native_tokens[pos]:
            raise ValueError("Latest target query differs from canonical Request")
        if (
            prepared["positions"][row] != pos
            or prepared["input_ids"][row] != native_tokens[pos + 1]
        ):
            raise ValueError("Latest MTP shifted query differs from Request token[p+1]")
        target_rows.append((pos, digest([inputs[row]])))
        draft_rows.append((pos, digest([prepared["input_ids"][row]])))
    if (
        not target_rows
        or target_rows[-1][0] != b - 1
        or len(target_rows) != b - last_start
    ):
        raise ValueError("No exact target/draft latest-frame row for B")
    logical = {
        "schema": "glm53f-native-accepted-frontier-v1",
        "imported_tokens": origin,
        "validated_steps": steps,
        "independent_target_frontier": target_end,
        "independent_draft_frontier": draft_end,
        "terminal_common_B": b,
        "last_frame_start": last_start,
        "native_Request_prefix_sha256": intent.prefix_sha256,
    }
    return {"frontier": logical, "latest_rows": target_rows}, {
        "frontier": logical,
        "latest_shifted_rows": draft_rows,
    }


def resolve_native_gdn_row(actual, batch_index):
    """Select the actual GDN spec/non-spec row without inventing a batch layout."""
    if type(batch_index) is not int or batch_index < 0:
        raise ValueError("Invalid native GDN request row")
    masks = actual.spec_sequence_masks_cpu
    row = batch_index
    is_spec = False
    if masks is not None:
        if masks.device.type != "cpu":
            raise ValueError("Native spec mask must be CPU metadata")
        mask = masks.tolist()
        if row >= len(mask):
            raise ValueError("Native GDN mask misses this batch row")
        is_spec = bool(mask[row])
        count = sum(bool(value) for value in mask[:row])
        row = count if is_spec else row - count
    if is_spec:
        indices = actual.spec_state_indices_tensor
        key = "spec_state_indices_tensor"
    else:
        indices = actual.non_spec_state_indices_tensor
        key = "non_spec_state_indices_tensor"
    if indices is None or row >= indices.shape[0]:
        raise ValueError("Actual native GDN state selector is absent")
    return {key: indices[row : row + 1]}


@dataclass
class CPUFrontierReference:
    """CPU boundary oracle; never a GPU/event/readiness receipt."""

    target_end: int
    draft_end: int
    target_valid: bool = True
    draft_valid: bool = True
    steps: int = 0
    last_start: int = 0
    last_end: int = 0

    def observe(
        self,
        target_positions,
        target_ids,
        draft_positions,
        draft_ids,
        last,
        rejected,
        raw_end,
        canonical_tokens,
    ):
        q = len(target_positions)
        count = q - rejected
        self.last_start = self.target_end
        common = 1 <= q <= 4 and 1 <= count <= q
        self.target_valid &= (
            common
            and raw_end == self.target_end + count
            and (target_positions == list(range(self.target_end, self.target_end + q)))
            and (target_positions[:count] == list(range(self.target_end, raw_end)))
            and (target_ids[:count] == canonical_tokens[self.target_end : raw_end])
        )
        self.draft_valid &= (
            common
            and 0 <= last < q
            and (last + 1 == count)
            and (raw_end == self.draft_end + last + 1)
            and (draft_positions[:count] == list(range(self.draft_end, raw_end)))
            and (
                draft_ids[:count] == canonical_tokens[self.draft_end + 1 : raw_end + 1]
            )
        )
        self.target_end = self.draft_end = self.last_end = raw_end
        self.steps += 1
