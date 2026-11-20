# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from kitepond.models.glm5next.frontier import (
    CPUFrontierReference,
    digest,
    validate_native_terminal_frontier,
)


def _observed_terminal(*, bad_draft=False):
    # Model runtime passes an accepted speculative query with rejected tail
    # rows and a terminal boundary that falls inside the accepted query.
    tokens = list(range(10, 21))
    origin = 3
    ref = CPUFrontierReference(origin, origin)
    ref.observe([3], [13], [3], [14], 0, 0, 4, tokens)
    ref.observe(
        [4, 5, 6, 7],
        [14, 15, 16, 17],
        [4, 5, 6, 7],
        [15, 30 if bad_draft else 16, 17, 18],
        2,
        1,
        7,
        tokens,
    )
    frame = {
        "native_req_index": 9,
        "target_positions": [4, 5, 6, 7],
        "target_input_ids": [14, 15, 16, 17],
        "draft_prepared": {
            "positions": [4, 5, 6, 7],
            "input_ids": [15, 16, 17, 18],
            "last_token_indices": [102],
            "query_start_loc": [100, 104],
        },
        "after_finalize": {"raw_gpu_computed": [7]},
    }
    snapshot = [
        ref.target_end,
        ref.draft_end,
        int(ref.target_valid),
        int(ref.draft_valid),
        ref.steps,
        origin,
        ref.last_start,
        ref.last_end,
        9,
        3,
        1,
    ]
    intent = SimpleNamespace(
        requested_common_tokens=6,
        prefix_sha256=digest(tokens[:6]),
        next_token_id=tokens[6],
    )
    return snapshot, frame, intent, origin, tokens[:7]


def test_terminal_boundary_uses_accepted_rows_and_shifted_draft_ids():
    target, draft = validate_native_terminal_frontier(*_observed_terminal())
    assert target["frontier"] == draft["frontier"]
    assert target["frontier"]["terminal_common_B"] == 6
    assert target["frontier"]["independent_draft_frontier"] == 7
    assert target["latest_rows"] == [(4, digest([14])), (5, digest([15]))]
    assert draft["latest_shifted_rows"] == [(4, digest([15])), (5, digest([16]))]


def test_valid_latest_draft_does_not_cover_invalid_previous_induction():
    with pytest.raises(ValueError, match="did not cover"):
        validate_native_terminal_frontier(*_observed_terminal(bad_draft=True))


@pytest.mark.parametrize("wrong", ["draft-token", "slot", "native-token", "counter"])
def test_terminal_proof_refuses_inconsistent_native_evidence(wrong):
    snap, frame, intent, origin, tokens = _observed_terminal()
    if wrong == "draft-token":
        frame["draft_prepared"]["input_ids"][1] = 999
    elif wrong == "slot":
        snap[8] = 10
    elif wrong == "native-token":
        tokens[6] = 999
    else:
        frame["after_finalize"]["raw_gpu_computed"] = [6]
    with pytest.raises(ValueError):
        validate_native_terminal_frontier(snap, frame, intent, origin, tokens)
