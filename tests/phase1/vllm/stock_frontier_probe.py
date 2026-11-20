# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the packaged GLM frontier operations on SM120 or in the interpreter."""

import argparse
import json

import torch
from kitepond.models.glm5next import frontier_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    device = parser.parse_args().device
    if device == "cuda" and torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This acceptance probe requires one SM120 device")

    def i64(values):
        return torch.tensor(values, dtype=torch.int64, device=device)

    for good_draft in (True, False):
        state = i64([[4, 4, 1, 1, 1, 3, 3, 4, 0, 1, 1]])
        frame = torch.full((1, 32), -1, dtype=torch.int64, device=device)
        slots = i64([-1])
        positions = i64([4, 5, 6, 7])
        query = i64([0, 4])
        frontier_gpu._capture_target[(1,)](
            i64([14, 15, 16, 17]),
            positions,
            query,
            i64([0]),
            frame,
            state,
            slots,
            1,
            1,
            4,
            2,
            num_warps=1,
        )
        frontier_gpu._capture_draft[(1,)](
            i64([15, 16 if good_draft else 99, 17, 18]),
            positions,
            query,
            i64([2]),
            i64([0]),
            frame,
            state,
            slots,
            1,
            num_warps=1,
        )
        frontier_gpu._advance_frontier[(1,)](
            frame,
            state,
            slots,
            i64([7]),
            i64([1]),
            i64([list(range(10, 21))]),
            1,
            11,
            11,
            1,
            num_warps=1,
        )
        assert state.tolist() == [[7, 7, 1, int(good_draft), 2, 3, 4, 7, 0, 3, 1]]
        assert frame[0, 29].item() == 1

    print(
        json.dumps(
            {
                "device": device,
                "kernel_type": type(frontier_gpu._capture_target).__name__,
                "accepted_frontier_checked": True,
                "shifted_draft_rejection_checked": True,
            }
        )
    )


if __name__ == "__main__":
    main()
