# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Address contracts for private interleave-4 HiSparse DCP shards."""

from vllm.v1.hisparse.types import SparseKVRowMirror


def localize_row_mirrors(
    mirrors: tuple[SparseKVRowMirror, ...],
    *,
    block_size: int,
    dcp_world_size: int,
    dcp_rank: int,
) -> tuple[SparseKVRowMirror, ...]:
    """Convert scheduler global-page envelopes to one rank's physical rows.

    Source and destination page ids are preserved. The scheduler numbers a
    physical B-row page over B*DCP logical tokens; DMA numbers its local B rows.
    Selecting this rank's interleaved rows makes each interval local-contiguous.
    """
    if dcp_world_size == 1:
        if dcp_rank != 0:
            raise ValueError("DCP1 has only rank zero.")
        return mirrors
    if (
        dcp_world_size != 4
        or not 0 <= dcp_rank < 4
        or block_size <= 0
        or block_size % 4
    ):
        raise ValueError("HiSparse row mirrors require DCP4 and interleave-4 pages.")
    global_block = block_size * dcp_world_size

    def local_prefix(tokens: int) -> int:
        groups, remainder = divmod(tokens, 16)
        return groups * 4 + min(max(remainder - dcp_rank * 4, 0), 4)

    result = []
    for mirror in mirrors:
        destination_block, offset = divmod(mirror.destination_start, global_block)
        if mirror.num_rows <= 0 or offset + mirror.num_rows > global_block:
            raise ValueError("HiSparse row mirror crosses a logical page boundary.")
        source_blocks = []
        for start in mirror.source_starts:
            source_block, source_offset = divmod(start, global_block)
            if start < 0 or source_offset != offset:
                raise ValueError(
                    "HiSparse mirror endpoints have different row offsets."
                )
            source_blocks.append(source_block)
        if mirror.destination_start < 0:
            raise ValueError("HiSparse mirror destination is negative.")
        begin = local_prefix(offset)
        count = local_prefix(offset + mirror.num_rows) - begin
        if count:
            result.append(
                SparseKVRowMirror(
                    source_starts=tuple(
                        block * block_size + begin for block in source_blocks
                    ),
                    destination_start=destination_block * block_size + begin,
                    num_rows=count,
                )
            )
    return tuple(result)
