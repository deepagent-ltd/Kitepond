# SPDX-License-Identifier: Apache-2.0
"""Provide exclusive, page-aligned host backing for single-node DP8 HiSparse."""

from __future__ import annotations

import fcntl
import json
import logging
import mmap
import os
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from vllm.v1.hisparse.runtime import HiSparseHostMemoryLease

POLICY_ENV = "KITEPOND_HISPARSE_HOST_POOL_POLICY"
DP_SIZE = 8
logger = logging.getLogger(__name__)


def read_policy(path: Path) -> tuple[Path, int, int]:
    """Resolve one policy file and validate its per-rank and aggregate budgets."""
    path = path.resolve(strict=True)
    with path.open(encoding="utf-8") as file:
        budgets = json.load(file)
    if not isinstance(budgets, dict) or budgets.keys() != {
        "rank_bytes",
        "total_bytes",
    }:
        raise ValueError("HiSparse host policy requires rank_bytes and total_bytes")

    rank_bytes = budgets["rank_bytes"]
    total_bytes = budgets["total_bytes"]
    if (
        type(rank_bytes) is not int
        or rank_bytes < mmap.PAGESIZE
        or rank_bytes % mmap.PAGESIZE
        or type(total_bytes) is not int
        or DP_SIZE * rank_bytes > total_bytes
    ):
        raise ValueError(
            "Eight positive page-aligned rank_bytes budgets must fit in total_bytes"
        )
    return path, rank_bytes, total_bytes


def _unlock_rank(fds: list[int]) -> None:
    for fd in reversed(fds):
        os.close(fd)


def _reserve_rank(path: Path, rank: int) -> list[int]:
    # Honor both legacy lock names. One policy budgets one set of eight
    # slots, even when workers use different provider entry points.
    fds: list[int] = []
    try:
        for suffix in (f"rank{rank}", f"dcp-rank{rank}"):
            lock_path = path.with_name(f"{path.name}.{suffix}.lock")
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            fds.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        _unlock_rank(fds)
        raise
    return fds


def _acquire_rank_backing(
    *, instance_id: str, rank: int, size: int, policy_env: str, owner: str
) -> tuple[torch.Tensor, Callable[[], None]]:
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError(f"HiSparse {owner} backing requires a non-empty instance_id")

    value = os.environ.get(policy_env)
    if not value or not Path(value).is_absolute():
        raise ValueError(f"{policy_env} must name an absolute policy file")
    policy, rank_bytes, total_bytes = read_policy(Path(value))
    mapped = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
    if mapped > rank_bytes:
        raise ValueError(
            f"HiSparse {owner} rank {rank} requested {mapped} mapped bytes "
            f"with a {rank_bytes}-byte rank budget"
        )

    import torch

    try:
        fds = _reserve_rank(policy, rank)
    except BlockingIOError as error:
        raise RuntimeError(
            f"HiSparse {owner} rank {rank} already holds a host lease"
        ) from error

    try:
        region: mmap.mmap | None = mmap.mmap(
            -1, mapped, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
        )
        backing = torch.frombuffer(region, dtype=torch.int8)
    except BaseException:
        _unlock_rank(fds)
        raise

    # Torch storage retains the mmap, so any live view also retains the locks.
    def release_rank() -> None:
        _unlock_rank(fds)
        logger.info(
            "HiSparse %s host backing released: instance=%s rank=%d mmap_bytes=%d",
            owner,
            instance_id,
            rank,
            mapped,
        )

    reservation = weakref.finalize(region, release_rank)
    reservation.atexit = False

    def release() -> None:
        nonlocal region
        region = None

    logger.info(
        "HiSparse %s host backing acquired: instance=%s rank=%d "
        "requested_bytes=%d mmap_bytes=%d rank_budget=%d total_budget=%d policy=%s",
        owner,
        instance_id,
        rank,
        size,
        mapped,
        rank_bytes,
        total_bytes,
        policy,
    )
    return backing, release


def acquire(
    *, instance_id: str, dp_rank: int, size_bytes: int
) -> HiSparseHostMemoryLease:
    """Reserve one RAM region in the configured DP8 deployment.

    vLLM registers the pages and calls release after unregistering them.
    A surviving tensor view keeps the reservation until its storage dies.
    """
    if type(dp_rank) is not int or not 0 <= dp_rank < DP_SIZE:
        raise ValueError("HiSparse host backing requires a DP rank from 0 through 7")
    if type(size_bytes) is not int or size_bytes <= 0 or size_bytes % mmap.PAGESIZE:
        raise ValueError("HiSparse host backing size must be positive and page-aligned")

    from vllm.v1.hisparse.runtime import HiSparseHostMemoryLease

    backing, release = _acquire_rank_backing(
        instance_id=instance_id,
        rank=dp_rank,
        size=size_bytes,
        policy_env=POLICY_ENV,
        owner="DP",
    )
    return HiSparseHostMemoryLease(buffer=backing, release=release)
