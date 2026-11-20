# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P's metadata-only view of pages from the native D host allocator."""

from collections import Counter
from collections.abc import Iterable, Sequence

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock

from .host_authority import NativeHostGrant


class BorrowedHostMetadata(BlockPool):
    """A single null block plus D-granted page metadata, and no free KV slots."""

    def __init__(
        self, *, engine_id: str, boot_id: str, capacity: int, hash_block_size: int
    ):
        super().__init__(
            num_gpu_blocks=1,
            enable_caching=False,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=False,
        )
        if not engine_id or not boot_id or type(capacity) is not int or capacity <= 1:
            raise ValueError(
                "Borrowed metadata requires actual D authority and capacity"
            )
        self.engine_id, self.boot_id, self.capacity = engine_id, boot_id, capacity
        self._views: dict[int, KVCacheBlock] = {}
        self._grants: dict[str, NativeHostGrant] = {}
        self._writable: set[int] = set()
        self._readers: Counter[int] = Counter()

    def install(self, grant: NativeHostGrant) -> tuple[KVCacheBlock, ...]:
        """Install the exact native control receipt already admitted by D."""
        if (grant.engine_id, grant.boot_id) != (
            self.engine_id,
            self.boot_id,
        ) or grant.request_id in self._grants:
            raise ValueError("P received a foreign or duplicate host grant")
        ordered = (*grant.readonly_pages, *grant.writable_pages)
        if (
            not ordered
            or len(set(ordered)) != len(ordered)
            or any(type(p) is not int or not 0 < p < self.capacity for p in ordered)
            or not self._writable.isdisjoint(ordered)
            or any(p in self._views for p in grant.writable_pages)
        ):
            raise ValueError("P received invalid or overlapping host grant pages")
        blocks = []
        for p in ordered:
            block = self._views.setdefault(p, KVCacheBlock(p, pool=self))
            block.ref_cnt += 1
            blocks.append(block)
        self._grants[grant.request_id] = grant
        self._writable.update(grant.writable_pages)
        self._readers.update(grant.readonly_pages)
        return tuple(blocks)

    def discard(self, grant: NativeHostGrant) -> None:
        """Forget a settled lease after P's native request refs have drained."""
        if self._grants.get(grant.request_id) is not grant:
            raise ValueError("Foreign or settled P host lease")
        if any(self._views[p].ref_cnt for p in grant.writable_pages):
            raise ValueError("P still holds writable host page references")
        if any(
            self._views[p].ref_cnt >= self._readers[p] for p in grant.readonly_pages
        ):
            raise ValueError("P still holds shared host page references")
        del self._grants[grant.request_id]
        self._writable.difference_update(grant.writable_pages)
        self._readers.subtract(grant.readonly_pages)
        for p in (*grant.readonly_pages, *grant.writable_pages):
            if not self._readers[p]:
                del self._views[p]
                self._readers.pop(p, None)

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        for block in blocks:
            if not block.is_null:
                self._require_live(block)
                block.ref_cnt += 1

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        # D still owns every physical ref. Validate the whole P free before
        # decrementing a local view, even if a native step submits an alias.
        counts: Counter[int] = Counter()
        for block in ordered_blocks:
            if not block.is_null:
                self._require_live(block)
                counts[block.block_id] += 1
        for page_id, count in counts.items():
            block = self._views.get(page_id)
            if block is None or count > block.ref_cnt:
                raise ValueError("Unbalanced P host metadata references")
        for page_id, count in counts.items():
            self._views[page_id].ref_cnt -= count

    def _require_live(self, block: KVCacheBlock) -> None:
        if self._views.get(block.block_id) is not block or block.ref_cnt <= 0:
            raise ValueError("P host metadata reference is foreign or already released")

    def is_block_writable(self, block: KVCacheBlock) -> bool:
        return (
            self._views.get(block.block_id) is block
            and block.ref_cnt == 1
            and block.block_id in self._writable
        )

    def get_num_free_blocks(self) -> int:
        return 0

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        if num_blocks == 0:
            return []
        raise RuntimeError("P may only install D-owned host page IDs")

    def cache_full_blocks(self, *args, **kwargs) -> None:
        # Only D publishes prefix hashes. P has no cached hashes in this pool.
        return None

    def cache_partial_block(self, *args, **kwargs) -> None:
        raise RuntimeError("Only D publishes shared host prefix hashes")
