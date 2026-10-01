# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Internal machinery for invoking policies and ordering eviction candidates."""

import heapq
import logging
import math
from collections import Counter
from collections.abc import Collection, Generator, Iterable
from dataclasses import dataclass

from .policy import KVCachePolicy
from .types import (
    BlockKey,
    BlockMeta,
    CacheTier,
    PlacementAction,
    PlacementDecision,
    PlacementFailure,
)

logger = logging.getLogger(__name__)


class _PolicyInvoker:
    def __init__(
        self,
        policy: KVCachePolicy,
        allowed_eviction_moves: Collection[tuple[CacheTier, CacheTier]] = (),
    ) -> None:
        self._policy = policy
        self._allowed_eviction_moves = frozenset(allowed_eviction_moves)

    def eviction_score(
        self,
        meta: BlockMeta,
        source: CacheTier,
        previous_score: float | None = None,
    ) -> float | None:
        try:
            score = self._policy.eviction_score(meta, source)
        except Exception:
            logger.warning("KVCR eviction_score failed", exc_info=True)
            return previous_score
        if type(score) not in (int, float) or not math.isfinite(score):
            logger.warning("KVCR eviction_score returned invalid score %r", score)
            return previous_score
        return float(score)

    def decide_ingest(
        self,
        meta: BlockMeta,
        source: CacheTier,
        required_local: bool,
        router_hints: object | None = None,
        framework_hints: object | None = None,
    ) -> PlacementDecision:
        try:
            decision = self._policy.decide_ingest(
                meta,
                source,
                required_local,
                router_hints,
                framework_hints,
            )
        except Exception:
            logger.warning("KVCR decide_ingest failed", exc_info=True)
            return (PlacementAction.KEEP, None)
        # TODO(kvcr-g3): Wire ingest-time COPY_TO and MOVE_TO placement.
        if decision not in (
            (PlacementAction.KEEP, None),
            (PlacementAction.DROP, None),
        ):
            logger.warning(
                "KVCR decide_ingest returned invalid decision %r",
                decision,
            )
            return (PlacementAction.KEEP, None)
        if required_local and decision[0] is PlacementAction.DROP:
            logger.warning("KVCR decide_ingest cannot drop required-local work")
            return (PlacementAction.KEEP, None)
        return decision

    def decide_eviction(
        self,
        meta: BlockMeta,
        source: CacheTier,
    ) -> PlacementDecision:
        try:
            decision = self._policy.decide_eviction(meta, source)
        except Exception:
            logger.warning("KVCR decide_eviction failed", exc_info=True)
            return (PlacementAction.KEEP, None)
        if decision in (
            (PlacementAction.KEEP, None),
            (PlacementAction.DROP, None),
        ):
            return decision
        if (
            isinstance(decision, tuple)
            and len(decision) == 2
            and decision[0] is PlacementAction.MOVE_TO
            and isinstance(decision[1], CacheTier)
            and (source, decision[1]) in self._allowed_eviction_moves
        ):
            return decision
        logger.warning(
            "KVCR decide_eviction returned invalid decision %r",
            decision,
        )
        return (PlacementAction.KEEP, None)

    def decide_recovery(
        self,
        meta: BlockMeta,
        failure: PlacementFailure,
    ) -> PlacementDecision:
        try:
            decision = self._policy.decide_recovery(meta, failure)
        except Exception:
            logger.warning("KVCR decide_recovery failed", exc_info=True)
            return (PlacementAction.DROP, None)
        if decision != (PlacementAction.DROP, None):
            logger.warning(
                "KVCR decide_recovery returned unsupported decision %r",
                decision,
            )
            return (PlacementAction.DROP, None)
        return decision

    def on_ingest(
        self,
        meta: BlockMeta,
        source: CacheTier,
    ) -> None:
        try:
            self._policy.on_ingest(meta, source)
        except Exception:
            logger.warning("KVCR on_ingest failed", exc_info=True)

    def on_remove(self, meta: BlockMeta) -> None:
        try:
            self._policy.on_remove(meta)
        except Exception:
            logger.warning("KVCR on_remove failed", exc_info=True)

    def on_align_sequence(
        self, blocks: list[BlockMeta], use_current_time: bool
    ) -> None:
        try:
            self._policy.on_align_sequence(blocks, use_current_time)
        except Exception:
            logger.warning("KVCR on_align_sequence failed", exc_info=True)


@dataclass(frozen=True)
class _Entry:
    score: float
    sequence: int
    pools: tuple[str, ...]


class _EvictionQueue:
    def __init__(self) -> None:
        self._heaps: dict[str, list[tuple[float, int, BlockKey]]] = {}
        self._pool_sizes: Counter[str] = Counter()
        self._live: dict[BlockKey, _Entry] = {}
        self._next_sequence = 0

    def __len__(self) -> int:
        return len(self._live)

    def score(self, key: BlockKey) -> float | None:
        entry = self._live.get(key)
        return entry.score if entry is not None else None

    def insert(self, key: BlockKey, score: float, pools: Iterable[str] = ("",)) -> bool:
        """Refresh a score, returning whether the key newly became evictable."""
        pools = tuple(sorted(set(pools)))
        previous = self._live.get(key)
        if previous is not None and previous.score == score and previous.pools == pools:
            return False
        if previous is not None:
            self._pool_sizes.subtract(previous.pools)
        self._pool_sizes.update(pools)
        entry = _Entry(score, self._next_sequence, pools)
        self._next_sequence += 1
        self._live[key] = entry
        item = (entry.score, entry.sequence, key)
        for pool in pools:
            heap = self._heaps.setdefault(pool, [])
            heapq.heappush(heap, item)
            if len(heap) > 2 * self._pool_sizes[pool]:
                # Stale entries amortize this O(n) pass within each pool.
                # Compact queued entries only; candidates() restores held ones.
                heap[:] = [
                    item
                    for item in heap
                    if (current := self._live.get(item[2])) is not None
                    and current.sequence == item[1]
                ]
                heapq.heapify(heap)
        return previous is None

    def remove(self, key: BlockKey) -> bool:
        previous = self._live.pop(key, None)
        if previous is not None:
            self._pool_sizes.subtract(previous.pools)
        return previous is not None

    def candidates(
        self, excluded: set[BlockKey], pools: Collection[str] | None = None
    ) -> Generator[BlockKey, None, None]:
        """Visit selected pools in global score order; close to restore entries.

        The caller may shrink pools as deficits are satisfied. A multi-pool key
        is yielded only once, with the same sequence tie-break as a single pool.
        """
        pools = self._heaps if pools is None else pools
        excluded = set(excluded)
        skipped: list[tuple[str, tuple[float, int, BlockKey]]] = []
        try:
            while heads := [
                (heap[0], pool) for pool in pools if (heap := self._heaps.get(pool))
            ]:
                _, pool = min(heads)
                item = heapq.heappop(self._heaps[pool])
                score, sequence, key = item
                current = self._live.get(key)
                if current is None or current.sequence != sequence:
                    continue
                skipped.append((pool, item))
                if key not in excluded:
                    excluded.add(key)
                    yield key
        finally:
            for pool, item in skipped:
                current = self._live.get(item[2])
                if current is not None and current.sequence == item[1]:
                    heapq.heappush(self._heaps[pool], item)
