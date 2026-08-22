# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Internal machinery for invoking policies and ordering eviction candidates."""

import heapq
import logging
import math
from collections.abc import Collection
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


@dataclass(frozen=True)
class _Entry:
    score: float
    sequence: int


class _EvictionQueue:
    def __init__(self) -> None:
        self._heap: list[tuple[float, int, BlockKey]] = []
        self._live: dict[BlockKey, _Entry] = {}
        self._next_sequence = 0

    def __len__(self) -> int:
        return len(self._live)

    def insert(self, key: BlockKey, score: float) -> None:
        entry = _Entry(score, self._next_sequence)
        self._next_sequence += 1
        self._live[key] = entry
        heapq.heappush(self._heap, (entry.score, entry.sequence, key))

    def remove(self, key: BlockKey) -> None:
        self._live.pop(key, None)

    def select(self, excluded: set[BlockKey]) -> BlockKey | None:
        skipped: list[tuple[float, int, BlockKey]] = []
        selected: BlockKey | None = None
        while self._heap:
            score, sequence, key = self._heap[0]
            entry = self._live.get(key)
            if entry != _Entry(score, sequence):
                heapq.heappop(self._heap)
                continue
            if key not in excluded:
                selected = key
                break
            skipped.append(heapq.heappop(self._heap))
        for item in skipped:
            heapq.heappush(self._heap, item)
        return selected
