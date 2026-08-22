# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Public policy API for controlling KVCR placement and eviction."""

import logging
from sys import float_info

from .types import (
    BlockMeta,
    CacheTier,
    PlacementAction,
    PlacementDecision,
    PlacementFailure,
)

logger = logging.getLogger(__name__)


class KVCachePolicy:
    """Policy decisions and lifecycle hooks invoked synchronously by KVCR."""

    required_tiers: frozenset[CacheTier] = frozenset()

    # Required core policy.
    def decide_ingest(
        self,
        meta: BlockMeta,
        source: CacheTier,
        required_local: bool,
        router_hints: object | None = None,
        framework_hints: object | None = None,
    ) -> PlacementDecision:
        """Choose placement before KVCR creates a managed residency."""
        raise NotImplementedError

    def eviction_score(
        self,
        meta: BlockMeta,
        source: CacheTier,
    ) -> float:
        """Score an evictable residency; lower values are selected first."""
        raise NotImplementedError

    def decide_eviction(
        self,
        meta: BlockMeta,
        source: CacheTier,
    ) -> PlacementDecision:
        """Choose placement after KVCR selects an eviction candidate."""
        raise NotImplementedError

    # Optional lifecycle hooks for policies that keep state.
    def on_ingest(
        self,
        meta: BlockMeta,
        source: CacheTier,
    ) -> None:
        """Observe a successful first admission into KVCR-managed storage."""
        pass

    def on_remove(self, meta: BlockMeta) -> None:
        """Observe removal of the block's final managed residency."""
        pass

    # Optional recovery override.
    def decide_recovery(
        self,
        meta: BlockMeta,
        failure: PlacementFailure,
    ) -> PlacementDecision:
        """Choose recovery after a policy-requested placement fails."""
        logger.warning(
            "KVCR placement failed (%s); dropping the source residency",
            failure.reason,
        )
        return (PlacementAction.DROP, None)


class FIFOPolicy(KVCachePolicy):
    """Built-in policy preserving KVCR's FIFO eviction behavior."""

    def decide_ingest(
        self,
        meta: BlockMeta,
        source: CacheTier,
        required_local: bool,
        router_hints: object | None = None,
        framework_hints: object | None = None,
    ) -> PlacementDecision:
        return (PlacementAction.KEEP, None)

    def eviction_score(
        self,
        meta: BlockMeta,
        source: CacheTier,
    ) -> float:
        return 0.0

    def decide_eviction(
        self,
        meta: BlockMeta,
        source: CacheTier,
    ) -> PlacementDecision:
        return (PlacementAction.DROP, None)


class G3FIFOPolicy(FIFOPolicy):
    """Built-in FIFO policy that spills local DRAM into configured G3."""

    required_tiers = FIFOPolicy.required_tiers | {CacheTier.G3}

    # Failed G3 moves inherit KVCachePolicy's default warning and DROP recovery.
    def decide_eviction(
        self,
        meta: BlockMeta,
        source: CacheTier,
    ) -> PlacementDecision:
        if source is CacheTier.LOCAL_G2:
            return (PlacementAction.MOVE_TO, CacheTier.G3)
        return super().decide_eviction(meta, source)


class LRUPolicy(FIFOPolicy):
    """Built-in least-recently-used eviction policy."""

    def eviction_score(
        self,
        meta: BlockMeta,
        source: CacheTier,
    ) -> float:
        return meta.last_access if meta.last_access is not None else -float_info.max


class G3LRUPolicy(LRUPolicy):
    """Built-in LRU policy that spills local DRAM into configured G3."""

    required_tiers = LRUPolicy.required_tiers | {CacheTier.G3}

    # Failed G3 moves inherit KVCachePolicy's default warning and DROP recovery.
    def decide_eviction(
        self,
        meta: BlockMeta,
        source: CacheTier,
    ) -> PlacementDecision:
        if source is CacheTier.LOCAL_G2:
            return (PlacementAction.MOVE_TO, CacheTier.G3)
        return super().decide_eviction(meta, source)
