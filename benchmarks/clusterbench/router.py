# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Synthetic KV router: a global inventory, worker selection and fetch hints.

The inventory keeps, per worker, tier and stream, the length of the
root-aligned run it holds. That matches how the workload uses streams and how
the engines evict (tail first), and it is all a prefix-overlap router needs.

Selection is either "sticky" (session affinity with a seeded migration
probability, independent of how busy workers are) or "cost" (Dynamo's default
KV-router formula). A migrating session goes to the worker with the fewest
sessions, as any load-aware router would send it: uniformly random targets
pile sessions onto a few workers, whose caches then thrash while the rest sit
idle. The hint rule is Dynamo's: after selection, name the peer holding the
longest root-aligned prefix if that beats what the chosen worker holds.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import NamedTuple

from .config import RouterConfig, TierConfig
from .ids import Segment, segment_blocks
from .protocol import TIER_DEVICE, TIER_G2, TIER_PRIMARY, InventoryDelta


class Inventory:
    def __init__(self, workers: int) -> None:
        self._held: list[list[dict[int, int]]] = [
            [{} for _ in range(workers)] for _ in range(3)
        ]
        # Per tier and stream, the workers holding any of it.
        self._holders: list[dict[int, set[int]]] = [{} for _ in range(3)]

    def apply(self, worker: int, deltas: Iterable[InventoryDelta]) -> None:
        for tier, stream, start, end, removed in deltas:
            held = self._held[tier][worker]
            have = held.get(stream, 0)
            if removed:
                if start < have:
                    if start:
                        held[stream] = start
                    else:
                        del held[stream]
                        holders = self._holders[tier][stream]
                        holders.discard(worker)
                        if not holders:
                            del self._holders[tier][stream]
            elif start <= have < end:
                held[stream] = end
                if not have:
                    holders = self._holders[tier].get(stream)
                    if holders is None:
                        holders = self._holders[tier][stream] = set()
                    holders.add(worker)

    def holders(self, stream: int, tiers: Sequence[int]) -> list[int]:
        """Workers holding any of ``stream`` in ``tiers``, lowest id first."""
        found: set[int] = set()
        for tier in tiers:
            found.update(self._holders[tier].get(stream, ()))
        return sorted(found)

    def prefix(
        self, worker: int, segments: Sequence[Segment], tiers: Sequence[int]
    ) -> int:
        """Blocks of the request's root-aligned prefix this worker holds."""
        maps = [self._held[tier][worker] for tier in tiers]
        total = 0
        for stream, start, end in segments:
            have = 0
            for held in maps:
                n = held.get(stream, 0)
                if n > have:
                    have = n
            if have <= start:
                break
            if have >= end:
                total += end - start
                continue
            total += have - start
            break
        return total


class Route(NamedTuple):
    worker: int
    local_blocks: int  # prefix the chosen worker already holds
    hint_source: int  # worker id, or -1
    hint_blocks: int
    migrated: bool


class Router:
    def __init__(
        self, config: RouterConfig, nodes: Sequence[str], tiers: TierConfig
    ) -> None:
        self._config = config
        self._nodes = list(nodes)
        self._workers = len(nodes)
        self._block = tiers.tokens_per_block
        self.inventory = Inventory(self._workers)
        self._local_tiers = (TIER_DEVICE, TIER_PRIMARY, TIER_G2)
        # A peer can only serve what it holds in host memory it can expose.
        source_tiers = []
        if tiers.framework_pinning:
            source_tiers.append(TIER_PRIMARY)
        if tiers.g2_blocks:
            source_tiers.append(TIER_G2)
        self._source_tiers = tuple(source_tiers)
        self._home: dict[int, int] = {}
        self._resident = [0] * self._workers  # sticky mode: sessions per worker
        self._rngs: dict[int, random.Random] = {}
        self._tie_rng = random.Random(config.seed)
        # Cost-mode load: projected active blocks and queued prefill tokens.
        self._active_blocks = [0] * self._workers
        self._prefill_tokens = [0] * self._workers
        self._inflight: dict[int, tuple[int, int, int]] = {}

    def same_node(self, a: int, b: int) -> bool:
        return self._nodes[a] == self._nodes[b]

    def _session_rng(self, session: int) -> random.Random:
        rng = self._rngs.get(session)
        if rng is None:
            seed = (self._config.seed << 32) ^ (session * 40503 + 0x5BD1E995)
            rng = self._rngs[session] = random.Random(seed)
        return rng

    def _sticky(self, session: int, turn: int) -> tuple[int, bool]:
        home = self._home.get(session)
        resident = self._resident
        if home is None:
            # The emptiest worker, starting the search at session % workers.
            workers = self._workers
            home = min(
                range(workers), key=lambda w: (resident[w], (w - session) % workers)
            )
            self._home[session] = home
            resident[home] += 1
            return home, False
        rng = self._session_rng(session)
        # Draw every turn so the stream does not depend on earlier outcomes.
        move, where, pick = rng.random(), rng.random(), rng.random()
        if self._workers < 2 or move >= self._config.migration_prob:
            return home, False
        # The emptiest other worker anywhere: a session takes turns faster on
        # a quiet worker, so quiet workers shed sessions faster and must be
        # refilled from the whole cluster or busy ones collect everything.
        least = min(count for w, count in enumerate(resident) if w != home)
        emptiest = [
            w for w, count in enumerate(resident) if count == least and w != home
        ]
        # Among equally empty workers, same_node_prob picks the node.
        nodes = self._nodes
        on_node = where < self._config.same_node_prob
        pool = [w for w in emptiest if (nodes[w] == nodes[home]) == on_node] or emptiest
        target = pool[int(pick * len(pool)) % len(pool)]
        resident[home] -= 1
        resident[target] += 1
        self._home[session] = target
        return target, True

    def _cost(self, segments: Sequence[Segment], prompt_tokens: int) -> int:
        config = self._config
        request_blocks = segment_blocks(segments)
        best: list[int] = []
        best_cost = float("inf")
        for worker in range(self._workers):
            device = self.inventory.prefix(worker, segments, (TIER_DEVICE,))
            local = self.inventory.prefix(worker, segments, self._local_tiers)
            credit = config.device_credit * device + config.host_credit * max(
                0, local - device
            )
            prefill = (self._prefill_tokens[worker] + prompt_tokens) / self._block
            cost = max(0.0, prefill - credit)
            cost += self._active_blocks[worker] + request_blocks
            if cost < best_cost:
                best, best_cost = [worker], cost
            elif cost == best_cost:
                best.append(worker)
        return best[0] if len(best) == 1 else self._tie_rng.choice(best)

    def route(
        self, session: int, turn: int, segments: Sequence[Segment], prompt_tokens: int
    ) -> Route:
        migrated = False
        if self._config.mode == "cost":
            previous = self._home.get(session)
            worker = self._cost(segments, prompt_tokens)
            migrated = previous is not None and previous != worker
            self._home[session] = worker
        else:
            worker, migrated = self._sticky(session, turn)
        local = self.inventory.prefix(worker, segments, self._local_tiers)
        source, source_blocks = -1, local
        if self._config.hints and self._source_tiers:
            # To hold a longer prefix a peer must hold the block the chosen
            # worker stops at, so only holders of that block's stream can win.
            inventory = self.inventory
            covered = 0
            for stream, start, end in segments:
                covered += end - start
                if local < covered:
                    for peer in inventory.holders(stream, self._source_tiers):
                        if peer == worker:
                            continue
                        held = inventory.prefix(peer, segments, self._source_tiers)
                        if held > source_blocks:  # longest wins, ties to the lower id
                            source, source_blocks = peer, held
                    break
        hint_blocks = source_blocks if source >= 0 else 0
        return Route(worker, local, source, hint_blocks, migrated)

    def dispatched(self, rid: int, route: Route, blocks: int, tokens: int) -> None:
        uncached = max(0, tokens - route.local_blocks * self._block)
        self._inflight[rid] = (route.worker, blocks, uncached)
        self._active_blocks[route.worker] += blocks
        self._prefill_tokens[route.worker] += uncached

    def done(self, rid: int) -> None:
        entry = self._inflight.pop(rid, None)
        if entry is not None:
            worker, blocks, uncached = entry
            self._active_blocks[worker] -= blocks
            self._prefill_tokens[worker] -= uncached

    def forget(self, session: int) -> None:
        home = self._home.pop(session, None)
        if home is not None and self._config.mode != "cost":
            self._resident[home] -= 1
        self._rngs.pop(session, None)
