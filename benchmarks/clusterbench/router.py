# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Synthetic KV router: a global inventory, worker selection and fetch hints.

The inventory keeps, per worker, tier and stream, the length of the
root-aligned run it holds. That matches how the workload uses streams, and it
is all a prefix-overlap router needs. Workers report a tier that can have
gaps through a ``PrefixTracker``.

Selection is sticky: a session stays on its home worker and moves with a
seeded probability per turn, however busy workers are. A migrating session
goes to the worker with the fewest sessions, as any load-aware router would
send it: uniformly random targets pile sessions onto a few workers, whose
caches then thrash while the rest sit idle. The hint rule is Dynamo's: after
selection, name the peer holding the longest root-aligned prefix if that beats
what the chosen worker holds.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import NamedTuple

from .config import RouterConfig
from .ids import INDEX_BITS, INDEX_MASK, Segment
from .protocol import TIER_DEVICE, TIER_G2, TIER_PRIMARY, InventoryDelta

_LOCAL_TIERS = (TIER_DEVICE, TIER_PRIMARY, TIER_G2)
# A peer can only serve what it holds in host memory: its CPU tier, through a
# pin, or its KVCR pool.
_SOURCE_TIERS = (TIER_PRIMARY, TIER_G2)


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


class PrefixTracker:
    """A worker's view of one tier, reported as root-aligned prefix changes.

    ``Inventory`` keeps one prefix length per stream, but a host tier can lose
    a block from the middle of a stream and get it back (a pinned tail, or
    KVCR evicting in access order). The tracker knows every block and reports
    only changes to the true prefix.
    """

    def __init__(self, tier: int) -> None:
        self._tier = tier
        self._blocks: dict[int, set[int]] = {}
        self._prefix: dict[int, int] = {}
        self.count = 0
        self._deltas: list[InventoryDelta] = []

    def take(self) -> list[InventoryDelta]:
        deltas, self._deltas = self._deltas, []
        return deltas

    def apply(self, removed: bool, ids: Sequence[int]) -> None:
        """Block ids that entered or left the tier, in the order they did."""
        if removed:
            for bid in ids:
                self.remove(bid >> INDEX_BITS, bid & INDEX_MASK)
            return
        first = previous = None
        for bid in ids:
            if first is None:
                first = bid
            elif bid != previous + 1 or not bid & INDEX_MASK:
                self._add_run(first, previous)
                first = bid
            previous = bid
        if first is not None:
            self._add_run(first, previous)

    def _add_run(self, first: int, last: int) -> None:
        start = first & INDEX_MASK
        self.add(first >> INDEX_BITS, start, start + last - first + 1)

    def add(self, stream: int, start: int, end: int) -> None:
        blocks = self._blocks.get(stream)
        if blocks is None:
            blocks = self._blocks[stream] = set()
        before = len(blocks)
        blocks.update(range(start, end))
        self.count += len(blocks) - before
        prefix = self._prefix.get(stream, 0)
        if start <= prefix:
            new = max(prefix, end)
            while new in blocks:
                new += 1
            if new > prefix:
                self._prefix[stream] = new
                self._delta(stream, prefix, new, False)

    def remove(self, stream: int, index: int) -> None:
        blocks = self._blocks.get(stream)
        if blocks is None or index not in blocks:
            return
        blocks.remove(index)
        self.count -= 1
        prefix = self._prefix.get(stream, 0)
        if index < prefix:
            self._delta(stream, index, prefix, True)
            if index:
                self._prefix[stream] = index
            else:
                del self._prefix[stream]
        if not blocks:
            del self._blocks[stream]

    def _delta(self, stream: int, start: int, end: int, removed: bool) -> None:
        deltas = self._deltas
        if deltas:
            # A run added or evicted block by block travels as one delta.
            _, last_stream, last_start, last_end, last_removed = deltas[-1]
            if last_stream == stream and last_removed == removed:
                if removed and end == last_start:
                    deltas[-1] = (self._tier, stream, start, last_end, True)
                    return
                if not removed and start == last_end:
                    deltas[-1] = (self._tier, stream, last_start, end, False)
                    return
        deltas.append((self._tier, stream, start, end, removed))


class Route(NamedTuple):
    worker: int
    local_blocks: int  # prefix the chosen worker already holds
    hint_source: int  # worker id, or -1
    hint_blocks: int
    migrated: bool


class Router:
    def __init__(self, config: RouterConfig, nodes: Sequence[str]) -> None:
        self._config = config
        self._nodes = list(nodes)
        self._workers = len(nodes)
        self.inventory = Inventory(self._workers)
        self._home: dict[int, int] = {}
        self._resident = [0] * self._workers  # sessions per worker
        self._rngs: dict[int, random.Random] = {}

    def same_node(self, a: int, b: int) -> bool:
        return self._nodes[a] == self._nodes[b]

    def _session_rng(self, session: int) -> random.Random:
        rng = self._rngs.get(session)
        if rng is None:
            rng = self._rngs[session] = random.Random(session * 40503 + 0x5BD1E995)
        return rng

    def _place(self, session: int) -> tuple[int, bool]:
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
        move, pick = rng.random(), rng.random()
        if self._workers < 2 or move >= self._config.migration_prob:
            return home, False
        # The emptiest other worker anywhere: a session takes turns faster on
        # a quiet worker, so quiet workers shed sessions faster and must be
        # refilled from the whole cluster or busy ones collect everything.
        least = min(count for w, count in enumerate(resident) if w != home)
        emptiest = [
            w for w, count in enumerate(resident) if count == least and w != home
        ]
        # Among equally empty workers, one on another node if there is one.
        nodes = self._nodes
        pool = [w for w in emptiest if nodes[w] != nodes[home]] or emptiest
        target = pool[int(pick * len(pool)) % len(pool)]
        resident[home] -= 1
        resident[target] += 1
        self._home[session] = target
        return target, True

    def route(self, session: int, segments: Sequence[Segment]) -> Route:
        worker, migrated = self._place(session)
        inventory = self.inventory
        local = inventory.prefix(worker, segments, _LOCAL_TIERS)
        source, source_blocks = -1, local
        # To hold a longer prefix a peer must hold the block the chosen worker
        # stops at, so only holders of that block's stream can win.
        covered = 0
        for stream, start, end in segments:
            covered += end - start
            if local < covered:
                for peer in inventory.holders(stream, _SOURCE_TIERS):
                    if peer == worker:
                        continue
                    held = inventory.prefix(peer, segments, _SOURCE_TIERS)
                    if held > source_blocks:  # longest wins, ties to the lower id
                        source, source_blocks = peer, held
                break
        hint_blocks = source_blocks if source >= 0 else 0
        return Route(worker, local, source, hint_blocks, migrated)

    def forget(self, session: int) -> None:
        home = self._home.pop(session, None)
        if home is not None:
            self._resident[home] -= 1
        self._rngs.pop(session, None)
