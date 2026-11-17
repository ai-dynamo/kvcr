# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The SGLang profile's device cache: a radix tree over real page slots.

Modelled on SGLang's unified radix cache with an external linker
(``unified_tree_core.py`` at the linker PR's revision), reduced to what the
workload can produce. Requests name whole stream prefixes, a stream only grows
at its end and a compaction starts a new stream, so a tree path is a chain of
nodes per stream and nodes never split:

- An insert appends one node per stream for the pages the tree lacks.
- A node is ``stored`` once KVCR holds every page of it and ``pending`` while
  an offload is in flight.
- A lock covers a node and everything it hangs from, as SGLang's
  ``inc_lock_ref`` walks to the root: a running request locks its path, a
  pending offload locks its node's ancestry.
- Eviction removes whole unlocked leaves, least recently used first. A leaf
  is the last node of a stream that no other stream continues from. What is
  unlocked is exactly what eviction can reach, so an allocation that cannot
  be met is refused before anything is evicted (SGLang's ``NO_TOKEN``).
- KVCR's copy outlives the node: evicting a node says nothing to the linker.

Slots index the device regions KVCR reads and writes, so they are real memory.
"""

from __future__ import annotations

import heapq
from bisect import bisect_right
from collections.abc import Iterable, Sequence

from .ids import Segment
from .primary_tier import StreamEvent


class Node:
    __slots__ = (
        "stream", "start", "end", "slots", "stored", "pending", "locks", "access",
    )  # fmt: skip

    def __init__(self, stream: int, start: int, slots: list[int], access: int) -> None:
        self.stream = stream
        self.start = start
        self.end = start + len(slots)
        self.slots = slots
        self.stored = False
        self.pending = False
        self.locks = 0
        self.access = access

    def __repr__(self) -> str:
        return f"Node({self.stream}, {self.start}:{self.end})"


class DeviceTree:
    def __init__(self, capacity: int, reserved: int = 0) -> None:
        self.capacity = capacity
        self._reserved = reserved
        # pop() hands out the lowest free slot.
        self._free: list[int] = list(range(capacity - 1, reserved - 1, -1))
        self._chains: dict[int, list[Node]] = {}
        self._parent: dict[int, int] = {}
        self._dependents: dict[int, int] = {}
        self._heap: list[tuple[int, int]] = []  # (access, stream), stale ones skipped
        self._tick = 0
        self._evictable = 0  # pages in unlocked nodes
        # Ordered residency changes.
        self.events: list[StreamEvent] = []
        self.evictions = 0
        self.evicted_pages = 0
        self.allocation_failures = 0
        self.duplicate_pages = 0

    def __len__(self) -> int:
        return self.capacity - self._reserved - len(self._free)

    def fill_level(self) -> float:
        usable = self.capacity - self._reserved
        return len(self) / usable if usable else 0.0

    def held(self, stream: int) -> int:
        chain = self._chains.get(stream)
        return chain[-1].end if chain else 0

    def prefix(self, segments: Iterable[Segment]) -> int:
        """Leading pages of the request that are cached."""
        chains = self._chains
        total = 0
        for stream, start, end in segments:
            chain = chains.get(stream)
            held = chain[-1].end if chain else 0
            if held >= end:
                total += end - start
                continue
            if held > start:
                total += held - start
            break
        return total

    def path(self, segments: Iterable[Segment], pages: int) -> list[Node]:
        """The nodes holding the request's first ``pages`` pages, root first."""
        nodes: list[Node] = []
        chains = self._chains
        for stream, start, end in segments:
            if pages <= 0:
                break
            last = min(end, start + pages)
            for node in chains.get(stream, ()):
                if node.start >= last:
                    break
                if node.end > start:
                    nodes.append(node)
            pages -= end - start
        return nodes

    def slots(self, segments: Iterable[Segment], first: int, pages: int) -> list[int]:
        """Slots of request pages ``[first, first + pages)``, which are cached."""
        out: list[int] = []
        chains = self._chains
        offset = 0
        last = first + pages
        for stream, start, end in segments:
            length = end - start
            low = max(first, offset) - offset + start
            high = min(last, offset + length) - offset + start
            if low < high:
                for node in chains[stream]:
                    if node.end <= low:
                        continue
                    if node.start >= high:
                        break
                    a, b = max(low, node.start), min(high, node.end)
                    out.extend(node.slots[a - node.start : b - node.start])
            offset += length
            if offset >= last:
                break
        return out

    # ---- recency and locks -------------------------------------------------

    def touch(self, nodes: Sequence[Node]) -> None:
        """A request used these nodes: they are the most recently used."""
        self._tick += 1
        tick = self._tick
        heap = self._heap
        seen = -1
        for node in nodes:
            node.access = tick
            if node.stream != seen:
                seen = node.stream
                heapq.heappush(heap, (tick, seen))
        if len(heap) > 4 * len(self._chains) + 1024:
            self._heap = [(chain[-1].access, s) for s, chain in self._chains.items()]
            heapq.heapify(self._heap)

    def lock(self, nodes: Iterable[Node]) -> None:
        for node in nodes:
            if not node.locks:
                self._evictable -= node.end - node.start
            node.locks += 1

    def unlock(self, nodes: Iterable[Node]) -> None:
        for node in nodes:
            node.locks -= 1
            if not node.locks:
                self._evictable += node.end - node.start

    def ancestry(self, node: Node) -> list[Node]:
        """The node and every node it hangs from, root first."""
        out: list[Node] = []
        stream = node.stream
        chain = self._chains.get(stream, ())
        own = [n for n in chain if n.start <= node.start]
        if not own or own[-1] is not node:
            return [node]  # not in the tree
        parent = self._parent.get(stream)
        lineage = []
        while parent is not None:
            lineage.append(parent)
            parent = self._parent.get(parent)
        for ancestor in reversed(lineage):
            out.extend(self._chains.get(ancestor, ()))
        out.extend(own)
        return out

    def evictable(self) -> int:
        return self._evictable

    # ---- slots -------------------------------------------------------------

    def allocate(self, pages: int) -> list[int] | None:
        """``pages`` free slots, evicting leaves to make room; None if it cannot."""
        free = self._free
        if len(free) + self._evictable < pages:
            self.allocation_failures += 1
            return None
        while len(free) < pages:
            if not self._evict_leaf():  # unreachable while the accounting holds
                self.allocation_failures += 1
                return None
        if not pages:
            return []
        taken = free[-pages:]
        del free[-pages:]
        taken.reverse()
        return taken

    def release(self, slots: Iterable[int]) -> None:
        self._free.extend(slots)

    def _evict_leaf(self) -> bool:
        heap, chains, dependents = self._heap, self._chains, self._dependents
        skipped: list[tuple[int, int]] = []
        evicted = False
        while heap:
            access, stream = heapq.heappop(heap)
            chain = chains.get(stream)
            if not chain or chain[-1].access != access:
                continue  # used again since, or gone
            if dependents.get(stream):
                continue  # not a leaf; re-queued when its last dependent goes
            node = chain[-1]
            if node.locks:
                skipped.append((access, stream))
                continue
            chain.pop()
            self._evictable -= node.end - node.start
            self._free.extend(node.slots)
            self.events.append((stream, node.start, node.end, True))
            self.evictions += 1
            self.evicted_pages += node.end - node.start
            if chain:
                heapq.heappush(heap, (chain[-1].access, stream))
            else:
                del chains[stream]
                parent = self._parent.pop(stream, None)
                if parent is not None:
                    dependents[parent] -= 1
                    if not dependents[parent]:
                        del dependents[parent]
                        rest = chains.get(parent)
                        if rest:
                            heapq.heappush(heap, (rest[-1].access, parent))
            evicted = True
            break
        for entry in skipped:
            heapq.heappush(heap, entry)
        return evicted

    # ---- inserts -----------------------------------------------------------

    def insert(
        self, stream: int, start: int, slots: list[int], parent: int | None = None
    ) -> Node | None:
        """Append pages ``[start, start + len(slots))`` to the stream.

        Pages the stream already holds (two requests computed the same prefix)
        give their slots back; pages past its end cannot be found again and are
        dropped. Returns the new node, if any page was new.
        """
        chain = self._chains.get(stream)
        held = chain[-1].end if chain else 0
        if start > held or not slots:
            self._free.extend(slots)
            return None
        if start < held:
            overlap = min(held - start, len(slots))
            self.duplicate_pages += overlap
            self._free.extend(slots[:overlap])
            slots = slots[overlap:]
            start += overlap
            if not slots:
                return None
        self._tick += 1
        node = Node(stream, start, slots, self._tick)
        if chain is None:
            self._chains[stream] = [node]
            if parent is not None:
                self._parent[stream] = parent
                self._dependents[parent] = self._dependents.get(parent, 0) + 1
        else:
            chain.append(node)
        heapq.heappush(self._heap, (node.access, stream))
        self._evictable += node.end - node.start
        events = self.events
        last = events[-1] if events else None
        if last and last[0] == stream and last[2] == start and not last[3]:
            events[-1] = (stream, last[1], node.end, False)  # still one run
        else:
            events.append((stream, start, node.end, False))
        return node

    def node_at(self, stream: int, index: int) -> Node | None:
        chain = self._chains.get(stream)
        if not chain:
            return None
        position = bisect_right(chain, index, key=_start) - 1
        if position < 0:
            return None
        node = chain[position]
        return node if index < node.end else None

    def take_events(self) -> list[StreamEvent]:
        events, self.events = self.events, []
        return events


def _start(node: Node) -> int:
    return node.start
