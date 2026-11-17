# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The engine's own caches, modelled on vLLM's.

Both evict like vLLM's ``LRUCachePolicy``: every block is ranked
``(request rank, -position)``, so the oldest request goes first and, within a
request, the tail goes before the head. What survives is a root-aligned
prefix, which a plain per-block LRU would not give.

``StreamCache`` is the device tier: membership only, tracked per stream.

``PrimaryTier`` is the CPU offload tier (vLLM
``CPUPrimaryTierOffloadingManager``): slots and reference counts, -1 while a
write is in flight, 0 when evictable, positive while something reads the slot.

This is bookkeeping the engine thread pays for on every block, next to the
KVCR calls being measured, so it avoids per-key heaps, tuples and calls.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Sequence
from itertools import islice

from .ids import Segment

MISS, HIT, PENDING = 0, 1, 2
_WRITING = -1

# A key's place in the eviction order is one integer, lowest first:
# ``rank << _SHIFT | (_MASK - position)``.
_SHIFT = 24
_MASK = (1 << _SHIFT) - 1
_MIN_BUCKET_LIMIT = 4096

# (stream, start, end, removed)
StreamEvent = tuple[int, int, int, bool]
# (removed, keys)
KeyEvent = tuple[bool, Sequence[bytes]]


class StreamCache:
    """Which leading blocks of each stream the device holds.

    Requests name whole stream prefixes and a stream only ever grows from its
    root, so one length and one rank per stream say what a per-block cache
    would: the newest request that used a stream ranks all of it. Streams
    named later in a request go before the ones named earlier, each from its
    tail, which is the block-level order.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._length: dict[int, int] = {}
        self._code: dict[int, int] = {}  # rank << 8 | (255 - place in request)
        self._heap: list[tuple[int, int]] = []  # (code, stream), stale ones skipped
        self._blocks = 0
        self._next_rank = 0
        # Ordered residency changes.
        self.events: list[StreamEvent] = []

    def __len__(self) -> int:
        return self._blocks

    def new_rank(self) -> int:
        self._next_rank += 1
        return self._next_rank

    def held(self, stream: int) -> int:
        return self._length.get(stream, 0)

    def prefix(self, segments: Iterable[Segment]) -> int:
        """Leading blocks of the request that are cached."""
        length = self._length
        total = 0
        for stream, start, end in segments:
            held = length.get(stream, 0)
            if held >= end:
                total += end - start
                continue
            if held > start:
                total += held - start
            break
        return total

    def insert(self, segments: Iterable[Segment], rank: int) -> None:
        """The request's blocks are now cached and rank with it."""
        length, codes, heap, events = self._length, self._code, self._heap, self.events
        for place, (stream, start, end) in enumerate(segments):
            held = length.get(stream, 0)
            if start > held:
                continue  # not root-aligned: it could never be found again
            if end > held:
                length[stream] = end
                self._blocks += end - held
                last = events[-1] if events else None
                if last and last[0] == stream and last[2] == held and not last[3]:
                    events[-1] = (stream, last[1], end, False)  # still one run
                else:
                    events.append((stream, held, end, False))
            code = (rank << 8) | (255 - min(place, 255))
            if codes.get(stream, -1) < code:
                codes[stream] = code
                heapq.heappush(heap, (code, stream))
        if self._blocks > self.capacity:
            self._evict()
        if len(heap) > 4 * len(codes) + 1024:
            self._heap = [(code, stream) for stream, code in codes.items()]
            heapq.heapify(self._heap)

    def _evict(self) -> None:
        length, codes, heap = self._length, self._code, self._heap
        over = self._blocks - self.capacity
        while over > 0 and heap:
            code, stream = heap[0]
            if codes.get(stream) != code:
                heapq.heappop(heap)  # the stream was used again, or is gone
                continue
            held = length[stream]
            take = min(over, held)
            if take == held:
                heapq.heappop(heap)
                del length[stream], codes[stream]
            else:
                length[stream] = held - take
            self.events.append((stream, held - take, held, True))
            self._blocks -= take
            over -= take


class PrimaryTier:
    """Slotted host tier with pins. Slot ids index the framework region.

    Evictable keys queue in one list per request rank. The lists are taken
    oldest rank first and sorted only when their turn comes, which is cheap
    because a request adds its keys in position order. An entry is stale once
    its key has left, been pinned or moved to a newer rank; stale entries are
    dropped when they surface.
    """

    def __init__(self, capacity: int, reserved: int = 0) -> None:
        self.capacity = capacity
        self._free = list(range(capacity - 1, reserved - 1, -1))
        self._slot: dict[bytes, int] = {}
        self._ref: dict[bytes, int] = {}
        self._codes: dict[bytes, int] = {}
        self._buckets: dict[int, list[bytes]] = {}
        self._bucket_ranks: list[int] = []  # heap of the ranks in _buckets
        self._victims: list[bytes] = []  # the oldest bucket, next victim last
        self._victim_rank = -1
        self._bucket_limit = _MIN_BUCKET_LIMIT
        self._next_rank = 0
        self._events: list[KeyEvent] = []
        self._evicted: list[bytes] = []  # evictions not yet in _events
        self.evictions = 0
        self.allocation_failures = 0

    def __len__(self) -> int:
        return len(self._slot)

    def new_rank(self) -> int:
        self._next_rank += 1
        return self._next_rank

    def take_events(self) -> list[KeyEvent]:
        """Ordered residency changes since the last call."""
        if self._evicted:
            self._events.append((True, self._evicted))
            self._evicted = []
        events, self._events = self._events, []
        return events

    # ---- eviction order ----------------------------------------------------

    def _bucket(self, rank: int) -> list[bytes]:
        bucket = self._buckets.get(rank)
        if bucket is None:
            if len(self._buckets) >= self._bucket_limit:
                self._compact()
                bucket = self._buckets.get(rank)
                if bucket is not None:
                    return bucket
            bucket = self._buckets[rank] = []
            heapq.heappush(self._bucket_ranks, rank)
        return bucket

    def _refill(self) -> bool:
        """Make ``_victims`` the oldest non-empty bucket; False if none is left."""
        ranks, buckets, get = self._bucket_ranks, self._buckets, self._codes.get
        if self._victims:
            # An older rank became evictable again: it goes first.
            rank = self._victim_rank
            waiting = buckets.get(rank)
            if waiting is None:
                buckets[rank] = self._victims
                heapq.heappush(ranks, rank)
            else:
                waiting.extend(self._victims)
            self._victims = []
        while ranks:
            rank = heapq.heappop(ranks)
            low = rank << _SHIFT
            high = low + _MASK
            live = [key for key in buckets.pop(rank) if low <= get(key, -1) <= high]
            if live:
                # Highest position last, so pop() yields the tail first.
                live.sort(key=self._codes.__getitem__, reverse=True)
                self._victims = live
                self._victim_rank = rank
                return True
        self._victim_rank = -1
        return False

    def _compact(self) -> None:
        """Rebuild the queues from the evictable keys, dropping stale entries."""
        buckets: dict[int, list[bytes]] = {}
        ref = self._ref
        for key, code in self._codes.items():
            if ref[key] == 0:
                rank = code >> _SHIFT
                bucket = buckets.get(rank)
                if bucket is None:
                    bucket = buckets[rank] = []
                bucket.append(key)
        self._buckets = buckets
        self._bucket_ranks[:] = sorted(buckets)
        self._victims = []
        self._victim_rank = -1
        self._bucket_limit = max(_MIN_BUCKET_LIMIT, 2 * len(buckets))

    def _take_slot(self) -> int | None:
        if self._free:
            return self._free.pop()
        codes, ref, ranks = self._codes, self._ref, self._bucket_ranks
        while True:
            victims = self._victims
            if not victims or (ranks and ranks[0] < self._victim_rank):
                if not self._refill():
                    return None
                victims = self._victims
            key = victims.pop()
            code = codes.get(key)
            if code is None or code >> _SHIFT != self._victim_rank or ref[key] != 0:
                continue  # stale entry, or no longer evictable
            slot = self._slot.pop(key)
            del ref[key], codes[key]
            self._evicted.append(key)
            self.evictions += 1
            return slot

    # ---- lookups -----------------------------------------------------------

    def lookup(self, key: bytes) -> int:
        ref = self._ref.get(key)
        if ref is None:
            return MISS
        return PENDING if ref == _WRITING else HIT

    def slot(self, key: bytes) -> int:
        return self._slot[key]

    def readable(self, keys: Iterable[bytes]) -> list[bytes]:
        """The keys that are present and not being written."""
        get = self._ref.get
        return [key for key in keys if get(key, _WRITING) != _WRITING]

    def scan(self, keys: Sequence[bytes], start: int, rank: int) -> tuple[int, bool]:
        """A request's lookup walks ``keys[start:]`` until one is missing.

        Returns where it stopped and whether any key on the way is still being
        written. Keys it found now rank with the request (the CPU tier records
        the access at lookup), so the request's own promotions cannot evict
        what it just found.
        """
        ref, codes = self._ref, self._codes
        code = ((rank << _SHIFT) | _MASK) - start
        pending = False
        bucket = None
        stop = start
        for key in islice(keys, start, None):
            state = ref.get(key)
            if state is None:
                break
            if state == _WRITING:
                pending = True
            elif codes[key] < code:
                codes[key] = code
                if state == 0:
                    if bucket is None:
                        bucket = self._bucket(rank)
                    bucket.append(key)
            code -= 1
            stop += 1
        return stop, pending

    # ---- writes ------------------------------------------------------------

    def insert_reserved(self, key: bytes, slot: int) -> None:
        """A permanently pinned entry in a reserved slot (never evicted)."""
        self._slot[key] = slot
        self._ref[key] = 1
        self._codes[key] = 0

    def reserve(self, key: bytes, rank: int, position: int) -> int | None:
        """A slot for one absent key, write in flight; None if the tier is full."""
        slot = self._free.pop() if self._free else self._take_slot()
        if slot is None:
            self.allocation_failures += 1
            return None
        self._slot[key] = slot
        self._ref[key] = _WRITING
        self._codes[key] = ((rank << _SHIFT) | _MASK) - position
        return slot

    def prepare_write(
        self, keys: Iterable[bytes], rank: int, positions: Iterable[int]
    ) -> list[tuple[bytes, int]] | None:
        """Allocate slots for keys not yet present; None if the tier is full.

        Nothing is allocated on failure, like vLLM's all-or-nothing store.
        """
        taken: list[tuple[bytes, int]] = []
        ref, codes, slots, free = self._ref, self._codes, self._slot, self._free
        base = (rank << _SHIFT) | _MASK
        for key, position in zip(keys, positions):
            if key in ref:
                continue
            slot = free.pop() if free else self._take_slot()
            if slot is None:
                for done, done_slot in taken:
                    del slots[done], ref[done], codes[done]
                    free.append(done_slot)
                self.allocation_failures += 1
                return None
            slots[key] = slot
            ref[key] = _WRITING
            codes[key] = base - position
            taken.append((key, slot))
        return taken

    def complete_write(
        self, keys: Iterable[bytes], success: bool, readers: int = 0
    ) -> None:
        """Finish in-flight writes; on success leave ``readers`` references."""
        ref, codes = self._ref, self._codes
        if not success:
            slots, free = self._slot, self._free
            for key in keys:
                if ref.get(key) == _WRITING:
                    free.append(slots.pop(key))
                    del ref[key], codes[key]
            return
        done: list[bytes] = []
        if readers:
            for key in keys:
                if ref.get(key) == _WRITING:
                    ref[key] = readers
                    done.append(key)
        else:
            bucket: list[bytes] = []
            bucket_rank = -1
            for key in keys:
                if ref.get(key) != _WRITING:
                    continue
                ref[key] = 0
                rank = codes[key] >> _SHIFT
                if rank != bucket_rank:
                    bucket = self._bucket(rank)
                    bucket_rank = rank
                bucket.append(key)
                done.append(key)
        if done:
            if self._evicted:
                self._events.append((True, self._evicted))
                self._evicted = []
            self._events.append((False, done))

    def prepare_read(self, keys: Iterable[bytes]) -> None:
        ref = self._ref
        for key in keys:
            ref[key] += 1

    def complete_read(self, keys: Iterable[bytes]) -> None:
        ref, codes = self._ref, self._codes
        bucket: list[bytes] = []
        bucket_rank = -1
        for key in keys:
            count = ref.get(key)
            if count is None or count <= 0:
                continue
            ref[key] = count - 1
            if count == 1:
                rank = codes[key] >> _SHIFT
                if rank != bucket_rank:
                    bucket = self._bucket(rank)
                    bucket_rank = rank
                bucket.append(key)

    def access(self, keys: Sequence[bytes], rank: int, first_position: int) -> None:
        """A request touched these keys: raise their rank (vLLM on_request_access)."""
        codes, ref = self._codes, self._ref
        append = self._bucket(rank).append
        code = ((rank << _SHIFT) | _MASK) - first_position
        for key in keys:
            old = codes.get(key)
            if old is not None and old < code:
                codes[key] = code
                if ref[key] == 0:
                    append(key)
            code -= 1

    def occupancy(self) -> float:
        return len(self._slot) / self.capacity

    def fill_level(self) -> float:
        """How full the tier has got: 1.0 once it has had to evict.

        The instantaneous occupancy dips whenever a failed load hands its
        slots back, so it does not say whether the fill phase was long enough.
        """
        return 1.0 if self.evictions else self.occupancy()
