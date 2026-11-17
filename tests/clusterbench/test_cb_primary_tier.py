# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from clusterbench.primary_tier import HIT, MISS, PENDING, PrimaryTier, StreamCache


def _keys(prefix, count):
    return [f"{prefix}{i:03d}".encode() for i in range(count)]


def _events(tier):
    """Residency changes since the last call, one (key, removed) per key."""
    return [(key, removed) for removed, keys in tier.take_events() for key in keys]


def _store(tier, keys, rank, start=0):
    taken = tier.prepare_write(keys, rank, range(start, start + len(keys)))
    assert taken is not None
    tier.complete_write([key for key, _ in taken], True)
    return taken


def test_write_is_pending_until_completed_and_failure_frees_the_slot():
    tier = PrimaryTier(4)
    (key,) = _keys("a", 1)
    ((_, slot),) = tier.prepare_write([key], tier.new_rank(), [0])
    assert tier.lookup(key) == PENDING
    tier.complete_write([key], False)
    assert tier.lookup(key) == MISS and len(tier) == 0
    ((_, again),) = tier.prepare_write([key], tier.new_rank(), [0])
    assert again == slot
    tier.complete_write([key], True)
    assert tier.lookup(key) == HIT
    assert _events(tier) == [(key, False)]


def test_eviction_takes_the_oldest_request_tail_first():
    tier = PrimaryTier(8)
    old, new = _keys("old", 4), _keys("new", 4)
    _store(tier, old, tier.new_rank())
    _store(tier, new, tier.new_rank())
    tier.take_events()
    _store(tier, _keys("x", 2), tier.new_rank())
    assert _events(tier)[:2] == [(old[3], True), (old[2], True)]
    assert tier.lookup(old[0]) == tier.lookup(old[1]) == HIT  # a prefix survives


def test_access_by_a_newer_request_protects_keys():
    tier = PrimaryTier(4)
    shared, mine = _keys("s", 2), _keys("m", 2)
    _store(tier, shared, tier.new_rank())
    _store(tier, mine, tier.new_rank())
    tier.access(shared, tier.new_rank(), 0)  # a later request reuses them
    _store(tier, _keys("x", 2), tier.new_rank())
    assert [tier.lookup(k) for k in shared] == [HIT, HIT]
    assert [tier.lookup(k) for k in mine] == [MISS, MISS]


def test_pinned_and_pending_entries_are_not_evicted():
    tier = PrimaryTier(4)
    keys = _keys("k", 4)
    _store(tier, keys[:3], tier.new_rank())
    tier.prepare_write(keys[3:], tier.new_rank(), [3])  # write in flight
    tier.prepare_read(keys[:3])
    assert tier.prepare_write(_keys("x", 1), tier.new_rank(), [0]) is None
    assert tier.allocation_failures == 1
    tier.complete_read(keys[2:3])
    taken = tier.prepare_write(_keys("x", 1), tier.new_rank(), [0])
    assert taken is not None and tier.lookup(keys[2]) == MISS


def test_fill_level_stays_full_once_the_tier_has_evicted():
    tier = PrimaryTier(4)
    _store(tier, _keys("a", 2), tier.new_rank())
    assert tier.fill_level() == tier.occupancy() == 0.5
    _store(tier, _keys("b", 3), tier.new_rank())  # evicts one of the first two
    assert tier.evictions == 1 and tier.occupancy() == 1.0
    # A failed load hands its slots back; the tier has still been full.
    taken = tier.prepare_write(_keys("c", 2), tier.new_rank(), [0, 1])
    tier.complete_write([key for key, _ in taken], False)
    assert tier.occupancy() == 0.5 and tier.fill_level() == 1.0


def test_failed_allocation_takes_nothing():
    tier = PrimaryTier(2)
    held = _keys("h", 2)
    _store(tier, held, tier.new_rank())
    tier.prepare_read(held[:1])
    before = len(tier)
    assert tier.prepare_write(_keys("x", 2), tier.new_rank(), [0, 1]) is None
    assert len(tier) == before - 1  # the evictable one was evicted, nothing added
    assert all(tier.lookup(k) == MISS for k in _keys("x", 2))


def test_present_keys_are_skipped_by_prepare_write():
    tier = PrimaryTier(4)
    keys = _keys("k", 3)
    _store(tier, keys[:2], tier.new_rank())
    taken = tier.prepare_write(keys, tier.new_rank(), range(3))
    assert [key for key, _ in taken] == keys[2:]


def test_reserved_slots_are_never_handed_out():
    tier = PrimaryTier(4, reserved=2)
    tier.insert_reserved(b"warm", 0)
    taken = _store(tier, _keys("k", 2), tier.new_rank())
    assert sorted(slot for _, slot in taken) == [2, 3]
    tier.prepare_read([b"warm"])
    tier.complete_read([b"warm"])
    _store(tier, _keys("x", 2), tier.new_rank())
    assert tier.lookup(b"warm") == HIT and tier.slot(b"warm") == 0


def test_a_lookup_hit_protects_the_key_from_its_own_requests_promotions():
    tier = PrimaryTier(4)
    old = _keys("o", 4)
    _store(tier, old, tier.new_rank())
    mine = tier.new_rank()
    # The request's scan finds the first block and stops at the one it lacks.
    request = [old[0], *_keys("p", 3)]
    assert tier.scan(request, 0, mine) == (1, False)
    # It now promotes three more blocks: the untouched ones go.
    for position, key in enumerate(request[1:], 1):
        assert tier.reserve(key, mine, position) is not None
    assert tier.lookup(old[0]) == HIT
    assert [tier.lookup(k) for k in old[1:]] == [MISS, MISS, MISS]
    assert tier.scan(request, 0, mine) == (4, True)  # promotions in flight


def test_reserve_fails_when_nothing_can_be_evicted():
    tier = PrimaryTier(2)
    held = _keys("h", 2)
    _store(tier, held, tier.new_rank())
    tier.prepare_read(held)
    assert tier.reserve(b"x", tier.new_rank(), 0) is None
    assert tier.allocation_failures == 1 and tier.lookup(b"x") == MISS


def test_a_write_can_complete_with_a_reader_already_holding_it():
    tier = PrimaryTier(2)
    keys = _keys("k", 2)
    taken = tier.prepare_write(keys, tier.new_rank(), range(2))
    tier.complete_write([key for key, _ in taken], True, 1)
    assert tier.readable(keys + [b"absent"]) == keys
    assert _events(tier) == [(keys[0], False), (keys[1], False)]
    assert tier.prepare_write(_keys("x", 1), tier.new_rank(), [0]) is None  # pinned
    tier.complete_read(keys)
    assert tier.prepare_write(_keys("x", 1), tier.new_rank(), [0]) is not None
    assert tier.lookup(keys[1]) == MISS and tier.lookup(keys[0]) == HIT


def test_an_older_request_released_late_is_still_evicted_first():
    tier = PrimaryTier(6)
    old, new = _keys("old", 2), _keys("new", 4)
    _store(tier, old, tier.new_rank())
    _store(tier, new, tier.new_rank())
    tier.prepare_read(old)  # pinned while the newer request starts to go
    _store(tier, _keys("x", 1), tier.new_rank())
    assert tier.lookup(new[3]) == MISS
    tier.complete_read(old)
    _store(tier, _keys("y", 2), tier.new_rank())
    assert [tier.lookup(k) for k in old] == [MISS, MISS]
    assert [tier.lookup(k) for k in new[:3]] == [HIT, HIT, HIT]


def test_keys_added_out_of_order_still_leave_tail_first():
    tier = PrimaryTier(4)
    keys = _keys("k", 4)
    rank = tier.new_rank()
    _store(tier, keys[2:], rank, start=2)
    _store(tier, keys[:2], rank)
    _store(tier, _keys("x", 2), tier.new_rank())
    assert [tier.lookup(k) for k in keys] == [HIT, HIT, MISS, MISS]


def test_stale_queue_entries_are_compacted_away():
    tier = PrimaryTier(8)
    keys = _keys("k", 4)
    _store(tier, keys, tier.new_rank())
    for _ in range(3 * 4096):  # every access queues the keys under a new rank
        tier.access(keys, tier.new_rank(), 0)
    assert len(tier._buckets) <= 4096
    _store(tier, _keys("x", 5), tier.new_rank())
    assert [tier.lookup(k) for k in keys] == [HIT, HIT, HIT, MISS]


def test_device_cache_counts_the_cached_prefix_of_a_request():
    cache = StreamCache(100)
    cache.insert([(1, 0, 4), (9, 0, 6)], cache.new_rank())
    assert len(cache) == 10
    assert cache.prefix([(1, 0, 4), (9, 0, 6)]) == 10
    assert cache.prefix([(1, 0, 4), (9, 0, 8)]) == 10  # the context grew
    assert cache.prefix([(2, 0, 4), (9, 0, 6)]) == 0  # another system prompt
    assert cache.prefix([(1, 0, 4), (7, 0, 3)]) == 4
    cache.insert([(1, 0, 4), (9, 0, 8), (9, 8, 9)], cache.new_rank())  # with its tail
    assert cache.held(9) == 9 and len(cache) == 13
    assert cache.events == [(1, 0, 4, False), (9, 0, 9, False)]


def test_device_cache_evicts_the_oldest_request_tail_first():
    cache = StreamCache(10)
    cache.insert([(1, 0, 4), (8, 0, 3)], cache.new_rank())
    cache.insert([(1, 0, 4), (9, 0, 3)], cache.new_rank())
    assert len(cache) == 10
    cache.events.clear()
    cache.insert([(2, 0, 2)], cache.new_rank())
    # The older session's context goes from its end; the shared system
    # prompt was used by the newer request and stays.
    assert cache.events == [(2, 0, 2, False), (8, 1, 3, True)]
    assert cache.prefix([(1, 0, 4), (8, 0, 3)]) == 5
    cache.insert([(2, 0, 2), (7, 0, 4)], cache.new_rank())
    assert cache.held(8) == 0 and cache.held(9) == 0 and cache.held(1) == 4
    assert len(cache) == 10


def test_device_cache_takes_a_requests_later_streams_before_its_earlier_ones():
    cache = StreamCache(6)
    cache.insert([(1, 0, 3), (9, 0, 3)], cache.new_rank())
    cache.insert([(2, 0, 4)], cache.new_rank())
    assert cache.held(9) == 0 and cache.held(1) == 2 and cache.held(2) == 4


def test_device_cache_ignores_blocks_that_are_not_root_aligned():
    cache = StreamCache(10)
    cache.insert([(5, 2, 4)], cache.new_rank())
    assert len(cache) == 0 and cache.prefix([(5, 0, 4)]) == 0
