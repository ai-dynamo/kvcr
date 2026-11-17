# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The SGLang profile's device radix tree."""

from clusterbench.device_tree import DeviceTree
from clusterbench.engine_sglang import _G2Prefix
from clusterbench.protocol import TIER_G2
from clusterbench.router import Inventory

SYSTEM, A, B = 1, 100, 200


def _grow(tree, stream, pages, parent=None):
    start = tree.held(stream)
    slots = tree.allocate(pages)
    assert slots is not None
    return tree.insert(stream, start, slots, parent)


def test_prefix_path_and_slots_follow_segments():
    tree = DeviceTree(64)
    system = _grow(tree, SYSTEM, 4)
    first = _grow(tree, A, 3, SYSTEM)
    second = _grow(tree, A, 2, SYSTEM)
    request = [(SYSTEM, 0, 4), (A, 0, 8)]
    assert tree.prefix(request) == 9
    assert tree.prefix([(SYSTEM, 0, 4), (B, 0, 8)]) == 4
    assert tree.path(request, 9) == [system, first, second]
    assert tree.path(request, 5) == [system, first]
    assert tree.path(request, 4) == [system]
    assert tree.slots(request, 3, 3) == [system.slots[3], *first.slots[:2]]
    assert tree.node_at(A, 3) is second
    assert tree.node_at(A, 5) is None
    assert len(tree) == 9 and tree.fill_level() == 9 / 64


def test_insert_gives_back_slots_of_pages_the_stream_already_has():
    tree = DeviceTree(32)
    _grow(tree, A, 4)
    slots = tree.allocate(6)
    node = tree.insert(A, 2, slots)  # pages 2..7: 2 and 3 are already there
    assert (node.start, node.end) == (4, 8)
    assert node.slots == slots[2:]
    assert tree.duplicate_pages == 2 and len(tree) == 8
    # Pages past the end of the stream could never be found again.
    assert tree.insert(A, 20, tree.allocate(2)) is None
    assert len(tree) == 8


def test_eviction_takes_unlocked_leaves_least_recently_used_first():
    tree = DeviceTree(12)
    system = _grow(tree, SYSTEM, 4)
    a = _grow(tree, A, 4, SYSTEM)
    _grow(tree, B, 4, SYSTEM)
    tree.take_events()
    tree.touch([system, a])  # A is now more recent than B
    assert tree.allocate(2) is not None
    assert tree.take_events() == [(B, 0, 4, True)]
    # The system prefix is not a leaf while A continues from it, and A is locked.
    tree.lock([a])
    assert tree.allocate(4) is None
    assert tree.allocation_failures == 1 and tree.held(SYSTEM) == 4
    tree.unlock([a])
    assert tree.allocate(8) is not None
    assert tree.take_events() == [(A, 0, 4, True), (SYSTEM, 0, 4, True)]
    assert tree.evictions == 3 and tree.evicted_pages == 12


def test_eviction_peels_a_stream_from_its_tail():
    tree = DeviceTree(8)
    _grow(tree, A, 3)
    _grow(tree, A, 3)
    tree.take_events()
    assert tree.allocate(4) is not None
    assert tree.take_events() == [(A, 3, 6, True)]
    assert tree.held(A) == 3


def test_reserved_slots_are_never_handed_out():
    tree = DeviceTree(8, reserved=3)
    assert sorted(tree.allocate(5)) == [3, 4, 5, 6, 7]
    assert tree.allocate(1) is None


def test_kvcr_prefix_survives_head_eviction_and_refill():
    g2, router = _G2Prefix(), Inventory(1)

    def flush():
        router.apply(0, g2.deltas)
        g2.deltas = []
        return router.prefix(0, [(A, 0, 10)], (TIER_G2,))

    g2.add(A, 0, 6)
    assert flush() == 6
    g2.remove(A, 0)  # KVCR evicts in access order: the head can go first
    assert flush() == 0
    g2.add(A, 0, 1)  # backed up again: pages 1..5 never left
    assert flush() == 6
    g2.remove(A, 3)
    assert flush() == 3
    g2.add(A, 6, 9)  # not contiguous with the prefix: not reported yet
    assert flush() == 3
    g2.add(A, 3, 4)
    assert flush() == 9
    assert g2.count == 9


def test_a_lock_covers_the_ancestry_and_refuses_before_evicting():
    tree = DeviceTree(12)
    system = _grow(tree, SYSTEM, 4)
    first = _grow(tree, A, 2, SYSTEM)
    second = _grow(tree, A, 2, SYSTEM)
    other = _grow(tree, B, 4, SYSTEM)
    assert tree.ancestry(second) == [system, first, second]
    assert tree.ancestry(first) == [system, first]
    assert tree.evictable() == 12
    tree.lock(tree.ancestry(first))  # a pending backup of A's first node
    assert tree.evictable() == 6
    tree.take_events()
    # Seven pages cannot be had: nothing is evicted for the attempt.
    assert tree.allocate(7) is None and tree.take_events() == []
    assert tree.allocate(6) is not None
    assert sorted(tree.take_events()) == [(A, 2, 4, True), (B, 0, 4, True)]
    assert tree.held(A) == 2 and other.locks == 0
    tree.unlock(tree.ancestry(first))
    assert tree.evictable() == 6
