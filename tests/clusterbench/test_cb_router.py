# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from clusterbench.config import RouterConfig, TierConfig
from clusterbench.protocol import TIER_DEVICE, TIER_G2, TIER_PRIMARY
from clusterbench.router import Inventory, Router

NODES = ["a", "a", "b", "b"]
SEGMENTS = [(1, 0, 10), (50, 0, 20)]
ALL = (TIER_DEVICE, TIER_PRIMARY, TIER_G2)


def test_inventory_tracks_root_aligned_runs():
    inventory = Inventory(1)
    inventory.apply(
        0, [(TIER_PRIMARY, 1, 0, 10, False), (TIER_PRIMARY, 50, 0, 8, False)]
    )
    assert inventory.prefix(0, SEGMENTS, ALL) == 18
    # A gap is not a prefix: blocks 12.. do not extend a run that ends at 8.
    inventory.apply(0, [(TIER_PRIMARY, 50, 12, 20, False)])
    assert inventory.prefix(0, SEGMENTS, ALL) == 18
    inventory.apply(0, [(TIER_PRIMARY, 50, 8, 12, False)])
    assert inventory.prefix(0, SEGMENTS, ALL) == 22
    # Tail-first eviction shortens the run; evicting the root drops it.
    inventory.apply(0, [(TIER_PRIMARY, 50, 5, 12, True)])
    assert inventory.prefix(0, SEGMENTS, ALL) == 15
    inventory.apply(0, [(TIER_PRIMARY, 1, 0, 10, True)])
    assert inventory.prefix(0, SEGMENTS, ALL) == 0


def test_tiers_union_and_a_missing_first_segment_stops_the_prefix():
    inventory = Inventory(1)
    inventory.apply(0, [(TIER_DEVICE, 1, 0, 4, False), (TIER_G2, 1, 0, 10, False)])
    assert inventory.prefix(0, SEGMENTS, (TIER_DEVICE,)) == 4
    assert inventory.prefix(0, SEGMENTS, ALL) == 10
    inventory = Inventory(1)
    inventory.apply(0, [(TIER_PRIMARY, 50, 0, 20, False)])
    assert inventory.prefix(0, SEGMENTS, ALL) == 0


def _router(**kwargs):
    return Router(RouterConfig(**kwargs), NODES, TierConfig())


def test_hint_names_the_longest_holder_with_ties_to_the_lower_id():
    router = _router(migration_prob=0.0)
    session = 2  # home worker 2
    assert router.route(session, 0, SEGMENTS, 2000).hint_source == -1
    router.inventory.apply(1, [(TIER_PRIMARY, 1, 0, 10, False)])
    router.inventory.apply(3, [(TIER_G2, 1, 0, 10, False), (TIER_G2, 50, 0, 6, False)])
    router.inventory.apply(0, [(TIER_G2, 1, 0, 10, False), (TIER_G2, 50, 0, 6, False)])
    route = router.route(session, 1, SEGMENTS, 2000)
    assert (route.worker, route.hint_source, route.hint_blocks) == (2, 0, 16)
    # One block more locally than the best peer: no hint.
    router.inventory.apply(2, [(TIER_DEVICE, 1, 0, 10, False)])
    router.inventory.apply(2, [(TIER_DEVICE, 50, 0, 6, False)])
    assert router.route(session, 2, SEGMENTS, 2000).hint_source == -1


def test_hint_search_matches_a_scan_of_every_worker():
    import random

    rng = random.Random(7)
    nodes = [f"n{w // 4}" for w in range(16)]
    router = Router(RouterConfig(migration_prob=0.0), nodes, TierConfig(g2_blocks=64))
    streams = [1, 2, 50, 51, 52, 53]
    hinted = 0
    for step in range(3000):
        worker = rng.randrange(16)
        tier = rng.choice(ALL)
        stream = rng.choice(streams)
        start = rng.randrange(0, 12)
        delta = (tier, stream, start, start + rng.randrange(1, 12), rng.random() < 0.4)
        router.inventory.apply(worker, [delta])
        if step % 3:
            continue
        segments = [(rng.choice(streams[:2]), 0, 10), (rng.choice(streams[2:]), 0, 20)]
        route = router.route(1000 + step, 0, segments, 2000)
        best, best_blocks = -1, route.local_blocks
        for peer in range(16):
            if peer != route.worker:
                held = router.inventory.prefix(peer, segments, (TIER_PRIMARY, TIER_G2))
                if held > best_blocks:
                    best, best_blocks = peer, held
        assert route.hint_source == best
        assert route.hint_blocks == (best_blocks if best >= 0 else 0)
        hinted += best >= 0
    assert hinted > 100
    # A stream nobody holds any more is forgotten.
    for worker in range(16):
        for tier in ALL:
            router.inventory.apply(worker, [(tier, s, 0, 99, True) for s in streams])
    assert router.inventory._holders == [{}, {}, {}]


def test_device_only_blocks_cannot_be_a_hint_source():
    router = _router(migration_prob=0.0)
    router.inventory.apply(1, [(TIER_DEVICE, 1, 0, 10, False)])
    assert router.route(2, 0, SEGMENTS, 2000).hint_source == -1


def test_pinless_peer_only_sources_are_ignored():
    tiers = TierConfig(g2_blocks=4096, framework_pinning=False)
    router = Router(RouterConfig(migration_prob=0.0), NODES, tiers)
    router.inventory.apply(1, [(TIER_PRIMARY, 1, 0, 10, False)])
    assert router.route(2, 0, SEGMENTS, 2000).hint_source == -1
    router.inventory.apply(1, [(TIER_G2, 1, 0, 10, False)])
    assert router.route(2, 1, SEGMENTS, 2000).hint_source == 1


def test_sticky_routing_is_deterministic_and_ignores_load():
    def plan(seed):
        router = _router(migration_prob=0.3, seed=seed)
        return [
            router.route(s, t, SEGMENTS, 2000).worker
            for t in range(40)
            for s in range(6)
        ]

    assert plan(1) == plan(1)
    assert plan(1) != plan(2)
    loaded = _router(migration_prob=0.3, seed=1)
    out = []
    for t in range(40):
        for s in range(6):
            route = loaded.route(s, t, SEGMENTS, 2000)
            loaded.dispatched(t * 6 + s, route, 30, 2000)  # never completes
            out.append(route.worker)
    assert out == plan(1)


def test_sticky_migration_rate():
    router = _router(migration_prob=0.25)
    moves = 0
    for session in range(200):
        home = router.route(session, 0, SEGMENTS, 2000).worker
        for turn in range(1, 20):
            route = router.route(session, turn, SEGMENTS, 2000)
            if route.migrated:
                moves += 1
                assert route.worker != home
            home = route.worker
    assert 0.2 < moves / (200 * 19) < 0.3


def test_same_node_prob_chooses_among_equally_empty_workers():
    for same_node_prob, expected in ((1.0, {1}), (0.0, {2, 3})):
        seen = set()
        for seed in range(8):
            router = _router(
                migration_prob=1.0, same_node_prob=same_node_prob, seed=seed
            )
            for session in range(4):  # one session on each worker
                assert router.route(session, 0, SEGMENTS, 2000).worker == session
            seen.add(router.route(0, 1, SEGMENTS, 2000).worker)
        assert seen == expected


def test_a_migrating_session_goes_to_the_emptiest_worker():
    nodes = [f"n{w // 4}" for w in range(12)]
    router = Router(RouterConfig(migration_prob=0.5), nodes, TierConfig())
    sessions = list(range(96))
    homes = {s: router.route(s, 0, SEGMENTS, 2000).worker for s in sessions}
    assert sorted(homes.values()) == sorted(list(range(12)) * 8)
    moves = 0
    for turn in range(1, 40):
        for s in sessions[:: 1 + turn % 3]:  # sessions do not take turns evenly
            counts = [list(homes.values()).count(w) for w in range(12)]
            home = homes[s]
            route = router.route(s, turn, SEGMENTS, 2000)
            if route.migrated:
                moves += 1
                others = [count for w, count in enumerate(counts) if w != home]
                assert counts[route.worker] == min(others)
            homes[s] = route.worker
        counts = [list(homes.values()).count(w) for w in range(12)]
        assert max(counts) - min(counts) <= 2, counts
    assert moves > 500
    # A session that ends frees its place; a new one takes the emptiest worker.
    counts = [list(homes.values()).count(w) for w in range(12)]
    least = min(counts)
    assert counts[router.route(1000, 0, SEGMENTS, 2000).worker] == least
    router.forget(1000)
    busiest = max(range(12), key=counts.__getitem__)
    for s, home in list(homes.items()):
        if home == busiest:
            router.forget(s)
            del homes[s]
    assert router.route(2000, 0, SEGMENTS, 2000).worker == busiest


def test_cost_mode_follows_overlap_until_load_outweighs_it():
    router = _router(mode="cost")
    router.inventory.apply(3, [(TIER_DEVICE, 1, 0, 10, False)])
    first = router.route(0, 0, SEGMENTS, 30 * 64)
    assert first.worker == 3
    for rid in range(50):
        router.dispatched(rid, first, 30, 30 * 64)
    assert router.route(1, 0, SEGMENTS, 30 * 64).worker != 3
    for rid in range(50):
        router.done(rid)
    assert router.route(2, 0, SEGMENTS, 30 * 64).worker == 3
