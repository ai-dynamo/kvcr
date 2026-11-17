# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from clusterbench import ids


def test_block_key_is_reversible_and_vllm_sized():
    bid = ids.block_id(ids.session_stream(7, 3), 12345)
    key = ids.id_to_key(bid)
    assert len(key) == 36
    assert ids.key_to_id(key) == bid
    assert ids.split_id(bid) == (ids.session_stream(7, 3), 12345)
    assert 0 <= bid < 1 << 64


def test_streams_do_not_collide():
    streams = {ids.system_stream(a) for a in range(8)}
    streams |= {ids.warm_stream(w) for w in range(512)}
    streams |= {ids.session_stream(s, g) for s in range(64) for g in range(4)}
    assert len(streams) == 8 + 512 + 64 * 4


def test_segments_expand_to_ids_in_order():
    segments = [(5, 0, 3), (9, 2, 4)]
    expanded = ids.segment_ids(segments)
    assert expanded == [ids.block_id(5, i) for i in range(3)] + [
        ids.block_id(9, 2),
        ids.block_id(9, 3),
    ]
    assert ids.segment_blocks(segments) == len(expanded)


def test_key_cache_returns_the_same_keys_for_a_growing_stream():
    cache = ids.KeyCache()
    first = cache.keys([(4, 0, 3)])
    grown = cache.keys([(4, 0, 5)])
    assert grown[:3] == first
    assert all(a is b for a, b in zip(first, grown))  # cached objects, not copies
    assert [ids.key_to_id(k) for k in grown] == [ids.block_id(4, i) for i in range(5)]
    assert cache.keys([(4, 2, 4)]) == grown[2:4]


def test_key_cache_is_bounded():
    cache = ids.KeyCache(max_streams=4)
    for stream in range(10):
        cache.keys([(stream, 0, 2)])
    assert len(cache._streams) == 4


def test_key_adapter_speaks_both_protocol_generations():
    adapter = ids.KeyAdapter()
    key = ids.id_to_key(ids.block_id(3, 1))
    assert adapter.encode_to_router(key) == adapter.decode(key) == ids.block_id(3, 1)
    assert adapter.decode_from_router(ids.block_id(3, 1)) == key
    assert adapter.decode_from_framework(adapter.encode_to_framework(key)) == key
    assert adapter.encode(key) == key
