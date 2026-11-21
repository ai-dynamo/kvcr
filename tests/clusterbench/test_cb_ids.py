# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from clusterbench import ids


def test_a_block_id_is_the_low_bytes_of_its_hash_and_survives_the_kvcr_key():
    bid = ids.block_id(ids.session_stream(7, 3), 12345)
    block_hash = ids.block_hash(bid)
    # What vLLM gives a router is the low 64 bits of a 32-byte block hash.
    assert len(block_hash) == 32 and int.from_bytes(block_hash[-8:], "big") == bid
    # The KVCR key is the hash plus vLLM's 4-byte group index.
    assert ids.key_to_id(block_hash + bytes(4)) == bid


def test_streams_do_not_collide():
    streams = {ids.system_stream(a) for a in range(8)}
    streams |= {ids.warm_stream(w) for w in range(512)}
    streams |= {ids.session_stream(s, g) for s in range(64) for g in range(4)}
    assert len(streams) == 8 + 512 + 64 * 4
    # Above every named stream, each request has ids of its own.
    assert max(streams) < ids.TRACE_BASE < ids.PRIVATE_BASE == ids.MAX_STREAM + 1


def test_segments_expand_to_ids_in_order():
    segments = [(5, 0, 3), (9, 2, 4)]
    expanded = ids.segment_ids(segments)
    assert expanded == [ids.block_id(5, i) for i in range(3)] + [
        ids.block_id(9, 2),
        ids.block_id(9, 3),
    ]
    assert ids.segment_blocks(segments) == len(expanded)
