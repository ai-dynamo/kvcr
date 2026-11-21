# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Block identity shared by the coordinator and the workers.

A block is ``(stream, index)``: a stream is one lineage of KV blocks (a shared
system prefix, or one session's context) and ``index`` is the block's position
in it. The coordinator only ever handles ``(stream, start, end)`` segments;
workers expand them into block hashes for the engine.

The 64-bit block id is the low 8 bytes of a 32-byte vLLM block hash. vLLM
appends a 4-byte group index to make the offload key, which is the KVCR key,
and gives a router the low 64 bits of the hash: the id again. So keys are 36
bytes, as in a real deployment, and reversible.
"""

from __future__ import annotations

from collections.abc import Iterable

INDEX_BITS = 24
INDEX_MASK = (1 << INDEX_BITS) - 1

# Stream id ranges. Sessions get ``SESSION_BASE + session * GENERATIONS + gen``;
# each play of a replayed trace takes the next free ids from ``TRACE_BASE`` up.
# From ``PRIVATE_BASE`` up, a stream belongs to one request: the blocks of it
# that no segment names (one straddling two streams, an answer block no later
# prompt uses).
SYSTEM_BASE = 1
WARM_BASE = 1 << 20
SESSION_BASE = 1 << 24
GENERATIONS = 64
TRACE_BASE = 1 << 32
PRIVATE_BASE = 1 << 39
MAX_STREAM = PRIVATE_BASE - 1

Segment = tuple[int, int, int]  # stream, start index, end index (exclusive)


def block_id(stream: int, index: int) -> int:
    return (stream << INDEX_BITS) | index


def block_hash(bid: int) -> bytes:
    return bid.to_bytes(32, "big")


def key_to_id(key: bytes) -> int:
    """The block id of a KVCR key (a block hash plus the group index)."""
    return int.from_bytes(key[24:32], "big")


def system_stream(agent_type: int) -> int:
    return SYSTEM_BASE + agent_type


def warm_stream(worker: int) -> int:
    return WARM_BASE + worker


def session_stream(session: int, generation: int) -> int:
    return SESSION_BASE + session * GENERATIONS + generation % GENERATIONS


def segment_blocks(segments: Iterable[Segment]) -> int:
    return sum(end - start for _, start, end in segments)


def segment_ids(segments: Iterable[Segment]) -> list[int]:
    ids: list[int] = []
    for stream, start, end in segments:
        base = stream << INDEX_BITS
        ids.extend(range(base + start, base + end))
    return ids
