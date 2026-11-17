# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Block identity shared by the coordinator and the workers.

A block is ``(stream, index)``: a stream is one lineage of KV blocks (a shared
system prefix, or one session's context) and ``index`` is the block's position
in it. The coordinator only ever handles ``(stream, start, end)`` segments;
workers expand them into KVCR keys.

The KVCR ``BlockKey`` is 36 bytes like vLLM's (a 32-byte hash plus a 4-byte
group index) so dict and set costs inside KVCR match a real deployment, and it
is reversible so no reverse map is needed. The router hash is the packed id.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable

INDEX_BITS = 24
INDEX_MASK = (1 << INDEX_BITS) - 1
MAX_STREAM = (1 << (64 - INDEX_BITS)) - 1

# Stream id ranges. Sessions get ``SESSION_BASE + session * GENERATIONS + gen``;
# each play of a replayed trace takes the next free ids from ``TRACE_BASE`` up.
SYSTEM_BASE = 1
WARM_BASE = 1 << 20
SESSION_BASE = 1 << 24
GENERATIONS = 64
TRACE_BASE = 1 << 32

Segment = tuple[int, int, int]  # stream, start index, end index (exclusive)

_KEY = struct.Struct(">Q24xI")
KEY_BYTES = _KEY.size
_ID = struct.Struct(">Q")
_KEY_IDS = struct.Struct(">Q28x")


def block_id(stream: int, index: int) -> int:
    return (stream << INDEX_BITS) | index


def split_id(bid: int) -> tuple[int, int]:
    return bid >> INDEX_BITS, bid & INDEX_MASK


def id_to_key(bid: int) -> bytes:
    return _KEY.pack(bid, 0)


def key_to_id(key: bytes) -> int:
    return _ID.unpack_from(key)[0]


def keys_to_ids(keys: Iterable[bytes]) -> list[int]:
    """Block ids of many keys at once."""
    return [bid for (bid,) in _KEY_IDS.iter_unpack(b"".join(keys))]


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


class KeyCache:
    """Per-stream key lists, so a request's keys are list slices."""

    def __init__(self, max_streams: int = 4096) -> None:
        self._streams: dict[int, list[bytes]] = {}
        self._max_streams = max_streams

    def keys(self, segments: Iterable[Segment]) -> list[bytes]:
        out: list[bytes] = []
        for stream, start, end in segments:
            cached = self._streams.pop(stream, None)
            if cached is None:
                cached = []
            if len(cached) < end:
                base = stream << INDEX_BITS
                pack = _KEY.pack
                cached.extend(pack(base + i, 0) for i in range(len(cached), end))
            # Reinsert at the back: the dict doubles as an LRU of streams.
            self._streams[stream] = cached
            out.extend(cached[start:end])
        while len(self._streams) > self._max_streams:
            del self._streams[next(iter(self._streams))]
        return out


class KeyAdapter:
    """KVCR ``KeyAdapter`` for both the current and the pre-#161 protocol."""

    # Current protocol.
    def encode_to_framework(self, key: bytes) -> object:
        return key

    def decode_from_framework(self, framework_key: object) -> bytes:
        return framework_key  # type: ignore[return-value]

    def encode_to_router(self, key: bytes) -> int:
        return key_to_id(key)

    def decode_from_router(self, router_key: int | bytes) -> bytes:
        if isinstance(router_key, bytes):
            return router_key
        return id_to_key(router_key)

    # Older trees.
    def encode(self, framework_key: object) -> bytes:
        return framework_key  # type: ignore[return-value]

    def decode(self, key: bytes) -> int:
        return key_to_id(key)


# ---- SGLang direct-linker identities ----------------------------------------
#
# SGLang names a page by a chained SHA-256 "storage hash" (64 hex digits). The
# linker's KVCR key is ``sglang-kvcr-v1:<layout namespace>:<pool>:<hash>`` and
# the router hash is the first 16 digits. Here the block id fills those 16
# digits, so a page hash is reversible like the vLLM key.

LINKER_NAMESPACE = "c1" * 32  # one model and layout for every worker
LINKER_KEY_PREFIX = f"sglang-kvcr-v1:{LINKER_NAMESPACE}:".encode()
LINKER_POOL = "kv"
_HASH_PAD = "0" * 48


def page_hash(bid: int) -> str:
    return f"{bid:016x}{_HASH_PAD}"


def hash_to_id(storage_hash: str) -> int:
    return int(storage_hash[:16], 16)


class HashCache:
    """Per-stream page-hash lists, so a request's hashes are list slices."""

    def __init__(self, max_streams: int = 4096) -> None:
        self._streams: dict[int, list[str]] = {}
        self._max_streams = max_streams

    def stream(self, stream: int, end: int) -> list[str]:
        """Hashes of the stream's pages ``[0, end)`` (the list may be longer)."""
        cached = self._streams.pop(stream, None)
        if cached is None:
            cached = []
        if len(cached) < end:
            base = stream << INDEX_BITS
            pad = _HASH_PAD
            cached.extend(f"{base + i:016x}{pad}" for i in range(len(cached), end))
        self._streams[stream] = cached
        if len(self._streams) > self._max_streams:
            del self._streams[next(iter(self._streams))]
        return cached

    def hashes(self, segments: Iterable[Segment]) -> list[str]:
        out: list[str] = []
        for stream, start, end in segments:
            out.extend(self.stream(stream, end)[start:end])
        return out
