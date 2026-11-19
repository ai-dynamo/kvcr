# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Block identity: shared block id."""

import msgspec
import pytest

from kvcr.identity import (
    KV_VERSION,
    MAX_HEADER_BYTES,
    RecordError,
    SharedBlockId,
    decode_header,
    encode_header,
)
from kvcr.types import BlockKey

_KEY = BlockKey(b"block-hash")
_BLOCK_ID = SharedBlockId(
    version=KV_VERSION, block_key=_KEY, block_size_bytes=16, prefix="tenant-a"
)


def test_same_id_round_trip() -> None:
    header = encode_header(_BLOCK_ID)
    short = encode_header(
        SharedBlockId(version=KV_VERSION, block_key=_KEY, block_size_bytes=16)
    )
    assert len(header) == len(short) == MAX_HEADER_BYTES
    found = decode_header(memoryview(header).cast("I"), block_id=_BLOCK_ID)
    assert found == _BLOCK_ID
    assert {found: "extent"}[_BLOCK_ID] == "extent"


def test_other_id_is_rejected() -> None:
    header = encode_header(_BLOCK_ID)
    other_key = SharedBlockId(
        version=KV_VERSION,
        block_key=BlockKey(b"other"),
        block_size_bytes=16,
        prefix="tenant-a",
    )
    other_size = SharedBlockId(
        version=KV_VERSION, block_key=_KEY, block_size_bytes=32, prefix="tenant-a"
    )
    other_prefix = SharedBlockId(
        version=KV_VERSION, block_key=_KEY, block_size_bytes=16, prefix=""
    )
    for other in (other_key, other_size, other_prefix):
        with pytest.raises(RecordError, match="incompatible"):
            decode_header(header, block_id=other)


def test_prefix_is_optional() -> None:
    omitted = SharedBlockId(version=KV_VERSION, block_key=_KEY, block_size_bytes=16)
    blank = SharedBlockId(
        version=KV_VERSION, block_key=_KEY, block_size_bytes=16, prefix=""
    )
    assert omitted.prefix == blank.prefix == ""
    assert decode_header(encode_header(omitted), block_id=blank) == omitted


@pytest.mark.parametrize("bad", [b"", b"x" * 257, "text", 1])
def test_bad_key_is_refused(bad) -> None:
    with pytest.raises(RecordError):
        SharedBlockId(version=KV_VERSION, block_key=bad, block_size_bytes=16)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_size", [0, -1, True, "16"])
def test_bad_block_size_is_refused(bad_size) -> None:
    with pytest.raises(RecordError, match="block size"):
        SharedBlockId(version=KV_VERSION, block_key=_KEY, block_size_bytes=bad_size)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_prefix", [1, b"tenant", "/*/", "ten/ant-a", "tenant a"])
def test_bad_prefix_is_refused(bad_prefix) -> None:
    with pytest.raises(RecordError, match="prefix"):
        SharedBlockId(  # type: ignore[arg-type]
            version=KV_VERSION, block_key=_KEY, block_size_bytes=16, prefix=bad_prefix
        )


def test_overlong_prefix_is_refused() -> None:
    with pytest.raises(RecordError, match="too large"):
        SharedBlockId(
            version=KV_VERSION,
            block_key=_KEY,
            block_size_bytes=16,
            prefix="a" * MAX_HEADER_BYTES,
        )


def test_length_cannot_run_past_the_header() -> None:
    # A body that fills the header still decodes if a too-large length is sliced
    # down to the buffer. The declared length has to be rejected first.
    block_id = None
    header = b""
    for count in range(1, MAX_HEADER_BYTES):
        try:
            candidate = SharedBlockId(
                version=KV_VERSION,
                block_key=_KEY,
                block_size_bytes=16,
                prefix="a" * count,
            )
            encoded = encode_header(candidate)
        except RecordError:
            break
        if encoded[-1] != 0:
            block_id, header = candidate, encoded
    assert block_id is not None and len(header) == MAX_HEADER_BYTES
    forged = bytearray(header)
    forged[:4] = (int.from_bytes(forged[:4], "little") + 1).to_bytes(4, "little")
    with pytest.raises(RecordError, match="malformed"):
        decode_header(forged, block_id=block_id)


def test_decode_rejects_untrusted_bytes() -> None:
    good = encode_header(_BLOCK_ID)
    dirty = bytearray(good)
    dirty[-1] = 1
    overrun = bytearray(MAX_HEADER_BYTES)
    overrun[:4] = MAX_HEADER_BYTES.to_bytes(4, "little")
    for bad in (
        b"",
        b"\xc1",
        good[:-1],
        good + b"x",
        bytes(MAX_HEADER_BYTES + 1),
        dirty,
        overrun,
    ):
        with pytest.raises(RecordError):
            decode_header(bad, block_id=_BLOCK_ID)
    # Fixed header: little-endian uint32 length, msgpack body, zero pad.
    length = int.from_bytes(good[:4], "little")
    record = msgspec.msgpack.decode(good[4 : 4 + length])
    record[0] = 2
    body = msgspec.msgpack.encode(record)
    forged = len(body).to_bytes(4, "little") + body
    forged += bytes(MAX_HEADER_BYTES - len(forged))
    with pytest.raises(RecordError, match="unsupported version"):
        decode_header(forged, block_id=_BLOCK_ID)
    oversized = memoryview(bytes(MAX_HEADER_BYTES * 2)).cast("I")
    with pytest.raises(RecordError, match="too large"):
        decode_header(oversized, block_id=_BLOCK_ID)
