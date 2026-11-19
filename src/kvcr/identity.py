# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Identity of a stored block: ``SharedBlockId``.

These helpers do not read the block bytes. A block extent has no name, so
KVCR stores the shared block id in a fixed-size header and rejects a header
that differs. The id is the record version, block key, block size, and optional
storage prefix. ``prefix`` is one path segment, or empty. File paths, object
keys, and this header use that same segment. The header is always 1 KiB, so a
payload written after it starts at a fixed offset. A caller that needs a
larger alignment pads outside this header. These helpers keep no cache.
"""

import struct

import msgspec

# Bump when the record layout changes.
KV_VERSION = 1

# Opaque framework keys are short hashes. Longer keys are refused.
MAX_BLOCK_KEY_BYTES = 256
# Reject before decode. Large enough for a longest key, the other fields,
# and msgpack tags; this is the only cap on prefix length. Stored headers are
# always this size.
MAX_HEADER_BYTES = 1024
# Msgpack body length. The rest is zero padding; a reader does not parse to
# find where the header ends. Msgpack itself rejects trailing bytes, and the
# body may end in zero, so the length is what makes the padding ignorable.
_HEADER_LEN = struct.Struct("<I")


class RecordError(ValueError):
    """A record is malformed, too large, or does not match this block."""


def _version(value: object) -> int:
    if type(value) is not int or value != KV_VERSION:
        raise ValueError(f"unsupported version {value!r}")
    return value


def _block_size(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise RecordError("block size must be a positive integer")
    return value


def _prefix(value: object) -> str:
    if type(value) is not str:
        raise RecordError("prefix must be a string")
    if not all(ch.isalnum() or ch in "_-" for ch in value):
        raise RecordError("prefix must be letters, digits, '_', or '-'")
    return value


def _block_key(value: object) -> bytes:
    if type(value) is not bytes or not 1 <= len(value) <= MAX_BLOCK_KEY_BYTES:
        raise RecordError(f"block key must be 1..{MAX_BLOCK_KEY_BYTES} bytes")
    return value


class SharedBlockId(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    array_like=True,
    forbid_unknown_fields=True,
):
    """One stored block's identity.

    ``prefix`` is the storage namespace shared by file paths, object keys, and
    this header: one segment of letters, digits, ``_``, or ``-``, or empty.
    """

    version: int
    block_key: bytes
    block_size_bytes: int
    prefix: str = ""

    def __post_init__(self) -> None:
        _version(self.version)
        _block_key(self.block_key)
        _block_size(self.block_size_bytes)
        _prefix(self.prefix)
        # Encoded size is the prefix cap, so a header that cannot be stored
        # never exists.
        encoded = _ENCODER.encode(self)
        if _HEADER_LEN.size + len(encoded) > MAX_HEADER_BYTES:
            raise RecordError("header is too large")


_ENCODER = msgspec.msgpack.Encoder()
_DECODER = msgspec.msgpack.Decoder(SharedBlockId)


def encode_header(block_id: SharedBlockId) -> bytes:
    """Header KVCR writes in front of a block-extent payload.

    Always ``MAX_HEADER_BYTES`` long. The payload begins at that offset.
    """
    if type(block_id) is not SharedBlockId:
        raise RecordError("block id must be a SharedBlockId")
    encoded = _ENCODER.encode(block_id)
    pad = MAX_HEADER_BYTES - _HEADER_LEN.size - len(encoded)
    return _HEADER_LEN.pack(len(encoded)) + encoded + bytes(pad)


def decode_header(
    data: bytes | bytearray | memoryview,
    *,
    block_id: SharedBlockId,
) -> SharedBlockId:
    """Reject a header that is malformed or is another shared block id."""
    if type(block_id) is not SharedBlockId:
        raise RecordError("block id must be a SharedBlockId")
    view = memoryview(data)
    if view.nbytes > MAX_HEADER_BYTES:
        raise RecordError("header is too large")
    if view.nbytes < MAX_HEADER_BYTES:
        raise RecordError("header is truncated")
    # Item size may not be one byte; the header layout is byte-oriented.
    view = view.cast("B")
    (length,) = _HEADER_LEN.unpack_from(view)
    body_end = _HEADER_LEN.size + length
    if body_end > MAX_HEADER_BYTES or any(view[body_end:]):
        raise RecordError("malformed record")
    try:
        found = _DECODER.decode(view[_HEADER_LEN.size : body_end])
    except (msgspec.DecodeError, ValueError, TypeError) as error:
        raise RecordError(f"malformed record: {error}") from error
    if found != block_id:
        raise RecordError("incompatible record")
    return found
