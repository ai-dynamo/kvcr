# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Recovery wire projection, mirroring, and bounded journal storage."""

import ctypes
import ctypes.util
import hashlib
import hmac
import logging
import mmap
import struct
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, Annotated, Any

import msgspec

from .config import KVCRBackendConfigs, KVCRConfig, KVCRGuardConfig
from .core import _BlockRecord, _KVCRCore
from .guard_protocol import KVCRClient, KVCRPoolHold
from .local_disk import _G3, _G3Residency
from .local_dram import _LocalDram, _LocalDramResidency, _LocalDramState
from .memory import _JOURNAL_HEADER_BYTES, KVCRPoolAttachment, KVCRPoolSpec
from .types import BlockKey, RecoveryMirrorError

if TYPE_CHECKING:
    from .api import KVCRBindings

logger = logging.getLogger(__name__)

_ACQUIRE = 2
_RELEASE = 3


@cache
def _atomics() -> tuple[Any, Any]:
    """Bind libatomic's 8-byte acquire/release operations on first use.

    Late because libatomic is a system library, not a declared dependency:
    importing kvcr must not require what only recovery uses.
    """
    library = ctypes.CDLL(ctypes.util.find_library("atomic") or "libatomic.so.1")

    def bind(name: str, argtypes: list[object], restype: object) -> Any:
        function = getattr(library, name)
        function.argtypes = argtypes
        function.restype = restype
        return function

    return (
        bind("__atomic_load_8", [ctypes.c_void_p, ctypes.c_int], ctypes.c_uint64),
        bind(
            "__atomic_store_8",
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int],
            None,
        ),
    )


class _AtomicU64:
    def __init__(self, address: int) -> None:
        self._address = address
        self._load, self._store = _atomics()

    def load_acquire(self) -> int:
        return int(self._load(self._address, _ACQUIRE))

    def store_release(self, value: int) -> None:
        self._store(self._address, value, _RELEASE)


_INVALID_OFFSET = 0
_PUBLISHED_OFFSET = 128
_CONSUMED_OFFSET = 192
_VALID = 0
_INVALID = 1
_ALIGNMENT = 8
_MAX_U64 = (1 << 64) - 1
_MAX_FRAME_SIZE = (1 << 16) - 1
# Size, record type, key size. Per frame because BlockKey has no declared
# width, so nothing would make a lease-wide key size true.
_FRAME_HEADER = struct.Struct("<HHH")

# A reader must never infer a record's kind from whether it happens to decode.
_RECORD_BLOCK = 1
_RECORD_TYPES = frozenset({_RECORD_BLOCK})


# Arrays, not maps: repeating field names costs ring space, and the ring
# filling ends recovery. 3 bytes a record instead of 21.
#
# Field order is the format. Append only -- never reorder or remove.
_NonNegativeInt = Annotated[int, msgspec.Meta(ge=0)]


class _RecoveryBlock(msgspec.Struct, frozen=True, array_like=True):
    # A slot per tier, or nothing. Bare ints: wrapping one costs a byte each.
    g2: _NonNegativeInt | None = None
    g3: _NonNegativeInt | None = None


_RECOVERY_ENCODER = msgspec.msgpack.Encoder()
_RECOVERY_DECODER = msgspec.msgpack.Decoder(_RecoveryBlock)


# TODO: Carry access history. Recovered blocks arrive with last_access unset,
# so LRU scores every one of them at -float_info.max until first touch and
# their relative recency is lost. Journaling it faithfully would cost a
# journal write per cache hit, since last_access changes on every access.
def _is_recoverable(record: _BlockRecord) -> bool:
    """Whether this record still names bytes a later holder of the pool can use.

    Both carriers of recovered state ask this -- the frames written to a
    handback region and the records a standing-down Guard keeps -- so that the
    two cannot come to describe different pools.
    """
    local_dram = record.local_dram
    return record.g3 is not None or (
        local_dram is not None and local_dram.state is _LocalDramState.READY
    )


def _project_recovery_record(record: _BlockRecord) -> _RecoveryBlock:
    local_dram = record.local_dram
    return _RecoveryBlock(
        g2=(
            local_dram.slot
            if local_dram is not None and local_dram.state is _LocalDramState.READY
            else None
        ),
        g3=record.g3.slot if record.g3 is not None else None,
    )


def _decode_recovery_record(payload: bytes) -> _BlockRecord:
    recovered = _RECOVERY_DECODER.decode(payload)
    return _BlockRecord(
        local_dram=(
            _LocalDramResidency(recovered.g2, _LocalDramState.READY)
            if recovered.g2 is not None
            else None
        ),
        g3=_G3Residency(recovered.g3) if recovered.g3 is not None else None,
    )


class RecoveryJournalError(RuntimeError):
    """The shared journal is incomplete or malformed."""


class RecoveryJournalTornError(RecoveryJournalError):
    """A handback region whose write never finished.

    Distinct from a region that disagrees about terms: this one is a write this
    service itself did not complete, so nobody is ever going to make sense of
    it and it can be thrown away. A region under other terms means the pool's
    configuration moved, which is not supposed to happen and is reported.
    """


class RecoveryJournal:
    """One-producer, one-consumer ring over a shared pool attachment.

    Each side caches only the cursor it advances and acquire-loads the peer's: a
    libatomic call through ctypes costs ~1.2us against ~140ns for the frame it
    orders. The invalid flag is read once per side, on first use.
    """

    def __init__(self, pool: KVCRPoolAttachment) -> None:
        mapping = pool._require_mapping()
        self._pool = pool
        self._capacity = pool._spec.journal_bytes - _JOURNAL_HEADER_BYTES
        base = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        self._published = _AtomicU64(base + _PUBLISHED_OFFSET)
        self._consumed = _AtomicU64(base + _CONSUMED_OFFSET)
        self._invalid = _AtomicU64(base + _INVALID_OFFSET)
        # On first use: this may attach to a journal already in progress.
        self._published_local: int | None = None
        self._consumed_local: int | None = None
        self._invalid_local = False

    def reset(self) -> None:
        """Reset while no producer or consumer is using the pool."""
        self._pool._require_mapping()
        self._consumed.store_release(0)
        self._published.store_release(0)
        self._invalid.store_release(_VALID)
        self._published_local = 0
        self._consumed_local = 0
        self._invalid_local = False

    def publish(self, record_type: int, key: bytes, payload: bytes) -> bool:
        """Publish one typed keyed frame, or permanently invalidate the journal."""
        mapping = self._pool._require_mapping()
        if record_type not in _RECORD_TYPES:
            raise ValueError(f"unknown journal record type: {record_type}")
        if not isinstance(key, bytes):
            raise TypeError("journal key must be bytes")
        if not isinstance(payload, bytes):
            raise TypeError("journal payload must be bytes")
        if self._invalid_local:
            return False

        frame_size = _FRAME_HEADER.size + len(key) + len(payload)
        if frame_size > _MAX_FRAME_SIZE:
            return self._invalidate(
                f"frame is {frame_size} bytes; maximum is {_MAX_FRAME_SIZE} bytes"
            )
        stored_size = _align_up(frame_size)
        published = self._published_local
        if published is None:
            if self.is_invalid():
                self._invalid_local = True
                return False
            published = self._published.load_acquire()
        # Acquire: what follows writes into space this cursor declares free, and
        # those writes are unordered against the consumer's reads on ARM.
        consumed = self._consumed.load_acquire()
        if not self._valid_cursors(published, consumed):
            return self._invalidate(
                f"publish cursors are invalid: {published=} {consumed=}"
            )
        if stored_size > self._capacity - (published - consumed):
            return self._invalidate(
                f"ring is full: {stored_size} bytes needed, "
                f"{self._capacity - (published - consumed)} available"
            )

        frame = _pack_frame(record_type, key, payload, stored_size)
        self._write_ring(mapping, published % self._capacity, frame)
        # Ordered: every byte above must be visible before the cursor offering it.
        self._published_local = published + stored_size
        self._published.store_release(self._published_local)
        return True

    def read_next(self) -> tuple[int, bytes, bytes] | None:
        mapping = self._pool._require_mapping()
        if self._invalid_local:
            raise RecoveryJournalError("recovery journal is invalid")
        consumed = self._consumed_local
        if consumed is None:
            if self.is_invalid():
                self._invalid_local = True
                raise RecoveryJournalError("recovery journal is invalid")
            consumed = self._consumed_local = self._consumed.load_acquire()
        # Acquire: the frame bytes below must not be hoisted above the cursor that
        # says they are there. x86 gives that to any load; ARM does not.
        published = self._published.load_acquire()
        if not self._valid_cursors(published, consumed):
            raise self._fail("recovery journal cursor is invalid")
        if published == consumed:
            # Caught up looks the same as given up on, so the flag is re-read here --
            # and only here, where it costs the idle poll rather than the reading path.
            if self.is_invalid():
                self._invalid_local = True
                raise RecoveryJournalError("recovery journal is invalid")
            return None

        position = consumed % self._capacity
        frame_size, record_type, key_size = _FRAME_HEADER.unpack(
            self._read_ring(mapping, position, _FRAME_HEADER.size)
        )
        if frame_size < _FRAME_HEADER.size + key_size:
            raise self._fail("journal frame has an invalid key size")
        if record_type not in _RECORD_TYPES:
            raise self._fail(f"journal frame has an unknown type: {record_type}")
        stored_size = _align_up(frame_size)
        available = published - consumed
        if stored_size > self._capacity or stored_size > available:
            raise self._fail("journal frame exceeds the published bytes")

        frame = self._read_ring(mapping, position, stored_size)
        if any(frame[frame_size:]):
            raise self._fail("journal frame padding is not zero")
        key_start = _FRAME_HEADER.size
        key_end = key_start + key_size
        self._consumed_local = consumed + stored_size
        self._consumed.store_release(self._consumed_local)
        return record_type, frame[key_start:key_end], frame[key_end:frame_size]

    def drain(self) -> Iterator[tuple[int, bytes, bytes]]:
        """Every frame the producer left behind, for one that cannot add more.

        Only correct once the writing process is reaped: that is what makes an empty
        ring a complete one. Invalidation raises out of read_next.
        """
        while (record := self.read_next()) is not None:
            yield record

    def is_invalid(self) -> bool:
        self._pool._require_mapping()
        return self._invalid.load_acquire() != _VALID

    def invalidate(self) -> bool:
        """Permanently disable recovery publication for this lease."""
        self._pool._require_mapping()
        return self._invalidate("invalidated by its owner")

    def _valid_cursors(self, published: int, consumed: int) -> bool:
        return (
            published % _ALIGNMENT == 0
            and consumed % _ALIGNMENT == 0
            and consumed <= published <= consumed + self._capacity
        )

    def _invalidate(self, reason: str) -> bool:
        logger.warning("KVCR recovery journal invalidated: %s", reason)
        self._invalid_local = True
        self._invalid.store_release(_INVALID)
        return False

    def _fail(self, message: str) -> RecoveryJournalError:
        self._invalidate(message)
        return RecoveryJournalError(message)

    def _write_ring(self, mapping: object, position: int, data: bytes) -> None:
        first = min(len(data), self._capacity - position)
        start = _JOURNAL_HEADER_BYTES + position
        mapping[start : start + first] = data[:first]
        if first < len(data):
            remaining = len(data) - first
            mapping[_JOURNAL_HEADER_BYTES : _JOURNAL_HEADER_BYTES + remaining] = data[
                first:
            ]

    def _read_ring(self, mapping: object, position: int, length: int) -> bytes:
        first = min(length, self._capacity - position)
        start = _JOURNAL_HEADER_BYTES + position
        data = bytes(mapping[start : start + first])
        if first < length:
            remaining = length - first
            data += bytes(
                mapping[_JOURNAL_HEADER_BYTES : _JOURNAL_HEADER_BYTES + remaining]
            )
        return data


class _RecoveryMirror:
    def __init__(self) -> None:
        self._records: dict[BlockKey, _BlockRecord] = {}

    def apply(self, record_type: int, key: bytes, payload: bytes) -> None:
        # Taken so a frame can be applied as it is read; the type is refused
        # where frames are published and read, not again here.
        del record_type
        try:
            record = _decode_recovery_record(payload)
        except (TypeError, ValueError, msgspec.DecodeError) as error:
            raise RecoveryMirrorError("recovery record is malformed") from error
        block_key = BlockKey(key)
        if record.local_dram is None and record.g3 is None:
            self._records.pop(block_key, None)
        else:
            self._records[block_key] = record

    def adopt(self, records: dict[BlockKey, _BlockRecord]) -> None:
        """Take a closed core's records as this mirror's state, canonicalized in place.

        In place because rebuilding costs a second copy of the set per million
        blocks. Canonicalizing is required: a half-written G2 slot would otherwise be
        seated as READY.
        """
        stale = []
        for key, record in records.items():
            local_dram = record.local_dram
            if local_dram is not None:
                if local_dram.state is not _LocalDramState.READY:
                    record.local_dram = None
                else:
                    local_dram.claim_count = 0
                    local_dram.retire_on_release = False
            if record.g3 is not None:
                record.g3.claim_count = 0
            record.fw_mem = None
            record.in_flight_ops = None
            record.access_count = 0
            record.last_access = None
            if not _is_recoverable(record):
                stale.append(key)
        for key in stale:
            del records[key]
        self._records = records

    def take_records(self) -> dict[BlockKey, _BlockRecord]:
        """Hand the records over, leaving this mirror empty.

        A promoted core is the sole owner from that point, so copying the table
        would leave two complete populations of the set alive at the moment a
        worker has just died.
        """
        records, self._records = self._records, {}
        return records


def _attach_journal(
    local_dram: _LocalDram,
    journal: RecoveryJournal,
    g3: _G3 | None = None,
) -> None:
    """Attach stable G2/G3 residency publication to one journal."""
    enabled = True

    def publish_frame(record_type: int, key: bytes, payload: bytes) -> None:
        nonlocal enabled
        if not enabled:
            return
        try:
            if not journal.publish(record_type, key, payload):
                logger.warning(
                    "KVCR recovery publication disabled after the journal "
                    "rejected a frame"
                )
                enabled = False
        except Exception:
            logger.warning("KVCR recovery publication failed", exc_info=True)
            enabled = False
            with suppress(Exception):
                journal.invalidate()

    def publish(key: BlockKey, record: _BlockRecord) -> None:
        # TODO: Publish per-tier deltas if full-record journal traffic is material.
        recovered = _project_recovery_record(record)
        publish_frame(_RECORD_BLOCK, bytes(key), _RECOVERY_ENCODER.encode(recovered))

    if g3 is not None:
        g3.observe_residency(publish)
    local_dram.observe_residency(publish)


def install_recovery_records(
    core: _KVCRCore, records: dict[BlockKey, _BlockRecord]
) -> None:
    """Seed a core that has not started with recovered residencies.

    The mechanics live on the core: seeding, admission and ranking touch
    invariants only the core owns.
    """
    core.adopt_recovery_records(records)


@dataclass
class ClaimedPool:
    """A pool leased from the KVCR-Service and whatever was left in it."""

    hold: KVCRPoolHold
    recovered: _RecoveryMirror
    adopt_listener: Callable[[int], None]


def claim_guarded_pool(
    guard_config: KVCRGuardConfig,
    bindings: "KVCRBindings",
    backend_configs: KVCRBackendConfigs,
) -> ClaimedPool:
    """Lease a service-owned pool and read the state a Guard left behind.

    A guard_config opts into both, so failing to get either is a failure. A Guard
    answers on a listener the service owns, so a control that cannot adopt one
    cannot be guarded.
    """
    if backend_configs.local_dram is not None:
        raise ValueError("guard_config conflicts with backend_configs.local_dram")
    # Duck-typed: what matters is whether the framework's control can hand its
    # endpoint over, not what class it is.
    framework_control = bindings.framework_control
    bind_address = getattr(framework_control, "control_bind_address", None)
    adopt_listener = getattr(framework_control, "adopt_listener", None)
    if not callable(bind_address) or not callable(adopt_listener):
        raise ValueError(
            "guard_config needs a framework control that can share its control "
            "endpoint, so that a Guard can answer on it after this worker dies"
        )
    control_bind = bind_address()
    hold = KVCRClient(guard_config.kvcr_service_socket_path).claim(
        guard_config.pool_index,
        guard_config.row_stride,
        guard_config.compatibility_digest,
        control_bind,
        backend_configs.g3,
    )
    # The lease is live from here, and the caller cannot release what it has not
    # been handed yet: anything that fails before this returns has to give the pool
    # back itself or leave it held until the process exits.
    try:
        recovered = read_handback(
            hold._attachment,
            guard_config.compatibility_digest,
            guard_config.row_stride,
        )
    except BaseException:
        # A failing release must not mask the error that made the claim unusable.
        with suppress(BaseException):
            hold.release()
        raise
    return ClaimedPool(hold, recovered, adopt_listener)


def claimed_core(
    config: KVCRConfig,
    bindings: "KVCRBindings",
    backend_configs: KVCRBackendConfigs,
    claimed: ClaimedPool,
) -> _KVCRCore:
    """A core over the pool this claim was granted, built but not yet wired.

    Only construction, so the caller owns the core before anything that can
    fail runs against it: a startup that released the pool without closing the
    core would leave it mapping bytes the next claimant is given.
    """
    return _KVCRCore(
        config,
        bindings,
        replace(backend_configs, local_dram=claimed.hold.local_dram),
    )


def adopt_claimed_pool(core: _KVCRCore, claimed: ClaimedPool) -> None:
    """Wire a core the caller already owns to the pool it was granted.

    The listener is adopted last, and disowned only once the channel has taken
    it: anything that fails before that leaves the hold owning the descriptor,
    which is what closes it on release.
    """
    hold = claimed.hold
    if core._local_dram is None:
        raise ValueError("a claimed pool must give the core its local DRAM tier")
    _attach_journal(core._local_dram, RecoveryJournal(hold._attachment), core._g3)
    install_recovery_records(core, claimed.recovered.take_records())
    hold.hand_listener_to(claimed.adopt_listener)


def commit_claimed_pool(claimed: ClaimedPool | None) -> None:
    """Take ownership of the recovered slots once the replacement is serving.

    Dropping the region before start would strand the cache if start failed.
    No inventory is announced for recovered blocks: the router learned them
    from the primary that stored them, on a control endpoint that never moves.
    """
    if claimed is None:
        return
    clear_recovery_snapshot(claimed.hold._attachment)


def _pack_frame(record_type: int, key: bytes, payload: bytes, size: int) -> bytearray:
    """One typed keyed frame, zero-padded out to size.

    The ring pads to its alignment and a handback region does not, but the reader
    parses one shape, so it is written in one place.
    """
    frame = bytearray(size)
    frame_size = _FRAME_HEADER.size + len(key) + len(payload)
    _FRAME_HEADER.pack_into(frame, 0, frame_size, record_type, len(key))
    key_end = _FRAME_HEADER.size + len(key)
    frame[_FRAME_HEADER.size : key_end] = key
    frame[key_end:frame_size] = payload
    return frame


def _align_up(value: int) -> int:
    return value + (-value % _ALIGNMENT)


def _recovery_frames(
    records: Mapping[BlockKey, _BlockRecord],
) -> Iterator[tuple[int, bytes, bytes]]:
    """Every frame a returning primary needs to rebuild this state."""
    for key, record in records.items():
        if not _is_recoverable(record):
            continue
        payload = _RECOVERY_ENCODER.encode(_project_recovery_record(record))
        yield _RECORD_BLOCK, bytes(key), payload


# Bound to the pool and to the geometry: a slot index only means the same
# bytes under the same file and layout. The generation stops a replay into a
# different pool of the same shape; the digest separates finished from filling.
_SNAPSHOT_HEADER = struct.Struct("<32sQ")
_SNAPSHOT_DOMAIN = b"KVCR-HANDBACK\0"
_SNAPSHOT_TERMS = struct.Struct("<QQQQQ")


def canonical_pool_terms(
    compatibility_digest: str, row_stride: int, spec: "KVCRPoolSpec"
) -> bytes:
    """Encode what a handback region must not be replayed across."""
    return (
        _SNAPSHOT_DOMAIN
        + compatibility_digest.encode()
        + b"\0"
        + bytes.fromhex(spec.generation)
        + _SNAPSHOT_TERMS.pack(
            row_stride,
            spec.journal_bytes,
            spec.mapping_bytes,
            spec.device,
            spec.inode,
        )
    )


# A slice at a time: hashing whole would copy a tier's worth of records, and
# a memoryview would keep the mmap exported so it could not be closed.
_DIGEST_CHUNK_BYTES = 1 << 20


def _snapshot_digest(terms: bytes, mapping: mmap.mmap, start: int, size: int) -> bytes:
    digest = hashlib.sha256(terms)
    for offset in range(start, start + size, _DIGEST_CHUNK_BYTES):
        digest.update(mapping[offset : min(offset + _DIGEST_CHUNK_BYTES, start + size)])
    return digest.digest()


def write_recovery_snapshot(
    pool: KVCRPoolAttachment,
    terms: bytes,
    frames: Iterable[tuple[int, bytes, bytes]],
) -> None:
    """Publish this pool's state as a handback region past the pool itself.

    In the pool's own file, so no second name can be orphaned or replaced. The
    header lands last: until the digest is there the region is unfinished.
    """
    body = bytearray()
    for record_type, key, payload in frames:
        frame_size = _FRAME_HEADER.size + len(key) + len(payload)
        if frame_size > _MAX_FRAME_SIZE:
            raise RecoveryJournalError(
                f"handback frame is {frame_size} bytes; "
                f"maximum is {_MAX_FRAME_SIZE} bytes"
            )
        body += _pack_frame(record_type, key, payload, frame_size)
    if not body:
        pool.release_snapshot_region()
        return

    digest = hashlib.sha256(terms)
    digest.update(body)
    with pool.snapshot_region(_SNAPSHOT_HEADER.size + len(body)) as region:
        # A previous region's header is retired before this body overwrites what it
        # describes. Left there, an interrupted rewrite would read as written for
        # other terms -- which is refused permanently -- instead of as unfinished,
        # which is discarded.
        region[: _SNAPSHOT_HEADER.size] = bytes(_SNAPSHOT_HEADER.size)
        region.flush()
        region[_SNAPSHOT_HEADER.size :] = body
        region[: _SNAPSHOT_HEADER.size] = _SNAPSHOT_HEADER.pack(
            digest.digest(), len(body)
        )
        region.flush()


def read_recovery_snapshot(
    pool: KVCRPoolAttachment, terms: bytes
) -> Iterator[tuple[int, bytes, bytes]]:
    """Replay the pool's handback region, if a finished one is there.

    Nothing there is the ordinary case. A digest that does not match these terms
    means a different geometry or an unfinished write; either is refused.
    """
    with pool.mapped_snapshot() as region:
        if region is None:
            return
        available = len(region)
        if available < _SNAPSHOT_HEADER.size:
            raise RecoveryJournalTornError("handback region is truncated")
        digest, size = _SNAPSHOT_HEADER.unpack_from(region, 0)
        if not size:
            # An empty snapshot is truncated away and the header lands last, so a region
            # here declaring nothing never finished being written.
            raise RecoveryJournalTornError("handback region is unfinished")
        if size > available - _SNAPSHOT_HEADER.size:
            raise RecoveryJournalTornError("handback region length disagrees")
        expected = _snapshot_digest(terms, region, _SNAPSHOT_HEADER.size, size)
        if not hmac.compare_digest(digest, expected):
            raise RecoveryJournalError(
                "handback region is unfinished or was written for other terms"
            )
        end = _SNAPSHOT_HEADER.size + size
        offset = _SNAPSHOT_HEADER.size
        while offset < end:
            if end - offset < _FRAME_HEADER.size:
                raise RecoveryJournalError("handback frame header is truncated")
            frame_size, record_type, key_size = _FRAME_HEADER.unpack_from(
                region, offset
            )
            if frame_size < _FRAME_HEADER.size + key_size or offset + frame_size > end:
                raise RecoveryJournalError("handback frame is malformed")
            if record_type not in _RECORD_TYPES:
                raise RecoveryJournalError(
                    f"handback frame has an unknown type: {record_type}"
                )
            key_start = offset + _FRAME_HEADER.size
            key_end = key_start + key_size
            yield (
                record_type,
                bytes(region[key_start:key_end]),
                bytes(region[key_end : offset + frame_size]),
            )
            offset += frame_size


def read_handback(
    pool: KVCRPoolAttachment, compatibility_digest: str, row_stride: int
) -> _RecoveryMirror:
    """Replay whatever the last Guard left for this pool, if anything.

    A region that disagrees about terms is refused, not discarded: every pool of
    a service runs one configuration for its lifetime, so terms that disagree
    mean something is wrong rather than something has moved on. A region whose
    write never finished is this service's own, and is thrown away -- nothing
    else ever would, and it would refuse every later claim on this pool too.
    """
    mirror = _RecoveryMirror()
    terms = canonical_pool_terms(compatibility_digest, row_stride, pool._spec)
    try:
        for frame in read_recovery_snapshot(pool, terms):
            mirror.apply(*frame)
    except RecoveryJournalTornError:
        logger.warning(
            "KVCR discarding a handback region that was never finished",
            exc_info=True,
        )
        clear_recovery_snapshot(pool)
        return _RecoveryMirror()
    return mirror


def clear_recovery_snapshot(pool: KVCRPoolAttachment) -> None:
    """Drop handback state once it has been installed."""
    pool.release_snapshot_region()
