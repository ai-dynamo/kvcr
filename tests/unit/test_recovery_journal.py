# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import mmap
import struct
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from _kvcr_test_utils import _recovered_record

from kvcr.core import _BlockRecord
from kvcr.memory import KVCRPoolAttachment, KVCRPoolSpec, _KVCRPoolOwner
from kvcr.recovery_journal import (
    _JOURNAL_HEADER_BYTES,
    _RECORD_BLOCK,
    _SNAPSHOT_HEADER,
    RecoveryJournal,
    RecoveryJournalError,
    RecoveryJournalTornError,
    _attach_journal,
    _decode_recovery_record,
    _recovery_frames,
    _RecoveryMirror,
    canonical_pool_terms,
    read_handback,
    read_recovery_snapshot,
    write_recovery_snapshot,
)
from kvcr.types import BlockKey

_TEST_JOURNAL_BYTES = 2 * _JOURNAL_HEADER_BYTES
_INVALID_OFFSET = 0
_PUBLISHED_OFFSET = 128
_CONSUMED_OFFSET = 192
_GENERATION = "a" * 32
_TEST_DIGEST = "Opaque-Digest"


def _attachment(mapping: mmap.mmap, journal_bytes: int) -> KVCRPoolAttachment:
    attachment = object.__new__(KVCRPoolAttachment)
    attachment._mapping = mapping
    attachment._spec = KVCRPoolSpec(
        pool_id="pool_0",
        path=f"/tmp/kvcr-pool_0-{_GENERATION}",
        generation=_GENERATION,
        device=0,
        inode=0,
        mapping_bytes=len(mapping),
        journal_bytes=journal_bytes,
    )
    return attachment


@pytest.fixture
def journal_and_mapping() -> Iterator[tuple[RecoveryJournal, mmap.mmap]]:
    mapping = mmap.mmap(-1, _TEST_JOURNAL_BYTES + mmap.PAGESIZE)
    journal = RecoveryJournal(_attachment(mapping, _TEST_JOURNAL_BYTES))
    journal.reset()
    try:
        yield journal, mapping
    finally:
        mapping.close()


def test_ring_round_trips_wraps_and_drains_streaming(
    journal_and_mapping: tuple[RecoveryJournal, mmap.mmap],
) -> None:
    """Frames land at documented offsets, replay in order, wrap, and drain lazily."""
    journal, mapping = journal_and_mapping
    records = [
        (b"", b"serialized block record"),
        (b"k", b""),
        (b"variable-width-key", bytes(range(31))),
    ]
    for key, payload in records:
        assert journal.publish(_RECORD_BLOCK, key, payload)

    assert struct.unpack_from("<Q", mapping, _INVALID_OFFSET) == (0,)
    assert struct.unpack_from("<Q", mapping, _PUBLISHED_OFFSET) == (
        sum((6 + len(key) + len(payload) + 7) // 8 * 8 for key, payload in records),
    )
    assert struct.unpack_from("<Q", mapping, _CONSUMED_OFFSET) == (0,)
    assert struct.unpack_from("<HHH", mapping, _JOURNAL_HEADER_BYTES) == (
        6 + len(records[0][0]) + len(records[0][1]),
        _RECORD_BLOCK,
        len(records[0][0]),
    )
    assert [journal.read_next() for _ in records] == [
        (_RECORD_BLOCK, key, payload) for key, payload in records
    ]
    assert journal.read_next() is None

    for index in range(400):
        key = index.to_bytes(4, "little")
        payload = bytes([index % 256]) * 9
        assert journal.publish(_RECORD_BLOCK, key, payload)
        assert journal.read_next() == (_RECORD_BLOCK, key, payload)
    assert journal._published.load_acquire() > journal._capacity
    assert journal._consumed.load_acquire() == journal._published.load_acquire()
    assert not journal.is_invalid()

    # Streaming, so a promotion does not hold a second copy of the ring.
    for index in range(3):
        assert journal.publish(_RECORD_BLOCK, bytes([index]), bytes([index + 3]))
    published = journal._published.load_acquire()
    drained = journal.drain()
    assert next(drained) == (_RECORD_BLOCK, b"\x00", b"\x03")
    assert journal._consumed.load_acquire() < published
    assert list(drained) == [
        (_RECORD_BLOCK, b"\x01", b"\x04"),
        (_RECORD_BLOCK, b"\x02", b"\x05"),
    ]
    assert journal._consumed.load_acquire() == published


class _Source:
    def __init__(self) -> None:
        self._observer: Callable[[BlockKey, _BlockRecord], None] | None = None

    def observe_residency(
        self, observer: Callable[[BlockKey, _BlockRecord], None]
    ) -> None:
        self._observer = observer

    def emit(self, key: BlockKey, record: _BlockRecord) -> None:
        assert self._observer is not None
        self._observer(key, record)


def test_publisher_streams_mutations_until_the_journal_refuses_or_fails(
    journal_and_mapping: tuple[RecoveryJournal, mmap.mmap],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Stable mutations stream through the ring; a refusal or failure stops all."""
    journal, _ = journal_and_mapping
    local_dram, g3 = _Source(), _Source()
    key = BlockKey(b"block")
    _attach_journal(local_dram, journal, g3)
    caplog.set_level("WARNING", logger="kvcr.recovery_journal")

    local_dram.emit(key, _recovered_record(g2=2))
    g3.emit(key, _recovered_record(g2=2, g3=7))
    local_dram.emit(key, _recovered_record(g3=7))
    g3.emit(key, _BlockRecord())
    frames = [journal.read_next() for _ in range(4)]
    assert journal.read_next() is None
    assert [_decode_recovery_record(payload) for _, _, payload in frames] == [
        _recovered_record(g2=2),
        _recovered_record(g2=2, g3=7),
        _recovered_record(g3=7),
        _BlockRecord(),
    ]
    assert caplog.messages == []

    # One refused attempt, then publication stays off with one warning: the
    # count is the contract, not the wording.
    journal.invalidate()
    caplog.clear()
    with patch.object(journal, "publish", wraps=journal.publish) as publish:
        local_dram.emit(key, _recovered_record(g2=0))
        local_dram.emit(key, _recovered_record(g2=1))
    assert publish.call_count == 1
    assert len(caplog.messages) == 1

    # A publish that raises invalidates recovery; the mutation itself survives.
    with mmap.mmap(-1, _TEST_JOURNAL_BYTES + mmap.PAGESIZE) as fresh_mapping:
        fresh = RecoveryJournal(_attachment(fresh_mapping, _TEST_JOURNAL_BYTES))
        fresh.reset()
        source = _Source()
        _attach_journal(source, fresh)
        assert not fresh.is_invalid()
        with patch.object(fresh, "publish", side_effect=RuntimeError("publish failed")):
            source.emit(BlockKey(b"still-serving"), _recovered_record(g2=0))
        assert fresh.is_invalid()


def test_invalidation_stops_every_role_until_an_owner_reset(
    journal_and_mapping: tuple[RecoveryJournal, mmap.mmap],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Oversize, overflow, or corruption kills the ring for every role until reset."""
    journal, mapping = journal_and_mapping
    caplog.set_level("WARNING", logger="kvcr.recovery_journal")
    consumer = RecoveryJournal(_attachment(mapping, _TEST_JOURNAL_BYTES))
    assert journal.publish(_RECORD_BLOCK, b"key0", b"record")
    assert consumer.read_next() is not None
    assert consumer.read_next() is None

    # A frame past the uint16 size header is refused without publication.
    before = journal._published.load_acquire()
    assert not journal.publish(_RECORD_BLOCK, b"key", bytes(1 << 16))
    assert journal._published.load_acquire() == before
    assert journal.is_invalid()
    assert "frame is 65545 bytes; maximum is 65535 bytes" in caplog.text

    # Caught up and finished look identical unless the shared flag is re-read.
    with pytest.raises(RecoveryJournalError, match="invalid"):
        consumer.read_next()
    # A role that attaches to a finished journal never starts.
    late = RecoveryJournal(_attachment(mapping, _TEST_JOURNAL_BYTES))
    assert not late.publish(_RECORD_BLOCK, b"key", b"payload")
    with pytest.raises(RecoveryJournalError, match="invalid"):
        RecoveryJournal(_attachment(mapping, _TEST_JOURNAL_BYTES)).read_next()

    journal.reset()
    assert journal._published.load_acquire() == 0
    assert journal._consumed.load_acquire() == 0
    assert not journal.is_invalid()

    # A ring without room for the next frame invalidates whole, not partially.
    published = 0
    while journal.publish(_RECORD_BLOCK, b"key", bytes(1000)):
        published = journal._published.load_acquire()
    assert published > 0
    assert journal._published.load_acquire() == published
    assert journal.is_invalid()
    with pytest.raises(RecoveryJournalError, match="invalid"):
        journal.read_next()
    with pytest.raises(RecoveryJournalError, match="invalid"):
        list(journal.drain())

    # A frame header declaring less than its own key is a corrupt ring.
    journal.reset()
    assert journal.publish(_RECORD_BLOCK, b"key", b"payload")
    struct.pack_into("<HH", mapping, _JOURNAL_HEADER_BYTES, 2, 0)
    with pytest.raises(RecoveryJournalError, match="key size"):
        journal.read_next()
    assert journal.is_invalid()

    # Alignment padding the producer left as zeros must still be zeros.
    journal.reset()
    assert journal.publish(_RECORD_BLOCK, b"keys", b"x")
    mapping[_JOURNAL_HEADER_BYTES + 12] = 1
    with pytest.raises(RecoveryJournalError, match="padding"):
        journal.read_next()
    assert journal.is_invalid()

    # An owner's own declaration kills the ring exactly the same way, and a
    # consumer that was caught up re-reads the shared flag to notice it.
    journal.reset()
    assert journal.publish(_RECORD_BLOCK, b"key1", b"record")
    fresh = RecoveryJournal(_attachment(mapping, _TEST_JOURNAL_BYTES))
    assert fresh.read_next() is not None
    assert fresh.read_next() is None
    assert journal.invalidate() is False
    with pytest.raises(RecoveryJournalError, match="invalid"):
        fresh.read_next()


@contextmanager
def _attached(tmp_path: Path) -> Iterator[KVCRPoolAttachment]:
    """An owned pool, mapped, the way a Guard or a claimant holds one."""
    owner = _KVCRPoolOwner.allocate(
        pool_id="pool_0",
        pool_size_bytes=8192 + 4096,
        journal_bytes=8192,
        pool_dir=tmp_path,
    )
    try:
        attachment = KVCRPoolAttachment.attach(owner.spec)
        try:
            yield attachment
        finally:
            attachment.close()
    finally:
        owner.close()


def _write_slot(pool: KVCRPoolAttachment, terms: bytes, key: bytes, slot: int) -> None:
    """One-slot handback region: the smallest finished snapshot."""
    frames = _recovery_frames({BlockKey(key * 32): _recovered_record(g2=slot)})
    write_recovery_snapshot(pool, terms, frames)


def test_a_handback_region_lives_and_dies_inside_the_pool_file(tmp_path: Path) -> None:
    """Replayed whole under its own terms, discardable when torn, gone once released."""
    with _attached(tmp_path) as pool:
        path = Path(pool._spec.path)
        terms = canonical_pool_terms(_TEST_DIGEST, 4096, pool._spec)
        assert list(read_recovery_snapshot(pool, terms)) == []

        records = {
            BlockKey(b"a" * 32): _recovered_record(g2=3),
            # One with both halves, one only on disk.
            BlockKey(b"b" * 32): _recovered_record(g2=4, g3=9),
            BlockKey(b"c" * 32): _recovered_record(g3=2),
        }
        write_recovery_snapshot(pool, terms, _recovery_frames(records))
        # Inside the pool file, so it has no name of its own to be found under.
        assert set(tmp_path.iterdir()) == {path}
        assert path.stat().st_size > pool._spec.mapping_bytes

        # The mirror the ring feeds is also what replays the region.
        mirror = _RecoveryMirror()
        for frame in read_recovery_snapshot(pool, terms):
            mirror.apply(*frame)
        assert mirror.take_records() == records

        # A slot number only means the same bytes under the same geometry.
        for other in (
            canonical_pool_terms("another-digest", 4096, pool._spec),
            canonical_pool_terms(_TEST_DIGEST, 8192, pool._spec),
        ):
            with pytest.raises(RecoveryJournalError, match="other terms"):
                list(read_recovery_snapshot(pool, other))

        # Stopped once the replacing body has landed but before its header has.
        interrupted = Mock(
            size=_SNAPSHOT_HEADER.size, pack=Mock(side_effect=OSError("interrupted"))
        )
        with patch("kvcr.recovery_journal._SNAPSHOT_HEADER", interrupted):
            with pytest.raises(OSError, match="interrupted"):
                _write_slot(pool, terms, b"b", 2)

        # The failed rewrite took the tail with it -- including the previous
        # snapshot. Old frames must not replay against a Guard whose mirror
        # already gave up on this handover.
        assert list(read_recovery_snapshot(pool, terms)) == []

        # A crash leaves torn bytes no exception path can truncate: a region
        # whose header never landed reads as unfinished, and read_handback
        # discards it rather than serving it as another handover's terms.
        _write_slot(pool, terms, b"c", 3)
        with pool.snapshot_region(_SNAPSHOT_HEADER.size + 64) as region:
            region[: _SNAPSHOT_HEADER.size] = bytes(_SNAPSHOT_HEADER.size)
        with pytest.raises(RecoveryJournalTornError, match="unfinished"):
            list(read_recovery_snapshot(pool, terms))
        assert read_handback(pool, _TEST_DIGEST, 4096)._records == {}
        assert list(read_recovery_snapshot(pool, terms)) == []

        # A released region is truncated away, so it replays nothing.
        _write_slot(pool, terms, b"d", 4)
        assert list(read_recovery_snapshot(pool, terms))
        pool.release_snapshot_region()
        assert list(read_recovery_snapshot(pool, terms)) == []
