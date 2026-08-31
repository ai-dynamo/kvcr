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
from kvcr.memory import (
    KVCRPoolAttachment,
    KVCRPoolSpec,
    _KVCRPoolOwner,
    _snapshot_offset,
)
from kvcr.recovery_journal import (
    _JOURNAL_HEADER_BYTES,
    _RECORD_BLOCK,
    _SNAPSHOT_HEADER,
    RecoveryJournal,
    RecoveryJournalError,
    RecoveryJournalTornError,
    _recovery_frames,
    _RecoveryMirror,
    canonical_pool_terms,
    clear_recovery_snapshot,
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


def test_a_journal_round_trips_wraps_and_dies_by_invalidation(
    journal_and_mapping: tuple[RecoveryJournal, mmap.mmap],
) -> None:
    """Publish/read, drain, wrap, then invalidation stops current and future roles."""
    journal, mapping = journal_and_mapping

    # Variable-width keys and opaque payloads land in the documented layout.
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

    # Drain streams, so a promotion does not hold a second copy of the ring:
    # _consumed/_published are the ring's byte cursors, and after one item the
    # read cursor sits strictly between where it started and the write cursor.
    for index in range(3):
        assert journal.publish(_RECORD_BLOCK, bytes([index]), bytes([index + 3]))
    consumed = journal._consumed.load_acquire()
    published = journal._published.load_acquire()
    drained = journal.drain()
    assert next(drained) == (_RECORD_BLOCK, b"\x00", b"\x03")
    assert consumed < journal._consumed.load_acquire() < published
    assert list(drained) == [
        (_RECORD_BLOCK, b"\x01", b"\x04"),
        (_RECORD_BLOCK, b"\x02", b"\x05"),
    ]
    assert journal._consumed.load_acquire() == published

    # The ring wraps without losing records: the write cursor passes capacity.
    for index in range(400):
        key = index.to_bytes(4, "little")
        payload = bytes([index % 256]) * 9
        assert journal.publish(_RECORD_BLOCK, key, payload)
        assert journal.read_next() == (_RECORD_BLOCK, key, payload)
    assert journal._published.load_acquire() > journal._capacity
    assert journal._consumed.load_acquire() == journal._published.load_acquire()
    assert not journal.is_invalid()

    # A caught-up mirror looks finished; only re-reading the shared flag tells
    # it the journal was invalidated by a role other than itself.
    consumer = RecoveryJournal(_attachment(mapping, _TEST_JOURNAL_BYTES))
    assert journal.publish(_RECORD_BLOCK, b"key", b"payload")
    assert consumer.read_next() is not None
    assert consumer.read_next() is None
    assert journal.invalidate() is False
    with pytest.raises(RecoveryJournalError, match="invalid"):
        consumer.read_next()

    # Roles that attach after the death never start at all.
    late_producer = RecoveryJournal(_attachment(mapping, _TEST_JOURNAL_BYTES))
    late_consumer = RecoveryJournal(_attachment(mapping, _TEST_JOURNAL_BYTES))
    assert not late_producer.publish(_RECORD_BLOCK, b"key", b"payload")
    with pytest.raises(RecoveryJournalError, match="invalid"):
        late_consumer.read_next()


def _publish_oversize_frame(
    journal: RecoveryJournal, mapping: mmap.mmap, caplog: pytest.LogCaptureFixture
) -> None:
    """A frame over uint16 is refused whole: _published, the write cursor, holds."""
    assert journal.publish(_RECORD_BLOCK, b"key0", b"record")
    published = journal._published.load_acquire()

    assert not journal.publish(_RECORD_BLOCK, b"key", bytes(1 << 16))

    assert journal._published.load_acquire() == published
    assert caplog.messages == [
        "KVCR recovery journal invalidated: frame is 65545 bytes; "
        "maximum is 65535 bytes"
    ]


def _fill_ring(
    journal: RecoveryJournal, mapping: mmap.mmap, caplog: pytest.LogCaptureFixture
) -> None:
    """The frame that finds the ring full is refused whole: the cursor holds."""
    published = 0
    while True:
        before = journal._published.load_acquire()
        if not journal.publish(_RECORD_BLOCK, b"key", bytes(1000)):
            break
        published += 1

    assert published > 0
    assert journal._published.load_acquire() == before


def _tear_frame_key_size(
    journal: RecoveryJournal, mapping: mmap.mmap, caplog: pytest.LogCaptureFixture
) -> None:
    """A published frame whose key size no longer fits its frame size."""
    assert journal.publish(_RECORD_BLOCK, b"key", b"payload")
    struct.pack_into("<HH", mapping, _JOURNAL_HEADER_BYTES, 2, 0)


def _tear_frame_padding(
    journal: RecoveryJournal, mapping: mmap.mmap, caplog: pytest.LogCaptureFixture
) -> None:
    """A published frame whose padding is no longer zero."""
    assert journal.publish(_RECORD_BLOCK, b"keys", b"x")
    mapping[_JOURNAL_HEADER_BYTES + 12] = 1


@pytest.mark.parametrize(
    "provoke,error",
    [
        (_publish_oversize_frame, "invalid"),
        (_fill_ring, "invalid"),
        (_tear_frame_key_size, "key size"),
        (_tear_frame_padding, "padding"),
    ],
    ids=["oversize-frame", "full-ring", "torn-key-size", "torn-padding"],
)
def test_a_frame_the_ring_cannot_carry_invalidates_it_until_reset(
    journal_and_mapping: tuple[RecoveryJournal, mmap.mmap],
    caplog: pytest.LogCaptureFixture,
    provoke: Callable[[RecoveryJournal, mmap.mmap, pytest.LogCaptureFixture], None],
    error: str,
) -> None:
    """Any frame the ring cannot carry invalidates it wholesale, until a reset."""
    journal, mapping = journal_and_mapping
    caplog.set_level("WARNING", logger="kvcr.recovery_journal")
    provoke(journal, mapping, caplog)

    with pytest.raises(RecoveryJournalError, match=error):
        journal.read_next()
    assert journal.is_invalid()
    with pytest.raises(RecoveryJournalError, match="invalid"):
        list(journal.drain())

    journal.reset()

    assert not journal.is_invalid()
    assert journal.read_next() is None
    assert journal.publish(_RECORD_BLOCK, b"key", b"payload")
    assert journal.read_next() == (_RECORD_BLOCK, b"key", b"payload")


def _pool(tmp_path: Path, pool_id: str = "pool_0") -> _KVCRPoolOwner:
    return _KVCRPoolOwner.allocate(
        pool_id=pool_id,
        pool_size_bytes=8192 + 4096,
        journal_bytes=8192,
        pool_dir=tmp_path,
    )


@contextmanager
def _attached(tmp_path: Path, pool_id: str = "pool_0") -> Iterator:
    """An owned pool, mapped, the way a Guard or a claimant holds one."""
    owner = _pool(tmp_path, pool_id)
    try:
        attachment = KVCRPoolAttachment.attach(owner.spec)
        try:
            yield attachment
        finally:
            attachment.close()
    finally:
        owner.close()


def test_a_handback_region_round_trips_and_replays_nothing_once_released(
    tmp_path: Path,
) -> None:
    """A snapshot hides inside the pool file, round trips, and dies with release."""
    with _attached(tmp_path) as pool:
        # Inside the pool file, so it has no name of its own to be found under.
        path = Path(pool._spec.path)
        assert path.stat().st_size == pool._spec.mapping_bytes
        assert set(tmp_path.iterdir()) == {path}

        terms = canonical_pool_terms(_TEST_DIGEST, 4096, pool._spec)
        assert list(read_recovery_snapshot(pool, terms)) == []

        records = {
            BlockKey(b"a" * 32): _recovered_record(g2=3),
            BlockKey(b"b" * 32): _recovered_record(g2=4, g3=9),
            BlockKey(b"gone"): _BlockRecord(),
        }
        write_recovery_snapshot(pool, terms, _recovery_frames(records))

        assert set(tmp_path.iterdir()) == {path}
        assert path.stat().st_size > pool._spec.mapping_bytes

        # The mirror the ring feeds, unchanged: one record format, two carriers.
        # Its _records table is the recovered state a claimant would start from.
        mirror = _RecoveryMirror()
        for frame in read_recovery_snapshot(pool, terms):
            mirror.apply(*frame)
        recovered = mirror._records
        assert set(recovered) == {BlockKey(b"a" * 32), BlockKey(b"b" * 32)}
        assert recovered[BlockKey(b"a" * 32)].local_dram.slot == 3
        assert recovered[BlockKey(b"b" * 32)].g3.slot == 9

        # A released region is truncated away, so it replays nothing.
        clear_recovery_snapshot(pool)

        assert list(read_recovery_snapshot(pool, terms)) == []


@pytest.mark.parametrize(
    "digest,row_stride",
    [(_TEST_DIGEST, 8192), ("another-digest", 4096)],
    ids=["stride", "digest"],
)
def test_handback_region_is_refused_across_a_change_of_terms(
    tmp_path: Path, digest: str, row_stride: int
) -> None:
    """A slot number only means the same bytes under the same geometry."""
    with _attached(tmp_path) as pool:
        write_recovery_snapshot(
            pool,
            canonical_pool_terms(_TEST_DIGEST, 4096, pool._spec),
            _recovery_frames({BlockKey(b"k" * 32): _recovered_record(g2=1)}),
        )

        other = canonical_pool_terms(digest, row_stride, pool._spec)
        with pytest.raises(RecoveryJournalError, match="other terms"):
            list(read_recovery_snapshot(pool, other))


def test_an_unfinished_handback_region_is_refused(tmp_path: Path) -> None:
    """The header lands last, so a torn region never reads as a whole one."""
    with _attached(tmp_path) as pool:
        terms = canonical_pool_terms(_TEST_DIGEST, 4096, pool._spec)
        write_recovery_snapshot(
            pool,
            terms,
            _recovery_frames({BlockKey(b"k" * 32): _recovered_record(g2=1)}),
        )
        # A byte of the digest, as a region whose last write never finished.
        offset = _snapshot_offset(pool._spec.mapping_bytes)
        with open(pool._spec.path, "r+b") as pool_file:
            pool_file.seek(offset)
            byte = pool_file.read(1)[0]
            pool_file.seek(offset)
            pool_file.write(bytes([byte ^ 0xFF]))

        with pytest.raises(RecoveryJournalError, match="unfinished"):
            list(read_recovery_snapshot(pool, terms))


def test_an_interrupted_rewrite_reads_as_unfinished_rather_than_as_other_terms(
    tmp_path: Path,
) -> None:
    """A rewrite that dies mid-way must be discardable, not poison for every claim."""
    with _attached(tmp_path) as pool:
        terms = canonical_pool_terms(_TEST_DIGEST, 4096, pool._spec)
        write_recovery_snapshot(
            pool,
            terms,
            _recovery_frames({BlockKey(b"a" * 32): _recovered_record(g2=1)}),
        )
        assert list(read_recovery_snapshot(pool, terms))

        # Stopped once the replacing body has landed but before its header has.
        interrupted = Mock(
            size=_SNAPSHOT_HEADER.size, pack=Mock(side_effect=OSError("interrupted"))
        )
        with patch("kvcr.recovery_journal._SNAPSHOT_HEADER", interrupted):
            with pytest.raises(OSError, match="interrupted"):
                write_recovery_snapshot(
                    pool,
                    terms,
                    _recovery_frames({BlockKey(b"b" * 32): _recovered_record(g2=2)}),
                )

        # Unfinished, not written-for-other-terms: the first is thrown away, the
        # second would refuse every later claim on this pool.
        with pytest.raises(RecoveryJournalTornError, match="unfinished"):
            list(read_recovery_snapshot(pool, terms))
        assert read_handback(pool, _TEST_DIGEST, 4096)._records == {}
        assert list(read_recovery_snapshot(pool, terms)) == []
