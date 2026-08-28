# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import mmap
import struct
from collections.abc import Iterator
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


def test_journal_wraps_without_losing_records(
    journal_and_mapping: tuple[RecoveryJournal, mmap.mmap],
) -> None:
    journal, _ = journal_and_mapping
    for index in range(400):
        key = index.to_bytes(4, "little")
        payload = bytes([index % 256]) * 9
        assert journal.publish(_RECORD_BLOCK, key, payload)
        assert journal.read_next() == (_RECORD_BLOCK, key, payload)

    assert journal._published.load_acquire() > journal._capacity
    assert journal._consumed.load_acquire() == journal._published.load_acquire()
    assert not journal.is_invalid()


def test_round_trip_variable_width_keys_and_opaque_payloads(
    journal_and_mapping: tuple[RecoveryJournal, mmap.mmap],
) -> None:
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


def test_frame_larger_than_uint16_invalidates_without_publication(
    journal_and_mapping: tuple[RecoveryJournal, mmap.mmap],
    caplog: pytest.LogCaptureFixture,
) -> None:
    journal, _ = journal_and_mapping
    caplog.set_level("WARNING", logger="kvcr.recovery_journal")
    assert journal.publish(_RECORD_BLOCK, b"key0", b"record")
    published = journal._published.load_acquire()

    assert not journal.publish(_RECORD_BLOCK, b"key", bytes(1 << 16))
    assert journal._published.load_acquire() == published
    assert journal.is_invalid()
    assert caplog.messages == [
        "KVCR recovery journal invalidated: frame is 65545 bytes; "
        "maximum is 65535 bytes"
    ]


def test_full_ring_invalidates_without_partial_publication(
    journal_and_mapping: tuple[RecoveryJournal, mmap.mmap],
) -> None:
    journal, _ = journal_and_mapping
    published = 0

    while True:
        before = journal._published.load_acquire()
        if not journal.publish(_RECORD_BLOCK, b"key", bytes(1000)):
            break
        published += 1

    assert published > 0
    assert journal._published.load_acquire() == before
    assert journal.is_invalid()
    with pytest.raises(RecoveryJournalError, match="invalid"):
        journal.read_next()
    with pytest.raises(RecoveryJournalError, match="invalid"):
        list(journal.drain())

    journal.reset()

    assert journal._published.load_acquire() == 0
    assert journal._consumed.load_acquire() == 0
    assert not journal.is_invalid()


def test_a_mirroring_consumer_notices_an_invalidation_it_did_not_make(
    journal_and_mapping: tuple[RecoveryJournal, mmap.mmap],
) -> None:
    """Caught up and finished look identical unless the shared flag is re-read."""
    producer, mapping = journal_and_mapping
    consumer = RecoveryJournal(_attachment(mapping, _TEST_JOURNAL_BYTES))

    assert producer.publish(_RECORD_BLOCK, b"key", b"payload")
    assert consumer.read_next() is not None
    assert consumer.read_next() is None

    assert producer.invalidate() is False

    with pytest.raises(RecoveryJournalError, match="invalid"):
        consumer.read_next()


def test_a_role_that_attaches_to_a_finished_journal_never_starts(
    journal_and_mapping: tuple[RecoveryJournal, mmap.mmap],
) -> None:
    """The other test covers noticing later; this one covers never starting."""
    owner, mapping = journal_and_mapping
    producer = RecoveryJournal(_attachment(mapping, _TEST_JOURNAL_BYTES))
    consumer = RecoveryJournal(_attachment(mapping, _TEST_JOURNAL_BYTES))
    owner.invalidate()

    assert not producer.publish(_RECORD_BLOCK, b"key", b"payload")
    with pytest.raises(RecoveryJournalError, match="invalid"):
        consumer.read_next()


def test_noncanonical_frame_invalidates_the_consumer(
    journal_and_mapping: tuple[RecoveryJournal, mmap.mmap],
) -> None:
    journal, mapping = journal_and_mapping
    assert journal.publish(_RECORD_BLOCK, b"key", b"payload")
    struct.pack_into("<HH", mapping, _JOURNAL_HEADER_BYTES, 2, 0)

    with pytest.raises(RecoveryJournalError, match="key size"):
        journal.read_next()
    assert journal.is_invalid()


def test_nonzero_frame_padding_invalidates_the_consumer(
    journal_and_mapping: tuple[RecoveryJournal, mmap.mmap],
) -> None:
    journal, mapping = journal_and_mapping
    assert journal.publish(_RECORD_BLOCK, b"keys", b"x")
    mapping[_JOURNAL_HEADER_BYTES + 12] = 1

    with pytest.raises(RecoveryJournalError, match="padding"):
        journal.read_next()
    assert journal.is_invalid()


def test_drain_is_streaming_and_takes_everything(
    journal_and_mapping: tuple[RecoveryJournal, mmap.mmap],
) -> None:
    """Streaming, so a promotion does not hold a second copy of the ring."""
    journal, _ = journal_and_mapping
    for index in range(3):
        assert journal.publish(_RECORD_BLOCK, bytes([index]), bytes([index + 3]))
    published = journal._published.load_acquire()

    records = journal.drain()
    assert next(records) == (_RECORD_BLOCK, b"\x00", b"\x03")
    assert 0 < journal._consumed.load_acquire() < published
    assert list(records) == [
        (_RECORD_BLOCK, b"\x01", b"\x04"),
        (_RECORD_BLOCK, b"\x02", b"\x05"),
    ]
    assert journal._consumed.load_acquire() == published


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


def test_a_handback_region_lives_past_the_pool_it_describes(tmp_path: Path) -> None:
    """Inside the pool file, so it has no name of its own to be found under."""
    with _attached(tmp_path) as pool:
        path = Path(pool._spec.path)
        assert path.stat().st_size == pool._spec.mapping_bytes
        assert set(tmp_path.iterdir()) == {path}

        write_recovery_snapshot(
            pool,
            canonical_pool_terms(_TEST_DIGEST, 4096, pool._spec),
            _recovery_frames({BlockKey(b"k" * 32): _recovered_record(g2=1)}),
        )

        assert set(tmp_path.iterdir()) == {path}
        assert path.stat().st_size > pool._spec.mapping_bytes


def test_handback_region_round_trips_through_the_ring_mirror(tmp_path: Path) -> None:
    with _attached(tmp_path) as pool:
        terms = canonical_pool_terms(_TEST_DIGEST, 4096, pool._spec)
        records = {
            BlockKey(b"a" * 32): _recovered_record(g2=3),
            BlockKey(b"b" * 32): _recovered_record(g2=4, g3=9),
            BlockKey(b"gone"): _BlockRecord(),
        }

        write_recovery_snapshot(pool, terms, _recovery_frames(records))

        # The mirror the ring feeds, unchanged: one record format, two carriers.
        mirror = _RecoveryMirror()
        for frame in read_recovery_snapshot(pool, terms):
            mirror.apply(*frame)

        recovered = mirror._records
        assert set(recovered) == {BlockKey(b"a" * 32), BlockKey(b"b" * 32)}
        assert recovered[BlockKey(b"a" * 32)].local_dram.slot == 3
        assert recovered[BlockKey(b"b" * 32)].g3.slot == 9


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


def test_a_released_handback_region_replays_nothing(tmp_path: Path) -> None:
    """A released region is truncated away, so it replays nothing."""
    with _attached(tmp_path) as pool:
        terms = canonical_pool_terms(_TEST_DIGEST, 4096, pool._spec)
        assert list(read_recovery_snapshot(pool, terms)) == []

        write_recovery_snapshot(
            pool,
            terms,
            _recovery_frames({BlockKey(b"k" * 32): _recovered_record(g2=1)}),
        )
        assert list(read_recovery_snapshot(pool, terms))

        clear_recovery_snapshot(pool)

        assert list(read_recovery_snapshot(pool, terms)) == []


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
