# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Aligned shared THP, buffer ownership, and best-effort fallback coverage."""

import ctypes
import errno
import mmap
import os
import sys
import tempfile
from unittest.mock import Mock, patch

import pytest

from kvcr import memmap
from kvcr.memmap import AlignedSharedMapping, thp_page_size
from kvcr.memory import KVCRPoolAttachment, _KVCRPoolOwner
from kvcr.recovery_journal import RecoveryJournal

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or sys.version_info < (3, 12),
    reason="aligned buffers require Linux and Python 3.12+",
)


@pytest.fixture
def thp_owner():
    try:
        page = thp_page_size()
    except OSError:
        pytest.skip("kernel PMD size is unavailable")
    if page > 32 << 20:
        pytest.skip("limit real unit probe to 64 MiB")
    with tempfile.TemporaryDirectory(prefix="kvcr-thp-", dir="/dev/shm") as directory:
        owner = _KVCRPoolOwner.allocate(
            pool_id="thp",
            pool_size_bytes=2 * page + 12345,
            journal_bytes=8192,
            pool_dir=directory,
        )
        try:
            yield owner
        finally:
            owner.close()


def test_shared_thp_complete_windows_tail_journal_and_snapshot(thp_owner):
    writer = KVCRPoolAttachment.attach(thp_owner.spec, collapse=True)
    reader = KVCRPoolAttachment.attach(thp_owner.spec)
    try:
        page = thp_page_size()
        assert isinstance(writer._require_mapping(), AlignedSharedMapping)
        assert writer.address % page == reader.address % page == 0
        producer, consumer = RecoveryJournal(writer), RecoveryJournal(reader)
        producer.reset()
        assert producer.publish(1, b"key", b"payload")
        assert consumer.read_next() == (1, b"key", b"payload")
        mapping = writer._require_mapping()
        mapping[-4:] = b"tail"
        assert reader._require_mapping()[-4:] == b"tail"
        assert len(mapping) == thp_owner.spec.mapping_bytes
        with writer.snapshot_region(12345) as snapshot:
            snapshot[:4] = b"snap"
        with reader.mapped_snapshot() as snapshot:
            assert snapshot[:4] == b"snap"
        writer.release_snapshot_region()
        assert os.stat(thp_owner.spec.path).st_size == (
            memmap.round_up(len(mapping), mmap.PAGESIZE)
        )
        assert mapping[-4:] == b"tail"
    finally:
        reader.close()
        writer.close()


def test_unidentifiable_backing_is_never_overlaid(thp_owner, monkeypatch):
    broken = Mock(fstatfs=Mock(return_value=-1))
    monkeypatch.setattr(memmap.ctypes, "CDLL", lambda *args, **kwargs: broken)
    with pytest.raises(OSError):
        KVCRPoolAttachment.attach(thp_owner.spec)
    broken.mmap.assert_not_called()  # no fixed-address overlay was attempted


@pytest.mark.parametrize("export", ["view", "slice", "ctypes"])
def test_export_blocks_close_and_allows_retry(thp_owner, export):
    attachment = KVCRPoolAttachment.attach(thp_owner.spec)
    mapping = attachment._require_mapping()
    assert isinstance(mapping, AlignedSharedMapping)
    base = ctypes.addressof(ctypes.c_char.from_buffer(mapping._reservation))
    assert attachment.address % thp_page_size() == 0
    assert base <= attachment.address
    assert attachment.address + len(mapping) <= base + len(mapping._reservation)
    value = (
        memoryview(mapping)
        if export in ("view", "slice")
        else ctypes.c_char.from_buffer(mapping)
    )
    if export == "slice":
        value = value[1:]
    with pytest.raises(BufferError):
        attachment.close()
    assert attachment._require_mapping() is mapping
    assert os.fstat(attachment._file_descriptor).st_ino == thp_owner.spec.inode
    if export in ("view", "slice"):
        value.release()
    del value
    attachment.close()
    attachment.close()
    with pytest.raises(ValueError):
        memoryview(mapping)


def test_failed_collapse_keeps_the_aligned_mapping(thp_owner, monkeypatch):
    reservations = []

    class Reservation(mmap.mmap):
        def __init__(self, *args, **kwargs):
            reservations.append(self)

        def madvise(self, option, *args):
            if option == memmap._MADV_COLLAPSE:
                raise OSError(errno.EIO, "collapse")
            return super().madvise(option, *args)

    monkeypatch.setattr(mmap, "mmap", Reservation)
    attachment = KVCRPoolAttachment.attach(thp_owner.spec, collapse=True)
    try:
        mapping = attachment._require_mapping()
        assert not reservations[0].closed and mapping._reservation is reservations[0]
    finally:
        attachment.close()


@pytest.mark.parametrize("failure", ["overlay", "overlay_dontfork"])
def test_failed_overlay_uses_base_pages(thp_owner, monkeypatch, failure):
    reservations = []

    class Reservation(mmap.mmap):
        def __init__(self, *args, **kwargs):
            reservations.append(self)

        def madvise(self, option, *args):
            # Only the aligned reservation fails; the base-page fallback must work.
            if (
                failure == "overlay_dontfork"
                and self is reservations[0]
                and option == mmap.MADV_DONTFORK
                and args
            ):
                raise OSError(errno.EIO, failure)
            return super().madvise(option, *args)

    monkeypatch.setattr(mmap, "mmap", Reservation)
    with patch("kvcr.memmap.ctypes.CDLL", wraps=ctypes.CDLL) as library:
        if failure == "overlay":
            native = Mock()
            native.mmap.return_value = ctypes.c_void_p(-1).value
            native.fstatfs.return_value = 0  # an ordinary filesystem
            library.return_value = native
            monkeypatch.setattr(ctypes, "get_errno", lambda: errno.EIO)
        attachment = KVCRPoolAttachment.attach(thp_owner.spec, collapse=True)
    try:
        assert reservations[0].closed and not isinstance(
            attachment._require_mapping(), AlignedSharedMapping
        )
    finally:
        attachment.close()


def test_old_runtime_and_missing_kernel_size_fall_back(monkeypatch):
    monkeypatch.setattr(memmap.sys, "version_info", (3, 11))
    assert memmap.try_aligned_shared_mapping(-1, 1 << 22, collapse=True) is None
    monkeypatch.setattr(memmap.sys, "version_info", (3, 12))
    monkeypatch.setattr(memmap, "thp_page_size", Mock(side_effect=FileNotFoundError))
    assert memmap.try_aligned_shared_mapping(-1, 1 << 22, collapse=True) is None
