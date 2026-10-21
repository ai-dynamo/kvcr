# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Aligned shared THP, buffer ownership, and best-effort fallback coverage."""

import ctypes
import errno
import mmap
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import msgspec
import pytest

from kvcr import hugepages
from kvcr.hugepages import AlignedSharedMapping, thp_page_size
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
        if os.getenv("KVCR_TEST_THP") == "1":
            raise
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


def _pmd_bytes(attachment):
    start = attachment.address
    end = start + (-(-attachment._spec.mapping_bytes // mmap.PAGESIZE) * mmap.PAGESIZE)
    total = 0
    for stanza in re.split(
        r"(?=^[0-9a-f]+-[0-9a-f]+ )", Path("/proc/self/smaps").read_text(), flags=re.M
    ):
        header = re.match(r"([0-9a-f]+)-([0-9a-f]+) ", stanza)
        if header and start <= int(header[1], 16) < int(header[2], 16) <= end:
            match = re.search(r"^ShmemPmdMapped:\s+(\d+) kB$", stanza, re.M)
            total += int(match[1]) * 1024 if match else 0
    return total


@pytest.mark.skipif(
    os.getenv("KVCR_TEST_THP") != "1",
    reason="set KVCR_TEST_THP=1 to require real tmpfs THP",
)
def test_shared_thp_complete_windows_tail_journal_and_snapshot(thp_owner):
    writer = KVCRPoolAttachment.attach(thp_owner.spec, collapse=True)
    reader = KVCRPoolAttachment.attach(thp_owner.spec)
    try:
        page = thp_page_size()
        assert isinstance(writer._require_mapping(), AlignedSharedMapping)
        assert writer.address % page == reader.address % page == 0
        assert _pmd_bytes(writer) == 2 * page
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
            -(-len(mapping) // mmap.PAGESIZE) * mmap.PAGESIZE
        )
        assert mapping[-4:] == b"tail"
    finally:
        reader.close()
        writer.close()


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


def test_worker_shares_backing_across_processes_without_collapse(thp_owner):
    attachment = KVCRPoolAttachment.attach(thp_owner.spec, collapse=True)
    try:
        ctypes.memmove(attachment.data_address, b"parent", 6)
        code = """
import ctypes,os,sys,msgspec
from pathlib import Path
from unittest.mock import patch
from kvcr.hugepages import AlignedSharedMapping
from kvcr.memory import KVCRPoolAttachment,KVCRPoolSpec
original=AlignedSharedMapping.madvise
def advice(self, option, length=None):
    assert option != 25, 'worker must not collapse'
    return original(self,option,length)
with patch.object(AlignedSharedMapping,'madvise',advice):
    a=KVCRPoolAttachment.attach(msgspec.json.decode(sys.argv[1],type=KVCRPoolSpec))
    assert ctypes.string_at(a.data_address,6)==b'parent'
    ctypes.memmove(a.data_address,b'child!',6)
    address=a.address
    child=os.fork()
    if child==0:
        inherited=any(int(line.split('-')[0],16)<=address<int(line.split('-')[1].split()[0],16)
            for line in Path('/proc/self/maps').read_text().splitlines())
        os._exit(1 if inherited else 0)
    assert os.waitpid(child,0)[1]==0
    a.close()
"""
        subprocess.run(
            [sys.executable, "-c", code, msgspec.json.encode(thp_owner.spec).decode()],
            check=True,
            capture_output=True,
            text=True,
        )
        assert ctypes.string_at(attachment.data_address, 6) == b"child!"
    finally:
        attachment.close()


@pytest.mark.parametrize(
    "error_number",
    [
        errno.EINVAL,
        errno.ENOSYS,
        errno.EOPNOTSUPP,
        errno.EBUSY,
        errno.ENOMEM,
        errno.EAGAIN,
        errno.EFAULT,
    ],
)
def test_collapse_failure_retains_mapping_except_unexpected_errors(
    thp_owner, monkeypatch, error_number
):
    reservations = []
    original_map, original_advise = mmap.mmap, AlignedSharedMapping.madvise

    def reserve(*args, **kwargs):
        mapping = original_map(*args, **kwargs)
        reservations.append(mapping)
        return mapping

    def advise(self, option, length=None):
        if option == 25:
            raise OSError(error_number, "collapse unavailable")
        return original_advise(self, option, length)

    monkeypatch.setattr(mmap, "mmap", reserve)
    monkeypatch.setattr(AlignedSharedMapping, "madvise", advise)
    if error_number == errno.EFAULT:
        with pytest.raises(OSError) as raised:
            KVCRPoolAttachment.attach(thp_owner.spec, collapse=True)
        assert raised.value.errno == errno.EFAULT
        assert len(reservations) == 1 and reservations[0].closed
        return
    attachment = KVCRPoolAttachment.attach(thp_owner.spec, collapse=True)
    try:
        assert len(reservations) == 1 and not reservations[0].closed
        assert attachment._require_mapping()._reservation is reservations[0]
        assert attachment._require_mapping()[:4] == bytes(4)
    finally:
        attachment.close()


@pytest.mark.parametrize(
    "failure", ["reservation_dontfork", "overlay", "overlay_dontfork"]
)
def test_failed_overlay_or_dontfork_closes_owned_reservation(
    thp_owner, monkeypatch, failure
):
    reservations = []
    error = OSError(errno.EIO, failure)

    class Reservation(mmap.mmap):
        def __init__(self, *args, **kwargs):
            reservations.append(self)

        def madvise(self, option, *args):
            if option == mmap.MADV_DONTFORK:
                overlay = bool(args)
                if failure == (
                    "overlay_dontfork" if overlay else "reservation_dontfork"
                ):
                    raise error
            return super().madvise(option, *args)

    monkeypatch.setattr(mmap, "mmap", Reservation)
    with patch("kvcr.hugepages.ctypes.CDLL", wraps=ctypes.CDLL) as library:
        if failure == "overlay":
            native = Mock()
            native.mmap.return_value = ctypes.c_void_p(-1).value
            library.return_value = native
            monkeypatch.setattr(ctypes, "get_errno", lambda: errno.EIO)
        with pytest.raises(OSError) as raised:
            KVCRPoolAttachment.attach(thp_owner.spec, collapse=True)
    assert raised.value.errno == errno.EIO
    assert len(reservations) == 1 and reservations[0].closed


def test_old_runtime_and_missing_kernel_size_fall_back(monkeypatch):
    monkeypatch.setattr(hugepages.sys, "version_info", (3, 11))
    assert hugepages.try_aligned_shared_mapping(-1, 1 << 22, collapse=True) is None
    monkeypatch.setattr(hugepages.sys, "version_info", (3, 12))
    monkeypatch.setattr(hugepages, "thp_page_size", Mock(side_effect=FileNotFoundError))
    assert hugepages.try_aligned_shared_mapping(-1, 1 << 22, collapse=True) is None
