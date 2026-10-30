# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HugeTLB-first pool placement; a plain directory stands in for a hugetlbfs mount."""

import errno
import os
import sys
from pathlib import Path

import pytest
from _kvcr_test_utils import _recovered_record

from kvcr.service import memmap, memory
from kvcr.service.memory import KVCRPoolAttachment, _KVCRPoolOwner
from kvcr.service.protocol import _PoolDescriptor
from kvcr.service.recovery_journal import (
    _recovery_frames,
    _RecoveryMirror,
    canonical_pool_terms,
    read_recovery_snapshot,
    write_recovery_snapshot,
)
from kvcr.service.server import _PoolRegistry
from kvcr.types import BlockKey

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="needs Linux")
_REAL_MOUNTS = memmap._hugetlbfs_mounts  # before conftest hides them
_PAGE = 2 << 20
_SIZE = _PAGE + 12345


def _allocate(base):
    return _KVCRPoolOwner.allocate(
        pool_id="huge", pool_size_bytes=_SIZE, journal_bytes=8192, pool_dir=base
    )


def _files(directory):
    return sorted(Path(directory).rglob("kvcr-*"))


def _hand_back_and_replay(pool):
    """The Guard's real snapshot writer and reader, through the pool's region."""
    pools = (_PoolDescriptor("pool0", 2048, 1024, pool._spec.journal_bytes),)
    terms = canonical_pool_terms("digest", pools, pool._spec)
    records = {BlockKey(b"a" * 32): _recovered_record(g2=[("pool0", 3)])}
    write_recovery_snapshot(pool, terms, _recovery_frames(records))
    mirror = _RecoveryMirror(("pool0",))
    for frame in read_recovery_snapshot(pool, terms):
        mirror.apply(*frame)
    assert mirror.take_records() == records


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    base, mount = tmp_path / "base", tmp_path / "hugetlbfs"
    base.mkdir()
    mount.mkdir()
    monkeypatch.setattr(memmap, "_hugetlbfs_mounts", lambda: [(mount, _PAGE)])
    monkeypatch.setattr(
        memmap,
        "hugetlb_page_size",
        lambda fd: (
            _PAGE if mount in Path(os.readlink(f"/proc/self/fd/{fd}")).parents else 0
        ),
    )
    return base, mount


def test_pool_on_hugetlbfs_is_rounded_hands_back_by_page_and_is_purged(dirs):
    base, mount = dirs
    owner, live = _allocate(base), _allocate(base)
    path = Path(owner.spec.path)
    attachment = KVCRPoolAttachment.attach(owner.spec)
    assert path.parent == mount / "kvcr" and not _files(base)
    assert owner.spec.mapping_bytes == _SIZE and path.stat().st_size == 2 * _PAGE
    _hand_back_and_replay(attachment)
    assert path.stat().st_size == 3 * _PAGE  # the region is one whole page
    attachment.release_snapshot_region()
    assert path.stat().st_size == 2 * _PAGE
    attachment.close()
    os.close(owner._file_descriptor)  # drops the flock like a crash
    registry = object.__new__(_PoolRegistry)
    registry._pool_dir = base
    try:
        registry._purge_orphaned_pools()
        assert not path.exists() and Path(live.spec.path).exists()
    finally:
        owner._file_descriptor = -1
        live.close()


@pytest.mark.parametrize("case", ["disabled", "not_a_directory", "reservation_failure"])
def test_unavailable_hugetlb_uses_pool_dir(dirs, monkeypatch, case):
    base, mount = dirs
    if case == "disabled":
        monkeypatch.setattr(memmap, "hugetlb_enabled", False)
    elif case == "not_a_directory":
        (mount / "kvcr").touch()
    else:
        original = memory._populate_pages

        def populate(fd, *args):
            if mount in Path(os.readlink(f"/proc/self/fd/{fd}")).parents:
                raise OSError(errno.ENOSPC, "no pages")  # after the file exists
            return original(fd, *args)

        monkeypatch.setattr(memory, "_populate_pages", populate)
    owner = _allocate(base)
    assert Path(owner.spec.path).parent == base.resolve() and not _files(mount)
    owner.close()


def test_purge_skips_an_unopenable_or_symlinked_hugetlb_dir(dirs, monkeypatch):
    base, mount = dirs
    (mount / "kvcr").symlink_to(base)
    assert memmap.pool_dirs() == []
    (mount / "kvcr").unlink()
    (mount / "kvcr").mkdir()
    original = _PoolRegistry._purge_orphaned_pools_in

    def purge(self, directory):
        if directory != base:
            raise PermissionError(errno.EACCES, "no access")
        original(self, directory)

    monkeypatch.setattr(_PoolRegistry, "_purge_orphaned_pools_in", purge)
    registry = object.__new__(_PoolRegistry)
    registry._pool_dir = base
    registry._purge_orphaned_pools()  # the HugeTLB directory is skipped, not fatal


def test_real_hugetlbfs_pool(request, tmp_path, monkeypatch):
    given = request.config.getoption("--hugetlbfs-dir")
    if not given:
        pytest.skip("pass --hugetlbfs-dir to run against a real hugetlbfs mount")
    directory = Path(given).resolve()
    mounts = [m for m in _REAL_MOUNTS() if m[0] == directory.parent]
    monkeypatch.setattr(memmap, "_hugetlbfs_mounts", lambda: mounts)
    monkeypatch.setattr(memmap, "_SUBDIR", directory.name)
    owner = _allocate(tmp_path)
    assert Path(owner.spec.path).parent == directory, f"HugeTLB failed under {given}"
    attachment = KVCRPoolAttachment.attach(owner.spec)
    try:
        assert memmap.hugetlb_page_size(attachment._file_descriptor)
        attachment._require_mapping()[-4:] = b"tail"
        _hand_back_and_replay(attachment)
        attachment.release_snapshot_region()
    finally:
        attachment.close()
        owner.close()
