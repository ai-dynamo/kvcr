# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pool memory mappings: HugeTLB, then THP, then base pages, best-effort."""

import contextlib
import ctypes
import logging
import mmap
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")
hugetlb_enabled = True  # the service clears it for --disable-hugepages
_MADV_COLLAPSE = 25  # Linux 6.1+, not exposed by all Python mmap builds.
_MAP_FIXED = 0x10  # Linux: replace only the interior of our owned reservation.
_SUBDIR = "kvcr"
_HUGETLBFS_MAGIC = 0x958458F6  # statfs f_type


def round_up(size: int, alignment: int) -> int:
    return (size + alignment - 1) // alignment * alignment


def _hugetlbfs_mounts() -> list[tuple[Path, int]]:
    """Visible hugetlbfs mounts as (mount point, page bytes), largest pages first."""
    try:
        lines = (
            Path("/proc/self/mounts").read_text(errors="surrogateescape").splitlines()
        )
    except OSError:
        return []
    mounts = []
    for line in lines:
        fields = line.split()
        if len(fields) >= 3 and fields[2] == "hugetlbfs":
            mount = Path(fields[1].replace("\\040", " "))
            with contextlib.suppress(OSError):
                mounts.append((mount, os.statvfs(mount).f_bsize))
    return sorted(mounts, key=lambda m: m[1], reverse=True)


def create_pool(
    pool_dir: str | os.PathLike[str],
    size: int,
    create: Callable[[Path, int], T],
) -> T:
    """Call ``create(directory, backing_bytes)`` on hugetlbfs, else in ``pool_dir``.

    Whether pages are free is left to the allocation: a HugeTLB failure warns and
    falls back to ``pool_dir``.
    """
    for mount, page in _hugetlbfs_mounts() if hugetlb_enabled else ():
        try:
            directory = mount / _SUBDIR
            directory.mkdir(mode=0o700, exist_ok=True)
            return create(directory, round_up(size, page))
        except OSError as error:
            logger.warning("HugeTLB pool unavailable; trying next: %s", error)
    return create(Path(pool_dir).resolve(), size)


def pool_dirs() -> list[Path]:
    """Our directories on hugetlbfs mounts, for orphan cleanup."""
    directories = []
    for mount, _ in _hugetlbfs_mounts():
        directory = mount / _SUBDIR
        # os.path never raises, unlike Path.is_dir on an unsearchable mount.
        if os.path.isdir(directory) and not os.path.islink(directory):
            directories.append(directory)
    return directories


def hugetlb_page_size(fd: int) -> int:
    """The huge-page size if ``fd`` is on hugetlbfs, else 0."""
    if sys.platform != "linux":  # the buffer below is sized for Linux's struct statfs
        return 0
    statfs = ctypes.create_string_buffer(256)  # struct statfs: f_type comes first
    if ctypes.CDLL(None, use_errno=True).fstatfs(fd, statfs) != 0:
        # Unknown backing must not take the fixed-address overlay or a guessed length.
        raise OSError(ctypes.get_errno(), "cannot identify the pool's filesystem")
    f_type = ctypes.c_long.from_buffer(statfs).value & 0xFFFFFFFF
    # hugetlbfs reports its huge-page size as the block size.
    return os.fstatvfs(fd).f_bsize if f_type == _HUGETLBFS_MAGIC else 0


def thp_page_size() -> int:
    """Read the kernel's PMD size rather than assuming x86's 2 MiB."""
    size = int(Path("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size").read_text())
    if size < mmap.PAGESIZE or size & (size - 1):
        raise ValueError(f"invalid THP page size: {size}")
    return size


class AlignedSharedMapping:
    """Keep Python mmap's buffer lifetime checks for an aligned file overlay."""

    def __init__(self, fd: int, length: int, page_bytes: int) -> None:
        self._length = length
        self._reservation = mmap.mmap(
            -1, length + page_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
        )
        try:
            # Both the reservation and its replacement must exclude forked children.
            self._reservation.madvise(mmap.MADV_DONTFORK)
            base = ctypes.addressof(ctypes.c_char.from_buffer(self._reservation))
            self._offset = -base % page_bytes
            libc = ctypes.CDLL(None, use_errno=True)
            libc.mmap.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_long,
            ]
            libc.mmap.restype = ctypes.c_void_p
            mapped = libc.mmap(
                base + self._offset,
                length,
                mmap.PROT_READ | mmap.PROT_WRITE,
                mmap.MAP_SHARED | _MAP_FIXED,
                fd,
                0,
            )
            if mapped == ctypes.c_void_p(-1).value:
                error = ctypes.get_errno()
                raise OSError(error, f"aligned pool mmap: {os.strerror(error)}")
            self.madvise(mmap.MADV_DONTFORK)
        except BaseException:
            self.close()
            raise

    def __buffer__(self, flags: int) -> memoryview:
        return memoryview(self._reservation)[self._offset : self._offset + self._length]

    def __getitem__(self, key: int | slice) -> int | bytes:
        with self.__buffer__(0) as view:
            value = view[key]
            return bytes(value) if isinstance(key, slice) else value

    def __setitem__(self, key: int | slice, value: int | bytes) -> None:
        with self.__buffer__(0) as view:
            view[key] = value

    def __len__(self) -> int:
        if self._reservation.closed:
            raise ValueError("mapping is closed")
        return self._length

    def madvise(self, advice: int, length: int | None = None) -> None:
        self._reservation.madvise(advice, self._offset, length or self._length)

    def close(self) -> None:
        # mmap.close raises BufferError while any view/ctypes export remains alive.
        self._reservation.close()


def try_aligned_shared_mapping(
    fd: int, length: int, *, collapse: bool
) -> AlignedSharedMapping | None:
    """Try full PMD regions; old runtimes/kernels and small pools fall back."""
    if (
        sys.platform != "linux"
        or sys.version_info < (3, 12)
        or not hasattr(mmap, "MADV_HUGEPAGE")
    ):
        return None
    try:
        page_bytes = thp_page_size()
    except (OSError, ValueError):
        return None
    collapse_bytes = length // page_bytes * page_bytes
    if not collapse_bytes or hugetlb_page_size(fd):
        return None
    try:
        mapping = AlignedSharedMapping(fd, length, page_bytes)
    except OSError as error:
        logger.warning("Pool THP unavailable; using base pages: %s", error)
        return None
    try:
        mapping.madvise(mmap.MADV_HUGEPAGE)
        if collapse:
            mapping.madvise(_MADV_COLLAPSE, collapse_bytes)
    except OSError as error:
        # Advice is an optimisation: the mapping works with whatever backing it has.
        logger.warning("Pool THP advice failed; retaining available backing: %s", error)
    return mapping


PoolMapping = mmap.mmap | AlignedSharedMapping
Region = mmap.mmap


def map_range(fd: int, size: int, offset: int = 0, *, writable: bool = True) -> Region:
    """Map ``size`` bytes of a pool file from ``offset``."""
    access = mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ
    return mmap.mmap(fd, size, offset=offset, access=access)


def region_offset(fd: int, mapping_bytes: int) -> int:
    """Where a region past a ``mapping_bytes`` pool starts."""
    granularity = mmap.ALLOCATIONGRANULARITY
    return align(fd, round_up(mapping_bytes, granularity))


def map_pool(fd: int, length: int, *, collapse: bool) -> PoolMapping:
    """Map a pool file: aligned THP when possible, else a plain mapping."""
    mapping = try_aligned_shared_mapping(fd, length, collapse=collapse)
    if mapping is None:
        # Whole pages only on a huge-page file, and not its snapshot tail.
        mapping = mmap.mmap(fd, align(fd, length), access=mmap.ACCESS_WRITE)
        try:
            # A forked child is not the pidfd-bound claimant.
            mapping.madvise(mmap.MADV_DONTFORK)
        except BaseException:
            mapping.close()
            raise
    return mapping


def align(fd: int, size: int) -> int:
    """``size`` rounded up to whole huge pages if ``fd`` is on hugetlbfs."""
    page = hugetlb_page_size(fd)
    return round_up(size, page) if page else size
