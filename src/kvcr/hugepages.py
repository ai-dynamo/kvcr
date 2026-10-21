# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Best-effort aligned shared THP mappings before native registration."""

import ctypes
import errno
import logging
import mmap
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
_MADV_COLLAPSE = 25  # Linux 6.1+, not exposed by all Python mmap builds.
_MAP_FIXED = 0x10  # Linux: replace only the interior of our owned reservation.


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
    if not collapse_bytes:
        return None
    mapping = AlignedSharedMapping(fd, length, page_bytes)
    try:
        mapping.madvise(mmap.MADV_HUGEPAGE)
        if collapse:
            mapping.madvise(_MADV_COLLAPSE, collapse_bytes)
    except OSError as error:
        if error.errno not in (
            errno.EINVAL,
            errno.ENOSYS,
            errno.EOPNOTSUPP,
            errno.ENOMEM,
            errno.EAGAIN,
            errno.EBUSY,
        ):
            mapping.close()
            raise
        logger.warning("Pool THP unavailable; retaining available backing: %s", error)
    return mapping
