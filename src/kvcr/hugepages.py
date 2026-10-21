# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Strict opt-in Linux THP mappings, verified before native registration."""

import ctypes
import mmap
import os
import re
import sys
from pathlib import Path


def thp_page_size() -> int:
    """Read the host PMD size; it differs between x86 and Grace kernels."""
    if sys.platform != "linux" or sys.version_info < (3, 12):
        raise RuntimeError("KVCR THP requires Linux and Python 3.12+")
    size = int(Path("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size").read_text())
    if size < mmap.PAGESIZE or size & (size - 1):
        raise RuntimeError(f"invalid kernel THP page size: {size}")
    return size


class AlignedSharedMapping:
    """Own an aligned shared mapping; PEP 688 preserves buffer lifetimes.

    MAP_FIXED replaces only the interior of our still-owned PROT_NONE
    reservation. It never overwrites memory allocated by another component.
    """

    def __init__(
        self,
        fd: int,
        length: int,
    ):
        page_bytes = thp_page_size()
        if length <= 0 or length % page_bytes:
            raise ValueError(
                f"THP mapping length must be a positive multiple of {page_bytes} bytes"
            )
        self._lib = ctypes.CDLL(None, use_errno=True)
        self._lib.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        self._lib.mmap.restype = ctypes.c_void_p
        self._lib.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._lib.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self._lib.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self._length = length
        self._logical_length = length
        self._exports = 0
        self._closed = False
        self._reservation_bytes = length + page_bytes
        reserved = self._lib.mmap(
            None,
            self._reservation_bytes,
            0,
            mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
            -1,
            0,
        )
        if reserved == ctypes.c_void_p(-1).value:
            self._raise_errno("reserve aligned mapping")
        self._reserved = reserved
        self._address = (reserved + page_bytes - 1) // page_bytes * page_bytes
        try:
            mapped = self._lib.mmap(
                self._address,
                length,
                mmap.PROT_READ | mmap.PROT_WRITE,
                mmap.MAP_SHARED | 0x10,
                fd,
                0,  # MAP_FIXED, owned reservation only
            )
            if mapped == ctypes.c_void_p(-1).value:
                self._raise_errno("map aligned Guard pool")
            self._array = (ctypes.c_ubyte * self._logical_length).from_address(
                self._address
            )
            self.madvise(mmap.MADV_DONTFORK)
            self.madvise(mmap.MADV_HUGEPAGE)
            self.madvise(23)  # MADV_POPULATE_WRITE: errors, never SIGBUS stores
            self.madvise(25)  # MADV_COLLAPSE: before native pinning
            self.huge_bytes = self._huge_bytes()
            if self.huge_bytes != length:
                raise RuntimeError(
                    f"Guard THP backing incomplete: {self.huge_bytes}/{length} bytes"
                )
        except BaseException:
            self.close()
            raise

    def _raise_errno(self, operation: str):
        error = ctypes.get_errno()
        raise OSError(error, f"{operation}: {os.strerror(error)}")

    def _huge_bytes(self) -> int:
        total = 0
        for stanza in re.split(
            r"(?=^[0-9a-f]+-[0-9a-f]+ )",
            Path("/proc/self/smaps").read_text(),
            flags=re.M,
        ):
            header = re.match(r"([0-9a-f]+)-([0-9a-f]+) ", stanza)
            if not header:
                continue
            begin, end = (int(value, 16) for value in header.groups())
            if begin >= self._address and end <= self._address + self._length:
                fields = ("ShmemPmdMapped", "FilePmdMapped")
                for field in fields:
                    match = re.search(rf"^{field}:\s+(\d+) kB$", stanza, re.M)
                    total += int(match[1]) * 1024 if match else 0
        return total

    def __buffer__(self, flags):
        if self._closed:
            raise ValueError("mapping is closed")
        self._exports += 1
        return memoryview(self._array).cast("B")

    def __release_buffer__(self, view):
        self._exports -= 1

    def __getitem__(self, key):
        if self._closed:
            raise ValueError("mapping is closed")
        value = self._array[key]
        return bytes(value) if isinstance(key, slice) else value

    def __setitem__(self, key, value):
        if self._closed:
            raise ValueError("mapping is closed")
        with memoryview(self._array).cast("B") as view:
            view[key] = value

    def __len__(self):
        if self._closed:
            raise ValueError("mapping is closed")
        return self._logical_length

    def madvise(self, advice: int):
        if self._closed:
            raise ValueError("mapping is closed")
        if self._lib.madvise(self._address, self._length, advice):
            self._raise_errno(f"Guard madvise({advice})")

    def flush(self):
        if self._closed:
            raise ValueError("mapping is closed")
        if self._lib.msync(self._address, self._length, 4):  # MS_SYNC
            self._raise_errno("flush aligned Guard mapping")

    def close(self):
        if self._closed:
            return
        if self._exports:
            raise BufferError("cannot close exported Guard mapping")
        if self._lib.munmap(self._reserved, self._reservation_bytes):
            self._raise_errno("unmap aligned Guard pool")
        self._closed = True
