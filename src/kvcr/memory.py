# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Server-owned shared-memory pools for KVCR local DRAM."""

import contextlib
import ctypes
import errno
import fcntl
import mmap
import os
import stat
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import msgspec

_POOL_MODE = 0o600
_POOL_PREFIX = "kvcr"


def _snapshot_offset(mapping_bytes: int) -> int:
    """Where a pool's handback region starts, past everything it was granted."""
    granularity = mmap.ALLOCATIONGRANULARITY
    return -(-mapping_bytes // granularity) * granularity


_JOURNAL_HEADER_BYTES = 4096
# The errnos that mean "another holder has this lock" rather than "locking is
# broken here". Anything outside this set must propagate.
_CONTENTION_ERRNOS = frozenset({errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES})
_OPEN_FLAGS = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
_POOL_ID_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
)


def _validate_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


class KVCRPoolSpec(msgspec.Struct, frozen=True):
    """Identity and physical layout of a server-owned KVCR memory pool."""

    pool_id: str
    path: str
    generation: Annotated[str, msgspec.Meta(pattern=r"\A[0-9a-f]{32}\Z")]
    device: Annotated[int, msgspec.Meta(ge=0)]
    inode: Annotated[int, msgspec.Meta(ge=0)]
    mapping_bytes: Annotated[int, msgspec.Meta(gt=0)]
    journal_bytes: Annotated[int, msgspec.Meta(gt=0)]

    @property
    def data_bytes(self) -> int:
        """How much of the mapping is cache rather than journal."""
        return self.mapping_bytes - self.journal_bytes

    def __post_init__(self) -> None:
        if not Path(self.path).is_absolute():
            raise ValueError("KVCR pool path must be absolute")
        if Path(self.path).name != _pool_filename(self.pool_id, self.generation):
            raise ValueError(
                "KVCR pool path does not match its identity and generation"
            )
        _validate_pool_layout(self.mapping_bytes, self.journal_bytes)


class KVCRPoolAttachment:
    """Map a KVCR pool without taking ownership of its backing file."""

    def __init__(
        self,
        *,
        _file_descriptor: int,
        _mapping: mmap.mmap,
        _spec: KVCRPoolSpec,
    ) -> None:
        self._file_descriptor = _file_descriptor
        self._mapping: mmap.mmap | None = _mapping
        self._spec = _spec

    @classmethod
    def attach(cls, spec: KVCRPoolSpec) -> "KVCRPoolAttachment":
        file_descriptor = os.open(spec.path, _OPEN_FLAGS)
        mapping: mmap.mmap | None = None
        try:
            # Held for the mapping's lifetime: a worker can outlive its
            # daemon, and this stops a replacement reclaiming the pool.
            if not _try_lock_pool(file_descriptor, exclusive=False):
                raise RuntimeError(
                    f"KVCR pool is exclusively locked by another process: {spec.path}"
                )
            file_stat = os.fstat(file_descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"KVCR pool is not a regular file: {spec.path}")
            if stat.S_IMODE(file_stat.st_mode) != _POOL_MODE:
                raise PermissionError(f"KVCR pool must have mode 0600: {spec.path}")
            if (file_stat.st_dev, file_stat.st_ino) != (spec.device, spec.inode):
                raise ValueError(
                    f"KVCR pool identity does not match the grant: {spec.path}"
                )
            if file_stat.st_size < spec.mapping_bytes:
                raise ValueError(
                    "KVCR pool is smaller than the grant describes: "
                    f"{file_stat.st_size} < {spec.mapping_bytes}"
                )
            mapping = mmap.mmap(
                file_descriptor,
                spec.mapping_bytes,
                access=mmap.ACCESS_WRITE,
            )
            # A forked child is not the pidfd-bound claimant.
            mapping.madvise(mmap.MADV_DONTFORK)
            return cls(
                _file_descriptor=file_descriptor,
                _mapping=mapping,
                _spec=spec,
            )
        except BaseException:
            if mapping is not None:
                mapping.close()
            os.close(file_descriptor)
            raise

    @property
    def data_address(self) -> int:
        """Where the cache starts, past the journal at the head of the pool."""
        return self.address + self._spec.journal_bytes

    @property
    def address(self) -> int:
        """Return the base address of the mapped pool."""
        mapping = self._require_mapping()
        return ctypes.addressof(ctypes.c_char.from_buffer(mapping))

    @contextlib.contextmanager
    def snapshot_region(self, size: int) -> Iterator[mmap.mmap]:
        """Map `size` writable bytes past the pool, backed before they are used.

        Past the pool rather than inside it, so the grant still describes exactly what
        a claimant maps. Mapped separately because the pool's own mapping is
        registered with NIXL and mremap may move it.
        """
        offset = _snapshot_offset(self._spec.mapping_bytes)
        os.ftruncate(self._file_descriptor, offset + size)
        try:
            _populate_pages(self._file_descriptor, offset, size)
            region = mmap.mmap(
                self._file_descriptor, size, offset=offset, access=mmap.ACCESS_WRITE
            )
        except BaseException:
            # Back to the pool: an unusable tail still reads as a region, and
            # the next claimant would try to replay it.
            with contextlib.suppress(OSError):
                os.ftruncate(self._file_descriptor, offset)
            raise
        try:
            yield region
        except BaseException:
            # A write that did not finish must not leave the previous snapshot
            # readable: the Guard that failed here dropped its mirror, and a
            # claimant replaying old frames would diverge from it. The mapping
            # closes before the truncate; shrinking under it would fault.
            region.close()
            with contextlib.suppress(OSError):
                os.ftruncate(self._file_descriptor, offset)
            raise
        else:
            region.close()

    @contextlib.contextmanager
    def mapped_snapshot(self) -> Iterator[mmap.mmap | None]:
        """Whatever a previous Guard left past the pool, if anything is there."""
        offset = _snapshot_offset(self._spec.mapping_bytes)
        size = os.fstat(self._file_descriptor).st_size - offset
        if size <= 0:
            yield None
            return
        region = mmap.mmap(
            self._file_descriptor, size, offset=offset, access=mmap.ACCESS_READ
        )
        try:
            yield region
        finally:
            region.close()

    def release_snapshot_region(self) -> None:
        """Give the region back once its records have been installed.

        Truncated, which both retires it and returns its pages. Safe because both
        readers map it inside a context manager and a pool grants itself to one
        claimant at a time.
        """
        os.ftruncate(self._file_descriptor, _snapshot_offset(self._spec.mapping_bytes))

    def close(self) -> None:
        """Unmap the pool without unlinking the server-owned file."""
        mapping = self._mapping
        if mapping is not None:
            mapping.close()
            self._mapping = None
        if self._file_descriptor >= 0:
            os.close(self._file_descriptor)
            self._file_descriptor = -1

    def _require_mapping(self) -> mmap.mmap:
        if self._mapping is None:
            raise RuntimeError("KVCR pool attachment is closed")
        return self._mapping


class _KVCRPoolOwner:
    """Own a pool's backing file and physical specification."""

    def __init__(
        self,
        spec: KVCRPoolSpec,
        file_descriptor: int,
    ) -> None:
        self.spec = spec
        self._file_descriptor = file_descriptor
        self._path = Path(spec.path)
        self._file_identity = (spec.device, spec.inode)

    @classmethod
    def allocate(
        cls,
        *,
        pool_id: str,
        pool_size_bytes: int,
        journal_bytes: int,
        pool_dir: str | os.PathLike[str],
    ) -> "_KVCRPoolOwner":
        """Create the file and reserve its data space.

        Created and locked under the directory guard, which a purge takes
        exclusively, so a pool under construction cannot be reclaimed.
        """
        _validate_pool_layout(pool_size_bytes, journal_bytes)
        generation = uuid.uuid4().hex
        directory = Path(pool_dir).resolve()
        path = directory / _pool_filename(pool_id, generation)
        flags = _OPEN_FLAGS | os.O_CREAT | os.O_EXCL
        with _pool_dir_guard(directory, exclusive=False):
            file_descriptor = os.open(path, flags, _POOL_MODE)
            file_stat = os.fstat(file_descriptor)
            file_identity = (file_stat.st_dev, file_stat.st_ino)
            try:
                spec = msgspec.convert(
                    {
                        "pool_id": pool_id,
                        "path": str(path),
                        "generation": generation,
                        "device": file_identity[0],
                        "inode": file_identity[1],
                        "mapping_bytes": pool_size_bytes,
                        "journal_bytes": journal_bytes,
                    },
                    type=KVCRPoolSpec,
                )
                # Dropped by the kernel on death, which is how another
                # daemon tells a live pool from a crashed one's.
                if not _try_lock_pool(file_descriptor, exclusive=False):
                    raise RuntimeError(f"KVCR pool is already owned: {path}")
            except BaseException:
                os.close(file_descriptor)
                _unlink_if_identity(path, file_identity)
                raise
        try:
            os.fchmod(file_descriptor, _POOL_MODE)
            os.ftruncate(file_descriptor, pool_size_bytes)
            _populate_pages(file_descriptor, 0, pool_size_bytes)
        except BaseException:
            os.close(file_descriptor)
            # Identity-guarded: something may have replaced the file since it
            # was created, and that replacement is not ours.
            _unlink_if_identity(path, file_identity)
            raise
        return cls(spec, file_descriptor)

    def close(self) -> None:
        """Close and unlink the owned pool."""
        if self._file_descriptor != -1:
            os.close(self._file_descriptor)
            self._file_descriptor = -1
        _unlink_if_identity(self._path, self._file_identity)


def _compute_pool_geometry(
    requested_bytes: int,
    row_stride: int,
) -> tuple[int, int]:
    _validate_positive_int("requested_bytes", requested_bytes)
    _validate_positive_int("row_stride", row_stride)
    rows = requested_bytes // row_stride
    if rows == 0:
        raise ValueError(
            "requested_bytes must hold at least one complete KV row "
            f"of {row_stride} bytes"
        )
    return rows * row_stride, rows


def _validate_pool_layout(mapping_bytes: int, journal_bytes: int) -> None:
    _validate_positive_int("mapping_bytes", mapping_bytes)
    _validate_positive_int("journal_bytes", journal_bytes)
    if journal_bytes <= _JOURNAL_HEADER_BYTES:
        raise ValueError("KVCR pool journal_bytes must exceed its 4096-byte header")
    if journal_bytes % _JOURNAL_HEADER_BYTES:
        raise ValueError("KVCR pool journal_bytes must be page aligned")
    if journal_bytes >= mapping_bytes:
        raise ValueError("KVCR pool journal_bytes must be smaller than mapping_bytes")


def _pool_filename(pool_id: str, generation: str) -> str:
    if not pool_id or any(character not in _POOL_ID_CHARS for character in pool_id):
        raise ValueError("KVCR pool_id contains a filename-unsafe character")
    return f"{_POOL_PREFIX}-{pool_id}-{generation}"


@contextlib.contextmanager
def _pool_dir_guard(directory: Path, *, exclusive: bool) -> Iterator[None]:
    """Serialize pool creation against a directory-wide purge.

    Creation takes it shared, a purge exclusive, closing the window between a
    staging file's creation and its own lock. Held on the directory's own
    descriptor, so it leaves no lock file behind.
    """
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    file_descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(file_descriptor, mode)
        yield
    finally:
        os.close(file_descriptor)


def _try_lock_pool(file_descriptor: int, *, exclusive: bool) -> bool:
    """Take the pool's advisory lock without blocking.

    Every legitimate holder takes it shared; exclusive is only a liveness
    probe, succeeding precisely when the pool is abandoned. Returns False only
    for genuine contention, so a broken environment cannot look like a busy pool.
    """
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(file_descriptor, mode | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno in _CONTENTION_ERRNOS:
            return False
        raise
    return True


def _reclaim_pool_if_orphaned(path: Path) -> bool:
    """Unlink ``path`` only if neither a daemon nor a worker holds it.

    Failing to take the exclusive lock means the pool is still in use; the
    unlink happens while holding it, so nobody can adopt the file in between.
    """
    try:
        file_descriptor = os.open(path, _OPEN_FLAGS)
    except OSError:
        return False
    try:
        if not _try_lock_pool(file_descriptor, exclusive=True):
            return False
        file_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            return False
        _unlink_if_identity(path, (file_stat.st_dev, file_stat.st_ino))
        return True
    finally:
        os.close(file_descriptor)


def _unlink_if_identity(path: Path, identity: tuple[int, int]) -> None:
    try:
        file_stat = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (file_stat.st_dev, file_stat.st_ino) == identity:
        path.unlink()


def _populate_pages(file_descriptor: int, offset: int, length: int) -> None:
    """Commit backing blocks so a later mapping touch cannot fault on ENOSPC.

    The range is explicit because the fallback writes: reserving past a live pool
    must not touch the pool itself.
    """
    posix_fallocate = getattr(os, "posix_fallocate", None)
    if posix_fallocate is not None:
        try:
            posix_fallocate(file_descriptor, offset, length)
            return
        except OSError as error:
            unsupported_errors = {
                errno.ENOSYS,
                errno.EOPNOTSUPP,
                getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
            }
            if error.errno not in unsupported_errors:
                raise
    # Through pwrite, not a mapping: a mapped store that cannot be backed raises SIGBUS,
    # where the same write through the descriptor reports ENOSPC. A whole page at a
    # time, so a filesystem with sub-page blocks does not leave every block but the
    # first sparse.
    end = offset + length
    chunk = bytes(mmap.PAGESIZE)
    position = offset
    while position < end:
        span = min(mmap.PAGESIZE, end - position)
        count = os.pwrite(file_descriptor, chunk[:span], position)
        if count == 0:
            raise OSError(errno.EIO, "zero-length write reserving KVCR pool pages")
        position += count
