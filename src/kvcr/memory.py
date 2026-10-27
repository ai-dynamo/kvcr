# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Server-owned shared-memory pools for KVCR local DRAM."""

import contextlib
import ctypes
import errno
import fcntl
import hmac
import logging
import mmap
import os
import re
import stat
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import msgspec

logger = logging.getLogger(__name__)

_POOL_MODE = 0o600
_POOL_PREFIX = "kvcr"


def _snapshot_offset(mapping_bytes: int) -> int:
    """Where a pool's handback region starts, past everything it was granted."""
    granularity = mmap.ALLOCATIONGRANULARITY
    return -(-mapping_bytes // granularity) * granularity


_JOURNAL_HEADER_BYTES = 4096
# Where a pool records the terms it was built for: in the journal header page,
# clear of the three cursors at 0, 128 and 192.
_ADOPT_STAMP_OFFSET = 256
_ADOPT_STAMP_BYTES = 32
# How many times a pool has been adopted. A grant carries the epoch it was made
# under; an attachment refuses a pool adopted since.
_ADOPT_EPOCH_OFFSET = _ADOPT_STAMP_OFFSET + _ADOPT_STAMP_BYTES
_ADOPT_EPOCH_BYTES = 8
# Never removed: it holds no state, and unlinking it would let two services
# lock different files. The purge ignores it (it is not a pool name).
_ADOPT_LOCK_NAME = "adopt.lock"
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
    epoch: Annotated[int, msgspec.Meta(ge=0)] = 0

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
        return cls.attach_fd(os.open(spec.path, _OPEN_FLAGS), spec)

    @classmethod
    def attach_fd(
        cls, file_descriptor: int, spec: KVCRPoolSpec
    ) -> "KVCRPoolAttachment":
        """Map a pool from a descriptor the service opened; spec.path is never used.

        Takes ownership of the descriptor, closing it on failure. Every check
        `attach` makes still runs against the descriptor itself.
        """
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
            # After the lock: an adopter holds it exclusively while it bumps the
            # epoch, so a grant made before an adoption cannot map the pool after.
            header_epoch = int.from_bytes(
                os.pread(file_descriptor, _ADOPT_EPOCH_BYTES, _ADOPT_EPOCH_OFFSET),
                "little",
            )
            if header_epoch != spec.epoch:
                raise ValueError(
                    f"KVCR pool was adopted since this grant: {spec.path} "
                    f"(grant epoch {spec.epoch}, pool epoch {header_epoch})"
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
            # Mapping must close before the truncate; shrinking under it would fault.
            with mmap.mmap(
                self._file_descriptor, size, offset=offset, access=mmap.ACCESS_WRITE
            ) as region:
                yield region
        except BaseException:
            # Truncate back: a partial or stale tail still reads as a region; a
            # claimant replaying it would diverge from the failed Guard's mirror.
            with contextlib.suppress(OSError):
                os.ftruncate(self._file_descriptor, offset)
            raise

    @contextlib.contextmanager
    def mapped_snapshot(self) -> Iterator[mmap.mmap | None]:
        """Whatever a previous Guard left past the pool, if anything is there."""
        offset = _snapshot_offset(self._spec.mapping_bytes)
        size = os.fstat(self._file_descriptor).st_size - offset
        if size <= 0:
            yield None
            return
        with mmap.mmap(
            self._file_descriptor, size, offset=offset, access=mmap.ACCESS_READ
        ) as region:
            yield region

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
        keep: bool = False,
        held: bool = False,
    ) -> None:
        self.spec = spec
        self._file_descriptor = file_descriptor
        self._keep = keep
        # Adopted while another process still held the pool: no lock of ours yet.
        self.held = held
        # What the header will say once this owner holds the pool.
        self._epoch = spec.epoch
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
        stamp: bytes | None = None,
        keep: bool = False,
    ) -> "_KVCRPoolOwner":
        """Create the file and reserve its data space.

        Created and locked under the directory guard, which a purge takes
        exclusively, so a pool under construction cannot be reclaimed. A stamp
        is recorded in the header for a later `adopt`; `keep` leaves the file
        in place on close so that one can happen.
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
                spec = KVCRPoolSpec(
                    pool_id=pool_id,
                    path=str(path),
                    generation=generation,
                    device=file_identity[0],
                    inode=file_identity[1],
                    mapping_bytes=pool_size_bytes,
                    journal_bytes=journal_bytes,
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
            if stamp is not None:
                os.pwrite(file_descriptor, stamp, _ADOPT_STAMP_OFFSET)
        except BaseException:
            os.close(file_descriptor)
            # Identity-guarded: something may have replaced the file since it
            # was created, and that replacement is not ours.
            _unlink_if_identity(path, file_identity)
            raise
        return cls(spec, file_descriptor, keep)

    @classmethod
    def adopt(
        cls,
        *,
        pool_id: str,
        pool_size_bytes: int,
        journal_bytes: int,
        pool_dir: str | os.PathLike[str],
        stamp: bytes,
    ) -> "_KVCRPoolOwner | None":
        """Take over the newest pool a previous service left for `pool_id`.

        Cold: the bytes and the file's identity carry over, nothing about what
        they hold does; a handback region past the pool is cut away. A file
        that is not exactly what this configuration would have allocated is
        refused (and logged) rather than repaired, and a refused file is left
        for the purge. Returns None when there is nothing to adopt. The owner
        may come back `held`: see `try_unhold`.

        "Newest" is the last write time: the pool the previous service was
        using, not the one it happened to create last.
        """
        _validate_pool_layout(pool_size_bytes, journal_bytes)
        directory = Path(pool_dir).resolve()
        name = re.compile(rf"{_POOL_PREFIX}-{re.escape(pool_id)}-([0-9a-f]{{32}})")

        def written(path: Path) -> int:
            try:
                return path.lstat().st_mtime_ns
            except OSError:
                return 0

        candidates = sorted(directory.glob(f"{_POOL_PREFIX}-{pool_id}-*"), key=written)
        for path in reversed(candidates):
            match = name.fullmatch(path.name)
            if match is None:
                continue
            try:
                # Shared, like creation: a purge cannot reclaim it under us.
                with _pool_dir_guard(directory, exclusive=False):
                    return cls._adopt_file(
                        path,
                        pool_id,
                        match.group(1),
                        pool_size_bytes,
                        journal_bytes,
                        stamp,
                    )
            except (OSError, ValueError, RuntimeError) as error:
                logger.warning("Refusing to adopt KVCR pool %s: %s", path, error)
        return None

    @classmethod
    def _adopt_file(
        cls,
        path: Path,
        pool_id: str,
        generation: str,
        pool_size_bytes: int,
        journal_bytes: int,
        stamp: bytes,
    ) -> "_KVCRPoolOwner":
        file_descriptor = os.open(path, _OPEN_FLAGS)
        try:
            file_stat = os.fstat(file_descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("it is not a regular file")
            if stat.S_IMODE(file_stat.st_mode) != _POOL_MODE:
                raise PermissionError("it does not have mode 0600")
            if file_stat.st_size < pool_size_bytes:
                raise ValueError(
                    f"it is smaller than configured: {file_stat.st_size} "
                    f"< {pool_size_bytes}"
                )
            recorded = os.pread(
                file_descriptor, _ADOPT_STAMP_BYTES, _ADOPT_STAMP_OFFSET
            )
            if not hmac.compare_digest(recorded, stamp):
                raise ValueError("it was not built for this configuration")
            spec = KVCRPoolSpec(
                pool_id=pool_id,
                path=str(path),
                generation=generation,
                device=file_stat.st_dev,
                inode=file_stat.st_ino,
                mapping_bytes=pool_size_bytes,
                journal_bytes=journal_bytes,
                epoch=1
                + int.from_bytes(
                    os.pread(file_descriptor, _ADOPT_EPOCH_BYTES, _ADOPT_EPOCH_OFFSET),
                    "little",
                ),
            )
            owner = cls(spec, file_descriptor, keep=True, held=True)
            unheld = owner.try_unhold()
        except BaseException:
            os.close(file_descriptor)
            raise
        if unheld:
            logger.info(
                "Adopted KVCR pool %s (cold: its bytes are kept, its contents are "
                "not recovered)",
                path,
            )
        else:
            logger.warning(
                "Adopted KVCR pool %s but a previous worker still holds it; "
                "it is not claimable until that worker is gone",
                path,
            )
        return owner

    def try_unhold(self) -> bool:
        """Make an adopted pool ours if nobody else holds it any more.

        Probed with an exclusive lock while we hold nothing of our own, since
        a shared one could not be told from a previous worker's. Under that
        lock the epoch is bumped, so a grant made before this adoption fails to
        attach after it, and the lock is then converted to the usual shared
        one. Under the shared directory guard, so no purge reclaims the file.
        """
        if not self.held:
            return True
        with _pool_dir_guard(self._path.parent, exclusive=False):
            if os.fstat(self._file_descriptor).st_nlink == 0:
                raise RuntimeError(f"KVCR pool was unlinked while held: {self._path}")
            if not _try_lock_pool(self._file_descriptor, exclusive=True):
                return False
            os.pwrite(
                self._file_descriptor,
                self._epoch.to_bytes(_ADOPT_EPOCH_BYTES, "little"),
                _ADOPT_EPOCH_OFFSET,
            )
            if not _try_lock_pool(self._file_descriptor, exclusive=False):
                # Linux's failed conversion leaves no lock; BSD/macOS may keep the old.
                fcntl.flock(self._file_descriptor, fcntl.LOCK_UN)
                return False
            if os.fstat(self._file_descriptor).st_size > self.spec.mapping_bytes:
                os.ftruncate(self._file_descriptor, self.spec.mapping_bytes)
        self.held = False
        return True

    def close(self) -> None:
        """Close the owned pool, unlinking it unless it is kept for adoption."""
        if self._file_descriptor != -1:
            os.close(self._file_descriptor)
            self._file_descriptor = -1
        if not self._keep:
            _unlink_if_identity(self._path, self._file_identity)


def _compute_pool_geometry(
    requested_bytes: int,
    block_size_bytes: int,
) -> tuple[int, int]:
    _validate_positive_int("requested_bytes", requested_bytes)
    _validate_positive_int("block_size_bytes", block_size_bytes)
    block_count = requested_bytes // block_size_bytes
    if block_count == 0:
        raise ValueError(
            "requested_bytes must hold at least one complete KV block "
            f"of {block_size_bytes} bytes"
        )
    return block_count * block_size_bytes, block_count


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


def _try_lock_adoption(directory: Path) -> int | None:
    """Take the right to adopt this directory's pools until the descriptor closes.

    Not the directory guard (every allocation takes that shared) and not the
    pools' locks (workers and services both hold those shared). Never mix
    adopting and legacy services on one directory: a legacy purge ignores this.
    """
    file_descriptor = os.open(
        directory / _ADOPT_LOCK_NAME,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        _POOL_MODE,
    )
    try:
        if _try_lock_pool(file_descriptor, exclusive=True):
            return file_descriptor
    except BaseException:
        os.close(file_descriptor)
        raise
    os.close(file_descriptor)
    return None


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
    must not touch the pool itself. The pwrite fallback is O(pages), latent on Linux.
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
    chunk, position, end = bytes(mmap.PAGESIZE), offset, offset + length
    while position < end:
        span = min(mmap.PAGESIZE, end - position)
        count = os.pwrite(file_descriptor, chunk[:span], position)
        if count == 0:
            raise OSError(errno.EIO, "zero-length write reserving KVCR pool pages")
        position += count
