# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ctypes
import errno
import fcntl
import mmap
import os
import stat
from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import msgspec
import pytest
from _kvcr_test_utils import manual_pool_spec

from kvcr import memory as kvcr_memory
from kvcr.memory import (
    _JOURNAL_HEADER_BYTES,
    KVCRPoolAttachment,
    KVCRPoolSpec,
    _compute_pool_geometry,
    _KVCRPoolOwner,
    _populate_pages,
    _reclaim_pool_if_orphaned,
    _try_lock_adoption,
)

_TEST_GENERATION = "0123456789abcdef0123456789abcdef"
_TEST_JOURNAL_BYTES = 8192


@pytest.fixture
def pool_owner(tmp_path: Path) -> Iterator[_KVCRPoolOwner]:
    owner = _KVCRPoolOwner.allocate(
        pool_id="engine",
        pool_size_bytes=12288,
        journal_bytes=_TEST_JOURNAL_BYTES,
        pool_dir=tmp_path,
    )
    try:
        yield owner
    finally:
        owner.close()


@pytest.mark.parametrize(
    ("requested_bytes", "block_size_bytes", "expected"),
    [
        (4096, 1024, nullcontext((4096, 4))),
        (4097, 1024, nullcontext((4096, 4))),
        (1023, 1024, pytest.raises(ValueError, match="one complete KV block")),
        (True, 1, pytest.raises(TypeError)),
        (0, 1, pytest.raises(ValueError)),
        (1, True, pytest.raises(TypeError)),
        (1, 0, pytest.raises(ValueError)),
    ],
)
def test_pool_geometry_is_row_aligned_and_validated(
    requested_bytes: int,
    block_size_bytes: int,
    expected: AbstractContextManager[tuple[int, int] | None],
) -> None:
    """Geometry snaps down to whole KV blocks and refuses invalid input."""
    with expected as geometry:
        assert _compute_pool_geometry(requested_bytes, block_size_bytes) == geometry


def test_an_allocated_pool_serves_attachments_and_only_its_owner_unlinks_it(
    tmp_path: Path,
) -> None:
    """A pool is 0600, fully backed, attachable, and unlinked only by its owner."""

    with patch("kvcr.memory.uuid.uuid4") as uuid4:
        uuid4.return_value.hex = _TEST_GENERATION
        with patch("kvcr.memory._populate_pages", wraps=_populate_pages) as populate:
            owner = _KVCRPoolOwner.allocate(
                pool_id="engine_dp0",
                pool_size_bytes=12289,
                journal_bytes=_TEST_JOURNAL_BYTES,
                pool_dir=tmp_path,
            )
        try:
            spec = owner.spec
            path = Path(spec.path)
            file_stat = path.stat()
            assert spec == KVCRPoolSpec(
                pool_id="engine_dp0",
                path=str(path),
                generation=_TEST_GENERATION,
                device=file_stat.st_dev,
                inode=file_stat.st_ino,
                mapping_bytes=12289,
                journal_bytes=_TEST_JOURNAL_BYTES,
            )
            assert stat.S_IMODE(file_stat.st_mode) == 0o600
            assert file_stat.st_size == spec.mapping_bytes
            populate.assert_called_once()
            assert populate.call_args.args[1:] == (0, spec.mapping_bytes)
            # A fresh pool's journal header reads empty: exclusive creation,
            # truncation and population all leave only zeros behind.
            header = path.read_bytes()[:_JOURNAL_HEADER_BYTES]
            assert header == bytes(_JOURNAL_HEADER_BYTES)

            journal_marker = b"journal starts at byte zero"
            data_marker = b"data starts after the journal"
            with path.open("r+b") as pool_file:
                pool_file.write(journal_marker)
                pool_file.seek(spec.journal_bytes)
                pool_file.write(data_marker)
            with patch("kvcr.memory.mmap.mmap", wraps=mmap.mmap) as map_file:
                attachment = KVCRPoolAttachment.attach(spec)
            map_file.assert_called_once()
            assert map_file.call_args.args[1:] == (spec.mapping_bytes,)
            assert map_file.call_args.kwargs == {"access": mmap.ACCESS_WRITE}
            try:
                address = attachment.address
                assert ctypes.string_at(address, len(journal_marker)) == journal_marker
                data_address = address + spec.journal_bytes
                assert ctypes.string_at(data_address, len(data_marker)) == data_marker
            finally:
                attachment.close()
            # Detaching unmaps but never unlinks the server-owned file.
            assert path.exists()
            with pytest.raises(RuntimeError, match="closed"):
                _ = attachment.address
            with pytest.raises(FileExistsError):
                _KVCRPoolOwner.allocate(
                    pool_id="engine_dp0",
                    pool_size_bytes=12288,
                    journal_bytes=_TEST_JOURNAL_BYTES,
                    pool_dir=tmp_path,
                )
        finally:
            owner.close()
        assert not path.exists()

        # A creation that fails after its file was swapped out must not unlink
        # the replacement: the unlink is identity-guarded.
        replacement = b"replacement"

        def replace_then_fail(
            file_descriptor: object, _offset: int, length: int
        ) -> None:
            del file_descriptor, length
            [staging_path] = tmp_path.iterdir()
            staging_path.unlink()
            staging_path.write_bytes(replacement)
            raise RuntimeError("page population failed")

        with patch("kvcr.memory._populate_pages", side_effect=replace_then_fail):
            with pytest.raises(RuntimeError, match="page population failed"):
                _KVCRPoolOwner.allocate(
                    pool_id="engine_dp0",
                    pool_size_bytes=12288,
                    journal_bytes=_TEST_JOURNAL_BYTES,
                    pool_dir=tmp_path,
                )
    [replaced_path] = tmp_path.iterdir()
    assert replaced_path.read_bytes() == replacement


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("device", True, "Expected `int`"),
        ("inode", 1.0, "Expected `int`"),
        ("mapping_bytes", 4096.0, "Expected `int`"),
        ("journal_bytes", 8192.0, "Expected `int`"),
        ("generation", "f" * 32, "path does not match"),
    ],
)
def test_pool_spec_rejects_a_corrupted_grant(
    pool_owner: _KVCRPoolOwner,
    field_name: str,
    value: object,
    match: str,
) -> None:
    """A grant with a non-integer field or a foreign generation fails validation."""
    fields = msgspec.structs.asdict(pool_owner.spec)
    fields[field_name] = value
    with pytest.raises(msgspec.ValidationError, match=match):
        msgspec.convert(fields, type=KVCRPoolSpec)


@pytest.mark.parametrize(
    ("tamper", "error", "match"),
    [
        pytest.param("truncate", ValueError, "smaller than the grant", id="undersized"),
        pytest.param("chmod", PermissionError, "mode 0600", id="wrong-mode"),
        pytest.param(
            "replace",
            ValueError,
            "identity does not match the grant",
            id="replaced-identity",
        ),
    ],
)
def test_attachment_refuses_a_tampered_pool_without_mapping_it(
    pool_owner: _KVCRPoolOwner,
    tamper: str,
    error: type[Exception],
    match: str,
) -> None:
    """Attach checks size, mode, and identity against the grant before any mmap."""
    spec = pool_owner.spec
    path = Path(spec.path)
    if tamper == "truncate":
        os.truncate(path, spec.mapping_bytes // 2)
    elif tamper == "chmod":
        os.chmod(path, 0o640)
    else:
        original_stat = path.stat()
        path.unlink()
        path.write_bytes(bytes(original_stat.st_size))
        path.chmod(0o600)
        # Same device, size, and mode: only the inode betrays the swap.
        replacement_stat = path.stat()
        assert replacement_stat.st_dev == spec.device
        assert replacement_stat.st_ino != spec.inode
        assert replacement_stat.st_size == original_stat.st_size
        assert stat.S_IMODE(replacement_stat.st_mode) == 0o600
    with (
        patch("kvcr.memory.mmap.mmap") as map_file,
        pytest.raises(error, match=match),
    ):
        KVCRPoolAttachment.attach(spec)
    map_file.assert_not_called()
    assert path.exists()


@pytest.mark.parametrize(
    ("fallocate_errno", "pwrite_mode", "error_match"),
    [
        pytest.param(
            errno.ENOSPC, "real", "no space left on device", id="enospc-propagates"
        ),
        pytest.param(errno.EOPNOTSUPP, "real", None, id="unsupported-posix-fallocate"),
        pytest.param(None, "real", None, id="missing-posix-fallocate"),
        pytest.param(None, "half", None, id="partial-writes-resume"),
        pytest.param(None, "zero", "zero-length", id="zero-write-propagates"),
    ],
)
def test_page_population_backs_only_the_asked_range_without_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fallocate_errno: int | None,
    pwrite_mode: str,
    error_match: str | None,
) -> None:
    """The pwrite fallback fills exactly the asked range and surfaces failures."""
    fallocate_error: OSError | None = None
    if fallocate_errno is None:
        monkeypatch.delattr(os, "posix_fallocate", raising=False)
    else:
        fallocate_error = OSError(
            fallocate_errno, error_match or "operation not supported"
        )
        monkeypatch.setattr(
            os, "posix_fallocate", Mock(side_effect=fallocate_error), raising=False
        )

    real_pwrite = os.pwrite
    half = mmap.PAGESIZE // 2

    def half_pwrite(fd: int, data: bytes, position: int) -> int:
        # os.pwrite may legally write less than asked; the remainder must follow.
        return real_pwrite(fd, data[:half], position)

    pwrite = {"real": real_pwrite, "half": half_pwrite, "zero": lambda *_: 0}[
        pwrite_mode
    ]

    path = tmp_path / "pool"
    with path.open("w+b") as pool_file:
        pool_file.write(b"\xaa" * (2 * mmap.PAGESIZE))
        pool_file.flush()
        # Written, not mapped: a store that cannot be backed raises SIGBUS.
        with (
            patch("kvcr.memory.mmap.mmap") as map_file,
            patch("kvcr.memory.os.pwrite", side_effect=pwrite) as write,
        ):
            if error_match is None:
                _populate_pages(pool_file.fileno(), mmap.PAGESIZE, mmap.PAGESIZE)
            else:
                with pytest.raises(OSError, match=error_match) as raised:
                    _populate_pages(pool_file.fileno(), mmap.PAGESIZE, mmap.PAGESIZE)
        map_file.assert_not_called()
        if fallocate_errno == errno.ENOSPC:
            # A genuine backing failure propagates untouched, before any write.
            assert raised.value is fallocate_error
            write.assert_not_called()
        if error_match is None:
            # The live page ahead of the range is untouched; the range is backed.
            pool_file.seek(0)
            assert pool_file.read(mmap.PAGESIZE) == b"\xaa" * mmap.PAGESIZE
            assert pool_file.read() == bytes(mmap.PAGESIZE)


@pytest.mark.parametrize(
    ("error_number", "contended"),
    [
        (errno.EACCES, True),
        (errno.EAGAIN, True),
        (errno.EWOULDBLOCK, True),
        (errno.ENOLCK, False),
    ],
)
def test_pool_lock_distinguishes_contention_from_lock_failure(
    error_number: int, contended: bool
) -> None:
    """Only lock-contention errors become a failed acquisition result."""
    error = OSError(error_number, "lock failed")
    with patch.object(kvcr_memory.fcntl, "flock", side_effect=error):
        if contended:
            assert not kvcr_memory._try_lock_pool(0, exclusive=False)
        else:
            with pytest.raises(OSError) as raised:
                kvcr_memory._try_lock_pool(0, exclusive=False)
            assert raised.value is error


def test_a_snapshot_region_is_tail_only_and_a_failed_one_is_retired(
    pool_owner: _KVCRPoolOwner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed reservation leaves no tail; a served one never touches the pool."""
    spec = pool_owner.spec
    attachment = KVCRPoolAttachment.attach(spec)
    try:
        # A tail with no header reads as a region, and nothing ever retires one.
        with (
            patch(
                "kvcr.memory._populate_pages",
                side_effect=OSError(errno.ENOSPC, "no space left on device"),
            ),
            pytest.raises(OSError),
        ):
            with attachment.snapshot_region(4096):
                pass
        # Truncated back, so the next claimant finds nothing rather than a
        # region it can neither replay nor get rid of.
        with attachment.mapped_snapshot() as region:
            assert region is None

        marker = bytes(range(1, 129))
        tail_offset = spec.mapping_bytes - len(marker)
        ctypes.memmove(attachment.address, marker, len(marker))
        ctypes.memmove(attachment.address + tail_offset, marker, len(marker))
        # Force the fallback: the path that writes rather than reserves.
        monkeypatch.delattr(os, "posix_fallocate", raising=False)
        with attachment.snapshot_region(4096) as region:
            region[:8] = b"snapshot"
        assert ctypes.string_at(attachment.address, len(marker)) == marker
        assert ctypes.string_at(attachment.address + tail_offset, len(marker)) == marker
    finally:
        attachment.close()


def _open_fresh(spec: KVCRPoolSpec) -> int:
    """What the service sends a claimant: its own open-file-description."""
    return os.open(spec.path, os.O_RDWR)


def _lock_free(path: str) -> bool:
    """Whether the pool's exclusive liveness probe succeeds, i.e. it is abandoned."""
    probe = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    finally:
        os.close(probe)
    return True


def test_attach_fd_maps_the_pool_without_its_path(
    pool_owner: _KVCRPoolOwner,
) -> None:
    """The worker never needs spec.path: the pool stays reachable by descriptor."""
    spec = pool_owner.spec
    marker = b"seen through the descriptor"
    with open(spec.path, "r+b") as pool_file:
        pool_file.seek(spec.journal_bytes)
        pool_file.write(marker)
    file_descriptor = _open_fresh(spec)
    # Stand in for a pool directory the worker cannot reach.
    hidden = Path(spec.path).with_name("hidden")
    os.rename(spec.path, hidden)
    try:
        with patch("kvcr.memory.os.open", side_effect=AssertionError("opened a path")):
            attachment = KVCRPoolAttachment.attach_fd(file_descriptor, spec)
        try:
            assert ctypes.string_at(attachment.data_address, len(marker)) == marker
        finally:
            attachment.close()
    finally:
        os.rename(hidden, spec.path)


def test_attach_fd_refuses_a_tampered_descriptor_and_closes_it(
    pool_owner: _KVCRPoolOwner, tmp_path: Path
) -> None:
    """A descriptor for some other file fails the grant's identity check."""
    spec = pool_owner.spec
    impostor = tmp_path / "impostor"
    impostor.write_bytes(bytes(spec.mapping_bytes))
    impostor.chmod(0o600)
    file_descriptor = os.open(impostor, os.O_RDWR)

    with pytest.raises(ValueError, match="identity does not match"):
        KVCRPoolAttachment.attach_fd(file_descriptor, spec)

    with pytest.raises(OSError) as closed:
        os.fstat(file_descriptor)
    assert closed.value.errno == errno.EBADF


def test_attach_fd_keeps_every_other_check(tmp_path: Path) -> None:
    spec = manual_pool_spec(tmp_path)
    os.chmod(spec.path, 0o644)
    with pytest.raises(PermissionError, match="0600"):
        KVCRPoolAttachment.attach_fd(_open_fresh(spec), spec)
    os.chmod(spec.path, 0o600)

    holder = _open_fresh(spec)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        with pytest.raises(RuntimeError, match="exclusively locked"):
            KVCRPoolAttachment.attach_fd(_open_fresh(spec), spec)
    finally:
        os.close(holder)

    bigger = msgspec.structs.replace(spec, mapping_bytes=spec.mapping_bytes * 2)
    with pytest.raises(ValueError, match="smaller than the grant"):
        KVCRPoolAttachment.attach_fd(_open_fresh(spec), bigger)


def test_a_workers_lock_is_independent_of_the_guards(tmp_path: Path) -> None:
    """Each holds its own open-file-description, so either can outlive the other."""
    spec = manual_pool_spec(tmp_path)
    path = Path(spec.path)
    guard = KVCRPoolAttachment.attach(spec)
    worker = KVCRPoolAttachment.attach_fd(_open_fresh(spec), spec)

    # Worker death releases its lock only; the Guard's still holds the pool.
    worker.close()
    assert not _lock_free(spec.path)
    assert _reclaim_pool_if_orphaned(path) is False

    # A worker that outlives the Guard keeps the pool from being reclaimed.
    worker = KVCRPoolAttachment.attach_fd(_open_fresh(spec), spec)
    guard.close()
    assert not _lock_free(spec.path)
    assert _reclaim_pool_if_orphaned(path) is False
    assert path.exists()

    # With neither holding it the pool is orphaned.
    worker.close()
    assert _lock_free(spec.path)
    assert _reclaim_pool_if_orphaned(path) is True


_ADOPT_STAMP = bytes(range(32))
_ADOPT_POOL_BYTES = 12288


def _kept_pool(tmp_path: Path, stamp: bytes = _ADOPT_STAMP) -> _KVCRPoolOwner:
    return _KVCRPoolOwner.allocate(
        pool_id="pool_0",
        pool_size_bytes=_ADOPT_POOL_BYTES,
        journal_bytes=_TEST_JOURNAL_BYTES,
        pool_dir=tmp_path,
        stamp=stamp,
        keep=True,
    )


def _adopt(tmp_path: Path, stamp: bytes = _ADOPT_STAMP) -> _KVCRPoolOwner | None:
    return _KVCRPoolOwner.adopt(
        pool_id="pool_0",
        pool_size_bytes=_ADOPT_POOL_BYTES,
        journal_bytes=_TEST_JOURNAL_BYTES,
        pool_dir=tmp_path,
        stamp=stamp,
    )


def test_a_kept_pool_is_adopted_with_its_bytes_and_identity(tmp_path: Path) -> None:
    first = _kept_pool(tmp_path)
    with open(first.spec.path, "r+b") as pool:
        pool.seek(_TEST_JOURNAL_BYTES)
        pool.write(b"cached")
        # A handback region left past the pool is cut away: adoption is cold.
        pool.truncate(_ADOPT_POOL_BYTES + 8192)
    first.close()
    assert Path(first.spec.path).exists()

    second = _adopt(tmp_path)

    assert second is not None
    assert not second.held
    assert second.spec == msgspec.structs.replace(first.spec, epoch=1)
    assert [path.name for path in tmp_path.iterdir()] == [Path(first.spec.path).name]
    with open(second.spec.path, "rb") as pool:
        pool.seek(_TEST_JOURNAL_BYTES)
        assert pool.read(6) == b"cached"
    assert Path(second.spec.path).stat().st_size == _ADOPT_POOL_BYTES
    # The adopter holds the pool shared, which is what keeps a purge off it.
    assert not _reclaim_pool_if_orphaned(Path(second.spec.path))
    second.close()
    assert Path(second.spec.path).exists()


def test_adoption_takes_the_newest_pool(tmp_path: Path) -> None:
    older = _kept_pool(tmp_path)
    older.close()
    newer = _kept_pool(tmp_path)
    newer.close()
    os.utime(older.spec.path, ns=(1, 1))

    adopted = _adopt(tmp_path)

    assert adopted is not None
    assert adopted.spec.generation == newer.spec.generation
    adopted.close()


def test_a_pool_without_a_keep_is_unlinked_and_nothing_is_adopted(
    tmp_path: Path,
) -> None:
    owner = _KVCRPoolOwner.allocate(
        pool_id="pool_0",
        pool_size_bytes=_ADOPT_POOL_BYTES,
        journal_bytes=_TEST_JOURNAL_BYTES,
        pool_dir=tmp_path,
        stamp=_ADOPT_STAMP,
    )
    owner.close()

    assert _adopt(tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def _smaller(path: Path) -> None:
    os.truncate(path, _ADOPT_POOL_BYTES - 4096)


def _torn_stamp(path: Path) -> None:
    with open(path, "r+b") as pool:
        pool.seek(256)
        pool.write(_ADOPT_STAMP[:16] + bytes(16))


def _wrong_mode(path: Path) -> None:
    path.chmod(0o644)


def _symlink(path: Path) -> None:
    target = path.with_name("elsewhere")
    path.rename(target)
    path.symlink_to(target)


def _directory(path: Path) -> None:
    path.unlink()
    path.mkdir()


@pytest.mark.parametrize(
    "damage", [_smaller, _torn_stamp, _wrong_mode, _symlink, _directory]
)
def test_a_pool_that_is_not_what_was_built_is_refused(
    tmp_path: Path, damage, caplog: pytest.LogCaptureFixture
) -> None:
    kept = _kept_pool(tmp_path)
    kept.close()
    damage(Path(kept.spec.path))

    assert _adopt(tmp_path) is None
    assert "Refusing to adopt KVCR pool" in caplog.text
    # Refused, not repaired: the file is left for the purge.
    assert os.path.lexists(kept.spec.path)


def test_a_pool_built_for_other_terms_is_refused(tmp_path: Path) -> None:
    kept = _kept_pool(tmp_path)
    kept.close()

    assert _adopt(tmp_path, stamp=bytes(32)) is None


def test_adoption_is_held_by_one_service_at_a_time(tmp_path: Path) -> None:
    first = _try_lock_adoption(tmp_path)
    assert first is not None
    assert _try_lock_adoption(tmp_path) is None
    os.close(first)
    second = _try_lock_adoption(tmp_path)
    assert second is not None
    os.close(second)


def test_a_pool_a_previous_worker_holds_is_adopted_held_until_it_lets_go(
    tmp_path: Path,
) -> None:
    kept = _kept_pool(tmp_path)
    kept.close()
    # A worker of the previous service: its own open file description.
    worker = os.open(kept.spec.path, os.O_RDWR)
    fcntl.flock(worker, fcntl.LOCK_SH)
    os.truncate(kept.spec.path, _ADOPT_POOL_BYTES + 8192)

    adopted = _adopt(tmp_path)

    assert adopted is not None
    assert adopted.held
    assert not adopted.try_unhold()
    # Nothing of ours is held yet, and the worker's pool is left as it was.
    assert Path(adopted.spec.path).stat().st_size == _ADOPT_POOL_BYTES + 8192
    os.close(worker)
    assert adopted.try_unhold()
    assert not adopted.held
    assert Path(adopted.spec.path).stat().st_size == _ADOPT_POOL_BYTES
    assert not _reclaim_pool_if_orphaned(Path(adopted.spec.path))
    adopted.close()


@pytest.fixture
def _madv_dontfork(monkeypatch: pytest.MonkeyPatch) -> None:
    # Absent on macOS; the attach path only needs the call to succeed.
    monkeypatch.setattr(mmap, "MADV_DONTFORK", mmap.MADV_NORMAL, raising=False)


def _attach(spec: KVCRPoolSpec) -> KVCRPoolAttachment:
    return KVCRPoolAttachment.attach_fd(os.open(spec.path, os.O_RDWR), spec)


@pytest.mark.usefixtures("_madv_dontfork")
def test_a_grant_from_before_an_adoption_cannot_attach_after_it(
    tmp_path: Path,
) -> None:
    first = _kept_pool(tmp_path)
    old_grant = first.spec
    assert old_grant.epoch == 0
    first.close()

    adopted = _adopt(tmp_path)

    assert adopted is not None
    assert adopted.spec.epoch == 1
    with pytest.raises(ValueError, match="adopted since this grant"):
        _attach(old_grant)
    _attach(adopted.spec).close()
    adopted.close()


@pytest.mark.usefixtures("_madv_dontfork")
def test_a_fresh_pool_and_a_spec_without_an_epoch_are_epoch_zero(
    pool_owner: _KVCRPoolOwner,
) -> None:
    assert pool_owner.spec.epoch == 0
    wire = msgspec.json.decode(msgspec.json.encode(pool_owner.spec))
    del wire["epoch"]
    legacy = msgspec.convert(wire, KVCRPoolSpec)
    assert legacy.epoch == 0
    _attach(legacy).close()
