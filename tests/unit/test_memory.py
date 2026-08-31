# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ctypes
import errno
import mmap
import os
import stat
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import Mock, patch

import msgspec
import pytest

from kvcr import memory as kvcr_memory
from kvcr.memory import (
    _JOURNAL_HEADER_BYTES,
    KVCRPoolAttachment,
    KVCRPoolSpec,
    _compute_pool_geometry,
    _KVCRPoolOwner,
    _populate_pages,
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
    ("requested_bytes", "row_stride", "expected"),
    [
        (4096, 1024, (4096, 4)),
        (4097, 1024, (4096, 4)),
        (1023, 1024, pytest.raises(ValueError, match="one complete KV row")),
        (True, 1, pytest.raises(TypeError)),
        (0, 1, pytest.raises(ValueError)),
        (1, True, pytest.raises(TypeError)),
        (1, 0, pytest.raises(ValueError)),
    ],
)
def test_pool_geometry_is_row_aligned_and_validated(
    requested_bytes: int,
    row_stride: int,
    expected: tuple[int, int] | AbstractContextManager[object],
) -> None:
    """Geometry snaps down to whole KV rows and refuses non-positive or bool input."""
    if isinstance(expected, tuple):
        assert _compute_pool_geometry(requested_bytes, row_stride) == expected
    else:
        with expected:
            _compute_pool_geometry(requested_bytes, row_stride)


def test_an_allocated_pool_serves_attachments_and_only_its_owner_unlinks_it(
    tmp_path: Path,
) -> None:
    """A pool is 0600, fully backed, attachable, and unlinked only by its owner."""

    def populate_then_dirty_header(
        file_descriptor: int, offset: int, length: int
    ) -> None:
        _populate_pages(file_descriptor, offset, length)
        os.pwrite(file_descriptor, b"\xff" * _JOURNAL_HEADER_BYTES, 0)

    with patch("kvcr.memory.uuid.uuid4") as uuid4:
        uuid4.return_value.hex = _TEST_GENERATION
        with patch(
            "kvcr.memory._populate_pages", side_effect=populate_then_dirty_header
        ) as populate:
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
            # Zeroed after page population dirtied it: the journal header is
            # written last, so a crash mid-creation cannot leave a stale one.
            with path.open("rb") as pool_file:
                assert pool_file.read(_JOURNAL_HEADER_BYTES) == bytes(
                    _JOURNAL_HEADER_BYTES
                )

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
                assert (
                    ctypes.string_at(attachment.address, len(journal_marker))
                    == journal_marker
                )
                assert (
                    ctypes.string_at(
                        attachment.address + spec.journal_bytes, len(data_marker)
                    )
                    == data_marker
                )
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
