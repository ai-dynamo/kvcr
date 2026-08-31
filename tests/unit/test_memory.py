# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ctypes
import errno
import mmap
import os
import stat
from collections.abc import Iterator
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
    _snapshot_offset,
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
        (1023, 1024, None),
    ],
)
def test_pool_geometry_is_row_aligned(
    requested_bytes: int,
    row_stride: int,
    expected: tuple[int, int] | None,
) -> None:
    if expected is None:
        with pytest.raises(ValueError, match="one complete KV row"):
            _compute_pool_geometry(requested_bytes, row_stride)
    else:
        assert _compute_pool_geometry(requested_bytes, row_stride) == expected


@pytest.mark.parametrize(
    ("requested_bytes", "row_stride", "error"),
    [
        (True, 1, TypeError),
        (0, 1, ValueError),
        (1, True, TypeError),
        (1, 0, ValueError),
    ],
)
def test_pool_geometry_rejects_invalid_values(
    requested_bytes: object,
    row_stride: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        _compute_pool_geometry(requested_bytes, row_stride)  # type: ignore[arg-type]


def test_pool_is_raw_private_and_page_populated(tmp_path: Path) -> None:
    with (
        patch("kvcr.memory.uuid.uuid4") as uuid4,
        patch("kvcr.memory._populate_pages", wraps=_populate_pages) as populate,
    ):
        uuid4.return_value.hex = _TEST_GENERATION
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
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert path.stat().st_size == spec.mapping_bytes
            populate.assert_called_once()
            assert populate.call_args.args[1:] == (0, spec.mapping_bytes)

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


def test_pool_creation_initializes_the_journal_header(tmp_path: Path) -> None:
    def dirty_header(file_descriptor: int, _offset: int, _length: int) -> None:
        os.pwrite(file_descriptor, b"\xff" * _JOURNAL_HEADER_BYTES, 0)

    with patch("kvcr.memory._populate_pages", side_effect=dirty_header):
        owner = _KVCRPoolOwner.allocate(
            pool_id="engine",
            pool_size_bytes=12288,
            journal_bytes=_TEST_JOURNAL_BYTES,
            pool_dir=tmp_path,
        )
    try:
        with open(owner._path, "rb") as pool_file:
            assert pool_file.read(_JOURNAL_HEADER_BYTES) == bytes(_JOURNAL_HEADER_BYTES)
    finally:
        owner.close()


def test_attachment_close_does_not_unlink(pool_owner: _KVCRPoolOwner) -> None:
    path = Path(pool_owner.spec.path)
    attachment = KVCRPoolAttachment.attach(pool_owner.spec)
    assert attachment.address > 0
    attachment.close()
    assert path.exists()
    with pytest.raises(RuntimeError, match="closed"):
        _ = attachment.address
    pool_owner.close()
    assert not path.exists()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("device", True),
        ("inode", 1.0),
        ("mapping_bytes", 4096.0),
        ("journal_bytes", 8192.0),
    ],
)
def test_pool_spec_rejects_non_integer_fields(
    pool_owner: _KVCRPoolOwner,
    field_name: str,
    value: object,
) -> None:
    fields = msgspec.structs.asdict(pool_owner.spec)
    fields[field_name] = value
    with pytest.raises(msgspec.ValidationError, match="Expected `int`"):
        msgspec.convert(fields, type=KVCRPoolSpec)


def test_attachment_rejects_wrong_size(pool_owner: _KVCRPoolOwner) -> None:
    os.truncate(pool_owner.spec.path, pool_owner.spec.mapping_bytes // 2)
    with pytest.raises(ValueError, match="smaller than the grant"):
        KVCRPoolAttachment.attach(pool_owner.spec)


def test_pool_spec_rejects_path_for_different_generation(
    pool_owner: _KVCRPoolOwner,
) -> None:
    with pytest.raises(ValueError, match="path does not match"):
        msgspec.structs.replace(pool_owner.spec, generation="f" * 32)


def test_attachment_rejects_replaced_pool_identity(
    pool_owner: _KVCRPoolOwner,
) -> None:
    spec = pool_owner.spec
    path = Path(spec.path)
    original_stat = path.stat()
    path.unlink()
    path.write_bytes(bytes(original_stat.st_size))
    path.chmod(0o600)

    replacement_stat = path.stat()
    assert replacement_stat.st_dev == spec.device
    assert replacement_stat.st_ino != spec.inode
    assert replacement_stat.st_size == original_stat.st_size
    assert stat.S_IMODE(replacement_stat.st_mode) == 0o600
    with (
        patch("kvcr.memory.mmap.mmap") as map_file,
        pytest.raises(ValueError, match="identity does not match the grant"),
    ):
        KVCRPoolAttachment.attach(spec)
    map_file.assert_not_called()
    assert path.exists()


def test_creation_failure_does_not_unlink_replacement(tmp_path: Path) -> None:
    replacement = b"replacement"

    def replace_then_fail(file_descriptor: object, _offset: int, length: int) -> None:
        del file_descriptor, length
        [path] = tmp_path.iterdir()
        path.unlink()
        path.write_bytes(replacement)
        raise RuntimeError("page population failed")

    with (
        patch("kvcr.memory.uuid.uuid4") as uuid4,
        patch("kvcr.memory._populate_pages", side_effect=replace_then_fail),
    ):
        uuid4.return_value.hex = _TEST_GENERATION
        with pytest.raises(RuntimeError, match="page population failed"):
            _KVCRPoolOwner.allocate(
                pool_id="engine",
                pool_size_bytes=12288,
                journal_bytes=_TEST_JOURNAL_BYTES,
                pool_dir=tmp_path,
            )
    [path] = tmp_path.iterdir()
    assert path.read_bytes() == replacement


def test_page_population_propagates_enospc_without_mapping() -> None:
    error = OSError(errno.ENOSPC, "no space left on device")
    with (
        patch(
            "kvcr.memory.os.posix_fallocate",
            side_effect=error,
            create=True,
        ),
        patch("kvcr.memory.mmap.mmap") as map_file,
        pytest.raises(OSError) as raised,
    ):
        _populate_pages(3, 0, mmap.PAGESIZE)

    assert raised.value is error
    map_file.assert_not_called()


@pytest.mark.parametrize(
    "has_posix_fallocate",
    [
        pytest.param(True, id="unsupported-posix-fallocate"),
        pytest.param(False, id="missing-posix-fallocate"),
    ],
)
def test_page_population_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_posix_fallocate: bool,
) -> None:
    if has_posix_fallocate:
        monkeypatch.setattr(
            os,
            "posix_fallocate",
            Mock(side_effect=OSError(errno.EOPNOTSUPP, "operation not supported")),
            raising=False,
        )
    else:
        monkeypatch.delattr(os, "posix_fallocate", raising=False)

    path = tmp_path / "pool"
    with path.open("w+b") as pool_file:
        pool_file.truncate(2 * mmap.PAGESIZE)
        # Written, not mapped: a store that cannot be backed raises SIGBUS.
        with (
            patch("kvcr.memory.mmap.mmap", wraps=mmap.mmap) as map_file,
            patch("kvcr.memory.os.pwrite", wraps=os.pwrite) as write,
        ):
            _populate_pages(pool_file.fileno(), mmap.PAGESIZE, mmap.PAGESIZE)

        map_file.assert_not_called()
        assert [call.args[2] for call in write.call_args_list] == [mmap.PAGESIZE]
        pool_file.seek(0)
        assert pool_file.read() == bytes(2 * mmap.PAGESIZE)


def test_page_population_survives_partial_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """os.pwrite may legally write less than asked; the remainder must follow."""
    monkeypatch.delattr(os, "posix_fallocate", raising=False)
    path = tmp_path / "pool"
    with path.open("w+b") as pool_file:
        pool_file.truncate(mmap.PAGESIZE)
        half = mmap.PAGESIZE // 2
        real_pwrite = os.pwrite

        def half_pwrite(fd: int, data: bytes, position: int) -> int:
            return real_pwrite(fd, data[:half], position)

        with patch("kvcr.memory.os.pwrite", side_effect=half_pwrite):
            _populate_pages(pool_file.fileno(), 0, mmap.PAGESIZE)

        pool_file.seek(0)
        assert pool_file.read() == bytes(mmap.PAGESIZE)

        with (
            patch("kvcr.memory.os.pwrite", return_value=0),
            pytest.raises(OSError, match="zero-length"),
        ):
            _populate_pages(pool_file.fileno(), 0, mmap.PAGESIZE)


def test_attachment_rejects_wrong_permissions(
    pool_owner: _KVCRPoolOwner,
) -> None:
    os.chmod(pool_owner.spec.path, 0o640)
    with pytest.raises(PermissionError, match="mode 0600"):
        KVCRPoolAttachment.attach(pool_owner.spec)


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


def test_a_snapshot_region_that_cannot_be_backed_leaves_no_tail(
    pool_owner: _KVCRPoolOwner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tail with no header reads as a region, and nothing ever retires one."""
    spec = pool_owner.spec
    attachment = KVCRPoolAttachment.attach(spec)
    try:
        monkeypatch.setattr(
            "kvcr.memory._populate_pages",
            Mock(side_effect=OSError(errno.ENOSPC, "no space left on device")),
        )

        with pytest.raises(OSError):
            with attachment.snapshot_region(4096):
                pass

        # Truncated back, so the next claimant finds nothing rather than a
        # region it can neither replay nor get rid of.
        assert os.fstat(attachment._file_descriptor).st_size == _snapshot_offset(
            spec.mapping_bytes
        )
        with attachment.mapped_snapshot() as region:
            assert region is None
    finally:
        attachment.close()


def test_reserving_a_snapshot_region_does_not_touch_the_live_pool(
    pool_owner: _KVCRPoolOwner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reservation fallback writes, so its range has to be the tail only."""
    spec = pool_owner.spec
    attachment = KVCRPoolAttachment.attach(spec)
    try:
        marker = bytes(range(1, 129))
        mapping = attachment._mapping
        assert mapping is not None
        mapping[: len(marker)] = marker
        tail_start = spec.mapping_bytes - len(marker)
        mapping[tail_start:] = marker

        # Force the fallback: the path that writes rather than reserves.
        monkeypatch.delattr(os, "posix_fallocate", raising=False)
        with attachment.snapshot_region(4096) as region:
            region[:8] = b"snapshot"

        assert bytes(mapping[: len(marker)]) == marker
        assert bytes(mapping[tail_start:]) == marker
    finally:
        attachment.close()
