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
    KVCRPoolAttachment,
    KVCRPoolSpec,
    _compute_pool_geometry,
    _KVCRPoolOwner,
    _populate_pages,
)

_TEST_GENERATION = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def pool_owner(tmp_path: Path) -> Iterator[_KVCRPoolOwner]:
    owner = _KVCRPoolOwner.allocate(
        pool_id="engine",
        pool_size_bytes=4096,
        pool_dir=tmp_path,
    )
    owner.finalize(row_stride=1024)
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


def test_pool_is_data_only_private_and_page_populated(tmp_path: Path) -> None:
    with (
        patch("kvcr.memory.uuid.uuid4") as uuid4,
        patch("kvcr.memory._populate_pages", wraps=_populate_pages) as populate,
    ):
        uuid4.return_value.hex = _TEST_GENERATION
        owner = _KVCRPoolOwner.allocate(
            pool_id="engine_dp0",
            pool_size_bytes=8193,
            pool_dir=tmp_path,
        )
        try:
            spec = owner.finalize(row_stride=4096)
            path = Path(spec.path)
            file_stat = path.stat()
            assert spec == KVCRPoolSpec(
                pool_id="engine_dp0",
                path=str(path),
                generation=_TEST_GENERATION,
                device=file_stat.st_dev,
                inode=file_stat.st_ino,
                effective_bytes=8192,
                rows=2,
                row_stride=4096,
            )
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert path.stat().st_size == 8193
            populate.assert_called_once()
            assert populate.call_args.args[1] == 8193

            marker = b"data starts at byte zero"
            with path.open("r+b") as pool_file:
                pool_file.write(marker)
            with patch("kvcr.memory.mmap.mmap", wraps=mmap.mmap) as map_file:
                attachment = KVCRPoolAttachment.attach(spec)
            map_file.assert_called_once()
            assert map_file.call_args.args[1:] == (spec.effective_bytes,)
            assert map_file.call_args.kwargs == {"access": mmap.ACCESS_WRITE}
            try:
                assert ctypes.string_at(attachment.address, len(marker)) == marker
            finally:
                attachment.close()
            with pytest.raises(FileExistsError):
                _KVCRPoolOwner.allocate(
                    pool_id="engine_dp0",
                    pool_size_bytes=8192,
                    pool_dir=tmp_path,
                )
        finally:
            owner.close()
    assert not path.exists()


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
        ("effective_bytes", 4096.0),
        ("rows", 4.0),
        ("row_stride", 1024.0),
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
    os.truncate(pool_owner.spec.path, pool_owner.spec.effective_bytes // 2)
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

    def replace_then_fail(file_descriptor: object, length: int) -> None:
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
                pool_size_bytes=4096,
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
        _populate_pages(3, mmap.PAGESIZE)

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
        pool_file.truncate(mmap.PAGESIZE)
        with patch("kvcr.memory.mmap.mmap", wraps=mmap.mmap) as map_file:
            _populate_pages(pool_file.fileno(), mmap.PAGESIZE)

        map_file.assert_called_once_with(
            pool_file.fileno(),
            mmap.PAGESIZE,
            access=mmap.ACCESS_WRITE,
        )
        pool_file.seek(0)
        assert pool_file.read() == bytes(mmap.PAGESIZE)


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
