# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Configuration and real shared-mapping coverage for opt-in THP."""

import ctypes
import mmap
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import msgspec
import pytest

from kvcr.kvcr_service import _parse_args
from kvcr.memory import KVCRPoolAttachment, KVCRPoolSpec, _KVCRPoolOwner
from kvcr.recovery_journal import RecoveryJournal


def service_args():
    return [
        "--socket-path",
        "/tmp/kvcr.sock",
        "--pool-dir",
        "/dev/shm",
        "--guard-count",
        "1",
        "--pool-sizes-gb",
        "1",
        "--compatibility-digest",
        "test",
    ]


def test_memory_config_defaults_and_thp_json():
    assert _parse_args(service_args()).memory_config.mode == "default"
    assert (
        _parse_args(
            service_args() + ["--memory-config", '{"mode":"thp"}']
        ).memory_config.mode
        == "thp"
    )


@pytest.mark.parametrize(
    "value",
    [
        '{"mode":"hugetlb"}',
        '{"mode":"typo"}',
        '{"mode":1}',
        '{"mod":"thp"}',
        "null",
        "{",
    ],
)
def test_memory_config_rejects_invalid_json(value):
    with pytest.raises(SystemExit):
        _parse_args(service_args() + ["--memory-config", value])


def test_direct_memory_config_rejects_unsupported_mode():
    from kvcr.config import KVCRMemoryConfig

    with pytest.raises(ValueError, match="mode"):
        KVCRMemoryConfig(mode="hugetlb")


@pytest.fixture
def thp_owner():
    from kvcr.hugepages import thp_page_size

    page = thp_page_size()
    if page > 32 << 20:
        pytest.skip("real unit probe limited to 64 MiB; use deployment tests on Grace")
    with tempfile.TemporaryDirectory(prefix="kvcr-thp-", dir="/dev/shm") as d:
        owner = _KVCRPoolOwner.allocate(
            pool_id="test",
            pool_size_bytes=page * 2,
            journal_bytes=8192,
            pool_dir=d,
            memory_mode="thp",
        )
        try:
            yield owner
        finally:
            owner.close()


def test_pool_spec_carries_mode_and_old_grants_default(tmp_path):
    owner = _KVCRPoolOwner.allocate(
        pool_id="test", pool_size_bytes=12288, journal_bytes=8192, pool_dir=tmp_path
    )
    try:
        data = msgspec.to_builtins(owner.spec)
        data.pop("memory_mode", None)
        assert msgspec.convert(data, type=KVCRPoolSpec).memory_mode == "default"
        attachment = KVCRPoolAttachment.attach(owner.spec)
        try:
            assert isinstance(attachment._require_mapping(), mmap.mmap)
        finally:
            attachment.close()
    finally:
        owner.close()


def test_thp_shared_journal_and_export_lifetime(thp_owner):
    from kvcr.hugepages import thp_page_size

    assert (
        msgspec.json.decode(
            msgspec.json.encode(thp_owner.spec), type=KVCRPoolSpec
        ).memory_mode
        == "thp"
    )
    writer = KVCRPoolAttachment.attach(thp_owner.spec)
    reader = KVCRPoolAttachment.attach(thp_owner.spec)
    try:
        assert writer.address % thp_page_size() == 0
        assert writer._require_mapping().huge_bytes == thp_owner.spec.mapping_bytes
        producer, consumer = RecoveryJournal(writer), RecoveryJournal(reader)
        producer.reset()
        assert producer.publish(1, b"key", b"payload")
        assert consumer.read_next() == (1, b"key", b"payload")
        view = memoryview(writer._require_mapping())
        with pytest.raises(BufferError):
            writer.close()
        view.release()
    finally:
        reader.close()
        writer.close()


def test_thp_mapping_shared_across_processes_and_snapshot(thp_owner):
    a = KVCRPoolAttachment.attach(thp_owner.spec)
    try:
        ctypes.memmove(a.data_address, b"parent", 6)
        code = """
import ctypes,sys,msgspec
from kvcr.memory import KVCRPoolAttachment,KVCRPoolSpec
spec=msgspec.json.decode(sys.argv[1],type=KVCRPoolSpec)
a=KVCRPoolAttachment.attach(spec)
assert ctypes.string_at(a.data_address,6)==b'parent'
ctypes.memmove(a.data_address,b'child!',6)
a.close()
"""
        subprocess.run(
            [sys.executable, "-c", code, msgspec.json.encode(thp_owner.spec).decode()],
            check=True,
            capture_output=True,
            text=True,
        )
        assert ctypes.string_at(a.data_address, 6) == b"child!"
        with a.snapshot_region(12345) as snapshot:
            snapshot[:4] = b"tail"
        with a.mapped_snapshot() as snapshot:
            assert snapshot[:4] == b"tail"
        a.release_snapshot_region()
        assert os.stat(thp_owner.spec.path).st_size == thp_owner.spec.mapping_bytes
        assert ctypes.string_at(a.data_address, 6) == b"child!"
    finally:
        a.close()


def test_incomplete_thp_backing_fails_without_fd_leak(thp_owner, monkeypatch):
    from kvcr.hugepages import AlignedSharedMapping

    monkeypatch.setattr(AlignedSharedMapping, "_huge_bytes", lambda self: 0)
    before = len(list(Path("/proc/self/fd").iterdir()))
    with pytest.raises(RuntimeError, match="backing incomplete"):
        KVCRPoolAttachment.attach(thp_owner.spec)
    assert len(list(Path("/proc/self/fd").iterdir())) == before


def test_bad_thp_geometry_leaves_no_file(tmp_path):
    with pytest.raises(ValueError, match="multiple"):
        _KVCRPoolOwner.allocate(
            pool_id="bad",
            pool_size_bytes=12288,
            journal_bytes=8192,
            pool_dir=tmp_path,
            memory_mode="thp",
        )
    assert not list(tmp_path.iterdir())


def test_service_config_reaches_guard_attachment(thp_owner, tmp_path):
    from kvcr.config import KVCRMemoryConfig
    from kvcr.kvcr_service import _KVCRService

    service = _KVCRService(
        socket_path=tmp_path / "service.sock",
        pool_dir=Path(thp_owner.spec.path).parent,
        guard_count=1,
        pool_sizes_bytes=(thp_owner.spec.mapping_bytes - 8192,),
        compatibility_digest="test",
        journal_bytes=8192,
        memory_config=KVCRMemoryConfig(mode="thp"),
    )
    try:
        guard = service._registry._guards[0]
        assert guard._spec.memory_mode == "thp"
        assert (
            guard._recovery.attachment._require_mapping().huge_bytes
            == guard._spec.mapping_bytes
        )
    finally:
        service.close()


def test_thp_size_uses_kernel_value_for_grace(monkeypatch):
    from unittest.mock import Mock

    from kvcr import hugepages

    path = Mock()
    path.return_value.read_text.return_value = "536870912\n"
    monkeypatch.setattr(hugepages, "Path", path)
    assert hugepages.thp_page_size() == 512 << 20
