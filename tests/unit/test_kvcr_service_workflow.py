# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Whole-workflow tests for the standalone KVCR service daemon."""

import ctypes
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from _kvcr_test_utils import (
    FakeNixlAgent,
    FakePrimaryPinning,
    _use_nixl_agent,
    free_port,
)

from kvcr import (
    KVCR,
    KVCRBindings,
    KVCRClient,
    KVCRPoolHold,
    KVCRServiceError,
    KVCRSocketError,
)
from kvcr.config import KVCRBackendConfigs, KVCRConfig, KVCRGuardConfig
from kvcr.control_channels import ZmqPeerControlChannel
from kvcr.kvcr_service import _DEFAULT_JOURNAL_BYTES, _KVCRService

_ROW_STRIDE = 1024
_DIGEST = "opaque workflow digest: Preserve-Me EXACTLY"
_JOURNAL_BYTES = 8192
_POOL_SIZE_BYTES = _JOURNAL_BYTES + 8192
_CLI_POOL_SIZE_BYTES = _DEFAULT_JOURNAL_BYTES + 8192
_CLI_POOL_SIZE_GB = str(_CLI_POOL_SIZE_BYTES / (1 << 30))
_STOP_TIMEOUT_SECONDS = 5.0
_START_TIMEOUT_SECONDS = 60.0


_POOL_BINDS: dict[int, tuple[str, int]] = {}


def _control_bind(pool_index: int) -> tuple[str, int]:
    """One address per pool, stable across the claims that reuse it."""
    bind = _POOL_BINDS.get(pool_index)
    if bind is None:
        bind = _POOL_BINDS[pool_index] = ("127.0.0.1", free_port())
    return bind


@pytest.fixture(autouse=True)
def _fresh_pool_binds() -> Iterator[None]:
    """One address per pool per test; the service holding it is gone by the next."""
    _POOL_BINDS.clear()
    yield
    _POOL_BINDS.clear()


def _socket_path() -> Path:
    """Return a fresh path under /tmp, short enough for AF_UNIX."""
    return Path("/tmp") / f"kvcr-{uuid.uuid4().hex}.sock"


def _claim_when_ready(
    process: subprocess.Popen[bytes], socket_path: Path, pool_index: int
) -> KVCRPoolHold:
    client = KVCRClient(socket_path)
    deadline = time.monotonic() + _START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            return client.claim(
                pool_index, _ROW_STRIDE, _DIGEST, _control_bind(pool_index)
            )
        except KVCRSocketError:
            if process.poll() is not None:
                output = process.stdout.read().decode() if process.stdout else ""
                raise AssertionError(f"daemon exited before accepting claims: {output}")
            time.sleep(0.05)
    raise AssertionError("daemon never accepted claims")


@contextmanager
def _running_daemon(pool_dir: Path) -> Iterator[tuple[subprocess.Popen[bytes], Path]]:
    socket_path = _socket_path()
    daemon = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "kvcr.kvcr_service",
            "--socket-path",
            str(socket_path),
            "--pool-dir",
            str(pool_dir),
            "--pool-count",
            "2",
            "--pool-size-gb",
            _CLI_POOL_SIZE_GB,
            "--compatibility-digest",
            _DIGEST,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        yield daemon, socket_path
    finally:
        if daemon.poll() is None:
            daemon.send_signal(signal.SIGTERM)
        daemon.wait(timeout=_STOP_TIMEOUT_SECONDS)
        if daemon.stdout is not None:
            daemon.stdout.close()
        socket_path.unlink(missing_ok=True)


@contextmanager
def _running_service(
    pool_dir: Path,
    pool_count: int = 2,
    socket_path: Path | None = None,
) -> Iterator[Path]:
    if socket_path is None:
        socket_path = _socket_path()
    server = _KVCRService(
        socket_path,
        pool_dir,
        pool_count=pool_count,
        pool_size_bytes=_POOL_SIZE_BYTES,
        compatibility_digest=_DIGEST,
        journal_bytes=_JOURNAL_BYTES,
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield socket_path
    finally:
        server.shutdown()
        thread.join(timeout=_STOP_TIMEOUT_SECONDS)
        server.close()
        assert not thread.is_alive()


def test_multi_pool_claim_release_and_persistent_reclaim(tmp_path: Path) -> None:
    """Two pools map independently and a released pool retains its bytes."""
    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()
    payload = b"kvcr-workflow"

    with _running_service(pool_dir) as socket_path:
        client = KVCRClient(socket_path)
        first = client.claim(0, _ROW_STRIDE, _DIGEST, _control_bind(0))
        second = client.claim(1, _ROW_STRIDE, _DIGEST, _control_bind(1))
        try:
            assert first.local_dram.address != second.local_dram.address
            ctypes.memmove(first.local_dram.address, payload, len(payload))

            first.release()
            replacement = client.claim(0, _ROW_STRIDE, _DIGEST, _control_bind(0))
            try:
                assert (
                    ctypes.string_at(replacement.local_dram.address, len(payload))
                    == payload
                )
            finally:
                replacement.release()
        finally:
            second.release()


def test_kvcr_uses_releases_and_reclaims_service_pool(tmp_path: Path) -> None:
    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()

    with _running_service(pool_dir, pool_count=1) as socket_path:
        pinning = FakePrimaryPinning()
        # The service binds this port later, so another test can take it first.
        for attempt in range(5):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = int(probe.getsockname()[1])
            control = ZmqPeerControlChannel("127.0.0.1", port, "127.0.0.1")
            try:
                with _use_nixl_agent(FakeNixlAgent()):
                    controller = KVCR(
                        KVCRConfig(nixl_agent_name="target", nixl_listen_port=1),
                        KVCRBindings(
                            pinning.request_pin,
                            pinning.poll_pin_results,
                            pinning.release_pin,
                            framework_control=control,
                        ),
                        KVCRBackendConfigs(),
                        KVCRGuardConfig(
                            kvcr_service_socket_path=str(socket_path),
                            pool_index=0,
                            row_stride=_ROW_STRIDE,
                            compatibility_digest=_DIGEST,
                        ),
                    )
                break
            except KVCRServiceError as error:
                control.close()
                if "unavailable" not in str(error) or attempt == 4:
                    raise
        try:
            with pytest.raises(KVCRServiceError, match="held"):
                KVCRClient(socket_path).claim(
                    0, _ROW_STRIDE, _DIGEST, ("127.0.0.1", port)
                )
        finally:
            controller.close()
            control.close()

        replacement = KVCRClient(socket_path).claim(
            0, _ROW_STRIDE, _DIGEST, ("127.0.0.1", port)
        )
        replacement.release()


def test_restart_reclaims_only_unattached_pools(tmp_path: Path) -> None:
    """A restart reclaims an eager pool but preserves an attached one."""
    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()
    with _running_daemon(pool_dir) as (crashed, socket_path):
        hold: KVCRPoolHold | None = None
        try:
            hold = _claim_when_ready(crashed, socket_path, 0)
            attached_pool = next(pool_dir.glob("kvcr-pool_0-*"))
            unclaimed_pool = next(pool_dir.glob("kvcr-pool_1-*"))

            crashed.send_signal(signal.SIGKILL)
            crashed.wait(timeout=_STOP_TIMEOUT_SECONDS)
            assert attached_pool.exists()
            assert unclaimed_pool.exists()

            # The successor reclaims the unclaimed pool but leaves the attached one.
            with _running_service(pool_dir, pool_count=1, socket_path=socket_path):
                assert attached_pool.exists()
                assert not unclaimed_pool.exists()
                assert len(list(pool_dir.iterdir())) == 2

            try:
                hold.release()
            except KVCRSocketError:
                pass
            hold = None

            # Once the worker unmaps it, the next successor reclaims the orphan.
            with _running_service(pool_dir, pool_count=1, socket_path=socket_path):
                assert not attached_pool.exists()
                assert len(list(pool_dir.iterdir())) == 1
        finally:
            if hold is not None:
                try:
                    hold.release()
                except KVCRSocketError:
                    pass


def test_cli_configures_data_geometry(tmp_path: Path) -> None:
    """The deployed flags produce the requested pool and data geometry."""
    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()
    with _running_daemon(pool_dir) as (daemon, socket_path):
        hold: KVCRPoolHold | None = None
        try:
            hold = _claim_when_ready(daemon, socket_path, 1)
            pools = list(pool_dir.iterdir())
            assert len(pools) == 2, "--pool-count pools at startup"
            requested = int(float(_CLI_POOL_SIZE_GB) * (1 << 30))
            client_rows = (requested - _DEFAULT_JOURNAL_BYTES) // _ROW_STRIDE
            data_bytes = client_rows * _ROW_STRIDE
            assert all(path.stat().st_size == requested for path in pools)
            assert hold.local_dram.length == data_bytes
            assert hold.local_dram.slot_count == client_rows
            hold.release()
            hold = None
        finally:
            if hold is not None:
                hold.release()
