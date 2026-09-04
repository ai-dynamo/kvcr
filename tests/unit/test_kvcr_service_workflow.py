# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Whole-workflow tests for the standalone KVCR service daemon."""

import ctypes
import signal
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
_POOL_SIZE_BYTES = 8192
_CLI_POOL_SIZE_BYTES = 8192
_CLI_POOL_SIZE_GB = str(_CLI_POOL_SIZE_BYTES / (1 << 30))
_STOP_TIMEOUT_SECONDS = 5.0
_START_TIMEOUT_SECONDS = 60.0


_GUARD_BINDS: dict[int, tuple[str, int]] = {}


def _control_bind(guard_index: int) -> tuple[str, int]:
    """One address per Guard, stable across the claims that reuse it."""
    bind = _GUARD_BINDS.get(guard_index)
    if bind is None:
        bind = _GUARD_BINDS[guard_index] = ("127.0.0.1", free_port())
    return bind


@pytest.fixture(autouse=True)
def _fresh_guard_binds() -> Iterator[None]:
    """One address per Guard per test; the service holding it is gone by the next."""
    _GUARD_BINDS.clear()
    yield
    _GUARD_BINDS.clear()


def _socket_path() -> Path:
    """Return a fresh path under /tmp, short enough for AF_UNIX."""
    return Path("/tmp") / f"kvcr-{uuid.uuid4().hex}.sock"


def _claim_when_ready(
    process: subprocess.Popen[bytes], socket_path: Path, guard_index: int
) -> KVCRPoolHold:
    client = KVCRClient(socket_path)
    deadline = time.monotonic() + _START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            return client.claim(
                guard_index,
                [(_ROW_STRIDE, "")],
                _DIGEST,
                _control_bind(guard_index),
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
            "--guard-count",
            "2",
            "--pool-sizes-gb",
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
    guard_count: int = 2,
    socket_path: Path | None = None,
) -> Iterator[Path]:
    if socket_path is None:
        socket_path = _socket_path()
    server = _KVCRService(
        socket_path,
        pool_dir,
        guard_count=guard_count,
        pool_sizes_bytes=(_POOL_SIZE_BYTES,),
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


def test_pools_persist_bytes_and_a_held_pool_refuses_claims(tmp_path: Path) -> None:
    """Independent pools keep bytes across reclaim and refuse claims while held."""
    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()
    payload = b"kvcr-workflow"

    with _running_service(pool_dir) as socket_path:
        client = KVCRClient(socket_path)
        first = client.claim(0, [(_ROW_STRIDE, "")], _DIGEST, _control_bind(0))
        second = client.claim(1, [(_ROW_STRIDE, "")], _DIGEST, _control_bind(1))
        try:
            assert first.local_dram.address != second.local_dram.address
            ctypes.memmove(first.local_dram.address, payload, len(payload))

            first.release()
            replacement = client.claim(
                0, [(_ROW_STRIDE, "")], _DIGEST, _control_bind(0)
            )
            try:
                assert (
                    ctypes.string_at(replacement.local_dram.address, len(payload))
                    == payload
                )
            finally:
                replacement.release()

            # A KVCR worker takes pool 0 next, over the endpoint the pool
            # answers on: a pool's control address is bound once and never moves.
            host, port = _control_bind(0)
            pinning = FakePrimaryPinning()
            control = ZmqPeerControlChannel(host, port, host)
            with _use_nixl_agent(FakeNixlAgent()):
                controller = KVCR(
                    KVCRConfig(
                        nixl_agent_name="target",
                        pool_layout=[(_ROW_STRIDE, "")],
                        nixl_listen_port=1,
                    ),
                    KVCRBindings(
                        pinning.request_pin,
                        pinning.poll_pin_results,
                        pinning.release_pin,
                        framework_control=control,
                    ),
                    KVCRBackendConfigs(),
                    KVCRGuardConfig(
                        kvcr_service_socket_path=str(socket_path),
                        guard_index=0,
                        compatibility_digest=_DIGEST,
                    ),
                )
            try:
                with pytest.raises(KVCRServiceError, match="held"):
                    client.claim(0, [(_ROW_STRIDE, "")], _DIGEST, (host, port))
            finally:
                controller.close()
                control.close()

            # Closing the worker released the pool: the next claim is served.
            reclaimed = client.claim(0, [(_ROW_STRIDE, "")], _DIGEST, (host, port))
            reclaimed.release()
        finally:
            second.release()


def test_cli_daemon_sets_geometry_and_restart_reclaims_only_unattached(
    tmp_path: Path,
) -> None:
    """CLI flags set pool geometry; a restart reclaims only unattached pools."""
    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()
    with _running_daemon(pool_dir) as (crashed, socket_path):
        hold: KVCRPoolHold | None = None
        try:
            hold = _claim_when_ready(crashed, socket_path, 0)

            # The deployed flags produce the requested pool and data geometry.
            pools = list(pool_dir.iterdir())
            assert len(pools) == 2, "--guard-count pools at startup"
            pool_bytes = int(float(_CLI_POOL_SIZE_GB) * (1 << 30))
            assert all(
                path.stat().st_size == _DEFAULT_JOURNAL_BYTES + pool_bytes
                for path in pools
            )
            assert hold.local_dram.pool_sizes_bytes == [(pool_bytes, "")]

            attached_pool = next(pool_dir.glob("kvcr-pool_0-*"))
            unclaimed_pool = next(pool_dir.glob("kvcr-pool_1-*"))

            crashed.send_signal(signal.SIGKILL)
            crashed.wait(timeout=_STOP_TIMEOUT_SECONDS)
            assert attached_pool.exists()
            assert unclaimed_pool.exists()

            # The successor reclaims the unclaimed pool but leaves the attached one.
            with _running_service(pool_dir, guard_count=1, socket_path=socket_path):
                assert attached_pool.exists()
                assert not unclaimed_pool.exists()
                assert len(list(pool_dir.iterdir())) == 2

            try:
                hold.release()
            except KVCRSocketError:
                pass
            hold = None

            # Once the worker unmaps it, the next successor reclaims the orphan.
            with _running_service(pool_dir, guard_count=1, socket_path=socket_path):
                assert not attached_pool.exists()
                assert len(list(pool_dir.iterdir())) == 1
        finally:
            if hold is not None:
                try:
                    hold.release()
                except KVCRSocketError:
                    pass
