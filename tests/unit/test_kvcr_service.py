# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import select
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import msgspec
import pytest

from kvcr import KVCRClient, KVCRServiceError
from kvcr.control_channels import FramedConnection
from kvcr.guard_protocol import (
    _CLAIM_RESPONSE_DECODER,
    _RELEASE_RESPONSE_DECODER,
    PidfdLiveness,
    _Claim,
    _Error,
    _Granted,
    _Released,
)
from kvcr.kvcr_service import (
    _KVCRService,
    _PoolRegistry,
    _RequestHandler,
    _ThreadingUnixServer,
)
from kvcr.memory import _KVCRPoolOwner

_SERVER_STOP_TIMEOUT_SECONDS = 5
_CONNECTION_POLL_INTERVAL_SECONDS = 0.001

_TEST_POOL_COUNT = 2
_TEST_POOL_SIZE_BYTES = 8192
_TEST_ROW_STRIDE = 1024
_TEST_DIGEST = "opaque digest: Preserve-Me EXACTLY"


class _FakeLiveness:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@dataclass
class _ServerHarness:
    server: _KVCRService
    client: KVCRClient
    thread: threading.Thread
    stopped: bool = False

    def stop(self) -> None:
        if self.stopped:
            return
        self.server.shutdown()
        self.thread.join(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        self.server.close()
        assert not self.thread.is_alive()
        self.stopped = True


def _test_socket_path() -> Path:
    """Return a fresh socket path under /tmp, short enough for AF_UNIX."""
    return Path("/tmp") / f"kvcr-{uuid.uuid4().hex}.sock"


@contextmanager
def _running_server(
    tmp_path: Path,
    pool_count: int = _TEST_POOL_COUNT,
    pool_size_bytes: int = _TEST_POOL_SIZE_BYTES,
) -> Iterator[_ServerHarness]:
    socket_path = _test_socket_path()
    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()
    server = _KVCRService(
        socket_path,
        pool_dir,
        pool_count=pool_count,
        pool_size_bytes=pool_size_bytes,
        compatibility_digest=_TEST_DIGEST,
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    harness = _ServerHarness(server, KVCRClient(socket_path), thread)
    try:
        yield harness
    finally:
        harness.stop()


def _send_raw_request(
    socket_path: Path,
    request: object,
) -> _Granted | _Error:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(socket_path))
        channel = FramedConnection(connection)
        channel.send(request)
        return channel.receive(_CLAIM_RESPONSE_DECODER)


def _claim_request(
    compatibility_digest: str = _TEST_DIGEST,
) -> _Claim:
    return _Claim(0, _TEST_ROW_STRIDE, compatibility_digest)


def _wait_for_connection_state(
    server: _KVCRService,
    *,
    connected: bool,
) -> None:
    deadline = time.monotonic() + _SERVER_STOP_TIMEOUT_SECONDS
    while bool(server._server._connections) is not connected:
        if time.monotonic() >= deadline:
            pytest.fail(f"server connection state did not become {connected}")
        time.sleep(_CONNECTION_POLL_INTERVAL_SECONDS)


def test_socket_is_private(tmp_path: Path) -> None:
    """Only the service owner can access its Unix socket."""
    with _running_server(tmp_path) as harness:
        assert stat.S_IMODE(harness.server.socket_path.stat().st_mode) == 0o600


def test_release_keeps_service_geometry_sticky(tmp_path: Path) -> None:
    registry = _PoolRegistry(tmp_path, 2, _TEST_POOL_SIZE_BYTES)
    first = _FakeLiveness()
    wrong = _FakeLiveness()
    second = _FakeLiveness()
    try:
        first_spec = registry.claim(0, _TEST_ROW_STRIDE, first)
        registry.release(0, first)
        with pytest.raises(KVCRServiceError, match="geometry mismatch"):
            registry.claim(1, _TEST_ROW_STRIDE * 2, wrong)
        assert registry._owners[1].spec is None
        assert 1 not in registry._bindings

        second_spec = registry.claim(0, _TEST_ROW_STRIDE, second)
        registry.release(0, second)

        assert first_spec == second_spec
        assert first.closed is True
        assert wrong.closed is False
        assert second.closed is True
    finally:
        registry.close()


def test_a_dead_holder_is_replaced_before_its_watcher_cleans_up(
    tmp_path: Path,
) -> None:
    registry = _PoolRegistry(tmp_path, 1, _TEST_POOL_SIZE_BYTES)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    first = PidfdLiveness(os.pidfd_open(child.pid))
    replacement = _FakeLiveness()
    try:
        registry.claim(0, _TEST_ROW_STRIDE, first)
        child.terminate()
        child.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        poller = select.poll()
        poller.register(first.fileno(), select.POLLIN)
        assert poller.poll(0)

        registry.claim(0, _TEST_ROW_STRIDE, replacement)
        registry.holder_died(0, first)

        assert registry._bindings[0] is replacement
        assert replacement.closed is False
        with pytest.raises(ValueError, match="pidfd is closed"):
            first.fileno()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        registry.close()


@pytest.mark.parametrize("tag", ["claim.v2", True, 1])
def test_unknown_claim_tag_does_not_mutate_registry(
    tmp_path: Path, tag: object
) -> None:
    with _running_server(tmp_path) as harness:
        request = msgspec.to_builtins(_claim_request())
        request["type"] = tag
        response = _send_raw_request(
            harness.server.socket_path,
            request,
        )
        assert isinstance(response, _Error)
        assert harness.server._registry._owners[0].spec is None
        assert harness.server._registry._bindings == {}


def test_recognized_messages_ignore_unknown_fields(
    tmp_path: Path,
) -> None:
    with _running_server(tmp_path) as harness:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(harness.server.socket_path))
            channel = FramedConnection(connection)
            claim = msgspec.to_builtins(_claim_request())
            claim["future"] = {"ignored": True}
            channel.send(claim)
            response = channel.receive(_CLAIM_RESPONSE_DECODER)
            assert isinstance(response, _Granted)
            assert 0 in harness.server._registry._bindings

            channel.send({"type": "release", "future": True})
            assert channel.receive(_RELEASE_RESPONSE_DECODER) == _Released()
            assert harness.server._registry._bindings == {}


def test_claim_refusals_and_internal_failures_do_not_finalize_or_bind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _running_server(tmp_path) as harness:
        with pytest.raises(KVCRServiceError, match="compatibility digest"):
            harness.client.claim(0, _TEST_ROW_STRIDE, _TEST_DIGEST.swapcase())
        with pytest.raises(KVCRServiceError, match="one complete KV row"):
            harness.client.claim(0, _TEST_POOL_SIZE_BYTES + 1, _TEST_DIGEST)
        assert "Unexpected failure while handling KVCR claim" not in caplog.text

        internal_error = AttributeError("internal invariant broke")
        with monkeypatch.context() as patcher:
            patcher.setattr(
                harness.server._server,
                "dispatch",
                Mock(side_effect=internal_error),
            )
            with pytest.raises(KVCRServiceError, match="internal KVCR service error"):
                harness.client.claim(0, _TEST_ROW_STRIDE, _TEST_DIGEST)

        log_record = next(
            record
            for record in caplog.records
            if record.message == "Unexpected failure while handling KVCR claim"
        )
        assert log_record.exc_info is not None
        assert log_record.exc_info[1] is internal_error
        harness.client.claim(0, _TEST_ROW_STRIDE * 2, _TEST_DIGEST).release()


def test_live_holder_makes_a_second_claim_busy(tmp_path: Path) -> None:
    with _running_server(tmp_path) as harness:
        hold = harness.client.claim(0, _TEST_ROW_STRIDE, _TEST_DIGEST)
        try:
            with pytest.raises(KVCRServiceError, match="held"):
                harness.client.claim(0, _TEST_ROW_STRIDE, _TEST_DIGEST)
        finally:
            hold.release()


def test_held_connection_accepts_only_release(tmp_path: Path) -> None:
    with _running_server(tmp_path, pool_count=1) as harness:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(harness.server.socket_path))
            channel = FramedConnection(connection)
            channel.send(_claim_request())
            assert isinstance(channel.receive(_CLAIM_RESPONSE_DECODER), _Granted)

            channel.send(_claim_request())
            assert isinstance(channel.receive(_RELEASE_RESPONSE_DECODER), _Error)
            assert 0 in harness.server._registry._bindings

            channel.send({"type": "release", "future": True})
            response = channel.receive(_RELEASE_RESPONSE_DECODER)
            assert response == _Released()
            assert harness.server._registry._bindings == {}


def test_fork_and_exec_do_not_preserve_claimant_access(
    tmp_path: Path,
) -> None:
    with _running_server(tmp_path) as harness:
        exec_program = "import time; print('execed', flush=True); time.sleep(60)"
        program = "\n".join(
            (
                "import os",
                "import sys",
                "import time",
                "from kvcr.guard_protocol import KVCRClient",
                f"hold = KVCRClient({str(harness.server.socket_path)!r}).claim("
                f"0, {_TEST_ROW_STRIDE}, {_TEST_DIGEST!r})",
                "forked_pid = os.fork()",
                "if forked_pid == 0:",
                "    hold._connection.close()",
                "    time.sleep(60)",
                "    os._exit(0)",
                "print(forked_pid, flush=True)",
                f"os.execl(sys.executable, sys.executable, '-c', {exec_program!r})",
            )
        )
        child = subprocess.Popen(
            [sys.executable, "-c", program],
            stdout=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        forked_pid: int | None = None
        forked_pidfd: int | None = None
        forked_poller: select.poll | None = None
        try:
            assert child.stdout is not None
            forked_pid_text = child.stdout.readline()
            assert child.stdout.readline() == "execed\n"
            forked_pid = int(forked_pid_text)
            forked_pidfd = os.pidfd_open(forked_pid)
            forked_poller = select.poll()
            forked_poller.register(forked_pidfd, select.POLLIN)
            spec = harness.server._registry._owners[0].spec
            assert spec is not None
            assert spec.path not in Path(f"/proc/{forked_pid}/maps").read_text()
            with pytest.raises(KVCRServiceError, match="held"):
                harness.client.claim(0, _TEST_ROW_STRIDE, _TEST_DIGEST)

            child.terminate()
            child.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
            assert not forked_poller.poll(0)
            harness.client.claim(0, _TEST_ROW_STRIDE, _TEST_DIGEST).release()
            assert not forked_poller.poll(0)
        finally:
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
            if forked_pidfd is not None:
                assert forked_poller is not None
                try:
                    assert forked_poller.poll(int(_SERVER_STOP_TIMEOUT_SECONDS * 1000))
                finally:
                    os.close(forked_pidfd)
            if child.stdout is not None:
                child.stdout.close()


def test_startup_allocation_failure_rolls_back_before_listener(
    tmp_path: Path,
) -> None:
    owners = [Mock(spec=_KVCRPoolOwner), Mock(spec=_KVCRPoolOwner)]
    allocation_error = OSError("allocation failed")

    with (
        patch.object(
            _KVCRPoolOwner,
            "allocate",
            side_effect=[*owners, allocation_error],
        ),
        patch("kvcr.kvcr_service._ThreadingUnixServer") as listener,
        pytest.raises(OSError) as raised,
    ):
        _KVCRService(
            _test_socket_path(),
            tmp_path,
            pool_count=3,
            pool_size_bytes=_TEST_POOL_SIZE_BYTES,
            compatibility_digest=_TEST_DIGEST,
        )

    assert raised.value is allocation_error
    listener.assert_not_called()
    for owner in owners:
        owner.close.assert_called_once_with()


def test_shutdown_unlinks_eager_pools_and_socket(tmp_path: Path) -> None:
    with _running_server(tmp_path) as harness:
        pool_paths = list((tmp_path / "pools").glob("kvcr-pool_*-*"))
        socket_path = harness.server.socket_path
        assert len(pool_paths) == _TEST_POOL_COUNT
        harness.stop()
        assert all(not path.exists() for path in pool_paths)
        assert not socket_path.exists()


def test_claim_after_shutdown_is_rejected(tmp_path: Path) -> None:
    request = _claim_request()

    with _running_server(tmp_path) as harness:
        pool_dir = tmp_path / "pools"
        harness.stop()
        with pytest.raises(RuntimeError, match="registry is closed"):
            harness.server._server.dispatch(request, _FakeLiveness())
        assert not list(pool_dir.iterdir())


def test_shutdown_does_not_unlink_replaced_socket_path(tmp_path: Path) -> None:
    replacement = b"replacement"
    with _running_server(tmp_path) as harness:
        socket_path = harness.server.socket_path
        socket_path.unlink()
        socket_path.write_bytes(replacement)
        harness.stop()
        assert socket_path.read_bytes() == replacement
        socket_path.unlink()


def test_idle_client_does_not_block_shutdown_cleanup(tmp_path: Path) -> None:
    with _running_server(tmp_path) as harness:
        pool_path = next((tmp_path / "pools").glob("kvcr-pool_0-*"))
        socket_path = harness.server.socket_path
        _wait_for_connection_state(harness.server, connected=False)
        idle_connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        idle_connection.connect(str(socket_path))
        _wait_for_connection_state(harness.server, connected=True)

        harness.stop()
        idle_connection.close()

        assert not pool_path.exists()
        assert not socket_path.exists()


def test_shutdown_continues_after_connection_cleanup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connection_cleanup() -> None:
        raise OSError("connection cleanup failed")

    with _running_server(tmp_path) as harness:
        pool_path = next((tmp_path / "pools").glob("kvcr-pool_0-*"))
        socket_path = harness.server.socket_path
        harness.server.shutdown()
        harness.thread.join(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        with monkeypatch.context() as patcher:
            patcher.setattr(
                harness.server._server,
                "close_connections",
                fail_connection_cleanup,
            )
            with pytest.raises(OSError, match="connection cleanup failed"):
                harness.server.close()
        assert not pool_path.exists()
        assert not socket_path.exists()


def test_grant_send_failure_rolls_back_the_exact_binding(tmp_path: Path) -> None:
    registry = _PoolRegistry(tmp_path, 1, _TEST_POOL_SIZE_BYTES)
    accepted, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

    class _Channel:
        def receive(self, _decoder: object) -> _Claim:
            return _claim_request()

        def send(self, response: object) -> None:
            assert isinstance(response, _Granted)
            raise RuntimeError("grant could not be delivered")

    class _Server:
        def __init__(self) -> None:
            self.registry = registry
            self.compatibility_digest = _TEST_DIGEST

        def dispatch(self, request, liveness):
            return _ThreadingUnixServer.dispatch(self, request, liveness)

    handler = object.__new__(_RequestHandler)
    handler.request = accepted
    handler.channel = _Channel()
    handler.server = _Server()
    try:
        handler.handle()
        assert registry._bindings == {}
        replacement = _FakeLiveness()
        registry.claim(0, _TEST_ROW_STRIDE, replacement)
        registry.release(0, replacement)
    finally:
        peer.close()
        accepted.close()
        registry.close()


@pytest.mark.parametrize("failure", ["setup", "poll", "pidfd"])
def test_ambiguous_hold_failure_stops_service_and_retains_binding(
    tmp_path: Path,
    failure: str,
) -> None:
    registry = _PoolRegistry(tmp_path, 1, _TEST_POOL_SIZE_BYTES)
    liveness = Mock()
    liveness.fileno.return_value = 11
    registry.claim(0, _TEST_ROW_STRIDE, liveness)

    server = object.__new__(_ThreadingUnixServer)
    server.registry = registry
    server._fatal_error = None
    server.shutdown = Mock()
    handler = object.__new__(_RequestHandler)
    handler.request = Mock()
    handler.request.fileno.return_value = 10
    handler.server = server
    poller = Mock()
    if failure == "setup":
        poller.register.side_effect = OSError("setup failed")
    elif failure == "poll":
        poller.poll.side_effect = OSError("poll failed")
    else:
        poller.poll.return_value = [(11, select.POLLERR | select.POLLNVAL)]

    try:
        with patch("kvcr.kvcr_service.select.poll", return_value=poller):
            handler._hold(0, liveness)

        fatal_error = server._fatal_error
        assert isinstance(fatal_error, OSError)
        assert failure in str(fatal_error)
        server.shutdown.assert_called_once_with()
        assert registry._bindings[0] is liveness
        liveness.close.assert_not_called()
        with pytest.raises(RuntimeError, match="held"):
            registry.claim(0, _TEST_ROW_STRIDE, _FakeLiveness())

        service = object.__new__(_KVCRService)
        service._server = server
        server.serve_forever = Mock()
        with pytest.raises(OSError) as raised:
            service.serve_forever()
        assert raised.value is fatal_error
    finally:
        registry.close()


def test_shutdown_drains_a_held_connection(tmp_path: Path) -> None:
    with _running_server(tmp_path) as harness:
        hold = harness.client.claim(0, _TEST_ROW_STRIDE, _TEST_DIGEST)
        harness.stop()
        _wait_for_connection_state(harness.server, connected=False)
        hold._attachment.close()
        hold._connection.close()
