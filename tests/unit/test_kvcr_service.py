# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import errno
import fcntl
import logging
import mmap
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
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import pytest
from _kvcr_test_utils import (
    _wait_until,
    free_port,
    listening_socket,
    manual_pool_spec,
)

from kvcr import KVCRClient, KVCRServiceError, guard_protocol
from kvcr.config import G3Options
from kvcr.control_channels import FramedConnection
from kvcr.guard import _Command, _Phase
from kvcr.guard_protocol import (
    _CLAIM_RESPONSE_DECODER,
    _RELEASE_RESPONSE_DECODER,
    PidfdLiveness,
    _Claim,
    _Error,
    _G3Config,
    _Granted,
    _PoolDescriptor,
    _Release,
    _Released,
    _TierConfig,
)
from kvcr.kvcr_service import (
    _KVCRService,
    _open_pool_fd,
    _parse_args,
    _PoolRegistry,
    _RequestHandler,
    _ThreadingUnixServer,
)
from kvcr.memory import KVCRPoolAttachment, _KVCRPoolOwner


def _holders_of(registry) -> dict[int, object]:
    """The Guards a worker holds, in the shape the old binding map had."""
    return {
        i: p._pool_lease.current
        for i, p in registry._guards.items()
        if p._pool_lease.current is not None
    }


_SERVER_STOP_TIMEOUT_SECONDS = 5
_CONNECTION_POLL_INTERVAL_SECONDS = 0.001
_PAGE_BLOCK_SIZE_BYTES = os.sysconf("SC_PAGE_SIZE")

_TEST_GUARD_COUNT = 2
_TEST_JOURNAL_BYTES = 2 * _PAGE_BLOCK_SIZE_BYTES
_TEST_POOL_SIZES_BYTES = (2 * _PAGE_BLOCK_SIZE_BYTES,)
_TEST_BLOCK_SIZE_BYTES = 1024
_TEST_POOL_LAYOUTS = [("pool0", _TEST_BLOCK_SIZE_BYTES)]
_TEST_DIGEST = "opaque digest: Preserve-Me EXACTLY"
_TEST_TIER_CONFIG = _TierConfig(_TEST_POOL_LAYOUTS, None)


class _FakeLiveness:
    """A pollable pidfd stand-in: a pipe that reads POLLIN once killed."""

    def __init__(self) -> None:
        self._read, self._write = os.pipe()
        self.closed = False
        self.incarnation = None
        self.mode = "so_peerpidfd"
        self.verified = True
        self.fenceable = True
        self.peer_uid = None

    def fileno(self) -> int:
        if self.closed:
            raise ValueError("pidfd is closed")
        return self._read

    def kill(self) -> None:
        os.write(self._write, b"x")

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            os.close(self._read)
            os.close(self._write)


def _claim(registry, guard_index, liveness, control_bind=None):
    """Claim through the registry, closing the granted fd the tests never send."""
    spec, _pools, listener_fd, lease = registry.claim(
        guard_index,
        _TEST_TIER_CONFIG,
        liveness,
        control_bind or _control_bind(guard_index),
    )
    os.close(listener_fd)
    return spec, lease


def _kill_and_wait(registry, guard_index, liveness) -> None:
    """Die the way a real claimant does: the Guard's own actor notices."""
    liveness.kill()
    guard = registry._guards[guard_index]
    _wait_until(
        lambda: guard._phase in (_Phase.STANDBY, _Phase.FAILED) or guard._failure,
        timeout=5,
    )


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
    guard_count: int = _TEST_GUARD_COUNT,
    pool_sizes_bytes: tuple[int, ...] = _TEST_POOL_SIZES_BYTES,
    journal_bytes: int = _TEST_JOURNAL_BYTES,
    heartbeat_timeout_seconds: float | None = None,
    peer_trust: str = "strict",
) -> Iterator[_ServerHarness]:
    socket_path = _test_socket_path()
    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()
    server = _KVCRService(
        socket_path,
        pool_dir,
        guard_count=guard_count,
        pool_sizes_bytes=pool_sizes_bytes,
        compatibility_digest=_TEST_DIGEST,
        journal_bytes=journal_bytes,
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
        peer_trust=peer_trust,
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


_GUARD_BINDS: dict[int, tuple[str, int]] = {}


def _control_bind(guard_index: int = 0) -> tuple[str, int]:
    """The address a Guard answers on, stable for as long as it exists."""
    if guard_index not in _GUARD_BINDS:
        _GUARD_BINDS[guard_index] = ("127.0.0.1", free_port())
    return _GUARD_BINDS[guard_index]


# A free address a client can ask the service to bind.
def _claim_request(
    compatibility_digest: str = _TEST_DIGEST,
    control_bind: tuple[str, int] | None = None,
) -> _Claim:
    host, port = control_bind or _control_bind(0)
    return _Claim(0, compatibility_digest, _TEST_TIER_CONFIG, host, port, 1)


@pytest.fixture(autouse=True)
def _channels_are_taken():
    """Close the duplicate a claim hands its Guard, as a real Guard would."""
    # Within a test a Guard's endpoint never moves; across tests it is gone.
    _GUARD_BINDS.clear()
    # A promoted Guard's poll loop drains this, so recv() must be iterable.
    channel = Mock(recv=Mock(return_value=[]))

    def _take(duplicate: socket.socket):
        duplicate.close()
        return channel

    with patch(
        "kvcr.guard.ZmqPeerControlChannel.from_shared_listener",
        side_effect=_take,
    ):
        yield
    _GUARD_BINDS.clear()


def _stand_in_pool(spec) -> Mock:
    """The pool-tail surface a Guard reaches for, without a real mapping."""
    attachment = Mock(address=1234, _spec=spec)
    attachment.mapped_snapshot.return_value = nullcontext(None)
    return attachment


@contextmanager
def _guard_patches(real_attach: bool = False) -> Iterator[None]:
    """Stand-in pool mappings, or real ones with a stand-in journal."""
    journal = Mock()
    journal.read_next.return_value = None
    with (
        patch(
            "kvcr.guard.KVCRPoolAttachment.attach",
            side_effect=KVCRPoolAttachment.attach if real_attach else _stand_in_pool,
        ),
        patch("kvcr.guard.RecoveryJournal", Mock(return_value=journal)),
    ):
        yield


@pytest.fixture
def _stand_in_guards() -> Iterator[None]:
    """Kept for the whole test: a held Guard attaches long after construction."""
    with _guard_patches():
        yield


@pytest.fixture
def _real_attach_guards(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Absent on macOS; real attachments only need the call to succeed.
    monkeypatch.setattr(mmap, "MADV_DONTFORK", mmap.MADV_NORMAL, raising=False)
    with _guard_patches(real_attach=True):
        yield


def _new_registry(
    tmp_path: Path,
    guard_count: int = 1,
    digest: str = _TEST_DIGEST,
    real_attach: bool = False,
    **options,
) -> _PoolRegistry:
    """A registry of real Guards over stand-in pool mappings (or real ones)."""
    with _guard_patches(real_attach):
        registry = _PoolRegistry(
            tmp_path,
            guard_count,
            _TEST_POOL_SIZES_BYTES,
            _TEST_JOURNAL_BYTES,
            digest,
            **options,
        )
    for guard in registry._guards.values():
        # Stand-in addresses cannot be registered with the native backend.
        guard._prepare_core = Mock()
    return registry


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


def test_client_claims_with_worker_pidfd_when_peer_option_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unknown option makes the host behave like Linux 5.15 for this claim.
    monkeypatch.setattr(guard_protocol, "_SO_PEERPIDFD", 0x7FFFFFFF)
    with _running_server(tmp_path) as harness:
        hold = harness.client.claim(
            0, _TEST_POOL_LAYOUTS, _TEST_DIGEST, _control_bind(0)
        )
        assert harness.server._registry._guards[0]._pool_lease.current is not None
        hold.release()


def test_client_claims_without_own_pidfd_when_server_has_peer_pidfd(
    tmp_path: Path,
) -> None:
    probe, peer = socket.socketpair()
    try:
        try:
            os.close(probe.getsockopt(socket.SOL_SOCKET, guard_protocol._SO_PEERPIDFD))
        except OSError:
            pytest.skip("SO_PEERPIDFD is unavailable")
    finally:
        probe.close()
        peer.close()
    with _running_server(tmp_path) as harness:
        with patch(
            "kvcr.guard_protocol.os.pidfd_open",
            side_effect=OSError(errno.EPERM, "blocked"),
        ):
            hold = harness.client.claim(
                0, _TEST_POOL_LAYOUTS, _TEST_DIGEST, _control_bind(0)
            )
        hold.release()


def test_legacy_service_refuses_client_without_own_pidfd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard_protocol, "_SO_PEERPIDFD", 0x7FFFFFFF)
    with _running_server(tmp_path) as harness:
        with patch(
            "kvcr.guard_protocol.os.pidfd_open",
            side_effect=OSError(errno.EPERM, "blocked"),
        ):
            with pytest.raises(KVCRServiceError, match="worker pidfd is required"):
                harness.client.claim(
                    0, _TEST_POOL_LAYOUTS, _TEST_DIGEST, _control_bind(0)
                )


@pytest.mark.parametrize("peer_trust", ["strict", "trusted-peer"])
def test_legacy_claim_rejects_wrong_process_or_non_pidfd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, peer_trust: str
) -> None:
    monkeypatch.setattr(guard_protocol, "_SO_PEERPIDFD", 0x7FFFFFFF)
    with _running_server(tmp_path, guard_count=1, peer_trust=peer_trust) as harness:
        open_fds = len(os.listdir("/proc/self/fd"))
        for descriptor_type in ("wrong-pid", "not-pidfd"):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(harness.server.socket_path))
                channel = FramedConnection(connection)
                descriptor = (
                    os.pidfd_open(os.getppid())
                    if descriptor_type == "wrong-pid"
                    else os.open(os.devnull, os.O_RDONLY)
                )
                try:
                    channel.send_with_fd(_claim_request(), descriptor)
                finally:
                    os.close(descriptor)
                response = channel.receive(_CLAIM_RESPONSE_DECODER)
                assert isinstance(response, _Error)
                assert "does not match socket peer" in response.message
                assert _holders_of(harness.server._registry) == {}
        _wait_until(lambda: len(os.listdir("/proc/self/fd")) <= open_fds, timeout=2)


def test_trusted_peer_grants_a_peer_in_another_pid_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real thing needs a second PID namespace; see the unshare tests below.
    monkeypatch.setattr(guard_protocol, "_SO_PEERPIDFD", 0x7FFFFFFF)
    monkeypatch.setattr(
        guard_protocol,
        "_peer_pidfd_verdict",
        lambda _connection, _fd: (guard_protocol._PeerVerdict.PEER_NOT_VISIBLE, 0),
    )
    with _running_server(tmp_path, guard_count=1, peer_trust="trusted-peer") as harness:
        hold = harness.client.claim(
            0, _TEST_POOL_LAYOUTS, _TEST_DIGEST, _control_bind(0)
        )
        lease = harness.server._registry._guards[0]._pool_lease.current
        assert (lease.mode, lease.verified, lease.fenceable) == (
            "pidfd_trusted",
            False,
            False,
        )
        hold.release()


def test_service_refuses_startup_without_pidfd_open(tmp_path: Path) -> None:
    socket_path = _test_socket_path()
    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()
    with patch(
        "kvcr.kvcr_service.os.pidfd_open",
        side_effect=OSError(errno.ENOSYS, "not implemented"),
    ):
        with pytest.raises(KVCRServiceError, match="pidfd_open"):
            _KVCRService(
                socket_path,
                pool_dir,
                guard_count=1,
                pool_sizes_bytes=_TEST_POOL_SIZES_BYTES,
                compatibility_digest=_TEST_DIGEST,
            )
    assert not socket_path.exists()


def test_held_connection_refreshes_heartbeats_until_release(tmp_path: Path) -> None:
    timeout = 1.0
    with _running_server(tmp_path, heartbeat_timeout_seconds=timeout) as harness:
        guard = harness.server._registry._guards[0]
        guard._promote = Mock()
        # The claimant here is the pytest process: a scheduling pause past the
        # timeout would otherwise SIGKILL the test run through the lease pidfd.
        guard._fence_missing_primary = Mock(return_value=True)
        hold = harness.client.claim(
            0, _TEST_POOL_LAYOUTS, _TEST_DIGEST, _control_bind(0)
        )
        lease = guard._pool_lease.current
        try:
            _wait_until(lambda: guard._last_heartbeat is not None, timeout=2)
            first_heartbeat = guard._last_heartbeat
            _wait_until(
                lambda: (
                    guard._last_heartbeat is not None
                    and guard._last_heartbeat >= first_heartbeat + 2 * timeout
                ),
                timeout=5,
            )
            guard._fence_missing_primary.assert_not_called()
            guard._promote.assert_not_called()
            assert guard._phase is _Phase.PRIMARY
            assert guard._pool_lease.current is lease
        finally:
            hold.release()
        assert guard._phase is _Phase.IDLE
        assert guard._pool_lease.current is None


def test_client_claims_one_grouped_allocation_with_independent_strides(
    tmp_path: Path,
) -> None:
    pool_sizes = (2 * _PAGE_BLOCK_SIZE_BYTES, 3 * _PAGE_BLOCK_SIZE_BYTES)
    pool_layouts = [
        ("pool0", _TEST_BLOCK_SIZE_BYTES),
        ("pool1", 3 * _TEST_BLOCK_SIZE_BYTES),
    ]
    with _running_server(
        tmp_path,
        guard_count=2,
        pool_sizes_bytes=pool_sizes,
    ) as harness:
        with pytest.raises(KVCRServiceError, match="out of range"):
            harness.client.claim(2, _TEST_POOL_LAYOUTS, _TEST_DIGEST, _control_bind(2))
        with pytest.raises(KVCRServiceError, match="pool layout"):
            harness.client.claim(1, _TEST_POOL_LAYOUTS, _TEST_DIGEST, _control_bind(1))

        hold = harness.client.claim(1, pool_layouts, _TEST_DIGEST, _control_bind(1))
        hold.release()
        guard = harness.server._registry._guards[1]
        spec = guard._owner.spec
        assert spec.mapping_bytes == _TEST_JOURNAL_BYTES + sum(pool_sizes)
        offsets = (_TEST_JOURNAL_BYTES, _TEST_JOURNAL_BYTES + pool_sizes[0])
        assert guard._recovery.pools == tuple(
            _PoolDescriptor(name, size, block_size, offset)
            for (name, block_size), size, offset in zip(
                pool_layouts, pool_sizes, offsets, strict=True
            )
        )


def test_registry_lifecycle_from_independent_leases_to_a_wedged_close(
    tmp_path: Path, caplog
) -> None:
    """Pools lease independently; close keeps, names, and can retry a wedged one."""
    caplog.set_level(logging.DEBUG, logger="kvcr.guard")
    registry = _new_registry(tmp_path, guard_count=2)
    guard = registry._guards[0]
    first, second, third = _FakeLiveness(), _FakeLiveness(), _FakeLiveness()
    first_spec, stale = _claim(registry, 0, first)
    assert any("KVCR_EVENT primary_attached " in message for message in caplog.messages)
    _spec, _pools, _fd, _lease = registry.claim(
        1,
        _TierConfig([("pool0", _TEST_BLOCK_SIZE_BYTES * 2)], None),
        second,
        _control_bind(1),
    )
    os.close(_fd)
    # One pool's geometry says nothing about another's.
    assert _holders_of(registry) == {0: first, 1: second}

    registry.release(0, stale)
    assert first.closed is True
    second_spec, current = _claim(registry, 0, third)
    assert second_spec is first_spec

    # The retried release of the old lease must not end the new one.
    registry.release(0, stale)
    assert registry._guards[0]._pool_lease.current is current
    assert third.closed is False

    first_path = Path(guard._owner.spec.path)
    second_path = Path(registry._guards[1]._owner.spec.path)
    # An unclosable mapping keeps its file: unlinking would hide committed RAM.
    guard._recovery.attachment.close.side_effect = RuntimeError("mapping held")
    with pytest.raises(RuntimeError, match="mapping held"):
        registry.close()
    assert registry._guards.keys() == {0}
    assert registry._guards[0]._recovery.attachment is not None
    assert first_path.exists()
    # The pool behind it still went, files and all.
    assert second.closed is True and not second_path.exists()

    # The kept pool must still name what it leaked, or nobody could retry it.
    guard._recovery.attachment.close.side_effect = None
    stubborn = Mock(close=Mock(side_effect=OSError("will not close")))
    guard._pool_lease.listener = stubborn
    with pytest.raises(OSError, match="will not close"):
        registry.close()
    assert registry._guards.keys() == {0}
    assert registry._guards[0]._pool_lease.listener is stubborn
    assert not first_path.exists()

    # With nothing left refusing, the retried close finally takes the pool.
    guard._pool_lease.listener = None
    registry.close()
    assert registry._guards == {} and third.closed is True

    registry = _new_registry(tmp_path)
    guard = registry._guards[0]
    owner = guard._owner
    with patch.object(owner, "close", side_effect=OSError("reclaim failed")):
        with pytest.raises(OSError, match="reclaim failed"):
            registry.close()
        assert guard._owner is owner and Path(owner.spec.path).exists()
    registry.close()
    assert registry._guards == {}
    assert not Path(owner.spec.path).exists()


@pytest.mark.parametrize("cleanup_failure", [None, "mapping", "lease"])
def test_shutdown_joins_all_guards_before_reclaiming(tmp_path, cleanup_failure) -> None:
    registry = _new_registry(tmp_path, guard_count=2)
    joined = []
    paths = [Path(guard._owner.spec.path) for guard in registry._guards.values()]
    for index, guard in registry._guards.items():
        finish = guard.finish_close
        owner_close = guard._owner.close

        def finish_close(deadline, index=index, finish=finish):
            try:
                return finish(deadline)
            finally:
                joined.append(index)

        def reclaim(owner_close=owner_close):
            assert joined == [0, 1]
            owner_close()

        guard.finish_close = finish_close
        guard._owner.close = reclaim
        if cleanup_failure == "mapping":
            guard._recovery.attachment.close.side_effect = RuntimeError(
                "cleanup failed"
            )
        elif cleanup_failure == "lease":
            guard._pool_lease.listener = Mock(
                close=Mock(side_effect=RuntimeError("cleanup failed"))
            )
    if cleanup_failure:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            registry.close()
        assert registry._guards.keys() == {0, 1}
        assert all(path.exists() == (cleanup_failure == "mapping") for path in paths)
        for guard in registry._guards.values():
            assert not guard._thread.is_alive()
            if cleanup_failure == "mapping":
                guard._recovery.attachment.close.side_effect = None
            else:
                guard._pool_lease.listener.close.side_effect = None
        joined.clear()
    registry.close()
    assert registry._guards == {}
    assert all(not path.exists() for path in paths)


def test_refused_claims_do_not_bind_the_pool(tmp_path: Path) -> None:
    """A hostname, bad G3 terms, or a taken endpoint refuse without binding."""
    # A name is refused where the claim is built, before it is sent.
    with pytest.raises(ValueError, match="literal IPv4 address"):
        _claim_request(control_bind=("localhost", free_port()))

    # Held for the whole test: the address must still be taken when the claim runs.
    with listening_socket() as squatter, _running_server(tmp_path) as harness:
        request = msgspec.to_builtins(_claim_request())
        request["tier_config"]["g3"] = {
            "paths": ["relative"],
            "capacity_bytes_per_file": 8192,
            "backend": "FILE",
            "backend_options": {},
        }
        assert isinstance(
            _send_raw_request(harness.server.socket_path, request), _Error
        )

        taken = (str(squatter.getsockname()[0]), int(squatter.getsockname()[1]))
        with pytest.raises(KVCRServiceError, match="control listener"):
            harness.client.claim(
                0, _TEST_POOL_LAYOUTS, _TEST_DIGEST, control_bind=taken
            )

        # The rejected claims left the pool free to take normally.
        harness.client.claim(
            0, _TEST_POOL_LAYOUTS, _TEST_DIGEST, _control_bind()
        ).release()


def test_recognized_messages_ignore_unknown_fields(
    tmp_path: Path,
) -> None:
    with _running_server(tmp_path) as harness:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(harness.server.socket_path))
            channel = FramedConnection(connection)
            claim = msgspec.to_builtins(_claim_request())
            claim["future"] = {"ignored": True}
            worker_pidfd = os.pidfd_open(os.getpid())
            try:
                channel.send_with_fd(claim, worker_pidfd)
            finally:
                os.close(worker_pidfd)
            response = channel.receive(_CLAIM_RESPONSE_DECODER)
            assert isinstance(response, _Granted)
            assert 0 in _holders_of(harness.server._registry)

            channel.send({"type": "release", "version": 1, "future": True})
            assert channel.receive(_RELEASE_RESPONSE_DECODER) == _Released(1)
            assert _holders_of(harness.server._registry) == {}


def test_a_grant_tells_the_pools_guard_and_a_clean_release_stands_it_down(
    tmp_path: Path,
) -> None:
    """The Guard is the pool's, so a grant configures it and a release parks it."""
    control = Mock()
    taken: list[tuple[int, object]] = []

    def _take_and_record(duplicate: socket.socket):
        # Recorded before it is closed, as the real one closes it too.
        taken.append((duplicate.fileno(), duplicate.getsockname()))
        duplicate.close()
        return control

    g3 = G3Options(
        paths=((tmp_path / "g3").resolve(),),
        capacity_bytes_per_file=2 * _PAGE_BLOCK_SIZE_BYTES,
        backend="FILE",
        backend_options={"mode": "direct"},
    )
    with (
        patch(
            "kvcr.guard.ZmqPeerControlChannel.from_shared_listener",
            side_effect=_take_and_record,
        ),
        _running_server(tmp_path) as harness,
    ):
        guard = harness.server._registry._guards[1]
        # Built and started with the pool, before any claim named it.
        assert guard._phase is _Phase.UNCONFIGURED

        hold = harness.client.claim(
            1, [("pool0", _PAGE_BLOCK_SIZE_BYTES)], _TEST_DIGEST, _control_bind(), g3
        )
        assert guard._phase is _Phase.PRIMARY and guard._control is control
        assert guard._configured == _TierConfig(
            [("pool0", _PAGE_BLOCK_SIZE_BYTES)],
            _G3Config(
                paths=(str(g3.paths[0]),),
                capacity_bytes_per_file=g3.capacity_bytes_per_file,
                backend=g3.backend,
                backend_options=dict(g3.backend_options),
            ),
        )
        taken_fd, taken_name = taken[0]
        # The Guard is handed a duplicate; the pool keeps the original.
        assert taken_fd != guard._pool_lease.listener.fileno()
        assert taken_name == guard._pool_lease.listener.getsockname()

        hold.release()

        assert guard._phase is _Phase.IDLE and guard._pool_lease.current is None
        # The adopted channel went with the lease; the endpoint did not.
        control.close.assert_called_once_with()
        assert guard._pool_lease.listener is not None


def test_a_failed_claim_pins_nothing_and_an_unfreeable_endpoint_is_fatal(
    tmp_path: Path,
) -> None:
    """A failed claim gives back lease and endpoint; an unfreeable address is fatal."""
    registry = _new_registry(tmp_path, guard_count=2)
    guard = registry._guards[0]
    bound: list[socket.socket] = []
    try:
        # Everything fallible runs before the lease exists.
        with patch("kvcr.guard.os.dup", side_effect=OSError("dup failed")):
            with pytest.raises(OSError, match="dup failed"):
                registry.claim(0, _TEST_TIER_CONFIG, _FakeLiveness(), _control_bind(0))
        assert guard._pool_lease.listener is None and guard._pool_lease.current is None
        assert guard._reserved is None and guard._phase is not _Phase.PRIMARY

        # The pool survived its failed grant: the retry simply works.
        replacement = _FakeLiveness()
        _spec, lease = _claim(registry, 0, replacement)
        assert guard._pool_lease.current is lease
        assert guard._pool_lease.bind_address == _control_bind(0)

        # A rollback that cannot free the pool's address must stop the service.
        fatal = registry._guards[1]
        adopt_failure = RuntimeError("adopt failed")
        unbind_failure = OSError("close failed")
        uncontained: list[BaseException] = []
        registry.on_uncontained_failure = uncontained.append
        fatal._adopt = Mock(side_effect=adopt_failure)
        real_bind = fatal._pool_lease.bind

        def bind_a_listener_that_will_not_close(control_bind):
            listener, bound_here = real_bind(control_bind)
            bound.append(listener)
            fatal._pool_lease.listener = Mock(
                fileno=bound[-1].fileno, close=Mock(side_effect=unbind_failure)
            )
            return fatal._pool_lease.listener, bound_here

        fatal._pool_lease.bind = bind_a_listener_that_will_not_close
        with pytest.raises(RuntimeError) as raised:
            _claim(registry, 1, _FakeLiveness())

        # The claim's own error, not the one its cleanup hit.
        assert raised.value is adopt_failure
        assert uncontained == [unbind_failure]
        # The transition ended: nothing is left reserved against this pool.
        assert fatal._reserved is None and fatal._pool_lease.current is None
    finally:
        registry._guards[1]._pool_lease.listener = None
        for listener in bound:
            listener.close()
        registry.close()


def test_a_standby_survives_failed_claims_and_hands_over_to_a_replacement(
    tmp_path: Path,
) -> None:
    """Refused or failed claims cost a standby nothing; a good one inherits all."""
    control_bind = _control_bind(0)
    registry = _new_registry(tmp_path)
    guard = registry._guards[0]
    # The promotion itself is stubbed: what this exercises is the lifecycle
    # around a standby, not the recovery machinery inside one.
    guard._promote = Mock()
    liveness = _FakeLiveness()
    try:
        _spec, lease = _claim(registry, 0, liveness, control_bind)

        # With the actor's observation held back, the dead window answers busy.
        guard._observe_holder = lambda: None
        liveness.kill()
        with pytest.raises(KVCRServiceError, match="busy"):
            _claim(registry, 0, _FakeLiveness())
        guard._promote.assert_not_called()
        assert liveness.closed is False

        del guard._observe_holder
        _wait_until(lambda: guard._phase is _Phase.STANDBY, timeout=5)
        guard._promote.assert_called_once_with()
        assert liveness.closed is True and _holders_of(registry) == {}

        # A release racing the death it lost to is a no-op forever after.
        registry.release(0, lease)
        assert guard._phase is _Phase.STANDBY

        # Tiers are fixed by the first claim; a mismatched replacement is refused.
        with pytest.raises(KVCRServiceError, match="another tier configuration"):
            registry.claim(
                0,
                _TierConfig([("pool0", _TEST_BLOCK_SIZE_BYTES * 2)], None),
                _FakeLiveness(),
                control_bind,
            )

        # Only a hand-over that actually began costs the standby its endpoint.
        with patch(
            "kvcr.guard.ZmqPeerControlChannel.from_shared_listener",
            side_effect=ValueError("cannot share this listener"),
        ):
            with pytest.raises(KVCRServiceError, match="cannot share this listener"):
                _claim(registry, 0, _FakeLiveness(), control_bind)
        assert guard._phase is _Phase.STANDBY

        # An adopt failure fences the claim rather than the pool.
        real_adopt = guard._adopt
        guard._adopt = Mock(side_effect=RuntimeError("adopt failed"))
        with pytest.raises(RuntimeError, match="adopt failed"):
            _claim(registry, 0, _FakeLiveness(), control_bind)
        assert guard._phase is _Phase.STANDBY
        assert guard._reserved is None and guard._pool_lease.listener is not None
        assert _holders_of(registry) == {}

        guard._adopt = real_adopt
        replacement = _FakeLiveness()
        _spec, lease = _claim(registry, 0, replacement, control_bind)
        assert guard._phase is _Phase.PRIMARY
        assert guard._pool_lease.current is lease
        # The same endpoint, never rebound: the replacement inherited it.
        assert guard._pool_lease.listener.getsockname() == control_bind
    finally:
        registry.close()


def test_slow_promotion_does_not_block_other_pools_or_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _new_registry(tmp_path, guard_count=2)
    guard = registry._guards[0]
    promotion_started = threading.Event()
    continue_promotion = threading.Event()
    first = _FakeLiveness()
    second = _FakeLiveness()

    def promote() -> None:
        promotion_started.set()
        assert continue_promotion.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)

    guard._promote = Mock(side_effect=promote)
    try:
        _claim(registry, 0, first)
        first.kill()
        assert promotion_started.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)

        # The other pool progresses while pool 0 promotes...
        _spec, second_lease = _claim(registry, 1, second)
        # ...and pool 0 answers busy immediately instead of queueing.
        with pytest.raises(KVCRServiceError, match="busy"):
            _claim(registry, 0, _FakeLiveness())

        # Small but not zero: the healthy pool finishes well inside the deadline.
        monkeypatch.setattr(
            "kvcr.kvcr_service._REGISTRY_TRANSITION_TIMEOUT_SECONDS", 1.0
        )
        with pytest.raises(TimeoutError, match="Guard transitions"):
            registry.close()
        # The wedged pool is kept and named; its neighbour still went.
        assert registry._guards.keys() == {0} and second.closed is True
    finally:
        continue_promotion.set()
        monkeypatch.undo()
        registry.close()

    assert first.closed is True


def test_claim_refusals_and_internal_failures_do_not_bind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _running_server(tmp_path) as harness:
        spec = harness.server._registry._guards[0]._owner.spec
        with pytest.raises(KVCRServiceError, match="compatibility digest"):
            harness.client.claim(
                0, _TEST_POOL_LAYOUTS, _TEST_DIGEST.swapcase(), _control_bind()
            )
        with pytest.raises(KVCRServiceError, match="one complete KV block"):
            harness.client.claim(
                0,
                [("pool0", _TEST_POOL_SIZES_BYTES[0] + 1)],
                _TEST_DIGEST,
                _control_bind(),
            )
        assert "Unexpected failure while handling KVCR claim" not in caplog.text

        internal_error = AttributeError("internal invariant broke")
        with monkeypatch.context() as patcher:
            patcher.setattr(
                harness.server._server,
                "dispatch",
                Mock(side_effect=internal_error),
            )
            with pytest.raises(KVCRServiceError, match="internal KVCR service error"):
                harness.client.claim(
                    0, _TEST_POOL_LAYOUTS, _TEST_DIGEST, _control_bind()
                )

        log_record = next(
            record
            for record in caplog.records
            if record.message == "Unexpected failure while handling KVCR claim"
        )
        assert log_record.exc_info is not None
        assert log_record.exc_info[1] is internal_error
        assert harness.server._registry._guards[0]._owner.spec is spec
        assert _holders_of(harness.server._registry) == {}
        harness.client.claim(
            0, _TEST_POOL_LAYOUTS, _TEST_DIGEST, _control_bind()
        ).release()


def test_held_connection_accepts_only_release(tmp_path: Path) -> None:
    with _running_server(tmp_path, guard_count=1) as harness:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(harness.server.socket_path))
            channel = FramedConnection(connection)
            worker_pidfd = os.pidfd_open(os.getpid())
            try:
                channel.send_with_fd(_claim_request(), worker_pidfd)
            finally:
                os.close(worker_pidfd)
            assert isinstance(channel.receive(_CLAIM_RESPONSE_DECODER), _Granted)

            channel.send(_claim_request())
            assert isinstance(channel.receive(_RELEASE_RESPONSE_DECODER), _Error)
            assert 0 in _holders_of(harness.server._registry)

            channel.send({"type": "release", "version": 1, "future": True})
            response = channel.receive(_RELEASE_RESPONSE_DECODER)
            assert response == _Released(1)
            assert _holders_of(harness.server._registry) == {}


def test_unexpected_release_failure_stops_the_service() -> None:
    error = RuntimeError("registry invariant failed")
    handler = object.__new__(_RequestHandler)
    handler.server = Mock()
    handler.server.registry.release.side_effect = error

    assert handler._release_or_fail(0, Mock()) is False

    handler.server.fail.assert_called_once_with(error)


def test_a_release_refused_by_a_closing_registry_is_not_fatal() -> None:
    """The claimant learns its release did not commit; the service does not die."""
    handler = object.__new__(_RequestHandler)
    handler.server = Mock()
    handler.channel = Mock()
    handler.server.registry.release.side_effect = KVCRServiceError("closed")

    assert handler._release_or_fail(0, Mock()) is False

    handler.server.fail.assert_not_called()
    handler.channel.send.assert_called_once()


@pytest.mark.parametrize("legacy_peer_option", [False, True])
def test_fork_and_exec_do_not_preserve_claimant_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_peer_option: bool,
) -> None:
    if legacy_peer_option:
        monkeypatch.setattr(guard_protocol, "_SO_PEERPIDFD", 0x7FFFFFFF)
    with _running_server(tmp_path) as harness:
        exec_program = "import time; print('execed', flush=True); time.sleep(60)"
        program = "\n".join(
            (
                "import os",
                "import sys",
                "import time",
                "from kvcr.guard_protocol import KVCRClient",
                f"hold = KVCRClient({str(harness.server.socket_path)!r}).claim("
                f"0, {_TEST_POOL_LAYOUTS!r}, {_TEST_DIGEST!r}, "
                f"{_control_bind()!r})",
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
            spec = harness.server._registry._guards[0]._owner.spec
            assert spec is not None
            assert spec.path not in Path(f"/proc/{forked_pid}/maps").read_text()
            with pytest.raises(KVCRServiceError, match="held"):
                harness.client.claim(
                    0, _TEST_POOL_LAYOUTS, _TEST_DIGEST, _control_bind()
                )

            child.terminate()
            child.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
            # The fork outlives the process that claimed the pool.
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


def test_startup_rollback_gives_back_every_pool_but_a_wedged_one(
    tmp_path: Path,
) -> None:
    """A failed startup closes what it built; an unclosable mapping keeps its file."""
    attached: list[Mock] = []

    def attach(spec):
        if len(attached) == 0:
            pool = _stand_in_pool(spec)
            # Unlinking under a mapping a Guard still holds would fault the process.
            pool.close.side_effect = RuntimeError("still holding the mapping")
        elif len(attached) == 1:
            pool = _stand_in_pool(spec)
        else:
            raise RuntimeError("attach failed")
        attached.append(pool)
        return pool

    journal = Mock()
    journal.read_next.return_value = None
    with (
        patch("kvcr.guard.KVCRPoolAttachment.attach", side_effect=attach),
        patch("kvcr.guard.RecoveryJournal", Mock(return_value=journal)),
        pytest.raises(RuntimeError, match="attach failed"),
    ):
        _PoolRegistry(
            tmp_path, 3, _TEST_POOL_SIZES_BYTES, _TEST_JOURNAL_BYTES, _TEST_DIGEST
        )

    # The pool that closed is gone with the one that never attached; the pool
    # whose mapping would not close keeps its file for the next start's purge.
    attached[1].close.assert_called_once_with()
    assert [path.name.split("-")[1] for path in tmp_path.iterdir()] == ["pool_0"]


def test_startup_allocation_failure_rolls_back_before_listener(
    tmp_path: Path,
) -> None:
    allocation_error = OSError("allocation failed")
    real_allocate = _KVCRPoolOwner.allocate.__func__
    calls: list[int] = []

    def allocate(cls, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise allocation_error
        return real_allocate(cls, **kwargs)

    journal = Mock()
    journal.read_next.return_value = None
    with (
        patch("kvcr.guard.KVCRPoolAttachment.attach", side_effect=_stand_in_pool),
        patch("kvcr.guard.RecoveryJournal", Mock(return_value=journal)),
        patch.object(_KVCRPoolOwner, "allocate", classmethod(allocate)),
        patch("kvcr.kvcr_service._ThreadingUnixServer") as listener,
        pytest.raises(OSError) as raised,
    ):
        _KVCRService(
            _test_socket_path(),
            tmp_path,
            guard_count=3,
            pool_sizes_bytes=_TEST_POOL_SIZES_BYTES,
            compatibility_digest=_TEST_DIGEST,
            journal_bytes=_TEST_JOURNAL_BYTES,
        )

    assert raised.value is allocation_error
    listener.assert_not_called()
    # The two pools that were built are given back, files and all.
    assert list(tmp_path.iterdir()) == []


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
        pool_paths = list((tmp_path / "pools").glob("kvcr-pool_*-*"))
        assert len(pool_paths) == _TEST_GUARD_COUNT
        socket_path = harness.server.socket_path
        _wait_for_connection_state(harness.server, connected=False)
        idle_connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        idle_connection.connect(str(socket_path))
        _wait_for_connection_state(harness.server, connected=True)

        harness.stop()
        idle_connection.close()

        assert all(not path.exists() for path in pool_paths)
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


def test_a_failed_grant_delivery_retracts_through_the_guard(tmp_path: Path) -> None:
    """An unactivated release retracts an undelivered grant via abort_grant."""
    registry = _new_registry(tmp_path)
    accepted, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

    class _Channel:
        def __init__(self) -> None:
            self.sent: list[object] = []
            self.messages = [_claim_request(), _Release(1, activated=False)]

        def receive(self, _decoder: object):
            return self.messages.pop(0)

        def receive_with_fd(self, _decoder: object, _expected_counts: object):
            return self.messages.pop(0), [os.pidfd_open(os.getpid())]

        def send(self, response: object) -> None:
            self.sent.append(response)

        def send_with_fd(self, response: object, *descriptors: int) -> None:
            assert isinstance(response, _Granted)
            (listener_fd,) = descriptors
            assert listener_fd >= 0
            self.granted_fd = listener_fd
            raise RuntimeError("grant could not be delivered")

    class _Server:
        def __init__(self) -> None:
            self.registry = registry
            self.compatibility_digest = _TEST_DIGEST
            self.peer_trust = "strict"

        def dispatch(self, request, liveness):
            return _ThreadingUnixServer.dispatch(self, request, liveness)

    handler = object.__new__(_RequestHandler)
    handler.request = accepted
    handler.channel = _Channel()
    handler.server = _Server()
    guard = registry._guards[0]
    guard.abort_grant = Mock(wraps=guard.abort_grant)
    guard.release = Mock(wraps=guard.release)
    try:
        with patch("kvcr.kvcr_service.os.close", wraps=os.close) as os_close:
            handler.handle()
        # The duplicated endpoint went back whatever the send did: one fd per
        # grant would otherwise leak until claims fail with EMFILE.
        os_close.assert_any_call(handler.channel.granted_fd)
        assert handler.channel.sent == [_Released(1)]
        # Routed as a retraction, not an ordinary release: the Guard is the
        # one that knows whether this grant stood a serving Guard down.
        guard.abort_grant.assert_called_once()
        guard.release.assert_not_called()
        assert _holders_of(registry) == {}
        replacement = _FakeLiveness()
        _spec, lease = _claim(registry, 0, replacement)
        registry.release(0, lease)
    finally:
        peer.close()
        accepted.close()
        registry.close()


def test_an_undelivered_grants_lease_survives_its_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EOF is not a release: only the claimant's word or its death frees a lease."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    registry = _new_registry(tmp_path)
    # Promotion's serving core is not under test; the freed lease is.
    registry._guards[0]._promote = Mock()
    server = object.__new__(_ThreadingUnixServer)
    server.registry = registry
    server.compatibility_digest = _TEST_DIGEST
    server.peer_trust = "strict"
    monkeypatch.setattr(
        PidfdLiveness,
        "from_peer_socket",
        classmethod(
            lambda _cls, _sock, _fd, _trust: PidfdLiveness(os.pidfd_open(child.pid))
        ),
    )
    handler = object.__new__(_RequestHandler)
    handler.request = Mock()
    handler.server = server
    handler.channel = Mock()
    handler.channel.receive_with_fd.return_value = (_claim_request(), [])
    handler.channel.receive.side_effect = [EOFError()]
    handler.channel.send_with_fd.side_effect = RuntimeError("undelivered")
    try:
        # Delivery fails and the connection then EOFs; handle() returns with
        # the lease still held.
        handler.handle()
        with pytest.raises(KVCRServiceError, match="held by another worker"):
            _claim(registry, 0, _FakeLiveness())

        child.kill()
        child.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        # Death freed it: the pool's own actor promotes, and the lease is gone.
        guard = registry._guards[0]
        _wait_until(lambda: guard._phase is _Phase.STANDBY, timeout=5)
        assert _holders_of(registry) == {}
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        registry.close()


def test_promotion_failure_fails_the_pool_and_stops_the_whole_service(
    tmp_path: Path,
) -> None:
    """A failed promotion is escalated, fences its pool, and keeps the first error."""
    registry = _new_registry(tmp_path)
    guard = registry._guards[0]
    failure = RuntimeError("promotion failed")
    guard._promote = Mock(side_effect=failure)

    server = object.__new__(_ThreadingUnixServer)
    server.registry = registry
    server._fatal_error = None
    server._fatal_lock = threading.Lock()
    server.shutdown = Mock()
    registry.on_uncontained_failure = server.fail
    liveness = _FakeLiveness()
    try:
        _claim(registry, 0, liveness)

        _kill_and_wait(registry, 0, liveness)

        assert server._fatal_error is failure
        server.shutdown.assert_called_once_with()
        assert registry._refusing.is_set() is True
        assert guard._phase is _Phase.FAILED
        assert liveness.closed is True and _holders_of(registry) == {}
        # The fenced pool refuses its next claim with its own failure.
        with pytest.raises(RuntimeError, match="promotion failed"):
            guard.claim(_FakeLiveness(), _TEST_TIER_CONFIG, _control_bind(0))
        # Shutdown ends other pools' work too; those failures never usurp the first.
        server.fail(RuntimeError("and then the shutdown did too"))
        assert server._fatal_error is failure
    finally:
        registry.close()


def test_one_uninspectable_pool_file_does_not_stop_the_service(
    tmp_path: Path,
) -> None:
    """The purge logs and skips a candidate it cannot inspect, then starts."""
    stale = tmp_path / ("kvcr-stale-" + "a" * 32)
    stale.write_bytes(b"")
    with patch(
        "kvcr.kvcr_service._reclaim_pool_if_orphaned",
        side_effect=OSError("uninspectable"),
    ):
        registry = _new_registry(tmp_path)
    try:
        assert stale.exists()
        liveness = _FakeLiveness()
        _spec, lease = _claim(registry, 0, liveness)
        registry.release(0, lease)
    finally:
        registry.close()


def test_a_pools_endpoint_never_moves_across_claims(tmp_path: Path) -> None:
    """A claim naming a different control address is refused, not migrated."""
    registry = _new_registry(tmp_path)
    liveness = _FakeLiveness()
    try:
        _spec, lease = _claim(registry, 0, liveness)
        registry.release(0, lease)

        moved = ("127.0.0.1", free_port())
        with pytest.raises(KVCRServiceError, match="cannot be moved"):
            _claim(registry, 0, _FakeLiveness(), control_bind=moved)
        # The refusal cost nothing: the original endpoint still grants.
        _spec, lease = _claim(registry, 0, _FakeLiveness())
        registry.release(0, lease)
    finally:
        registry.close()


def test_a_lease_older_than_the_idle_timeout_still_releases_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The accept-time idle timeout must not sever a held connection."""
    monkeypatch.setattr("kvcr.kvcr_service._CLIENT_IDLE_TIMEOUT_SECONDS", 0.2)
    with _running_server(tmp_path) as harness:
        hold = harness.client.claim(
            0, _TEST_POOL_LAYOUTS, _TEST_DIGEST, _control_bind()
        )
        time.sleep(0.5)

        hold.release()

        assert harness.server._registry._guards[0]._pool_lease.current is None


def test_refuse_claims_is_a_barrier_no_grant_can_cross(tmp_path: Path) -> None:
    """refuse_claims waits out in-flight commits; nothing grants once it returns."""
    registry = _new_registry(tmp_path)
    guard = registry._guards[0]
    in_adopt = threading.Event()
    resume = threading.Event()
    returned = threading.Event()
    real_adopt = guard._adopt

    def gated_adopt(control, tier_config) -> None:
        real_adopt(control, tier_config)
        in_adopt.set()
        assert resume.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)

    guard._adopt = gated_adopt
    outcome: dict = {}

    def claim() -> None:
        try:
            outcome["grant"] = registry.claim(
                0, _TEST_TIER_CONFIG, _FakeLiveness(), _control_bind(0)
            )
        except BaseException as error:  # noqa: BLE001 - recorded for the assert
            outcome["error"] = error

    claimant = threading.Thread(target=claim)
    refuser = threading.Thread(
        target=lambda: (registry.refuse_claims(), returned.set())
    )
    claimant.start()
    try:
        assert in_adopt.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        with guard._phase_lock:
            # Held by us, exactly as a commit in flight would hold it.
            refuser.start()
            assert not returned.wait(timeout=0.3)
        assert returned.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        # refuse_claims has RETURNED; the in-flight claim must fail at commit.
        resume.set()
        claimant.join(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        assert not claimant.is_alive()
        assert "grant" not in outcome
        assert isinstance(outcome["error"], KVCRServiceError)
        assert guard._pool_lease.current is None and guard._reserved is None
        # The refusal rolled the adoption back: the channel the claim adopted
        # was closed and released, not left holding the pool's endpoint.
        assert guard._control is None
    finally:
        resume.set()
        refuser.join(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        registry.close()


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("hand-back failed"), TimeoutError("hand-back timed out")],
    ids=["plain", "timeout"],
)
def test_a_failed_release_reports_promptly_and_fences_the_pool_first(
    tmp_path: Path, failure: BaseException
) -> None:
    """A hand-back failure is the answer now, reported once nothing can claim."""
    registry = _new_registry(tmp_path)
    guard = registry._guards[0]
    guard._release = Mock(side_effect=failure)
    claimable_when_reported: list[bool] = []
    registry.on_uncontained_failure = lambda _error: claimable_when_reported.append(
        guard._pool_lease.current is None and guard._failure is None
    )
    liveness = _FakeLiveness()
    try:
        _spec, lease = _claim(registry, 0, liveness)

        started = time.monotonic()
        with pytest.raises(type(failure), match="hand-back") as raised:
            registry.release(0, lease)

        # The handler's own TimeoutError is the answer, not a wait to keep waiting.
        assert raised.value is failure
        assert time.monotonic() - started < 2
        # Reported while the pool was already fenced, so nothing could claim it.
        assert claimable_when_reported == [False]
        assert guard._phase is _Phase.FAILED
        with pytest.raises(type(failure), match="hand-back"):
            _claim(registry, 0, _FakeLiveness())
    finally:
        registry.close()


def test_a_pidfd_that_breaks_while_its_process_lives_is_service_fatal(
    tmp_path: Path,
) -> None:
    """Promotion on a broken descriptor could seat a second server on the pool."""
    registry = _new_registry(tmp_path)
    guard = registry._guards[0]
    guard._promote = Mock()
    uncontained: list[BaseException] = []
    registry.on_uncontained_failure = uncontained.append
    liveness = _FakeLiveness()
    try:
        _claim(registry, 0, liveness)
        poller = Mock()
        poller.poll.return_value = [(liveness.fileno(), select.POLLNVAL)]
        with patch("kvcr.guard.select.poll", return_value=poller):
            _wait_until(
                lambda: guard._phase is _Phase.FAILED and bool(uncontained), timeout=5
            )

        guard._promote.assert_not_called()
        _wait_until(lambda: len(uncontained) == 1, timeout=5)
        (error,) = uncontained
        assert isinstance(error, OSError) and "without POLLIN" in str(error)
    finally:
        registry.close()


def test_a_close_in_progress_absorbs_races_and_answers_stragglers(
    tmp_path: Path,
) -> None:
    """Races against a close are absorbed, and queued work is still answered."""
    registry = _new_registry(tmp_path)
    guard = registry._guards[0]
    liveness = _FakeLiveness()
    _spec, lease = _claim(registry, 0, liveness)
    gate = threading.Event()
    entered = threading.Event()
    real_close_resources = guard._close_resources
    real_owner_close = guard._owner.close

    def reclaim_pool() -> None:
        assert not guard._thread.is_alive()
        assert guard._recovery.attachment is None
        real_owner_close()

    guard._owner.close = reclaim_pool

    def slow_close() -> None:
        entered.set()
        assert gate.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        real_close_resources()

    guard._close_resources = slow_close
    guard._promote = Mock()
    try:
        registry.refuse_claims()
        guard.begin_close()
        assert entered.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)

        # Silently absorbed, exactly as the old registry did under its lock.
        registry.release(0, lease)
        # A death in the same window is the close's to clean up, not a promotion.
        liveness.kill()
        # A claim in the same window is refused with a typed error.
        with pytest.raises(KVCRServiceError, match="closed"):
            registry.claim(0, _TEST_TIER_CONFIG, _FakeLiveness(), _control_bind(0))

        # A command that slipped in behind the close is answered, not abandoned.
        stray = _Command("release", (object(),))
        guard._commands.put(stray)
        gate.set()
        deadline = time.monotonic() + _SERVER_STOP_TIMEOUT_SECONDS
        assert guard.finish_close(deadline) is False
        assert stray.future.done()
        assert isinstance(stray.future.exception(), KVCRServiceError)
    finally:
        gate.set()
        registry.close()
    guard._promote.assert_not_called()
    assert liveness.closed is True


def test_shutdown_drains_a_held_connection(tmp_path: Path) -> None:
    with _running_server(tmp_path) as harness:
        hold = harness.client.claim(
            0, _TEST_POOL_LAYOUTS, _TEST_DIGEST, _control_bind()
        )
        harness.stop()
        _wait_for_connection_state(harness.server, connected=False)
        hold._attachment.close()
        hold._connection.close()


def _service_args(pool_sizes_gb: str) -> list[str]:
    return (
        "--socket-path s --pool-dir d --guard-count 2 --pool-sizes-gb".split()
        + [pool_sizes_gb]
        + "--compatibility-digest g".split()
    )


def test_pool_size_list_preserves_order_and_floors_each_item_to_pages() -> None:
    raw_sizes = (2 * _PAGE_BLOCK_SIZE_BYTES + 123, 3 * _PAGE_BLOCK_SIZE_BYTES + 456)
    parsed = _parse_args(
        _service_args(",".join(str(size / (1 << 30)) for size in raw_sizes))
    )

    expected = (2 * _PAGE_BLOCK_SIZE_BYTES, 3 * _PAGE_BLOCK_SIZE_BYTES)
    assert parsed.pool_sizes_bytes == expected


def test_service_log_level_accepts_cli_and_environment(monkeypatch) -> None:
    monkeypatch.setenv("KVCR_LOG_LEVEL", "warning")
    assert _parse_args(_service_args("1")).log_level == "WARNING"
    assert _parse_args(_service_args("1") + ["--log-level", "debug"]).log_level == (
        "DEBUG"
    )


def test_service_heartbeat_timeout_is_opt_in() -> None:
    assert _parse_args(_service_args("1")).heartbeat_timeout_ms is None
    with pytest.raises(SystemExit):
        _parse_args(_service_args("1") + ["--heartbeat-timeout-ms", "999"])
    assert (
        _parse_args(
            _service_args("1") + ["--heartbeat-timeout-ms", "1000"]
        ).heartbeat_timeout_ms
        == 1000
    )


def test_service_startup_timeout_defaults_to_sixty_seconds() -> None:
    assert _parse_args(_service_args("1")).startup_timeout_seconds == 60
    with pytest.raises(SystemExit):
        _parse_args(_service_args("1") + ["--startup-timeout-seconds", "0"])
    assert (
        _parse_args(
            _service_args("1") + ["--startup-timeout-seconds", "600"]
        ).startup_timeout_seconds
        == 600
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "1,,2",
        "nan",
        "0",
        str((_PAGE_BLOCK_SIZE_BYTES - 1) / (1 << 30)),
        "1e1000000",
        ",".join([str(sys.maxsize // (1 << 30))] * 2),
    ],
)
def test_invalid_pool_size_items_are_rejected(value: str) -> None:
    with pytest.raises(SystemExit):
        _parse_args(_service_args(value))


def test_fd_claim_maps_the_pool_without_opening_its_path(tmp_path: Path) -> None:
    """Linux only: pidfd liveness. The worker gets the pool as a descriptor."""
    with _running_server(tmp_path) as harness:
        guard = harness.server._registry._guards[0]
        with patch(
            "kvcr.guard_protocol.KVCRPoolAttachment.attach",
            side_effect=AssertionError("fd mode must not open the pool path"),
        ):
            hold = harness.client.claim(
                0, _TEST_POOL_LAYOUTS, _TEST_DIGEST, _control_bind(0), attach_mode="fd"
            )
        try:
            assert hold._attachment.address
            assert hold._attachment._spec == guard._spec
        finally:
            hold.release()


def test_each_grant_gets_a_fresh_open_file_description(tmp_path: Path) -> None:
    """A dup would share the Guard's lock; a fresh description contends with it."""
    spec = manual_pool_spec(tmp_path)
    first, second = _open_pool_fd(spec), _open_pool_fd(spec)
    try:
        fcntl.flock(first, fcntl.LOCK_SH)
        # On one shared description this would convert the lock and succeed.
        with pytest.raises(OSError):
            fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(first)
        os.close(second)
    tampered = msgspec.structs.replace(spec, inode=spec.inode + 1)
    with pytest.raises(KVCRServiceError, match="does not match"):
        _open_pool_fd(tampered)


def test_a_pool_that_cannot_be_opened_is_reported_and_the_grant_aborted() -> None:
    """Before anything is sent: the claimant gets an error, not a stalled grant."""
    claimant, service_side = socket.socketpair()
    listener_fd = os.dup(os.open(os.devnull, os.O_RDONLY))
    lease = object()
    server = Mock()
    server.dispatch.return_value = (
        SimpleNamespace(attach_mode="fd", spec=None),
        (0, listener_fd, lease),
    )
    with claimant, service_side:
        FramedConnection(claimant).send(
            _Claim(0, _TEST_DIGEST, _TEST_TIER_CONFIG, "127.0.0.1", 5555, 1, "x", "fd")
        )
        with (
            patch("kvcr.kvcr_service.PidfdLiveness"),
            patch(
                "kvcr.kvcr_service._open_pool_fd",
                side_effect=OSError(errno.EMFILE, "too many open files"),
            ),
        ):
            _RequestHandler(service_side, ("", 0), server)

        response = FramedConnection(claimant).receive(_CLAIM_RESPONSE_DECODER)

    assert isinstance(response, _Error)
    server.registry.abort_grant.assert_called_once_with(0, lease)
    with pytest.raises(OSError):
        os.fstat(listener_fd)


def _pool_files(directory: Path) -> dict[str, int]:
    return {
        path.name: path.stat().st_ino for path in sorted(directory.glob("kvcr-pool_*"))
    }


def test_a_restarted_service_adopts_its_pools_and_serves_them(
    tmp_path: Path,
) -> None:
    first = _new_registry(tmp_path, guard_count=2, adopt_pools=True)
    specs = {rank: guard._spec for rank, guard in first._guards.items()}
    files = _pool_files(tmp_path)
    assert len(files) == 2
    # A worker's mapping, which a guard restart must not invalidate.
    with open(specs[0].path, "r+b") as pool:
        worker = mmap.mmap(pool.fileno(), specs[0].mapping_bytes)
    worker[-5:] = b"bytes"
    first.close()
    # Kept on close, as a crash would leave them.
    assert _pool_files(tmp_path) == files

    second = _new_registry(tmp_path, guard_count=2, adopt_pools=True)
    try:
        assert _pool_files(tmp_path) == files
        for rank, guard in second._guards.items():
            assert guard._spec == msgspec.structs.replace(specs[rank], epoch=1)
            assert not guard._held
        assert worker[-5:] == b"bytes"
        spec, lease = _claim(second, 0, _FakeLiveness())
        assert (spec.generation, spec.inode, spec.epoch) == (
            specs[0].generation,
            specs[0].inode,
            1,
        )
        second.release(0, lease)
    finally:
        worker.close()
        second.close()
    assert _pool_files(tmp_path) == files


def test_without_the_flag_a_restart_allocates_fresh_and_unlinks(
    tmp_path: Path,
) -> None:
    first = _new_registry(tmp_path)
    generation = first._guards[0]._spec.generation
    first.close()
    assert _pool_files(tmp_path) == {}

    second = _new_registry(tmp_path)
    try:
        assert second._guards[0]._spec.generation != generation
    finally:
        second.close()


def test_a_changed_configuration_refuses_the_old_pool_and_purges_it(
    tmp_path: Path,
) -> None:
    first = _new_registry(tmp_path, adopt_pools=True)
    old = _pool_files(tmp_path)
    first.close()

    second = _new_registry(tmp_path, digest="another model", adopt_pools=True)
    try:
        now = _pool_files(tmp_path)
        assert len(now) == 1
        assert not set(now) & set(old)
    finally:
        second.close()


def test_a_second_live_service_does_not_adopt_the_first_ones_pools(
    tmp_path: Path,
) -> None:
    first = _new_registry(tmp_path, adopt_pools=True)
    mine = _pool_files(tmp_path)
    second = _new_registry(tmp_path, adopt_pools=True)
    try:
        assert second._adoption_lock is None
        assert second._guards[0]._spec.path != first._guards[0]._spec.path
        assert set(mine) <= set(_pool_files(tmp_path))
    finally:
        second.close()
    # The loser unlinks its own pool as a service always did; the first is untouched.
    assert _pool_files(tmp_path) == mine
    first.close()


@pytest.mark.usefixtures("_stand_in_guards")
def test_adoption_leaves_a_live_workers_pool_and_purges_true_orphans(
    tmp_path: Path,
) -> None:
    first = _new_registry(tmp_path, adopt_pools=True)
    spec = first._guards[0]._spec
    first.close()
    # A worker of the previous service still holds its pool.
    worker = os.open(spec.path, os.O_RDWR)
    fcntl.flock(worker, fcntl.LOCK_SH)
    older = tmp_path / ("kvcr-pool_0-" + "b" * 32)
    older.write_bytes(b"")
    other_rank = tmp_path / ("kvcr-pool_7-" + "c" * 32)
    other_rank.write_bytes(b"")
    for orphan in (older, other_rank):
        orphan.chmod(0o600)
    os.utime(older, ns=(1, 1))

    second = _new_registry(tmp_path, adopt_pools=True)
    try:
        assert second._guards[0]._spec.inode == spec.inode
        # The worker still holds it, so it is adopted but not yet claimable.
        assert second._guards[0]._held
        assert not older.exists()
        assert not other_rank.exists()
        assert Path(spec.path).exists()
    finally:
        second.close()
        os.close(worker)


def test_adopt_pools_is_an_opt_in_flag() -> None:
    assert _parse_args(_service_args("1")).adopt_pools is False
    assert _parse_args(_service_args("1") + ["--adopt-pools"]).adopt_pools is True


@pytest.mark.usefixtures("_real_attach_guards")
def test_a_pool_a_previous_worker_still_holds_is_refused_until_it_exits(
    tmp_path: Path,
) -> None:
    first = _new_registry(tmp_path, adopt_pools=True, real_attach=True)
    spec = first._guards[0]._spec
    first.close()
    worker = os.open(spec.path, os.O_RDWR)
    fcntl.flock(worker, fcntl.LOCK_SH)

    second = _new_registry(tmp_path, adopt_pools=True, real_attach=True)
    try:
        guard = second._guards[0]
        assert guard._spec.inode == spec.inode
        assert guard._held
        # Not attached while held: that would block its own takeover probe.
        assert guard._recovery.attachment is None
        with pytest.raises(KVCRServiceError, match="still attached by a previous"):
            _claim(second, 0, _FakeLiveness())
        # Its holder going away must not expose it to a purge before the Guard
        # has taken it: the purge skips it by identity, not only by lock.
        os.close(worker)
        worker = -1
        second._purge_orphaned_pools({(spec.device, spec.inode)})
        assert Path(spec.path).exists()
        _wait_until(lambda: not guard._held, timeout=5)
        _spec, lease = _claim(second, 0, _FakeLiveness())
        assert _spec.inode == spec.inode
        second.release(0, lease)
    finally:
        if worker >= 0:
            os.close(worker)
        second.close()


@pytest.mark.usefixtures("_stand_in_guards")
def test_a_service_that_lost_adoption_does_not_purge_the_winners_held_pool(
    tmp_path: Path,
) -> None:
    _new_registry(tmp_path, adopt_pools=True).close()
    spec_path = next(iter(_pool_files(tmp_path)))
    worker = os.open(tmp_path / spec_path, os.O_RDWR)
    fcntl.flock(worker, fcntl.LOCK_SH)
    winner = _new_registry(tmp_path, adopt_pools=True)
    orphan = tmp_path / ("kvcr-pool_9-" + "d" * 32)
    orphan.write_bytes(b"")
    orphan.chmod(0o600)
    try:
        assert winner._guards[0]._held
        loser = _new_registry(tmp_path, adopt_pools=True)
        try:
            # Nothing it did not allocate is touched, orphans included.
            assert orphan.exists()
            os.close(worker)
            worker = -1
            assert (tmp_path / spec_path).exists()
            _wait_until(lambda: not winner._guards[0]._held, timeout=5)
            assert winner._guards[0]._failure is None
        finally:
            loser.close()
        assert (tmp_path / spec_path).exists()
    finally:
        if worker >= 0:
            os.close(worker)
        winner.close()


def test_a_plain_service_still_purges_true_orphans(tmp_path: Path) -> None:
    orphan = tmp_path / ("kvcr-pool_9-" + "e" * 32)
    orphan.write_bytes(b"")
    orphan.chmod(0o600)
    registry = _new_registry(tmp_path)
    try:
        assert not orphan.exists()
    finally:
        registry.close()


def _unshare_pid_namespace() -> list[str]:
    """The argv prefix that runs a command as pid 1 of a fresh PID namespace.

    Skips, with the probe's own error, when the host cannot: no unshare (macOS),
    user namespaces disabled, or /proc cannot be remounted (AppArmor).
    """
    prefix = ["unshare", "--pid", "--fork", "--mount-proc"]
    if os.geteuid() != 0:
        prefix = ["unshare", "--user", "--map-root-user", *prefix[1:]]
    try:
        probe = subprocess.run(
            [*prefix, "true"], capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        pytest.skip(f"SKIPPED, no PID namespace ({' '.join(prefix)}): {error}")
    if probe.returncode != 0:
        pytest.skip(
            f"SKIPPED, no PID namespace ({' '.join(prefix)} true exited "
            f"{probe.returncode}): {probe.stderr.strip()}"
        )
    return prefix


def _read_line(process: subprocess.Popen, timeout: float) -> str:
    assert process.stdout is not None
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    assert ready, "no output within the timeout"
    return process.stdout.readline()


_CLAIM_AND_WAIT = """
import sys, time
from kvcr.guard_protocol import KVCRClient
hold = KVCRClient(sys.argv[1]).claim(0, {layouts!r}, {digest!r}, {bind!r})
print("claimed", flush=True)
time.sleep(60)
"""


def test_claimant_in_a_child_pid_namespace_is_visible_and_its_death_promotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service is an ancestor of the claimant, so it can still verify it.

    Only a peer outside the service's own PID subtree reads as pid 0.
    """
    prefix = _unshare_pid_namespace()
    monkeypatch.setattr(guard_protocol, "_SO_PEERPIDFD", 0x7FFFFFFF)
    with _running_server(tmp_path, guard_count=1, peer_trust="trusted-peer") as harness:
        guard = harness.server._registry._guards[0]
        guard._promote = Mock()
        program = _CLAIM_AND_WAIT.format(
            layouts=_TEST_POOL_LAYOUTS, digest=_TEST_DIGEST, bind=_control_bind(0)
        )
        child = subprocess.Popen(
            [*prefix, sys.executable, "-c", program, str(harness.server.socket_path)],
            stdout=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        try:
            assert _read_line(child, 20) == "claimed\n"
            lease = guard._pool_lease.current
            assert (lease.mode, lease.verified, lease.fenceable) == (
                "pidfd_verified",
                True,
                True,
            )
            os.killpg(child.pid, signal.SIGKILL)
            _wait_until(lambda: guard._phase is _Phase.STANDBY, timeout=5)
            guard._promote.assert_called_once_with()
        finally:
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
            if child.stdout is not None:
                child.stdout.close()


# The Guard side of the real topology: the service runs in its own PID
# namespace, so a worker outside it reads as pid 0 on both sides of the check.
# Stand-in pools and a recording promotion keep nixl out of it.
_SERVICE_IN_PID_NAMESPACE = """
import sys
from unittest.mock import Mock, patch
sys.path.insert(0, {tests!r})
import test_kvcr_service as t
from kvcr import guard, guard_protocol
from kvcr.kvcr_service import _KVCRService

guard_protocol._SO_PEERPIDFD = 0x7FFFFFFF
channel = Mock(recv=Mock(return_value=[]))

def take(duplicate):
    duplicate.close()
    return channel

def promote_for(self, lease):
    print("promoted", flush=True)

socket_path, pool_dir, peer_trust = sys.argv[1:4]
with (
    t._guard_patches(),
    patch("kvcr.guard.ZmqPeerControlChannel.from_shared_listener", side_effect=take),
    patch.object(guard._Guard, "_promote_for", promote_for),
):
    service = _KVCRService(
        socket_path,
        pool_dir,
        guard_count=1,
        pool_sizes_bytes=t._TEST_POOL_SIZES_BYTES,
        compatibility_digest=t._TEST_DIGEST,
        journal_bytes=t._TEST_JOURNAL_BYTES,
        peer_trust=peer_trust,
    )
    print("ready", flush=True)
    service.serve_forever()
"""


@contextmanager
def _service_in_pid_namespace(
    tmp_path: Path, peer_trust: str
) -> Iterator[tuple[subprocess.Popen, Path]]:
    prefix = _unshare_pid_namespace()
    socket_path = _test_socket_path()
    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()
    program = _SERVICE_IN_PID_NAMESPACE.format(tests=str(Path(__file__).parent))
    service = subprocess.Popen(
        [
            *prefix,
            sys.executable,
            "-c",
            program,
            str(socket_path),
            str(pool_dir),
            peer_trust,
        ],
        stdout=subprocess.PIPE,
        start_new_session=True,
        text=True,
    )
    try:
        assert _read_line(service, 30) == "ready\n"
        yield service, socket_path
    finally:
        with suppress(ProcessLookupError):
            os.killpg(service.pid, signal.SIGKILL)
        service.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        if service.stdout is not None:
            service.stdout.close()
        with suppress(OSError):
            socket_path.unlink()


def test_service_in_its_own_pid_namespace_refuses_a_worker_it_cannot_see_when_strict(
    tmp_path: Path,
) -> None:
    with _service_in_pid_namespace(tmp_path, "strict") as (_service, socket_path):
        with pytest.raises(KVCRServiceError, match="does not match socket peer"):
            KVCRClient(socket_path).claim(
                0, _TEST_POOL_LAYOUTS, _TEST_DIGEST, _control_bind(0)
            )


def test_service_in_its_own_pid_namespace_grants_a_worker_it_cannot_see_when_trusted(
    tmp_path: Path,
) -> None:
    with _service_in_pid_namespace(tmp_path, "trusted-peer") as (service, socket_path):
        program = _CLAIM_AND_WAIT.format(
            layouts=_TEST_POOL_LAYOUTS, digest=_TEST_DIGEST, bind=_control_bind(0)
        )
        worker = subprocess.Popen(
            [sys.executable, "-c", program, str(socket_path)],
            stdout=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        try:
            assert _read_line(worker, 20) == "claimed\n"
            # Death crosses the namespace through the pidfd and nothing else.
            worker.kill()
            assert _read_line(service, 10) == "promoted\n"
        finally:
            worker.kill()
            worker.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
            if worker.stdout is not None:
                worker.stdout.close()
