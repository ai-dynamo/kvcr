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
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import msgspec
import pytest
from _kvcr_test_utils import _wait_until, free_port, listening_socket

from kvcr import KVCRClient, KVCRServiceError
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
    _Release,
    _Released,
    _TierConfig,
)
from kvcr.kvcr_service import (
    _KVCRService,
    _parse_args,
    _PoolRegistry,
    _RequestHandler,
    _ThreadingUnixServer,
)
from kvcr.memory import _KVCRPoolOwner


def _holders_of(registry) -> dict[int, object]:
    """The pools a worker holds, in the shape the old binding map had."""
    return {i: p._lease for i, p in registry._pools.items() if p._lease is not None}


_SERVER_STOP_TIMEOUT_SECONDS = 5
_CONNECTION_POLL_INTERVAL_SECONDS = 0.001

_TEST_POOL_COUNT = 2
_TEST_JOURNAL_BYTES = 8192
_TEST_POOL_SIZE_BYTES = _TEST_JOURNAL_BYTES + 8192
_TEST_ROW_STRIDE = 1024
_TEST_DIGEST = "opaque digest: Preserve-Me EXACTLY"
_TEST_TIER_CONFIG = _TierConfig(_TEST_ROW_STRIDE, None)
# G3 terms are refused at decode unless a real claimant could open them, so
# the one claim that carries G3 uses a page-aligned stride.
_PAGE_STRIDE = os.sysconf("SC_PAGE_SIZE")


class _FakeLiveness:
    """A pollable pidfd stand-in: a pipe that reads POLLIN once killed."""

    def __init__(self) -> None:
        self._read, self._write = os.pipe()
        self.closed = False

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


def _claim(registry, pool_index, liveness, control_bind=None):
    """Claim through the registry, closing the granted fd the tests never send."""
    spec, listener_fd, lease = registry.claim(
        pool_index,
        _TEST_TIER_CONFIG,
        liveness,
        control_bind or _control_bind(pool_index),
    )
    os.close(listener_fd)
    return spec, lease


def _kill_and_wait(registry, pool_index, liveness) -> None:
    """Die the way a real claimant does: the pool's own actor notices."""
    liveness.kill()
    guard = registry._pools[pool_index]
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
    pool_count: int = _TEST_POOL_COUNT,
    pool_size_bytes: int = _TEST_POOL_SIZE_BYTES,
    journal_bytes: int = _TEST_JOURNAL_BYTES,
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
        journal_bytes=journal_bytes,
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


_POOL_BINDS: dict[int, tuple[str, int]] = {}


def _control_bind(pool_index: int = 0) -> tuple[str, int]:
    """The address a pool answers on, stable for as long as it exists."""
    if pool_index not in _POOL_BINDS:
        _POOL_BINDS[pool_index] = ("127.0.0.1", free_port())
    return _POOL_BINDS[pool_index]


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
    # Within a test a pool's endpoint never moves; across tests it is gone.
    _POOL_BINDS.clear()
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
    _POOL_BINDS.clear()


def _stand_in_pool(spec) -> Mock:
    """The pool-tail surface a Guard reaches for, without a real mapping."""
    attachment = Mock(address=1234, data_address=1234 + spec.journal_bytes, _spec=spec)
    attachment.mapped_snapshot.return_value = nullcontext(None)
    return attachment


def _new_registry(tmp_path: Path, pool_count: int = 1) -> _PoolRegistry:
    """A registry of real Guards over stand-in pool mappings."""
    journal = Mock()
    journal.read_next.return_value = None
    with (
        patch("kvcr.guard.KVCRPoolAttachment.attach", side_effect=_stand_in_pool),
        patch("kvcr.guard.RecoveryJournal", Mock(return_value=journal)),
    ):
        return _PoolRegistry(
            tmp_path,
            pool_count,
            _TEST_POOL_SIZE_BYTES,
            _TEST_JOURNAL_BYTES,
            _TEST_DIGEST,
        )


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


def test_registry_lifecycle_from_independent_leases_to_a_wedged_close(
    tmp_path: Path,
) -> None:
    """Pools lease independently; close keeps, names, and can retry a wedged one."""
    registry = _new_registry(tmp_path, pool_count=2)
    guard = registry._pools[0]
    first, second, third = _FakeLiveness(), _FakeLiveness(), _FakeLiveness()
    first_spec, stale = _claim(registry, 0, first)
    _spec, _fd, _lease = registry.claim(
        1, _TierConfig(_TEST_ROW_STRIDE * 2, None), second, _control_bind(1)
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
    assert registry._pools[0]._lease is current
    assert third.closed is False

    first_path = Path(guard._owner.spec.path)
    second_path = Path(registry._pools[1]._owner.spec.path)
    # An unclosable mapping keeps its file: unlinking would hide committed RAM.
    guard._attachment.close.side_effect = RuntimeError("still holding the mapping")
    with pytest.raises(RuntimeError, match="still holding the mapping"):
        registry.close()
    assert registry._pools.keys() == {0}
    assert registry._pools[0]._attachment is not None
    assert first_path.exists()
    # The pool behind it still went, files and all.
    assert second.closed is True and not second_path.exists()

    # The kept pool must still name what it leaked, or nobody could retry it.
    guard._attachment.close.side_effect = None
    stubborn = Mock(close=Mock(side_effect=OSError("will not close")))
    guard._listener = stubborn
    with pytest.raises(OSError, match="will not close"):
        registry.close()
    assert registry._pools.keys() == {0}
    assert registry._pools[0]._listener is stubborn

    # With nothing left refusing, the retried close finally takes the pool.
    guard._listener = None
    registry.close()
    assert registry._pools == {} and third.closed is True


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
            harness.client.claim(0, _TEST_ROW_STRIDE, _TEST_DIGEST, control_bind=taken)

        # The rejected claims left the pool free to take normally.
        harness.client.claim(
            0, _TEST_ROW_STRIDE, _TEST_DIGEST, _control_bind()
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
            channel.send(claim)
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
        capacity_bytes_per_file=2 * _PAGE_STRIDE,
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
        guard = harness.server._registry._pools[1]
        # Built and started with the pool, before any claim named it.
        assert guard._phase is _Phase.UNCONFIGURED

        hold = harness.client.claim(1, _PAGE_STRIDE, _TEST_DIGEST, _control_bind(), g3)
        assert guard._phase is _Phase.PRIMARY and guard._control is control
        assert guard._configured == _TierConfig(
            _PAGE_STRIDE,
            _G3Config(
                paths=(str(g3.paths[0]),),
                capacity_bytes_per_file=g3.capacity_bytes_per_file,
                backend=g3.backend,
                backend_options=dict(g3.backend_options),
            ),
        )
        taken_fd, taken_name = taken[0]
        # The Guard is handed a duplicate; the pool keeps the original.
        assert taken_fd != guard._listener.fileno()
        assert taken_name == guard._listener.getsockname()

        hold.release()

        assert guard._phase is _Phase.IDLE and guard._lease is None
        # The adopted channel went with the lease; the endpoint did not.
        control.close.assert_called_once_with()
        assert guard._listener is not None


def test_a_failed_claim_pins_nothing_and_an_unfreeable_endpoint_is_fatal(
    tmp_path: Path,
) -> None:
    """A failed claim gives back lease and endpoint; an unfreeable address is fatal."""
    registry = _new_registry(tmp_path, pool_count=2)
    guard = registry._pools[0]
    bound: list[socket.socket] = []
    try:
        # Everything fallible runs before the lease exists.
        with patch("kvcr.guard.os.dup", side_effect=OSError("dup failed")):
            with pytest.raises(OSError, match="dup failed"):
                registry.claim(0, _TEST_TIER_CONFIG, _FakeLiveness(), _control_bind(0))
        assert guard._listener is None and guard._lease is None
        assert guard._reserved is None and guard._phase is not _Phase.PRIMARY

        # The pool survived its failed grant: the retry simply works.
        replacement = _FakeLiveness()
        _spec, lease = _claim(registry, 0, replacement)
        assert guard._lease is lease and guard._bind == _control_bind(0)

        # A rollback that cannot free the pool's address must stop the service.
        fatal = registry._pools[1]
        adopt_failure = RuntimeError("adopt failed")
        unbind_failure = OSError("close failed")
        uncontained: list[BaseException] = []
        registry.on_uncontained_failure = uncontained.append
        fatal._adopt = Mock(side_effect=adopt_failure)
        real_bind = fatal._bind_listener

        def bind_a_listener_that_will_not_close(control_bind):
            bound.append(real_bind(control_bind))
            fatal._listener = Mock(
                fileno=bound[-1].fileno, close=Mock(side_effect=unbind_failure)
            )
            return fatal._listener

        fatal._bind_listener = bind_a_listener_that_will_not_close
        with pytest.raises(RuntimeError) as raised:
            _claim(registry, 1, _FakeLiveness())

        # The claim's own error, not the one its cleanup hit.
        assert raised.value is adopt_failure
        assert uncontained == [unbind_failure]
        # The transition ended: nothing is left reserved against this pool.
        assert fatal._reserved is None and fatal._lease is None
    finally:
        registry._pools[1]._listener = None
        for listener in bound:
            listener.close()
        registry.close()


def test_a_standby_survives_failed_claims_and_hands_over_to_a_replacement(
    tmp_path: Path,
) -> None:
    """Refused or failed claims cost a standby nothing; a good one inherits all."""
    control_bind = _control_bind(0)
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
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
                _TierConfig(_TEST_ROW_STRIDE * 2, None),
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
        assert guard._reserved is None and guard._listener is not None
        assert _holders_of(registry) == {}

        guard._adopt = real_adopt
        replacement = _FakeLiveness()
        _spec, lease = _claim(registry, 0, replacement, control_bind)
        assert guard._phase is _Phase.PRIMARY and guard._lease is lease
        # The same endpoint, never rebound: the replacement inherited it.
        assert guard._listener.getsockname() == control_bind
    finally:
        registry.close()


def test_slow_promotion_does_not_block_other_pools_or_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _new_registry(tmp_path, pool_count=2)
    guard = registry._pools[0]
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
        with pytest.raises(TimeoutError, match="pool transitions"):
            registry.close()
        # The wedged pool is kept and named; its neighbour still went.
        assert registry._pools.keys() == {0} and second.closed is True
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
        spec = harness.server._registry._pools[0]._owner.spec
        with pytest.raises(KVCRServiceError, match="compatibility digest"):
            harness.client.claim(
                0, _TEST_ROW_STRIDE, _TEST_DIGEST.swapcase(), _control_bind()
            )
        with pytest.raises(KVCRServiceError, match="one complete KV row"):
            harness.client.claim(
                0, _TEST_POOL_SIZE_BYTES + 1, _TEST_DIGEST, _control_bind()
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
                harness.client.claim(0, _TEST_ROW_STRIDE, _TEST_DIGEST, _control_bind())

        log_record = next(
            record
            for record in caplog.records
            if record.message == "Unexpected failure while handling KVCR claim"
        )
        assert log_record.exc_info is not None
        assert log_record.exc_info[1] is internal_error
        assert harness.server._registry._pools[0]._owner.spec is spec
        assert _holders_of(harness.server._registry) == {}
        harness.client.claim(
            0, _TEST_ROW_STRIDE, _TEST_DIGEST, _control_bind()
        ).release()


def test_held_connection_accepts_only_release(tmp_path: Path) -> None:
    with _running_server(tmp_path, pool_count=1) as harness:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(harness.server.socket_path))
            channel = FramedConnection(connection)
            channel.send(_claim_request())
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
                f"0, {_TEST_ROW_STRIDE}, {_TEST_DIGEST!r}, {_control_bind()!r})",
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
            spec = harness.server._registry._pools[0]._owner.spec
            assert spec is not None
            assert spec.path not in Path(f"/proc/{forked_pid}/maps").read_text()
            with pytest.raises(KVCRServiceError, match="held"):
                harness.client.claim(0, _TEST_ROW_STRIDE, _TEST_DIGEST, _control_bind())

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
            tmp_path, 3, _TEST_POOL_SIZE_BYTES, _TEST_JOURNAL_BYTES, _TEST_DIGEST
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
            pool_count=3,
            pool_size_bytes=_TEST_POOL_SIZE_BYTES,
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
        assert len(pool_paths) == _TEST_POOL_COUNT
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

        def send(self, response: object) -> None:
            self.sent.append(response)

        def send_with_fd(self, response: object, listener_fd: int) -> None:
            assert isinstance(response, _Granted)
            assert listener_fd >= 0
            self.granted_fd = listener_fd
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
    guard = registry._pools[0]
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
    registry._pools[0]._promote = Mock()
    server = object.__new__(_ThreadingUnixServer)
    server.registry = registry
    server.compatibility_digest = _TEST_DIGEST
    monkeypatch.setattr(
        PidfdLiveness,
        "from_peer_socket",
        classmethod(lambda _cls, _sock: PidfdLiveness(os.pidfd_open(child.pid))),
    )
    handler = object.__new__(_RequestHandler)
    handler.request = Mock()
    handler.server = server
    handler.channel = Mock()
    handler.channel.receive.side_effect = [_claim_request(), EOFError()]
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
        guard = registry._pools[0]
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
    guard = registry._pools[0]
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
        hold = harness.client.claim(0, _TEST_ROW_STRIDE, _TEST_DIGEST, _control_bind())
        time.sleep(0.5)

        hold.release()

        assert harness.server._registry._pools[0]._lease is None


def test_refuse_claims_is_a_barrier_no_grant_can_cross(tmp_path: Path) -> None:
    """refuse_claims waits out in-flight commits; nothing grants once it returns."""
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
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
        assert guard._lease is None and guard._reserved is None
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
    guard = registry._pools[0]
    guard._release = Mock(side_effect=failure)
    claimable_when_reported: list[bool] = []
    registry.on_uncontained_failure = lambda _error: claimable_when_reported.append(
        guard._lease is None and guard._failure is None
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
    guard = registry._pools[0]
    guard._promote = Mock()
    uncontained: list[BaseException] = []
    registry.on_uncontained_failure = uncontained.append
    liveness = _FakeLiveness()
    try:
        _claim(registry, 0, liveness)
        poller = Mock()
        poller.poll.return_value = [(liveness.fileno(), select.POLLNVAL)]
        with patch("kvcr.guard.select.poll", return_value=poller):
            _wait_until(lambda: guard._phase is _Phase.FAILED, timeout=5)

        guard._promote.assert_not_called()
        (error,) = uncontained
        assert isinstance(error, OSError) and "without POLLIN" in str(error)
    finally:
        registry.close()


def test_a_close_in_progress_absorbs_races_and_answers_stragglers(
    tmp_path: Path,
) -> None:
    """Races against a close are absorbed, and queued work is still answered."""
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    liveness = _FakeLiveness()
    _spec, lease = _claim(registry, 0, liveness)
    gate = threading.Event()
    entered = threading.Event()
    real_close_resources = guard._close_resources

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
        hold = harness.client.claim(0, _TEST_ROW_STRIDE, _TEST_DIGEST, _control_bind())
        harness.stop()
        _wait_for_connection_state(harness.server, connected=False)
        hold._attachment.close()
        hold._connection.close()


def test_a_pool_no_bigger_than_its_journal_is_rejected_at_the_flag() -> None:
    """Sized at or below the journal, a pool has nothing left to cache with."""
    flags = "--socket-path s --pool-dir d --pool-count 1 --compatibility-digest g"
    with pytest.raises(SystemExit):
        _parse_args([*flags.split(), "--pool-size-gb", "0.05"])
    parsed = _parse_args([*flags.split(), "--pool-size-gb", "0.2"])
    assert parsed.pool_size_bytes > 100 * (1 << 20)
