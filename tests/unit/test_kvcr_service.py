# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
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
from _kvcr_test_utils import free_port, listening_socket

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
    return {
        i: p._lease.liveness for i, p in registry._pools.items() if p._lease is not None
    }


def _listeners_of(registry) -> dict[int, object]:
    return {
        i: p._listener for i, p in registry._pools.items() if p._listener is not None
    }


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
    """A pollable stand-in for a claimant's pidfd.

    A real pidfd reads POLLIN once its process exits; a pipe read end reads
    POLLIN once a byte is written. kill() is that byte.
    """

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


def _await_phase(guard, predicate, timeout=5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate(guard):
        if time.monotonic() >= deadline:
            pytest.fail(f"pool never reached the expected state: {guard._phase}")
        time.sleep(0.001)


def _kill_and_wait(registry, pool_index, liveness) -> None:
    """Die the way a real claimant does: the pool's own actor notices."""
    from kvcr.guard import _Phase

    liveness.kill()
    _await_phase(
        registry._pools[pool_index],
        lambda g: g._phase in (_Phase.STANDBY, _Phase.FAILED) or g._failure,
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


def _control_address() -> tuple[str, int]:
    """A free address a client can ask the service to bind."""
    bind = _control_bind()
    return bind


def _takes(*channels):
    """Stand in for from_shared_listener, which detaches what it is given."""
    remaining = list(channels)

    def _take(duplicate: socket.socket):
        duplicate.close()
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return _take


_POOL_BINDS: dict[int, tuple[str, int]] = {}


def _control_bind(pool_index: int = 0) -> tuple[str, int]:
    """The address a pool answers on, stable for as long as it exists."""
    if pool_index not in _POOL_BINDS:
        _POOL_BINDS[pool_index] = ("127.0.0.1", free_port())
    return _POOL_BINDS[pool_index]


def _claim_request(
    compatibility_digest: str = _TEST_DIGEST,
    g3: _G3Config | None = None,
    control_bind: tuple[str, int] | None = None,
) -> _Claim:
    host, port = control_bind or _control_bind(0)
    return _Claim(
        0,
        compatibility_digest,
        _TierConfig(_TEST_ROW_STRIDE, g3),
        host,
        port,
        1,
    )


@pytest.fixture(autouse=True)
def _channels_are_taken():
    """Close the duplicate a claim hands its Guard, as a real Guard would."""
    # Within a test a pool's endpoint never moves; across tests it is gone.
    _POOL_BINDS.clear()
    # A promoted Guard's poll loop drains this, so recv() must be iterable.
    channel = Mock()
    channel.recv.return_value = []
    with patch(
        "kvcr.guard.ZmqPeerControlChannel.from_shared_listener",
        side_effect=_takes(channel),
    ):
        yield
    _POOL_BINDS.clear()


def _stand_in_pool(spec) -> Mock:
    """The pool-tail surface a Guard reaches for, without a real mapping."""
    attachment = Mock(address=1234, data_address=1234 + spec.journal_bytes, _spec=spec)
    attachment.mapped_snapshot.return_value = nullcontext(None)
    return attachment


def _new_registry(tmp_path: Path, pool_count: int = 1) -> _PoolRegistry:
    """A registry of real Guards over stand-in pool mappings.

    A pool's lifecycle lives in its Guard now, so the seam moved: these tests
    fake the mapping a Guard attaches, not the Guard itself.
    """
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


def test_release_preserves_the_physical_pool_spec(tmp_path: Path) -> None:
    registry = _new_registry(tmp_path, 2)
    first = _FakeLiveness()
    second = _FakeLiveness()
    try:
        first_spec, first_lease = _claim(registry, 0, first)
        registry.release(0, first_lease)
        second_spec, second_lease = _claim(registry, 0, second)
        registry.release(0, second_lease)

        assert first_spec is second_spec
        assert first.closed is True
        assert second.closed is True
    finally:
        registry.close()


def test_pools_are_claimed_independently_of_each_other(tmp_path: Path) -> None:
    """One pool's geometry says nothing about another's."""
    registry = _new_registry(tmp_path, pool_count=2)
    first = _FakeLiveness()
    second = _FakeLiveness()
    try:
        _claim(registry, 0, first)
        _spec, _fd, lease = registry.claim(
            1, _TierConfig(_TEST_ROW_STRIDE * 2, None), second, _control_bind(1)
        )
        os.close(_fd)
        del lease

        assert _holders_of(registry) == {0: first, 1: second}
    finally:
        registry.close()


def test_a_stale_lease_cannot_touch_a_replacement(tmp_path: Path) -> None:
    """Identity is the authority: an ended lease is a no-op forever after."""
    registry = _new_registry(tmp_path)
    first = _FakeLiveness()
    replacement = _FakeLiveness()
    try:
        _spec, stale = _claim(registry, 0, first)
        registry.release(0, stale)
        assert first.closed is True

        _spec, current = _claim(registry, 0, replacement)
        # The retried release of the old lease must not end the new one.
        registry.release(0, stale)

        assert registry._pools[0]._lease is current
        assert replacement.closed is False
    finally:
        registry.close()


def test_shutdown_releases_every_pool_and_unlinks_its_file(tmp_path: Path) -> None:
    liveness = _FakeLiveness()
    registry = _new_registry(tmp_path)
    pool_path = Path(registry._pools[0]._owner.spec.path)
    _claim(registry, 0, liveness)
    assert pool_path.exists()

    registry.close()

    # The pidfd is spent either way, so this must not interrupt the shutdown.
    assert registry._pools == {}
    assert _listeners_of(registry) == {}
    assert not pool_path.exists()


def test_an_endpoint_must_be_named_by_address(tmp_path: Path) -> None:
    """A name is refused where the claim is built, before it is sent."""
    with pytest.raises(ValueError, match="literal IPv4 address"):
        _claim_request(control_bind=("localhost", free_port()))


def test_invalid_g3_claim_does_not_bind(tmp_path: Path) -> None:
    with _running_server(tmp_path) as harness:
        request = msgspec.to_builtins(_claim_request())
        request["tier_config"]["g3"] = {
            "paths": ["relative"],
            "capacity_bytes_per_file": 8192,
            "backend": "FILE",
            "backend_options": {},
        }

        response = _send_raw_request(harness.server.socket_path, request)

        assert isinstance(response, _Error)
        harness.client.claim(
            0, _TEST_ROW_STRIDE, _TEST_DIGEST, _control_address()
        ).release()


def test_unbindable_control_endpoint_does_not_bind_the_pool(tmp_path: Path) -> None:
    # Held for the whole test: the address has to still be taken when the claim runs.
    with listening_socket() as squatter:
        taken = (str(squatter.getsockname()[0]), int(squatter.getsockname()[1]))
        with _running_server(tmp_path) as harness:
            with pytest.raises(KVCRServiceError, match="control listener"):
                harness.client.claim(
                    0,
                    _TEST_ROW_STRIDE,
                    _TEST_DIGEST,
                    control_bind=taken,
                )
            # A rejected claim must leave the pool free to take normally.
            harness.client.claim(
                0, _TEST_ROW_STRIDE, _TEST_DIGEST, _control_address()
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
    taken: list[tuple[int, int, object]] = []

    def _take_and_record(duplicate: socket.socket):
        # Recorded before it is closed, as the real one closes it too.
        taken.append((duplicate.family, duplicate.fileno(), duplicate.getsockname()))
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

        hold = harness.client.claim(
            1, _PAGE_STRIDE, _TEST_DIGEST, _control_address(), g3
        )
        assert guard._phase is _Phase.PRIMARY
        assert guard._control is control
        assert guard._configured.tier_config == _TierConfig(
            _PAGE_STRIDE,
            _G3Config(
                paths=(str(g3.paths[0]),),
                capacity_bytes_per_file=g3.capacity_bytes_per_file,
                backend=g3.backend,
                backend_options=dict(g3.backend_options),
            ),
        )
        (family, taken_fd, taken_name) = taken[0]
        assert family == socket.AF_INET
        # The Guard is handed a duplicate; the pool keeps the original.
        owned = guard._listener
        assert taken_fd != owned.fileno()
        assert taken_name == owned.getsockname()

        hold.release()

        assert guard._phase is _Phase.IDLE
        assert guard._lease is None
        # The adopted channel went with the lease; the endpoint did not.
        control.close.assert_called_once_with()
        assert guard._listener is not None


def test_a_claim_that_cannot_give_its_endpoint_back_is_fatal(
    tmp_path: Path,
) -> None:
    """A rollback that cannot prove the address is gone must stop the service."""
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    adopt_failure = RuntimeError("adopt failed")
    unbind_failure = OSError("close failed")
    uncontained: list[BaseException] = []
    registry.on_uncontained_failure = uncontained.append
    guard._adopt = Mock(side_effect=adopt_failure)
    real_bind = guard._bind_listener
    bound: list[socket.socket] = []

    def bind_a_listener_that_will_not_close(control_bind):
        listener = real_bind(control_bind)
        bound.append(listener)
        guard._listener = Mock(
            fileno=listener.fileno, close=Mock(side_effect=unbind_failure)
        )
        return guard._listener

    guard._bind_listener = bind_a_listener_that_will_not_close
    try:
        with pytest.raises(RuntimeError) as raised:
            _claim(registry, 0, _FakeLiveness())

        # The claim's own error, not the one its cleanup hit.
        assert raised.value is adopt_failure
        assert uncontained == [unbind_failure]
        # The transition ended: nothing is left reserved against this pool.
        assert guard._reserved is None
        assert guard._lease is None
    finally:
        guard._listener = None
        for listener in bound:
            listener.close()
        registry.close()


@pytest.mark.parametrize(
    "promotion_error",
    [None, RuntimeError("promotion failed")],
    ids=["promoted", "promotion_failed"],
)
def test_holder_death_retains_guard_and_pool_stays_busy(
    tmp_path: Path,
    promotion_error: RuntimeError | None,
) -> None:
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    uncontained: list[BaseException] = []
    registry.on_uncontained_failure = uncontained.append
    promote = Mock(side_effect=promotion_error)
    guard._promote = promote
    liveness = _FakeLiveness()
    try:
        _claim(registry, 0, liveness)

        _kill_and_wait(registry, 0, liveness)

        promote.assert_called_once_with()
        assert liveness.closed is True
        assert _holders_of(registry) == {}
        if promotion_error is None:
            assert uncontained == []
            assert guard._phase is _Phase.STANDBY
            # Claimable again: the pool waits for a replacement primary.
            replacement = _FakeLiveness()
            _spec, lease = _claim(registry, 0, replacement)
            assert guard._lease is lease
        else:
            # Promotion is fatal, on purpose: escalated, and the pool refuses.
            assert uncontained == [promotion_error]
            assert guard._phase is _Phase.FAILED
            with pytest.raises(RuntimeError, match="promotion failed"):
                _claim(registry, 0, _FakeLiveness())
    finally:
        registry.close()


def _claim_after_promotion(
    registry: _PoolRegistry, control_bind: tuple[str, int]
) -> _FakeLiveness:
    """Leave pool 0 standing by, with nobody holding it.

    The promotion itself is stubbed: what these tests exercise is the
    lifecycle around a standby, not the recovery machinery inside one.
    """
    guard = registry._pools[0]
    guard._promote = Mock()
    liveness = _FakeLiveness()
    _spec, _fd, _lease = registry.claim(0, _TEST_TIER_CONFIG, liveness, control_bind)
    os.close(_fd)
    _kill_and_wait(registry, 0, liveness)
    assert _holders_of(registry) == {}
    return liveness


def test_a_standby_pool_hands_itself_to_a_replacement(tmp_path: Path) -> None:
    control_bind = _control_bind(0)
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    try:
        _claim_after_promotion(registry, control_bind)
        replacement = _FakeLiveness()

        _spec, lease = _claim(registry, 0, replacement, control_bind)

        assert guard._phase is _Phase.PRIMARY
        assert guard._lease is lease
        # The same endpoint, never rebound: the replacement inherits it.
        assert guard._listener.getsockname() == (control_bind[0], control_bind[1])
    finally:
        registry.close()


def test_a_claim_that_fails_before_the_hand_over_leaves_the_guard_serving(
    tmp_path: Path,
) -> None:
    """Only a hand-over that actually began costs the standby its endpoint."""
    control_bind = _control_bind(0)
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    try:
        _claim_after_promotion(registry, control_bind)
        with patch(
            "kvcr.guard.ZmqPeerControlChannel.from_shared_listener",
            side_effect=ValueError("cannot share this listener"),
        ):
            with pytest.raises(KVCRServiceError, match="cannot share this listener"):
                _claim(registry, 0, _FakeLiveness(), control_bind)

        assert guard._phase is _Phase.STANDBY
        assert guard._reserved is None
        assert guard._listener is not None
    finally:
        registry.close()


def test_a_guard_that_cannot_adopt_a_replacement_fails_the_claim(
    tmp_path: Path,
) -> None:
    """A refused claim costs the pool nothing -- not its standby, not its address."""
    control_bind = _control_bind(0)
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    failure = RuntimeError("adopt failed")
    try:
        _claim_after_promotion(registry, control_bind)
        real_adopt = guard._adopt
        guard._adopt = Mock(side_effect=failure)
        with pytest.raises(RuntimeError) as raised:
            _claim(registry, 0, _FakeLiveness(), control_bind)
        assert raised.value is failure
        assert guard._phase is _Phase.STANDBY
        assert _holders_of(registry) == {}
        assert guard._listener is not None

        # The failure fenced the claim rather than the pool.
        guard._adopt = real_adopt
        replacement = _FakeLiveness()
        _spec, lease = _claim(registry, 0, replacement, control_bind)
        assert guard._lease is lease
    finally:
        registry.close()


def test_a_guard_will_not_hand_a_pool_to_differently_configured_tiers(
    tmp_path: Path,
) -> None:
    control_bind = _control_bind(0)
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    try:
        _claim_after_promotion(registry, control_bind)
        # Tiers are fixed by the first claim; a mismatched replacement is refused
        # by the real predicate, and the standby keeps serving.
        with pytest.raises(KVCRServiceError, match="another tier configuration"):
            registry.claim(
                0,
                _TierConfig(_TEST_ROW_STRIDE * 2, None),
                _FakeLiveness(),
                control_bind,
            )
        assert guard._phase is _Phase.STANDBY
        assert _holders_of(registry) == {}
    finally:
        registry.close()


def test_dead_guarded_holder_stays_busy_until_the_actor_promotes(
    tmp_path: Path,
) -> None:
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    guard._promote = Mock()
    # Hold the actor's own observation back, so the dead-but-not-yet-promoted
    # window is stable enough to assert against.
    guard._observe_holder = lambda: None
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    first = PidfdLiveness(os.pidfd_open(child.pid))
    try:
        _claim(registry, 0, first)
        child.terminate()
        child.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)

        # Dead, but the actor has not promoted: busy, not "held".
        with pytest.raises(KVCRServiceError, match="busy"):
            _claim(registry, 0, _FakeLiveness())
        assert first.fileno() >= 0
        guard._promote.assert_not_called()

        del guard._observe_holder
        _await_phase(registry._pools[0], lambda g: g._phase is _Phase.STANDBY)

        guard._promote.assert_called_once_with()
        assert _holders_of(registry) == {}
        with pytest.raises(ValueError, match="pidfd is closed"):
            first.fileno()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        first.close()
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

        # Small but not zero: the deadline is absolute, so the healthy pool's
        # actor finishes well inside it while the wedged one burns it.
        monkeypatch.setattr(
            "kvcr.kvcr_service._REGISTRY_TRANSITION_TIMEOUT_SECONDS", 1.0
        )
        with pytest.raises(TimeoutError, match="pool transitions"):
            registry.close()
        # The wedged pool is kept and named; its neighbour still went.
        assert registry._pools.keys() == {0}
        assert second.closed is True
    finally:
        continue_promotion.set()
        monkeypatch.undo()
        registry.close()

    assert first.closed is True


def test_shutdown_leaks_only_the_pool_it_cannot_release(tmp_path: Path) -> None:
    """One pool that will not go keeps its own file, and nobody else's."""
    failure = RuntimeError("close failed")
    registry = _new_registry(tmp_path, pool_count=2)
    stuck = registry._pools[0]
    stuck._close_resources = Mock(side_effect=failure)
    first = _FakeLiveness()
    second = _FakeLiveness()
    _claim(registry, 0, first)
    _claim(registry, 1, second)
    first_path = Path(registry._pools[0]._owner.spec.path)
    second_path = Path(registry._pools[1]._owner.spec.path)

    with pytest.raises(RuntimeError, match="close failed") as raised:
        registry.close()

    assert raised.value is failure
    # The stuck pool keeps its file and lease; the one behind it still goes.
    assert registry._pools.keys() == {0}
    assert first.closed is False
    assert first_path.exists()
    assert second.closed is True
    assert not second_path.exists()

    # Clean up the pool this test deliberately leaves behind.
    type(stuck)._close_resources(stuck)


def test_claim_refusals_and_internal_failures_do_not_bind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _running_server(tmp_path) as harness:
        spec = harness.server._registry._pools[0]._owner.spec
        with pytest.raises(KVCRServiceError, match="compatibility digest"):
            harness.client.claim(
                0, _TEST_ROW_STRIDE, _TEST_DIGEST.swapcase(), _control_address()
            )
        with pytest.raises(KVCRServiceError, match="one complete KV row"):
            harness.client.claim(
                0, _TEST_POOL_SIZE_BYTES + 1, _TEST_DIGEST, _control_address()
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
                    0, _TEST_ROW_STRIDE, _TEST_DIGEST, _control_address()
                )

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
            0, _TEST_ROW_STRIDE, _TEST_DIGEST, _control_address()
        ).release()


def test_live_holder_makes_a_second_claim_busy(tmp_path: Path) -> None:
    with _running_server(tmp_path) as harness:
        hold = harness.client.claim(
            0, _TEST_ROW_STRIDE, _TEST_DIGEST, _control_address()
        )
        try:
            with pytest.raises(KVCRServiceError, match="held"):
                harness.client.claim(
                    0, _TEST_ROW_STRIDE, _TEST_DIGEST, _control_address()
                )
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
            assert 0 in _holders_of(harness.server._registry)

            channel.send({"type": "release", "version": 1, "future": True})
            response = channel.receive(_RELEASE_RESPONSE_DECODER)
            assert response == _Released(1)
            assert _holders_of(harness.server._registry) == {}


def test_a_release_racing_a_death_is_stale_after_promotion(tmp_path: Path) -> None:
    """Whichever ending reserves first wins; the loser is a no-op forever."""
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    guard._promote = Mock()
    liveness = _FakeLiveness()
    try:
        _spec, lease = _claim(registry, 0, liveness)
        _kill_and_wait(registry, 0, liveness)

        # The claimant's release arrives after its death already promoted.
        registry.release(0, lease)

        assert guard._phase is _Phase.STANDBY
        guard._promote.assert_called_once_with()
    finally:
        registry.close()


def test_the_first_fatal_failure_is_the_one_reported() -> None:
    """Shutdown ends other pools' work, and those endings fail too."""
    server = object.__new__(_ThreadingUnixServer)
    server._fatal_error = None
    server._fatal_lock = threading.Lock()
    server.registry = Mock()
    server.shutdown = Mock()
    first = RuntimeError("Guard failed")

    server.fail(first)
    server.fail(RuntimeError("and then the shutdown did too"))

    assert server._fatal_error is first


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
                f"0, {_TEST_ROW_STRIDE}, {_TEST_DIGEST!r}, {_control_address()!r})",
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
                harness.client.claim(
                    0, _TEST_ROW_STRIDE, _TEST_DIGEST, _control_address()
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


def test_a_guard_that_cannot_prepare_gives_its_pool_back(tmp_path: Path) -> None:
    """The rank that failed is rolled back too, not just the ones before it."""
    attached: list[object] = []

    def attach(spec):
        if attached:
            raise RuntimeError("attach failed")
        pool = _stand_in_pool(spec)
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
            tmp_path,
            2,
            _TEST_POOL_SIZE_BYTES,
            _TEST_JOURNAL_BYTES,
            _TEST_DIGEST,
        )

    # Both pool files are gone: the one that prepared and the one that failed.
    assert list(tmp_path.iterdir()) == []
    attached[0].close.assert_called_once_with()


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


def test_a_failed_grant_delivery_retracts_through_the_guard(tmp_path: Path) -> None:
    """A held undelivered grant is retracted by the claimant's own word: an
    unactivated release routes through abort_grant, and the pool is free."""
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
        handler.handle()
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
    """EOF is not a release: a claimant that mapped the pool may have dropped
    its connection while alive, so only its word or its death frees the lease."""
    accepted, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    registry = _new_registry(tmp_path)
    # Promotion's serving core is not under test; the freed lease is.
    registry._pools[0]._promote = Mock()

    class _Server:
        def __init__(self) -> None:
            self.registry = registry
            self.compatibility_digest = _TEST_DIGEST

        def dispatch(self, request, liveness):
            return _ThreadingUnixServer.dispatch(self, request, liveness)

    monkeypatch.setattr(
        PidfdLiveness,
        "from_peer_socket",
        classmethod(lambda _cls, _sock: PidfdLiveness(os.pidfd_open(child.pid))),
    )
    handler = object.__new__(_RequestHandler)
    handler.request = accepted
    handler.server = _Server()
    handler.channel = Mock()
    handler.channel.receive.side_effect = [_claim_request(), EOFError()]
    handler.channel.send_with_fd.side_effect = RuntimeError("undelivered")
    try:
        # Delivery fails and the connection then EOFs; handle() returns with
        # the lease still held.
        handler.handle()
        rival = _FakeLiveness()
        with pytest.raises(KVCRServiceError, match="held by another worker"):
            _claim(registry, 0, rival)

        child.kill()
        child.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        # Death freed it: the pool's own actor promotes, and the lease is gone.
        _await_phase(registry._pools[0], lambda g: g._phase is _Phase.STANDBY)
        assert _holders_of(registry) == {}
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        peer.close()
        accepted.close()
        registry.close()


def test_a_rollback_that_cannot_stop_a_guard_leaves_its_pool_in_place(
    tmp_path: Path,
) -> None:
    """Unlinking under a Guard that would not close would fault the mapping."""
    wedged, failing = Mock(), Mock()
    wedged.close.side_effect = RuntimeError("still holding the mapping")
    failing.start.side_effect = RuntimeError("attach failed")

    def build(spec, _callback, *, pool_index, owner, **_kwargs):
        guard = wedged if pool_index == 0 else failing
        guard._spec = spec
        if guard is failing:
            # A Guard that closes gives its pool back with everything else.
            failing.close.side_effect = owner.close
        return guard

    with (
        patch("kvcr.kvcr_service._Guard", side_effect=build),
        pytest.raises(RuntimeError, match="attach failed"),
    ):
        _PoolRegistry(
            tmp_path,
            2,
            _TEST_POOL_SIZE_BYTES,
            _TEST_JOURNAL_BYTES,
            _TEST_DIGEST,
        )

    # The wedged pool's file is left for the next start's purge; the pool
    # whose Guard closed is gone with its owner.
    assert len(list(tmp_path.iterdir())) == 1


def test_promotion_failure_stops_the_whole_service(tmp_path: Path) -> None:
    """A pool that cannot be promoted is fatal, on purpose."""
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
        assert registry.is_closed() is True
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
            0, _TEST_ROW_STRIDE, _TEST_DIGEST, _control_address()
        )
        time.sleep(0.5)

        hold.release()

        guard = harness.server._registry._pools[0]
        assert guard._lease is None


def test_refuse_claims_returns_only_after_inflight_commits_are_decided(
    tmp_path: Path,
) -> None:
    """The barrier: refuse_claims passes every pool's phase lock on its way out."""
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    returned = threading.Event()
    with guard._phase_lock:
        refuser = threading.Thread(
            target=lambda: (registry.refuse_claims(), returned.set())
        )
        refuser.start()
        # Held by us, exactly as a commit in flight would hold it.
        assert not returned.wait(timeout=0.3)
    assert returned.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
    refuser.join(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
    registry.close()


def test_a_release_that_times_out_reports_the_timeout(tmp_path: Path) -> None:
    """A handler's own TimeoutError is the answer, not a wait to keep waiting."""
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    liveness = _FakeLiveness()
    try:
        _spec, lease = _claim(registry, 0, liveness)
        guard._release = Mock(side_effect=TimeoutError("hand-back timed out"))

        started = time.monotonic()
        with pytest.raises(TimeoutError, match="hand-back timed out"):
            registry.release(0, lease)

        assert time.monotonic() - started < 2
    finally:
        registry.close()


def test_a_dup_failure_on_a_fresh_bind_does_not_pin_the_endpoint(
    tmp_path: Path,
) -> None:
    """A claim that bound the address and then failed must give it back."""
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    try:
        with patch("kvcr.guard.os.dup", side_effect=OSError("dup failed")):
            with pytest.raises(OSError, match="dup failed"):
                registry.claim(0, _TEST_TIER_CONFIG, _FakeLiveness(), _control_bind(0))
        assert guard._listener is None

        # In particular, a retry on a DIFFERENT address must be honoured.
        replacement = _FakeLiveness()
        other_bind = _control_bind(0)
        _spec, lease = _claim(registry, 0, replacement, other_bind)
        assert guard._bind == other_bind
        assert guard._lease is lease
    finally:
        registry.close()


def test_a_listener_that_will_not_close_stays_visible_on_the_kept_pool(
    tmp_path: Path,
) -> None:
    """A kept pool must still name what it leaked, or nobody can ever retry."""
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    _claim(registry, 0, _FakeLiveness())
    real_listener = guard._listener
    stubborn = Mock(
        fileno=real_listener.fileno, close=Mock(side_effect=OSError("will not close"))
    )
    guard._listener = stubborn
    try:
        with pytest.raises(OSError, match="will not close"):
            registry.close()

        # The pool is kept, and the resource that failed is still reachable.
        assert registry._pools.keys() == {0}
        assert registry._pools[0]._listener is stubborn
    finally:
        guard._listener = real_listener
        type(guard)._close_resources(guard)


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
            _await_phase(guard, lambda g: g._phase is _Phase.FAILED)

        guard._promote.assert_not_called()
        assert len(uncontained) == 1
        assert isinstance(uncontained[0], OSError)
        assert "without POLLIN" in str(uncontained[0])
    finally:
        registry.close()


def test_a_release_racing_shutdown_is_absorbed(tmp_path: Path) -> None:
    """Once close has begun, a release is the close's problem, not a new fatal."""
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

        # A death in the same window is the close's to clean up, not a
        # promotion: the pool is being torn down, not failed over.
        liveness.kill()

        # And a claim in the same window is refused with a typed error.
        with pytest.raises(KVCRServiceError, match="closed"):
            registry.claim(0, _TEST_TIER_CONFIG, _FakeLiveness(), _control_bind(0))
    finally:
        gate.set()
        registry.close()
    guard._promote.assert_not_called()
    assert liveness.closed is True


def test_a_command_queued_behind_close_is_answered_not_abandoned(
    tmp_path: Path,
) -> None:
    """Every exit of the actor loop answers what is still queued."""
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    gate = threading.Event()
    entered = threading.Event()
    real_close_resources = guard._close_resources

    def slow_close() -> None:
        entered.set()
        assert gate.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        real_close_resources()

    guard._close_resources = slow_close
    guard.begin_close()
    assert entered.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
    # A raw command that slipped in behind the close, bypassing every gate.
    stray = _Command("release", (object(),))
    guard._commands.put(stray)
    gate.set()

    assert guard.finish_close(time.monotonic() + _SERVER_STOP_TIMEOUT_SECONDS) is False
    assert stray.future.done()
    assert isinstance(stray.future.exception(), KVCRServiceError)
    registry.close()


def test_a_grant_that_cannot_dup_its_endpoint_leaves_the_pool_claimable(
    tmp_path: Path,
) -> None:
    """Everything fallible runs before the lease exists."""
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    liveness = _FakeLiveness()
    try:
        with patch("kvcr.guard.os.dup", side_effect=OSError("dup failed")):
            with pytest.raises(OSError, match="dup failed"):
                registry.claim(0, _TEST_TIER_CONFIG, liveness, _control_bind(0))

        assert guard._lease is None
        assert guard._phase is not _Phase.PRIMARY
        assert guard._reserved is None

        # The pool survived its failed grant: the next claim simply works.
        replacement = _FakeLiveness()
        _spec, lease = _claim(registry, 0, replacement)
        assert guard._lease is lease
    finally:
        registry.close()


def test_no_grant_is_produced_after_refuse_claims_returns(tmp_path: Path) -> None:
    """The grant commits under the same lock the refusal is read under."""
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    in_adopt = threading.Event()
    resume = threading.Event()
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
    claimant.start()
    try:
        assert in_adopt.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        registry.refuse_claims()
        # refuse_claims has RETURNED; the claim past every earlier check must
        # still be refused at its commit.
        resume.set()
        claimant.join(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        assert not claimant.is_alive()

        assert "grant" not in outcome
        assert isinstance(outcome["error"], KVCRServiceError)
        assert guard._lease is None
        assert guard._reserved is None
    finally:
        resume.set()
        registry.close()


def test_shutdown_drains_a_held_connection(tmp_path: Path) -> None:
    with _running_server(tmp_path) as harness:
        hold = harness.client.claim(
            0, _TEST_ROW_STRIDE, _TEST_DIGEST, _control_address()
        )
        harness.stop()
        _wait_for_connection_state(harness.server, connected=False)
        hold._attachment.close()
        hold._connection.close()


def test_a_pool_no_bigger_than_its_journal_is_rejected_at_the_flag() -> None:
    """Sized at or below the journal, a pool has nothing left to cache with."""

    def parse(pool_size_gb: str) -> argparse.Namespace:
        return _parse_args(
            [
                "--socket-path",
                "/run/kvcr/memory.sock",
                "--pool-dir",
                "/dev/shm/kvcr",
                "--pool-count",
                "1",
                "--compatibility-digest",
                "digest",
                "--pool-size-gb",
                pool_size_gb,
            ]
        )

    with pytest.raises(SystemExit):
        parse("0.05")
    assert parse("0.2").pool_size_bytes > 100 * (1 << 20)


def test_a_failed_release_refuses_claims_before_it_frees_the_pool(
    tmp_path: Path,
) -> None:
    """The window between a failed hand-back and the shutdown it causes."""
    registry = _new_registry(tmp_path)
    guard = registry._pools[0]
    failure = RuntimeError("hand-back failed")
    guard._release = Mock(side_effect=failure)
    claimable_when_reported: list[bool] = []
    registry.on_uncontained_failure = lambda _error: claimable_when_reported.append(
        guard._lease is None and guard._failure is None
    )
    liveness = _FakeLiveness()
    try:
        _spec, lease = _claim(registry, 0, liveness)

        with pytest.raises(RuntimeError) as raised:
            registry.release(0, lease)

        assert raised.value is failure
        # Reported while the pool was already fenced, so nothing could claim it.
        assert claimable_when_reported == [False]
        assert guard._phase is _Phase.FAILED
        with pytest.raises(RuntimeError, match="hand-back failed"):
            _claim(registry, 0, _FakeLiveness())
    finally:
        registry.close()
