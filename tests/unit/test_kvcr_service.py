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
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import msgspec
import pytest
from _kvcr_test_utils import free_port, listening_socket

from kvcr import KVCRClient, KVCRServiceError
from kvcr.config import G3Options
from kvcr.control_channels import FramedConnection
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
from kvcr.recovery_journal import RecoveryMirrorError


def _holders_of(registry) -> dict[int, object]:
    """The pools a worker holds, in the shape the old binding map had."""
    return {i: p.holder for i, p in registry._pools.items() if p.holder is not None}


def _listeners_of(registry) -> dict[int, object]:
    return {i: p.listener for i, p in registry._pools.items() if p.listener is not None}


_SERVER_STOP_TIMEOUT_SECONDS = 5
_CONNECTION_POLL_INTERVAL_SECONDS = 0.001

_TEST_POOL_COUNT = 2
_TEST_JOURNAL_BYTES = 8192
_TEST_POOL_SIZE_BYTES = _TEST_JOURNAL_BYTES + 8192
_TEST_ROW_STRIDE = 1024
_TEST_DIGEST = "opaque digest: Preserve-Me EXACTLY"
_TEST_TIER_CONFIG = _TierConfig(_TEST_ROW_STRIDE, None)


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
        "kvcr.kvcr_service.ZmqPeerControlChannel.from_shared_listener",
        side_effect=_takes(channel),
    ):
        yield
    _POOL_BINDS.clear()


def _new_registry(
    tmp_path: Path,
    pool_count: int = 1,
    guards: list | None = None,
) -> _PoolRegistry:
    """A registry whose pools come with stand-in Guards."""
    supplied = list(guards or ())

    def build(*_args, **_kwargs):
        return supplied.pop(0) if supplied else Mock()

    with patch("kvcr.kvcr_service._Guard", side_effect=build):
        return _PoolRegistry(
            tmp_path,
            pool_count,
            _TEST_POOL_SIZE_BYTES,
            _TEST_JOURNAL_BYTES,
            _TEST_DIGEST,
        )


def _assert_registry_lock_available(registry: _PoolRegistry) -> None:
    assert registry._lock.acquire(blocking=False)
    registry._lock.release()


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
        first_spec = registry.claim(0, _TEST_TIER_CONFIG, first, _control_bind(0))
        registry.release(0, first)
        second_spec = registry.claim(0, _TEST_TIER_CONFIG, second, _control_bind(0))
        registry.release(0, second)

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
        registry.claim(0, _TEST_TIER_CONFIG, first, _control_bind(0))
        registry.claim(
            1, _TierConfig(_TEST_ROW_STRIDE * 2, None), second, _control_bind(1)
        )

        assert _holders_of(registry) == {0: first, 1: second}
    finally:
        registry.close()


@pytest.mark.parametrize("action", ["release", "holder_died"])
def test_stale_liveness_close_failure_does_not_affect_replacement(
    tmp_path: Path,
    action: str,
) -> None:
    registry = _new_registry(tmp_path)
    replacement = _FakeLiveness()
    stale = Mock()
    try:
        registry.claim(0, _TEST_TIER_CONFIG, replacement, _control_bind(0))

        getattr(registry, action)(0, stale)

        assert registry._pools[0].holder is replacement
        assert replacement.closed is False
        stale.close.assert_called_once_with()
    finally:
        registry.close()


def test_shutdown_releases_every_pool_and_unlinks_its_file(tmp_path: Path) -> None:
    liveness = Mock()
    registry = _new_registry(tmp_path)
    pool_path = Path(registry._pools[0].owner.spec.path)
    registry.claim(0, _TEST_TIER_CONFIG, liveness, _control_bind(0))
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
    """The Guard is the pool's, so a grant tells it and a release hands it back."""
    guard = Mock()
    control = Mock()
    taken: list[tuple[int, int, object]] = []

    def _take_and_record(duplicate: socket.socket):
        # Recorded before it is closed, as the real one closes it too.
        taken.append((duplicate.family, duplicate.fileno(), duplicate.getsockname()))
        duplicate.close()
        return control

    g3 = G3Options(
        paths=((tmp_path / "g3").resolve(),),
        capacity_bytes_per_file=8192,
        backend="FILE",
        backend_options={"mode": "direct"},
    )
    with (
        patch("kvcr.kvcr_service._Guard", return_value=guard),
        patch(
            "kvcr.kvcr_service.ZmqPeerControlChannel.from_shared_listener",
            side_effect=_take_and_record,
        ),
        _running_server(tmp_path) as harness,
    ):
        # Built and started with the pool, before any claim named it.
        guard.start.assert_called_with()
        guard.adopt.assert_not_called()

        hold = harness.client.claim(
            1, _TEST_ROW_STRIDE, _TEST_DIGEST, _control_address(), g3
        )
        assert guard.adopt.call_args.args == (
            control,
            _TierConfig(
                _TEST_ROW_STRIDE,
                _G3Config(
                    paths=(str(g3.paths[0]),),
                    capacity_bytes_per_file=g3.capacity_bytes_per_file,
                    backend=g3.backend,
                    backend_options=dict(g3.backend_options),
                ),
            ),
        )
        (family, taken_fd, taken_name) = taken[0]
        assert family == socket.AF_INET
        # The Guard is handed a duplicate; the service keeps the original.
        owned = harness.server._registry._pools[1].listener
        assert taken_fd != owned.fileno()
        assert taken_name == owned.getsockname()

        hold.release()

        guard.release.assert_called_once_with()
        guard.close.assert_not_called()
        guard.promote_after_death.assert_not_called()
        assert _holders_of(harness.server._registry) == {}
        # Both stay with the pool: the primary left, the pool did not.
        assert harness.server._registry._pools[1].guard is guard
        assert harness.server._registry._pools[1].listener is not None


def test_a_guard_failure_stops_the_service(tmp_path: Path) -> None:
    """A pool without its standby is a pool the service cannot honour."""
    guard = Mock()
    liveness = _FakeLiveness()
    registry = _new_registry(tmp_path, guards=[guard])
    uncontained: list[BaseException] = []
    registry.on_uncontained_failure = uncontained.append
    try:
        registry.claim(0, _TEST_TIER_CONFIG, liveness, _control_bind(0))
        failure = RuntimeError("core close failed")

        registry._guard_failed(0, guard, failure)

        assert uncontained == [failure]
        # Closing the registry's copy would not reach the Guard's duplicate.
        assert registry._pools[0].listener is not None
    finally:
        registry.close()


def test_a_claim_that_cannot_give_its_endpoint_back_frees_the_pool_and_is_fatal(
    tmp_path: Path,
) -> None:
    """A rollback that cannot prove the address is gone still ends the transition."""
    guard = Mock()
    adopt_failure = RuntimeError("adopt failed")
    guard.adopt.side_effect = adopt_failure
    unbind_failure = OSError("close failed")
    registry = _new_registry(tmp_path, guards=[guard])
    uncontained: list[tuple[BaseException, bool, bool]] = []

    def report(error: BaseException) -> None:
        # Reported from outside the registry lock, which refuse_claims retakes.
        unlocked = registry._condition.acquire(blocking=False)
        if unlocked:
            registry._condition.release()
        uncontained.append((error, registry._closed, unlocked))

    registry.on_uncontained_failure = report
    bind_locked = registry._bind_control_listener_locked
    bound: list[socket.socket] = []

    def bind_a_listener_that_will_not_close(pool, pool_index, control_bind):  # type: ignore[no-untyped-def]
        listener = bind_locked(pool, pool_index, control_bind)
        bound.append(listener)
        pool.listener = Mock(
            fileno=listener.fileno, close=Mock(side_effect=unbind_failure)
        )
        return pool.listener

    try:
        with patch.object(
            registry,
            "_bind_control_listener_locked",
            bind_a_listener_that_will_not_close,
        ):
            with pytest.raises(RuntimeError) as raised:
                registry.claim(0, _TEST_TIER_CONFIG, _FakeLiveness(), _control_bind(0))

        # The claim's own error, not the one its cleanup hit.
        assert raised.value is adopt_failure
        # Refused before the pool was let go, so nothing could claim the address
        # this failed to unbind.
        assert uncontained == [(unbind_failure, True, True)]
        assert not registry._pools[0].in_transition
    finally:
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
    guard = Mock()
    guard.promote_after_death.side_effect = promotion_error
    control = Mock()
    guard.close.side_effect = control.close
    liveness = _FakeLiveness()
    registry = _new_registry(tmp_path, guards=[guard])
    try:
        registry.claim(
            0,
            _TEST_TIER_CONFIG,
            liveness,
            _control_bind(0),
        )

        if promotion_error is None:
            registry.holder_died(0, liveness)
        else:
            # The watcher routes this to server.fail(): promotion is fatal.
            with pytest.raises(RuntimeError) as raised:
                registry.holder_died(0, liveness)
            assert raised.value is promotion_error

        guard.promote_after_death.assert_called_once_with()
        assert liveness.closed is True
        assert _holders_of(registry) == {}
        assert registry._pools[0].guard is guard
        # Claimable again: the Guard is now waiting for a replacement primary.
        replacement = _FakeLiveness()
        registry.claim(0, _TEST_TIER_CONFIG, replacement, _control_bind(0))
        assert registry._pools[0].holder is replacement
    finally:
        registry.close()


def _claim_after_promotion(
    registry: _PoolRegistry, guard: Mock, control_bind: tuple[str, int]
) -> _FakeLiveness:
    """Leave one Guard serving pool 0 with nobody holding it."""
    liveness = _FakeLiveness()
    registry.claim(0, _TEST_TIER_CONFIG, liveness, control_bind)
    registry.holder_died(0, liveness)
    assert _holders_of(registry) == {}
    return liveness


def test_a_guard_serving_alone_mirrors_for_a_guarded_claimant(tmp_path: Path) -> None:
    serving = Mock()
    control_bind = _control_bind(0)
    registry = _new_registry(tmp_path, guards=[serving])
    try:
        _claim_after_promotion(registry, serving, control_bind)
        replacement = _FakeLiveness()
        # Constructing a second Guard would rebuild records this one still has.
        with patch("kvcr.kvcr_service._Guard", side_effect=AssertionError):
            _ = registry.claim(0, _TEST_TIER_CONFIG, replacement, control_bind)
        # Its serving core closed the channel, so it adopts onto a fresh one.
        assert serving.adopt.call_count == 2
        assert registry._pools[0].listener.getsockname() == (
            control_bind[0],
            control_bind[1],
        )
        serving.close.assert_not_called()
        assert registry._pools[0].guard is serving
        assert registry._pools[0].holder is replacement
    finally:
        registry.close()


def test_a_claim_that_fails_before_the_hand_over_leaves_the_guard_serving(
    tmp_path: Path,
) -> None:
    """Only a hand-over that actually began costs the Guard its endpoint."""
    serving = Mock()
    control_bind = _control_bind(0)
    registry = _new_registry(tmp_path, guards=[serving])
    try:
        _claim_after_promotion(registry, serving, control_bind)
        with patch(
            "kvcr.kvcr_service.ZmqPeerControlChannel.from_shared_listener",
            side_effect=ValueError("cannot share this listener"),
        ):
            with pytest.raises(KVCRServiceError, match="cannot share this listener"):
                registry.claim(0, _TEST_TIER_CONFIG, _FakeLiveness(), control_bind)

        serving.close.assert_not_called()
        assert registry._pools[0].guard is serving
        assert registry._pools[0].listener is not None
    finally:
        registry.close()


def test_a_guard_that_cannot_adopt_a_replacement_fails_the_claim(
    tmp_path: Path,
) -> None:
    """A refused claim costs the pool nothing -- not its Guard, not its address."""
    serving = Mock()
    failure = RuntimeError("adopt failed")
    control_bind = _control_bind(0)
    registry = _new_registry(tmp_path, guards=[serving])
    try:
        _claim_after_promotion(registry, serving, control_bind)
        serving.adopt.side_effect = failure
        with pytest.raises(RuntimeError) as raised:
            registry.claim(0, _TEST_TIER_CONFIG, _FakeLiveness(), control_bind)
        assert raised.value is failure
        serving.close.assert_not_called()
        assert registry._pools[0].guard is serving
        assert _holders_of(registry) == {}
        assert registry._pools[0].listener is not None

        # The failure fenced the claim rather than the pool.
        serving.adopt.side_effect = None
        replacement = _FakeLiveness()
        registry.claim(0, _TEST_TIER_CONFIG, replacement, control_bind)
        assert registry._pools[0].holder is replacement
    finally:
        registry.close()


def test_a_guard_will_not_hand_a_pool_to_differently_configured_tiers(
    tmp_path: Path,
) -> None:
    serving = Mock()
    control_bind = _control_bind(0)
    registry = _new_registry(tmp_path, guards=[serving])
    try:
        _claim_after_promotion(registry, serving, control_bind)
        serving.adopt.side_effect = RecoveryMirrorError(
            "KVCR pool holds recovered state for different tiers"
        )
        # Tiers are fixed at build time; a mismatched claimant could not promote.
        with pytest.raises(KVCRServiceError, match="different tiers"):
            registry.claim(0, _TEST_TIER_CONFIG, _FakeLiveness(), control_bind)
        serving.release.assert_not_called()
        assert registry._pools[0].guard is serving
        assert _holders_of(registry) == {}
    finally:
        registry.close()


def test_dead_guarded_holder_stays_busy_until_watcher_promotes(
    tmp_path: Path,
) -> None:
    guard = Mock()
    control = Mock()
    guard.close.side_effect = control.close
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    first = PidfdLiveness(os.pidfd_open(child.pid))
    replacement = _FakeLiveness()
    registry = _new_registry(tmp_path, guards=[guard])
    try:
        registry.claim(0, _TEST_TIER_CONFIG, first, _control_bind(0))
        child.terminate()
        child.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)

        with pytest.raises(KVCRServiceError, match="busy"):
            registry.claim(0, _TEST_TIER_CONFIG, replacement, _control_bind(0))
        assert first.fileno() >= 0
        guard.promote_after_death.assert_not_called()
        registry.holder_died(0, first)

        guard.promote_after_death.assert_called_once_with()
        assert _holders_of(registry) == {}
        assert registry._pools[0].guard is guard
        with pytest.raises(ValueError, match="pidfd is closed"):
            first.fileno()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        first.close()
        replacement.close()
        registry.close()


def test_slow_promotion_does_not_block_other_pools_or_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion_started = threading.Event()
    continue_promotion = threading.Event()
    promotion_errors: list[BaseException] = []
    claim_done = threading.Event()
    claim_errors: list[BaseException] = []
    guard = Mock()
    first = _FakeLiveness()
    second = _FakeLiveness()
    registry = _new_registry(tmp_path, pool_count=2, guards=[guard])
    promotion_thread: threading.Thread | None = None
    claim_thread: threading.Thread | None = None

    def promote() -> None:
        _assert_registry_lock_available(registry)
        promotion_started.set()
        assert continue_promotion.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)

    guard.promote_after_death.side_effect = promote
    try:
        registry.claim(0, _TEST_TIER_CONFIG, first, _control_bind(0))

        def holder_died() -> None:
            try:
                registry.holder_died(0, first)
            except BaseException as error:
                promotion_errors.append(error)

        promotion_thread = threading.Thread(target=holder_died)
        promotion_thread.start()
        assert promotion_started.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)

        def claim_other_pool() -> None:
            try:
                registry.claim(1, _TEST_TIER_CONFIG, second, _control_bind(1))
            except BaseException as error:
                claim_errors.append(error)
            finally:
                claim_done.set()

        claim_thread = threading.Thread(target=claim_other_pool)
        claim_thread.start()
        assert claim_done.wait(timeout=1)
        claim_thread.join(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        assert not claim_thread.is_alive()
        assert claim_errors == []
        with pytest.raises(KVCRServiceError, match="busy"):
            registry.claim(0, _TEST_TIER_CONFIG, _FakeLiveness(), _control_bind(0))

        monkeypatch.setattr("kvcr.kvcr_service._REGISTRY_TRANSITION_TIMEOUT_SECONDS", 0)
        with pytest.raises(TimeoutError, match="pool transitions"):
            registry.close()
        assert registry._pools[0].guard is guard
        assert registry._pools[0].holder is first
        assert registry._pools.keys() == {0}
        assert second.closed is True
    finally:
        continue_promotion.set()
        for thread in (claim_thread, promotion_thread):
            if thread is not None:
                thread.join(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
                assert not thread.is_alive()
        registry.close()

    assert promotion_errors == []
    assert first.closed is True


def test_shutdown_leaks_only_the_pool_it_cannot_release(tmp_path: Path) -> None:
    """One pool that will not go keeps its own file, and nobody else's."""
    failure = RuntimeError("close failed")
    guard = Mock()
    guard.close.side_effect = failure
    first = _FakeLiveness()
    second = _FakeLiveness()
    registry = _new_registry(tmp_path, pool_count=2, guards=[guard])
    registry.claim(0, _TEST_TIER_CONFIG, first, _control_bind(0))
    registry.claim(1, _TEST_TIER_CONFIG, second, _control_bind(1))
    first_path = Path(registry._pools[0].owner.spec.path)
    second_path = Path(registry._pools[1].owner.spec.path)

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
    pool = registry._pools[0]
    if pool.listener is not None:
        pool.listener.close()
    pool.owner.close()


def test_claim_refusals_and_internal_failures_do_not_bind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _running_server(tmp_path) as harness:
        spec = harness.server._registry._pools[0].owner.spec
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
        assert harness.server._registry._pools[0].owner.spec is spec
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


def test_simultaneous_release_and_death_uses_the_valid_release() -> None:
    socket_fd = 10
    pidfd = 11
    poller = Mock()
    poller.poll.return_value = [
        (pidfd, select.POLLIN),
        (socket_fd, select.POLLIN),
    ]
    liveness = Mock()
    liveness.fileno.return_value = pidfd
    registry = Mock()
    handler = object.__new__(_RequestHandler)
    handler.request = Mock()
    handler.request.fileno.return_value = socket_fd
    handler.channel = Mock()
    handler.channel.receive.return_value = _Release(1)
    handler.server = Mock(registry=registry)

    with patch("kvcr.kvcr_service.select.poll", return_value=poller):
        handler._hold(0, liveness)

    registry.release.assert_called_once_with(0, liveness)
    registry.holder_died.assert_not_called()
    handler.channel.send.assert_called_once_with(_Released(1))


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
    liveness = Mock()

    assert handler._release_or_fail(0, liveness) is False

    handler.server.fail.assert_called_once_with(error)


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
            spec = harness.server._registry._pools[0].owner.spec
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
    first, failing = Mock(), Mock()
    failing.start.side_effect = RuntimeError("attach failed")

    with (
        patch("kvcr.kvcr_service._Guard", side_effect=[first, failing]),
        pytest.raises(RuntimeError, match="attach failed"),
    ):
        _PoolRegistry(
            tmp_path,
            2,
            _TEST_POOL_SIZE_BYTES,
            _TEST_JOURNAL_BYTES,
            _TEST_DIGEST,
        )

    first.close.assert_called_once_with()
    failing.close.assert_called_once_with()
    assert list(tmp_path.iterdir()) == []


def test_startup_allocation_failure_rolls_back_before_listener(
    tmp_path: Path,
) -> None:
    owners = [Mock(), Mock()]
    allocation_error = OSError("allocation failed")

    with (
        patch("kvcr.kvcr_service._Guard"),
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
    registry = _new_registry(tmp_path)
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
        assert _holders_of(registry) == {}
        replacement = _FakeLiveness()
        registry.claim(0, _TEST_TIER_CONFIG, replacement, _control_bind(0))
        registry.release(0, replacement)
    finally:
        peer.close()
        accepted.close()
        registry.close()


def test_promotion_failure_stops_the_whole_service(tmp_path: Path) -> None:
    """A Guard that cannot be promoted is fatal, on purpose."""
    registry = _new_registry(tmp_path)
    liveness = Mock()
    liveness.fileno.return_value = 11
    failure = RuntimeError("promotion failed")
    registry.holder_died = Mock(side_effect=failure)

    server = object.__new__(_ThreadingUnixServer)
    server.registry = registry
    server._fatal_error = None
    server._fatal_lock = threading.Lock()
    server.shutdown = Mock()
    handler = object.__new__(_RequestHandler)
    handler.request = Mock()
    handler.request.fileno.return_value = 10
    handler.server = server
    poller = Mock()
    poller.poll.return_value = [(11, select.POLLIN)]

    try:
        with patch("kvcr.kvcr_service.select.poll", return_value=poller):
            handler._hold(0, liveness)
        assert server._fatal_error is failure
        server.shutdown.assert_called_once_with()
    finally:
        registry.close()


@pytest.mark.parametrize("failure", ["setup", "poll", "pidfd"])
def test_ambiguous_hold_failure_stops_service_and_retains_binding(
    tmp_path: Path,
    failure: str,
) -> None:
    registry = _new_registry(tmp_path, 1)
    liveness = Mock()
    liveness.fileno.return_value = 11
    registry.claim(0, _TEST_TIER_CONFIG, liveness, _control_bind(0))

    server = object.__new__(_ThreadingUnixServer)
    server.registry = registry
    server._fatal_error = None
    server._fatal_lock = threading.Lock()
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
        assert registry._pools[0].holder is liveness
        liveness.close.assert_not_called()
        # Now refused for the stronger reason: failing closes the registry first.
        with pytest.raises(RuntimeError, match="closed"):
            registry.claim(0, _TEST_TIER_CONFIG, _FakeLiveness(), _control_bind(0))

        service = object.__new__(_KVCRService)
        service._server = server
        server.serve_forever = Mock()
        with pytest.raises(OSError) as raised:
            service.serve_forever()
        assert raised.value is fatal_error
    finally:
        registry.close()


@pytest.mark.parametrize("noticed_by", ["setup", "poll"])
def test_shutdown_closing_a_watched_pidfd_is_not_a_failure(
    tmp_path: Path,
    noticed_by: str,
) -> None:
    """close() breaks the descriptors _hold watches; neither end may latch a cause."""
    registry = _new_registry(tmp_path, 1)
    liveness = Mock()
    liveness.fileno.return_value = 11
    registry.claim(0, _TEST_TIER_CONFIG, liveness, _control_bind(0))

    server = object.__new__(_ThreadingUnixServer)
    server.registry = registry
    server._fatal_error = None
    server._fatal_lock = threading.Lock()
    server.shutdown = Mock()
    handler = object.__new__(_RequestHandler)
    handler.request = Mock()
    handler.request.fileno.return_value = 10
    handler.server = server
    poller = Mock()
    if noticed_by == "setup":
        # What an already-closed pidfd raises when _hold first asks for it.
        liveness.fileno.side_effect = ValueError("pidfd is closed")
    else:
        poller.poll.return_value = [(11, select.POLLERR | select.POLLNVAL)]

    registry.close()
    with patch("kvcr.kvcr_service.select.poll", return_value=poller):
        handler._hold(0, liveness)

    # Whichever end noticed, the shutdown is the cause and it is already under way.
    assert server._fatal_error is None
    server.shutdown.assert_not_called()


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


def test_a_claim_that_loses_a_race_with_shutdown_leaves_no_transition(
    tmp_path: Path,
) -> None:
    """Closing while a pool is mid-adoption must not strand it in transition."""
    adopting = threading.Event()
    finish_adopt = threading.Event()
    guard = Mock()
    guard.adopt.side_effect = lambda *_args: (
        adopting.set(),
        finish_adopt.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS),
    )
    registry = _new_registry(tmp_path, guards=[guard])
    liveness = _FakeLiveness()
    failures: list[BaseException] = []

    def claim() -> None:
        try:
            registry.claim(0, _TEST_TIER_CONFIG, liveness, _control_bind(0))
        except BaseException as error:  # noqa: BLE001 - recorded for the assert
            failures.append(error)

    claimant = threading.Thread(target=claim)
    claimant.start()
    try:
        assert adopting.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        registry._closed = True
        finish_adopt.set()
        claimant.join(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        assert not claimant.is_alive()

        # Refused, so nothing was granted -- but the pool is not left mid-flight.
        assert failures and isinstance(failures[0], KVCRServiceError)
        assert registry._pools[0].in_transition is False
        assert registry._pools[0].holder is liveness
    finally:
        finish_adopt.set()
        registry._closed = False
        registry.close()


def test_a_failed_release_refuses_claims_before_it_frees_the_pool(
    tmp_path: Path,
) -> None:
    """The window between a failed hand-back and the shutdown it causes."""
    guard = Mock()
    failure = RuntimeError("hand-back failed")
    guard.release.side_effect = failure
    registry = _new_registry(tmp_path, guards=[guard])
    claimable_when_reported: list[bool] = []
    registry.on_uncontained_failure = lambda _error: claimable_when_reported.append(
        registry._pools[0].holder is None and not registry._pools[0].in_transition
    )
    liveness = _FakeLiveness()
    try:
        registry.claim(0, _TEST_TIER_CONFIG, liveness, _control_bind(0))

        with pytest.raises(RuntimeError) as raised:
            registry.release(0, liveness)

        assert raised.value is failure
        # Reported while the pool was still held, so nothing could claim it.
        assert claimable_when_reported == [False]
    finally:
        registry.close()
