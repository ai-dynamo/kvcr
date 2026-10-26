# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import http.client
import threading
import time
from contextlib import closing, contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _kvcr_test_utils import FakeNixlAgent, _use_nixl_agent, _wait_until
from test_kvcr_service import (
    _claim,
    _FakeLiveness,
    _running_server,
    _service_args,
)

from kvcr import kvcr_service
from kvcr.health import _serve_health
from kvcr.kvcr_service import _KVCRService


def _get(server, path):
    with closing(
        http.client.HTTPConnection(*server.server_address, timeout=2)
    ) as client:
        client.request("GET", path)
        response = client.getresponse()
        return response.status, response.read().decode()


@pytest.mark.parametrize(
    "options",
    [
        ["--health-port", "0"],
        ["--health-timeout-seconds", "nan"],
        ["--health-dependency-url", "http://127.0.0.1:9091/live"],
    ],
)
def test_invalid_health_options_fail_before_allocation(options):
    with pytest.raises(SystemExit) as error:
        kvcr_service._parse_args(_service_args("1") + options)
    assert error.value.code == 2


def test_health_http_routes_dependency_and_shutdown():
    status = Mock(return_value=(200, "ok"))
    with _serve_health(("127.0.0.1", 0), status) as peer:
        peer_url = "http://%s:%s/ready" % peer.server_address
        with _serve_health(("127.0.0.1", 0), status, peer_url) as server:
            assert _get(server, "/live") == (200, "ok\n")
            assert _get(server, "/ready") == (200, "ok\n")
            status.return_value = (503, "stopping")
            assert _get(server, "/ready") == (503, "stopping\n")
            assert _get(server, "/missing")[0] == 404
        with _serve_health(("127.0.0.1", 0), lambda: (200, "ok"), peer_url) as server:
            assert _get(server, "/ready") == (503, "dependency unavailable\n")
            status.return_value = (200, "ok")
            peer.RequestHandlerClass.protocol_version = "INVALID"
            assert _get(server, "/ready") == (503, "dependency unavailable\n")
    assert not any(t.name == "kvcr-health" for t in threading.enumerate())


def test_health_does_not_claim_or_release_attached_pool(tmp_path):
    with (
        _use_nixl_agent(FakeNixlAgent()),
        _running_server(tmp_path, guard_count=1) as harness,
    ):
        service = harness.server
        # The accept loop has not necessarily completed its first iteration yet.
        service._server.service_actions()
        assert service._health_status(30) == (200, "ok")
        holder = _FakeLiveness()
        _, lease = _claim(service._registry, 0, holder)
        guard = service._registry._guards[0]
        before = (guard._phase, guard._pool_lease.current, guard._configured)
        for _ in range(3):
            assert service._health_status(30) == (200, "ok")
        assert (guard._phase, guard._pool_lease.current, guard._configured) == before
        service._registry.release(0, lease)


def test_health_tracks_progress_during_preparation_and_promotion(tmp_path):
    entered, finish = threading.Event(), threading.Event()

    class RegisteringAgent(FakeNixlAgent):
        def register_memory(self, descs, mem_type="DRAM"):
            entered.set()
            assert finish.wait(20)
            return super().register_memory(descs, mem_type)

    with (
        _use_nixl_agent(RegisteringAgent()),
        _running_server(tmp_path, guard_count=1) as harness,
    ):
        service = harness.server
        with _serve_health(
            ("127.0.0.1", 0), lambda: service._health_status(2)
        ) as health:
            try:
                _wait_until(lambda: _get(health, "/ready")[0] == 200)
                holder = _FakeLiveness()
                _claim(service._registry, 0, holder)
                assert entered.wait(1)
                guard = service._registry._guards[0]
                assert not guard._core._progress._prepare_ready.is_set()
                time.sleep(2.2)
                assert _get(health, "/ready")[0] == 200
                holder.kill()
                _wait_until(
                    lambda: _get(health, "/ready") == (503, "guard stalled\n"),
                    timeout=5,
                )
                assert not guard._serving
            finally:
                finish.set()
            _wait_until(lambda: guard._serving, timeout=5)
            _wait_until(lambda: _get(health, "/ready")[0] == 200, timeout=5)


@pytest.mark.parametrize(
    "failure", ["accept_loop", "guard_loop", "guard_failure", "shutdown"]
)
def test_health_fails_closed_without_waiting_on_guard(failure):
    # State snapshots, without racing the live loops that update progress clocks.
    now = time.monotonic()
    guard = SimpleNamespace(
        _failure=None, _thread=Mock(is_alive=lambda: True), _last_progress=now
    )
    service = object.__new__(_KVCRService)
    service._closed = False
    service._server = SimpleNamespace(_fatal_error=None, _last_progress=now)
    service._registry = SimpleNamespace(
        _refusing=threading.Event(), _guards={0: guard}, _guard_count=1
    )
    if failure == "accept_loop":
        service._server._last_progress = now - 31
    elif failure == "guard_loop":
        guard._last_progress = now - 31
    elif failure == "guard_failure":
        guard._failure = RuntimeError("test")
    else:
        service._registry._refusing.set()
    assert service._health_status(30)[0] == 503


def test_service_health_listener_tracks_startup_idle_progress_and_shutdown(
    tmp_path, monkeypatch
):
    listening = threading.Event()
    listeners = []

    @contextmanager
    def capture_listener(*args):
        with _serve_health(*args) as listener:
            listeners.append(listener)
            listening.set()
            yield listener

    monkeypatch.setattr(kvcr_service, "_serve_health", capture_listener)
    service = _KVCRService(
        tmp_path / "service.sock", tmp_path, 1, (8192,), "test", journal_bytes=8192
    )
    thread = threading.Thread(target=service.serve_forever, args=(("127.0.0.1", 0),))
    try:
        assert service._health_status(30) == (503, "service not serving")
        thread.start()
        assert listening.wait(2)
        _wait_until(lambda: _get(listeners[0], "/ready")[0] == 200, timeout=2)
        guard = service._registry._guards[0]
        initial_progress = guard._last_progress
        _wait_until(lambda: guard._last_progress > initial_progress, timeout=2)
        assert _get(listeners[0], "/live") == (200, "ok\n")
    finally:
        if thread.is_alive():
            service.shutdown()
        thread.join(timeout=3)
        service.close()
    assert not thread.is_alive()
    assert listeners[0].fileno() == -1
    assert service._health_status(30)[0] == 503
