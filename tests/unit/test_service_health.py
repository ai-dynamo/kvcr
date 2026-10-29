# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import functools
import http.client
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from _kvcr_test_utils import FakeNixlAgent, _use_nixl_agent, _wait_until
from test_kvcr_service import (
    _claim,
    _FakeLiveness,
    _running_server,
    _service_args,
)

from kvcr.service import memmap
from kvcr.service import server as kvcr_service
from kvcr.service.health import _serve_health
from kvcr.service.memory import _KVCRPoolOwner
from kvcr.service.server import _KVCRService


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
        ["--health-port", "9092", "--health-dependency-url", "http://["],
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
        with _serve_health(
            ("127.0.0.1", 0), lambda _ready: (200, "ok"), peer_url
        ) as server:
            assert _get(server, "/ready") == (503, "dependency unavailable\n")
            assert _get(server, "/live") == (200, "ok\n")
            status.return_value = (200, "ok")
            peer.RequestHandlerClass.protocol_version = "INVALID"
            assert _get(server, "/ready") == (503, "dependency unavailable\n")
        peer.RequestHandlerClass.protocol_version = "HTTP/1.0"

        def redirect(handler):
            handler.send_response(302)
            handler.send_header("Location", "http://[invalid/")
            handler.end_headers()

        peer.RequestHandlerClass.do_GET = redirect
        dependency_urls = {
            "malformed redirect": peer_url,
            "non-ASCII request path": peer_url + "/non-ascii-\u00e9",
        }
        for case, url in dependency_urls.items():
            with _serve_health(
                ("127.0.0.1", 0), lambda _ready: (200, "ok"), url
            ) as server:
                assert _get(server, "/ready") == (503, "dependency unavailable\n"), case
                assert _get(server, "/live") == (200, "ok\n"), case
    assert not any(t.name == "kvcr-health" for t in threading.enumerate())


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
            ("127.0.0.1", 0), lambda ready: service._health_status(2, ready=ready)
        ) as health:
            try:
                _wait_until(lambda: _get(health, "/ready")[0] == 200)
                holder = _FakeLiveness()
                _claim(service._registry, 0, holder)
                assert entered.wait(1)
                guard = service._registry._guards[0]
                assert not guard._core._progress._prepare_ready.is_set()
                before = (guard._phase, guard._pool_lease.current, guard._configured)
                time.sleep(2.2)
                for _ in range(3):
                    assert _get(health, "/live")[0] == 200
                    assert _get(health, "/ready")[0] == 200
                assert (
                    guard._phase,
                    guard._pool_lease.current,
                    guard._configured,
                ) == before
                holder.kill()
                _wait_until(
                    lambda: _get(health, "/ready") == (503, "guard promoting\n"),
                    timeout=1,
                )
                assert _get(health, "/live")[0] == 200
                _wait_until(
                    lambda: time.monotonic() - guard._last_progress > 2, timeout=3
                )
                assert _get(health, "/live")[0] == 200
                assert _get(health, "/ready") == (503, "guard promoting\n")
                assert not guard._serving
            finally:
                finish.set()
            _wait_until(lambda: guard._serving, timeout=5)
            _wait_until(lambda: _get(health, "/ready")[0] == 200, timeout=5)
            assert _get(health, "/live")[0] == 200


@pytest.mark.parametrize(
    "failure", ["accept_loop", "guard_loop", "guard_failure", "dead_guard", "shutdown"]
)
def test_health_fails_closed_without_waiting_on_guard(failure):
    # State snapshots, without racing the live loops that update progress clocks.
    now = time.monotonic()
    guard = SimpleNamespace(
        _failure=None,
        _thread=Mock(is_alive=lambda: True),
        _last_progress=now,
        _reserved=None,
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
    elif failure == "dead_guard":
        guard._thread.is_alive = lambda: False
    else:
        service._registry._refusing.set()
    assert service._health_status(30)[0] == 503
    assert service._health_status(30, ready=False)[0] == (
        200 if failure == "guard_loop" else 503
    )


def _health_cli():
    """Run the real CLI, with pipe-controlled allocation and cleanup boundaries."""
    memmap._hugetlbfs_mounts = lambda: []  # keep host hugetlbfs out of the daemon
    allocate_pool, close_service = _KVCRPoolOwner.allocate, _KVCRService.close

    def allocate(**kwargs):
        print("allocating", flush=True)
        if sys.stdin.readline().strip() == "fail":
            raise OSError("test allocation failed")
        return allocate_pool(**kwargs)

    def close(service):
        print("stopping", flush=True)
        assert sys.stdin.readline().strip() == "close"
        close_service(service)

    with (
        patch.object(_KVCRPoolOwner, "allocate", side_effect=allocate),
        patch.object(_KVCRService, "close", close),
        patch.object(
            kvcr_service,
            "_KVCRService",
            functools.partial(_KVCRService, journal_bytes=8192),
        ),
    ):
        kvcr_service.main()
    assert not any(t.name == "kvcr-health" for t in threading.enumerate())


@pytest.mark.parametrize("outcome", ["ready", "allocation_failure", "bind_failure"])
def test_service_health_startup_and_cleanup(tmp_path, outcome):
    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()
    socket_path = tmp_path / "service.sock"
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        health = SimpleNamespace(server_address=reservation.getsockname())
        if outcome != "bind_failure":
            reservation.close()
        bootstrap = (
            "import sys; sys.path.insert(0, sys.argv.pop(1)); "
            "from test_service_health import _health_cli; _health_cli()"
        )
        with subprocess.Popen(
            [sys.executable, "-c", bootstrap, str(Path(__file__).parent)]
            + _service_args(str(8192 / (1 << 30)))
            + [
                "--pool-dir",
                str(pool_dir),
                "--socket-path",
                str(socket_path),
                "--health-port",
                str(health.server_address[1]),
                "--health-timeout-seconds",
                "2",
                "--disable-hugepages",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as child:

            def expect(marker):
                assert select.select([child.stdout], [], [], 5)[0], marker
                assert child.stdout.readline().strip() == marker

            def reply(command):
                child.stdin.write(command + "\n")
                child.stdin.flush()

            try:
                if outcome != "bind_failure":
                    for rank in range(2):
                        expect("allocating")
                        assert _get(health, "/live") == (200, "service starting\n")
                        assert _get(health, "/ready") == (503, "service starting\n")
                        assert not socket_path.exists()
                        assert len(list(pool_dir.iterdir())) == rank
                        reply(
                            "fail"
                            if outcome == "allocation_failure" and rank == 1
                            else "allocate"
                        )
                    if outcome == "ready":
                        _wait_until(lambda: _get(health, "/ready")[0] == 200, timeout=5)
                        # An unclaimed, idle Guard must remain ready past the deadline.
                        time.sleep(2.2)
                        assert _get(health, "/ready")[0] == 200
                        assert _get(health, "/live")[0] == 200
                        assert len(list(pool_dir.iterdir())) == 2
                        child.send_signal(signal.SIGTERM)
                        expect("stopping")
                        for path in ("/live", "/ready"):
                            assert _get(health, path) == (503, "service stopping\n")
                        reply("close")
                output, errors = child.communicate(timeout=5)
                if outcome == "ready":
                    assert child.returncode == 0, errors
                else:
                    assert child.returncode != 0
                    assert (
                        "test allocation failed" in errors
                        if outcome == "allocation_failure"
                        else "Address already in use" in errors
                    )
                    assert "allocating" not in output
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
        assert not list(pool_dir.iterdir())
        assert not socket_path.exists()
        if outcome != "bind_failure":
            with pytest.raises(OSError):
                with socket.create_connection(health.server_address, timeout=1):
                    pass
