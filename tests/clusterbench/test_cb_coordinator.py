# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The coordinator against workers that only speak the protocol: no KVCR, and
a request takes exactly as long as the test says."""

import contextlib
import heapq
import json
import signal
import subprocess
import threading
import time

import msgspec
import pytest
import zmq
from clusterbench import launch_local, protocol
from clusterbench.config import load_config
from clusterbench.coordinator import (
    REQUEST_FIELDS,
    REQUEST_LOG,
    Coordinator,
    RunFailed,
)
from clusterbench.launch_local import run_local
from clusterbench.metrics import Window

OVERRIDES = [
    "workload.sessions_per_worker=2",
    "workload.think_median_s=0.02",
    "workload.think_sigma=0.3",
    "workload.think_cap_s=0.05",
    "workload.ramp_s=0.05",
    # Short sessions, so some end and are replaced inside the run.
    "workload.turns_median=30",
    "phases.mesh_warm=none",
    "phases.fill_s=0.3",
    "phases.measure_s=1.0",
    "phases.drain_s=0.2",
]
WORKERS, PER_NODE = 4, 2
SESSIONS = 2 * WORKERS  # live at any time: workload.sessions_per_worker each
ANSWERED = 3 * SESSIONS


class Workers(threading.Thread):
    """Every worker of a run. The n-th request to arrive is answered
    ``service(n)`` seconds later, or never if that is None. The workers in
    ``mute`` do not acknowledge ``measure``."""

    def __init__(self, endpoint: str, service=lambda n: 0.02, mute=()) -> None:
        super().__init__(daemon=True)
        self._endpoint = endpoint
        self._service = service
        self._mute = set(mute)
        self._quit = threading.Event()
        self.error = None

    def stop(self) -> None:
        self._quit.set()
        self.join(10)

    def run(self) -> None:
        try:
            self._serve()
        except Exception as error:  # surfaced by the test, not lost in a thread
            self.error = error

    def _serve(self) -> None:
        context = zmq.Context()
        poller = zmq.Poller()
        ids: dict[zmq.Socket, int] = {}
        for index in range(WORKERS):
            sock = context.socket(zmq.DEALER)
            sock.setsockopt(zmq.IDENTITY, b"w%d" % index)
            sock.linger = 0
            sock.connect(self._endpoint)
            poller.register(sock, zmq.POLLIN)
            node, local_id = divmod(index, PER_NODE)
            hello = protocol.Hello(
                node=f"n{node}",
                local_id=local_id,
                host="h",
                pid=0,
                control_endpoint="",
                agent_name=f"a{index}",
                kvcr_file="/k",
                kvcr_sha="abc",
                cpus=[],
                limits={},
            )
            sock.send(protocol.encode(hello))
            ids[sock] = index
        pending: list[tuple[float, int, zmq.Socket, protocol.Done]] = []
        served = dict.fromkeys(ids, 0)
        stopped = set()
        arrived = 0
        while ids and not self._quit.is_set():
            now = time.monotonic()
            while pending and pending[0][0] <= now:
                _, _, sock, done = heapq.heappop(pending)
                if sock not in stopped:  # a stopped worker drops its requests
                    sock.send(protocol.encode(done))
            wait_ms = 50
            if pending:
                left_ms = int((pending[0][0] - time.monotonic()) * 1e3)
                wait_ms = min(wait_ms, max(0, left_ms))
            for sock, _ in poller.poll(wait_ms):
                message = protocol.decode(sock.recv())
                if isinstance(message, protocol.Configure):
                    ids[sock] = message.worker
                    sock.send(protocol.encode(protocol.Ready("configure")))
                elif isinstance(message, protocol.Request):
                    served[sock] += 1
                    service_s = self._service(arrived)
                    arrived += 1
                    if service_s is None:
                        continue
                    blocks = sum(end - start for _, start, end in message.segments)
                    done = protocol.Done(
                        rid=message.rid,
                        ttft_ms=service_s * 1e3,
                        load_ms=0.0,
                        prompt_blocks=blocks,
                        gpu_blocks=blocks,
                        primary_blocks=0,
                        local_blocks=0,
                        remote_blocks=0,
                        failed_blocks=0,
                        recomputed_tokens=0,
                    )
                    finish = time.monotonic() + service_s
                    heapq.heappush(pending, (finish, message.rid, sock, done))
                elif message.name == "measure":
                    if ids[sock] not in self._mute:
                        ready = protocol.Ready("measure", {"primary_occupancy": 1.0})
                        sock.send(protocol.encode(ready))
                elif message.name == "stop":
                    stopped.add(sock)
                    report = self._report(ids[sock], served[sock])
                    sock.send(protocol.encode(protocol.Report(report)))
                elif message.name == "exit":
                    poller.unregister(sock)
                    del ids[sock]
        context.destroy()

    @staticmethod
    def _report(worker: int, served: int) -> bytes:
        window = Window()
        window.call("query", 0.004, calls=4000, items=4000)
        window.counters.update(requests=served, forward_seconds=1.0, idle_seconds=0.5)
        for name in ("ttft", "step_kvcr", "deliver_remote"):
            window.hists[name].add(0.01)
        window.hists["cache_resolve_remote"].add(0.02)
        report = window.to_json()
        report.update(
            window_s=1.0,
            worker=worker,
            threads={
                role: {"cpu_s": 0.1, "delay_s": 0.0, "threads": 1}
                for role in ("engine", "progress", "native")
            },
            engine={"primary_occupancy": 0.99},
            rss_mb=100.0,
            drained=True,
            profiled=False,
        )
        return json.dumps(report).encode()


@contextlib.contextmanager
def _cell(out, *overrides, **fake):
    """A coordinator with the fake workers dialled in to it."""
    coordinator = Coordinator(load_config(None, [*OVERRIDES, *overrides]), out)
    workers = Workers(coordinator.bind("tcp://127.0.0.1:*"), **fake)
    workers.start()
    try:
        yield coordinator
    finally:
        coordinator.close()
        workers.stop()
    assert workers.error is None and not workers.is_alive(), workers.error


def _run(out, *overrides, **fake):
    """Run one cell against fake workers and return its summary."""
    with _cell(out, *overrides, **fake) as coordinator:
        return coordinator.run(expect=WORKERS, per_node=PER_NODE)


@contextlib.contextmanager
def _within(seconds):
    """Fail, rather than hang, if the body is still running by then."""

    def expired(signum, frame):
        raise TimeoutError(f"still running after {seconds} s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _rows(out):
    """The request log as one dict per request."""
    header, *rows = map(json.loads, (out / REQUEST_LOG).read_text().splitlines())
    assert header == {"clusterbench_requests": 1, "fields": list(REQUEST_FIELDS)}
    return [dict(zip(REQUEST_FIELDS, row)) for row in rows]


def _inside_the_window(n):
    """Every answer arrives while the window is open: each session's first
    request is held until well after fill, and after ``ANSWERED`` requests
    nothing is answered, long before the window closes."""
    if n < SESSIONS:
        return 0.8
    return 0.02 if n < ANSWERED else None


@pytest.fixture(scope="module")
def cell(tmp_path_factory):
    out = tmp_path_factory.mktemp("cell")
    summary = _run(out, "phases.measure_s=1.5", service=_inside_the_window)
    return out, summary, _rows(out)


def test_every_request_is_logged_with_where_it_went_and_how_it_ended(cell):
    _, _, rows = cell
    # Every session ends on a request that is never answered.
    assert len(rows) == ANSWERED + SESSIONS
    assert len({(row["session"], row["turn"]) for row in rows}) == len(rows)
    assert {row["worker"] for row in rows} == set(range(WORKERS))
    finished = [row for row in rows if "aborted" in row]
    assert len(finished) == ANSWERED
    # Only the held requests went out during fill, and each row has its own
    # request's outcome.
    assert [row["ttft_ms"] for row in finished if not row["in_window"]] == (
        [800] * SESSIONS
    )
    assert {row["ttft_ms"] for row in finished if row["in_window"]} == {20}
    assert not any(row["aborted"] for row in finished)
    # A request the run ended on is logged with no outcome.
    unfinished = [list(row) for row in rows if "aborted" not in row]
    assert unfinished == [list(REQUEST_FIELDS[:9])] * SESSIONS


def test_requests_dispatched_and_finished_in_the_window_are_tallied(cell):
    _, summary, rows = cell
    in_window = [row for row in rows if row["in_window"]]
    offered = summary["offered"]
    # All but the held requests: those answered and each session's last.
    assert offered["dispatched"] == len(in_window) == ANSWERED
    assert offered["dispatched_blocks"] == sum(r["prompt_blocks"] for r in in_window)
    blocks = sum(row["prompt_blocks"] for row in rows if "aborted" in row)
    assert blocks > 0
    assert summary["completed"] == {
        **Coordinator._new_tally()["completed"],
        "completed": ANSWERED,
        "prompt_blocks": blocks,
        "gpu_blocks": blocks,
    }


def test_dispatch_lag_is_sampled_only_while_the_window_is_open(cell):
    _, summary, rows = cell
    assert summary["coordinator_lag"]["n"] == ANSWERED < len(rows)


def test_workers_json_names_its_fields(cell):
    out, _, _ = cell
    assert json.loads((out / "workers.json").read_text())[1] == {
        "node": "n0",
        "local_id": 1,
        "host": "h",
        "pid": 0,
        "control_endpoint": "",
        "agent_name": "a1",
        "kvcr_file": "/k",
        "kvcr_sha": "abc",
        "cpus": [],
        "limits": {},
    }


def test_requests_that_fall_due_together_are_timed_one_by_one(tmp_path):
    # No ramp: every session's first request is due at the same instant.
    _run(tmp_path, "workload.ramp_s=0", "phases.fill_s=0.1", "phases.measure_s=0.2")
    burst = sorted(row["t_ms"] for row in _rows(tmp_path))[:SESSIONS]
    assert len(set(burst)) == SESSIONS


def _done(**fields):
    done = protocol.Done(
        rid=0,
        ttft_ms=1.0,
        load_ms=0.5,
        prompt_blocks=10,
        gpu_blocks=4,
        primary_blocks=3,
        local_blocks=2,
        remote_blocks=0,
        failed_blocks=0,
        recomputed_tokens=64,
    )
    return msgspec.structs.replace(done, **fields)


def test_finished_requests_are_tallied_by_where_their_blocks_came_from():
    completed = Coordinator._new_tally()["completed"]
    nodes = ["a", "a", "b"]
    for source, done in (
        (-1, _done()),
        (1, _done(remote_blocks=1)),  # from a worker on the same node
        (2, _done(remote_blocks=5, failed_blocks=2, timed_out_blocks=1)),
    ):
        Coordinator._tally_done(completed, done, 0, source, nodes)
    assert completed == {
        "completed": 3,
        "aborted": 0,
        "prompt_blocks": 30,
        "gpu_blocks": 12,
        "primary_blocks": 9,
        "local_blocks": 6,
        "remote_requests": 2,
        "remote_blocks_same_node": 1,
        "remote_blocks_cross_node": 5,
        "failed_blocks": 2,
        "timed_out_blocks": 1,
        "prefill_tokens": 192,
    }


def test_an_aborted_request_still_counts_its_failed_and_timed_out_blocks():
    completed = Coordinator._new_tally()["completed"]
    done = _done(aborted=True, failed_blocks=3, timed_out_blocks=2)
    Coordinator._tally_done(completed, done, 0, 1, ["a", "b"])
    assert completed == {
        **Coordinator._new_tally()["completed"],
        "aborted": 1,
        "failed_blocks": 3,
        "timed_out_blocks": 2,
    }


def test_a_worker_that_never_acknowledges_measure_fails_the_run_and_is_named(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("clusterbench.coordinator.PHASE_TIMEOUT_S", 0.5)
    # A window that never opens never closes either.
    with _within(10), _cell(tmp_path, "phases.fill_s=0.1", mute={3}) as coordinator:
        with pytest.raises(
            RunFailed, match=r"phase 'measure': no ack from workers \[3\]"
        ):
            coordinator.run(expect=WORKERS)


def test_a_worker_that_reports_a_failure_fails_the_run_with_its_text(tmp_path):
    coordinator = Coordinator(load_config(None, OVERRIDES), tmp_path)
    context = zmq.Context()
    sock = context.socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, b"n0/1")
    sock.linger = 0
    sock.connect(coordinator.bind("tcp://127.0.0.1:*"))
    sock.send(protocol.encode(protocol.Failure("RuntimeError: no NIXL agent")))
    try:
        with pytest.raises(
            RunFailed, match="worker n0/1 failed:\nRuntimeError: no NIXL agent"
        ):
            coordinator.run(expect=1)
    finally:
        context.destroy()
        coordinator.close()


def test_a_send_to_a_worker_that_is_gone_fails_the_run_and_names_it(tmp_path):
    coordinator = Coordinator(load_config(None, OVERRIDES), tmp_path)
    coordinator._idents = [b"n0/0", b"n0/1"]  # registered, but no longer connected
    try:
        with pytest.raises(RunFailed, match=r"cannot reach worker 1 \(n0/1\)"):
            coordinator._send(1, protocol.Phase("stop"))
    finally:
        coordinator.close()


def test_workers_that_loaded_another_kvcr_tree_fail_the_run(tmp_path):
    with _cell(tmp_path) as coordinator:  # the workers report kvcr from /k
        with pytest.raises(RunFailed, match="not from /checkout/src"):
            coordinator.run(expect=WORKERS, expect_kvcr="/checkout/src")


def test_a_path_that_is_not_a_kvcr_checkout_is_refused_before_anything_starts(
    tmp_path,
):
    config = load_config(None, OVERRIDES)
    with pytest.raises(ValueError, match="not a KVCR checkout"):
        run_local(config, tmp_path / "run", kvcr_src=str(tmp_path))
    assert not (tmp_path / "run").exists()


def test_a_pin_the_machine_cannot_give_is_refused(tmp_path, monkeypatch):
    # Eight physical cores of two hyperthreads each.
    cores = [[core, core + 8] for core in range(8)]
    monkeypatch.setattr(launch_local, "core_groups", lambda cpus: cores)
    assert launch_local._cpu_sets(4, 0) == [[], [], [], []]  # 0: no pinning
    assert launch_local._cpu_sets(2, 4) == [
        [0, 1, 2, 3, 8, 9, 10, 11],
        [4, 5, 6, 7, 12, 13, 14, 15],
    ]
    with pytest.raises(ValueError, match="at least 3 physical cores"):
        launch_local._cpu_sets(2, 2)
    config = load_config(None, OVERRIDES)
    with pytest.raises(
        ValueError,
        match="pinning 4 workers to 3 physical cores each needs 12; 8 are available",
    ):
        run_local(config, tmp_path / "run", cores_per_worker=3)
    assert not (tmp_path / "run").exists()


def test_the_workers_of_a_failed_run_are_stopped_without_a_grace_period(
    tmp_path, monkeypatch
):
    clock = [100.0]
    waits, signals = [], []

    class Child:
        """A worker that never exits by itself."""

        pid = 2**30  # no such process

        def wait(self, timeout):
            waits.append(timeout)
            clock[0] += timeout
            raise subprocess.TimeoutExpired("worker", timeout)

    def fail(self, **kwargs):
        raise RunFailed("only 3 of 4 workers registered")

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(launch_local.os, "killpg", lambda pid, sig: signals.append(sig))
    monkeypatch.setattr(launch_local.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(Coordinator, "run", fail)
    with pytest.raises(RunFailed, match="only 3 of 4 workers registered"):
        run_local(load_config(None, OVERRIDES), tmp_path)
    assert signals == [signal.SIGTERM] * 4 + [signal.SIGKILL] * 4
    # No wait before SIGTERM, then ten seconds between them before SIGKILL.
    assert waits == [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
