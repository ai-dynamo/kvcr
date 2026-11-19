# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The coordinator against workers that only speak the protocol: no KVCR, and
a request takes exactly as long as the test says."""

import heapq
import json
import threading
import time

import pytest
import zmq
from clusterbench import protocol
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


class Workers(threading.Thread):
    """Every worker of a run: each request is answered ``service_s`` later,
    or ``stall[session]`` seconds later if it is that session's first."""

    def __init__(self, endpoint: str, service_s: float, stall=None) -> None:
        super().__init__(daemon=True)
        self._endpoint = endpoint
        self._service_s = service_s
        self._stall = dict(stall or {})
        self.error = None

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
                arm="",
                kvcr_file="/k",
                kvcr_sha="abc",
                cpus=[],
                limits={},
            )
            sock.send(protocol.encode(hello))
            ids[sock] = index
        pending: list[tuple[float, int, zmq.Socket, protocol.Done]] = []
        served = dict.fromkeys(ids, 0)
        while ids:
            now = time.monotonic()
            while pending and pending[0][0] <= now:
                _, _, sock, done = heapq.heappop(pending)
                sock.send(protocol.encode(done))
            wait_ms = 50
            if pending:
                wait_ms = max(0, int((pending[0][0] - time.monotonic()) * 1e3))
            for sock, _ in poller.poll(wait_ms):
                message = protocol.decode(sock.recv())
                if isinstance(message, protocol.Configure):
                    ids[sock] = message.worker
                    sock.send(protocol.encode(protocol.Ready("configure")))
                elif isinstance(message, protocol.Request):
                    served[sock] += 1
                    blocks = sum(end - start for _, start, end in message.segments)
                    done = protocol.Done(
                        rid=message.rid,
                        ttft_ms=self._service_s * 1e3,
                        load_ms=0.0,
                        prompt_blocks=blocks,
                        gpu_blocks=blocks,
                        primary_blocks=0,
                        local_blocks=0,
                        remote_blocks=0,
                        failed_blocks=0,
                        recomputed_tokens=0,
                    )
                    service_s = self._stall.pop(message.session, self._service_s)
                    finish = time.monotonic() + service_s
                    heapq.heappush(pending, (finish, message.rid, sock, done))
                elif message.name == "measure":
                    ready = protocol.Ready("measure", {"primary_occupancy": 1.0})
                    sock.send(protocol.encode(ready))
                elif message.name == "stop":
                    report = self._report(ids[sock], served[sock])
                    sock.send(protocol.encode(protocol.Report(report)))
                elif message.name == "exit":
                    poller.unregister(sock)
                    del ids[sock]
                    sock.close()
        context.term()

    @staticmethod
    def _report(worker: int, served: int) -> bytes:
        window = Window()
        window.call("query", 0.004, calls=4000, items=4000)
        window.counters.update(requests=served, forward_seconds=1.0, idle_seconds=0.5)
        for name in ("ttft", "step_kvcr", "step_scheduler", "deliver_remote"):
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
        )
        return json.dumps(report).encode()


def _run(out, service_ms, *overrides, stall=None):
    """Run one cell against fake workers and return its summary."""
    config = load_config(None, [*OVERRIDES, *overrides])
    coordinator = Coordinator(config, out)
    endpoint = coordinator.bind("tcp://127.0.0.1:*")
    workers = Workers(endpoint, service_ms * 1e-3, stall)
    workers.start()
    try:
        summary = coordinator.run(expect=WORKERS, per_node=PER_NODE)
    finally:
        coordinator.close()
    workers.join(10)
    assert workers.error is None and not workers.is_alive(), workers.error
    return summary


def _rows(out):
    """The request log as one dict per request."""
    header, *rows = map(json.loads, (out / REQUEST_LOG).read_text().splitlines())
    assert header == {"clusterbench_requests": 1, "fields": list(REQUEST_FIELDS)}
    return [dict(zip(REQUEST_FIELDS, row)) for row in rows]


def test_every_request_is_logged_with_where_it_went_and_how_it_ended(tmp_path):
    # Session 0's first request does not come back before the run ends.
    summary = _run(tmp_path, 20, stall={0: 60.0})
    rows = _rows(tmp_path)
    assert len({(row["session"], row["turn"]) for row in rows}) == len(rows) > 100
    assert {row["worker"] for row in rows} == set(range(WORKERS))
    assert sum(row["in_window"] for row in rows) == summary["offered"]["dispatched"]
    finished = [row for row in rows if "aborted" in row]
    assert len(finished) > 100
    assert all(row["ttft_ms"] == 20 and not row["aborted"] for row in finished)
    # A request the run ended on is logged with no outcome.
    (stuck,) = [row for row in rows if row["session"] == 0]
    assert list(stuck) == list(REQUEST_FIELDS[:9])


def test_requests_that_fall_due_together_are_timed_one_by_one(tmp_path):
    # No ramp: every session's first request is due at the same instant.
    _run(tmp_path, 20, "workload.ramp_s=0")
    burst = sorted(row["t_ms"] for row in _rows(tmp_path))[: 2 * WORKERS]
    assert len(set(burst)) == len(burst)


def test_workers_that_loaded_another_kvcr_tree_fail_the_run(tmp_path):
    coordinator = Coordinator(load_config(None, OVERRIDES), tmp_path)
    endpoint = coordinator.bind("tcp://127.0.0.1:*")
    Workers(endpoint, 0.02).start()  # they report kvcr from /k
    try:
        with pytest.raises(RunFailed, match="not from /checkout/src"):
            coordinator.run(expect=WORKERS, expect_kvcr="/checkout/src")
    finally:
        coordinator.close()


def test_a_path_that_is_not_a_kvcr_checkout_is_refused_before_anything_starts(
    tmp_path,
):
    config = load_config(None, OVERRIDES)
    with pytest.raises(ValueError, match="not a KVCR checkout"):
        run_local(config, tmp_path / "run", kvcr_src=str(tmp_path))
    assert not (tmp_path / "run").exists()
