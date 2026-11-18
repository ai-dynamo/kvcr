# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The coordinator: workload generator, synthetic router and run phases.

Never imports ``kvcr``, so it is the same program whichever KVCR tree the
workers run. Workers dial in; every phase change is an acknowledged barrier.
"""

from __future__ import annotations

import heapq
import json
import random
import time
from pathlib import Path

import msgspec
import zmq

from . import protocol
from .config import RunConfig, encode_config
from .ids import segment_blocks
from .metrics import LogHist
from .report import summarize
from .router import Router
from .workload import make_workload

# requests.jsonl: a header naming these fields, then one row per request. A
# request the run ended on has no outcome, so its row stops at held_blocks.
REQUEST_LOG = "requests.jsonl"
REQUEST_FIELDS = (
    "session", "turn", "worker", "hint_source", "hint_blocks",
    "t_ms",  # sent this long after the workload started
    "in_window",
    "prompt_blocks",
    "held_blocks",  # of the prompt, what the router believed the worker held
    "remote_blocks", "failed_blocks", "timed_out_blocks", "load_ms", "ttft_ms",
    "aborted",
)  # fmt: skip


class RunFailed(RuntimeError):
    pass


class Coordinator:
    def __init__(self, config: RunConfig, out_dir: str | Path, arm: str = "") -> None:
        self.config = config
        self.out = Path(out_dir)
        self.arm = arm
        # First, so a bad trace file fails before any worker is started.
        self._workload = make_workload(config)
        self._context = zmq.Context()
        self._sock = self._context.socket(zmq.ROUTER)
        self._sock.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self._sock.linger = 2000
        self._sock.set_hwm(0)

    def bind(self, endpoint: str) -> str:
        """Bind and return the endpoint workers should dial."""
        if endpoint.endswith(":*") or endpoint.endswith(":0"):
            base = endpoint.rsplit(":", 1)[0]
            port = self._sock.bind_to_random_port(base)
            return f"{base}:{port}"
        self._sock.bind(endpoint)
        return endpoint

    def close(self) -> None:
        self._sock.close()
        self._context.term()

    # ---- plumbing ----------------------------------------------------------

    def _send(self, worker: int, message) -> None:
        self._sock.send_multipart([self._idents[worker], protocol.encode(message)])

    def _recv(self, timeout_s: float):
        """Next (worker, message), or None on timeout."""
        # Under a millisecond there is nothing to sleep on: just try to read.
        if timeout_s >= 0.001 and not self._sock.poll(int(timeout_s * 1000)):
            return None
        try:
            ident, data = self._sock.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            return None
        message = protocol.decode(data)
        if isinstance(message, protocol.Failure):
            raise RunFailed(f"worker {ident.decode()} failed:\n{message.text}")
        return self._ids.get(ident, ident), message

    def _barrier(self, phase: str, timeout_s: float, handler=None) -> dict:
        """Wait for every worker's Ready(phase); other messages go to handler."""
        infos: dict[int, dict] = {}
        deadline = time.monotonic() + timeout_s
        while len(infos) < self._workers:
            left = deadline - time.monotonic()
            if left <= 0:
                missing = sorted(set(range(self._workers)) - set(infos))
                raise RunFailed(f"phase {phase!r}: no ack from workers {missing}")
            received = self._recv(min(left, 0.5))
            if received is None:
                continue
            worker, message = received
            if isinstance(message, protocol.Ready) and message.phase == phase:
                infos[worker] = message.info
            elif handler is not None:
                handler(worker, message)
        return infos

    # ---- run ---------------------------------------------------------------

    def run(self, expect: int, expect_sha: str = "", per_node: int = 0) -> dict:
        config, phases = self.config, self.config.phases
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_bytes(
            msgspec.json.format(encode_config(config), indent=2)
        )

        # Register.
        self._ids = {}
        hellos: dict[bytes, protocol.Hello] = {}
        deadline = time.monotonic() + phases.register_timeout_s
        while len(hellos) < expect:
            left = deadline - time.monotonic()
            if left <= 0:
                raise RunFailed(f"only {len(hellos)} of {expect} workers registered")
            received = self._recv(min(left, 0.5))
            if received is None:
                continue
            ident, message = received
            if isinstance(message, protocol.Hello):
                hellos[ident] = message
        order = sorted(hellos, key=lambda i: (hellos[i].node, hellos[i].local_id))
        self._idents = order
        self._ids = {ident: worker for worker, ident in enumerate(order)}
        self._workers = workers = len(order)
        hello = [hellos[ident] for ident in order]
        nodes = [h.node for h in hello]
        shas = sorted({h.kvcr_sha for h in hello})
        files = sorted({h.kvcr_file for h in hello})
        if len(shas) > 1 or len(files) > 1:
            raise RunFailed(f"workers run different KVCR trees: {files} {shas}")
        counts = {node: nodes.count(node) for node in set(nodes)}
        if per_node and any(count != per_node for count in counts.values()):
            # srun can place a step's tasks on the wrong node of the job.
            raise RunFailed(f"expected {per_node} workers per node, got {counts}")
        if expect_sha and shas != [expect_sha]:
            raise RunFailed(f"expected KVCR {expect_sha}, workers run {shas}")
        (self.out / "workers.json").write_text(
            json.dumps([msgspec.to_builtins(h) for h in hello], indent=1)
        )

        # Configure: every worker starts its KVCR and learns the peer table.
        peers = [(h.node, h.control_endpoint) for h in hello]
        blob = encode_config(config)
        router = Router(config.router, nodes, config.tiers)
        self._router = router
        self._timeseries = open(self.out / "timeseries.jsonl", "w")
        self._reports: dict[int, dict] = {}
        for worker in range(workers):
            self._send(worker, protocol.Configure(worker, blob, peers))
        self._barrier("configure", phases.register_timeout_s, self._background)

        # Mesh warm.
        warm = {}
        if phases.mesh_warm == "full" and workers > 1:
            for worker in range(workers):
                self._send(worker, protocol.Phase("warm"))
            warm = self._barrier("warm", phases.warm_timeout_s, self._background)

        summary = self._drive(router, nodes)
        self._timeseries.close()

        summary_in = {
            "name": config.name,
            "arm": self.arm,
            "kvcr_sha": shas[0],
            "kvcr_file": files[0],
            "nodes": nodes,
            "warm": {
                "ok": sum(i.get("ok", 0) for i in warm.values()),
                "failed": sum(i.get("failed", 0) for i in warm.values()),
                "max_seconds": max(
                    (i.get("seconds", 0) for i in warm.values()), default=0.0
                ),
            },
            "coordinator": summary,
        }
        reports = [self._reports[w] for w in sorted(self._reports)]
        reports_dir = self.out / "reports"
        reports_dir.mkdir(exist_ok=True)
        for report in reports:
            (reports_dir / f"{report['worker']:04d}.json").write_text(
                json.dumps(report)
            )
        (self.out / "coordinator.json").write_text(json.dumps(summary_in, indent=1))
        result = summarize(msgspec.to_builtins(config), summary_in, reports)
        (self.out / "summary.json").write_text(json.dumps(result, indent=1))
        return result

    def _background(self, worker, message) -> None:
        """Messages that can arrive in any phase."""
        if isinstance(message, protocol.Inventory):
            self._router.inventory.apply(worker, message.deltas)
        elif isinstance(message, protocol.Stats):
            row = {"t": message.t, "worker": worker, **message.values}
            self._timeseries.write(json.dumps(row) + "\n")
        elif isinstance(message, protocol.Report):
            self._reports[worker] = json.loads(message.payload)

    def _drive(self, router: Router, nodes: list[str]) -> dict:
        config, phases = self.config, self.config.phases
        workers = self._workers
        workload = self._workload
        rng = random.Random(config.workload.seed)
        clock = time.monotonic
        due: list[tuple[float, int]] = []
        start = clock()
        for _ in range(max(1, round(config.workload.sessions_per_worker * workers))):
            session = workload.new_session(aged=config.workload.aged_start)
            heapq.heappush(
                due, (start + rng.random() * config.workload.ramp_s, session.id)
            )
        inflight: dict[int, tuple] = {}
        inflight_max = 0
        next_rid = 0
        measuring = False
        measure_acks: set[int] = set()
        measure_at = start + phases.fill_s
        measure_sent = False
        # Time spent waiting for a message or for the next turn to fall due;
        # the rest of the window is the coordinator's own work.
        waiting = waiting_before = 0.0
        stop_at = None
        stopped = False
        report_deadline = None
        lag = LogHist()
        tally = self._new_tally()
        log = open(self.out / REQUEST_LOG, "wb")
        encode = msgspec.json.Encoder().encode
        log.write(encode({"clusterbench_requests": 1, "fields": REQUEST_FIELDS}))

        def on_message(worker, message) -> None:
            nonlocal measuring, stop_at, waiting_before
            if isinstance(message, protocol.Done):
                entry = inflight.pop(message.rid, None)
                if entry is None:
                    return
                turn, sent = entry
                source, hinted = sent[3], sent[4]
                outcome = (
                    message.remote_blocks,
                    message.failed_blocks,
                    message.timed_out_blocks,
                    round(message.load_ms, 3),
                    round(message.ttft_ms, 3),
                    message.aborted,
                )
                log.write(b"\n" + encode(sent + outcome))
                router.done(message.rid)
                if measuring and not stopped:
                    self._tally_done(tally, message, worker, source, hinted, nodes)
                if stopped:
                    return
                # The workload says what this sets off: the session's next
                # turn, sub-agents starting, a parent resuming, a new session.
                step = workload.completed(turn)
                for session in step.ended:
                    router.forget(session)
                now = clock()
                for delay, session in step.wake:
                    heapq.heappush(due, (now + delay, session))
            elif isinstance(message, protocol.Ready) and message.phase == "measure":
                measure_acks.add(worker)
                occupancy = message.info.get("primary_occupancy", 1.0)
                tally["primary_occupancy_min"] = min(
                    tally["primary_occupancy_min"], occupancy
                )
                if len(measure_acks) == workers:
                    measuring = True
                    stop_at = clock() + phases.measure_s
                    tally["measure_started"] = time.time()
                    waiting_before = waiting
            else:
                self._background(worker, message)

        while True:
            now = clock()
            if not stopped:
                while due and due[0][0] <= now:
                    when, session = heapq.heappop(due)
                    turn = workload.next_turn(session)
                    if turn is None:  # only if a session had zero turns left
                        router.forget(session)
                        heapq.heappush(due, (now, workload.new_session().id))
                        continue
                    lag.add(now - when)
                    route = router.route(
                        turn.session, turn.turn, turn.segments, turn.prompt_tokens
                    )
                    rid = next_rid
                    next_rid += 1
                    blocks = segment_blocks(turn.segments)
                    router.dispatched(rid, route, blocks, turn.prompt_tokens)
                    self._send(
                        route.worker,
                        protocol.Request(
                            rid=rid,
                            session=turn.session,
                            segments=list(turn.segments),
                            tail=turn.tail,
                            prompt_tokens=turn.prompt_tokens,
                            output_tokens=turn.output_tokens,
                            hint_source=route.hint_source,
                            hint_blocks=route.hint_blocks,
                        ),
                    )
                    inflight[rid] = (
                        turn,
                        (
                            turn.session,
                            turn.turn,
                            route.worker,
                            route.hint_source,
                            route.hint_blocks,
                            round((now - start) * 1e3, 3),
                            measuring,
                            blocks,
                            route.local_blocks,
                        ),
                    )
                    if len(inflight) > inflight_max:
                        inflight_max = len(inflight)
                    if measuring:
                        tally["dispatched"] += 1
                        tally["dispatched_blocks"] += blocks
                        tally["dispatched_tokens"] += turn.prompt_tokens
                        tally["migrations"] += route.migrated
                        if route.hint_source >= 0:
                            tally["hinted"] += 1
                            tally["hinted_blocks"] += route.hint_blocks
                            same = router.same_node(route.worker, route.hint_source)
                            tally[
                                "hinted_same_node" if same else "hinted_cross_node"
                            ] += 1
                if not measure_sent and now >= measure_at:
                    measure_sent = True
                    for worker in range(workers):
                        self._send(worker, protocol.Phase("measure"))
                if stop_at is not None and now >= stop_at:
                    stopped = True
                    tally["measure_s"] = now - (stop_at - phases.measure_s)
                    tally["busy_s"] = tally["measure_s"] - (waiting - waiting_before)
                    for worker in range(workers):
                        self._send(worker, protocol.Phase("stop"))
                    report_deadline = now + phases.drain_s + 60
            elif len(self._reports) == workers:
                # Everyone has drained and reported: now they may close KVCR.
                for worker in range(workers):
                    self._send(worker, protocol.Phase("exit"))
                break
            elif now > report_deadline:
                missing = sorted(set(range(workers)) - set(self._reports))
                raise RunFailed(f"no report from workers {missing}")

            wait = 0.05
            if due and not stopped:
                wait = min(wait, max(0.0, due[0][0] - clock()))
            before = clock()
            received = self._recv(wait)
            waiting += clock() - before
            # Drain what is queued before dispatching again.
            while received is not None:
                on_message(*received)
                received = self._recv(0)

        for _, sent in inflight.values():
            log.write(b"\n" + encode(sent))
        log.write(b"\n")
        log.close()
        tally["lag"] = lag.to_json()
        tally["sessions_started"] = workload.started
        tally["inflight_at_stop"] = len(inflight)
        tally["inflight_max"] = inflight_max
        tally["workload"] = workload.summary()
        tally["ttft"] = tally["ttft"].to_json()
        tally["ttft_remote"] = tally["ttft_remote"].to_json()
        return tally

    @staticmethod
    def _new_tally() -> dict:
        tally = dict.fromkeys(
            (
                "dispatched",
                "dispatched_blocks",
                "dispatched_tokens",
                "migrations",
                "hinted",
                "hinted_blocks",
                "hinted_same_node",
                "hinted_cross_node",
                "completed",
                "aborted",
                "timed_out_blocks",
                "remote_blocks_same_node",
                "remote_blocks_cross_node",
                "remote_requests",
                "failed_blocks",
                "local_blocks",
                "gpu_blocks",
                "primary_blocks",
                "prompt_blocks",
                "prefill_tokens",
                "hinted_blocks_completed",
            ),
            0,
        )
        tally["ttft"] = LogHist()
        tally["ttft_remote"] = LogHist()
        tally["primary_occupancy_min"] = 1.0  # emptiest host tier at window start
        tally["busy_s"] = 0.0  # window time not spent waiting
        return tally

    @staticmethod
    def _tally_done(tally, done, worker, source, hinted, nodes) -> None:
        if done.aborted:
            # Dropped by the engine: no first token and nothing reused.
            tally["aborted"] += 1
            return
        tally["completed"] += 1
        tally["prompt_blocks"] += done.prompt_blocks
        tally["gpu_blocks"] += done.gpu_blocks
        tally["primary_blocks"] += done.primary_blocks
        tally["local_blocks"] += done.local_blocks
        tally["failed_blocks"] += done.failed_blocks
        tally["timed_out_blocks"] += done.timed_out_blocks
        tally["prefill_tokens"] += done.recomputed_tokens
        tally["ttft"].add(done.ttft_ms / 1e3)
        if source >= 0:
            tally["hinted_blocks_completed"] += hinted
        if done.remote_blocks:
            tally["remote_requests"] += 1
            tally["ttft_remote"].add(done.ttft_ms / 1e3)
            same = source >= 0 and nodes[worker] == nodes[source]
            tally[
                "remote_blocks_same_node" if same else "remote_blocks_cross_node"
            ] += done.remote_blocks
