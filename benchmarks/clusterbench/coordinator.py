# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The coordinator: workload generator, synthetic router and run phases.

Never imports ``kvcr``, so it is the same program whichever KVCR tree the
workers run. Workers dial in. They acknowledge ``configure`` and ``measure``;
``stop`` is answered with a report and ``exit`` with nothing.
"""

from __future__ import annotations

import heapq
import json
import random
import time
from collections import deque
from pathlib import Path

import msgspec
import zmq

from . import protocol
from .config import RunConfig, encode_config
from .ids import segment_blocks, warm_stream
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
)  # fmt: skip


# How long the workers have to register, to acknowledge a phase and to warm.
PHASE_TIMEOUT_S = 300.0
# Mesh warm: a pull that fails is tried again this often, each time for a
# block the puller has not seen, and its requests have ids of their own.
WARM_ATTEMPTS = 5
WARM_RID = 1 << 38


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
        ident = self._idents[worker]
        try:
            self._sock.send_multipart([ident, protocol.encode(message)])
        except zmq.ZMQError as error:  # ROUTER_MANDATORY: its connection is gone
            raise RunFailed(
                f"cannot reach worker {worker} ({ident.decode()}): {error}"
            ) from None

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

    def _no_ack(self, phase: str, acked) -> RunFailed:
        missing = sorted(set(range(self._workers)) - set(acked))
        return RunFailed(f"phase {phase!r}: no ack from workers {missing}")

    def _barrier(self, phase: str) -> dict:
        """Wait for every worker's Ready(phase)."""
        infos: dict[int, dict] = {}
        deadline = time.monotonic() + PHASE_TIMEOUT_S
        while len(infos) < self._workers:
            left = deadline - time.monotonic()
            if left <= 0:
                raise self._no_ack(phase, infos)
            received = self._recv(min(left, 0.5))
            if received is None:
                continue
            worker, message = received
            if isinstance(message, protocol.Ready) and message.phase == phase:
                infos[worker] = message.info
            else:
                self._background(worker, message)
        return infos

    # ---- run ---------------------------------------------------------------

    def run(self, expect: int, per_node: int = 0, expect_kvcr: str = "") -> dict:
        """Run the cell. ``expect_kvcr`` is the directory every worker must
        have imported ``kvcr`` from (an arm's ``src``)."""
        config, phases = self.config, self.config.phases
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_bytes(
            msgspec.json.format(encode_config(config), indent=2)
        )

        # Register.
        self._ids = {}
        hellos: dict[bytes, protocol.Hello] = {}
        deadline = time.monotonic() + PHASE_TIMEOUT_S
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
        if expect_kvcr:
            # A wrong path falls back to the installed kvcr without a sound.
            root = Path(expect_kvcr).resolve()
            if any(root not in Path(file).parents for file in files):
                raise RunFailed(f"workers loaded KVCR from {files}, not from {root}")
        (self.out / "workers.json").write_text(
            json.dumps([msgspec.structs.asdict(h) for h in hello], indent=1)
        )

        # Configure: every worker starts its KVCR and learns the peer table.
        peers = [(h.node, h.control_endpoint) for h in hello]
        blob = encode_config(config)
        router = Router(config.router, nodes)
        self._router = router
        self._reports: dict[int, dict] = {}
        with (
            open(self.out / "timeseries.jsonl", "w") as self._timeseries,
            open(self.out / REQUEST_LOG, "wb") as log,
        ):
            for worker in range(workers):
                self._send(worker, protocol.Configure(worker, blob, peers))
            self._barrier("configure")

            warm = {"ok": 0, "failed": 0, "max_seconds": 0.0}
            if phases.mesh_warm == "full" and workers > 1:
                warm = self._warm(router)

            summary = self._drive(router, log)

        summary_in = {
            "name": config.name,
            "arm": self.arm,
            "kvcr_sha": shas[0],
            "kvcr_file": files[0],
            "nodes": nodes,
            "warm": warm,
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

    def _warm(self, router: Router) -> dict:
        """Mesh warm: every worker pulls one block from every other, so the
        window holds no first-contact connection setup. They are ordinary
        requests: each worker first computes a few blocks of its own, then is
        sent a one-block prompt per peer with a hint naming that peer."""
        workers = self._workers
        block = self.config.tiers.tokens_per_block
        deadline = time.monotonic() + PHASE_TIMEOUT_S
        rids = iter(range(WARM_RID, 2 * WARM_RID))

        def send(worker, stream, start, end, source=-1) -> int:
            rid = next(rids)
            request = protocol.Request(
                rid=rid,
                session=-1,
                segments=[(stream, start, end)],
                tail=(stream, end, end),
                prompt_tokens=(end - start) * block + 1,
                output_tokens=1,
                hint_source=source,
                hint_blocks=int(source >= 0),
            )
            self._send(worker, request)
            return rid

        def receive(what: str):
            left = deadline - time.monotonic()
            if left <= 0:
                raise RunFailed(f"mesh warm: {what} did not finish")
            return self._recv(min(left, 0.5))

        # Each worker stores its own blocks where a peer can pull them.
        own = [[(warm_stream(w), 0, WARM_ATTEMPTS)] for w in range(workers)]
        for worker in range(workers):
            send(worker, *own[worker][0])
        while any(router.servable(w, own[w]) < WARM_ATTEMPTS for w in range(workers)):
            received = receive("storing the workers' own blocks")
            if received is not None and not isinstance(received[1], protocol.Done):
                self._background(*received)

        todo = [deque(s for s in range(workers) if s != w) for w in range(workers)]
        pulls: dict[int, tuple[int, int, int]] = {}  # rid -> puller, source, attempt
        open_pulls = [0] * workers
        started = time.monotonic()
        ok = failed = 0
        slowest = 0.0

        def pull(worker, source, attempt) -> None:
            rid = send(worker, warm_stream(source), attempt, attempt + 1, source)
            pulls[rid] = (worker, source, attempt)
            open_pulls[worker] += 1

        limit = self.config.phases.warm_concurrency  # pulls a worker has open
        while pulls or any(todo):
            for worker, sources in enumerate(todo):
                while sources and open_pulls[worker] < limit:
                    pull(worker, sources.popleft(), 0)
            received = receive("pulling a block from every peer")
            if received is None:
                continue
            if not isinstance(received[1], protocol.Done):
                self._background(*received)
                continue
            pulled = pulls.pop(received[1].rid, None)
            if pulled is None:
                continue  # a worker's request for its own blocks
            worker, source, attempt = pulled
            open_pulls[worker] -= 1
            if received[1].remote_blocks:
                ok += 1
            elif attempt + 1 < WARM_ATTEMPTS:
                pull(worker, source, attempt + 1)
            else:
                failed += 1
            slowest = time.monotonic() - started
        return {"ok": ok, "failed": failed, "max_seconds": slowest}

    def _background(self, worker, message) -> None:
        """Messages that can arrive in any phase."""
        if isinstance(message, protocol.Inventory):
            self._router.inventory.apply(worker, message.deltas)
        elif isinstance(message, protocol.Stats):
            row = {"t": message.t, "worker": worker, **message.values}
            self._timeseries.write(json.dumps(row) + "\n")
        elif isinstance(message, protocol.Report):
            self._reports[worker] = json.loads(message.payload)

    def _drive(self, router: Router, log) -> dict:
        config, phases = self.config, self.config.phases
        workers = self._workers
        workload = self._workload
        rng = random.Random(config.workload.seed)
        clock = time.monotonic
        due: list[tuple[float, int]] = []
        start = clock()
        for _ in range(max(1, round(config.workload.sessions_per_worker * workers))):
            # The first sessions start mid-life (contexts already grown), so
            # the tiers reach their steady state within the fill phase.
            session = workload.new_session(aged=True)
            heapq.heappush(
                due, (start + rng.random() * config.workload.ramp_s, session.id)
            )
        inflight: dict[int, tuple] = {}
        inflight_max = 0
        next_rid = 0
        measuring = False
        measure_acks: set[int] = set()
        measure_at = start + phases.fill_s
        measure_deadline = None  # set when the workers are told to measure
        # Time spent waiting for a message or for the next turn to fall due;
        # the rest of the window is the coordinator's own work.
        waiting = waiting_before = 0.0
        stop_at = None
        stopped = False
        report_deadline = None
        lag = LogHist()
        tally = self._new_tally()
        offered = tally["offered"]
        encode = msgspec.json.Encoder().encode
        log.write(encode({"clusterbench_requests": 1, "fields": REQUEST_FIELDS}))

        def on_message(worker, message) -> None:
            nonlocal measuring, stop_at, waiting_before
            if isinstance(message, protocol.Done):
                entry = inflight.pop(message.rid, None)
                if entry is None:
                    return
                turn, sent = entry
                outcome = (
                    message.remote_blocks,
                    message.failed_blocks,
                    message.timed_out_blocks,
                    round(message.load_ms, 3),
                    round(message.ttft_ms, 3),
                )
                log.write(b"\n" + encode(sent + outcome))
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
                    waiting_before = waiting
            else:
                self._background(worker, message)

        while True:
            now = clock()
            if not stopped:
                while due and due[0][0] <= now:
                    when, session = heapq.heappop(due)
                    turn = workload.next_turn(session)
                    route = router.route(turn.session, turn.segments)
                    rid = next_rid
                    next_rid += 1
                    blocks = segment_blocks(turn.segments)
                    # Per request: turns that fall due together still go out
                    # one after another.
                    sent_at = clock()
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
                            round((sent_at - start) * 1e3, 3),
                            measuring,
                            blocks,
                            route.local_blocks,
                        ),
                    )
                    if len(inflight) > inflight_max:
                        inflight_max = len(inflight)
                    if measuring:
                        lag.add(sent_at - when)
                        offered["dispatched"] += 1
                        offered["dispatched_blocks"] += blocks
                        offered["dispatched_tokens"] += turn.prompt_tokens
                        offered["migrations"] += route.migrated
                        if route.hint_source >= 0:
                            offered["hinted"] += 1
                            offered["hinted_blocks"] += route.hint_blocks
                            same = router.same_node(route.worker, route.hint_source)
                            offered[
                                "hinted_same_node" if same else "hinted_cross_node"
                            ] += 1
                if measure_deadline is None:
                    if now >= measure_at:
                        measure_deadline = now + PHASE_TIMEOUT_S
                        for worker in range(workers):
                            self._send(worker, protocol.Phase("measure"))
                elif not measuring and now > measure_deadline:
                    # The window only opens, and so only closes, once every
                    # worker has acknowledged.
                    raise self._no_ack("measure", measure_acks)
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
        tally["lag"] = lag.to_json()
        tally["sessions_started"] = workload.started
        tally["inflight_max"] = inflight_max
        tally["workload"] = workload.summary()
        return tally

    @staticmethod
    def _new_tally() -> dict:
        """What the window offered (the summary carries that dict as it is).
        What came of it is counted by the workers."""
        offered = (
            "dispatched",
            "dispatched_blocks",
            "dispatched_tokens",
            "hinted",
            "hinted_blocks",
            "hinted_same_node",
            "hinted_cross_node",
            "migrations",
        )
        return {
            "offered": dict.fromkeys(offered, 0),
            "primary_occupancy_min": 1.0,  # emptiest host tier at window start
            "measure_s": 0.0,
            "busy_s": 0.0,  # window time not spent waiting
        }
