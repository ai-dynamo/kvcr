# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""One emulated engine rank: a real KVCR driven by a fake engine.

The process has the thread layout of a real rank: this (engine) thread next to
KVCR's progress thread and NIXL's own threads, plus the linker's KVCR-owner
thread in the SGLang profile. Coordinator traffic is handled between engine
steps, where a real engine reads its input queue.
"""

from __future__ import annotations

import cProfile
import gc
import json
import os
import socket
import time
import traceback

import zmq

from . import protocol
from .config import decode_config
from .engine import make_engine
from .kvcr_host import (
    auto_cpus,
    kvcr_identity,
    limits,
    make_host,
    prepare_environment,
)
from .metrics import Telemetry, Window, rss_bytes, thread_delta, thread_snapshot


def parse_cpus(spec: str) -> list[int]:
    cpus: set[int] = set()
    for part in filter(None, spec.split(",")):
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return sorted(cpus)


def run_worker(args) -> int:
    if args.cpus == "auto":
        cpus, socket_index = auto_cpus(args.local_id, args.per_node, args.reserve_cores)
    else:
        cpus, socket_index = parse_cpus(args.cpus), 0
    if args.net_devices:
        # One RDMA device per socket: use the one next to this worker's cores.
        devices = args.net_devices.split(",")
        os.environ["UCX_NET_DEVICES"] = devices[socket_index % len(devices)]
    prepare_environment()
    advertise = args.advertise_host or args.bind_host
    endpoint = f"tcp://{advertise}:{args.control_port}"
    agent = f"cb-{args.node}-{args.local_id}-{os.getpid()}"

    context = zmq.Context()  # private: KVCR owns zmq.Context.instance()
    sock = context.socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, f"{args.node}/{args.local_id}".encode())
    sock.linger = 5000
    sock.connect(args.coordinator)
    send = sock.send
    encode = protocol.encode

    host = engine = None
    try:
        path, sha = kvcr_identity()
        send(
            encode(
                protocol.Hello(
                    node=args.node,
                    local_id=args.local_id,
                    host=socket.gethostname(),
                    pid=os.getpid(),
                    control_endpoint=endpoint,
                    agent_name=agent,
                    arm=args.arm,
                    kvcr_file=path,
                    kvcr_sha=sha,
                    cpus=cpus,
                    limits=limits(),
                )
            )
        )
        if not sock.poll(args.configure_timeout * 1000):
            raise RuntimeError("no Configure from the coordinator")
        configure = protocol.decode(sock.recv())
        if not isinstance(configure, protocol.Configure):
            raise RuntimeError(f"expected Configure, got {type(configure).__name__}")
        config = decode_config(configure.config)
        me = configure.worker
        phases = config.phases

        window = Window()
        window.watch_gc()
        engine = make_engine(config, window, me)
        engine.set_peers(configure.peers)
        host = make_host(
            config.tiers,
            config.engine,
            agent_name=agent,
            bind_host=args.bind_host,
            advertise_host=advertise,
            control_port=args.control_port,
            nixl_port=args.nixl_port,
            framework=engine,
            stats_factory=Telemetry,
            cpus=cpus,
        )
        engine.attach(host)
        send(encode(protocol.Ready("configure")))

        perf = time.perf_counter
        warm_started = None
        stop_deadline = None
        exit_deadline = None
        final = None
        threads_before = thread_snapshot()
        threads_after = threads_before
        # CB_PROFILE_DIR: profile the engine thread over the measured window.
        profile_dir = os.environ.get("CB_PROFILE_DIR")
        profiler = None
        flush_s = phases.inventory_flush_ms / 1e3
        next_inventory = next_stats = perf()
        while True:
            while True:
                try:
                    data = sock.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                message = protocol.decode(data)
                if isinstance(message, protocol.Request):
                    engine.submit(message)
                elif isinstance(message, protocol.Phase):
                    name = message.name
                    if name == "warm":
                        engine.start_warm(
                            [
                                (peer, peer_endpoint)
                                for peer, (_, peer_endpoint) in enumerate(
                                    configure.peers
                                )
                                if peer != me
                            ]
                        )
                        warm_started = perf()
                    elif name == "measure":
                        # Start the window from a settled heap, like a
                        # long-running engine: startup garbage is collected
                        # and what survives is not rescanned.
                        gc.collect()
                        gc.freeze()
                        window.reset()
                        engine.begin_window()
                        threads_before = thread_snapshot()
                        fill = engine.fill_level()
                        ready = protocol.Ready("measure", {"primary_occupancy": fill})
                        send(encode(ready))
                        if profile_dir:
                            profiler = cProfile.Profile()
                            profiler.enable()
                    elif name == "stop":
                        if profiler is not None:
                            profiler.disable()
                            profiler.dump_stats(f"{profile_dir}/worker-{me}.prof")
                            profiler = None
                        window.ended = perf()
                        threads_after = thread_snapshot()
                        engine.end_window()
                        final = window.to_json()
                        final["threads"] = thread_delta(threads_before, threads_after)
                        final["engine"] = engine.gauges()
                        # The window is closed: drop the requests, and let the
                        # KVCR operations already issued run to completion.
                        engine.abort_requests()
                        stop_deadline = perf() + phases.drain_s
                    elif name == "exit":
                        exit_deadline = 0.0
                    else:
                        send(encode(protocol.Ready(name)))

            for done in engine.step():
                send(encode(done))

            now = perf()
            if warm_started is not None:
                pending, ok, failed = engine.warm_status()
                if pending == 0:
                    send(
                        encode(
                            protocol.Ready(
                                "warm",
                                {
                                    "ok": ok,
                                    "failed": failed,
                                    "seconds": now - warm_started,
                                },
                            )
                        )
                    )
                    warm_started = None
            if now >= next_inventory:
                deltas = engine.take_inventory()
                if deltas:
                    send(encode(protocol.Inventory(deltas)))
                next_inventory = now + flush_s
            if now >= next_stats:
                values = engine.gauges()
                for key, value in window.gauges.items():
                    if key.startswith("kvcr_state|"):
                        values[key.partition("|")[2]] = value
                values["rss_mb"] = rss_bytes() / 1e6
                send(encode(protocol.Stats(time.time(), values)))
                next_stats = now + phases.stats_interval_s
            if stop_deadline is not None and (not engine.busy() or now > stop_deadline):
                # Report, then keep serving peers until the coordinator says
                # every worker is done: closing KVCR while a peer still has an
                # operation against it would strand that peer.
                final.update(
                    worker=me,
                    node=args.node,
                    local_id=args.local_id,
                    host=socket.gethostname(),
                    arm=args.arm,
                    kvcr_file=path,
                    kvcr_sha=sha,
                    affinity=host.affinity,
                    rss_mb=rss_bytes() / 1e6,
                    drained=not engine.busy(),
                    resilience_samples=engine.resilience_samples(),
                    ucx_tls=os.environ.get("UCX_TLS", ""),
                    ucx_net_devices=os.environ.get("UCX_NET_DEVICES", ""),
                )
                send(encode(protocol.Report(json.dumps(final).encode())))
                stop_deadline = None
                exit_deadline = now + phases.drain_s + 120
            if exit_deadline is not None and now > exit_deadline:
                break

        status = 0
    except Exception:
        text = traceback.format_exc()
        print(text, flush=True)
        try:
            send(encode(protocol.Failure(text[-4000:])))
        except zmq.ZMQError:
            pass
        status = 1
    finally:
        # The engine first: a linker owner thread closes KVCR itself.
        for closing in (engine, host):
            if closing is not None:
                try:
                    closing.close()
                except Exception:
                    traceback.print_exc()
        sock.close()
        context.term()
    return status
