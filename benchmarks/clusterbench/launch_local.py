# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run a whole cell on one machine: the coordinator here, workers as children.

"Nodes" are only labels locally (they drive the same-node/cross-node split and
router behaviour); every transfer goes over UCX shared memory or TCP.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

from .config import RunConfig
from .coordinator import Coordinator

BENCHMARKS = Path(__file__).resolve().parents[1]


def _free_port_base(count: int) -> int:
    """A base port with ``count`` consecutive ports free right now."""
    for base in range(21000, 31000, max(64, count)):
        try:
            probes = []
            for port in range(base, base + count):
                probe = socket.socket()
                probes.append(probe)
                probe.bind(("127.0.0.1", port))
        except OSError:
            continue
        finally:
            for probe in probes:
                probe.close()
        return base
    raise RuntimeError("no free port range")


def _cpu_sets(workers: int, cores_per_worker: int) -> list[list[int]]:
    """Disjoint blocks of physical cores per worker, or no pinning if short."""
    cores: dict[tuple[int, int], list[int]] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = (
                int((base / "physical_package_id").read_text()),
                int((base / "core_id").read_text()),
            )
        except (OSError, ValueError):
            key = (0, cpu)
        cores.setdefault(key, []).append(cpu)
    groups = [cores[key] for key in sorted(cores)]
    if cores_per_worker < 3 or len(groups) < workers * cores_per_worker:
        return [[] for _ in range(workers)]
    return [
        sorted(
            cpu
            for group in groups[w * cores_per_worker : (w + 1) * cores_per_worker]
            for cpu in group
        )
        for w in range(workers)
    ]


def run_local(
    config: RunConfig,
    out_dir: str | Path,
    *,
    nodes: int = 2,
    per_node: int = 2,
    kvcr_src: str | None = None,
    arm: str = "",
    cores_per_worker: int = 0,
) -> dict:
    out = Path(out_dir)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    workers = nodes * per_node
    coordinator = Coordinator(config, out, arm=arm)
    endpoint = coordinator.bind("tcp://127.0.0.1:*")
    base = _free_port_base(2 * workers)
    cpu_sets = _cpu_sets(workers, cores_per_worker)

    env = dict(os.environ)
    paths = [str(BENCHMARKS)]
    if kvcr_src:
        paths.append(str(Path(kvcr_src).resolve() / "src"))
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)

    children: list[subprocess.Popen] = []
    try:
        for index in range(workers):
            node, local_id = divmod(index, per_node)
            log = open(logs / f"w-node{node}-{local_id}.log", "w")
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "clusterbench",
                        "worker",
                        "--coordinator",
                        endpoint,
                        "--node",
                        f"node{node}",
                        "--local-id",
                        str(local_id),
                        "--bind-host",
                        "127.0.0.1",
                        "--control-port",
                        str(base + index),
                        "--nixl-port",
                        str(base + workers + index),
                        "--cpus",
                        ",".join(map(str, cpu_sets[index])),
                        "--arm",
                        arm,
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
            log.close()
        return coordinator.run(expect=workers)
    finally:
        coordinator.close()
        for child in children:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
