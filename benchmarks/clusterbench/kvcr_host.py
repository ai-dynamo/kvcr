# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Process setup around the engine: environment, CPU topology and affinity,
and which KVCR tree was imported."""

from __future__ import annotations

import os
import resource
import subprocess
import threading
from pathlib import Path


def prepare_environment() -> None:
    """Process-wide setup that must precede the first NIXL agent."""
    # NIXL rejects a UCX_TLS list without cuda* when NVML sees GPUs; hiding the
    # devices and keeping cuda_copy in the list satisfies both.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("UCX_TLS", "self,sm,tcp,cuda_copy")
    # vLLM: the CPU platform whatever NVML sees, and no model hub.
    os.environ["VLLM_TARGET_DEVICE"] = "cpu"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def limits() -> dict[str, int]:
    nofile = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    memlock = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    return {"nofile": nofile, "memlock": -1 if memlock < 0 else memlock}


def kvcr_identity() -> tuple[str, str]:
    """Where ``kvcr`` was imported from and that tree's revision."""
    import kvcr

    path = Path(kvcr.__file__).resolve()
    root = path.parents[2]
    marker = root / ".kvcr-sha"
    if marker.exists():
        return str(path), marker.read_text().strip()
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", "src"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        return str(path), sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return str(path), "unknown"


def physical_cores(cpus) -> dict[tuple[int, int], list[int]]:
    """The CPUs by physical core: (package, core id) -> its hyperthreads."""
    cores: dict[tuple[int, int], list[int]] = {}
    for cpu in sorted(cpus):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = (
                int((base / "physical_package_id").read_text()),
                int((base / "core_id").read_text()),
            )
        except (OSError, ValueError):
            key = (0, cpu)
        cores.setdefault(key, []).append(cpu)
    return cores


def core_groups(cpus) -> list[list[int]]:
    """The CPUs grouped by physical core, in (package, core) order."""
    return [group for _, group in sorted(physical_cores(cpus).items())]


def plan_affinity(cpus: list[int]) -> tuple[int | None, int | None, list[int]]:
    """Split a worker's CPUs: engine thread, kvcr-progress, native threads.

    The engine and progress threads each get the first hyperthread of their
    own physical core and its sibling is left idle; everything else (NIXL, UCX
    and ZMQ threads) shares the remaining cores. With fewer than three cores
    nothing is dedicated.
    """
    groups = core_groups(cpus)
    if len(groups) < 3:
        return None, None, sorted(cpus)
    native = [cpu for group in groups[2:] for cpu in group]
    return groups[0][0], groups[1][0], native


def auto_cpus(local_id: int, per_node: int) -> tuple[list[int], int]:
    """CPUs for worker ``local_id`` of ``per_node`` on this machine, and its socket.

    Workers alternate between sockets and each gets a contiguous block of
    physical cores (all hyperthreads). The first two cores of every socket
    are left for the OS, the coordinator and profilers.
    """
    reserve = 2
    sockets: dict[int, list[list[int]]] = {}
    for (package, _), group in sorted(physical_cores(os.sched_getaffinity(0)).items()):
        sockets.setdefault(package, []).append(group)
    packages = sorted(sockets)
    socket = local_id % len(packages)
    slot = local_id // len(packages)
    per_socket = -(-per_node // len(packages))
    cores = sockets[packages[socket]]
    usable = cores[reserve:] if len(cores) >= reserve + 3 * per_socket else cores
    width = len(usable) // per_socket
    if width < 3:
        return [], socket
    block = usable[slot * width : (slot + 1) * width]
    return sorted(cpu for core in block for cpu in core), socket


def confine_new_threads(cpus: list[int]) -> dict:
    """Before the engine starts: threads KVCR and NIXL start inherit the mask
    left here. Returns the plan ``pin_engine_threads`` completes."""
    engine_cpu, progress_cpu, native = plan_affinity(cpus)
    if native:
        os.sched_setaffinity(0, native)
    return {"engine": engine_cpu, "progress": progress_cpu, "native": native}


def pin_engine_threads(affinity: dict) -> None:
    """After the engine started: this thread and KVCR's progress thread each
    get their own core."""
    if affinity["engine"] is None:
        return
    os.sched_setaffinity(0, {affinity["engine"]})
    for thread in threading.enumerate():
        if thread.name == "kvcr-progress" and thread.native_id:
            os.sched_setaffinity(thread.native_id, {affinity["progress"]})
