# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Build one real KVCR instance on host memory, the way an engine does.

``Host`` mirrors ``KVCRSecondaryTierManager.__init__`` in vLLM: one unnamed
pool, the primary CPU tier as the single framework region, an optional
anonymous mmap as the KVCR-owned DRAM pool, a ZMQ peer control channel.
Unlike vLLM's default, KVCR's telemetry is on: the report reads it.

Small probes keep this working against older KVCR trees so A/B arms can differ.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import resource
import struct
import subprocess
import threading
from pathlib import Path

from .config import EngineConfig, TierConfig
from .ids import KeyAdapter, keys_to_ids

_STAMP = struct.Struct(">Q")


def prepare_environment() -> None:
    """Process-wide setup that must precede the first NIXL agent."""
    # NIXL rejects a UCX_TLS list without cuda* when NVML sees GPUs; hiding the
    # devices and keeping cuda_copy in the list satisfies both.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("UCX_TLS", "self,sm,tcp,cuda_copy")
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


def _filtered(struct_type, **kwargs):
    """Construct a msgspec struct with only the fields this KVCR tree has."""
    fields = set(struct_type.__struct_fields__)
    return struct_type(**{k: v for k, v in kwargs.items() if k in fields})


class Host:
    """A started KVCR plus the memory it was given."""

    def __init__(
        self,
        tiers: TierConfig,
        engine: EngineConfig,
        *,
        agent_name: str,
        bind_host: str,
        control_port: int,
        nixl_port: int,
        framework,
        stats_factory,
        cpus: list[int],
    ) -> None:
        from kvcr import KVCR, KVCRBindings
        from kvcr.config import (
            KVCRBackendConfigs,
            KVCRConfig,
            LocalDramOptions,
            RemoteFWDramOptions,
        )
        from kvcr.types import RegionDescriptor

        try:
            from kvcr import ZmqPeerControlChannel
        except ImportError:  # older trees
            from kvcr.control_channels import ZmqPeerControlChannel

        self.agent_name = agent_name
        self.block_bytes = tiers.block_bytes
        self.primary = mmap.mmap(-1, tiers.primary_blocks * tiers.block_bytes)
        self._primary_view = memoryview(self.primary)
        primary_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.primary))
        self.g2 = None
        local_dram = None
        if tiers.g2_blocks:
            self.g2 = mmap.mmap(-1, tiers.g2_blocks * tiers.block_bytes)
            g2_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.g2))
            local_dram = LocalDramOptions(pools=[("", g2_addr, len(self.g2))])

        self.control = ZmqPeerControlChannel(bind_host, control_port, bind_host)

        engine_cpu, progress_cpu, native = plan_affinity(cpus)
        if native:
            # Threads KVCR and NIXL start inherit this mask.
            os.sched_setaffinity(0, native)

        self.kvcr = KVCR(
            _filtered(
                KVCRConfig,
                nixl_agent_name=agent_name,
                nixl_listen_port=nixl_port,
                pool_layouts=[("", tiers.block_bytes)],
                enable_telemetry=True,
                operation_timeout_ms=engine.operation_timeout_ms,
                abandon_timeout_ms=engine.abandon_timeout_ms,
            ),
            _filtered(
                KVCRBindings,
                request_pin=framework.request_pin,
                poll_pin_results=framework.poll_pin_results,
                release_pin=framework.release_pin,
                cancel_pin_request=framework.cancel_pin_request,
                framework_control=self.control,
                key_adapter=KeyAdapter(),
                inventory_sink=framework.record_inventory,
                stats_factory=stats_factory,
                on_resilience_event=framework.on_resilience_event,
            ),
            _filtered(
                KVCRBackendConfigs,
                framework_regions=[
                    RegionDescriptor(
                        addr=primary_addr,
                        size=tiers.block_bytes,
                        count=tiers.primary_blocks,
                    )
                ],
                local_dram=local_dram,
                remote_fw_dram=_filtered(
                    RemoteFWDramOptions,
                    # vLLM's adapter defaults.
                    eager_ctrl_connect=True,
                    opportunistic_query=True,
                    metadata_retry_interval_ms=100,
                ),
            ),
        )

        self.affinity = {
            "engine": engine_cpu,
            "progress": progress_cpu,
            "native": native,
        }
        if engine_cpu is not None:
            os.sched_setaffinity(0, {engine_cpu})
            for thread in threading.enumerate():
                if thread.name == "kvcr-progress" and thread.native_id:
                    os.sched_setaffinity(thread.native_id, {progress_cpu})

    def stamp(self, slot: int, bid: int) -> None:
        _STAMP.pack_into(self._primary_view, slot * self.block_bytes, bid)

    def stamp_many(self, slots, keys) -> None:
        """Stamp each slot with the block id of its key."""
        pack, view, size = _STAMP.pack_into, self._primary_view, self.block_bytes
        for slot, bid in zip(slots, keys_to_ids(keys)):
            pack(view, slot * size, bid)

    def stamp_at(self, slot: int) -> int:
        return _STAMP.unpack_from(self._primary_view, slot * self.block_bytes)[0]

    def close(self) -> None:
        self.kvcr.close()
