# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Construction and integration configuration for KVCR."""

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

import msgspec

from .types import (
    BlockKey,
    InventoryEvent,
    LocalDramRegions,
    PoolBlockLayouts,
    RegionDescriptor,
)

InventorySink = Callable[[InventoryEvent], None]


def _validate_pool_layouts(pool_layouts: PoolBlockLayouts) -> None:
    if not pool_layouts:
        raise ValueError("pool_layouts must contain at least one pool")
    names = []
    for pool_name, block_size_bytes in pool_layouts:
        if not isinstance(pool_name, str):
            raise ValueError("pool_layouts pool name must be a string")
        if ":" in pool_name:
            raise ValueError("pool_layouts pool names cannot contain a colon")
        if type(block_size_bytes) is not int or block_size_bytes <= 0:
            raise ValueError("pool_layouts block size must be a positive integer")
        names.append(pool_name)
    if len(names) != len(set(names)):
        raise ValueError("pool_layouts pool names must be unique")
    if len(names) > 1 and "" in names:
        raise ValueError("pool_layouts cannot use an empty name with multiple pools")


class LocalDramOptions(msgspec.Struct, frozen=True):
    pools: LocalDramRegions
    backend: str = "UCX"


# Early pinning optimization was considered, but its complexity outweighed the benefit.
class RemoteFWDramOptions(msgspec.Struct, frozen=True):
    eager_ctrl_connect: bool = True
    opportunistic_query: bool = False
    metadata_retry_interval_ms: int = 100
    backend: str = "UCX"


class RemoteFWGPUOptions(msgspec.Struct, frozen=True, kw_only=True):
    """An isolated GPU framework-memory endpoint with an explicit transport.

    Host transports in ``ucx_tls`` carry control metadata only. CUDA copy,
    GDR copy and alternate GPU transports are excluded, so a missing NVLink
    route must fail rather than stage KV payloads in host memory. Peers share the
    explicit compatibility domain. The default profile requires the same host;
    MNNVL requires fabric-capable allocations in the same observed fabric domain.
    Its TCP transport carries control messages, with software RMA and TCP puts
    disabled by the dedicated native backend. Framework callbacks fence producer
    writes before returning pin results and retain CUDA allocations until release.
    """

    compatibility_domain: str
    ucx_tls: tuple[str, ...] = ("cuda_ipc", "posix", "self")
    eager_ctrl_connect: bool = True
    opportunistic_query: bool = False
    metadata_retry_interval_ms: int = 100
    backend: str = "UCX"
    gpu_transport: str = "cuda_ipc"
    fabric_domain: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.compatibility_domain, str)
            or not self.compatibility_domain
            or self.compatibility_domain.strip() != self.compatibility_domain
            or "\x00" in self.compatibility_domain
        ):
            raise ValueError("GPU compatibility_domain must be explicit and non-empty")
        if self.gpu_transport not in ("cuda_ipc", "mnnvl"):
            raise ValueError("GPU transport must be cuda_ipc or mnnvl")
        if self.gpu_transport == "mnnvl":
            match = (
                re.fullmatch(
                    r"mnnvl:([0-9a-f]{32}):(0|[1-9][0-9]{0,9})", self.fabric_domain
                )
                if isinstance(self.fabric_domain, str)
                else None
            )
            if match is None or int(match[1], 16) == 0 or int(match[2]) > 0xFFFFFFFF:
                raise ValueError(
                    "MNNVL requires an observed nonzero fabric UUID and clique"
                )
            if self.ucx_tls != ("cuda_ipc", "tcp", "self"):
                raise ValueError("MNNVL ucx_tls must be exactly cuda_ipc,tcp,self")
        elif self.fabric_domain is not None:
            raise ValueError("CUDA IPC does not accept an MNNVL fabric domain")
        allowed = (
            {"cuda_ipc", "tcp", "self"}
            if self.gpu_transport == "mnnvl"
            else {"cuda_ipc", "posix", "sysv", "self"}
        )
        if (
            not isinstance(self.ucx_tls, tuple)
            or not self.ucx_tls
            or any(
                type(token) is not str or token not in allowed for token in self.ucx_tls
            )
            or len(set(self.ucx_tls)) != len(self.ucx_tls)
            or "cuda_ipc" not in self.ucx_tls
        ):
            raise ValueError(
                "GPU ucx_tls must select CUDA IPC and only host control transports"
            )
        if self.backend != "UCX":
            raise ValueError("direct GPU transfer currently requires the UCX backend")


class G3Options(msgspec.Struct, frozen=True, kw_only=True):
    """Bounded file-backed cache storage owned by this KVCR process.

    Per-file capacity is rounded down to complete slots and must fit at least one.
    """

    paths: tuple[Path, ...]
    capacity_bytes_per_file: int
    backend: str = "GDS_MT"
    backend_options: Mapping[str, str] = msgspec.field(default_factory=dict)


class KVCRBackendConfigs(msgspec.Struct, frozen=True):
    framework_regions: list[RegionDescriptor] = msgspec.field(default_factory=list)
    local_dram: LocalDramOptions | None = None
    g3: G3Options | None = None
    remote_fw_dram: RemoteFWDramOptions = msgspec.field(
        default_factory=RemoteFWDramOptions
    )
    remote_fw_gpu: RemoteFWGPUOptions | None = None


class TelemetryStats(Protocol):
    """Telemetry seam between KVCR and a framework-specific wrapper."""

    def increase_counter(
        self,
        name: str,
        value: int | float,
        labelvalues: tuple[str, ...] = (),
    ) -> None: ...

    def set_gauge(
        self,
        name: str,
        value: int | float,
        labelvalues: tuple[str, ...] = (),
    ) -> None: ...

    def observe_histogram(
        self,
        name: str,
        value: int | float,
        labelvalues: tuple[str, ...] = (),
    ) -> None: ...

    # Wrappers call these on returned interval snapshots. They aggregate
    # snapshots and reset their accumulator; KVCR only records and replaces
    # its current snapshot.
    def reduce(self) -> dict[str, int | float]: ...

    def is_empty(self) -> bool: ...


class FrameworkControl(Protocol):
    def send(self, endpoint: str, message: bytes) -> bool: ...

    def recv(self) -> list[bytes]: ...


class KeyAdapter(Protocol):
    """Framework-specific key conversion."""

    def encode(self, framework_key: object) -> BlockKey: ...

    def decode(self, key: BlockKey) -> int | bytes: ...


class KVCRConfig(msgspec.Struct, frozen=True):
    nixl_agent_name: str
    pool_layouts: PoolBlockLayouts
    enable_telemetry: bool = False
    operation_timeout_ms: int = 1000
    abandon_timeout_ms: int = 5000
    capacity_low_watermark_percent: float = 0
    nixl_listen_port: int | None = None
    startup_timeout_seconds: int | None = None


class KVCRGuardConfig(msgspec.Struct, frozen=True):
    kvcr_service_socket_path: str
    guard_index: int
    compatibility_digest: str
