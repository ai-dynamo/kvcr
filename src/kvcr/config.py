# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Construction and integration configuration for KVCR."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, Protocol

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
    # "fd": receive the pool file from the service instead of opening its path,
    # so the worker needs no pool-directory mount.
    attach_mode: Literal["path", "fd"] = "path"
