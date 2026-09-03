# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Construction and integration configuration for KVCR."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .types import (
    BlockKey,
    InventoryEvent,
)

InventorySink = Callable[[InventoryEvent], None]


@dataclass(frozen=True)
class LocalDramInfo:
    address: int
    length: int
    slot_count: int


@dataclass(frozen=True)
class FrameworkDramInput:
    address: int
    length: int


# Early pinning optimization was considered, but its complexity outweighed the benefit.
@dataclass(frozen=True)
class RemoteFWDramOptions:
    eager_ctrl_connect: bool = True
    opportunistic_query: bool = False
    metadata_retry_interval_ms: int = 100


@dataclass(frozen=True, slots=True, kw_only=True)
class G3Options:
    """Bounded file-backed cache storage owned by this KVCR process."""

    paths: tuple[Path, ...]
    capacity_bytes_per_file: int
    backend: str = "GDS_MT"
    backend_options: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class KVCRBackendConfigs:
    framework_dram: FrameworkDramInput | None = None
    local_dram: LocalDramInfo | None = None
    g3: G3Options | None = None
    remote_fw_dram: RemoteFWDramOptions = field(default_factory=RemoteFWDramOptions)


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


@dataclass(frozen=True)
class KVCRConfig:
    nixl_agent_name: str
    enable_telemetry: bool = False
    operation_timeout_ms: int = 1000
    inventory_report_interval_ms: int = 10
    capacity_low_watermark_percent: float = 0
    nixl_listen_port: int | None = None


@dataclass(frozen=True)
class KVCRGuardConfig:
    kvcr_service_socket_path: str
    pool_index: int
    row_stride: int
    compatibility_digest: str
