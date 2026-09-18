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
    LocalDramRegions,
    PoolBlockLayouts,
)

InventorySink = Callable[[InventoryEvent], None]


def _validate_pool_layouts(pool_layouts: PoolBlockLayouts) -> None:
    if not pool_layouts:
        raise ValueError("pool_layouts must contain at least one pool")
    names = []
    for pool_name, block_size_bytes in pool_layouts:
        if not isinstance(pool_name, str):
            raise ValueError("pool_layouts pool name must be a string")
        if type(block_size_bytes) is not int or block_size_bytes <= 0:
            raise ValueError("pool_layouts block size must be a positive integer")
        names.append(pool_name)
    if len(names) != len(set(names)):
        raise ValueError("pool_layouts pool names must be unique")
    if len(names) > 1 and "" in names:
        raise ValueError("pool_layouts cannot use an empty name with multiple pools")


@dataclass(frozen=True)
class LocalDramOptions:
    pools: LocalDramRegions
    backend: str = "UCX"


@dataclass(frozen=True)
class FrameworkDramInput:
    address: int
    length: int


# NIXL memory segment names a framework endpoint may use. Storage segments
# (FILE, BLOCK, OBJ) are KVCR-owned tiers, never framework transfer endpoints.
FRAMEWORK_MEMORY_TYPES = frozenset({"DRAM", "VRAM"})


@dataclass(frozen=True)
class FrameworkMemoryRegion:
    """One framework-owned allocation KVCR registers as a transfer endpoint.

    ``mem_type`` and ``device_id`` are the NIXL segment type and device index
    of the allocation; a GPU allocation is ``("VRAM", cuda_device_index)``.
    ``owner`` is an opaque reference (for example the tensor whose storage this
    region describes) that KVCR keeps alive until the registration is released,
    so the address cannot be recycled while NIXL still knows it.
    """

    address: int
    length: int
    mem_type: str = "DRAM"
    device_id: int = 0
    owner: object | None = field(default=None, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.address) is not int or self.address <= 0:
            raise ValueError("framework region address must be a positive integer")
        if type(self.length) is not int or self.length <= 0:
            raise ValueError("framework region length must be a positive integer")
        if self.mem_type not in FRAMEWORK_MEMORY_TYPES:
            raise ValueError(
                "framework region mem_type must be one of "
                f"{sorted(FRAMEWORK_MEMORY_TYPES)}, got {self.mem_type!r}"
            )
        if type(self.device_id) is not int or self.device_id < 0:
            raise ValueError("framework region device_id must be a non-negative int")

    @property
    def endpoint(self) -> tuple[str, int]:
        return (self.mem_type, self.device_id)

    @property
    def end(self) -> int:
        return self.address + self.length


def resolve_framework_regions(
    backend_configs: "KVCRBackendConfigs",
) -> tuple[FrameworkMemoryRegion, ...]:
    """Return the framework endpoints to register, deduplicated and validated.

    ``framework_dram`` is the legacy single DRAM region; ``framework_regions``
    is the typed list. Configuring both is refused rather than merged, so one
    allocation can never be registered twice through two spellings. Exact
    duplicates (two views of one allocation) collapse to one registration;
    partially overlapping regions of the same endpoint are refused because
    NIXL cannot tell which registration a descriptor inside the overlap means.
    """
    legacy = backend_configs.framework_dram
    typed = tuple(backend_configs.framework_regions)
    if legacy is not None and typed:
        raise ValueError(
            "configure either framework_dram or framework_regions, not both"
        )
    if legacy is not None:
        typed = (FrameworkMemoryRegion(legacy.address, legacy.length),)
    if not all(isinstance(region, FrameworkMemoryRegion) for region in typed):
        raise TypeError("framework_regions must contain FrameworkMemoryRegion values")

    unique: list[FrameworkMemoryRegion] = []
    seen: set[tuple[int, int, str, int]] = set()
    for region in typed:
        identity = (region.address, region.length, region.mem_type, region.device_id)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(region)

    by_endpoint: dict[tuple[str, int], list[FrameworkMemoryRegion]] = {}
    for region in unique:
        by_endpoint.setdefault(region.endpoint, []).append(region)
    for regions in by_endpoint.values():
        ordered = sorted(regions, key=lambda region: region.address)
        for left, right in zip(ordered, ordered[1:]):
            if left.end > right.address:
                raise ValueError(
                    "framework regions must not overlap: "
                    f"[{left.address:#x}, {left.end:#x}) and "
                    f"[{right.address:#x}, {right.end:#x}) on {left.endpoint}"
                )
    return tuple(unique)


# Early pinning optimization was considered, but its complexity outweighed the benefit.
@dataclass(frozen=True)
class RemoteFWDramOptions:
    eager_ctrl_connect: bool = True
    opportunistic_query: bool = False
    metadata_retry_interval_ms: int = 100
    backend: str = "UCX"


@dataclass(frozen=True, slots=True, kw_only=True)
class G3Options:
    """Bounded file-backed cache storage owned by this KVCR process."""

    paths: tuple[Path, ...]
    capacity_bytes_per_file: int
    backend: str = "GDS_MT"
    backend_options: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class KVCRBackendConfigs:
    # Legacy single DRAM endpoint; prefer framework_regions. Setting both is refused.
    framework_dram: FrameworkDramInput | None = None
    framework_regions: tuple[FrameworkMemoryRegion, ...] = ()
    local_dram: LocalDramOptions | None = None
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
    pool_layouts: PoolBlockLayouts
    enable_telemetry: bool = False
    operation_timeout_ms: int = 1000
    abandon_timeout_ms: int = 5000
    capacity_low_watermark_percent: float = 0
    nixl_listen_port: int | None = None


@dataclass(frozen=True)
class KVCRGuardConfig:
    kvcr_service_socket_path: str
    guard_index: int
    compatibility_digest: str
