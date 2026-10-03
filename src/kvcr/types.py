# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime and policy value types for KVCR."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal, NewType

BlockKey = NewType("BlockKey", bytes)
PinHandle = str
PinRequestId = NewType("PinRequestId", int)
OpHandle = int
ReleaseHandle = NewType("ReleaseHandle", int)
ReleaseResult = tuple[ReleaseHandle, bool]
LocalDramRegions = list[tuple[str, int, int]]  # name, address, size in bytes
PoolBlockLayouts = list[tuple[str, int]]  # name, block size in bytes


@dataclass(frozen=True, kw_only=True)
class RegionDescriptor:
    """Registered memory containing ``count`` fixed-size transfer elements.

    ``stride=0`` means contiguous elements. The registered extent includes
    gaps between elements. ``label`` names the pool; names must be unique
    among framework registrations.
    """

    mem_type: str = "DRAM"
    device_Id: int = 0
    addr: int
    stride: int = 0
    count: int = 1
    size: int
    label: str = ""


@dataclass(frozen=True, kw_only=True)
class MemoryRef:
    """One element in a framework's named registered buffer.

    ``label`` selects the registered pool by exact name.
    """

    end_point_name: str
    label: str = ""
    element_index: int


PinResult = tuple[PinHandle, Mapping[BlockKey, list[MemoryRef] | None]] | None


class KVCRStartupError(RuntimeError):
    """Startup could not stop native work; retain caller buffers until process exit."""


class TransferError(RuntimeError):
    """Lifecycle report for memory exposed by a failed transfer.

    References identify local framework buffers only; KVCR-owned buffers are
    omitted. Source reports retain every original key, with an empty list for
    keys using only KVCR memory. Destination reports can likewise be empty.
    ``quiesced`` clears this operation's hazard; it never makes the failed data
    valid or clears other operations.
    Handles are local to this KVCR instance and report side (source/destination).
    """

    def __init__(
        self,
        message: str,
        op_handle: OpHandle,
        *,
        state: Literal["uncertain", "quiesced"] = "uncertain",
        source_blocks: dict[BlockKey, list[MemoryRef]] | None = None,
        destination_regions: list[MemoryRef] | None = None,
    ) -> None:
        self.op_handle = op_handle
        self.state = state
        self.source_blocks = source_blocks
        self.destination_regions = destination_regions
        super().__init__(
            f"{message}: state={state}, op={op_handle}, sources={source_blocks!r}, "
            f"destinations={destination_regions!r}"
        )


class OpEntryStatus(Enum):
    # TODO: Add specific statuses for timeout, abort, capacity, and unavailable sources.
    SUCCESS = "SUCCESS"
    DROPPED = "DROPPED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class OpEntryResult:
    status: OpEntryStatus
    release_handle: ReleaseHandle | None = None

    @property
    def success(self) -> bool:
        return self.status is OpEntryStatus.SUCCESS


OpResult = tuple[OpHandle, Mapping[BlockKey, OpEntryResult]]


class QueryStatus(Enum):
    HIT = "HIT"
    FETCHING = "FETCHING"
    FETCHABLE = "FETCHABLE"
    MISS = "MISS"


class CacheTier(Enum):
    FW_G1 = "HBM"
    FW_G2 = "FW_DRAM"
    LOCAL_G2 = "DRAM"
    REMOTE_G2 = "REMOTE_G2"
    G3 = "G3"
    # TODO: G4 is not supported; enable it with its backend.
    # G4 = "G4"


@dataclass(frozen=True)
class InventoryEvent:
    keys: tuple[BlockKey, ...]
    tier: CacheTier
    removed: bool


@dataclass(frozen=True, slots=True)
class BlockMeta:
    block_key: BlockKey
    size_bytes: int
    access_count: int
    last_access: float | None
    resident_tiers: frozenset[CacheTier]
    position: int = -1


class PlacementAction(Enum):
    KEEP = "KEEP"
    DROP = "DROP"
    COPY_TO = "COPY_TO"
    MOVE_TO = "MOVE_TO"


PlacementDecision = tuple[PlacementAction, CacheTier | None]


@dataclass(frozen=True)
class PlacementFailure:
    attempted: PlacementDecision
    source: CacheTier
    reason: str
    failure_count: int


class RecoveryMirrorError(RuntimeError):
    """The recovery stream cannot describe a valid cache state."""
