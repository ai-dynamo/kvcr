# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime and policy value types for KVCR."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, NewType

import msgspec

BlockKey = NewType("BlockKey", bytes)
PinHandle = str
PinRequestId = NewType("PinRequestId", int)
OpHandle = int
ReleaseHandle = NewType("ReleaseHandle", int)
ReleaseResult = tuple[ReleaseHandle, bool]


@dataclass(frozen=True)
class MemDescriptor:
    """Transport-addressable memory span for a pinned KV block.

    Field constraints are enforced by msgspec only when decoding or converting
    wire data. Direct construction is trusted and unvalidated.

    Endpoint, memory type, and info stay per span to keep descriptor lists flat.
    In a scenario where one key spans multiple workers and NIXL agents, grouping
    its spans by endpoint and memory type would add two hierarchy levels merely
    to factor out values typically shared by reference.
    """

    end_point_name: Annotated[str, msgspec.Meta(min_length=1)]
    mem_type: Annotated[str, msgspec.Meta(min_length=1)]
    addr: Annotated[int, msgspec.Meta(ge=0)]
    size: Annotated[int, msgspec.Meta(gt=0)]
    device_Id: Annotated[int, msgspec.Meta(ge=0)]
    info: str = ""


PinResult = tuple[PinHandle, Mapping[BlockKey, MemDescriptor | None]] | None


class OpEntryStatus(Enum):
    # TODO: Add specific statuses for timeout, abort, capacity, and unavailable sources.
    SUCCESS = "SUCCESS"
    DROPPED = "DROPPED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class OpEntryResult:
    status: OpEntryStatus
    descriptor: MemDescriptor | None = None
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
