# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime and policy value types for KVCR."""

from collections.abc import Mapping
from enum import Enum
from typing import Literal, NewType

import msgspec

BlockKey = NewType("BlockKey", bytes)
PinHandle = str
PinRequestId = NewType("PinRequestId", int)
OpHandle = int
ReleaseHandle = NewType("ReleaseHandle", int)
ReleaseResult = tuple[ReleaseHandle, bool]
LocalDramRegions = list[tuple[str, int, int]]  # name, address, size in bytes
PoolBlockLayouts = list[tuple[str, int]]  # name, block size in bytes


class RegionDescriptor(msgspec.Struct, frozen=True, kw_only=True):
    """Registered memory containing ``count`` fixed-size transfer elements.

    ``stride=0`` means contiguous elements. The registered extent includes
    gaps between elements. Framework registration names must be unique: ``pool``,
    ``pool:part``, or ``pool:*`` to share a buffer across parts.
    """

    mem_type: str = "DRAM"
    device_Id: int = 0
    addr: int
    stride: int = 0
    count: int = 1
    size: int
    label: str = ""


class MemoryRef(msgspec.Struct, frozen=True, kw_only=True):
    """One element in a named registered buffer.

    ``label`` identifies the key piece and selects an exact framework registration,
    or ``pool:*`` for unmatched ``pool:part`` labels. ``element_index`` is relative
    to the selected buffer. The ``*`` part is reserved for registration.
    """

    end_point_name: str
    label: str = ""
    element_index: int


PinResult = tuple[PinHandle, Mapping[BlockKey, list[MemoryRef] | None]] | None


class KVCRStartupError(RuntimeError):
    """Startup could not stop native work; retain caller buffers until process exit."""


class TransferError(RuntimeError):
    """Lifecycle report for memory exposed by a failed transfer.

    References identify local buffers. ``source_owners`` identifies each key's
    owner; ``destination_owner`` applies to all destination regions. Owners are
    ``framework`` or ``kvcr`` and select the corresponding registration namespace.
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
        source_owners: dict[BlockKey, Literal["framework", "kvcr"]] | None = None,
        destination_regions: list[MemoryRef] | None = None,
        destination_owner: Literal["framework", "kvcr"] | None = None,
    ) -> None:
        self.op_handle = op_handle
        self.state = state
        self.source_blocks = source_blocks
        self.source_owners = source_owners
        self.destination_regions = destination_regions
        self.destination_owner = destination_owner
        super().__init__(
            f"{message}: state={state}, op={op_handle}, sources={source_blocks!r}, "
            f"source_owners={source_owners!r}, destinations={destination_regions!r}, "
            f"destination_owner={destination_owner!r}"
        )


class OpEntryStatus(Enum):
    """Outcome for one block."""

    SUCCESS = "SUCCESS"
    DROPPED = "DROPPED"
    UNAVAILABLE = "UNAVAILABLE"  # Expected miss or no usable source.
    TIMEOUT = "TIMEOUT"  # Operation deadline expired.
    ABORTED = "ABORTED"  # Cancelled or interrupted.
    FAILED = "FAILED"  # Unexpected error on either worker.


class Reason(str, Enum):
    """Why a transfer, or one of its blocks, did not complete."""

    UNKNOWN = "unknown"
    NONE = "none"
    CALLBACK_ERROR = "callback_error"
    DUPLICATE_REQUEST = "duplicate_request"
    INVALID_RESULT = "invalid_result"
    DEADLINE = "deadline"
    CANCELLED = "cancelled"
    SHUTDOWN = "shutdown"
    NO_WAITERS = "no_waiters"
    SUBMIT_REJECTED = "submit_rejected"
    SUBMIT_AMBIGUOUS = "submit_ambiguous"
    CREATE_ERROR = "create_error"
    PROGRESS_ERROR = "progress_error"
    RELEASE_ERROR = "release_error"
    CONTROL_ERROR = "control_error"
    INVALID_NOTIFICATION = "invalid_notification"
    REMOTE_FAILURE = "remote_failure"
    ROUTE_CHANGED = "route_changed"
    HINT_UNAVAILABLE = "hint_unavailable"
    HINT_CONFLICT = "hint_conflict"
    NOT_RESIDENT = "not_resident"
    PIN_FAILED = "pin_failed"
    LAYOUT_MISMATCH = "layout_mismatch"
    STOPPING = "stopping"
    TRANSFER_FAILED = "transfer_failed"
    MALFORMED = "malformed"
    SETUP_FAILED = "setup_failed"
    REMEMBERED_MISS = "remembered_miss"
    HINT_INVALIDATED = "hint_invalidated"
    REFUSED = "refused"
    METADATA_REQUIRED = "metadata_required"
    SOURCE_RESTARTED = "source_restarted"

    def __str__(self) -> str:
        return self.value


class OpEntryResult(msgspec.Struct, frozen=True):
    status: OpEntryStatus
    # Fetch locations in KVCR-owned memory, valid until release.
    descriptors: list[MemoryRef] | None = None
    release_handle: ReleaseHandle | None = None
    # Why the block was not delivered: a lower-cased ``Reason`` name, or None.
    reason: str | None = None

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
    G4 = "G4"


class InventoryEvent(msgspec.Struct, frozen=True):
    keys: tuple[BlockKey, ...]
    tier: CacheTier
    removed: bool


class BlockMeta(msgspec.Struct, frozen=True):
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


class PlacementFailure(msgspec.Struct, frozen=True):
    attempted: PlacementDecision
    source: CacheTier
    reason: str
    failure_count: int


class RecoveryMirrorError(RuntimeError):
    """The recovery stream cannot describe a valid cache state."""
