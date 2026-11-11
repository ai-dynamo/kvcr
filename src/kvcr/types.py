# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime and policy value types for KVCR."""

from collections.abc import Mapping
from enum import Enum, IntEnum
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
    SUCCESS = "SUCCESS"
    DROPPED = "DROPPED"
    # Not deliverable (miss, no route, or the peer could not serve it); nothing
    # on this worker broke, so the caller just recomputes.
    UNAVAILABLE = "UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    # Given up deliberately, or the peer could not run the transfer.
    ABORTED = "ABORTED"
    # A fault on this worker.
    FAILED = "FAILED"


class Reason(IntEnum):
    """Why a transfer, or one of its blocks, did not complete.

    The lower-cased name is the shared vocabulary on the wire
    (``write_done.reason``, ``write_done.unserved``), in ``OpEntryResult.reason``,
    in metric labels, in logs and in NVTX payloads. Values are NVTX codes.
    """

    UNKNOWN = 0
    NONE = 1
    CALLBACK_ERROR = 2
    DUPLICATE_REQUEST = 3
    INVALID_RESULT = 4
    DEADLINE = 5
    CANCELLED = 6
    SHUTDOWN = 7
    NO_WAITERS = 8
    SUBMIT_REJECTED = 9
    SUBMIT_AMBIGUOUS = 10
    CREATE_ERROR = 11
    PROGRESS_ERROR = 12
    RELEASE_ERROR = 13
    CONTROL_ERROR = 14
    INVALID_NOTIFICATION = 15
    REMOTE_FAILURE = 16
    ROUTE_CHANGED = 17
    SOURCE_STALLED = 18
    HINT_UNAVAILABLE = 19
    HINT_CONFLICT = 20
    # Why a source left blocks out; names match the write_done unserved reasons.
    NOT_RESIDENT = 21
    PIN_FAILED = 22
    LAYOUT_MISMATCH = 23
    # Remaining write_done and pull reasons, so this enum is their only vocabulary.
    STOPPING = 24
    TRANSFER_FAILED = 25
    MALFORMED = 26
    SETUP_FAILED = 27
    REMEMBERED_MISS = 28
    HINT_INVALIDATED = 29
    REFUSED = 30
    METADATA_REQUIRED = 31
    SOURCE_RESTARTED = 32

    @property
    def wire(self) -> str:
        """Name used in write_done messages, logs and metric labels."""
        return self.name.lower()


class OpEntryResult(msgspec.Struct, frozen=True):
    status: OpEntryStatus
    # Fetch locations in KVCR-owned memory, valid until release.
    descriptors: list[MemoryRef] | None = None
    release_handle: ReleaseHandle | None = None
    # Why the block was not delivered: a lower-cased ``Reason`` name, or None
    # when the cause is unspecified. Detail only; ``status`` says what to do.
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
    # TODO: G4 is not supported; enable it with its backend.
    # G4 = "G4"


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
