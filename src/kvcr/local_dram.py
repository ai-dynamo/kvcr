# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KVCR-owned local DRAM slots, claims, and transfers."""

import logging
from collections import deque
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, cast

from .config import LocalDramInfo
from .policy_runtime import _EvictionQueue
from .progress import _KVCRProgress, _Op, _OpId, _ProgressOp
from .types import (
    BlockKey,
    CacheTier,
    MemDescriptor,
    OpEntryResult,
    OpEntryStatus,
    OpHandle,
    PlacementAction,
    ReleaseHandle,
    ReleaseResult,
)

if TYPE_CHECKING:
    from .core import _BlockRecord, _KVCRCore

logger = logging.getLogger(__name__)

_Clock = Callable[[], float]


class _LocalDramState(Enum):
    FILLING = auto()
    READY = auto()
    DISCARDING = auto()


@dataclass
class _LocalDramArena:
    address: int
    length: int
    slot_size: int
    free_slots: deque[int]
    evictable: _EvictionQueue = field(default_factory=_EvictionQueue)
    unscored: set[BlockKey] = field(default_factory=set)
    capacity_waiters: deque["_CapacityWaiter"] = field(default_factory=deque)
    capacity_eviction_key: BlockKey | None = None
    resuming_capacity_waiters: bool = False

    @property
    def slot_count(self) -> int:
        return self.length // self.slot_size


@dataclass(slots=True)
class _LocalDramResidency:
    slot: int
    state: _LocalDramState
    claim_count: int = 0
    retire_on_release: bool = False
    # None is the legacy single-arena representation used by recovery records.
    # Multi-arena residencies name their unique size class explicitly.
    arena_size: int | None = None


@dataclass
class _PendingResidencyOp(_Op):
    deadline: float
    claim_on_ready: bool
    request_id: str | None = None
    results: dict[BlockKey, OpEntryResult] = field(default_factory=dict)
    remote_fill_keys: set[BlockKey] = field(default_factory=set)
    capacity_waiters: set[BlockKey] = field(default_factory=set)


@dataclass
class _PendingDeliverOp(_Op):
    deadline: float
    destinations: Mapping[BlockKey, MemDescriptor]
    results: dict[BlockKey, OpEntryResult] = field(default_factory=dict)
    active_keys: set[BlockKey] = field(default_factory=set)


@dataclass(frozen=True)
class _CapacityWaiter:
    op: _PendingResidencyOp
    key: BlockKey
    source: MemDescriptor | CacheTier


@dataclass
class _LocalCopyOp(_ProgressOp):
    deliver_op_id: _OpId | None
    ordered_keys: tuple[BlockKey, ...]
    local_slots: tuple[int, ...]
    arena_size: int
    src_descriptors: tuple[MemDescriptor, ...]
    dst_descriptors: tuple[MemDescriptor, ...]
    deadline: float
    clock: _Clock = field(repr=False, compare=False)
    started_at: float | None = field(repr=False, compare=False)
    transfer_id: int | None = None
    success: bool = False
    cancellation_requested: bool = False

    def progress(
        self, progress: _KVCRProgress, event: object | None
    ) -> tuple[bool, bool]:
        if event is not None:
            raise RuntimeError(f"unexpected local-copy event: {event!r}")
        observed_work = False
        if self.transfer_id is None:
            if self.clock() >= self.deadline:
                return True, True
            try:
                transfer_id, submitted = progress.submit_transfer(
                    "WRITE",
                    self.src_descriptors,
                    self.dst_descriptors,
                    remote_side_agent=progress.nixl_agent_name,
                )
                self.transfer_id = transfer_id
                self.cancellation_requested = not submitted
                observed_work = True
            except Exception:
                logger.warning("KVCR local transfer submission failed", exc_info=True)
                return True, True

        transfer_id = self.transfer_id
        if transfer_id is None:
            raise RuntimeError(f"KVCR local copy {self.op_id!r} lost transfer")
        if not self.cancellation_requested and self.clock() >= self.deadline:
            self.cancellation_requested = True
            observed_work = True
        result = progress.poll_transfer(
            transfer_id,
            cancellation_requested=self.cancellation_requested,
        )
        if result is None:
            return False, observed_work
        self.transfer_id = None
        self.success, _ = result
        return True, True

    def close(self, progress: _KVCRProgress) -> bool:
        if self.transfer_id is not None:
            if not progress.cancel_transfer(self.transfer_id):
                return False
            self.transfer_id = None
        return True


class _LocalDram:
    """Main-thread metadata for fixed-slot DRAM arenas keyed by slot size."""

    def __init__(
        self,
        kvcr: "_KVCRCore",
        regions: Collection[LocalDramInfo],
    ) -> None:
        self._kvcr = kvcr
        arenas: list[_LocalDramArena] = []
        arena_by_size: dict[int, _LocalDramArena] = {}
        memory_ranges: list[tuple[int, int]] = []
        for region in regions:
            if region.address <= 0:
                raise ValueError("local DRAM address must be positive")
            if region.length <= 0:
                raise ValueError("local DRAM length must be positive")
            if region.slot_count <= 0:
                raise ValueError("local DRAM slot_count must be positive")
            if region.length % region.slot_count:
                raise ValueError("local DRAM length must divide evenly into slots")
            slot_size = region.length // region.slot_count
            if slot_size in arena_by_size:
                raise ValueError(f"duplicate local DRAM arena slot size {slot_size}")
            start, end = region.address, region.address + region.length
            if any(
                start < other_end and other_start < end
                for other_start, other_end in memory_ranges
            ):
                raise ValueError("local DRAM arena memory regions must not overlap")
            arena = _LocalDramArena(
                address=region.address,
                length=region.length,
                slot_size=slot_size,
                free_slots=deque(range(region.slot_count)),
            )
            arenas.append(arena)
            arena_by_size[slot_size] = arena
            memory_ranges.append((start, end))
        if not arenas:
            raise ValueError("at least one local DRAM arena is required")

        self._arenas = tuple(arenas)
        self._arena_by_size = arena_by_size
        self._single_arena = arenas[0] if len(arenas) == 1 else None
        # Private legacy attributes remain available for the single-arena path.
        self._address = arenas[0].address if len(arenas) == 1 else None
        self._length = arenas[0].length if len(arenas) == 1 else None
        self._slot_size = arenas[0].slot_size if len(arenas) == 1 else None
        self._pending_residency_ops: dict[_OpId, _PendingResidencyOp] = {}
        self._pending_deliver_ops: dict[_OpId, _PendingDeliverOp] = {}
        self._public_claims: dict[
            ReleaseHandle, tuple[BlockKey, _LocalDramResidency]
        ] = {}
        self._next_copy_id = 1
        self._next_release_handle = 1
        # A no-op until something attaches: the tiers publish residency
        # changes unconditionally, and only recovery cares to hear them.
        self._residency_observer: Callable[[BlockKey, "_BlockRecord"], None] = (
            lambda key, record: None
        )

    @property
    def memory_region(self) -> tuple[int, int]:
        arena = self._require_single_arena()
        return arena.address, arena.length

    @property
    def memory_regions(self) -> tuple[tuple[int, int], ...]:
        return tuple((arena.address, arena.length) for arena in self._arenas)

    @property
    def _free_slots(self) -> deque[int]:
        return self._require_single_arena().free_slots

    @property
    def _evictable(self) -> _EvictionQueue:
        return self._require_single_arena().evictable

    @property
    def _unscored(self) -> set[BlockKey]:
        return self._require_single_arena().unscored

    @property
    def _capacity_waiters(self) -> deque[_CapacityWaiter]:
        return self._require_single_arena().capacity_waiters

    @property
    def _capacity_eviction_key(self) -> BlockKey | None:
        return self._require_single_arena().capacity_eviction_key

    @_capacity_eviction_key.setter
    def _capacity_eviction_key(self, key: BlockKey | None) -> None:
        self._require_single_arena().capacity_eviction_key = key

    @property
    def _total_slots(self) -> int:
        return sum(arena.slot_count for arena in self._arenas)

    def observe_residency(
        self, observer: Callable[[BlockKey, "_BlockRecord"], None]
    ) -> None:
        self._residency_observer = observer

    def adopt_recovery_slots(self, records: Mapping[BlockKey, "_BlockRecord"]) -> None:
        """Take the rows already-recovered records name, before the core starts.

        The records carry the residencies; this only makes the allocator agree
        with them. Ranking them is rank_recovered, which needs the policy to have
        seen every block first.
        """
        arena = self._require_single_arena()
        slot_count = arena.slot_count
        occupied: set[int] = set()
        for record in records.values():
            residency = record.local_dram
            if residency is None:
                continue
            slot = residency.slot
            if (
                residency.state is not _LocalDramState.READY
                or type(slot) is not int
                or not 0 <= slot < slot_count
                or slot in occupied
            ):
                raise ValueError("invalid local DRAM recovery slots")
            occupied.add(slot)
        arena.free_slots = deque(
            slot for slot in range(slot_count) if slot not in occupied
        )

    def rank_recovered(self, records: Mapping[BlockKey, "_BlockRecord"]) -> None:
        """Make recovered rows evictable, once the policy can score them.

        Separate from adopt_recovery_slots because a score is asked of the policy,
        and the policy only knows a block once it has been admitted. Without this a
        pool recovered full has no free row and no victim, so it refuses every
        deposit until a reader happens to release one of the recovered rows.
        """
        for key, record in records.items():
            if record.local_dram is not None:
                self._make_evictable(key)

    def telemetry_state(self) -> dict[str, int]:
        total_slots = self._total_slots
        free_slots = sum(len(arena.free_slots) for arena in self._arenas)
        state = {
            "local_g2_total_slots": total_slots,
            "local_g2_free_slots": free_slots,
            "local_g2_allocated_slots": total_slots - free_slots,
            "local_g2_evictable_slots": sum(
                len(arena.evictable) for arena in self._arenas
            ),
            "local_g2_total_bytes": sum(arena.length for arena in self._arenas),
            "local_g2_free_bytes": sum(
                len(arena.free_slots) * arena.slot_size for arena in self._arenas
            ),
            "local_g2_allocated_bytes": sum(
                (arena.slot_count - len(arena.free_slots)) * arena.slot_size
                for arena in self._arenas
            ),
            "local_g2_evictable_bytes": sum(
                len(arena.evictable) * arena.slot_size for arena in self._arenas
            ),
        }
        for arena in self._arenas:
            prefix = f"local_g2_arena_{arena.slot_size}"
            state.update(
                {
                    f"{prefix}_total_slots": arena.slot_count,
                    f"{prefix}_free_slots": len(arena.free_slots),
                    f"{prefix}_allocated_slots": arena.slot_count
                    - len(arena.free_slots),
                    f"{prefix}_evictable_slots": len(arena.evictable),
                    f"{prefix}_total_bytes": arena.length,
                    f"{prefix}_free_bytes": len(arena.free_slots) * arena.slot_size,
                    f"{prefix}_allocated_bytes": (
                        arena.slot_count - len(arena.free_slots)
                    )
                    * arena.slot_size,
                    f"{prefix}_evictable_bytes": len(arena.evictable) * arena.slot_size,
                }
            )
        return state

    def _require_single_arena(self) -> _LocalDramArena:
        arena = self._single_arena
        if arena is None:
            raise RuntimeError("operation requires a single local DRAM arena")
        return arena

    def _arena_for_residency(self, residency: _LocalDramResidency) -> _LocalDramArena:
        arena_size = residency.arena_size
        if arena_size is None:
            return self._require_single_arena()
        arena = self._arena_by_size.get(arena_size)
        if arena is None:
            raise RuntimeError(f"unknown local DRAM arena size {arena_size}")
        return arena

    def _new_residency(
        self, arena: _LocalDramArena, slot: int, state: _LocalDramState
    ) -> _LocalDramResidency:
        return _LocalDramResidency(
            slot,
            state,
            arena_size=None if self._single_arena is not None else arena.slot_size,
        )

    def deposit(
        self,
        op_handle: OpHandle,
        blocks: Mapping[BlockKey, MemDescriptor],
        *,
        no_evict: bool,
        hints: object | None,
    ) -> None:
        keys = set(blocks)
        if not keys:
            self._kvcr._complete(op_handle, {})
            return

        deadline = self._kvcr._operation_deadline()
        op = _PendingResidencyOp(
            op_id=("deposit", op_handle),
            keys=keys,
            deadline=deadline,
            claim_on_ready=no_evict,
        )
        self._pending_residency_ops[op.op_id] = op
        self._kvcr._add_block_dependencies(op, new_operation=True)

        copy_groups: dict[
            int, list[tuple[BlockKey, int, MemDescriptor, MemDescriptor]]
        ] = {}
        evicted: list[BlockKey] = []
        for key, src in blocks.items():
            record = self._kvcr._block_record(key)
            residency = record.local_dram
            arena = self._arena_by_size.get(src.size)
            if arena is None:
                op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
                continue
            if residency is not None:
                resident_arena = self._arena_for_residency(residency)
                if resident_arena is not arena:
                    op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
                elif residency.state is _LocalDramState.READY:
                    op.results[key] = (
                        self._new_public_claim(key, residency)
                        if no_evict
                        else OpEntryResult(OpEntryStatus.SUCCESS)
                    )
                elif residency.state is _LocalDramState.DISCARDING:
                    op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
                continue
            decision = self._kvcr._policy.decide_ingest(
                self._kvcr._block_meta(key, record, arena.slot_size),
                CacheTier.FW_G2,
                required_local=no_evict,
                framework_hints=hints,
            )
            if decision[0] is PlacementAction.DROP:
                op.results[key] = OpEntryResult(OpEntryStatus.DROPPED)
                continue
            slot, evicted_key, eviction_pending = self._allocate_slot(
                arena, keys, deadline
            )
            if slot is None:
                if eviction_pending:
                    self._enqueue_capacity_waiter(arena, op, key, src)
                else:
                    op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
                continue
            if evicted_key is not None:
                evicted.append(evicted_key)
            self._kvcr._block_record(key).local_dram = self._new_residency(
                arena, slot, _LocalDramState.FILLING
            )
            copy_groups.setdefault(arena.slot_size, []).append(
                (key, slot, src, self._descriptor(arena, slot))
            )

        self._update_capacity_pressure()
        self._kvcr._publish_inventory(evicted, CacheTier.LOCAL_G2, removed=True)
        self._finish_residency_if_ready(op)
        for arena_size, copies in copy_groups.items():
            copy_keys, slots, src_descriptors, dst_descriptors = zip(*copies)
            self._kvcr._progress.submit(
                _LocalCopyOp(
                    op_id=("local_copy", self._next_copy_id),
                    keys=set(copy_keys),
                    deliver_op_id=None,
                    ordered_keys=tuple(copy_keys),
                    local_slots=tuple(slots),
                    arena_size=arena_size,
                    src_descriptors=tuple(src_descriptors),
                    dst_descriptors=tuple(dst_descriptors),
                    deadline=deadline,
                    clock=self._kvcr._clock,
                    started_at=self._kvcr._timer(),
                )
            )
            self._next_copy_id += 1

    def fetch(
        self,
        op_handle: OpHandle,
        keys: Collection[BlockKey],
        sources: Mapping[BlockKey, CacheTier],
        request_id: str | None,
        deadline: float,
        *,
        hints: object | None,
    ) -> dict[BlockKey, MemDescriptor]:
        ordered_keys = tuple(dict.fromkeys(keys))
        key_set = set(ordered_keys)
        if not key_set:
            self._kvcr._complete(op_handle, {})
            return {}

        op = _PendingResidencyOp(
            op_id=("fetch", op_handle),
            keys=key_set,
            deadline=deadline,
            claim_on_ready=True,
            request_id=request_id,
        )
        self._pending_residency_ops[op.op_id] = op
        self._kvcr._add_block_dependencies(op, new_operation=True)
        to_reserve: list[BlockKey] = []
        for key in ordered_keys:
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if residency is None:
                if self._single_arena is None:
                    # A key and source tier do not reveal which heterogeneous
                    # arena should receive a cold fill.
                    op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
                elif key in sources:
                    to_reserve.append(key)
                else:
                    op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
            elif residency.state is _LocalDramState.READY:
                self._kvcr._record_access((key,))
                op.results[key] = self._new_public_claim(key, residency)
            elif residency.state is _LocalDramState.DISCARDING:
                # A discarded fill still owns its slot, so this block cannot be
                # reserved yet. Wait for the slot instead of failing a key a
                # lower tier can still serve.
                if self._single_arena is None:
                    op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
                elif key in sources:
                    self._enqueue_capacity_waiter(
                        self._require_single_arena(), op, key, sources[key]
                    )
                else:
                    op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
        if self._single_arena is None:
            self._finish_residency_if_ready(op)
            return {}
        destinations, eviction_pending = self.reserve_fill(
            to_reserve,
            sources=sources,
            required_local=True,
            deadline=deadline,
            framework_hints=hints,
        )
        op.remote_fill_keys.update(destinations)
        for key in eviction_pending:
            self._enqueue_capacity_waiter(
                self._require_single_arena(), op, key, sources[key]
            )
        for key in to_reserve:
            if key not in destinations and key not in eviction_pending:
                op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
        self._finish_residency_if_ready(op)
        return destinations

    def complete_fill(self, keys: Collection[BlockKey], *, success: bool) -> None:
        if self._single_arena is None:
            raise RuntimeError(
                "cold fetch into multiple local DRAM arenas requires expected sizes"
            )
        ordered_keys = tuple(keys)
        slots: list[int] = []
        for key in ordered_keys:
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if (
                residency is None
                or residency.state
                not in (
                    _LocalDramState.FILLING,
                    _LocalDramState.DISCARDING,
                )
                or (success and residency.state is not _LocalDramState.FILLING)
            ):
                raise RuntimeError(f"local DRAM fill state lost for {key!r}")
            slots.append(residency.slot)
        self._apply_fill_result(
            ordered_keys,
            tuple(slots),
            self._require_single_arena().slot_size,
            success,
            CacheTier.REMOTE_G2,
        )

    def deliver(
        self,
        op_handle: OpHandle,
        blocks: Mapping[BlockKey, MemDescriptor],
        *,
        deadline: float,
    ) -> None:
        op = _PendingDeliverOp(
            op_id=("local_deliver", op_handle),
            keys=set(blocks),
            deadline=deadline,
            destinations=blocks,
        )
        self._pending_deliver_ops[op.op_id] = op
        self._kvcr._add_block_dependencies(op, new_operation=True)
        self._start_deliveries(op, blocks)

    def release(self, handles: Collection[ReleaseHandle]) -> list[ReleaseResult]:
        results: list[ReleaseResult] = []
        for handle in handles:
            claim = self._public_claims.pop(handle, None)
            if claim is None:
                results.append((handle, False))
                continue
            key, residency = claim
            self._release_claim(key, residency)
            results.append((handle, True))
        self._update_capacity_pressure()
        return results

    def acquire_sources(
        self, keys: Collection[BlockKey]
    ) -> dict[BlockKey, MemDescriptor]:
        sources: dict[BlockKey, MemDescriptor] = {}
        for key in keys:
            if key in sources:
                continue
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if residency is None or residency.state is not _LocalDramState.READY:
                continue
            self._acquire_claim(key, residency)
            arena = self._arena_for_residency(residency)
            sources[key] = self._descriptor(arena, residency.slot)
        self._update_capacity_pressure()
        return sources

    def release_sources(self, keys: Collection[BlockKey]) -> None:
        for key in keys:
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if residency is None:
                raise RuntimeError(f"local DRAM source state lost for {key!r}")
            self._release_claim(key, residency)
        self._update_capacity_pressure()

    def retire_sources(self, keys: Collection[BlockKey]) -> None:
        """Retire claimed sources when their final internal claim is released."""
        for key in dict.fromkeys(keys):
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if (
                residency is None
                or residency.state is not _LocalDramState.READY
                or residency.claim_count <= 0
            ):
                raise RuntimeError(f"local DRAM source cannot retire {key!r}")
            residency.retire_on_release = True

    def abandon_capacity_eviction(self, key: BlockKey) -> None:
        """Stop blocking local admission on an eviction that will not land.

        Waiters queue behind the slot a MOVE_TO eviction is about to free. When the
        move is abandoned, that reservation has to be dropped or every later admission
        is refused for the life of the process.
        """
        for arena in self._arenas:
            if arena.capacity_eviction_key == key:
                arena.capacity_eviction_key = None

    def discard_fill(self, keys: Collection[BlockKey]) -> None:
        residency_ops: dict[_OpId, _PendingResidencyOp] = {}
        deliver_ops: dict[_OpId, _PendingDeliverOp] = {}
        for key in dict.fromkeys(keys):
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if (
                record is None
                or residency is None
                or residency.state
                not in (
                    _LocalDramState.FILLING,
                    _LocalDramState.DISCARDING,
                )
            ):
                raise RuntimeError(f"local DRAM fill state lost for {key!r}")
            residency.state = _LocalDramState.DISCARDING
            for op_id in record.active_op_ids:
                residency_op = self._pending_residency_ops.get(op_id)
                if (
                    residency_op is not None
                    and key in residency_op.keys
                    # Capacity waiters never owned this fill, and the slot it
                    # holds is exactly what they are queued for.
                    and key not in residency_op.capacity_waiters
                ):
                    residency_op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
                    residency_ops[op_id] = residency_op
                deliver_op = self._pending_deliver_ops.get(op_id)
                if deliver_op is not None and key in deliver_op.keys:
                    deliver_op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
                    deliver_ops[op_id] = deliver_op

        for residency_op in residency_ops.values():
            self._finish_residency_if_ready(residency_op)
        for deliver_op in deliver_ops.values():
            self._finish_deliver_if_ready(deliver_op)

    def close(self) -> None:
        # Policy state ends with KVCR; teardown emits no per-block removals.
        self._public_claims.clear()

    def poll_main(self, items: Collection[object]) -> list[object]:
        unhandled: list[object] = []
        for item in items:
            if isinstance(item, _LocalCopyOp):
                self._finish_copy(item)
            else:
                unhandled.append(item)
        self._expire_pending_ops(self._kvcr._clock())
        return unhandled

    def _finish_copy(self, copy: _LocalCopyOp) -> None:
        self._kvcr._record_transfer(
            "local_deliver" if copy.deliver_op_id is not None else "local_fill",
            copy.started_at,
            copy.success,
            len(copy.ordered_keys),
            sum(descriptor.size for descriptor in copy.src_descriptors),
        )
        if copy.deliver_op_id is not None:
            self._finish_delivery_copy(copy)
            return

        self._apply_fill_result(
            copy.ordered_keys,
            copy.local_slots,
            copy.arena_size,
            copy.success,
            CacheTier.FW_G2,
        )

    def _apply_fill_result(
        self,
        ordered_keys: tuple[BlockKey, ...],
        local_slots: tuple[int, ...],
        arena_size: int,
        success: bool,
        source: CacheTier,
    ) -> None:
        committed: list[BlockKey] = []
        affected_residency_ops: dict[_OpId, _PendingResidencyOp] = {}
        affected_deliver_ops: dict[_OpId, _PendingDeliverOp] = {}
        deliver_keys: dict[_OpId, list[BlockKey]] = {}
        now = self._kvcr._clock()
        arena = self._arena_by_size[arena_size]
        for key, slot in zip(ordered_keys, local_slots):
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if (
                record is None
                or residency is None
                or residency.slot != slot
                or self._arena_for_residency(residency) is not arena
                or residency.state
                not in (
                    _LocalDramState.FILLING,
                    _LocalDramState.DISCARDING,
                )
                or (success and residency.state is not _LocalDramState.FILLING)
            ):
                raise RuntimeError(f"local DRAM fill state lost for {key!r}")
            if success:
                record.last_access = now
                residency.state = _LocalDramState.READY
                self._residency_observer(key, record)
                meta = self._kvcr._block_meta(key, record, arena.slot_size)
                self._kvcr._on_ingest(meta, source)
                self._make_evictable(key)
                committed.append(key)
            else:
                record.local_dram = None
                arena.free_slots.append(slot)

            for op_id in record.active_op_ids:
                residency_op = self._pending_residency_ops.get(op_id)
                if residency_op is not None and key in residency_op.keys:
                    if success and (
                        residency_op.op_id[0] == "deposit"
                        or now < residency_op.deadline
                    ):
                        if residency_op.op_id[0] == "fetch":
                            self._kvcr._record_access((key,))
                        residency_op.results[key] = (
                            self._new_public_claim(key, residency)
                            if residency_op.claim_on_ready
                            else OpEntryResult(OpEntryStatus.SUCCESS)
                        )
                        affected_residency_ops[op_id] = residency_op
                    elif key not in residency_op.capacity_waiters:
                        residency_op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
                        affected_residency_ops[op_id] = residency_op
                    # A capacity waiter is queued for the slot this failed fill
                    # just freed; _resume_capacity_waiters retries it below.

                deliver_op = self._pending_deliver_ops.get(op_id)
                if deliver_op is not None and key in deliver_op.keys:
                    if success:
                        deliver_keys.setdefault(op_id, []).append(key)
                    else:
                        deliver_op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
                    affected_deliver_ops[op_id] = deliver_op

        self._update_capacity_pressure()
        self._kvcr._publish_inventory(committed, CacheTier.LOCAL_G2, removed=False)
        for residency_op in affected_residency_ops.values():
            self._finish_residency_if_ready(residency_op)
        for op_id, deliver_op in affected_deliver_ops.items():
            self._start_deliveries(deliver_op, deliver_keys.get(op_id, ()))
        if not success:
            for key in ordered_keys:
                self._kvcr._prune_block_record(key)
        self._resume_capacity_waiters(arena)

    def reserve_fill(
        self,
        keys: Collection[BlockKey],
        *,
        sources: Mapping[BlockKey, CacheTier],
        required_local: bool,
        deadline: float,
        framework_hints: object | None = None,
    ) -> tuple[dict[BlockKey, MemDescriptor], set[BlockKey]]:
        arena = self._require_single_arena()
        keys = tuple(dict.fromkeys(keys))
        protected = set(keys)
        destinations: dict[BlockKey, MemDescriptor] = {}
        eviction_pending: set[BlockKey] = set()
        evicted: list[BlockKey] = []
        for key in keys:
            record = self._kvcr._block_record_map.get(key)
            if record is None:
                raise RuntimeError(f"missing block record for {key!r}")
            if record.local_dram is not None:
                continue
            decision = self._kvcr._policy.decide_ingest(
                self._kvcr._block_meta(key, record, arena.slot_size),
                sources[key],
                required_local,
                framework_hints=framework_hints,
            )
            if decision[0] is PlacementAction.DROP:
                continue
            slot, evicted_key, waiting = self._allocate_slot(arena, protected, deadline)
            if slot is None:
                if waiting:
                    eviction_pending.add(key)
                continue
            if evicted_key is not None:
                evicted.append(evicted_key)
            self._kvcr._block_record(key).local_dram = self._new_residency(
                arena, slot, _LocalDramState.FILLING
            )
            destinations[key] = self._descriptor(arena, slot)
        self._update_capacity_pressure()
        self._kvcr._publish_inventory(evicted, CacheTier.LOCAL_G2, removed=True)
        return destinations, eviction_pending

    def _start_deliveries(
        self, op: _PendingDeliverOp, keys: Collection[BlockKey]
    ) -> None:
        copy_groups: dict[
            int, list[tuple[BlockKey, int, MemDescriptor, MemDescriptor]]
        ] = {}
        now = self._kvcr._clock()
        for key in keys:
            if key in op.results or key in op.active_keys:
                continue
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if residency is None:
                op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
            elif residency.state is _LocalDramState.FILLING:
                continue
            else:
                arena = self._arena_for_residency(residency)
                if (
                    residency.state is _LocalDramState.DISCARDING
                    or op.destinations[key].size != arena.slot_size
                    or now >= op.deadline
                ):
                    op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
                    continue
                self._acquire_claim(key, residency)
                op.active_keys.add(key)
                copy_groups.setdefault(arena.slot_size, []).append(
                    (
                        key,
                        residency.slot,
                        self._descriptor(arena, residency.slot),
                        op.destinations[key],
                    )
                )

        self._update_capacity_pressure()
        for arena_size, copies in copy_groups.items():
            copy_keys, local_slots, src_descriptors, dst_descriptors = zip(*copies)
            self._kvcr._progress.submit(
                _LocalCopyOp(
                    op_id=("local_copy", self._next_copy_id),
                    keys=set(copy_keys),
                    deliver_op_id=op.op_id,
                    ordered_keys=tuple(copy_keys),
                    local_slots=tuple(local_slots),
                    arena_size=arena_size,
                    src_descriptors=tuple(src_descriptors),
                    dst_descriptors=tuple(dst_descriptors),
                    deadline=op.deadline,
                    clock=self._kvcr._clock,
                    started_at=self._kvcr._timer(),
                )
            )
            self._next_copy_id += 1
        self._finish_deliver_if_ready(op)

    def _finish_delivery_copy(self, copy: _LocalCopyOp) -> None:
        if copy.deliver_op_id is None:
            raise RuntimeError("local delivery has no owning operation")
        op = self._pending_deliver_ops[copy.deliver_op_id]
        arena = self._arena_by_size[copy.arena_size]
        for key, slot in zip(copy.ordered_keys, copy.local_slots):
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if (
                residency is None
                or residency.slot != slot
                or self._arena_for_residency(residency) is not arena
                or residency.state is not _LocalDramState.READY
            ):
                raise RuntimeError(f"local DRAM delivery state lost for {key!r}")
            if copy.success:
                self._kvcr._record_access((key,))
            self._release_claim(key, residency)
            op.active_keys.discard(key)
            op.results[key] = OpEntryResult(
                OpEntryStatus.SUCCESS if copy.success else OpEntryStatus.FAILED
            )
        self._update_capacity_pressure()
        self._finish_deliver_if_ready(op)

    def _finish_residency_if_ready(self, op: _PendingResidencyOp) -> None:
        if len(op.results) != len(op.keys):
            return
        op.capacity_waiters.clear()
        self._pending_residency_ops.pop(op.op_id)
        self._kvcr._remove_block_dependencies(op)
        self._kvcr._complete(cast(OpHandle, op.op_id[1]), op.results)

    def _finish_deliver_if_ready(self, op: _PendingDeliverOp) -> None:
        if len(op.results) != len(op.keys):
            return
        self._pending_deliver_ops.pop(op.op_id)
        self._kvcr._remove_block_dependencies(op)
        self._kvcr._complete(cast(OpHandle, op.op_id[1]), op.results)

    def _expire_pending_ops(self, now: float) -> None:
        for residency_op in list(self._pending_residency_ops.values()):
            if now < residency_op.deadline:
                continue
            if residency_op.op_id[0] == "deposit":
                for key in residency_op.keys - residency_op.results.keys():
                    if key in residency_op.capacity_waiters:
                        residency_op.capacity_waiters.remove(key)
                        residency_op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
                self._finish_residency_if_ready(residency_op)
                continue
            if residency_op.op_id[0] != "fetch":
                continue
            waiting_keys = residency_op.keys - residency_op.results.keys()
            remote_fill_keys = waiting_keys & residency_op.remote_fill_keys
            for key in waiting_keys - remote_fill_keys:
                residency_op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
            if remote_fill_keys:
                self.discard_fill(remote_fill_keys)
            else:
                self._finish_residency_if_ready(residency_op)

        for deliver_op in list(self._pending_deliver_ops.values()):
            if now < deliver_op.deadline:
                continue
            waiting_keys = (
                deliver_op.keys - deliver_op.results.keys() - deliver_op.active_keys
            )
            for key in waiting_keys:
                deliver_op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
            self._finish_deliver_if_ready(deliver_op)

    def _enqueue_capacity_waiter(
        self,
        arena: _LocalDramArena,
        op: _PendingResidencyOp,
        key: BlockKey,
        source: MemDescriptor | CacheTier,
    ) -> None:
        if key in op.capacity_waiters:
            raise RuntimeError(f"duplicate local capacity waiter for {key!r}")
        arena.capacity_waiters.append(_CapacityWaiter(op, key, source))
        op.capacity_waiters.add(key)

    def _resume_capacity_waiters(self, arena: _LocalDramArena) -> None:
        if arena.resuming_capacity_waiters:
            return
        arena.resuming_capacity_waiters = True
        try:
            while arena.capacity_waiters:
                waiter = arena.capacity_waiters[0]
                op = waiter.op
                if (
                    waiter.key not in op.capacity_waiters
                    or self._pending_residency_ops.get(op.op_id) is not op
                ):
                    arena.capacity_waiters.popleft()
                    continue
                if waiter.key in op.results:
                    arena.capacity_waiters.popleft()
                    op.capacity_waiters.remove(waiter.key)
                    continue
                if self._kvcr._clock() >= op.deadline:
                    arena.capacity_waiters.popleft()
                    op.capacity_waiters.remove(waiter.key)
                    op.results[waiter.key] = OpEntryResult(OpEntryStatus.FAILED)
                    self._finish_residency_if_ready(op)
                    continue

                record = self._kvcr._block_record(waiter.key)
                residency = record.local_dram
                if residency is not None:
                    arena.capacity_waiters.popleft()
                    op.capacity_waiters.remove(waiter.key)
                    if self._arena_for_residency(residency) is not arena:
                        op.results[waiter.key] = OpEntryResult(OpEntryStatus.FAILED)
                    elif residency.state is _LocalDramState.READY:
                        op.results[waiter.key] = (
                            self._new_public_claim(waiter.key, residency)
                            if op.claim_on_ready
                            else OpEntryResult(OpEntryStatus.SUCCESS)
                        )
                    elif residency.state is _LocalDramState.DISCARDING:
                        op.results[waiter.key] = OpEntryResult(OpEntryStatus.FAILED)
                    self._finish_residency_if_ready(op)
                    continue

                evicted_key: BlockKey | None = None
                if arena.free_slots:
                    slot = arena.free_slots.popleft()
                elif arena.capacity_eviction_key is not None:
                    break
                else:
                    slot, evicted_key, eviction_pending = self._allocate_slot(
                        arena, op.keys, op.deadline
                    )
                    if slot is None:
                        if eviction_pending:
                            break
                        arena.capacity_waiters.popleft()
                        op.capacity_waiters.remove(waiter.key)
                        op.results[waiter.key] = OpEntryResult(OpEntryStatus.FAILED)
                        self._finish_residency_if_ready(op)
                        continue

                arena.capacity_waiters.popleft()
                op.capacity_waiters.remove(waiter.key)
                if evicted_key is not None:
                    self._kvcr._publish_inventory(
                        (evicted_key,), CacheTier.LOCAL_G2, removed=True
                    )
                record.local_dram = self._new_residency(
                    arena, slot, _LocalDramState.FILLING
                )
                if isinstance(waiter.source, CacheTier):
                    op.remote_fill_keys.add(waiter.key)
                    self._kvcr._start_local_fill(
                        waiter.source,
                        {waiter.key: self._descriptor(arena, slot)},
                        op.request_id,
                        op.deadline,
                    )
                else:
                    self._kvcr._progress.submit(
                        _LocalCopyOp(
                            op_id=("local_copy", self._next_copy_id),
                            keys={waiter.key},
                            deliver_op_id=None,
                            ordered_keys=(waiter.key,),
                            local_slots=(slot,),
                            arena_size=arena.slot_size,
                            src_descriptors=(waiter.source,),
                            dst_descriptors=(self._descriptor(arena, slot),),
                            deadline=op.deadline,
                            clock=self._kvcr._clock,
                            started_at=self._kvcr._timer(),
                        )
                    )
                    self._next_copy_id += 1
        finally:
            arena.resuming_capacity_waiters = False
            self._update_capacity_pressure()

    def _new_public_claim(
        self, key: BlockKey, residency: _LocalDramResidency
    ) -> OpEntryResult:
        self._acquire_claim(key, residency)
        handle = ReleaseHandle(self._next_release_handle)
        self._next_release_handle += 1
        self._public_claims[handle] = (key, residency)
        arena = self._arena_for_residency(residency)
        return OpEntryResult(
            OpEntryStatus.SUCCESS,
            self._descriptor(arena, residency.slot),
            handle,
        )

    def _acquire_claim(self, key: BlockKey, residency: _LocalDramResidency) -> None:
        if residency.state is not _LocalDramState.READY:
            raise RuntimeError(f"cannot claim unready local DRAM entry {key!r}")
        self._remove_evictable(key, residency)
        residency.claim_count += 1

    def _release_claim(self, key: BlockKey, residency: _LocalDramResidency) -> None:
        record = self._kvcr._block_record_map.get(key)
        if (
            record is None
            or record.local_dram is not residency
            or residency.claim_count <= 0
        ):
            raise RuntimeError(f"invalid local DRAM claim for {key!r}")
        arena = self._arena_for_residency(residency)
        residency.claim_count -= 1
        if residency.claim_count == 0:
            if residency.retire_on_release:
                record.local_dram = None
                self._residency_observer(key, record)
                arena.free_slots.append(residency.slot)
                self.abandon_capacity_eviction(key)
                self._kvcr._on_remove(
                    self._kvcr._block_meta(key, record, arena.slot_size)
                )
                self._kvcr._publish_inventory((key,), CacheTier.LOCAL_G2, removed=True)
                self._kvcr._prune_block_record(key)
                self._resume_capacity_waiters(arena)
            else:
                self._make_evictable(key)

    def _allocate_slot(
        self,
        arena: _LocalDramArena,
        protected: set[BlockKey],
        deadline: float,
    ) -> tuple[int | None, BlockKey | None, bool]:
        if arena.free_slots:
            return arena.free_slots.popleft(), None, False
        if arena.capacity_eviction_key is not None:
            return None, None, True
        self._retry_unscored(arena)
        skipped = set(protected)
        while (key := arena.evictable.select(skipped)) is not None:
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if (
                record is None
                or residency is None
                or self._arena_for_residency(residency) is not arena
                or residency.state is not _LocalDramState.READY
                or residency.claim_count
            ):
                raise RuntimeError(f"invalid evictable local DRAM entry {key!r}")
            decision, eviction_pending = self._kvcr._decide_eviction(
                self._kvcr._block_meta(key, record, arena.slot_size),
                CacheTier.LOCAL_G2,
                deadline,
            )
            if arena.free_slots:
                return arena.free_slots.popleft(), None, False
            if eviction_pending:
                arena.capacity_eviction_key = key
                return None, None, True
            if decision[0] is PlacementAction.KEEP:
                skipped.add(key)
                continue
            arena.evictable.remove(key)
            record.local_dram = None
            self._residency_observer(key, record)
            self._kvcr._on_remove(self._kvcr._block_meta(key, record, arena.slot_size))
            self._kvcr._prune_block_record(key)
            return residency.slot, key, False
        return None, None, False

    def _make_evictable(self, key: BlockKey) -> None:
        record = self._kvcr._block_record_map.get(key)
        residency = record.local_dram if record is not None else None
        if record is None or residency is None:
            raise RuntimeError(f"missing block record for {key!r}")
        arena = self._arena_for_residency(residency)
        score = self._kvcr._policy.eviction_score(
            self._kvcr._block_meta(key, record, arena.slot_size),
            CacheTier.LOCAL_G2,
        )
        if score is None:
            arena.unscored.add(key)
            return
        arena.unscored.discard(key)
        arena.evictable.insert(key, score)

    def _remove_evictable(self, key: BlockKey, residency: _LocalDramResidency) -> None:
        arena = self._arena_for_residency(residency)
        arena.unscored.discard(key)
        arena.evictable.remove(key)

    def _retry_unscored(self, arena: _LocalDramArena) -> None:
        for key in tuple(arena.unscored):
            self._make_evictable(key)

    def _descriptor(self, arena: _LocalDramArena, slot: int) -> MemDescriptor:
        return MemDescriptor(
            end_point_name=self._kvcr.nixl_agent_name,
            mem_type="DRAM",
            addr=arena.address + slot * arena.slot_size,
            size=arena.slot_size,
            device_Id=0,
            info="",
        )

    def _update_capacity_pressure(self) -> None:
        self._kvcr._update_capacity_pressure(
            sum(len(arena.free_slots) + len(arena.evictable) for arena in self._arenas)
        )
