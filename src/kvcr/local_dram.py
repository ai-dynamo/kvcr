# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KVCR-owned local DRAM slots, claims, and transfers."""

import logging
from collections import Counter, deque
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, cast

from .config import LocalDramOptions
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


@dataclass(slots=True)
class _LocalDramResidency:
    """One logical block resident in one or more named allocator pools.

    The legacy recovery format only stores a slot number. Accepting an integer
    here keeps that format usable for the single-pool recovery path;
    ``adopt_recovery_slots`` assigns the configured pool name before use.
    """

    extents: tuple[tuple[str, int], ...] | int
    state: _LocalDramState
    claim_count: int = 0
    retire_on_release: bool = False

    def __post_init__(self) -> None:
        if type(self.extents) is int:
            self.extents = (("", self.extents),)
        elif not isinstance(self.extents, tuple) or not self.extents:
            raise ValueError("local DRAM residency requires at least one extent")

    @property
    def slot(self) -> int:
        """Legacy single-pool slot accessor used by recovery."""
        if not isinstance(self.extents, tuple) or len(self.extents) != 1:
            raise RuntimeError("operation requires a single local DRAM extent")
        return self.extents[0][1]


@dataclass(slots=True)
class _LocalDramPool:
    name: str
    address: int
    length: int
    slot_size: int
    free_slots: deque[int]

    @property
    def slot_count(self) -> int:
        return self.length // self.slot_size


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
    destinations: Mapping[BlockKey, tuple[MemDescriptor, ...]]
    results: dict[BlockKey, OpEntryResult] = field(default_factory=dict)
    active_keys: set[BlockKey] = field(default_factory=set)


@dataclass(frozen=True)
class _CapacityWaiter:
    op: _PendingResidencyOp
    key: BlockKey
    source: tuple[MemDescriptor, ...] | CacheTier
    layout: tuple[str, ...]


@dataclass
class _LocalCopyOp(_ProgressOp):
    deliver_op_id: _OpId | None
    ordered_keys: tuple[BlockKey, ...]
    local_extents: tuple[tuple[tuple[str, int], ...], ...]
    src_descriptors: tuple[MemDescriptor, ...]
    dst_descriptors: tuple[MemDescriptor, ...]
    deadline: float
    backend: str
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
                    backend=self.backend,
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
    """Main-thread metadata for named, fixed-slot local DRAM pools."""

    def __init__(
        self,
        kvcr: "_KVCRCore",
        region: LocalDramOptions,
    ) -> None:
        if not region.backend:
            raise ValueError("local DRAM NIXL backend must be non-empty")

        self._kvcr = kvcr
        self._backend = region.backend
        block_sizes = dict(kvcr.pool_layouts)
        configured_names = set(block_sizes)
        region_names = [name for name, _address, _length in region.pools]
        if len(region_names) != len(set(region_names)):
            raise ValueError("local DRAM pool names must be unique")
        if set(region_names) != configured_names:
            raise ValueError("local DRAM pool names must exactly match pool_layouts")

        pools: list[_LocalDramPool] = []
        memory_ranges: list[tuple[int, int]] = []
        for pool_name, address, length in region.pools:
            if type(address) is not int or address <= 0:
                raise ValueError("local DRAM address must be a positive integer")
            if type(length) is not int or length <= 0:
                raise ValueError("local DRAM pool size must be a positive integer")
            start, end = address, address + length
            if any(
                start < other_end and other_start < end
                for other_start, other_end in memory_ranges
            ):
                raise ValueError("local DRAM pool memory regions must not overlap")
            slot_size = block_sizes[pool_name]
            if length % slot_size:
                raise ValueError(
                    f"local DRAM pool {pool_name!r} size must contain complete blocks"
                )
            slot_count = length // slot_size
            if not slot_count:
                raise ValueError(
                    f"local DRAM pool {pool_name!r} must hold at least one block"
                )
            pools.append(
                _LocalDramPool(
                    name=pool_name,
                    address=address,
                    length=length,
                    slot_size=slot_size,
                    free_slots=deque(range(slot_count)),
                )
            )
            memory_ranges.append((start, end))

        self._pools = tuple(pools)
        self._pool_by_name = {pool.name: pool for pool in pools}
        self._single_pool = pools[0] if len(pools) == 1 else None
        # Retain these private aliases for the single-pool recovery/test path.
        self._address = pools[0].address if len(pools) == 1 else None
        self._length = pools[0].length if len(pools) == 1 else None
        self._slot_size = pools[0].slot_size if len(pools) == 1 else None
        self._evictable = _EvictionQueue()
        self._evictable_keys: set[BlockKey] = set()
        self._unscored: set[BlockKey] = set()
        self._pending_residency_ops: dict[_OpId, _PendingResidencyOp] = {}
        self._pending_deliver_ops: dict[_OpId, _PendingDeliverOp] = {}
        self._capacity_waiters: deque[_CapacityWaiter] = deque()
        self._capacity_eviction_key: BlockKey | None = None
        self._resuming_capacity_waiters = False
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
        pool = self._require_single_pool()
        return pool.address, pool.length

    @property
    def memory_regions(self) -> tuple[tuple[int, int], ...]:
        return tuple((pool.address, pool.length) for pool in self._pools)

    @property
    def _free_slots(self) -> deque[int]:
        return self._require_single_pool().free_slots

    @property
    def _total_slots(self) -> int:
        return sum(pool.slot_count for pool in self._pools)

    def _require_single_pool(self) -> _LocalDramPool:
        pool = self._single_pool
        if pool is None:
            raise RuntimeError("operation requires a single local DRAM pool")
        return pool

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
        pool = self._require_single_pool()
        slot_count = pool.slot_count
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
            residency.extents = ((pool.name, slot),)
            occupied.add(slot)
        pool.free_slots = deque(
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
        free_slots = sum(len(pool.free_slots) for pool in self._pools)
        evictable_slots = sum(
            len(residency.extents)
            for key in self._evictable_keys
            if (residency := self._residency(key)) is not None
        )
        state = {
            "local_g2_total_slots": total_slots,
            "local_g2_free_slots": free_slots,
            "local_g2_allocated_slots": total_slots - free_slots,
            "local_g2_evictable_slots": evictable_slots,
            "local_g2_total_bytes": sum(pool.length for pool in self._pools),
            "local_g2_free_bytes": sum(
                len(pool.free_slots) * pool.slot_size for pool in self._pools
            ),
            "local_g2_allocated_bytes": sum(
                (pool.slot_count - len(pool.free_slots)) * pool.slot_size
                for pool in self._pools
            ),
            "local_g2_evictable_bytes": sum(
                self._residency_size(residency)
                for key in self._evictable_keys
                if (residency := self._residency(key)) is not None
            ),
        }
        for pool in self._pools:
            prefix = f"local_g2_pool_{pool.name or 'default'}"
            pool_evictable_slots = sum(
                sum(name == pool.name for name, _slot in residency.extents)
                for key in self._evictable_keys
                if (residency := self._residency(key)) is not None
            )
            state.update(
                {
                    f"{prefix}_total_slots": pool.slot_count,
                    f"{prefix}_free_slots": len(pool.free_slots),
                    f"{prefix}_allocated_slots": pool.slot_count - len(pool.free_slots),
                    f"{prefix}_evictable_slots": pool_evictable_slots,
                    f"{prefix}_total_bytes": pool.length,
                    f"{prefix}_free_bytes": len(pool.free_slots) * pool.slot_size,
                    f"{prefix}_allocated_bytes": (
                        pool.slot_count - len(pool.free_slots)
                    )
                    * pool.slot_size,
                    f"{prefix}_evictable_bytes": pool_evictable_slots * pool.slot_size,
                }
            )
        return state

    def deposit(
        self,
        op_handle: OpHandle,
        blocks: Mapping[BlockKey, tuple[MemDescriptor, ...]],
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

        copy_keys: list[BlockKey] = []
        copy_extents: list[tuple[tuple[str, int], ...]] = []
        src_descriptors: list[MemDescriptor] = []
        dst_descriptors: list[MemDescriptor] = []
        evicted: list[BlockKey] = []
        for key, source_descriptors in blocks.items():
            sources = tuple(source_descriptors)
            layout = self._layout_for_descriptors(sources)
            record = self._kvcr._block_record(key)
            residency = record.local_dram
            if residency is not None:
                if self._residency_layout(residency) != layout:
                    op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
                elif residency.state is _LocalDramState.READY:
                    op.results[key] = (
                        self._new_public_claim(
                            key, residency, include_descriptors=False
                        )
                        if no_evict
                        else OpEntryResult(OpEntryStatus.SUCCESS)
                    )
                elif residency.state is _LocalDramState.DISCARDING:
                    op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
                continue
            size_bytes = sum(source.size for source in sources)
            decision = self._kvcr._policy.decide_ingest(
                self._kvcr._block_meta(key, record, size_bytes),
                CacheTier.FW_G2,
                required_local=no_evict,
                framework_hints=hints,
            )
            if decision[0] is PlacementAction.DROP:
                op.results[key] = OpEntryResult(OpEntryStatus.DROPPED)
                continue
            extents, evicted_keys, eviction_pending = self._allocate_extents(
                layout, keys, deadline
            )
            evicted.extend(evicted_keys)
            if extents is None:
                if eviction_pending:
                    self._enqueue_capacity_waiter(op, key, sources, layout)
                else:
                    op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
                continue
            self._kvcr._block_record(key).local_dram = _LocalDramResidency(
                extents, _LocalDramState.FILLING
            )
            copy_keys.append(key)
            copy_extents.append(extents)
            src_descriptors.extend(sources)
            dst_descriptors.extend(self._descriptors_for_extents(extents))

        self._update_capacity_pressure()
        self._kvcr._publish_inventory(
            tuple(dict.fromkeys(evicted)), CacheTier.LOCAL_G2, removed=True
        )
        self._finish_residency_if_ready(op)
        if copy_keys:
            self._kvcr._progress.submit(
                _LocalCopyOp(
                    op_id=("local_copy", self._next_copy_id),
                    keys=set(copy_keys),
                    deliver_op_id=None,
                    ordered_keys=tuple(copy_keys),
                    local_extents=tuple(copy_extents),
                    src_descriptors=tuple(src_descriptors),
                    dst_descriptors=tuple(dst_descriptors),
                    deadline=deadline,
                    backend=self._backend,
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
        expected_layout: Collection[str],
        *,
        hints: object | None,
    ) -> dict[BlockKey, tuple[MemDescriptor, ...]]:
        ordered_keys = tuple(dict.fromkeys(keys))
        layout = tuple(expected_layout)
        self._validate_layout(layout)
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
                if key in sources:
                    to_reserve.append(key)
                else:
                    op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
            elif self._residency_layout(residency) != layout:
                op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
            elif residency.state is _LocalDramState.READY:
                self._kvcr._record_access((key,))
                op.results[key] = self._new_public_claim(
                    key, residency, include_descriptors=True
                )
            elif residency.state is _LocalDramState.DISCARDING:
                # A discarded fill still owns its slot, so this block cannot be
                # reserved yet. Wait for the slot instead of failing a key a
                # lower tier can still serve.
                if key in sources:
                    self._enqueue_capacity_waiter(op, key, sources[key], layout)
                else:
                    op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
        destinations, eviction_pending = self.reserve_fill(
            to_reserve,
            sources=sources,
            required_local=True,
            deadline=deadline,
            expected_layout=layout,
            framework_hints=hints,
        )
        op.remote_fill_keys.update(destinations)
        for key in eviction_pending:
            self._enqueue_capacity_waiter(op, key, sources[key], layout)
        for key in to_reserve:
            if key not in destinations and key not in eviction_pending:
                op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
        self._finish_residency_if_ready(op)
        return destinations

    def complete_fill(self, keys: Collection[BlockKey], *, success: bool) -> None:
        ordered_keys = tuple(keys)
        extents: list[tuple[tuple[str, int], ...]] = []
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
            extents.append(residency.extents)
        self._apply_fill_result(
            ordered_keys, tuple(extents), success, CacheTier.REMOTE_G2
        )

    def deliver(
        self,
        op_handle: OpHandle,
        blocks: Mapping[BlockKey, tuple[MemDescriptor, ...]],
        *,
        deadline: float,
    ) -> None:
        op = _PendingDeliverOp(
            op_id=("local_deliver", op_handle),
            keys=set(blocks),
            deadline=deadline,
            destinations={key: tuple(value) for key, value in blocks.items()},
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
    ) -> dict[BlockKey, tuple[MemDescriptor, ...]]:
        sources: dict[BlockKey, tuple[MemDescriptor, ...]] = {}
        for key in keys:
            if key in sources:
                continue
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if residency is None or residency.state is not _LocalDramState.READY:
                continue
            self._acquire_claim(key, residency)
            sources[key] = self._descriptors_for_extents(residency.extents)
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
        if self._capacity_eviction_key == key:
            self._capacity_eviction_key = None

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
            copy.local_extents,
            copy.success,
            CacheTier.FW_G2,
        )

    def _apply_fill_result(
        self,
        ordered_keys: tuple[BlockKey, ...],
        local_extents: tuple[tuple[tuple[str, int], ...], ...],
        success: bool,
        source: CacheTier,
    ) -> None:
        committed: list[BlockKey] = []
        affected_residency_ops: dict[_OpId, _PendingResidencyOp] = {}
        affected_deliver_ops: dict[_OpId, _PendingDeliverOp] = {}
        deliver_keys: dict[_OpId, list[BlockKey]] = {}
        now = self._kvcr._clock()
        for key, extents in zip(ordered_keys, local_extents):
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if (
                record is None
                or residency is None
                or residency.extents != extents
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
                meta = self._kvcr._block_meta(
                    key, record, self._residency_size(residency)
                )
                self._kvcr._on_ingest(meta, source)
                self._make_evictable(key)
                committed.append(key)
            else:
                record.local_dram = None
                self._free_extents(extents)

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
                            self._new_public_claim(
                                key,
                                residency,
                                include_descriptors=residency_op.op_id[0] == "fetch",
                            )
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
        self._resume_capacity_waiters()

    def reserve_fill(
        self,
        keys: Collection[BlockKey],
        *,
        sources: Mapping[BlockKey, CacheTier],
        required_local: bool,
        deadline: float,
        expected_layout: Collection[str],
        framework_hints: object | None = None,
    ) -> tuple[dict[BlockKey, tuple[MemDescriptor, ...]], set[BlockKey]]:
        keys = tuple(dict.fromkeys(keys))
        layout = tuple(expected_layout)
        self._validate_layout(layout)
        protected = set(keys)
        destinations: dict[BlockKey, tuple[MemDescriptor, ...]] = {}
        eviction_pending: set[BlockKey] = set()
        evicted: list[BlockKey] = []
        for key in keys:
            record = self._kvcr._block_record_map.get(key)
            if record is None:
                raise RuntimeError(f"missing block record for {key!r}")
            if record.local_dram is not None:
                continue
            decision = self._kvcr._policy.decide_ingest(
                self._kvcr._block_meta(key, record, self._layout_size(layout)),
                sources[key],
                required_local,
                framework_hints=framework_hints,
            )
            if decision[0] is PlacementAction.DROP:
                continue
            extents, evicted_keys, waiting = self._allocate_extents(
                layout, protected, deadline
            )
            evicted.extend(evicted_keys)
            if extents is None:
                if waiting:
                    eviction_pending.add(key)
                continue
            self._kvcr._block_record(key).local_dram = _LocalDramResidency(
                extents, _LocalDramState.FILLING
            )
            destinations[key] = self._descriptors_for_extents(extents)
        self._update_capacity_pressure()
        self._kvcr._publish_inventory(
            tuple(dict.fromkeys(evicted)), CacheTier.LOCAL_G2, removed=True
        )
        return destinations, eviction_pending

    def _start_deliveries(
        self, op: _PendingDeliverOp, keys: Collection[BlockKey]
    ) -> None:
        copy_keys: list[BlockKey] = []
        local_extents: list[tuple[tuple[str, int], ...]] = []
        src_descriptors: list[MemDescriptor] = []
        dst_descriptors: list[MemDescriptor] = []
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
            elif (
                residency.state is _LocalDramState.DISCARDING
                or not self._descriptors_match_residency(
                    op.destinations[key], residency
                )
                or now >= op.deadline
            ):
                op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
            else:
                self._acquire_claim(key, residency)
                op.active_keys.add(key)
                copy_keys.append(key)
                local_extents.append(residency.extents)
                src_descriptors.extend(self._descriptors_for_extents(residency.extents))
                dst_descriptors.extend(op.destinations[key])

        self._update_capacity_pressure()
        if copy_keys:
            self._kvcr._progress.submit(
                _LocalCopyOp(
                    op_id=("local_copy", self._next_copy_id),
                    keys=set(copy_keys),
                    deliver_op_id=op.op_id,
                    ordered_keys=tuple(copy_keys),
                    local_extents=tuple(local_extents),
                    src_descriptors=tuple(src_descriptors),
                    dst_descriptors=tuple(dst_descriptors),
                    deadline=op.deadline,
                    backend=self._backend,
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
        for key, extents in zip(copy.ordered_keys, copy.local_extents):
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if (
                residency is None
                or residency.extents != extents
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
        self._capacity_waiters = deque(
            waiter for waiter in self._capacity_waiters if waiter.op is not op
        )
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
        op: _PendingResidencyOp,
        key: BlockKey,
        source: tuple[MemDescriptor, ...] | CacheTier,
        layout: tuple[str, ...],
    ) -> None:
        if key in op.capacity_waiters:
            raise RuntimeError(f"duplicate local capacity waiter for {key!r}")
        self._capacity_waiters.append(_CapacityWaiter(op, key, source, layout))
        op.capacity_waiters.add(key)

    def _resume_capacity_waiters(self) -> None:
        if self._resuming_capacity_waiters:
            return
        self._resuming_capacity_waiters = True
        try:
            # Visit each waiter that existed at entry once. A blocked pool must
            # not prevent a later waiter for an independent pool from using its
            # free capacity.
            for _ in range(len(self._capacity_waiters)):
                waiter = self._capacity_waiters.popleft()
                op = waiter.op
                if (
                    waiter.key not in op.capacity_waiters
                    or self._pending_residency_ops.get(op.op_id) is not op
                ):
                    continue
                if waiter.key in op.results:
                    op.capacity_waiters.remove(waiter.key)
                    continue
                if self._kvcr._clock() >= op.deadline:
                    op.capacity_waiters.remove(waiter.key)
                    op.results[waiter.key] = OpEntryResult(OpEntryStatus.FAILED)
                    self._finish_residency_if_ready(op)
                    continue

                record = self._kvcr._block_record(waiter.key)
                residency = record.local_dram
                if residency is not None:
                    if (
                        self._residency_layout(residency) == waiter.layout
                        and residency.state is _LocalDramState.DISCARDING
                    ):
                        # The failed fill still owns these extents. Its terminal
                        # completion invokes this method again after freeing them.
                        self._capacity_waiters.append(waiter)
                        continue
                    op.capacity_waiters.remove(waiter.key)
                    if self._residency_layout(residency) != waiter.layout:
                        op.results[waiter.key] = OpEntryResult(OpEntryStatus.FAILED)
                    elif residency.state is _LocalDramState.READY:
                        op.results[waiter.key] = (
                            self._new_public_claim(
                                waiter.key,
                                residency,
                                include_descriptors=op.op_id[0] == "fetch",
                            )
                            if op.claim_on_ready
                            else OpEntryResult(OpEntryStatus.SUCCESS)
                        )
                    self._finish_residency_if_ready(op)
                    continue

                extents, evicted_keys, eviction_pending = self._allocate_extents(
                    waiter.layout, op.keys, op.deadline
                )
                if evicted_keys:
                    self._kvcr._publish_inventory(
                        tuple(evicted_keys), CacheTier.LOCAL_G2, removed=True
                    )
                if extents is None:
                    if eviction_pending:
                        self._capacity_waiters.append(waiter)
                        continue
                    op.capacity_waiters.remove(waiter.key)
                    op.results[waiter.key] = OpEntryResult(OpEntryStatus.FAILED)
                    self._finish_residency_if_ready(op)
                    continue

                op.capacity_waiters.remove(waiter.key)
                record.local_dram = _LocalDramResidency(
                    extents, _LocalDramState.FILLING
                )
                destinations = self._descriptors_for_extents(extents)
                if isinstance(waiter.source, CacheTier):
                    op.remote_fill_keys.add(waiter.key)
                    self._kvcr._start_local_fill(
                        waiter.source,
                        {waiter.key: destinations},
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
                            local_extents=(extents,),
                            src_descriptors=waiter.source,
                            dst_descriptors=destinations,
                            deadline=op.deadline,
                            backend=self._backend,
                            clock=self._kvcr._clock,
                            started_at=self._kvcr._timer(),
                        )
                    )
                    self._next_copy_id += 1
        finally:
            self._resuming_capacity_waiters = False
            self._update_capacity_pressure()

    def _new_public_claim(
        self,
        key: BlockKey,
        residency: _LocalDramResidency,
        *,
        include_descriptors: bool,
    ) -> OpEntryResult:
        self._acquire_claim(key, residency)
        handle = ReleaseHandle(self._next_release_handle)
        self._next_release_handle += 1
        self._public_claims[handle] = (key, residency)
        return OpEntryResult(
            OpEntryStatus.SUCCESS,
            list(self._descriptors_for_extents(residency.extents))
            if include_descriptors
            else None,
            handle,
        )

    def _acquire_claim(self, key: BlockKey, residency: _LocalDramResidency) -> None:
        if residency.state is not _LocalDramState.READY:
            raise RuntimeError(f"cannot claim unready local DRAM entry {key!r}")
        self._remove_evictable(key)
        residency.claim_count += 1

    def _release_claim(self, key: BlockKey, residency: _LocalDramResidency) -> None:
        record = self._kvcr._block_record_map.get(key)
        if (
            record is None
            or record.local_dram is not residency
            or residency.claim_count <= 0
        ):
            raise RuntimeError(f"invalid local DRAM claim for {key!r}")
        residency.claim_count -= 1
        if residency.claim_count == 0:
            if residency.retire_on_release:
                record.local_dram = None
                self._residency_observer(key, record)
                self._free_extents(residency.extents)
                self.abandon_capacity_eviction(key)
                self._kvcr._on_remove(
                    self._kvcr._block_meta(key, record, self._residency_size(residency))
                )
                self._kvcr._publish_inventory((key,), CacheTier.LOCAL_G2, removed=True)
                self._kvcr._prune_block_record(key)
                self._resume_capacity_waiters()
            else:
                self._make_evictable(key)

    def _allocate_extents(
        self,
        layout: tuple[str, ...],
        protected: set[BlockKey],
        deadline: float,
    ) -> tuple[
        tuple[tuple[str, int], ...] | None,
        tuple[BlockKey, ...],
        bool,
    ]:
        """Reserve every extent for one key or reserve none of them."""
        self._validate_layout(layout)
        needed = Counter(layout)
        if any(
            count > self._pool_by_name[name].slot_count
            for name, count in needed.items()
        ):
            return None, (), False

        if self._capacity_eviction_key is not None and not self._has_free_capacity(
            needed
        ):
            return None, (), True
        self._retry_unscored()
        if not self._has_reclaimable_capacity(needed, protected):
            return None, (), False

        planned: list[tuple[BlockKey, _LocalDramResidency, int]] = []
        skipped = set(protected)
        while True:
            available = Counter(
                {name: len(self._pool_by_name[name].free_slots) for name in needed}
            )
            for _key, residency, _size_bytes in planned:
                available.update(
                    name for name, _slot in residency.extents if name in needed
                )
            if all(available[name] >= count for name, count in needed.items()):
                break
            if self._capacity_eviction_key is not None:
                return None, (), True
            deficits = {
                name for name, count in needed.items() if available[name] < count
            }
            key = self._select_contributing_victim(deficits, skipped)
            if key is None:
                return None, (), False
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if (
                record is None
                or residency is None
                or residency.state is not _LocalDramState.READY
                or residency.claim_count
            ):
                raise RuntimeError(f"invalid evictable local DRAM entry {key!r}")
            size_bytes = self._residency_size(residency)
            decision, eviction_pending = self._kvcr._decide_eviction(
                self._kvcr._block_meta(key, record, size_bytes),
                CacheTier.LOCAL_G2,
                deadline,
            )
            # A MOVE_TO can complete synchronously and retire the residency.
            if self._residency(key) is not residency:
                continue
            if eviction_pending:
                self._capacity_eviction_key = key
                return None, (), True
            if decision[0] is PlacementAction.KEEP:
                skipped.add(key)
                continue
            skipped.add(key)
            planned.append((key, residency, size_bytes))

        evicted: list[BlockKey] = []
        for key, residency, size_bytes in planned:
            record = self._kvcr._block_record_map.get(key)
            if record is None or record.local_dram is not residency:
                raise RuntimeError(f"local DRAM eviction state changed for {key!r}")
            self._remove_evictable(key)
            record.local_dram = None
            self._residency_observer(key, record)
            self._free_extents(residency.extents)
            self._kvcr._on_remove(self._kvcr._block_meta(key, record, size_bytes))
            self._kvcr._prune_block_record(key)
            evicted.append(key)

        reserved: list[tuple[str, int]] = []
        for name in layout:
            reserved.append((name, self._pool_by_name[name].free_slots.popleft()))
        return tuple(reserved), tuple(evicted), False

    def _make_evictable(self, key: BlockKey) -> None:
        record = self._kvcr._block_record_map.get(key)
        residency = record.local_dram if record is not None else None
        if record is None or residency is None:
            raise RuntimeError(f"missing block record for {key!r}")
        score = self._kvcr._policy.eviction_score(
            self._kvcr._block_meta(key, record, self._residency_size(residency)),
            CacheTier.LOCAL_G2,
        )
        if score is None:
            self._evictable_keys.discard(key)
            self._unscored.add(key)
            return
        self._unscored.discard(key)
        self._evictable_keys.add(key)
        self._evictable.insert(key, score)

    def _remove_evictable(self, key: BlockKey) -> None:
        self._unscored.discard(key)
        self._evictable_keys.discard(key)
        self._evictable.remove(key)

    def _retry_unscored(self) -> None:
        for key in tuple(self._unscored):
            self._make_evictable(key)

    def _residency(self, key: BlockKey) -> _LocalDramResidency | None:
        record = self._kvcr._block_record_map.get(key)
        return record.local_dram if record is not None else None

    def _validate_layout(self, layout: tuple[str, ...]) -> None:
        if not layout:
            raise ValueError("a block layout must contain at least one pool")
        unknown = set(layout) - self._pool_by_name.keys()
        if unknown:
            names = ", ".join(repr(name) for name in sorted(unknown))
            raise ValueError(f"unknown local DRAM pool(s): {names}")

    def _layout_for_descriptors(
        self, descriptors: tuple[MemDescriptor, ...]
    ) -> tuple[str, ...]:
        layout = tuple(descriptor.info for descriptor in descriptors)
        self._validate_layout(layout)
        for descriptor in descriptors:
            pool = self._pool_by_name[descriptor.info]
            if descriptor.size != pool.slot_size:
                raise ValueError(
                    f"descriptor for pool {pool.name!r} has the wrong byte count"
                )
        return layout

    def _residency_layout(self, residency: _LocalDramResidency) -> tuple[str, ...]:
        return tuple(name for name, _slot in residency.extents)

    def _layout_size(self, layout: tuple[str, ...]) -> int:
        return sum(self._pool_by_name[name].slot_size for name in layout)

    def _residency_size(self, residency: _LocalDramResidency) -> int:
        return self._layout_size(self._residency_layout(residency))

    def _descriptors_match_residency(
        self,
        descriptors: tuple[MemDescriptor, ...],
        residency: _LocalDramResidency,
    ) -> bool:
        if len(descriptors) != len(residency.extents):
            return False
        return all(
            descriptor.info == pool_name
            and descriptor.size == self._pool_by_name[pool_name].slot_size
            for descriptor, (pool_name, _slot) in zip(descriptors, residency.extents)
        )

    def _descriptor(self, pool_name: str, slot: int) -> MemDescriptor:
        pool = self._pool_by_name[pool_name]
        return MemDescriptor(
            end_point_name=self._kvcr.nixl_agent_name,
            mem_type="DRAM",
            addr=pool.address + slot * pool.slot_size,
            size=pool.slot_size,
            device_Id=0,
            info=pool.name,
        )

    def _descriptors_for_extents(
        self, extents: tuple[tuple[str, int], ...]
    ) -> tuple[MemDescriptor, ...]:
        return tuple(self._descriptor(name, slot) for name, slot in extents)

    def _free_extents(self, extents: tuple[tuple[str, int], ...]) -> None:
        for name, slot in extents:
            self._pool_by_name[name].free_slots.append(slot)

    def _has_free_capacity(self, needed: Counter[str]) -> bool:
        return all(
            len(self._pool_by_name[name].free_slots) >= count
            for name, count in needed.items()
        )

    def _has_reclaimable_capacity(
        self, needed: Counter[str], protected: set[BlockKey]
    ) -> bool:
        available = Counter(
            {name: len(self._pool_by_name[name].free_slots) for name in needed}
        )
        for key in self._evictable_keys - protected:
            residency = self._residency(key)
            if residency is None:
                continue
            available.update(
                name for name, _slot in residency.extents if name in needed
            )
        return all(available[name] >= count for name, count in needed.items())

    def _select_contributing_victim(
        self, deficits: set[str], skipped: set[BlockKey]
    ) -> BlockKey | None:
        while (key := self._evictable.select(skipped)) is not None:
            residency = self._residency(key)
            if residency is None:
                raise RuntimeError(f"invalid evictable local DRAM entry {key!r}")
            if any(name in deficits for name, _slot in residency.extents):
                return key
            skipped.add(key)
        return None

    def _update_capacity_pressure(self) -> None:
        self._kvcr._update_capacity_pressure(
            sum(len(pool.free_slots) for pool in self._pools)
            + sum(
                len(residency.extents)
                for key in self._evictable_keys
                if (residency := self._residency(key)) is not None
            )
        )
