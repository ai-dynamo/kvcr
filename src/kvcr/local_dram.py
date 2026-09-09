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
    slots: list[tuple[str, int]]
    state: _LocalDramState
    claim_count: int = 0
    retire_on_release: bool = False


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
    destinations: Mapping[BlockKey, list[MemDescriptor]]
    results: dict[BlockKey, OpEntryResult] = field(default_factory=dict)
    active_keys: set[BlockKey] = field(default_factory=set)


@dataclass(frozen=True)
class _CapacityWaiter:
    op: _PendingResidencyOp
    key: BlockKey
    source: list[MemDescriptor] | CacheTier
    layout: list[str]


@dataclass
class _LocalCopyOp(_ProgressOp):
    deliver_op_id: _OpId | None
    ordered_keys: tuple[BlockKey, ...]
    local_slots: tuple[tuple[tuple[str, int], ...], ...]
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
    """Main-thread metadata for KVCR-owned DRAM pools."""

    def __init__(
        self,
        kvcr: "_KVCRCore",
        region: LocalDramOptions,
    ) -> None:
        if [pool[0] for pool in region.pools] != [
            pool[0] for pool in kvcr.pool_layouts
        ]:
            raise ValueError("local DRAM pools must match pool_layouts")
        if not region.backend:
            raise ValueError("local DRAM NIXL backend must be non-empty")

        self._kvcr = kvcr
        self._backend = region.backend
        self._pools: dict[str, tuple[int, int, int]] = {}
        self._slot_counts: dict[str, int] = {}
        self._free_slots: dict[str, deque[int]] = {}
        for (pool_name, address, length), (_, slot_size) in zip(
            region.pools, kvcr.pool_layouts
        ):
            if address <= 0:
                raise ValueError("local DRAM address must be positive")
            if type(length) is not int or length <= 0:
                raise ValueError("local DRAM pool size must be a positive integer")
            slot_count = length // slot_size
            if not slot_count:
                raise ValueError("local DRAM pool must hold at least one block")
            self._pools[pool_name] = (address, length, slot_size)
            self._slot_counts[pool_name] = slot_count
            self._free_slots[pool_name] = deque(range(slot_count))
        self._evictable = _EvictionQueue()
        self._evictable_slots: Counter[str] = Counter()
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
    def memory_regions(self) -> tuple[tuple[int, int], ...]:
        return tuple((address, length) for address, length, _ in self._pools.values())

    @property
    def _total_slots(self) -> int:
        return sum(length // slot_size for _, length, slot_size in self._pools.values())

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
        occupied = {pool_name: set() for pool_name in self._pools}
        for record in records.values():
            residency = record.local_dram
            if residency is None:
                continue
            if residency.state is not _LocalDramState.READY or not residency.slots:
                raise ValueError("invalid local DRAM recovery slots")
            for pool_name, slot in residency.slots:
                pool = self._pools.get(pool_name)
                if (
                    pool is None
                    or type(slot) is not int
                    or not 0 <= slot < pool[1] // pool[2]
                    or slot in occupied[pool_name]
                ):
                    raise ValueError("invalid local DRAM recovery slots")
                occupied[pool_name].add(slot)
        self._free_slots = {
            pool_name: deque(
                slot
                for slot in range(length // slot_size)
                if slot not in occupied[pool_name]
            )
            for pool_name, (_, length, slot_size) in self._pools.items()
        }

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
        free_slots = sum(map(len, self._free_slots.values()))
        return {
            "local_g2_total_slots": total_slots,
            "local_g2_free_slots": free_slots,
            "local_g2_allocated_slots": total_slots - free_slots,
            "local_g2_evictable_slots": sum(self._evictable_slots.values()),
        }

    def deposit(
        self,
        op_handle: OpHandle,
        blocks: Mapping[BlockKey, list[MemDescriptor]],
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
        slots: list[tuple[tuple[str, int], ...]] = []
        src_descriptors: list[MemDescriptor] = []
        dst_descriptors: list[MemDescriptor] = []
        evicted: list[BlockKey] = []
        for key, sources in blocks.items():
            record = self._kvcr._block_record(key)
            residency = record.local_dram
            if residency is not None:
                if not self._same_layout(residency.slots, sources):
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
            locations, evicted_keys, eviction_pending = self._allocate_slots(
                [source.info for source in sources], keys, deadline
            )
            evicted.extend(evicted_keys)
            if locations is None:
                if eviction_pending:
                    self._enqueue_capacity_waiter(
                        op, key, sources, [source.info for source in sources]
                    )
                else:
                    op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
                continue
            self._kvcr._block_record(key).local_dram = _LocalDramResidency(
                locations, _LocalDramState.FILLING
            )
            copy_keys.append(key)
            slots.append(tuple(locations))
            src_descriptors.extend(sources)
            dst_descriptors.extend(self._descriptors(locations))

        self._update_capacity_pressure()
        self._kvcr._publish_inventory(evicted, CacheTier.LOCAL_G2, removed=True)
        self._finish_residency_if_ready(op)
        if copy_keys:
            self._kvcr._progress.submit(
                _LocalCopyOp(
                    op_id=("local_copy", self._next_copy_id),
                    keys=set(copy_keys),
                    deliver_op_id=None,
                    ordered_keys=tuple(copy_keys),
                    local_slots=tuple(slots),
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
        *,
        hints: object | None,
        layout: list[str],
    ) -> dict[BlockKey, list[MemDescriptor]]:
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
                if key in sources:
                    to_reserve.append(key)
                else:
                    op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
            elif residency.state is _LocalDramState.READY:
                if [name for name, _ in residency.slots] == layout:
                    self._kvcr._record_access((key,))
                    op.results[key] = self._new_public_claim(
                        key, residency, include_descriptors=True
                    )
                else:
                    op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
            elif [name for name, _ in residency.slots] != layout:
                op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
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
            framework_hints=hints,
            layouts={key: layout for key in to_reserve},
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
        slots: list[tuple[tuple[str, int], ...]] = []
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
            slots.append(tuple(residency.slots))
        self._apply_fill_result(
            ordered_keys, tuple(slots), success, CacheTier.REMOTE_G2
        )

    def deliver(
        self,
        op_handle: OpHandle,
        blocks: Mapping[BlockKey, list[MemDescriptor]],
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
    ) -> dict[BlockKey, list[MemDescriptor]]:
        sources: dict[BlockKey, list[MemDescriptor]] = {}
        for key in keys:
            if key in sources:
                continue
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if residency is None or residency.state is not _LocalDramState.READY:
                continue
            self._acquire_claim(key, residency)
            sources[key] = self._descriptors(residency.slots)
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
        byte_count = sum(descriptor.size for descriptor in copy.src_descriptors)
        self._kvcr._record_transfer(
            "local_deliver" if copy.deliver_op_id is not None else "local_fill",
            copy.started_at,
            copy.success,
            len(copy.ordered_keys),
            byte_count,
        )
        if copy.deliver_op_id is not None:
            self._finish_delivery_copy(copy)
            return

        self._apply_fill_result(
            copy.ordered_keys,
            copy.local_slots,
            copy.success,
            CacheTier.FW_G2,
        )

    def _apply_fill_result(
        self,
        ordered_keys: tuple[BlockKey, ...],
        local_slots: tuple[tuple[tuple[str, int], ...], ...],
        success: bool,
        source: CacheTier,
    ) -> None:
        committed: list[BlockKey] = []
        affected_residency_ops: dict[_OpId, _PendingResidencyOp] = {}
        affected_deliver_ops: dict[_OpId, _PendingDeliverOp] = {}
        deliver_keys: dict[_OpId, list[BlockKey]] = {}
        now = self._kvcr._clock()
        for key, slots in zip(ordered_keys, local_slots):
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if (
                record is None
                or residency is None
                or residency.slots != list(slots)
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
                    key, record, self._size_bytes(residency.slots)
                )
                self._kvcr._on_ingest(meta, source)
                self._make_evictable(key)
                committed.append(key)
            else:
                record.local_dram = None
                self._free(residency.slots)

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
        framework_hints: object | None = None,
        layouts: Mapping[BlockKey, list[str]],
    ) -> tuple[dict[BlockKey, list[MemDescriptor]], set[BlockKey]]:
        keys = tuple(dict.fromkeys(keys))
        protected = set(keys)
        destinations: dict[BlockKey, list[MemDescriptor]] = {}
        eviction_pending: set[BlockKey] = set()
        evicted: list[BlockKey] = []
        for key in keys:
            record = self._kvcr._block_record_map.get(key)
            if record is None:
                raise RuntimeError(f"missing block record for {key!r}")
            if record.local_dram is not None:
                continue
            decision = self._kvcr._policy.decide_ingest(
                self._kvcr._block_meta(
                    key,
                    record,
                    sum(self._pools[name][2] for name in layouts[key]),
                ),
                sources[key],
                required_local,
                framework_hints=framework_hints,
            )
            if decision[0] is PlacementAction.DROP:
                continue
            locations, evicted_keys, waiting = self._allocate_slots(
                layouts[key], protected, deadline
            )
            evicted.extend(evicted_keys)
            if locations is None:
                if waiting:
                    eviction_pending.add(key)
                continue
            self._kvcr._block_record(key).local_dram = _LocalDramResidency(
                locations, _LocalDramState.FILLING
            )
            destinations[key] = self._descriptors(locations)
        self._update_capacity_pressure()
        self._kvcr._publish_inventory(evicted, CacheTier.LOCAL_G2, removed=True)
        return destinations, eviction_pending

    def _start_deliveries(
        self, op: _PendingDeliverOp, keys: Collection[BlockKey]
    ) -> None:
        copy_keys: list[BlockKey] = []
        local_slots: list[tuple[tuple[str, int], ...]] = []
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
                or not self._same_layout(residency.slots, op.destinations[key])
                or now >= op.deadline
            ):
                op.results[key] = OpEntryResult(OpEntryStatus.FAILED)
            else:
                self._acquire_claim(key, residency)
                op.active_keys.add(key)
                copy_keys.append(key)
                local_slots.append(tuple(residency.slots))
                src_descriptors.extend(self._descriptors(residency.slots))
                dst_descriptors.extend(op.destinations[key])

        self._update_capacity_pressure()
        if copy_keys:
            self._kvcr._progress.submit(
                _LocalCopyOp(
                    op_id=("local_copy", self._next_copy_id),
                    keys=set(copy_keys),
                    deliver_op_id=op.op_id,
                    ordered_keys=tuple(copy_keys),
                    local_slots=tuple(local_slots),
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
        for key, slots in zip(copy.ordered_keys, copy.local_slots):
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if (
                residency is None
                or residency.slots != list(slots)
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
        op: _PendingResidencyOp,
        key: BlockKey,
        source: list[MemDescriptor] | CacheTier,
        layout: list[str],
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
            while self._capacity_waiters:
                waiter = self._capacity_waiters[0]
                op = waiter.op
                if (
                    waiter.key not in op.capacity_waiters
                    or self._pending_residency_ops.get(op.op_id) is not op
                ):
                    self._capacity_waiters.popleft()
                    continue
                if waiter.key in op.results:
                    self._capacity_waiters.popleft()
                    op.capacity_waiters.remove(waiter.key)
                    continue
                if self._kvcr._clock() >= op.deadline:
                    self._capacity_waiters.popleft()
                    op.capacity_waiters.remove(waiter.key)
                    op.results[waiter.key] = OpEntryResult(OpEntryStatus.FAILED)
                    self._finish_residency_if_ready(op)
                    continue

                record = self._kvcr._block_record(waiter.key)
                residency = record.local_dram
                if residency is not None:
                    self._capacity_waiters.popleft()
                    op.capacity_waiters.remove(waiter.key)
                    if [name for name, _ in residency.slots] != waiter.layout:
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
                    elif residency.state is _LocalDramState.DISCARDING:
                        op.results[waiter.key] = OpEntryResult(OpEntryStatus.FAILED)
                    self._finish_residency_if_ready(op)
                    continue

                locations, evicted_keys, eviction_pending = self._allocate_slots(
                    waiter.layout, op.keys, op.deadline
                )
                if evicted_keys:
                    self._kvcr._publish_inventory(
                        evicted_keys, CacheTier.LOCAL_G2, removed=True
                    )
                if locations is None:
                    if eviction_pending:
                        break
                    self._capacity_waiters.popleft()
                    op.capacity_waiters.remove(waiter.key)
                    op.results[waiter.key] = OpEntryResult(OpEntryStatus.FAILED)
                    self._finish_residency_if_ready(op)
                    continue

                self._capacity_waiters.popleft()
                op.capacity_waiters.remove(waiter.key)
                record.local_dram = _LocalDramResidency(
                    locations, _LocalDramState.FILLING
                )
                if isinstance(waiter.source, CacheTier):
                    op.remote_fill_keys.add(waiter.key)
                    self._kvcr._start_local_fill(
                        waiter.source,
                        {waiter.key: self._descriptors(locations)},
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
                            local_slots=(tuple(locations),),
                            src_descriptors=tuple(waiter.source),
                            dst_descriptors=tuple(self._descriptors(locations)),
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
            self._descriptors(residency.slots) if include_descriptors else None,
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
        residency.claim_count -= 1
        if residency.claim_count == 0:
            if residency.retire_on_release:
                record.local_dram = None
                self._residency_observer(key, record)
                self._free(residency.slots)
                self.abandon_capacity_eviction(key)
                self._kvcr._on_remove(
                    self._kvcr._block_meta(
                        key, record, self._size_bytes(residency.slots)
                    )
                )
                self._kvcr._publish_inventory((key,), CacheTier.LOCAL_G2, removed=True)
                self._kvcr._prune_block_record(key)
                self._resume_capacity_waiters()
            else:
                self._make_evictable(key)

    def _allocate_slots(
        self, pool_names: list[str], protected: set[BlockKey], deadline: float
    ) -> tuple[list[tuple[str, int]] | None, list[BlockKey], bool]:
        required = Counter(pool_names)
        available = {name: len(self._free_slots[name]) for name in required}
        if all(available[name] >= count for name, count in required.items()):
            return (
                [(name, self._free_slots[name].popleft()) for name in pool_names],
                [],
                False,
            )
        if self._capacity_eviction_key is not None:
            return None, [], True
        self._retry_unscored()
        skipped = set(protected)
        victims: list[tuple[BlockKey, "_BlockRecord", _LocalDramResidency, int]] = []
        while (key := self._evictable.select(skipped)) is not None:
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if (
                record is None
                or residency is None
                or residency.state is not _LocalDramState.READY
                or residency.claim_count
            ):
                raise RuntimeError(f"invalid evictable local DRAM entry {key!r}")
            if not any(
                name in required and available[name] < required[name]
                for name, _ in residency.slots
            ):
                skipped.add(key)
                continue
            size_bytes = self._size_bytes(residency.slots)
            free_before = {name: len(self._free_slots[name]) for name in required}
            decision, eviction_pending = self._kvcr._decide_eviction(
                self._kvcr._block_meta(key, record, size_bytes),
                CacheTier.LOCAL_G2,
                deadline,
            )
            for name in required:
                available[name] += len(self._free_slots[name]) - free_before[name]
            if all(available[name] >= count for name, count in required.items()):
                break
            if eviction_pending:
                self._capacity_eviction_key = key
                return None, [], True
            if decision[0] is PlacementAction.KEEP:
                skipped.add(key)
                continue
            victims.append((key, record, residency, size_bytes))
            skipped.add(key)
            for name, _ in residency.slots:
                if name in available:
                    available[name] += 1
            if all(available[name] >= count for name, count in required.items()):
                break
        else:
            return None, [], False

        for key, record, residency, size_bytes in victims:
            self._remove_evictable(key, residency)
            record.local_dram = None
            self._residency_observer(key, record)
            self._kvcr._on_remove(self._kvcr._block_meta(key, record, size_bytes))
            self._kvcr._prune_block_record(key)
            self._free(residency.slots)
        return (
            [(name, self._free_slots[name].popleft()) for name in pool_names],
            [victim[0] for victim in victims],
            False,
        )

    def _make_evictable(self, key: BlockKey) -> None:
        record = self._kvcr._block_record_map.get(key)
        if record is None:
            raise RuntimeError(f"missing block record for {key!r}")
        score = self._kvcr._policy.eviction_score(
            self._kvcr._block_meta(
                key, record, self._size_bytes(record.local_dram.slots)
            ),
            CacheTier.LOCAL_G2,
        )
        if score is None:
            self._unscored.add(key)
            return
        self._unscored.discard(key)
        if self._evictable.insert(key, score):
            self._evictable_slots.update(name for name, _ in record.local_dram.slots)

    def _remove_evictable(self, key: BlockKey, residency: _LocalDramResidency) -> None:
        self._unscored.discard(key)
        if self._evictable.remove(key):
            self._evictable_slots.subtract(name for name, _ in residency.slots)

    def _retry_unscored(self) -> None:
        for key in tuple(self._unscored):
            self._make_evictable(key)

    def _descriptors(
        self, locations: Collection[tuple[str, int]]
    ) -> list[MemDescriptor]:
        return [self._descriptor(pool_name, slot) for pool_name, slot in locations]

    def _descriptor(self, pool_name: str, slot: int) -> MemDescriptor:
        address, _, slot_size = self._pools[pool_name]
        return MemDescriptor(
            end_point_name=self._kvcr.nixl_agent_name,
            mem_type="DRAM",
            addr=address + slot * slot_size,
            size=slot_size,
            device_Id=0,
            info=pool_name,
        )

    def _free(self, locations: Collection[tuple[str, int]]) -> None:
        for pool_name, slot in locations:
            self._free_slots[pool_name].append(slot)

    def _size_bytes(self, locations: Collection[tuple[str, int]]) -> int:
        return sum(self._pools[pool_name][2] for pool_name, _ in locations)

    @staticmethod
    def _same_layout(
        locations: Collection[tuple[str, int]], descriptors: Collection[MemDescriptor]
    ) -> bool:
        return [name for name, _ in locations] == [
            descriptor.info for descriptor in descriptors
        ]

    def _update_capacity_pressure(self) -> None:
        if not self._kvcr._capacity_pressure_enabled:
            return
        self._kvcr._update_capacity_pressure(
            {
                name: len(slots) + self._evictable_slots[name]
                for name, slots in self._free_slots.items()
            }
        )
