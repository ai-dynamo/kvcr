# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KVCR-owned local DRAM slots, claims, and transfers."""

import logging
from collections import Counter, deque
from collections.abc import Callable, Collection, Mapping
from contextlib import closing
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from itertools import islice
from typing import TYPE_CHECKING, Literal, cast

from .config import LocalDramOptions
from .policy_runtime import _EvictionQueue
from .progress import (
    _DROPPED_RESULT,
    _FAILED_RESULT,
    _SUCCESS_RESULT,
    _KVCRProgress,
    _Op,
    _OpId,
    _ProgressOp,
    _TransferRef,
)
from .types import (
    BlockKey,
    CacheTier,
    MemoryRef,
    OpEntryResult,
    OpEntryStatus,
    OpHandle,
    PlacementAction,
    RegionDescriptor,
    ReleaseHandle,
    ReleaseResult,
    TransferError,
)

if TYPE_CHECKING:
    from .core import _BlockRecord, _KVCRCore

logger = logging.getLogger(__name__)

_Clock = Callable[[], float]


def _layout_indices(
    layout: list[str],
    requested: list[str],
    *,
    indices_by_label: dict[str, int | None] | None = None,
) -> list[int] | None:
    """Match full layouts or unambiguous pieces in the requested order."""
    if layout == requested:
        return list(range(len(layout)))
    if not requested:
        return None
    labels = set(requested)
    if len(labels) != len(requested):
        # Repeated pools are positional: require every occurrence in layout order.
        indices = [index for index, label in enumerate(layout) if label in labels]
        return indices if [layout[index] for index in indices] == requested else None
    if indices_by_label is not None:
        indices = [indices_by_label.get(label) for label in requested]
        return None if None in indices else cast(list[int], indices)
    if any(layout.count(label) != 1 for label in requested):
        return None
    return [layout.index(label) for label in requested]


class _LocalDramState(Enum):
    FILLING = auto()
    READY = auto()
    DISCARDING = auto()


@dataclass(slots=True)
class _LocalDramResidency:
    slots: list[tuple[str, int]]  # Full descriptor label and parent-pool slot.
    state: _LocalDramState
    claim_count: int = 0
    retire_on_release: bool = False
    key_layout: tuple[str, ...] | None = None
    # Reserved pieces become readable only after a successful copy.
    pending_labels: set[str] = field(default_factory=set, repr=False)
    pending_deposits: int = 0
    writers: list[int] = field(default_factory=list, repr=False)
    remote_indices: tuple[int, ...] | None = field(default=None, repr=False)
    remote_cancelled: bool = False
    # Cached references name committed physical slots.
    refs: dict[int, _TransferRef] = field(
        default_factory=dict, repr=False, compare=False
    )
    indices_by_label: dict[str, int | None] | None = field(
        default=None, repr=False, compare=False
    )
    _layout: list[str] | None = field(default=None, repr=False, compare=False)
    _pool_counts: Counter[str] | None = field(default=None, repr=False, compare=False)
    _cached_size: int | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.key_layout is None:
            self.key_layout = tuple(label for label, _ in self.slots)
        if self.state is _LocalDramState.FILLING:
            self.pending_labels.update(label for label, _ in self.slots)
        if self.pending_labels and not self.writers:
            self.writers = [0] * len(self.slots)

    @property
    def layout(self) -> list[str]:
        if self._layout is None:
            self._layout = [name for name, _ in self.slots]
        return self._layout

    @property
    def pool_counts(self) -> Counter[str]:
        if self._pool_counts is None:
            self._pool_counts = Counter(
                label.partition(":")[0] for label, _ in self.slots
            )
        return self._pool_counts

    def prepare_label_index(self) -> None:
        # The reservation and its indices stay fixed for this residency.
        if self.indices_by_label is None:
            indices: dict[str, int | None] = {}
            for index, label in enumerate(self.layout):
                indices[label] = None if label in indices else index
            self.indices_by_label = indices

    def ready_indices(self, requested: list[str]) -> list[int] | None:
        if self.pending_labels and not self.pending_labels.isdisjoint(requested):
            return None
        return _layout_indices(
            self.layout, requested, indices_by_label=self.indices_by_label
        )

    def can_wait_for_pieces(self, requested: list[str]) -> bool:
        return (
            _layout_indices(
                self.layout, requested, indices_by_label=self.indices_by_label
            )
            is not None
        )


@dataclass
class _PendingResidencyOp(_Op):
    deadline: float
    claim_on_ready: bool
    request_id: str | None = None
    results: dict[BlockKey, OpEntryResult] = field(default_factory=dict)
    remote_fill_keys: set[BlockKey] = field(default_factory=set)
    capacity_waiters: set[BlockKey] = field(default_factory=set)
    sources: Mapping[BlockKey, list[_TransferRef]] | None = None
    hints: object | None = None
    active_keys: set[BlockKey] = field(default_factory=set)
    requested_layout: list[str] | None = None
    key_layout: tuple[str, ...] | None = None
    fill_sources: Mapping[BlockKey, CacheTier] = field(default_factory=dict)
    claimed_keys: set[BlockKey] = field(default_factory=set)


@dataclass
class _PendingDeliverOp(_Op):
    deadline: float
    destinations: Mapping[BlockKey, list[_TransferRef]]
    results: dict[BlockKey, OpEntryResult] = field(default_factory=dict)
    active_keys: set[BlockKey] = field(default_factory=set)
    claimed_keys: set[BlockKey] = field(default_factory=set)


@dataclass(frozen=True)
class _CapacityWaiter:
    op: _PendingResidencyOp
    key: BlockKey


@dataclass
class _LocalCopyOp(_ProgressOp):
    deliver_op_id: _OpId | None
    ordered_keys: tuple[BlockKey, ...]
    local_indices: tuple[tuple[int, ...], ...]
    src_descriptors: tuple[_TransferRef, ...]
    dst_descriptors: tuple[_TransferRef, ...]
    deadline: float
    abandon_deadline: float
    backend: str
    clock: _Clock = field(repr=False, compare=False)
    started_at: float | None = field(repr=False, compare=False)
    transfer_id: int | None = None
    success: bool = False
    failure_latched: bool = False
    uncertain: bool = False
    deposit_op: _PendingResidencyOp | None = None

    def progress(
        self, progress: _KVCRProgress, event: object | None
    ) -> tuple[bool, bool]:
        if event is not None:
            raise RuntimeError(f"unexpected local-copy event: {event!r}")
        observed_work = False
        if self.transfer_id is None:
            if self.clock() >= self.deadline:
                logger.debug(
                    "KVCR_EVENT local_copy_expired_before_submit op=%s", self.op_id
                )
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
                self.failure_latched = not submitted
                if not submitted:
                    logger.debug(
                        "KVCR_EVENT local_copy_submit_failed op=%s", self.op_id
                    )
                observed_work = True
            except Exception:
                logger.warning("KVCR local transfer submission failed", exc_info=True)
                return True, True

        transfer_id = self.transfer_id
        if transfer_id is None:
            raise RuntimeError(f"KVCR local copy {self.op_id!r} lost transfer")
        if not self.failure_latched and self.clock() >= self.deadline:
            logger.debug("KVCR_EVENT local_copy_deadline_expired op=%s", self.op_id)
            self.failure_latched = True
            observed_work = True
        # shortcut: cached generic errors retain memory until exit; reclaim only
        # after backend-specific error handling proves native access has stopped.
        result = progress.poll_transfer(
            transfer_id,
            require_completion=True,
        )
        if result is None:
            if (
                self.failure_latched
                and not self.uncertain
                and self.clock() >= self.abandon_deadline
            ):
                self.uncertain = True
                self._report_memory(progress, "uncertain")
                # Report failure without returning the native handle or pool memory.
                progress._completed_backlog.append(replace(self))
            return False, observed_work
        self.transfer_id = None
        self.success = result[0] and not self.failure_latched
        if self.uncertain:
            self._report_memory(progress, "quiesced")
        return True, True

    def close(self, progress: _KVCRProgress) -> bool:
        self.failure_latched = True
        self.success = False
        return self.transfer_id is None

    def _report_memory(
        self, progress: _KVCRProgress, state: Literal["uncertain", "quiesced"]
    ) -> None:
        sources = iter(self.src_descriptors)
        source_blocks = (
            {
                key: [
                    MemoryRef(
                        end_point_name=ref.end_point_name,
                        label=ref.label,
                        element_index=ref.element_index,
                    )
                    for ref in islice(sources, len(indices))
                ]
                for key, indices in zip(self.ordered_keys, self.local_indices)
            }
            if self.deliver_op_id is None
            else None
        )
        progress._completed_backlog.append(
            TransferError(
                "KVCR local copy memory",
                self.op_id[1],
                state=state,
                source_blocks=source_blocks,
                source_owners=dict.fromkeys(source_blocks, "framework")
                if source_blocks is not None
                else None,
                destination_regions=[
                    MemoryRef(
                        end_point_name=ref.end_point_name,
                        label=ref.label,
                        element_index=ref.element_index,
                    )
                    for ref in self.dst_descriptors
                ],
                destination_owner="kvcr" if self.deliver_op_id is None else "framework",
            )
        )


class _LocalDram:
    """Caller-owned DRAM metadata, with synchronized progress-side source claims."""

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
            self._free_slots[pool_name] = deque(range(slot_count))
        ranges = sorted(
            (address, address + length) for address, length, _ in self._pools.values()
        )
        if any(
            left_end > right_start
            for (_, left_end), (right_start, _) in zip(ranges, ranges[1:])
        ):
            raise ValueError("local DRAM pools must not overlap")
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
        self._abandon_grace = (
            kvcr.config.abandon_timeout_ms - kvcr.config.operation_timeout_ms
        ) / 1000
        self._next_release_handle = 1
        # A no-op until something attaches: the tiers publish residency
        # changes unconditionally, and only recovery cares to hear them.
        self._residency_observer: Callable[[BlockKey, "_BlockRecord"], None] = (
            lambda key, record: None
        )

    @property
    def memory_regions(self) -> tuple[RegionDescriptor, ...]:
        return tuple(
            RegionDescriptor(addr=address, count=length // size, size=size, label=name)
            for name, (address, length, size) in self._pools.items()
        )

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
            if (
                residency.state is not _LocalDramState.READY
                or not residency.slots
                or residency.pending_labels
            ):
                raise ValueError("invalid local DRAM recovery slots")
            labels: set[str] = set()
            for label, slot in residency.slots:
                pool_name, separator, part = label.partition(":")
                pool = self._pools.get(pool_name)
                if (
                    pool is None
                    or (separator and (part in ("", "*") or label in labels))
                    or type(slot) is not int
                    or not 0 <= slot < pool[1] // pool[2]
                    or slot in occupied[pool_name]
                ):
                    raise ValueError("invalid local DRAM recovery slots")
                labels.add(label)
                occupied[pool_name].add(slot)
            declaration = self._kvcr._validate_expected_layout(
                list(residency.key_layout)
            )
            if list(declaration) != residency.layout:
                raise ValueError("invalid local DRAM recovery layout")
            # Byte counts depend on the receiving core's pool geometry.
            residency._cached_size = None
            residency.prepare_label_index()
        self._free_slots = {
            pool_name: deque(
                slot
                for slot in range(length // slot_size)
                if slot not in occupied[pool_name]
            )
            for pool_name, (_, length, slot_size) in self._pools.items()
        }

    def rank_recovered(self, records: Mapping[BlockKey, "_BlockRecord"]) -> None:
        """Make complete recovered rows evictable once the policy can score them.

        Separate from adopt_recovery_slots because a score is asked of the policy,
        and the policy only knows a block once it has been admitted. Without this a
        pool recovered full has no free row and no victim, so it refuses every
        deposit until a reader happens to release one of the recovered rows.
        """
        for key, record in records.items():
            if record.local_dram is not None:
                self._make_evictable(key)

    def telemetry_state(self) -> dict[str, int]:
        total_slots = sum(
            length // slot_size for _, length, slot_size in self._pools.values()
        )
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
        blocks: Mapping[BlockKey, list[_TransferRef]],
        *,
        no_evict: bool,
        hints: object | None,
        key_layout: tuple[str, ...] | None,
    ) -> None:
        keys = frozenset(blocks)
        if not keys:
            self._kvcr._complete(op_handle, {})
            return

        deadline = self._kvcr._operation_deadline()
        op = _PendingResidencyOp(
            op_id=("deposit", op_handle),
            keys=keys,
            deadline=deadline,
            claim_on_ready=no_evict,
            sources=blocks,
            hints=hints,
            key_layout=key_layout,
        )
        self._pending_residency_ops[op.op_id] = op
        self._kvcr._add_block_dependencies(op, new_operation=True)
        self._start_deposits(op, blocks)

    def _start_deposits(
        self, op: _PendingResidencyOp, keys: Collection[BlockKey]
    ) -> None:
        if op.sources is None:
            raise RuntimeError("deposit has no source descriptors")
        copy_keys: list[BlockKey] = []
        copy_indices: list[tuple[int, ...]] = []
        src_descriptors: list[_TransferRef] = []
        dst_descriptors: list[_TransferRef] = []
        evicted: list[BlockKey] = []
        declared_local = list(op.key_layout) if op.key_layout is not None else None
        for key in keys:
            if key in op.results or key in op.capacity_waiters or key in op.active_keys:
                continue
            if self._kvcr._clock() >= op.deadline:
                op.results[key] = _FAILED_RESULT
                continue
            sources = op.sources[key]
            layout = [descriptor.label for descriptor in sources]
            record = self._kvcr._block_record(key)
            residency = record.local_dram
            full_deposit = False
            if residency is not None:
                if (
                    residency.state is _LocalDramState.DISCARDING
                    or residency.retire_on_release
                    or op.key_layout is not None
                    and residency.key_layout != op.key_layout
                ):
                    op.results[key] = _FAILED_RESULT
                    continue
            else:
                declaration = op.key_layout or tuple(layout)
                local_layout = declared_local if declared_local is not None else layout
                size_bytes = sum(
                    self._pools[label.partition(":")[0]][2] for label in local_layout
                )
                decision = self._kvcr._policy.decide_ingest(
                    self._kvcr._block_meta(key, record, size_bytes),
                    CacheTier.FW_G2,
                    required_local=op.claim_on_ready,
                    framework_hints=op.hints,
                )
                if decision[0] is PlacementAction.DROP:
                    op.results[key] = _DROPPED_RESULT
                    continue
                locations, evicted_keys, eviction_pending = self._allocate_slots(
                    local_layout, op.keys, op.deadline
                )
                evicted.extend(evicted_keys)
                if locations is None:
                    if eviction_pending:
                        self._enqueue_capacity_waiter(op, key)
                    else:
                        op.results[key] = _FAILED_RESULT
                    continue
                full_deposit = layout == local_layout
                residency = record.local_dram = _LocalDramResidency(
                    locations,
                    _LocalDramState.FILLING,
                    key_layout=declaration,
                    writers=[1 if full_deposit else 0] * len(locations),
                )
            if full_deposit:
                selected_slots = residency.slots
                selected_indices = tuple(range(len(selected_slots)))
                src_descriptors.extend(sources)
            else:
                residency.prepare_label_index()
                indices = _layout_indices(
                    residency.layout,
                    layout,
                    indices_by_label=residency.indices_by_label,
                )
                if indices is None:
                    op.results[key] = _FAILED_RESULT
                    continue
                selected_indices = tuple(
                    index
                    for index in indices
                    if residency.slots[index][0] in residency.pending_labels
                )
                if not selected_indices:
                    op.results[key] = (
                        self._new_public_claim(key, residency)
                        if op.claim_on_ready
                        else _SUCCESS_RESULT
                    )
                    continue
                if all(residency.writers[index] for index in selected_indices):
                    continue  # Attach duplicates; new pieces can still proceed.
                self._remove_evictable(key, residency)
                selected_slots = []
                # Mixed deposits may copy an active slot too; bytes must be identical.
                for ref, index in zip(sources, indices):
                    if residency.slots[index][0] in residency.pending_labels:
                        selected_slots.append(residency.slots[index])
                        src_descriptors.append(ref)
                        residency.writers[index] += 1
            residency.claim_count += 1
            residency.pending_deposits += 1
            op.active_keys.add(key)
            copy_keys.append(key)
            copy_indices.append(selected_indices)
            dst_descriptors.extend(self._descriptors(selected_slots))

        self._update_capacity_pressure()
        self._kvcr._publish_inventory(evicted, CacheTier.LOCAL_G2, removed=True)
        self._finish_residency_if_ready(op)
        if copy_keys:
            self._kvcr._progress.submit(
                _LocalCopyOp(
                    op_id=("local_copy", self._kvcr._next_fill_handle),
                    keys=frozenset(copy_keys),
                    deliver_op_id=None,
                    ordered_keys=tuple(copy_keys),
                    local_indices=tuple(copy_indices),
                    src_descriptors=tuple(src_descriptors),
                    dst_descriptors=tuple(dst_descriptors),
                    deadline=op.deadline,
                    abandon_deadline=op.deadline + self._abandon_grace,
                    backend=self._backend,
                    clock=self._kvcr._clock,
                    started_at=self._kvcr._timer(),
                    deposit_op=op,
                )
            )
            self._kvcr._next_fill_handle -= 1

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
        key_layout: tuple[str, ...] | None = None,
    ) -> dict[BlockKey, list[_TransferRef]]:
        ordered_keys = tuple(dict.fromkeys(keys))
        if not ordered_keys:
            self._kvcr._complete(op_handle, {})
            return {}
        op = _PendingResidencyOp(
            op_id=("fetch", op_handle),
            keys=frozenset(ordered_keys),
            deadline=deadline,
            claim_on_ready=True,
            request_id=request_id,
            requested_layout=layout,
            key_layout=key_layout,
            fill_sources=sources,
            hints=hints,
        )
        self._pending_residency_ops[op.op_id] = op
        self._kvcr._add_block_dependencies(op, new_operation=True)
        destinations = {}
        for key in ordered_keys:
            refs = self._prepare_fetch(op, key)
            if refs:
                destinations[key] = refs
        self._finish_residency_if_ready(op)
        return destinations

    def _prepare_fetch(
        self, op: _PendingResidencyOp, key: BlockKey
    ) -> list[_TransferRef] | None:
        layout = op.requested_layout
        record = self._kvcr._block_record(key)
        residency = record.local_dram
        source = op.fill_sources.get(key)
        if self._kvcr._clock() >= op.deadline:
            op.results[key] = _FAILED_RESULT
            return None
        if residency is not None and residency.state is _LocalDramState.DISCARDING:
            if source is not None:
                self._enqueue_capacity_waiter(op, key)
            else:
                op.results[key] = _FAILED_RESULT
            return None
        if residency is None:
            if source is None:
                op.results[key] = _FAILED_RESULT
                return None
            reserved, waiting = self._reserve_fill(key, op, source)
            if not reserved:
                if waiting:
                    self._enqueue_capacity_waiter(op, key)
                else:
                    op.results[key] = _FAILED_RESULT
                return None
            residency = record.local_dram
        elif op.key_layout is not None and op.key_layout != residency.key_layout:
            op.results[key] = _FAILED_RESULT
            return None
        residency.prepare_label_index()
        indices = _layout_indices(
            residency.layout, layout, indices_by_label=residency.indices_by_label
        )
        if indices is None:
            op.results[key] = _FAILED_RESULT
        elif not residency.pending_labels or residency.pending_labels.isdisjoint(
            layout
        ):
            ready = [residency.slots[index] for index in indices]
            self._kvcr._record_access((key,))
            op.results[key] = self._new_public_claim(key, residency, slots=ready)
        else:
            if key not in op.claimed_keys:
                self._acquire_claim(key, residency)
                op.claimed_keys.add(key)
            if source is None or residency.remote_indices is not None:
                return None
            # One remote fill owns these destinations until its native terminal.
            missing = [
                index
                for index in indices
                if residency.slots[index][0] in residency.pending_labels
                and not residency.writers[index]
            ]
            if missing:
                self._remove_evictable(key, residency)
                residency.remote_indices = tuple(missing)
                for index in missing:
                    residency.writers[index] += 1
                residency.claim_count += 1
                op.remote_fill_keys.add(key)
                return self._descriptors([residency.slots[index] for index in missing])
        return None

    def complete_fill(self, keys: Collection[BlockKey], *, success: bool) -> None:
        ordered_keys = tuple(keys)
        indices = []
        for key in ordered_keys:
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if residency is None or residency.remote_indices is None:
                raise RuntimeError(f"local DRAM fill state lost for {key!r}")
            indices.append(residency.remote_indices)
        self._apply_fill_result(
            ordered_keys, tuple(indices), success, CacheTier.REMOTE_G2
        )

    def deliver(
        self,
        op_handle: OpHandle,
        blocks: Mapping[BlockKey, list[_TransferRef]],
        *,
        deadline: float,
    ) -> None:
        op = _PendingDeliverOp(
            op_id=("local_deliver", op_handle),
            keys=frozenset(blocks),
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
        self,
        keys: Collection[BlockKey],
        *,
        notify_capacity: bool = True,
        allow_filling: bool = False,
    ) -> dict[BlockKey, _LocalDramResidency]:
        sources: dict[BlockKey, _LocalDramResidency] = {}
        for key in keys:
            if key in sources:
                continue
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if residency is None or (
                residency.state is not _LocalDramState.READY
                and not (allow_filling and residency.state is _LocalDramState.FILLING)
            ):
                continue
            self._acquire_claim(key, residency)
            sources[key] = residency
        if notify_capacity:
            self._update_capacity_pressure()
        return sources

    def source_refs(
        self, residency: _LocalDramResidency, indices: list[int]
    ) -> tuple[_TransferRef, ...]:
        refs = []
        for index in indices:
            ref = residency.refs.get(index)
            if ref is None:
                ref = residency.refs[index] = self._descriptor(*residency.slots[index])
            refs.append(ref)
        return tuple(refs)

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
                or (
                    residency.remote_indices is None
                    and residency.state
                    not in (_LocalDramState.FILLING, _LocalDramState.DISCARDING)
                )
            ):
                raise RuntimeError(f"local DRAM fill state lost for {key!r}")
            if residency.remote_indices is None or (
                residency.state is _LocalDramState.FILLING
                and not residency.pending_deposits
            ):
                residency.state = _LocalDramState.DISCARDING
            if residency.remote_indices is not None:
                residency.remote_cancelled = True
            for op_id in record.active_op_ids:
                residency_op = self._pending_residency_ops.get(op_id)
                if (
                    residency_op is not None
                    and key in residency_op.keys
                    and key not in residency_op.results
                    # Capacity waiters never owned this fill, and the slot it
                    # holds is exactly what they are queued for.
                    and key not in residency_op.capacity_waiters
                    and (
                        key in residency_op.remote_fill_keys
                        or residency.state is _LocalDramState.DISCARDING
                    )
                ):
                    residency_op.results[key] = _FAILED_RESULT
                    residency_ops[op_id] = residency_op
                deliver_op = self._pending_deliver_ops.get(op_id)
                if (
                    deliver_op is not None
                    and key in deliver_op.keys
                    and key not in deliver_op.results
                    and residency.state is _LocalDramState.DISCARDING
                ):
                    deliver_op.results[key] = _FAILED_RESULT
                    deliver_ops[op_id] = deliver_op

        for residency_op in residency_ops.values():
            self._finish_residency_if_ready(residency_op)
        for deliver_op in deliver_ops.values():
            self._finish_deliver_if_ready(deliver_op)

    def close(self) -> None:
        # Policy state ends with KVCR; teardown emits no per-block removals.
        self._public_claims.clear()

    def poll_main(self, items: Collection[object]) -> list[object]:
        # Progress-side source claims defer framework callbacks to the caller.
        self._update_capacity_pressure()
        unhandled: list[object] = []
        for item in items:
            if isinstance(item, _LocalCopyOp):
                self._finish_copy(item)
            else:
                unhandled.append(item)
        self._expire_pending_ops(self._kvcr._clock())
        return unhandled

    def _finish_copy(self, copy: _LocalCopyOp) -> None:
        if copy.transfer_id is not None:
            if copy.deliver_op_id is None:
                # Fail this writer, retaining its slots and any committed pieces.
                op = copy.deposit_op
                for key in copy.ordered_keys:
                    op.results[key] = _FAILED_RESULT
                self._finish_residency_if_ready(op)
            else:
                op = self._pending_deliver_ops[copy.deliver_op_id]
                for key in copy.ordered_keys:
                    op.results[key] = _FAILED_RESULT
                self._finish_deliver_if_ready(op)
            return
        byte_count = self._kvcr._descriptor_bytes(copy.src_descriptors)
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
            copy.local_indices,
            copy.success,
            CacheTier.FW_G2,
            writer=copy.deposit_op,
        )

    def _apply_fill_result(
        self,
        ordered_keys: tuple[BlockKey, ...],
        local_indices: tuple[tuple[int, ...], ...],
        success: bool,
        source: CacheTier,
        *,
        writer: _PendingResidencyOp | None = None,
    ) -> None:
        committed: list[BlockKey] = []
        failed: list[BlockKey] = []
        affected_residency_ops: dict[_OpId, _PendingResidencyOp] = {}
        queued_deposits: dict[_OpId, list[BlockKey]] = {}
        affected_deliver_ops: dict[_OpId, _PendingDeliverOp] = {}
        deliver_keys: dict[_OpId, list[BlockKey]] = {}
        now = self._kvcr._clock()
        for key, indices in zip(ordered_keys, local_indices):
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if record is None or residency is None:
                raise RuntimeError(f"local DRAM fill state lost for {key!r}")
            was_ready = residency.state is _LocalDramState.READY
            key_success = (
                success
                and residency.state is not _LocalDramState.DISCARDING
                and (writer is not None or not residency.remote_cancelled)
            )
            for index in indices:
                residency.writers[index] -= 1
                if residency.writers[index] < 0:
                    raise RuntimeError("local DRAM writer ownership lost")
            if writer is not None:
                writer.active_keys.remove(key)
                residency.pending_deposits -= 1
            else:
                residency.remote_indices = None
                residency.remote_cancelled = False
            if key_success:
                pending_count = len(residency.pending_labels)
                if len(indices) == len(residency.slots):
                    residency.pending_labels.clear()
                else:
                    residency.pending_labels.difference_update(
                        residency.slots[index][0] for index in indices
                    )
                if len(residency.pending_labels) < pending_count:
                    residency.state = _LocalDramState.READY
                    residency.prepare_label_index()
                    record.last_access = now
                    if not was_ready:
                        self._kvcr._on_ingest(
                            self._kvcr._block_meta(
                                key, record, self._size_bytes(residency)
                            ),
                            source,
                        )
                    if not residency.pending_labels:
                        self._residency_observer(key, record)
                        committed.append(key)
            if writer is not None:
                if key not in writer.results:
                    writer.results[key] = (
                        self._new_public_claim(key, residency)
                        if key_success and writer.claim_on_ready
                        else _SUCCESS_RESULT
                        if key_success
                        else _FAILED_RESULT
                    )
                    affected_residency_ops[writer.op_id] = writer
            else:
                for op_id in record.active_op_ids:
                    op = self._pending_residency_ops.get(op_id)
                    if op is not None and key in op.remote_fill_keys:
                        op.remote_fill_keys.remove(key)
                        if not key_success:
                            op.results[key] = _FAILED_RESULT
                            affected_residency_ops[op_id] = op
            self._release_claim(key, residency)
            if record.local_dram is None:
                failed.append(key)

            # Releasing the final claim may retire this residency and wake a waiter.
            for op_id in record.active_op_ids:
                if writer is not None and op_id == writer.op_id:
                    continue
                residency_op = self._pending_residency_ops.get(op_id)
                if (
                    residency_op is not None
                    and key in residency_op.keys
                    and key not in residency_op.results
                    and key not in residency_op.capacity_waiters
                    and key not in residency_op.active_keys
                ):
                    if residency_op.sources is not None:
                        queued_deposits.setdefault(op_id, []).append(key)
                        continue
                    if not self._resolve_fetch_waiter(key, residency_op):
                        continue
                    affected_residency_ops[op_id] = residency_op

                deliver_op = self._pending_deliver_ops.get(op_id)
                if (
                    deliver_op is not None
                    and key in deliver_op.keys
                    and key not in deliver_op.results
                ):
                    if record.local_dram is not None:
                        deliver_keys.setdefault(op_id, []).append(key)
                    else:
                        deliver_op.results[key] = _FAILED_RESULT
                    affected_deliver_ops[op_id] = deliver_op

        self._update_capacity_pressure()
        if self._kvcr._key_history_enabled:
            self._kvcr._log_key_history("ready", committed, CacheTier.LOCAL_G2)
            self._kvcr._log_key_history("fill_failed", failed, CacheTier.LOCAL_G2)
        self._kvcr._publish_inventory(committed, CacheTier.LOCAL_G2, removed=False)
        for residency_op in affected_residency_ops.values():
            self._finish_residency_if_ready(residency_op)
        for op_id, keys in queued_deposits.items():
            pending = self._pending_residency_ops.get(op_id)
            if pending is not None:
                self._start_deposits(pending, keys)
        for op_id, deliver_op in affected_deliver_ops.items():
            self._start_deliveries(deliver_op, deliver_keys.get(op_id, ()))
        for key in failed:
            self._kvcr._prune_block_record(key)
        self._resume_capacity_waiters()

    def _resolve_fetch_waiter(
        self,
        key: BlockKey,
        residency_op: _PendingResidencyOp,
    ) -> bool:
        refs = self._prepare_fetch(residency_op, key)
        if refs:
            self._kvcr._start_local_fill(
                residency_op.fill_sources[key],
                {key: refs},
                residency_op.request_id,
                residency_op.deadline,
            )
        return key in residency_op.results

    def _reserve_fill(
        self,
        key: BlockKey,
        op: _PendingResidencyOp,
        source: CacheTier,
    ) -> tuple[bool, bool]:
        layout = op.requested_layout
        record = self._kvcr._block_record(key)
        size_bytes = sum(self._pools[label.partition(":")[0]][2] for label in layout)
        decision = self._kvcr._policy.decide_ingest(
            self._kvcr._block_meta(key, record, size_bytes),
            source,
            required_local=True,
            framework_hints=op.hints,
        )
        if decision[0] is PlacementAction.DROP:
            return False, False
        locations, evicted, waiting = self._allocate_slots(layout, op.keys, op.deadline)
        if locations is not None:
            record.local_dram = _LocalDramResidency(
                locations, _LocalDramState.FILLING, key_layout=op.key_layout
            )
        self._update_capacity_pressure()
        self._kvcr._publish_inventory(evicted, CacheTier.LOCAL_G2, removed=True)
        return locations is not None, waiting

    def _start_deliveries(
        self, op: _PendingDeliverOp, keys: Collection[BlockKey]
    ) -> None:
        copy_keys: list[BlockKey] = []
        src_descriptors: list[_TransferRef] = []
        dst_descriptors: list[_TransferRef] = []
        now = self._kvcr._clock()
        for key in keys:
            if key in op.results or key in op.active_keys:
                continue
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if residency is None:
                op.results[key] = _FAILED_RESULT
            elif residency.state is _LocalDramState.DISCARDING or now >= op.deadline:
                op.results[key] = _FAILED_RESULT
            else:
                labels = [descriptor.label for descriptor in op.destinations[key]]
                residency.prepare_label_index()
                indices = residency.ready_indices(labels)
                if indices is None and not residency.can_wait_for_pieces(labels):
                    op.results[key] = _FAILED_RESULT
                    continue
                if key not in op.claimed_keys:
                    self._acquire_claim(key, residency)
                    op.claimed_keys.add(key)
                if indices is None:
                    # A producer may submit the next piece after this delivery.
                    continue
                op.active_keys.add(key)
                copy_keys.append(key)
                src_descriptors.extend(self.source_refs(residency, indices))
                dst_descriptors.extend(op.destinations[key])

        self._update_capacity_pressure()
        if copy_keys:
            self._kvcr._progress.submit(
                _LocalCopyOp(
                    op_id=("local_copy", self._kvcr._next_fill_handle),
                    keys=frozenset(copy_keys),
                    deliver_op_id=op.op_id,
                    ordered_keys=tuple(copy_keys),
                    local_indices=(),  # The claim protects immutable source slots.
                    src_descriptors=tuple(src_descriptors),
                    dst_descriptors=tuple(dst_descriptors),
                    deadline=op.deadline,
                    abandon_deadline=op.deadline + self._abandon_grace,
                    backend=self._backend,
                    clock=self._kvcr._clock,
                    started_at=self._kvcr._timer(),
                )
            )
            self._kvcr._next_fill_handle -= 1
        self._finish_deliver_if_ready(op)

    def _finish_delivery_copy(self, copy: _LocalCopyOp) -> None:
        if copy.deliver_op_id is None:
            raise RuntimeError("local delivery has no owning operation")
        op = self._pending_deliver_ops.get(copy.deliver_op_id)
        for key in copy.ordered_keys:
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if (
                residency is None
                or (op is not None and key not in op.claimed_keys)
                or residency.state is not _LocalDramState.READY
            ):
                raise RuntimeError(f"local DRAM delivery state lost for {key!r}")
            if copy.success:
                self._kvcr._record_access((key,))
            self._release_claim(key, residency)
            if op is not None:
                op.claimed_keys.remove(key)
                op.active_keys.discard(key)
                op.results[key] = _SUCCESS_RESULT if copy.success else _FAILED_RESULT
        self._update_capacity_pressure()
        if op is not None:
            self._finish_deliver_if_ready(op)

    def _finish_residency_if_ready(self, op: _PendingResidencyOp) -> None:
        if (
            len(op.results) != len(op.keys)
            or self._pending_residency_ops.get(op.op_id) is not op
        ):
            return
        if op.op_id[0] == "fetch":
            remote_keys = self._kvcr._remote_delivery_keys.get(op.request_id)
            if remote_keys:
                remote_keys.difference_update(
                    key for key, result in op.results.items() if result.success
                )
        op.capacity_waiters.clear()
        self._pending_residency_ops.pop(op.op_id)
        claimed, op.claimed_keys = op.claimed_keys, set()
        if claimed:
            self.release_sources(claimed)
        self._kvcr._remove_block_dependencies(op)
        self._kvcr._complete(cast(OpHandle, op.op_id[1]), op.results)

    def _finish_deliver_if_ready(self, op: _PendingDeliverOp) -> None:
        if len(op.results) != len(op.keys):
            return
        # Submitted copies retain their claims until native completion.
        self.release_sources(op.claimed_keys - op.active_keys)
        op.claimed_keys.clear()
        self._pending_deliver_ops.pop(op.op_id)
        self._kvcr._remove_block_dependencies(op)
        self._kvcr._complete(cast(OpHandle, op.op_id[1]), op.results)

    def _expire_pending_ops(self, now: float) -> None:
        for residency_op in list(self._pending_residency_ops.values()):
            # discard_fill() can finish other operations from this snapshot.
            if residency_op.op_id not in self._pending_residency_ops:
                continue
            if now < residency_op.deadline:
                continue
            if residency_op.op_id[0] == "deposit":
                for key in residency_op.keys - residency_op.results.keys():
                    if key not in residency_op.active_keys:
                        residency_op.capacity_waiters.discard(key)
                        residency_op.results[key] = _FAILED_RESULT
                self._finish_residency_if_ready(residency_op)
                continue
            if residency_op.op_id[0] != "fetch":
                continue
            waiting_keys = residency_op.keys - residency_op.results.keys()
            remote_fill_keys = waiting_keys & residency_op.remote_fill_keys
            for key in waiting_keys - remote_fill_keys:
                residency_op.results[key] = _FAILED_RESULT
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
                deliver_op.results[key] = _FAILED_RESULT
            self._finish_deliver_if_ready(deliver_op)

    def _enqueue_capacity_waiter(self, op: _PendingResidencyOp, key: BlockKey) -> None:
        if key in op.capacity_waiters:
            raise RuntimeError(f"duplicate local capacity waiter for {key!r}")
        self._capacity_waiters.append(_CapacityWaiter(op, key))
        op.capacity_waiters.add(key)

    def _resume_capacity_waiters(self) -> None:
        if self._resuming_capacity_waiters:
            return
        self._resuming_capacity_waiters = True
        try:
            # One pass lets other waiters run while discarded fills retain slots.
            for _ in range(len(self._capacity_waiters)):
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
                    op.results[waiter.key] = _FAILED_RESULT
                    self._finish_residency_if_ready(op)
                    continue

                record = self._kvcr._block_record(waiter.key)
                residency = record.local_dram
                if (
                    residency is not None
                    and residency.state is _LocalDramState.DISCARDING
                ):
                    self._capacity_waiters.rotate(-1)
                    continue

                self._capacity_waiters.popleft()
                op.capacity_waiters.remove(waiter.key)
                if op.sources is not None:
                    self._start_deposits(op, (waiter.key,))
                    continue

                self._resolve_fetch_waiter(waiter.key, op)
                if waiter.key in op.capacity_waiters:
                    # Keep a pending spill ahead of later capacity waiters.
                    self._capacity_waiters.rotate(1)
                    break
                self._finish_residency_if_ready(op)
        finally:
            self._resuming_capacity_waiters = False
            self._update_capacity_pressure()

    def _new_public_claim(
        self,
        key: BlockKey,
        residency: _LocalDramResidency,
        *,
        slots: Collection[tuple[str, int]] | None = None,
    ) -> OpEntryResult:
        self._acquire_claim(key, residency)
        handle = ReleaseHandle(self._next_release_handle)
        self._next_release_handle += 1
        self._public_claims[handle] = (key, residency)
        return OpEntryResult(
            OpEntryStatus.SUCCESS,
            [
                MemoryRef(
                    end_point_name=self._kvcr.nixl_agent_name,
                    label=pool_name,
                    element_index=slot,
                )
                for pool_name, slot in slots
            ]
            if slots is not None
            else None,
            handle,
        )

    def _acquire_claim(self, key: BlockKey, residency: _LocalDramResidency) -> None:
        if residency.state is _LocalDramState.DISCARDING:
            raise RuntimeError(f"cannot claim discarded local DRAM entry {key!r}")
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
            if residency.state is not _LocalDramState.READY:
                record.local_dram = None
                self._free(residency.slots)
                self._kvcr._prune_block_record(key)
                self._resume_capacity_waiters()
                return
            if residency.retire_on_release:
                if self._kvcr._key_history_enabled:
                    self._kvcr._log_key_history(
                        "remove", (key,), CacheTier.LOCAL_G2, "retire_on_release"
                    )
                record.local_dram = None
                self._residency_observer(key, record)
                self._free(residency.slots)
                self.abandon_capacity_eviction(key)
                self._kvcr._on_remove(
                    self._kvcr._block_meta(key, record, self._size_bytes(residency))
                )
                if not residency.pending_labels:
                    self._kvcr._publish_inventory(
                        (key,), CacheTier.LOCAL_G2, removed=True
                    )
                self._kvcr._prune_block_record(key)
                self._resume_capacity_waiters()
            else:
                self._make_evictable(key)

    def _allocate_slots(
        self, layout: list[str], protected: frozenset[BlockKey], deadline: float
    ) -> tuple[list[tuple[str, int]] | None, list[BlockKey], bool]:
        required = Counter(label.partition(":")[0] for label in layout)
        if all(
            len(self._free_slots[name]) >= count for name, count in required.items()
        ):
            return (
                [
                    (label, self._free_slots[label.partition(":")[0]].popleft())
                    for label in layout
                ],
                [],
                False,
            )
        if self._capacity_eviction_key is not None:
            return None, [], True
        self._retry_unscored()
        victims: list[tuple[BlockKey, "_BlockRecord", _LocalDramResidency, int]] = []
        freed: Counter[str] = Counter()

        def short() -> set[str]:
            return {
                name
                for name, count in required.items()
                if len(self._free_slots[name]) + freed[name] < count
            }

        deficient = short()
        with closing(self._evictable.candidates(protected)) as candidates:
            while deficient:
                key = next(candidates, None)
                if key is None:
                    return None, [], False
                record = self._kvcr._block_record_map.get(key)
                residency = record.local_dram if record is not None else None
                if (
                    record is None
                    or residency is None
                    or residency.state is not _LocalDramState.READY
                    or residency.claim_count
                ):
                    raise RuntimeError(f"invalid evictable local DRAM entry {key!r}")
                if deficient.isdisjoint(residency.pool_counts):
                    continue
                size_bytes = self._size_bytes(residency)
                decision, eviction_pending = self._kvcr._decide_eviction(
                    self._kvcr._block_meta(key, record, size_bytes),
                    CacheTier.LOCAL_G2,
                    deadline,
                )
                deficient = short()
                if not deficient:
                    break
                if eviction_pending:
                    self._capacity_eviction_key = key
                    return None, [], True
                if decision[0] is PlacementAction.KEEP:
                    continue
                victims.append((key, record, residency, size_bytes))
                freed.update(residency.pool_counts)
                deficient = short()

        for key, record, residency, size_bytes in victims:
            self._remove_evictable(key, residency)
            if self._kvcr._key_history_enabled:
                self._kvcr._log_key_history(
                    "remove", (key,), CacheTier.LOCAL_G2, "capacity_eviction"
                )
            record.local_dram = None
            self._residency_observer(key, record)
            self._kvcr._on_remove(self._kvcr._block_meta(key, record, size_bytes))
            self._kvcr._prune_block_record(key)
            self._free(residency.slots)
        return (
            [
                (label, self._free_slots[label.partition(":")[0]].popleft())
                for label in layout
            ],
            [victim[0] for victim in victims],
            False,
        )

    def _make_evictable(self, key: BlockKey) -> None:
        record = self._kvcr._block_record_map.get(key)
        if record is None:
            raise RuntimeError(f"missing block record for {key!r}")
        residency = record.local_dram
        if (
            residency is None
            or residency.state is not _LocalDramState.READY
            or residency.claim_count
            or residency.pending_labels
        ):
            return
        score = self._kvcr._policy.eviction_score(
            self._kvcr._block_meta(key, record, self._size_bytes(residency)),
            CacheTier.LOCAL_G2,
            previous_score=self._evictable.score(key),
        )
        if score is None:
            self._unscored.add(key)
            return
        self._unscored.discard(key)
        if self._evictable.insert(key, score):
            self._evictable_slots.update(residency.pool_counts)

    def _remove_evictable(self, key: BlockKey, residency: _LocalDramResidency) -> None:
        self._unscored.discard(key)
        if self._evictable.remove(key):
            self._evictable_slots.subtract(residency.pool_counts)

    def _retry_unscored(self) -> None:
        for key in tuple(self._unscored):
            self._make_evictable(key)

    def _descriptors(
        self, locations: Collection[tuple[str, int]]
    ) -> list[_TransferRef]:
        return [self._descriptor(label, slot) for label, slot in locations]

    def _descriptor(self, label: str, slot: int) -> _TransferRef:
        return _TransferRef(
            self._kvcr.nixl_agent_name,
            slot,
            label,
            framework=False,
        )

    def _free(self, locations: Collection[tuple[str, int]]) -> None:
        for label, slot in locations:
            self._free_slots[label.partition(":")[0]].append(slot)

    def _size_bytes(self, residency: _LocalDramResidency) -> int:
        if residency._cached_size is None:
            residency._cached_size = sum(
                self._pools[pool][2] * count
                for pool, count in residency.pool_counts.items()
            )
        return residency._cached_size

    def _update_capacity_pressure(self) -> None:
        if self._kvcr._capacity_needed_callback is None:
            return
        self._kvcr._update_capacity_pressure(
            {
                name: len(slots) + self._evictable_slots[name]
                for name, slots in self._free_slots.items()
            }
        )
