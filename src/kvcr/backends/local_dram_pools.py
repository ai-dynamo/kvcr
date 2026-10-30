# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Local DRAM pool allocation, residency, claims, and eviction."""

import queue
import threading
from collections import Counter, deque
from collections.abc import Callable, Collection, Mapping
from contextlib import closing
from enum import Enum, auto
from typing import TYPE_CHECKING, cast

import msgspec

from ..config import LocalDramOptions
from ..runtime.policy_runtime import _EvictionQueue
from ..runtime.progress import _TransferRef
from ..types import (
    BlockKey,
    CacheTier,
    MemoryRef,
    OpEntryResult,
    OpEntryStatus,
    PlacementAction,
    RegionDescriptor,
    ReleaseHandle,
    ReleaseResult,
)

if TYPE_CHECKING:
    from ..runtime.core import _BlockRecord, _KVCRCore


def _layout_indices(
    layout: tuple[str, ...],
    requested: list[str],
    *,
    indices_by_label: dict[str, int | None] | None = None,
) -> list[int] | None:
    """Match full layouts or unambiguous pieces in the requested order."""
    if len(layout) == len(requested) and layout == tuple(requested):
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


class _LocalDramResidency(msgspec.Struct, gc=False, eq=False):
    """Cycle-free per-block DRAM slots; skip cyclic GC to avoid full-GC stalls.

    Reserved pieces become readable only after successful copies. Once all
    pieces are readable, pending_labels is None; writers is also None when
    no copies are in flight. refs is sparse until a completed residency is
    published for remote reads, then holds its immutable descriptor tuple.
    """

    slots: tuple[tuple[str, int], ...]  # Full descriptor label and parent-pool slot.
    state: _LocalDramState
    claim_count: int = 0
    retire_on_release: bool = False
    key_layout: tuple[str, ...] | None = None
    pending_labels: set[str] | None = None
    pending_deposits: int = 0
    writers: list[int] | None = None
    remote_indices: tuple[int, ...] | None = None
    remote_cancelled: bool = False
    refs: dict[int, _TransferRef] | tuple[_TransferRef, ...] = msgspec.field(
        default_factory=dict
    )
    indices_by_label: dict[str, int | None] | None = None
    _layout: tuple[str, ...] | None = None
    _pool_counts: tuple[tuple[str, int], ...] | None = None
    _cached_size: int | None = None
    source_readers: int = 0  # Includes completed reads awaiting access accounting.

    def __post_init__(self) -> None:
        self.slots = tuple(self.slots)
        if self.key_layout is None:
            self.key_layout = tuple(label for label, _ in self.slots)
        if self.state is _LocalDramState.FILLING:
            self.pending_labels = {label for label, _ in self.slots}
        if self.pending_labels and not self.writers:
            self.writers = [0] * len(self.slots)

    def __eq__(self, other: object) -> bool:
        # The cached fields (refs onward) do not take part in equality.
        if not isinstance(other, _LocalDramResidency):
            return NotImplemented
        return (
            self.slots == other.slots
            and self.state is other.state
            and self.claim_count == other.claim_count
            and self.retire_on_release == other.retire_on_release
            and self.key_layout == other.key_layout
            and self.pending_labels == other.pending_labels
            and self.pending_deposits == other.pending_deposits
            and self.writers == other.writers
            and self.remote_indices == other.remote_indices
            and self.remote_cancelled == other.remote_cancelled
        )

    # The Struct's own __ne__ compares identity (eq=False); derive it from __eq__.
    __ne__ = object.__ne__

    def settle(self) -> None:
        """Drop the fill bookkeeping once nothing is pending or in flight."""
        if not self.pending_labels:
            self.pending_labels = None
            # Writer counts are held only by pending deposits and a remote fill.
            if not self.pending_deposits and self.remote_indices is None:
                self.writers = None

    @property
    def layout(self) -> tuple[str, ...]:
        if self._layout is None:
            self._layout = tuple(name for name, _ in self.slots)
        return self._layout

    @property
    def pool_counts(self) -> tuple[tuple[str, int], ...]:
        """(pool, slot count) pairs; a tuple so it stays out of the GC."""
        if self._pool_counts is None:
            slots = self.slots
            if len(slots) == 1:
                self._pool_counts = ((slots[0][0].partition(":")[0], 1),)
            else:
                self._pool_counts = tuple(
                    Counter(label.partition(":")[0] for label, _ in slots).items()
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


class _SourceState:
    """Reader state shared with progress; completions return work to the caller."""

    def __init__(self) -> None:
        # Owner paths take _state_lock before this lock; progress takes only this.
        self.lock = threading.Lock()
        self.ready: dict[BlockKey, _LocalDramResidency] = {}
        self.active: dict[BlockKey, _LocalDramResidency] = {}
        self.operations = 0
        self.completions: queue.SimpleQueue[
            tuple[tuple[BlockKey, ...], tuple[_LocalDramResidency, ...], bool]
        ] = queue.SimpleQueue()


class _LocalDramPools:
    """Own pool slots and claims independently of pending operations."""

    def __init__(
        self,
        kvcr: "_KVCRCore",
        region: LocalDramOptions,
        *,
        resume_capacity_waiters: Callable[[], None],
    ) -> None:
        if [pool[0] for pool in region.pools] != [
            pool[0] for pool in kvcr.pool_layouts
        ]:
            raise ValueError("local DRAM pools must match pool_layouts")
        if not region.backend:
            raise ValueError("local DRAM NIXL backend must be non-empty")

        self._kvcr = kvcr
        self._source_state = _SourceState()
        self._draining_source_completions = False
        self._regions: dict[str, tuple[int, int, int]] = {}
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
            self._regions[pool_name] = (address, length, slot_size)
            self._free_slots[pool_name] = deque(range(slot_count))
        ranges = sorted(
            (address, address + length) for address, length, _ in self._regions.values()
        )
        if any(
            left_end > right_start
            for (_, left_end), (right_start, _) in zip(ranges, ranges[1:])
        ):
            raise ValueError("local DRAM pools must not overlap")
        self._evictable = _EvictionQueue()
        self._evictable_slots: Counter[str] = Counter()
        self._unscored: set[BlockKey] = set()
        self._capacity_eviction_key: BlockKey | None = None
        self._public_claims: dict[
            ReleaseHandle, tuple[BlockKey, _LocalDramResidency]
        ] = {}
        self._next_release_handle = 1
        # A no-op until something attaches: the tiers publish residency
        # changes unconditionally, and only recovery cares to hear them.
        self._residency_observer: Callable[[BlockKey, "_BlockRecord"], None] = (
            lambda key, record: None
        )
        # Final claim release must wake waiters before returning to the caller.
        self._resume_capacity_waiters = resume_capacity_waiters

    @property
    def memory_regions(self) -> tuple[RegionDescriptor, ...]:
        return tuple(
            RegionDescriptor(addr=address, count=length // size, size=size, label=name)
            for name, (address, length, size) in self._regions.items()
        )

    @property
    def source_operations(self) -> int:
        state = self._source_state
        with state.lock:
            return state.operations

    def _publish_source(
        self,
        key: BlockKey,
        residency: _LocalDramResidency,
        refs: tuple[_TransferRef, ...] | None = None,
    ) -> None:
        if self._kvcr.framework_control is None or residency.pending_labels:
            return
        if refs is None:
            refs = (
                self.source_refs(residency, range(len(residency.slots)))
                if residency.refs
                else tuple(self._descriptors(residency.slots))
            )
        state = self._source_state
        with state.lock:
            if (
                residency.state is _LocalDramState.READY
                and not residency.retire_on_release
            ):
                residency.refs = refs
                state.ready[key] = residency

    def acquire_readers(
        self,
        keys: tuple[BlockKey, ...],
        destinations: tuple[tuple[_TransferRef, ...], ...],
        key_layout: tuple[str, ...],
    ) -> tuple[tuple[_LocalDramResidency, tuple[_TransferRef, ...]], ...] | None:
        """Claim every requested piece; None retries contention, () uses the owner."""
        state = self._source_state
        if not state.lock.acquire(blocking=False):
            return None
        try:
            claims = []
            for key, destination in zip(keys, destinations):
                residency = state.ready.get(key)
                if residency is None:
                    return ()
                available = cast(tuple[_TransferRef, ...], residency.refs)
                if key_layout and key_layout != residency.key_layout:
                    return ()
                requested = [ref.label for ref in destination]
                if residency.layout == tuple(requested):
                    refs = available
                else:
                    indices = _layout_indices(
                        residency.layout,
                        requested,
                        indices_by_label=residency.indices_by_label,
                    )
                    if indices is None:
                        return ()
                    refs = tuple(available[index] for index in indices)
                claims.append((residency, refs))
            result = tuple(claims)
            for key, (residency, _) in zip(keys, result):
                residency.source_readers += 1
                state.active[key] = residency
            state.operations += 1
            return result
        finally:
            state.lock.release()

    def release_readers(
        self,
        keys: tuple[BlockKey, ...],
        sources: tuple[_LocalDramResidency, ...],
        *,
        success: bool,
    ) -> None:
        """Queue reader release; retain eviction protection until access accounting."""
        state = self._source_state
        with state.lock:
            state.operations -= 1
            state.completions.put((keys, sources, success))

    def _drain_source_completions(self) -> None:
        if self._draining_source_completions:
            return
        self._draining_source_completions = True
        state = self._source_state
        completions = state.completions
        freed = False
        try:
            while True:
                try:
                    keys, sources, success = completions.get_nowait()
                except queue.Empty:
                    break
                if success:
                    self._kvcr._record_access(keys)
                with state.lock:
                    for key, residency in zip(keys, sources):
                        if residency.source_readers <= 0:
                            raise RuntimeError(
                                "local DRAM source reader ownership lost"
                            )
                        residency.source_readers -= 1
                        if not residency.source_readers:
                            state.active.pop(key)
                for key, residency in zip(keys, sources):
                    record = self._kvcr._block_record_map.get(key)
                    if (
                        record is not None
                        and record.local_dram is residency
                        and not residency.claim_count
                    ):
                        freed |= self._after_last_claim(key, residency)
        finally:
            self._draining_source_completions = False
        if freed and not self._kvcr._closed:
            self._resume_capacity_waiters()

    def _reclaimable_slots(self) -> dict[str, int]:
        slots = {
            name: len(free) + self._evictable_slots[name]
            for name, free in self._free_slots.items()
        }
        state = self._source_state
        with state.lock:
            for key, residency in state.active.items():
                if self._evictable.score(key) is not None:
                    for pool, count in residency.pool_counts:
                        slots[pool] -= count
        return slots

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
        occupied = {pool_name: set() for pool_name in self._regions}
        for key, record in records.items():
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
                pool = self._regions.get(pool_name)
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
            if tuple(declaration) != residency.layout:
                raise ValueError("invalid local DRAM recovery layout")
            # Byte counts depend on the receiving core's pool geometry.
            residency._cached_size = None
            residency.refs = {}
            residency.prepare_label_index()
            self._publish_source(key, residency)
        self._free_slots = {
            pool_name: deque(
                slot
                for slot in range(length // slot_size)
                if slot not in occupied[pool_name]
            )
            for pool_name, (_, length, slot_size) in self._regions.items()
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
            length // slot_size for _, length, slot_size in self._regions.values()
        )
        free_slots = sum(map(len, self._free_slots.values()))
        return {
            "local_g2_total_slots": total_slots,
            "local_g2_free_slots": free_slots,
            "local_g2_allocated_slots": total_slots - free_slots,
            "local_g2_evictable_slots": sum(self._reclaimable_slots().values())
            - free_slots,
        }

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
        self._update_capacity_pressure()
        return sources

    def source_refs(
        self, residency: _LocalDramResidency, indices: Collection[int]
    ) -> tuple[_TransferRef, ...]:
        cached = residency.refs
        if isinstance(cached, tuple):
            return tuple(cached[index] for index in indices)
        refs = []
        for index in indices:
            ref = cached.get(index)
            if ref is None:
                ref = cached[index] = self._descriptor(*residency.slots[index])
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
        state = self._source_state
        for key in dict.fromkeys(keys):
            record = self._kvcr._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if (
                residency is None
                or residency.state is not _LocalDramState.READY
                or residency.claim_count <= 0
            ):
                raise RuntimeError(f"local DRAM source cannot retire {key!r}")
            with state.lock:
                residency.retire_on_release = True
                state.ready.pop(key, None)

    def abandon_capacity_eviction(self, key: BlockKey) -> None:
        """Stop blocking local admission on an eviction that will not land.

        Waiters queue behind the slot a MOVE_TO eviction is about to free. When the
        move is abandoned, that reservation has to be dropped or every later admission
        is refused for the life of the process.
        """
        if self._capacity_eviction_key == key:
            self._capacity_eviction_key = None

    def close(self) -> None:
        # Drain pending source cleanup before dropping public claims.
        self._drain_source_completions()
        self._public_claims.clear()

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
            self._after_last_claim(key, residency)

    def _after_last_claim(self, key: BlockKey, residency: _LocalDramResidency) -> bool:
        state = self._source_state
        record = self._kvcr._block_record_map[key]
        retiring = (
            residency.state is not _LocalDramState.READY or residency.retire_on_release
        )
        with state.lock:
            if retiring:
                state.ready.pop(key, None)
            if residency.source_readers:
                return False
            if retiring:
                # Withdraw the source before its slots can be reused.
                record.local_dram = None
        if not retiring:
            self._make_evictable(key)
            return False
        if residency.state is _LocalDramState.READY:
            if self._kvcr._key_history_enabled:
                self._kvcr._log_key_history(
                    "remove", (key,), CacheTier.LOCAL_G2, "retire_on_release"
                )
            self._residency_observer(key, record)
            self._free(residency.slots)
            self.abandon_capacity_eviction(key)
            self._kvcr._on_remove(
                self._kvcr._block_meta(key, record, self._size_bytes(residency))
            )
            if not residency.pending_labels:
                self._kvcr._publish_inventory((key,), CacheTier.LOCAL_G2, removed=True)
        else:
            self._free(residency.slots)
        self._kvcr._prune_block_record(key)
        if not self._kvcr._closed and not self._draining_source_completions:
            self._resume_capacity_waiters()
        return True

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
        state = self._source_state
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
                if deficient.isdisjoint(pool for pool, _ in residency.pool_counts):
                    continue
                with state.lock:
                    if residency.source_readers:
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
                for pool, count in residency.pool_counts:
                    freed[pool] += count
                deficient = short()

        with state.lock:
            # Policy runs unlocked; a source can claim a selected victim meanwhile.
            if any(residency.source_readers for _, _, residency, _ in victims):
                # Accept transient admission failure to avoid another policy scan.
                return None, [], False
            for key, _, _, _ in victims:
                state.ready.pop(key, None)
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
            or residency.retire_on_release
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
            evictable_slots = self._evictable_slots
            for pool, count in residency.pool_counts:
                evictable_slots[pool] += count

    def _remove_evictable(self, key: BlockKey, residency: _LocalDramResidency) -> None:
        self._unscored.discard(key)
        if self._evictable.remove(key):
            evictable_slots = self._evictable_slots
            for pool, count in residency.pool_counts:
                evictable_slots[pool] -= count

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
                self._regions[pool][2] * count for pool, count in residency.pool_counts
            )
        return residency._cached_size

    def _update_capacity_pressure(self) -> None:
        if self._kvcr._capacity_needed_callback is None:
            return
        self._kvcr._update_capacity_pressure(self._reclaimable_slots())
