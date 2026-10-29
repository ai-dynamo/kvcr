# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Local DRAM requests, copy operations, and completion handling."""

import logging
from collections import deque
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field, replace
from itertools import islice
from typing import TYPE_CHECKING, Literal, cast

from ..config import LocalDramOptions
from ..runtime.progress import (
    _DROPPED_RESULT,
    _FAILED_RESULT,
    _SUCCESS_RESULT,
    _KVCRProgress,
    _Op,
    _OpId,
    _ProgressOp,
    _TransferRef,
)
from ..types import (
    BlockKey,
    CacheTier,
    MemoryRef,
    OpEntryResult,
    OpHandle,
    PlacementAction,
    TransferError,
)
from .local_dram_pools import (
    _layout_indices,
    _LocalDramPools,
    _LocalDramResidency,
    _LocalDramState,
)

if TYPE_CHECKING:
    from ..runtime.core import _KVCRCore

logger = logging.getLogger(__name__)

_Clock = Callable[[], float]


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
    """Coordinate requests while pool claims protect their memory."""

    def __init__(
        self,
        kvcr: "_KVCRCore",
        region: LocalDramOptions,
    ) -> None:
        self._kvcr = kvcr
        self._pools = _LocalDramPools(
            kvcr, region, resume_capacity_waiters=self._resume_capacity_waiters
        )
        self._backend = region.backend
        self._pending_residency_ops: dict[_OpId, _PendingResidencyOp] = {}
        self._pending_deliver_ops: dict[_OpId, _PendingDeliverOp] = {}
        self._capacity_waiters: deque[_CapacityWaiter] = deque()
        self._resuming_capacity_waiters = False
        self._abandon_grace = (
            kvcr.config.abandon_timeout_ms - kvcr.config.operation_timeout_ms
        ) / 1000

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
                    self._pools._regions[label.partition(":")[0]][2]
                    for label in local_layout
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
                locations, evicted_keys, eviction_pending = self._pools._allocate_slots(
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
                pending = residency.pending_labels or ()
                selected_indices = tuple(
                    index for index in indices if residency.slots[index][0] in pending
                )
                if not selected_indices:
                    op.results[key] = (
                        self._pools._new_public_claim(key, residency)
                        if op.claim_on_ready
                        else _SUCCESS_RESULT
                    )
                    continue
                # Pending pieces imply writer counts (see _LocalDramResidency.settle).
                writers = cast(list[int], residency.writers)
                if all(writers[index] for index in selected_indices):
                    continue  # Attach duplicates; new pieces can still proceed.
                self._pools._remove_evictable(key, residency)
                selected_slots = []
                # Mixed deposits may copy an active slot too; bytes must be identical.
                for ref, index in zip(sources, indices):
                    if residency.slots[index][0] in pending:
                        selected_slots.append(residency.slots[index])
                        src_descriptors.append(ref)
                        writers[index] += 1
            residency.claim_count += 1
            residency.pending_deposits += 1
            op.active_keys.add(key)
            copy_keys.append(key)
            copy_indices.append(selected_indices)
            dst_descriptors.extend(self._pools._descriptors(selected_slots))

        self._pools._update_capacity_pressure()
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
            op.results[key] = self._pools._new_public_claim(key, residency, slots=ready)
        else:
            if key not in op.claimed_keys:
                self._pools._acquire_claim(key, residency)
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
                self._pools._remove_evictable(key, residency)
                residency.remote_indices = tuple(missing)
                for index in missing:
                    residency.writers[index] += 1
                residency.claim_count += 1
                op.remote_fill_keys.add(key)
                return self._pools._descriptors(
                    [residency.slots[index] for index in missing]
                )
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

    def poll_main(self, items: Collection[object]) -> list[object]:
        # Progress-side source claims defer framework callbacks to the caller.
        self._pools._update_capacity_pressure()
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
            # This copy holds a writer count, so writers cannot have been settled.
            writers = cast(list[int], residency.writers)
            for index in indices:
                writers[index] -= 1
                if writers[index] < 0:
                    raise RuntimeError("local DRAM writer ownership lost")
            if writer is not None:
                writer.active_keys.remove(key)
                residency.pending_deposits -= 1
            else:
                residency.remote_indices = None
                residency.remote_cancelled = False
            pending = residency.pending_labels
            if key_success and pending:
                pending_count = len(pending)
                if len(indices) == len(residency.slots):
                    pending.clear()
                else:
                    pending.difference_update(
                        residency.slots[index][0] for index in indices
                    )
                if len(pending) < pending_count:
                    residency.state = _LocalDramState.READY
                    residency.prepare_label_index()
                    record.last_access = now
                    if not was_ready:
                        self._kvcr._on_ingest(
                            self._kvcr._block_meta(
                                key, record, self._pools._size_bytes(residency)
                            ),
                            source,
                        )
                    if not pending:
                        self._pools._residency_observer(key, record)
                        committed.append(key)
            residency.settle()
            if writer is not None:
                if key not in writer.results:
                    writer.results[key] = (
                        self._pools._new_public_claim(key, residency)
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
            self._pools._release_claim(key, residency)
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

        self._pools._update_capacity_pressure()
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
        size_bytes = sum(
            self._pools._regions[label.partition(":")[0]][2] for label in layout
        )
        decision = self._kvcr._policy.decide_ingest(
            self._kvcr._block_meta(key, record, size_bytes),
            source,
            required_local=True,
            framework_hints=op.hints,
        )
        if decision[0] is PlacementAction.DROP:
            return False, False
        locations, evicted, waiting = self._pools._allocate_slots(
            layout, op.keys, op.deadline
        )
        if locations is not None:
            record.local_dram = _LocalDramResidency(
                locations, _LocalDramState.FILLING, key_layout=op.key_layout
            )
        self._pools._update_capacity_pressure()
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
                    self._pools._acquire_claim(key, residency)
                    op.claimed_keys.add(key)
                if indices is None:
                    # A producer may submit the next piece after this delivery.
                    continue
                op.active_keys.add(key)
                copy_keys.append(key)
                src_descriptors.extend(self._pools.source_refs(residency, indices))
                dst_descriptors.extend(op.destinations[key])

        self._pools._update_capacity_pressure()
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
            self._pools._release_claim(key, residency)
            if op is not None:
                op.claimed_keys.remove(key)
                op.active_keys.discard(key)
                op.results[key] = _SUCCESS_RESULT if copy.success else _FAILED_RESULT
        self._pools._update_capacity_pressure()
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
            self._pools.release_sources(claimed)
        self._kvcr._remove_block_dependencies(op)
        self._kvcr._complete(cast(OpHandle, op.op_id[1]), op.results)

    def _finish_deliver_if_ready(self, op: _PendingDeliverOp) -> None:
        if len(op.results) != len(op.keys):
            return
        # Submitted copies retain their claims until native completion.
        self._pools.release_sources(op.claimed_keys - op.active_keys)
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
            self._pools._update_capacity_pressure()
