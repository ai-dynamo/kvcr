# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Drop moved source G2 copies after readers and transfers drain."""

from collections.abc import Collection
from typing import TYPE_CHECKING

from ..types import BlockKey, OpEntryResult
from .progress import _FAILED_RESULT

if TYPE_CHECKING:
    from .core import _BlockRecord, _KVCRCore


class _InvalidationHandler:
    """Remove sources under the owner lock after existing readers drain.
    Submission rejects new reads and deposits; a failed move reopens its keys.
    """

    def __init__(self, core: "_KVCRCore") -> None:
        self._core = core
        # The layout the destination holds; None means unrestricted removal.
        self._pending: dict[
            BlockKey, tuple["_BlockRecord", tuple[str, ...] | None]
        ] = {}
        # Recheck only on submission or release, not every idle poll.
        self._dirty: dict[BlockKey, None] = {}

    def has_pending(self) -> bool:
        return bool(self._pending)

    def rejected_entries(
        self, keys: Collection[BlockKey]
    ) -> dict[BlockKey, OpEntryResult]:
        """Check new requests once; admitted operations may resume during removal."""
        if not self._pending:
            return {}
        return {key: _FAILED_RESULT for key in keys if key in self._pending}

    def close(self) -> None:
        """Abandon pending removals after progress stops."""
        for record, _ in self._pending.values():
            record.pending_invalidation = False
        self._core._outstanding_operations -= len(self._pending)
        self._pending.clear()
        self._dirty.clear()

    def submit(
        self,
        keys: Collection[BlockKey],
        *,
        key_layout: tuple[str, ...] | None = None,
    ) -> tuple[BlockKey, ...]:
        """Queue removal of G2 copies held in key_layout; return the keys marked.

        Keys protected by no_evict or held in another complete layout are left
        alone. Keys without a G2 copy are skipped too unless an admitted deposit
        or fetch may still create one: framework and G3 copies are never touched.
        """
        core = self._core
        local_dram = core._local_dram
        if local_dram is None:
            return ()
        state = local_dram._pools._source_state
        marked = []
        for key in keys:
            record = core._block_record_map.get(key)
            if record is None or key in self._pending:
                continue
            local = record.local_dram
            if local is None:
                # A batch dependency can outlive this key's terminal result.
                if not any(
                    (op := local_dram._pending_residency_ops.get(op_id)) is not None
                    and key not in op.results
                    for op_id in record.active_op_ids
                ):
                    continue
            elif local.retention_claims or (
                key_layout is not None and local.key_layout != key_layout
            ):
                continue
            self._pending[key] = record, key_layout
            # Close admission atomically with progress acquiring readers.
            # No slot reclamation or callbacks run under this lock.
            with state.lock:
                record.pending_invalidation = True
                state.index.pop(key, None)
            core._outstanding_operations += 1
            self._dirty[key] = None
            marked.append(key)
        return tuple(marked)

    def reopen(self, keys: Collection[BlockKey]) -> None:
        """Cancel a failed move's own pending keys; their copies stay readable."""
        for key in keys:
            pending = self._pending.get(key)
            if pending is not None:
                self._settle(key, pending[0], drop=False)

    def _settle(self, key: BlockKey, record: "_BlockRecord", *, drop: bool) -> bool:
        """Clear the pending mark, then drop or republish the copy; True if
        capacity changed."""
        core = self._core
        pools = core._local_dram._pools
        # Clear intent before callbacks can redeposit this key or poll again.
        del self._pending[key]
        self._dirty.pop(key, None)
        record.pending_invalidation = False
        core._outstanding_operations -= 1
        local = record.local_dram
        if local is None:
            core._prune_block_record(key)
            return False
        if drop:
            pools._drop(key)
            return True
        pools._publish_source(key, local)
        if local.retention_claims or local.source_readers:
            return False
        pools._make_evictable(key)
        return True

    def notify(self, key: BlockKey) -> None:
        if key in self._pending:
            self._dirty[key] = None

    @staticmethod
    def _is_ready(record: "_BlockRecord") -> bool:
        # Submission withdrew the source under the reader lock; its reader count
        # can now only decrease during owner-side completion processing.
        local = record.local_dram
        return not (
            record.in_flight_ops
            or local is not None
            and (local.claim_count > local.retention_claims or local.source_readers)
        )

    def drain(self) -> None:
        pools = self._core._local_dram._pools
        dirty, self._dirty = self._dirty, {}
        capacity_changed = False
        for key in dirty:
            # A preceding callback may have removed this key.
            pending = self._pending.get(key)
            if pending is None:
                continue
            record, layout = pending
            if not self._is_ready(record):
                continue
            local = record.local_dram
            # An earlier producer can grant retention or create a layout the
            # destination never held; keep that copy and reopen admission.
            keep = local is not None and (
                local.retention_claims
                or (layout is not None and local.key_layout != layout)
            )
            capacity_changed |= self._settle(key, record, drop=not keep)
        if capacity_changed:
            pools._resume_capacity_waiters()
            pools._update_capacity_pressure()
