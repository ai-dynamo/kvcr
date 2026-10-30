# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Carry move intent with ordinary fetch; remove source copies independently."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .runtime.progress import _OpId, _TransferRef
from .types import BlockKey, OpEntryResult, OpHandle

if TYPE_CHECKING:
    from .backends.remote_ops import _SourceWriteOp, _StartWrite
    from .runtime.core import _KVCRCore


@dataclass
class _PendingMove:
    source: str
    layout: tuple[str, ...]
    keys: frozenset[BlockKey]
    # Cache hits can appear during fill setup or completion; send intent once.
    requested_keys: set[BlockKey] = field(default_factory=set)


class _MoveHandler:
    """Carry removal intent through fetch without delaying destination completion.
    The source invalidates after copying, or when the destination has the data.
    """

    def __init__(self, core: "_KVCRCore") -> None:
        self._core = core
        # Public destination operations can launch several internal fills.
        self._pending: dict[OpHandle, _PendingMove] = {}
        self._request_ids: set[str] = set()
        # Selected move keys and complete copies already held by the target.
        self._target_transfers: dict[
            OpHandle, tuple[frozenset[BlockKey], tuple[BlockKey, ...]]
        ] = {}
        # Source native operations retain intent until their own cleanup.
        self._source_transfers: dict[_OpId, tuple[BlockKey, ...]] = {}

    def record_hint(self, request_id: str | None, *, move: bool) -> None:
        if request_id is not None:
            if move:
                self._request_ids.add(request_id)
            else:
                self._request_ids.discard(request_id)

    def discard_hint(self, request_id: str) -> None:
        self._request_ids.discard(request_id)

    def begin(
        self,
        handle: OpHandle,
        keys: tuple[BlockKey, ...],
        request_id: str | None,
        layout: tuple[str, ...],
        *,
        force: bool = False,
    ) -> bool:
        if not keys:
            return True
        remote = self._core._remote_fw_dram
        hint = remote._request_hints.get(request_id)
        adapter = remote._key_adapter
        if hint is None or not hint.source or hint.failed or adapter is None:
            return False
        selected = frozenset(
            key for key in keys if adapter.decode(key) in hint.block_hashes
        )
        # Even a destination cache hit needs advice naming the source to remove.
        if force and len(selected) != len(keys):
            return False
        if selected:
            self._pending[handle] = _PendingMove(hint.source, layout, selected)
        return True

    def start_fill(
        self,
        handle: OpHandle,
        fill_handle: OpHandle,
        blocks: Mapping[BlockKey, list[_TransferRef]],
        request_id: str | None,
        deadline: float,
    ) -> bool:
        move = self._pending[handle]
        selected = move.keys.intersection(blocks)
        fetch = self._core._local_dram._pending_residency_ops[("fetch", handle)]
        # Successful local results already hold complete copies with fetch claims.
        held = tuple(
            key
            for key, result in fetch.results.items()
            if result.success and key in move.keys and key not in move.requested_keys
        )
        move.requested_keys.update(selected)
        move.requested_keys.update(held)
        if selected or held:
            self._target_transfers[fill_handle] = selected.union(held), held
        started = False
        try:
            started = self._core._remote_fw_dram.fetch(
                blocks, request_id, deadline, op_handle=fill_handle
            )
            return started
        finally:
            if not started:
                self._target_transfers.pop(fill_handle, None)
                move.requested_keys.difference_update(held)

    def transfer_request(self, request: "_StartWrite") -> "_StartWrite":
        selected, held = self._target_transfers[request.op_handle]
        # Empty descriptors mark complete destination cache hits on the wire;
        # native writes and completion indices still cover only transferred keys.
        request.keys += held
        request.dst_descriptors += ((),) * len(held)
        request.move_keys = tuple(key for key in request.keys if key in selected)
        return request

    @staticmethod
    def validate_request(request: "_StartWrite") -> bool:
        keys, selected = set(request.keys), set(request.move_keys)
        if (
            not selected
            or len(selected) != len(request.move_keys)
            or not selected.issubset(keys)
            or not request.key_layout
        ):
            return False
        if not request.dst_descriptors:
            return selected == keys
        return len(request.keys) == len(request.dst_descriptors) and all(
            descriptors or key in selected
            for key, descriptors in zip(request.keys, request.dst_descriptors)
        )

    def complete(
        self, handle: OpHandle, entries: Mapping[BlockKey, OpEntryResult]
    ) -> None:
        """Publish ordinary fetch results and queue any unsent cache-hit intent."""
        from .backends.remote_ops import _StartWrite

        move = self._pending.pop(handle)
        held = tuple(
            key
            for key, result in entries.items()
            if result.success and key in move.keys and key not in move.requested_keys
        )
        if held:
            # Cache hits still need a source command, but no completion reply.
            self._core._progress.submit(
                _StartWrite(
                    op_handle=handle,
                    remaining_timeout_ms=self._core.config.operation_timeout_ms,
                    keys=held,
                    dst_descriptors=(),
                    key_layout=move.layout,
                    move_keys=held,
                    source_control_endpoint=move.source,
                )
            )
        self._core._complete(handle, entries)

    def source_finished(self, transfer: "_SourceWriteOp") -> None:
        """Invalidate before releasing source claims; fast G2 uses its reader queue."""
        move_keys = self._source_transfers.pop(transfer.op_id)
        if transfer.success:
            selected = set(move_keys)
            # The destination holds the other pieces ready before requesting a
            # partial move, so success permits removal of the entire source key.
            self._core._invalidation_handler.submit(
                tuple(key for key in transfer.source_keys if key in selected)
            )

    def source_cached(self, request: "_StartWrite") -> None:
        """Remove eligible source copies already held at the destination."""
        keys = []
        for index, key in enumerate(request.keys):
            if request.dst_descriptors and request.dst_descriptors[index]:
                continue
            record = self._core._block_record_map.get(key)
            if (
                record is not None
                and record.local_dram is not None
                and record.local_dram.key_layout != request.key_layout
            ):
                continue
            keys.append(key)
        self._core._invalidation_handler.submit(keys)

    def close(self) -> None:
        self._pending.clear()
        self._request_ids.clear()
        self._target_transfers.clear()
        self._source_transfers.clear()
