# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The hint handlers: request-scoped advice stored at the destination, and the
independent ``kv.prefetch`` / ``kv.move`` actions, which the source resolves
and asks the named destination to fetch; the destination transfers and reports
the outcome."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import msgspec

from .hint_types import _Hint, _KVFetchHint, _KVMoveHint, _KVPrefetchHint
from .parser import reject_hint
from .results import HintResult, HintStatus

if TYPE_CHECKING:
    from ..backends.remote_ops import _KeysToFetch
    from ..runtime.core import _KVCRCore
    from ..types import BlockKey, OpEntryResult, OpHandle
    from .dispatcher import HintDispatcher


class _HintHandler(ABC):
    """Base for handlers of one or more action types.

    ``submit`` runs on the caller's thread without the state lock and reports
    admission once. Handlers acquire the lock as needed for cache operations.
    """

    @abstractmethod
    def submit(
        self,
        hint: _Hint,
        request_id: str | None,
        message_id: str | None,
    ) -> None: ...

    def close(self) -> None:
        """Release handler state after the progress loop is quiescent."""


class _FetchHintHandler(_HintHandler):
    """Store ``kv.fetch`` / ``kv.evict_on_fetch`` advice at the destination.

    Inline only: no state lock, no block records, and no transfer beyond the
    store's own eager control connect.
    """

    def __init__(self, core: _KVCRCore, dispatcher: HintDispatcher) -> None:
        self._core = core  # backend access only
        self._dispatcher = dispatcher  # results

    def submit(
        self,
        hint: _Hint,
        request_id: str | None,
        message_id: str | None,
    ) -> None:
        hint = cast(_KVFetchHint, hint)
        # One store call per hint; the store's own rules apply (a later hint
        # replaces the request's hashes, a different source fails the request).
        self._core._remote_fw_dram.submit_hint(
            hint.source_endpoint,
            hint.block_hashes,
            request_id,
            evict_on_finish=hint.ACTION_TYPE == "kv.evict_on_fetch",
        )
        self._dispatcher.report(
            HintResult(
                action_id=hint.action_id,
                action_type=hint.ACTION_TYPE,
                status=HintStatus.ACCEPTED,
                request_id=request_id,
                message_id=message_id,
            )
        )


@dataclass
class _PendingHint:
    receipt: HintResult
    remaining: int


class _IndependentHintHandler(_HintHandler):
    """Owner-thread lifecycle of ``kv.prefetch`` / ``kv.move`` on both peers;
    the dispatcher's handler for both action types."""

    def __init__(self, core: _KVCRCore) -> None:
        self._core = core
        self._received: deque[_KeysToFetch] = deque()
        # Layout fetches share one receipt and completion count per hint.
        self._pending_operations: dict[
            OpHandle, tuple[_PendingHint, dict[BlockKey, int]]
        ] = {}

    # Source: the hint's recipient.

    def submit(
        self,
        hint: _Hint,
        request_id: str | None,
        message_id: str | None,
    ) -> None:
        """Resolve the hashes here and ask the destination to fetch the blocks.

        Admission reports the hashes that did not resolve; the destination
        reports one combined outcome for all forwarded groups under the same
        action and message IDs.
        """
        from ..backends.remote_ops import _FetchGroup, _KeysToFetch

        hint = cast(_KVPrefetchHint | _KVMoveHint, hint)
        core = self._core
        if core._local_dram is None or core._key_adapter is None:
            reason = (
                "source_has_no_managed_dram"
                if core._local_dram is None
                else "key_adapter_unavailable"
            )
        elif hint.destination_endpoint == getattr(
            core.framework_control, "endpoint", None
        ):
            reason = "self_destination"
        else:
            reason = None
        if reason is not None:
            core._hints_dispatcher.report(
                reject_hint(
                    hint.action_id, hint.ACTION_TYPE, reason, message_id, request_id
                )
            )
            return
        with core._state_lock:
            groups, failures = core.resolve_hashes(hint.block_hashes)
        # Queue before reporting: a failed submission must raise before any
        # report. One-way: a send failure is logged, not reported.
        if groups:
            core._progress.submit(
                _KeysToFetch(
                    action_id=hint.action_id,
                    message_id=message_id,
                    move=hint.ACTION_TYPE == "kv.move",
                    groups=tuple(
                        _FetchGroup(
                            keys=tuple(hashes_by_key),
                            block_hashes=tuple(hashes_by_key.values()),
                            key_layout=layout,
                            sizes=sizes,
                        )
                        for (layout, sizes), hashes_by_key in groups.items()
                    ),
                    source_control_endpoint=hint.destination_endpoint,
                )
            )
        core._hints_dispatcher.report(
            HintResult(
                action_id=hint.action_id,
                action_type=hint.ACTION_TYPE,
                status=HintStatus.ACCEPTED if groups else HintStatus.FAILED,
                request_id=request_id,
                message_id=message_id,
                failures=failures or None,
            )
        )

    # Destination: the request's recipient.

    def receive(self, request: _KeysToFetch) -> None:
        """Queue a source's request; ``poll`` starts it outside the state lock."""
        self._received.append(request)

    def poll(self) -> None:
        while self._received:
            self.requested(self._received.popleft())

    def requested(self, request: _KeysToFetch) -> None:
        """Fetch the requested blocks through the existing copy/move path."""
        core = self._core
        failures: dict[int, str] = {}
        receipt = HintResult(
            action_id=request.action_id,
            action_type="kv.move" if request.move else "kv.prefetch",
            status=HintStatus.ACCEPTED,
            message_id=request.message_id,
            failures=failures,
        )
        source = request.sender_control_endpoint
        keys = tuple(key for group in request.groups for key in group.keys)
        hashes = tuple(h for group in request.groups for h in group.block_hashes)
        if core._local_dram is None:
            reason = "destination_has_no_managed_dram"
        elif not source or not request.sender_incarnation:
            reason = "invalid_request"
        elif (
            request.sender_incarnation == core._remote_fw_dram._dangling_ops.incarnation
        ):
            # An alias of this node's own endpoint: a move must not drop its
            # own copies through a self-addressed invalidation.
            reason = "self_source"
        elif (
            not request.groups
            or any(
                not group.keys or len(group.keys) != len(group.block_hashes)
                for group in request.groups
            )
            or len(set(keys)) != len(keys)
            or len(set(hashes)) != len(hashes)
        ):
            reason = "invalid_request"
        else:
            reason = None
        if reason is not None:
            core._hints_dispatcher.report(
                msgspec.structs.replace(
                    receipt,
                    status=HintStatus.FAILED,
                    reason=reason,
                    failures=dict.fromkeys(hashes, reason) or None,
                )
            )
            return

        pending = _PendingHint(receipt, len(request.groups))
        for group in request.groups:
            operation = pending, dict(zip(group.keys, group.block_hashes))
            if self._layout_sizes(group.key_layout) != group.sizes:
                self._complete(*operation, {}, reason="incompatible_layout")
                continue
            try:
                handle = core.fetch(
                    group.keys,
                    expected_layout=list(group.key_layout),
                    source_endpoint=source,
                    move=request.move,
                )
            except Exception as error:
                # Fatal progress failures retain their ordinary API behavior.
                core._progress.raise_if_failed()
                # Fail this group without escaping the owner's poll. Layout
                # conflicts are expected; anything else warns once.
                if not isinstance(error, ValueError):
                    core._hints_dispatcher.warn_handler_error(
                        receipt.action_type, "admission"
                    )
                self._complete(*operation, {})
            else:
                # Synchronous hits are queued until polling, so registration
                # after fetch returns is safe.
                self._pending_operations[handle] = operation

    def _layout_sizes(self, layout: tuple[str, ...]) -> tuple[int, ...] | None:
        """Return piece sizes for a layout this destination can store."""
        core = self._core
        try:
            core._validate_expected_layout(list(layout))
        except ValueError:
            return None
        return tuple(core._block_sizes[label.partition(":")[0]] for label in layout)

    def finish(self, completed):
        """Consume independent completions and return framework operations."""
        framework = []
        for handle, entries in completed:
            operation = self._pending_operations.pop(handle, None)
            if operation is None:
                framework.append((handle, entries))
                continue
            self._complete(*operation, entries)
        return framework

    def _complete(
        self,
        pending: _PendingHint,
        selected: dict[BlockKey, int],
        results: dict[BlockKey, OpEntryResult],
        *,
        reason: str = "operation_failed",
    ) -> None:
        """Release this group's claims; report once all groups have settled."""
        handles = tuple(
            result.release_handle
            for result in results.values()
            if result.release_handle is not None
        )
        if handles:
            self._core.release(handles)
        receipt = pending.receipt
        failures = receipt.failures
        assert failures is not None
        for key, block_hash in selected.items():
            result = results.get(key)
            if result is None or not result.success:
                failures[block_hash] = reason
        pending.remaining -= 1
        if pending.remaining:
            return
        self._core._hints_dispatcher.report(
            msgspec.structs.replace(
                receipt,
                status=HintStatus.FAILED if failures else HintStatus.COMPLETED,
                failures=failures or None,
            )
        )

    def close(self) -> None:
        self._received.clear()
        self._pending_operations.clear()
