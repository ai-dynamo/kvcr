# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hint dispatch infrastructure.

``HintDispatcher`` owns the hint type table and reports every hint's
result; core holds one dispatcher and delegates to it. ``_HINT_TYPES`` binds
the supported hint types to their payload parsers and handlers.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import msgspec

from .hint_handler import _FetchHintHandler, _HintHandler
from .hint_types import (
    HintType,
    _Hint,
    _KVEvictOnFetchHint,
    _KVFetchHint,
    _KVMoveHint,
    _KVPrefetchHint,
)
from .parser import _parse_payload, hint_message_id, parse_hints
from .results import HintResult, HintStatus

if TYPE_CHECKING:
    from ..runtime.core import _KVCRCore

KV_FETCH = HintType(
    _KVFetchHint.ACTION_TYPE,
    "1.0",
    partial(_parse_payload, peer="source", hint_factory=_KVFetchHint),
    _FetchHintHandler,
)
KV_EVICT_ON_FETCH = HintType(
    _KVEvictOnFetchHint.ACTION_TYPE,
    "1.0",
    partial(_parse_payload, peer="source", hint_factory=_KVEvictOnFetchHint),
    _FetchHintHandler,
)
# Core builds the independent hint handler before the dispatcher; it handles
# both independent actions.
KV_PREFETCH = HintType(
    _KVPrefetchHint.ACTION_TYPE,
    "1.0",
    partial(_parse_payload, peer="destination", hint_factory=_KVPrefetchHint),
    lambda core, dispatcher: core._independent_hint_handler,
)
KV_MOVE = HintType(
    _KVMoveHint.ACTION_TYPE,
    "1.0",
    partial(_parse_payload, peer="destination", hint_factory=_KVMoveHint),
    lambda core, dispatcher: core._independent_hint_handler,
)

# The supported hint types. Every dispatcher builds one handler per entry.
_HINT_TYPES: tuple[HintType, ...] = (KV_FETCH, KV_EVICT_ON_FETCH, KV_PREFETCH, KV_MOVE)

logger = logging.getLogger(__name__)
# Handlers run on hot paths; warn once per (phase, action type) about errors.
_LOGGED_HANDLER_ERRORS: set[str] = set()


def _warn_handler_error(action_type: str, phase: str) -> None:
    """Log the active exception once per ``(phase, action_type)``."""
    issue = f"{phase}:{action_type}"
    if issue in _LOGGED_HANDLER_ERRORS:
        return
    _LOGGED_HANDLER_ERRORS.add(issue)
    logger.warning(
        "KV hint handler failed during %s; action_type=%s",
        phase,
        action_type,
        exc_info=True,
    )


class HintDispatcher:
    """Routes parsed hints to their handlers and reports their results.

    Core owns one dispatcher and calls ``submit`` from the caller's thread and
    ``close`` under its state lock once the progress loop is quiescent, with
    the other main-thread resources. Which hint types exist is decided here,
    by ``_HINT_TYPES``, not by core.
    """

    def __init__(self, core: _KVCRCore) -> None:
        self._core = core
        self._types = {hint_type.action_type: hint_type for hint_type in _HINT_TYPES}
        self._handlers: dict[str, _HintHandler] = {
            action_type: hint_type.handler_factory(core, self)
            for action_type, hint_type in self._types.items()
        }
        self._closed = False

    # Caller thread, no state lock.

    def submit(
        self,
        envelope: msgspec.Struct,
        request_id: str | None,
    ) -> None:
        """Parse an envelope, report its rejections, dispatch its hints."""
        self._core._progress.raise_if_failed()
        hints, rejected = parse_hints(envelope, self._types)
        for result in rejected:
            self.report(msgspec.structs.replace(result, request_id=request_id))
        message_id = hint_message_id(envelope)
        for hint in hints:
            handler = self._handlers[type(hint).ACTION_TYPE]
            self._submit_to_handler(handler, hint, request_id, message_id)

    def _submit_to_handler(
        self,
        handler: _HintHandler,
        hint: _Hint,
        request_id: str | None,
        message_id: str | None,
    ) -> None:
        # Contract: a handler reports admission exactly once, or raises before
        # doing so. ACCEPTED means admission, not asynchronous completion.
        # A raise means REJECTED, unless it is the fatal progress-thread
        # failure, which propagates to the caller as it does without hints.
        try:
            handler.submit(hint, request_id, message_id)
        except Exception:
            self._core._progress.raise_if_failed()
            _warn_handler_error(hint.ACTION_TYPE, "submit")
            self.report(
                HintResult(
                    action_id=hint.action_id,
                    action_type=hint.ACTION_TYPE,
                    status=HintStatus.REJECTED,
                    reason="handler_error",
                    request_id=request_id,
                    message_id=message_id,
                )
            )

    # Handlers warn through the dispatcher's once-per-issue logger.
    warn_handler_error = staticmethod(_warn_handler_error)

    def report(self, result: HintResult) -> None:
        """Record a hint's result; called by handlers and by the dispatcher.

        Results are logged at debug level only; rejections and handler errors
        already warn once per issue. No metric is emitted: integrations
        forward KVCR stats and may reject names they do not declare.
        """
        logger.debug(
            "KV hint result; action_id=%s action_type=%s status=%s reason=%s "
            "failures=%s",
            result.action_id,
            result.action_type,
            result.status.value,
            result.reason,
            result.failures,
        )

    # Main thread, under the state lock.

    def close(self) -> None:
        """Close handlers; idempotent."""
        if self._closed:
            return
        self._closed = True
        closed: set[int] = set()
        for action_type, handler in self._handlers.items():
            if id(handler) in closed:
                continue
            closed.add(id(handler))
            try:
                handler.close()
            except Exception:
                _warn_handler_error(action_type, "close")
