# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""One physical framework pin and its independent wait range."""

from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

from .payload import Reason, Status

if TYPE_CHECKING:
    from . import _PinTracer

_PIN_RESULTS = {
    "success": (Status.SUCCESS, Reason.NONE),
    "failed": (Status.FAILED, Reason.UNKNOWN),
    "timeout": (Status.TIMEOUT, Reason.DEADLINE),
    "cancelled": (Status.CANCELLED, Reason.CANCELLED),
}


@dataclass
class _PinTrace:
    tracer: "_PinTracer"
    pin_id: int
    requested_blocks: int
    source_op_id: int
    op_handle: int
    request_id: int | None = None
    finished: bool = False

    context_emitted: bool = False
    _wait_range_id: int | None = field(default=None, repr=False)
    _finish_lock: object = field(default_factory=Lock, repr=False)

    def _context(self):
        if self.context_emitted:
            return
        self.context_emitted = True
        known = self.request_id is not None and -(2**63) <= self.request_id < 2**63
        try:
            self.tracer._emit(
                "source.pin.context",
                self.pin_id,
                dict(
                    pin_id=self.pin_id,
                    pin_request_known=known,
                    pin_request_id=self.request_id if known else 0,
                    source_op_id=self.source_op_id,
                    op_handle=self.op_handle,
                    requested_blocks=self.requested_blocks,
                ),
                method=self.tracer.domain.mark,
                category=self.tracer.category,
                context=True,
            )
        except Exception:
            pass

    def _mark(self, name, **fields):
        try:
            self.tracer._emit(
                name,
                self.pin_id,
                fields,
                method=self.tracer.domain.mark,
                category=self.tracer.category,
            )
        except Exception:
            pass  # Annotation errors never alter KVCR results or resources.

    def push(self):
        try:
            self.tracer._emit(
                "source.pin.framework",
                self.pin_id,
                {},
                category=self.tracer.category,
                method=self.tracer.domain.push_range,
            )
            return True
        except Exception:
            return False

    def registered(self, request_id):
        self.request_id = request_id
        self._context()
        self._mark("source.pin.registered")

    def waiter(self, source_op_id, op_handle):
        self._mark("source.pin.waiter", source_op_id=source_op_id, op_handle=op_handle)

    def detached(self, source_op_id, op_handle):
        if self.tracer.level == "medium":
            self._mark(
                "source.pin.detached",
                source_op_id=source_op_id,
                op_handle=op_handle,
            )

    def finish(self, result, *, reason=None, completed_blocks=-1):
        with self._finish_lock:
            if self.finished:
                return
            self.finished = True
        self._context()
        status, default_reason = _PIN_RESULTS.get(
            result, (Status.UNRESOLVED, Reason.UNKNOWN)
        )
        if status is Status.SUCCESS and 0 <= completed_blocks < self.requested_blocks:
            status = Status.FAILED if completed_blocks == 0 else Status.PARTIAL
        if reason is None:
            reason = default_reason
        if status in (Status.PARTIAL, Status.FAILED) and reason is Reason.NONE:
            reason = Reason.UNKNOWN
        self._mark(
            "source.pin.completed",
            status=status,
            reason=reason,
            completed_blocks=completed_blocks,
        )
        if self._wait_range_id is not None:
            try:
                self.tracer.domain.end_range(self._wait_range_id)
            except Exception:
                pass
