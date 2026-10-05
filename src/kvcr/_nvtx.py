# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optional, best-effort NVTX annotations for framework pin lifetimes.

Ranges cover a single synchronous callback. Marks carry an independent pin
identity so shared waits and reused framework request IDs remain distinguishable.
Neither profiler presence nor telemetry configuration controls payload work.
"""

import logging
import os
from dataclasses import dataclass
from enum import IntEnum
from importlib import import_module
from itertools import count
from uuid import uuid4

_LOGGER = logging.getLogger(__name__)
_WARNED: set[str] = set()
_NAMES = (
    "source.pin.framework",
    "source.pin.registered",
    "source.pin.waiter",
    "source.pin.completed",
    "source.pin.detached",
)


class Status(IntEnum):
    PENDING = 1
    SUCCESS = 2
    PARTIAL = 3
    FAILED = 4
    TIMEOUT = 5
    CANCELLED = 6


class Reason(IntEnum):
    UNKNOWN = 0
    NONE = 1
    CALLBACK_ERROR = 2
    DUPLICATE_REQUEST = 3
    INVALID_RESULT = 4
    DEADLINE = 5
    CANCELLED = 6
    SHUTDOWN = 7
    NO_WAITERS = 8


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        _LOGGER.warning(message)


def _load_backend():
    nvtx = import_module("nvtx")
    numpy = import_module("numpy")
    return nvtx.get_domain("KVCR"), numpy


def pin_result_blocks(result):
    """Keep diagnostic traversal of a framework-supplied mapping best effort."""
    try:
        return sum(refs is not None for refs in result[1].values())
    except Exception:
        return -1


def create_tracer():
    """Return None before optional imports when off; absence is harmless."""
    configured = os.getenv("KVCR_NVTX_LEVEL")
    level = configured if configured is not None else "low"
    if level == "off":
        return None
    if level not in ("low", "medium"):
        _warn_once("level", "Invalid KVCR_NVTX_LEVEL; expected off, low or medium.")
        return None
    try:
        domain, numpy = _load_backend()
        return _PinTracer(domain, numpy, level)
    except Exception:
        if configured is not None:
            _warn_once(
                "backend",
                "KVCR NVTX unavailable; install the profiling extra to enable it.",
            )
        return None


class _PinTracer:
    def __init__(self, domain, numpy, level):
        self.domain, self.numpy, self.level = domain, numpy, level
        instance = uuid4().int
        self.instance_hi, self.instance_lo = instance >> 64, instance & (2**64 - 1)
        self._ids = count(1)
        self.category = domain.get_category_id("framework_pin")
        for name in _NAMES:
            domain.get_registered_string(name)
        self.dtype = numpy.dtype(
            [
                ("schema_version", "u2"),
                ("instance_hi", "u8"),
                ("instance_lo", "u8"),
                ("pin_id", "u8"),
                ("pin_request_known", "u1"),
                ("pin_request_id", "i8"),
                ("source_op_id", "i8"),
                ("op_handle", "i8"),
                ("requested_blocks", "u8"),
                ("completed_blocks", "i8"),
                ("status", "u1"),
                ("reason", "u1"),
                ("fw_dram_utilization_known", "u1"),
            ]
        )

    def begin(self, block_count, *, source_op_id=0, op_handle=0):
        return _PinTrace(self, next(self._ids), block_count, source_op_id, op_handle)


@dataclass
class _PinTrace:
    tracer: _PinTracer
    pin_id: int
    requested_blocks: int
    source_op_id: int
    op_handle: int
    request_id: int | None = None
    finished: bool = False

    def _attributes(
        self,
        name,
        status=Status.PENDING,
        reason=Reason.NONE,
        completed_blocks=-1,
        source_op_id=None,
        op_handle=None,
    ):
        tracer = self.tracer
        # Framework IDs are unrestricted Python ints. An unrepresentable ID must
        # not suppress the independent pin identity or terminal status event.
        request_known = (
            self.request_id is not None and -(2**63) <= self.request_id < 2**63
        )
        # Each event owns its buffer and attributes. Never mutate shared payloads.
        payload = tracer.numpy.array(
            (
                1,
                tracer.instance_hi,
                tracer.instance_lo,
                self.pin_id,
                request_known,
                self.request_id if request_known else 0,
                self.source_op_id if source_op_id is None else source_op_id,
                self.op_handle if op_handle is None else op_handle,
                self.requested_blocks,
                completed_blocks,
                status,
                reason,
                0,
            ),
            dtype=tracer.dtype,
        )
        return tracer.domain.get_event_attributes(
            message=name,
            category=tracer.category,
            payload=payload,
        )

    def _mark(self, name, **fields):
        try:
            self.tracer.domain.mark(self._attributes(name, **fields))
        except Exception:
            pass  # Annotation errors never alter KVCR results or resources.

    def push(self):
        try:
            self.tracer.domain.push_range(self._attributes("source.pin.framework"))
            return True
        except Exception:
            return False

    def pop(self, pushed):
        if pushed:
            try:
                self.tracer.domain.pop_range()
            except Exception:
                pass

    def registered(self, request_id):
        self.request_id = request_id
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
        if self.finished:
            return
        self.finished = True
        status = {
            "success": Status.SUCCESS,
            "failed": Status.FAILED,
            "timeout": Status.TIMEOUT,
            "cancelled": Status.CANCELLED,
        }[result]
        if status is Status.SUCCESS and 0 <= completed_blocks < self.requested_blocks:
            status = Status.PARTIAL
        if reason is None:
            reason = {
                "success": Reason.NONE,
                "failed": Reason.UNKNOWN,
                "timeout": Reason.DEADLINE,
                "cancelled": Reason.CANCELLED,
            }[result]
        elif isinstance(reason, str):
            reason = Reason[reason.upper()]
        if status is Status.PARTIAL and reason is Reason.NONE:
            reason = Reason.UNKNOWN
        self._mark(
            "source.pin.completed",
            status=status,
            reason=reason,
            completed_blocks=completed_blocks,
        )
