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
from hashlib import blake2b
from importlib import import_module
from itertools import count
from types import MappingProxyType
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
    REJECTED = 7
    AMBIGUOUS = 8
    UNRESOLVED = 9


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
    SUBMIT_REJECTED = 9
    SUBMIT_AMBIGUOUS = 10
    CREATE_ERROR = 11
    PROGRESS_ERROR = 12
    RELEASE_ERROR = 13
    CONTROL_ERROR = 14
    INVALID_NOTIFICATION = 15
    REMOTE_FAILURE = 16
    ROUTE_CHANGED = 17
    SOURCE_STALLED = 18
    HINT_UNAVAILABLE = 19
    HINT_CONFLICT = 20


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
        self.lifecycle_category = domain.get_category_id("remote_deliver")
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
        self.event_dtype = numpy.dtype(
            [
                ("schema_version", "u2"),
                ("instance_hi", "u8"),
                ("instance_lo", "u8"),
                ("trace_id", "u8"),
                ("request_hi", "u8"),
                ("request_lo", "u8"),
                ("request_known", "u1"),
                ("session_known", "u1"),
                ("parent_session_known", "u1"),
                ("op_handle", "i8"),
                ("source_op_id", "i8"),
                ("target_agent_hi", "u8"),
                ("target_agent_lo", "u8"),
                ("target_incarnation_hi", "u8"),
                ("target_incarnation_lo", "u8"),
                ("route_generation", "u8"),
                ("transfer_id", "u8"),
                ("hint_trace_id", "u8"),
                ("local_blocks", "i8"),
                ("remote_blocks", "i8"),
                ("g3_blocks", "i8"),
                ("requested_blocks", "i8"),
                ("requested_bytes", "i8"),
                ("selected_blocks", "i8"),
                ("completed_blocks", "i8"),
                ("selected_bytes", "i8"),
                ("completed_bytes", "i8"),
                ("status", "u1"),
                ("reason", "u1"),
                ("local_fill", "u1"),
                ("source_tier", "u1"),
                ("destination_tier", "u1"),
                ("source_memory", "u1"),
                ("destination_memory", "u1"),
                ("native_error_known", "u1"),
                ("native_error", "i8"),
            ]
        )
        # Nsight 2025.3 exports NumPy Unicode payloads as empty strings. A
        # bounded byte array and explicit length also preserve embedded NULs.
        self.context_dtype = numpy.dtype(
            [
                ("schema_version", "u2"),
                ("instance_hi", "u8"),
                ("instance_lo", "u8"),
                ("trace_id", "u8"),
                ("request_hi", "u8"),
                ("request_lo", "u8"),
                ("request_known", "u1"),
                ("length", "u2"),
                ("truncated", "u1"),
                ("value_utf8", "u1", (256,)),
            ]
        )

    def begin(self, block_count, *, source_op_id=0, op_handle=0):
        return _PinTrace(self, next(self._ids), block_count, source_op_id, op_handle)

    def name_progress_thread(self):
        # Python 3.12 does not propagate Thread.name to Linux. Nsight displays
        # the OS name; nvtx 0.2.16 has no Python thread-naming API.
        try:
            import ctypes

            ctypes.CDLL(None).prctl(15, ctypes.c_char_p(b"kvcr-progress"), 0, 0, 0)
        except Exception:
            pass

    def lifecycle(
        self, *, target_agent=None, target_incarnation=None, request_id=None, **fields
    ):
        try:
            hi, lo = identity(target_agent)
            inc_hi, inc_lo = identity(target_incarnation)
            req_hi, req_lo = identity(request_id)
            trace = _LifecycleTrace(
                self,
                next(self._ids),
                {
                    "target_agent_hi": hi,
                    "target_agent_lo": lo,
                    "target_incarnation_hi": inc_hi,
                    "target_incarnation_lo": inc_lo,
                    "request_hi": req_hi,
                    "request_lo": req_lo,
                    "request_known": request_id is not None,
                    **fields,
                },
            )
            trace.context(request_id)
            return trace
        except Exception:
            return None

    def source(self, op, kvcr):
        """Prepare context once; diagnostics cannot interrupt a source write."""
        try:
            refs = tuple(ref for block in op.src_descriptors for ref in block)
            targets = tuple(ref for block in op.dst_descriptors for ref in block)
            return self.lifecycle(
                target_agent=op.route[0] or None,
                target_incarnation=op.target_incarnation,
                op_handle=op.op_handle,
                source_op_id=op.op_id[1],
                route_generation=op.route[1],
                requested_blocks=op.requested_blocks,
                selected_blocks=len(op.source_keys),
                selected_bytes=kvcr._descriptor_bytes(refs),
                source_tier=tier(refs),
                destination_tier=tier(targets),
                source_memory=memory_kind(refs, kvcr._memory_regions),
            )
        except Exception:
            return None

    def operation(
        self, kvcr, handle, blocks, request_id, *, local_fill=False, hint_trace_id=0
    ):
        try:
            refs = tuple(ref for block in blocks.values() for ref in block)
            return self.lifecycle(
                target_agent=kvcr.nixl_agent_name,
                target_incarnation=kvcr._remote_fw_dram._dangling_ops.incarnation,
                request_id=request_id,
                op_handle=handle,
                local_fill=local_fill,
                hint_trace_id=hint_trace_id,
                requested_blocks=len(blocks),
                requested_bytes=kvcr._descriptor_bytes(refs),
                destination_tier=tier(refs),
                destination_memory=memory_kind(refs, kvcr._memory_regions),
            )
        except Exception:
            return None


def identity(value):
    """Stable 128-bit label identity; no Python hash randomization or truncation."""
    if value is None:
        return 0, 0
    digest = blake2b(value.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest[:8], "big"), int.from_bytes(digest[8:], "big")


def tier(refs):
    """Storage ownership, deliberately independent of DRAM versus VRAM."""
    kinds = {1 if ref.framework else 2 for ref in refs}
    return next(iter(kinds)) if len(kinds) == 1 else 3 if kinds else 0


def memory_kind(refs, regions):
    kinds = set()
    for ref in refs:
        table = regions[0 if ref.framework else 1]
        region = table.get(ref.label)
        if region is None:
            region = table.get(
                ref.label.partition(":")[0] + (":*" if ref.framework else "")
            )
        kinds.add(
            {"DRAM": 1, "VRAM": 2, "FILE": 3}.get(getattr(region, "mem_type", None), 0)
        )
    return next(iter(kinds)) if len(kinds) == 1 else 0


class _LifecycleTrace:
    """Immutable context plus event bookkeeping owned by the operation lifecycle."""

    def __init__(self, tracer, trace_id, fields):
        self.tracer, self.trace_id = tracer, trace_id
        self._fields = MappingProxyType(fields)
        self._observed = set()
        self.failure_reason = Reason.UNKNOWN
        self.completion_fields = {}

    def completed(self, entries):
        try:
            completed = sum(entry.success for entry in entries.values())
            all_completed = completed == len(entries)
            self.completion_fields = dict(
                status=Status.SUCCESS
                if all_completed
                else Status.PARTIAL
                if completed
                else Status.FAILED,
                reason=Reason.NONE if all_completed else self.failure_reason,
                completed_blocks=completed,
                completed_bytes=self._fields.get("requested_bytes", -1)
                if all_completed
                else -1
                if completed
                else 0,
            )
            self.mark("op.completion_queued", once=True, **self.completion_fields)
        except Exception:
            pass

    def returned(self):
        self.mark("op.completion_returned", once=True, **self.completion_fields)

    def target_result(self, op, reason):
        try:
            complete = len(op.completed_keys)
            byte_count = op._backend._kvcr._descriptor_bytes(
                ref
                for key, refs in zip(op.ordered_keys, op.dst_descriptors)
                if key in op.completed_keys
                for ref in refs
            )
            self.mark(
                "target.write_done.received",
                once=True,
                status=Status.SUCCESS
                if op.success and complete == len(op.keys)
                else Status.PARTIAL
                if complete
                else Status.FAILED,
                reason=reason,
                completed_blocks=complete,
                completed_bytes=byte_count,
            )
        except Exception:
            pass

    def context(self, request_id):
        try:
            t = self.tracer
            encoded = b"" if request_id is None else request_id.encode("utf-8")
            payload = t.numpy.zeros((), dtype=t.context_dtype)
            values = dict(
                self._fields,
                schema_version=2,
                instance_hi=t.instance_hi,
                instance_lo=t.instance_lo,
                trace_id=self.trace_id,
                length=min(len(encoded), 256),
                truncated=len(encoded) > 256,
            )
            for key in t.context_dtype.names:
                if key in values:
                    payload[key] = values[key]
            payload["value_utf8"][: min(len(encoded), 256)] = t.numpy.frombuffer(
                encoded[:256], dtype="u1"
            )
            t.domain.mark(
                t.domain.get_event_attributes(
                    message="request.context",
                    category=t.lifecycle_category,
                    payload=payload,
                )
            )
        except Exception:
            pass

    def complete_source(self, success, blocks):
        requested = self._fields.get("requested_blocks", -1)
        status = Status.FAILED
        if success:
            status = (
                (Status.FAILED if blocks == 0 else Status.PARTIAL)
                if blocks < requested
                else Status.SUCCESS
            )
        self.mark(
            "source.write.completed",
            once=True,
            status=status,
            reason=Reason.NONE
            if success and blocks >= requested
            else self.failure_reason,
            completed_blocks=blocks if success else 0,
            completed_bytes=self._fields.get("selected_bytes", -1) if success else 0,
        )

    def _attributes(self, name, fields):
        t = self.tracer
        values = {
            "schema_version": 2,
            "instance_hi": t.instance_hi,
            "instance_lo": t.instance_lo,
            "trace_id": self.trace_id,
            "requested_blocks": -1,
            "requested_bytes": -1,
            "selected_blocks": -1,
            "completed_blocks": -1,
            "selected_bytes": -1,
            "completed_bytes": -1,
            "reason": Reason.NONE,
            "status": Status.PENDING,
            **self._fields,
            **fields,
        }
        payload = t.numpy.array(
            tuple(values.get(key, 0) for key in t.event_dtype.names),
            dtype=t.event_dtype,
        )
        return t.domain.get_event_attributes(
            message=name, category=t.lifecycle_category, payload=payload
        )

    def mark(self, name, *, once=False, detail=False, retain_cause=False, **fields):
        reason = fields.get("reason", Reason.NONE)
        if not retain_cause and reason not in (
            Reason.NONE,
            Reason.UNKNOWN,
            Reason.RELEASE_ERROR,
            Reason.SHUTDOWN,
        ):
            self.failure_reason = reason
        if detail and self.tracer.level != "medium":
            return
        if once:
            if name in self._observed:
                return
            self._observed.add(name)
        try:
            self.tracer.domain.mark(self._attributes(name, fields))
        except Exception:
            pass

    def push(self, name, **fields):
        try:
            self.tracer.domain.push_range(self._attributes(name, fields))
            return True
        except Exception:
            return False

    def pop(self, pushed):
        if pushed:
            try:
                self.tracer.domain.pop_range()
            except Exception:
                pass


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
            status = Status.FAILED if completed_blocks == 0 else Status.PARTIAL
        if reason is None:
            reason = {
                "success": Reason.NONE,
                "failed": Reason.UNKNOWN,
                "timeout": Reason.DEADLINE,
                "cancelled": Reason.CANCELLED,
            }[result]
        elif isinstance(reason, str):
            reason = Reason[reason.upper()]
        if status in (Status.PARTIAL, Status.FAILED) and reason is Reason.NONE:
            reason = Reason.UNKNOWN
        self._mark(
            "source.pin.completed",
            status=status,
            reason=reason,
            completed_blocks=completed_blocks,
        )
