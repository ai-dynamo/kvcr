# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Reusable NVTX payload storage; optional dependencies are injected."""

from enum import IntEnum
from itertools import count
from operator import itemgetter
from threading import local
from uuid import uuid4


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


class _ThreadBuffers(local):
    def __init__(self):
        self.buffers = {}


class _PayloadEmitter:
    def __init__(self, domain, numpy, level):
        self.domain, self.numpy, self.level = domain, numpy, level
        instance = uuid4().int
        self.instance_hi, self.instance_lo = instance >> 64, instance & (2**64 - 1)
        self._ids = count(1)
        self.category = domain.get_category_id("framework_pin")
        self.lifecycle_category = domain.get_category_id("remote_deliver")
        self._thread = _ThreadBuffers()
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
            self.event_dtype.descr
            + [
                ("pin_id", "u8"),
                ("pin_request_known", "u1"),
                ("pin_request_id", "i8"),
                ("fw_dram_utilization_known", "u1"),
                ("request_display_known", "u1"),
                ("length", "u2"),
                ("truncated", "u1"),
                ("value_utf8", "u1", (256,)),
            ]
        )
        self._field_types = {
            name: self.context_dtype.fields[name][0]
            for name in self.context_dtype.names
        }
        self._defaults = dict(
            schema_version=3,
            instance_hi=self.instance_hi,
            instance_lo=self.instance_lo,
            trace_id=0,
            status=Status.PENDING,
            reason=Reason.NONE,
        )
        self._prefix_names = frozenset(self._defaults)
        self._prefix = [(name, self._field_types[name]) for name in self._defaults]
        context_defaults = {
            name: -1 if name.endswith(("_blocks", "_bytes")) else 0
            for name in self.context_dtype.names
        }
        context_defaults.update(self._defaults)
        template = numpy.array(
            itemgetter(*self.context_dtype.names)(context_defaults),
            dtype=self.context_dtype,
        )
        template.flags.writeable = False
        self._context_template = template[()]

    def _emit(self, name, trace_id, fields, *, method, category=None, context=False):
        """Mutate bound storage once; isolate concurrent and nested emissions."""
        category = self.lifecycle_category if category is None else category
        field_names = tuple(fields)
        key = (name, category, context, () if context else field_names)
        buffers = self._thread.buffers
        pool = buffers.get(key)
        if pool is None:
            dtype = (
                self.context_dtype
                if context
                else self.numpy.dtype(
                    self._prefix
                    + [
                        (key, self._field_types[key])
                        for key in field_names
                        if key not in self._prefix_names
                    ]
                )
            )
            pool = (dtype, [])
            buffers[key] = pool
        dtype, available = pool
        if available:
            slot = available.pop()
        else:
            payload = self.numpy.zeros((), dtype=dtype)
            if not context:
                for field_name, value in self._defaults.items():
                    payload[field_name] = value
            # The factory registers/caches the textual name internally.
            # Keep the original array bound: resetting the payload property
            # traverses dtype metadata again on every event.
            attributes = self.domain.get_event_attributes(
                message=name, category=category, payload=payload
            )
            slot = (payload, attributes, {})
        payload, attributes, views = slot
        try:
            update_names = (
                field_names if "trace_id" in fields else ("trace_id", *field_names)
            )
            updates = views.get(update_names)
            if updates is None:
                updates = payload[list(update_names)]
                views[update_names] = updates
            if context:
                payload[()] = self._context_template
            updates[()] = (
                tuple(fields.values())
                if "trace_id" in fields
                else (trace_id, *fields.values())
            )
            return method(attributes)
        finally:
            available.append(slot)
