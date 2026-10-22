# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import threading
from types import SimpleNamespace

from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    PendingPrimaryPinning,
    _new_kvcr,
    _poll_until,
    _start_write_message,
)

from kvcr.types import BlockKey


class RecordingDomain:
    def __init__(self):
        self.events = []
        self.stack = []
        self.fail = None
        self.ranges = {}
        self.ended = []
        self._next_range = 0

    def get_category_id(self, category):
        return 1

    def get_event_attributes(self, **kwargs):
        if not isinstance(kwargs["message"], str):
            raise TypeError("expected str, not RegisteredString")
        if self.fail == "attributes":
            raise RuntimeError("annotation failure")
        return SimpleNamespace(**kwargs)

    def _record(self, kind, attrs):
        if self.fail == kind:
            raise RuntimeError("annotation failure")
        # NVTX consumes payloads synchronously. Keep the values observed at the
        # boundary, rather than a reference to mutable annotation storage.
        self.events.append(
            (kind, attrs.message, attrs.payload.copy(), threading.get_ident())
        )

    def mark(self, attrs):
        self._record("mark", attrs)

    def push_range(self, attrs):
        self._record("push", attrs)
        self.stack.append(threading.get_ident())

    def pop_range(self):
        assert self.stack.pop() == threading.get_ident()
        if self.fail == "pop":
            raise RuntimeError("annotation failure")

    def start_range(self, attrs):
        self._record("start", attrs)
        range_id = self._next_range
        self._next_range += 1
        self.ranges[range_id] = (attrs.message, attrs.payload.copy())
        return range_id

    def end_range(self, range_id):
        self.ended.append(range_id)
        name, payload = self.ranges.pop(range_id)
        if self.fail == "end":
            raise RuntimeError("annotation failure")
        self.events.append(("end", name, payload, threading.get_ident()))


def spans(domain, name, kind="start"):
    return [event for event in domain.events if event[0] == kind and event[1] == name]


def payloads(domain, name):
    contexts = {}
    for _, message, payload, _ in domain.events:
        if message in ("request.context", "source.pin.context"):
            key = tuple(
                int(payload[f]) for f in ("instance_hi", "instance_lo", "trace_id")
            )
            contexts[key] = {f: payload[f] for f in payload.dtype.names}
    result = []
    for _, message, payload, _ in domain.events:
        if message != name:
            continue
        event = {f: payload[f] for f in payload.dtype.names}
        if int(event["schema_version"]) == 3 and "trace_id" in event:
            key = tuple(
                int(event[f]) for f in ("instance_hi", "instance_lo", "trace_id")
            )
            event = {**contexts.get(key, {}), **event}
        result.append(event)
    return result


def wait_for_payload(kvcr, recording, name, *, ended=None):
    _poll_until(
        kvcr,
        lambda _: (
            bool(payloads(recording, name))
            and (ended is None or bool(spans(recording, ended, "end")))
        ),
    )
    return payloads(recording, name)[0]


def make_source(handles=(9,), *, request_id=0):
    agent = FakeNixlAgent(metadata=b"source-md")
    pinning, control = PendingPrimaryPinning(), FakeBytesControl()
    pinning._next_request_id = request_id
    source = _new_kvcr(agent, pinning, control, name="source")
    for handle in handles:
        control.incoming.append(_start_write_message(handle, BlockKey(b"k0")))
    _poll_until(
        source,
        lambda _: len(source._core._remote_fw_dram._source_pin_ops) == len(handles),
    )
    return source, agent, pinning
