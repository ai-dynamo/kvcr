# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""NVTX must describe pin ownership without changing it."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    PendingPrimaryPinning,
    _new_kvcr,
    _poll_until,
    _start_write_message,
)

from kvcr import _nvtx
from kvcr.types import BlockKey


class RecordingDomain:
    def __init__(self):
        self.events = []
        self.registered = set()
        self.stack = []
        self.fail = None

    def get_registered_string(self, message):
        self.registered.add(message)
        return message

    def get_category_id(self, category):
        return 1

    def get_event_attributes(self, **kwargs):
        if self.fail == "attributes":
            raise RuntimeError("annotation failure")
        return SimpleNamespace(**kwargs)

    def _record(self, kind, attrs):
        if self.fail == kind:
            raise RuntimeError("annotation failure")
        self.events.append((kind, attrs.message, attrs.payload, threading.get_ident()))

    def mark(self, attrs):
        self._record("mark", attrs)

    def push_range(self, attrs):
        self._record("push", attrs)
        self.stack.append(threading.get_ident())

    def pop_range(self):
        assert self.stack.pop() == threading.get_ident()
        if self.fail == "pop":
            raise RuntimeError("annotation failure")


@pytest.fixture
def recording(monkeypatch):
    np = pytest.importorskip("numpy")
    domain = RecordingDomain()
    monkeypatch.setenv("KVCR_NVTX_LEVEL", "low")
    monkeypatch.setattr(_nvtx, "_load_backend", lambda: (domain, np))
    return domain


def payloads(domain, name):
    return [p for _, message, p, _ in domain.events if message == name]


def make_source(domain, handles=(9,), *, request_id=0):
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


def test_off_does_not_load_optional_dependencies(monkeypatch):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", "off")

    def forbidden():
        pytest.fail("off mode imported the optional backend")

    monkeypatch.setattr(_nvtx, "_load_backend", forbidden)
    assert _nvtx.create_tracer() is None


@pytest.mark.parametrize("level", [None, "low", "medium", "invalid"])
def test_missing_binding_is_safe_and_explicit_request_warns_once(
    monkeypatch, caplog, level
):
    _nvtx._WARNED.clear()
    if level is None:
        monkeypatch.delenv("KVCR_NVTX_LEVEL", raising=False)
    else:
        monkeypatch.setenv("KVCR_NVTX_LEVEL", level)

    def absent():
        raise ImportError("no nvtx")

    monkeypatch.setattr(_nvtx, "_load_backend", absent)
    assert _nvtx.create_tracer() is None
    assert _nvtx.create_tracer() is None
    assert len(caplog.records) == (0 if level is None else 1)


def test_default_low_and_fresh_payloads(recording, monkeypatch):
    monkeypatch.delenv("KVCR_NVTX_LEVEL")
    tracer = _nvtx.create_tracer()
    first = tracer.begin(2, source_op_id=3, op_handle=-(2**60))
    second = tracer.begin(5, source_op_id=4, op_handle=2**60)
    first.registered(0)
    second.registered(0)
    first.finish("success", completed_blocks=2)
    second.finish("failed", reason="unknown")
    first.finish("cancelled")
    registered = payloads(recording, "source.pin.registered")
    completed = payloads(recording, "source.pin.completed")
    assert [int(p["requested_blocks"]) for p in registered] == [2, 5]
    assert [int(p["op_handle"]) for p in registered] == [-(2**60), 2**60]
    assert registered[0]["pin_id"] != registered[1]["pin_id"]
    assert len(completed) == 2
    assert all(int(p["fw_dram_utilization_known"]) == 0 for p in completed)
    assert not recording.stack


@pytest.mark.parametrize("failure", ["attributes", "push", "mark", "pop"])
def test_annotation_failures_preserve_pin_callback_and_release(recording, failure):
    recording.fail = failure
    source, agent, pinning = make_source(recording)
    assert pinning.searches == [(BlockKey(b"k0"),)]
    agent.state = "DONE"
    pinning.complete(0)
    _poll_until(source, lambda _: pinning.unpins == ["pin"])
    assert not recording.stack


def test_shared_pin_has_one_lifetime_and_two_waiter_associations(recording):
    source, agent, pinning = make_source(recording, (9, 10))
    assert len(pinning.searches) == 1
    agent.state = "DONE"
    pinning.complete(0)
    _poll_until(source, lambda _: pinning.unpins == ["pin"])
    registered = payloads(recording, "source.pin.registered")
    waiters = payloads(recording, "source.pin.waiter")
    completed = payloads(recording, "source.pin.completed")
    assert len(registered) == len(completed) == 1
    assert {int(p["op_handle"]) for p in waiters} == {9, 10}
    assert {int(p["pin_id"]) for p in waiters} == {int(registered[0]["pin_id"])}
    assert int(completed[0]["completed_blocks"]) == 1
    assert int(completed[0]["status"]) == int(_nvtx.Status.SUCCESS)
    assert not recording.stack


@pytest.mark.parametrize("finish", ["success", "failed", "timeout", "cancelled"])
def test_terminal_pin_event_is_not_duplicated_by_late_result(recording, finish):
    source, agent, pinning = make_source(recording)
    backend = source._core._remote_fw_dram
    agent.state = "DONE"
    if finish == "success":
        pinning.complete(0)
        _poll_until(source, lambda _: pinning.unpins == ["pin"])
    elif finish == "failed":
        pinning.completed.append((0, None))
        _poll_until(source, lambda _: not backend._pending_pin_ops)
    elif finish == "timeout":
        source._core._clock = lambda: float("inf")
        _poll_until(source, lambda _: not backend._pending_pin_ops)
    else:
        source.close()
    # The late result is discarded/released, never a second terminal event.
    pinning.completed.append((0, None))
    backend._process_pending_pin_results()
    completed = payloads(recording, "source.pin.completed")
    assert len(completed) == 1
    assert int(completed[0]["status"]) == int(
        {
            "success": _nvtx.Status.SUCCESS,
            "failed": _nvtx.Status.FAILED,
            "timeout": _nvtx.Status.TIMEOUT,
            "cancelled": _nvtx.Status.CANCELLED,
        }[finish]
    )
    if finish == "failed":
        assert int(completed[0]["reason"]) == int(_nvtx.Reason.UNKNOWN)
    assert not recording.stack


@pytest.mark.parametrize("level,expected", [("low", 0), ("medium", 1)])
def test_detail_level_gates_waiter_detach(recording, monkeypatch, level, expected):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    source, agent, _ = make_source(recording, (9, 10))
    backend = source._core._remote_fw_dram
    op_id, op = next(iter(backend._source_pin_ops.items()))
    backend._cancel_pending_pin_for_op(op_id, op)
    assert len(payloads(recording, "source.pin.detached")) == expected
    assert not payloads(recording, "source.pin.completed")
    agent.state = "DONE"


def test_reused_framework_request_id_gets_new_trace_identity(recording):
    source, agent, pinning = make_source(recording)
    agent.state = "DONE"
    pinning.complete(0)
    _poll_until(source, lambda _: pinning.unpins == ["pin"])
    pinning._next_request_id = 0
    source._core.framework_control.incoming.append(
        _start_write_message(10, BlockKey(b"k1"))
    )
    _poll_until(source, lambda _: len(pinning.searches) == 2)
    pinning.complete(0)
    _poll_until(source, lambda _: len(pinning.unpins) == 2)
    records = payloads(recording, "source.pin.registered")
    assert [int(p["pin_request_id"]) for p in records] == [0, 0]
    assert records[0]["pin_id"] != records[1]["pin_id"]


@pytest.mark.parametrize(
    "request_id", [-(2**63) - 1, -(2**63), -1, 0, 2**63 - 1, 2**63, 2**128]
)
def test_framework_request_id_range_cannot_drop_lifecycle_events(recording, request_id):
    source, agent, pinning = make_source(recording, request_id=request_id)
    agent.state = "DONE"
    pinning.complete(request_id)
    _poll_until(source, lambda _: pinning.unpins == ["pin"])
    representable = -(2**63) <= request_id < 2**63
    identities = set()
    for name in ("registered", "waiter", "completed"):
        events = payloads(recording, f"source.pin.{name}")
        assert len(events) == 1
        event = events[0]
        assert bool(event["pin_request_known"]) == representable
        assert int(event["pin_request_id"]) == (request_id if representable else 0)
        identities.add(
            tuple(int(event[f]) for f in ("instance_hi", "instance_lo", "pin_id"))
        )
    assert len(identities) == 1
    assert int(payloads(recording, "source.pin.completed")[0]["status"]) == int(
        _nvtx.Status.SUCCESS
    )
    assert not recording.stack


def test_callback_failure_is_correlated_before_registration(recording):
    source, agent, pinning = make_source(recording)
    agent.state = "DONE"
    pinning.complete(0)
    _poll_until(source, lambda _: pinning.unpins == ["pin"])

    def fails(keys):
        raise RuntimeError("framework error is not a capacity diagnosis")

    source._core._request_pin_callback = fails
    source._core.framework_control.incoming.append(
        _start_write_message(11, BlockKey(b"other"))
    )
    _poll_until(source, lambda _: len(payloads(recording, "source.pin.completed")) == 2)
    event = payloads(recording, "source.pin.completed")[-1]
    assert int(event["op_handle"]) == 11
    assert int(event["source_op_id"]) > 0
    assert not int(event["pin_request_known"])
    assert int(event["reason"]) == int(_nvtx.Reason.CALLBACK_ERROR)
    assert not recording.stack


def test_partial_result_counts_available_blocks_without_guessing_cause(recording):
    agent, pinning, control = (
        FakeNixlAgent(),
        PendingPrimaryPinning(),
        FakeBytesControl(),
    )
    source = _new_kvcr(agent, pinning, control, name="source")
    message = msgspec.msgpack.decode(_start_write_message(9, BlockKey(b"k0")))
    message["keys"].append(b"k1")
    message["dst_descriptors"] *= 2
    control.incoming.append(msgspec.msgpack.encode(message))
    _poll_until(source, lambda _: bool(pinning.pending))
    agent.state = "DONE"
    pinning.complete(0, missing_indices=(1,))
    _poll_until(source, lambda _: pinning.unpins == ["pin"])
    event = payloads(recording, "source.pin.completed")[0]
    assert int(event["requested_blocks"]) == 2
    assert int(event["completed_blocks"]) == 1
    assert int(event["status"]) == int(_nvtx.Status.PARTIAL)
    assert int(event["reason"]) == int(_nvtx.Reason.UNKNOWN)


def test_invalid_result_has_bounded_reason_and_releases_handle(recording):
    source, agent, pinning = make_source(recording)
    agent.state = "DONE"
    pinning.completed.append((0, ("pin", {})))
    _poll_until(source, lambda _: pinning.unpins == ["pin"])
    event = payloads(recording, "source.pin.completed")[0]
    assert int(event["status"]) == int(_nvtx.Status.FAILED)
    assert int(event["reason"]) == int(_nvtx.Reason.INVALID_RESULT)


def test_payload_counting_failure_cannot_interrupt_pin_processing(recording):
    class ValuesUnavailable(dict):
        def values(self):
            raise RuntimeError("diagnostic-only traversal failed")

    source, agent, pinning = make_source(recording)
    agent.state = "DONE"
    pinning.complete(0)
    request, (handle, mapping) = pinning.completed.pop()
    pinning.completed.append((request, (handle, ValuesUnavailable(mapping))))
    _poll_until(source, lambda _: pinning.unpins == ["pin"])
    event = payloads(recording, "source.pin.completed")[0]
    assert int(event["status"]) == int(_nvtx.Status.SUCCESS)
    assert int(event["completed_blocks"]) == -1


def test_off_records_no_pin_state_or_events(recording, monkeypatch):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", "off")
    source, agent, pinning = make_source(recording)
    wait = next(iter(source._core._remote_fw_dram._pending_pin_ops.values()))
    assert wait.trace is None
    agent.state = "DONE"
    pinning.complete(0)
    _poll_until(source, lambda _: pinning.unpins == ["pin"])
    assert recording.events == []


def test_concurrent_annotations_have_independent_identities_and_payloads(recording):
    tracer = _nvtx.create_tracer()

    def annotate(value):
        trace = tracer.begin(value, op_handle=value)
        trace.registered(value)
        trace.finish("success", completed_blocks=value)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(annotate, range(1, 33)))
    events = payloads(recording, "source.pin.completed")
    assert len({int(p["pin_id"]) for p in events}) == 32
    assert {int(p["pin_request_id"]) for p in events} == set(range(1, 33))
    assert all(p["completed_blocks"] == p["requested_blocks"] for p in events)
    assert recording.registered == {
        "source.pin.framework",
        "source.pin.registered",
        "source.pin.waiter",
        "source.pin.completed",
        "source.pin.detached",
    }
