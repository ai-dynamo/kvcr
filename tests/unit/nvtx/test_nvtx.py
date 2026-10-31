# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""NVTX must describe pin ownership without changing it."""

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    PendingPrimaryPinning,
    _decode_notif,
    _new_kvcr,
    _poll_until,
    _start_write_message,
)
from _nvtx_test_utils import make_source, payloads, spans

from kvcr import _nvtx
from kvcr.types import BlockKey


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


@pytest.mark.parametrize("level", ["low", "medium"])
def test_explicit_level_and_fresh_payloads(recording, monkeypatch, level):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    tracer = _nvtx.create_tracer()
    assert tracer.level == level
    first = tracer.begin(2, source_op_id=3, op_handle=-(2**60))
    second = tracer.begin(5, source_op_id=4, op_handle=2**60)
    first.registered(0)
    second.registered(0)
    registered = payloads(recording, "source.pin.registered")
    assert [int(p["requested_blocks"]) for p in registered] == [2, 5]
    assert [int(p["op_handle"]) for p in registered] == [-(2**60), 2**60]
    assert all(int(p["fw_dram_utilization_known"]) == 0 for p in registered)


@pytest.mark.parametrize(
    "failure", ["attributes", "push", "mark", "pop", "start", "end"]
)
@pytest.mark.parametrize("after_context", [False, True])
def test_annotation_failures_preserve_pin_callback_and_release(
    recording, monkeypatch, failure, after_context
):
    if after_context:
        source_trace = _nvtx._PinTracer.source

        def fail_after_context(*args):
            trace = source_trace(*args)
            recording.fail = failure
            return trace

        monkeypatch.setattr(_nvtx._PinTracer, "source", fail_after_context)
    else:
        recording.fail = failure
    source, agent, pinning = make_source()
    assert pinning.searches == [(BlockKey(b"k0"),)]
    agent.state = "DONE"
    pinning.complete(0)
    _poll_until(source, lambda _: pinning.unpins == ["pin"])
    assert agent.transfers == agent.released_xfers == [1]
    assert _decode_notif(agent.xfers[0][-1])["success"] is True
    assert not agent.sent_notifs
    assert not source._core._progress._active_transfers
    assert not recording.stack


def test_unknown_pin_result_closes_range_without_raising(recording):
    pin = _nvtx.create_tracer().begin(1)
    pin.finish("unknown-result")
    pin.finish("success", completed_blocks=1)
    completed = payloads(recording, "source.pin.completed")
    assert len(completed) == 1
    assert int(completed[0]["status"]) == int(_nvtx.Status.UNRESOLVED)
    assert int(completed[0]["reason"]) == int(_nvtx.Reason.UNKNOWN)
    assert recording.ended == [0]
    assert not recording.ranges


@pytest.mark.parametrize("request_id", [0, 2**128])
def test_shared_pin_has_one_lifetime_and_two_waiter_associations(recording, request_id):
    source, agent, pinning = make_source((9, 10), request_id=request_id)
    assert len(pinning.searches) == 1
    assert len(spans(recording, "source.pin.wait")) == 1
    assert not spans(recording, "source.pin.wait", "end")
    agent.state = "DONE"
    pinning.complete(request_id)
    _poll_until(source, lambda _: pinning.unpins == ["pin"])
    registered = payloads(recording, "source.pin.registered")
    waiters = payloads(recording, "source.pin.waiter")
    completed = payloads(recording, "source.pin.completed")
    assert len(registered) == len(completed) == 1
    assert len(waiters) == 2
    assert all(
        bool(p["pin_request_known"]) == (request_id == 0)
        for p in registered + waiters + completed
    )
    assert {int(p["pin_id"]) for p in registered + waiters + completed} == {
        int(registered[0]["pin_id"])
    }
    assert {int(p["op_handle"]) for p in waiters} == {9, 10}
    assert int(completed[0]["completed_blocks"]) == 1
    assert int(completed[0]["status"]) == int(_nvtx.Status.SUCCESS)
    assert len(spans(recording, "source.pin.wait", "end")) == 1
    assert 0 in recording.ended  # Zero is a valid native range ID.
    assert not recording.stack


@pytest.mark.parametrize("finish", ["success", "failed", "timeout", "cancelled"])
def test_terminal_pin_event_is_not_duplicated_by_late_result(recording, finish):
    source, agent, pinning = make_source()
    backend = source._core._remote_fw_dram
    agent.state = "DONE"
    if finish == "success":
        pinning.complete(0)
        _poll_until(source, lambda _: pinning.unpins == ["pin"])
    elif finish == "failed":
        pinning.completed.append((0, None))
        _poll_until(source, lambda _: not backend._sources._pending_pin_ops)
    elif finish == "timeout":
        source._core._clock = lambda: float("inf")
        _poll_until(source, lambda _: not backend._sources._pending_pin_ops)
    else:
        source.close()
    # The late result is discarded/released, never a second terminal event.
    pinning.completed.append((0, None))
    backend._sources._process_pending_pin_results()
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
def test_shared_pin_detach_and_late_result(recording, monkeypatch, level, expected):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    source, agent, pinning = make_source((9, 10))
    backend = source._core._remote_fw_dram
    first, last = backend._sources._source_pin_ops.items()
    backend._sources._cancel_pending_pin_for_op(*first)
    assert len(payloads(recording, "source.pin.detached")) == expected
    assert not payloads(recording, "source.pin.completed")
    assert not pinning.cancelled
    backend._sources._cancel_pending_pin_for_op(*last)
    assert pinning.cancelled == [0]
    assert len(payloads(recording, "source.pin.detached")) == 2 * expected
    assert len(payloads(recording, "source.pin.completed")) == 1
    assert len(spans(recording, "source.pin.wait", "end")) == 1
    pinning.complete(0)
    backend._sources._process_pending_pin_results()
    assert pinning.unpins == ["pin"]
    [completed] = payloads(recording, "source.pin.completed")
    assert int(completed["status"]) == int(_nvtx.Status.CANCELLED)
    assert int(completed["reason"]) == int(_nvtx.Reason.CANCELLED)
    assert len(spans(recording, "source.pin.wait", "end")) == 1
    agent.state = "DONE"


def test_reused_framework_request_id_gets_new_trace_identity(recording):
    source, agent, pinning = make_source()
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


@pytest.mark.parametrize("request_id", [-(2**63) - 1, -(2**63), 0, 2**63 - 1, 2**63])
def test_framework_request_id_boundaries(recording, request_id):
    known = -(2**63) <= request_id < 2**63
    trace = _nvtx.create_tracer().begin(1)
    trace.registered(request_id)
    trace.waiter(2, 9)
    trace.finish("success", completed_blocks=1)
    for name in ("registered", "waiter", "completed"):
        [event] = payloads(recording, f"source.pin.{name}")
        assert int(event["pin_id"]) == trace.pin_id
        assert bool(event["pin_request_known"]) == known
        assert int(event["pin_request_id"]) == (request_id if known else 0)
    assert int(event["status"]) == 2


def test_callback_failure_is_correlated_before_registration(recording):
    source, agent, pinning = make_source()
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
    for name in ("source.pin.completed", "source.write.completed"):
        event = payloads(recording, name)[0]
        assert int(event["requested_blocks"]) == 2
        assert int(event["completed_blocks"]) == 1
        assert int(event["status"]) == int(_nvtx.Status.PARTIAL)
        assert int(event["reason"]) == int(_nvtx.Reason.UNKNOWN)


def test_invalid_result_has_bounded_reason_and_releases_handle(recording):
    source, agent, pinning = make_source()
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

    source, agent, pinning = make_source()
    agent.state = "DONE"
    pinning.complete(0)
    request, (handle, mapping) = pinning.completed.pop()
    pinning.completed.append((request, (handle, ValuesUnavailable(mapping))))
    _poll_until(source, lambda _: pinning.unpins == ["pin"])
    event = payloads(recording, "source.pin.completed")[0]
    assert int(event["status"]) == int(_nvtx.Status.SUCCESS)
    assert int(event["completed_blocks"]) == -1


@pytest.mark.parametrize("level", [None, "off"])
def test_off_loads_no_backend_and_records_no_pin_state_or_events(monkeypatch, level):
    if level is None:
        monkeypatch.delenv("KVCR_NVTX_LEVEL", raising=False)
    else:
        monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    monkeypatch.setattr(
        _nvtx, "_load_backend", lambda: pytest.fail("off mode loaded optional backend")
    )
    monkeypatch.setattr(
        "kvcr.backends.remote_fw_dram._NvtxState",
        lambda **kwargs: pytest.fail("off mode allocated tracing state"),
    )
    source, agent, pinning = make_source((9, 10))
    assert all(
        op.nvtx_state is None
        for op in source._core._remote_fw_dram._sources._source_pin_ops.values()
    )
    wait = next(iter(source._core._remote_fw_dram._sources._pending_pin_ops.values()))
    assert wait.trace is None
    assert wait.op_ids == set(source._core._remote_fw_dram._sources._source_pin_ops)
    agent.state = "DONE"
    pinning.complete(0)
    _poll_until(source, lambda _: pinning.unpins == ["pin"])
    assert source._core._remote_fw_dram._nvtx is None
