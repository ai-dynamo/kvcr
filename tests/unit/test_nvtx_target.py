# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Target events must retain asynchronous and caller completion boundaries."""

import ctypes
from types import SimpleNamespace

import pytest
import test_kvcr_remote_target
import test_nvtx
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _mem_descriptor,
    _new_kvcr,
    _poll_until,
    _router_hint,
    _wait_until,
    _write_done_notification,
)
from test_nvtx import payloads, spans

from kvcr import _nvtx
from kvcr.config import LocalDramOptions, RemoteFWDramOptions
from kvcr.types import BlockKey, RegionDescriptor

recording = test_nvtx.recording


@pytest.mark.parametrize("level", ["low", "medium"])
def test_target_low_avoids_descriptor_scans_but_preserves_result(
    recording, monkeypatch, level
):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    calls = []
    monkeypatch.setattr(_nvtx, "tier", lambda refs: calls.append("tier") or 1)
    monkeypatch.setattr(
        _nvtx, "memory_kind", lambda refs, regions: calls.append("memory") or 1
    )
    kvcr = SimpleNamespace(
        nixl_agent_name="target",
        _memory_regions=({}, {}),
        _remote_fw_dram=SimpleNamespace(
            _dangling_ops=SimpleNamespace(incarnation="inc")
        ),
        _descriptor_bytes=lambda refs: calls.append("bytes") or 16,
    )
    trace = _nvtx.create_tracer().operation(kvcr, 41, {b"key": (object(),)}, "request")
    assert trace is not None
    op = SimpleNamespace(
        completed_keys={b"key"},
        keys={b"key"},
        success=True,
        ordered_keys=(b"key",),
        dst_descriptors=((object(),),),
        _backend=SimpleNamespace(_kvcr=kvcr),
    )
    trace.target_result(op, _nvtx.Reason.NONE)
    assert bool(calls) == (level == "medium")
    event = payloads(recording, "target.write_done.received")[0]
    assert int(event["completed_blocks"]) == 1
    assert int(event["completed_bytes"]) == (16 if level == "medium" else -1)


def target(*, eager=False):
    agent, control = FakeNixlAgent(), FakeBytesControl()
    kvcr = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        control,
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=eager),
    )
    return kvcr, agent, control


@pytest.mark.parametrize("failure", ["metadata", "bookkeeping"])
def test_return_annotation_failure_preserves_completion_batch(recording, failure):
    kvcr, _, _ = target()
    handles = [kvcr.deliver({}) for _ in range(2)]
    trace = kvcr._core._nvtx_operations[handles[0]]
    if failure == "metadata":
        del trace.completion_fields
    else:

        class FailedBookkeeping(set):
            def add(self, name):
                raise RuntimeError("annotation bookkeeping failed")

        trace._observed = FailedBookkeeping(trace._observed)
    assert list(kvcr.poll_completed()) == [(handle, {}) for handle in handles]
    assert not list(kvcr.poll_completed())
    assert not kvcr._core._nvtx_operations
    assert len(spans(recording, "op.deliver.lifecycle", "end")) == 2
    assert not recording.ranges


def test_rejected_progress_submission_closes_unaccepted_target_range(
    recording, monkeypatch
):
    kvcr, _, _ = target()
    request = "submission-failure"
    kvcr.submit_hint(_router_hint("tcp://source:1"), request_id=request)

    def reject(_op):
        for name in ("op.deliver.lifecycle", "target.remote"):
            assert len(spans(recording, name)) == 1
            assert not spans(recording, name, "end")
        raise RuntimeError("progress submission rejected")

    monkeypatch.setattr(kvcr._core._progress, "submit", reject)
    with pytest.raises(RuntimeError, match="progress submission rejected"):
        kvcr.deliver({BlockKey(b"a"): [_mem_descriptor()]}, request_id=request)
    assert len(spans(recording, "op.deliver.lifecycle", "end")) == 1
    assert len(spans(recording, "target.remote")) == 1
    assert len(spans(recording, "target.remote", "end")) == 1
    rejected = payloads(recording, "target.queued.rejected")
    assert len(rejected) == 1
    assert int(rejected[0]["status"]) == int(_nvtx.Status.REJECTED)
    assert not payloads(recording, "target.start_write.enqueued")
    assert not kvcr._core._nvtx_operations
    assert not recording.stack


@pytest.mark.parametrize("failure", ["expired", "false", "exception"])
def test_target_rejection_before_control_enqueue(recording, monkeypatch, failure):
    kvcr, _, control = target()
    control.send_result = False
    if failure == "expired":
        submit = kvcr._core._progress.submit

        def expire(op):
            op.deadline = 0
            submit(op)

        monkeypatch.setattr(kvcr._core._progress, "submit", expire)
    elif failure == "exception":

        def fail_send(endpoint, message):
            control.sent.append((endpoint, message))
            raise RuntimeError("control enqueue failed")

        monkeypatch.setattr(control, "send", fail_send)
    kvcr.submit_hint(_router_hint("tcp://source:1"), request_id="req")
    handle = kvcr.deliver({BlockKey(b"k"): [_mem_descriptor()]}, request_id="req")
    result = dict(_poll_until(kvcr, lambda entries: handle in dict(entries)))
    assert not result[handle][BlockKey(b"k")].success
    rejected = payloads(recording, "target.start_write.rejected")
    assert len(rejected) == 1
    assert int(rejected[0]["status"]) == int(
        _nvtx.Status.TIMEOUT if failure == "expired" else _nvtx.Status.REJECTED
    )
    assert int(rejected[0]["reason"]) == int(
        _nvtx.Reason.DEADLINE if failure == "expired" else _nvtx.Reason.CONTROL_ERROR
    )
    assert len(control.sent) == (failure != "expired")
    assert not payloads(recording, "target.start_write.enqueued")
    assert len(spans(recording, "target.remote", "end")) == 1
    assert not recording.stack


@pytest.mark.parametrize(
    "indices,expected",
    [
        ((0, 1), _nvtx.Status.SUCCESS),
        ((0,), _nvtx.Status.PARTIAL),
        ((9,), _nvtx.Status.FAILED),
    ],
)
def test_target_receipt_consumption_and_caller_return(recording, indices, expected):
    kvcr, agent, control = target()
    request = "real-request-α"
    kvcr.submit_hint(_router_hint("tcp://source:1"), request_id=request)
    handle = kvcr.deliver(
        {BlockKey(b"a"): [_mem_descriptor()], BlockKey(b"b"): [_mem_descriptor(1)]},
        request_id=request,
    )
    _wait_until(lambda: bool(control.sent))
    agent.notifs["source"] = [
        _write_done_notification(handle, completed_indices=indices)
    ]
    _wait_until(lambda: ("target", handle) not in kvcr._core._progress._in_flight_ops)
    received = payloads(recording, "target.write_done.received")
    assert len(received) == 1
    assert not payloads(recording, "op.completion_returned")
    assert len(spans(recording, "op.deliver.lifecycle")) == 1
    assert not spans(recording, "op.deliver.lifecycle", "end")
    assert len(spans(recording, "target.remote", "end")) == 1
    result = _poll_until(kvcr, bool)
    assert len(result) == 1
    for name in (
        "target.main.consume",
        "op.completion_queued",
        "op.completion_returned",
    ):
        events = payloads(recording, name)
        assert len(events) == 1
        assert int(events[0]["op_handle"]) == handle
    returned = payloads(recording, "op.completion_returned")[0]
    assert int(returned["status"]) == int(expected)
    assert (int(returned["request_hi"]), int(returned["request_lo"])) == _nvtx.identity(
        request
    )
    assert int(received[0]["target_incarnation_hi"]) != 0
    assert not kvcr._core._nvtx_operations
    assert len(spans(recording, "op.deliver.lifecycle", "end")) == 1
    assert not recording.stack
    kvcr.discard_hint(request)
    assert len(payloads(recording, "hint.discarded")) == 1


@pytest.mark.parametrize(
    "first,remote_success", [("local", True), ("remote", True), ("remote", False)]
)
def test_public_mixed_delivery_waits_for_both_branches(
    recording, first, remote_success
):
    primary = ctypes.create_string_buffer(b"a" * 16 + b"\0" * 16, 32)
    local = ctypes.create_string_buffer(16)
    agent, control = FakeNixlAgent(), FakeBytesControl()
    kvcr = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        control,
        local_dram=LocalDramOptions([("", ctypes.addressof(local), len(local))]),
        framework_regions=[
            RegionDescriptor(addr=ctypes.addressof(primary), size=16, count=2)
        ],
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
    )
    local_key, remote_key = BlockKey(b"local"), BlockKey(b"remote")
    deposit = kvcr.deposit({local_key: [_mem_descriptor()]})
    _wait_until(lambda: bool(agent.transfers))
    agent.state = "DONE"
    assert deposit in dict(_poll_until(kvcr, bool))
    recording.events.clear()
    agent.state = "PROC"
    request = "mixed-request-α"
    kvcr.submit_hint(_router_hint("tcp://source:1"), request_id=request)
    handle = kvcr.deliver(
        {local_key: [_mem_descriptor()], remote_key: [_mem_descriptor(1)]},
        request_id=request,
    )
    _wait_until(lambda: bool(control.sent) and len(agent.transfers) == 2)
    dispatched = payloads(recording, "op.dispatched")[0]
    assert (
        int(dispatched["local_blocks"]),
        int(dispatched["remote_blocks"]),
        int(dispatched["g3_blocks"]),
    ) == (1, 1, 0)
    submitted = payloads(recording, "hint.submitted")[0]
    used = payloads(recording, "hint.used")[0]
    queued = payloads(recording, "target.queued")[0]
    assert (
        int(submitted["trace_id"])
        == int(used["trace_id"])
        == int(queued["hint_trace_id"])
    )
    assert int(used["op_handle"]) == handle
    notification = _write_done_notification(handle, success=remote_success)
    if first == "local":
        agent.state = "DONE"
        _wait_until(lambda: agent.transfers[-1] in agent.released_xfers)
        assert list(kvcr.poll_completed()) == []
        assert not payloads(recording, "op.completion_queued")
        agent.notifs["source"] = [notification]
    else:
        agent.notifs["source"] = [notification]
        _wait_until(lambda: bool(payloads(recording, "target.write_done.received")))
        assert list(kvcr.poll_completed()) == []
        assert not payloads(recording, "op.completion_queued")
        agent.state = "DONE"
    result = dict(_poll_until(kvcr, bool))
    assert set(result) == {handle}
    assert result[handle][local_key].success
    assert result[handle][remote_key].success == remote_success
    assert set(result[handle]) == {local_key, remote_key}
    assert len(payloads(recording, "op.completion_queued")) == 1
    returned = payloads(recording, "op.completion_returned")
    assert len(returned) == 1
    assert int(returned[0]["completed_blocks"]) == 1 + remote_success
    assert int(returned[0]["status"]) == int(
        _nvtx.Status.SUCCESS if remote_success else _nvtx.Status.PARTIAL
    )
    assert (
        int(returned[0]["request_hi"]),
        int(returned[0]["request_lo"]),
    ) == _nvtx.identity(request)
    assert list(kvcr.poll_completed()) == []
    kvcr.discard_hint(request)


def test_replaced_hint_is_used_and_conflicting_hint_is_rejected(recording):
    kvcr, agent, control = target()
    request = "reused-request"
    kvcr.submit_hint(_router_hint("tcp://source:1"), request)
    kvcr.submit_hint(_router_hint("tcp://source:1"), request)
    submitted = payloads(recording, "hint.submitted")
    assert len(submitted) == 2
    assert int(submitted[0]["trace_id"]) != int(submitted[1]["trace_id"])
    assert int(payloads(recording, "hint.replaced")[0]["trace_id"]) == int(
        submitted[0]["trace_id"]
    )
    for _ in range(2):
        handle = kvcr.deliver({BlockKey(b"k"): [_mem_descriptor()]}, request)
        _wait_until(lambda: ("target", handle) in kvcr._core._progress._in_flight_ops)
        agent.notifs["source"] = [_write_done_notification(handle)]
        assert handle in dict(_poll_until(kvcr, bool))
    used = payloads(recording, "hint.used")
    assert len(used) == 2
    assert all(
        int(event["trace_id"]) == int(submitted[1]["trace_id"]) for event in used
    )
    assert len({int(event["op_handle"]) for event in used}) == 2
    control.sent.clear()
    kvcr.submit_hint(_router_hint("tcp://other:2"), request)
    rejected = payloads(recording, "hint.rejected")[0]
    assert int(rejected["trace_id"]) == int(submitted[1]["trace_id"])
    assert int(rejected["reason"]) == int(_nvtx.Reason.HINT_CONFLICT)
    failed = kvcr.deliver({BlockKey(b"k"): [_mem_descriptor()]}, request)
    assert not dict(kvcr.poll_completed())[failed][BlockKey(b"k")].success
    assert len(payloads(recording, "hint.used")) == 2
    assert not control.sent
    kvcr.discard_hint(request)


def test_timeout_quarantine_and_late_quiescence_have_distinct_events(recording):
    test_kvcr_remote_target.test_kvcr_deliver_timeout_probes_source_before_finishing(
        False
    )
    for name in ("target.cancel_requested", "target.quarantined", "target.quiesced"):
        assert len(payloads(recording, name)) == 1
    handle = int(payloads(recording, "target.quarantined")[0]["op_handle"])
    returned = [
        p
        for p in payloads(recording, "op.completion_returned")
        if int(p["op_handle"]) == handle
    ]
    assert len(returned) == 1
    assert int(returned[0]["status"]) == int(_nvtx.Status.FAILED)
    assert int(returned[0]["reason"]) == int(_nvtx.Reason.DEADLINE)


def test_cancellation_cause_survives_grace_deadline(recording):
    kvcr, agent, control = target(eager=True)
    now = 0.0
    kvcr._core._clock = lambda: now
    handle, _ = test_kvcr_remote_target._acked_deliver(
        control, kvcr, "tcp://source:1", BlockKey(b"k")
    )
    agent.notifs["source"] = [
        _write_done_notification(handle, success=False, terminal=False)
    ]
    _wait_until(lambda: bool(payloads(recording, "target.cancel_requested")))
    now = 1000.0
    result = dict(_poll_until(kvcr, bool))
    assert not result[handle][BlockKey(b"k")].success
    assert len(payloads(recording, "target.quarantined")) == 1
    reason = int(payloads(recording, "op.completion_returned")[0]["reason"])
    assert len(spans(recording, "op.deliver.lifecycle", "end")) == 1
    assert not spans(recording, "target.remote", "end")
    agent.notifs["source"] = [_write_done_notification(handle)]
    _poll_until(kvcr, lambda _: bool(payloads(recording, "target.quiesced")))
    assert len(spans(recording, "target.remote", "end")) == 1
    assert len(payloads(recording, "op.completion_returned")) == 1
    kvcr.close()
    assert reason == int(_nvtx.Reason.CANCELLED)


def test_negative_fill_handle_and_copy_do_not_duplicate_completion(recording):
    test_kvcr_remote_target.test_remote_fetch_timeout_keeps_slot_until_source_is_terminal(
        "guard"
    )
    queued = payloads(recording, "target.queued")
    assert len(queued) == 1
    assert int(queued[0]["op_handle"]) < 0
    assert int(queued[0]["local_fill"]) == 1
    assert len(payloads(recording, "target.quiesced")) == 1

    assert not recording.stack


def test_missing_hint_has_specific_reason(recording):
    kvcr, _, _ = target()
    handle = kvcr.deliver({BlockKey(b"k"): [_mem_descriptor()]})
    result = dict(kvcr.poll_completed())
    assert not result[handle][BlockKey(b"k")].success
    assert int(payloads(recording, "op.completion_returned")[0]["reason"]) == int(
        _nvtx.Reason.HINT_UNAVAILABLE
    )


@pytest.mark.parametrize("failure", ["attributes", "push", "mark", "pop"])
def test_target_annotations_cannot_break_empty_or_missing_hint_delivery(
    recording, failure
):
    kvcr, _, _ = target()
    recording.fail = failure
    empty = kvcr.deliver({})
    missing = kvcr.deliver({BlockKey(b"k"): [_mem_descriptor()]})
    results = dict(kvcr.poll_completed())
    assert results[empty] == {}
    assert not results[missing][BlockKey(b"k")].success
    assert not kvcr._core._nvtx_operations
    assert not recording.stack


def test_target_context_reuse_clears_hint_and_block_counts(recording):
    tracer = _nvtx.create_tracer()
    first = tracer.lifecycle(
        hint_trace_id=123, local_blocks=1, remote_blocks=2, g3_blocks=3
    )
    second = tracer.lifecycle()
    contexts = payloads(recording, "request.context")
    assert [int(p["trace_id"]) for p in contexts] == [
        first.trace_id,
        second.trace_id,
    ]
    assert [
        tuple(
            int(p[name])
            for name in ("hint_trace_id", "local_blocks", "remote_blocks", "g3_blocks")
        )
        for p in contexts
    ] == [(123, 1, 2, 3), (0, -1, -1, -1)]
