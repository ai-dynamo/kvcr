# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Target events must retain asynchronous and caller completion boundaries."""

import ctypes
from types import SimpleNamespace

import msgspec
import pytest
import test_kvcr_remote_target
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
from _nvtx_test_utils import make_target, payloads, spans, wait_for_payload

from kvcr import _nvtx
from kvcr.backends.remote_fw_dram import _RemoteFWDram, _TargetPullOp, _TargetPullState
from kvcr.config import LocalDramOptions, RemoteFWDramOptions
from kvcr.types import BlockKey, RegionDescriptor


@pytest.mark.parametrize(
    "factory,range_name",
    [("lifecycle", None), ("operation", None), ("operation", "target.remote")],
)
def test_trace_factory_failure_preserves_transfer(
    recording, monkeypatch, factory, range_name
):
    kvcr, agent, control = make_target()
    kvcr.submit_hint(_router_hint("tcp://source:1"), request_id="request")
    [submitted] = payloads(recording, "hint.submitted")
    tracer = kvcr._core._nvtx
    original = getattr(tracer, factory)
    monkeypatch.setattr(
        tracer,
        factory,
        lambda *args, **kwargs: (
            None
            if kwargs.get("range_name") == range_name
            else original(*args, **kwargs)
        ),
    )
    if factory == "lifecycle":
        kvcr.submit_hint(_router_hint("tcp://source:1"), request_id="request")
        [replaced] = payloads(recording, "hint.replaced")
        assert int(replaced["trace_id"]) == int(submitted["trace_id"])
        assert len(payloads(recording, "hint.submitted")) == 1
    handle = kvcr.deliver({BlockKey(b"k"): [_mem_descriptor()]}, request_id="request")
    caller_traced = bool(kvcr._core._nvtx_operations)
    _wait_until(lambda: bool(control.sent))
    agent.notifs["source"] = [_write_done_notification(handle)]
    [(completed_handle, result)] = _poll_until(kvcr, bool)
    assert completed_handle == handle
    assert result[BlockKey(b"k")].success
    assert caller_traced == (range_name == "target.remote")
    assert [int(event["trace_id"]) for event in payloads(recording, "hint.used")] == (
        [] if factory == "lifecycle" else [int(submitted["trace_id"])]
    )
    assert not list(kvcr.poll_completed())
    assert not kvcr._core._nvtx_operations
    assert not recording.ranges
    assert not recording.stack


@pytest.mark.parametrize(
    "local_fill,results,prior,reason",
    [
        (True, (True,), _nvtx.Reason.UNKNOWN, _nvtx.Reason.DEADLINE),
        (False, (False,), _nvtx.Reason.DEADLINE, _nvtx.Reason.DEADLINE),
        (False, (False,), _nvtx.Reason.UNKNOWN, _nvtx.Reason.REMOTE_FAILURE),
        (True, (False, True), _nvtx.Reason.UNKNOWN, _nvtx.Reason.REMOTE_FAILURE),
        (False, (True,), _nvtx.Reason.UNKNOWN, _nvtx.Reason.NONE),
        (False, (True,), _nvtx.Reason.CANCELLED, _nvtx.Reason.CANCELLED),
    ],
)
def test_target_terminal_reason_preserves_cause(
    recording, local_fill, results, prior, reason
):
    key, handle = BlockKey(b"k"), -7 if local_fill else 7
    trace = _nvtx.create_tracer().lifecycle(requested_blocks=1, op_handle=handle)
    trace.start("target.remote")
    trace.failure_reason = prior
    backend = SimpleNamespace(
        _kvcr=SimpleNamespace(_clock=lambda: 10.0, _descriptor_bytes=lambda refs: 0),
        _record_progress_duration=lambda *args: None,
        _record_progress_counter=lambda *args: None,
    )
    op = _TargetPullOp(
        op_id=("target", handle),
        keys=frozenset({key}),
        started_at=0.0,
        deadline=1.0,
        state=_TargetPullState.WAITING_WRITE_DONE
        if prior is _nvtx.Reason.UNKNOWN
        else _TargetPullState.WAITING_TERMINAL,
        local_fill=local_fill,
        remote_ctrl_ep="tcp://source:1",
        _backend=backend,
        ordered_keys=(key,),
        dst_descriptors=((),),
        trace=trace,
    )
    notifications = [_write_done_notification(handle, success=ok) for ok in results]
    progress = SimpleNamespace(
        nixl_agent=SimpleNamespace(get_new_notifs=lambda: {"source": notifications})
    )
    event = _RemoteFWDram._poll_notifications(backend, progress)[op.op_id]
    assert op.progress(progress, event) == (True, True)
    assert op.state is _TargetPullState.FINISHED
    assert op.success is (reason is _nvtx.Reason.NONE)
    [received] = payloads(recording, "target.write_done.received")
    assert int(received["reason"]) == reason
    assert int(received["status"]) == (
        _nvtx.Status.SUCCESS if op.success else _nvtx.Status.FAILED
    )
    assert len(spans(recording, "target.remote", "end")) == 1
    assert not recording.ranges


def test_shutdown_closes_unreturned_caller_range(recording):
    kvcr, _, _ = make_target()
    handle = kvcr.deliver({})
    assert len(payloads(recording, "op.completion_queued")) == 1
    assert not payloads(recording, "op.completion_returned")
    kvcr.close()
    kvcr.close()
    [unreturned] = payloads(recording, "op.shutdown_unreturned")
    assert int(unreturned["op_handle"]) == handle
    assert int(unreturned["status"]) == _nvtx.Status.UNRESOLVED
    assert int(unreturned["reason"]) == _nvtx.Reason.SHUTDOWN
    assert len(spans(recording, "op.deliver.lifecycle", "end")) == 1
    assert not kvcr._core._nvtx_operations
    assert not recording.ranges


def test_rejected_progress_submission_closes_unaccepted_target_range(
    recording, monkeypatch
):
    kvcr, _, _ = make_target()
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
    [rejected] = payloads(recording, "target.queued.rejected")
    assert int(rejected["status"]) == int(_nvtx.Status.REJECTED)
    assert not payloads(recording, "target.start_write.enqueued")
    assert not kvcr._core._nvtx_operations
    assert not recording.stack


@pytest.mark.parametrize("failure", ["expired", "false", "exception"])
def test_target_rejection_before_control_enqueue(recording, monkeypatch, failure):
    kvcr, _, control = make_target()
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
    [rejected] = payloads(recording, "target.start_write.rejected")
    assert int(rejected["status"]) == int(
        _nvtx.Status.TIMEOUT if failure == "expired" else _nvtx.Status.REJECTED
    )
    assert int(rejected["reason"]) == int(
        _nvtx.Reason.DEADLINE if failure == "expired" else _nvtx.Reason.CONTROL_ERROR
    )
    assert len(control.sent) == (failure != "expired")
    assert not payloads(recording, "target.start_write.enqueued")
    assert len(spans(recording, "target.remote", "end")) == 1
    assert not recording.stack


@pytest.mark.parametrize(
    "indices,expected,retry",
    [
        ((0, 1), _nvtx.Status.SUCCESS, False),
        ((0,), _nvtx.Status.PARTIAL, False),
        ((9,), _nvtx.Status.FAILED, False),
        ((0, 1), _nvtx.Status.SUCCESS, True),
    ],
)
def test_target_receipt_consumption_and_caller_return(
    recording, indices, expected, retry
):
    kvcr, agent, control = make_target()
    request = "real-request-α"
    kvcr.submit_hint(_router_hint("tcp://source:1"), request_id=request)
    handle = kvcr.deliver(
        {BlockKey(b"a"): [_mem_descriptor()], BlockKey(b"b"): [_mem_descriptor(1)]},
        request_id=request,
    )
    _wait_until(lambda: bool(control.sent))
    if retry:
        ranges = set(recording.ranges)
        assert len(ranges) == 2
        refusal = test_kvcr_remote_target._metadata_refusal("tcp://source:1", handle)
        control.incoming.append(msgspec.msgpack.encode(refusal))
        _wait_until(lambda: len(control.sent) == 2)
        assert not list(kvcr.poll_completed())
        assert set(recording.ranges) == ranges
        assert not payloads(recording, "target.write_done.received")
    agent.notifs["source"] = [
        _write_done_notification(handle, completed_indices=indices)
    ]
    _wait_until(lambda: ("target", handle) not in kvcr._core._progress._in_flight_ops)
    [received] = payloads(recording, "target.write_done.received")
    assert not payloads(recording, "op.completion_returned")
    assert len(spans(recording, "op.deliver.lifecycle")) == 1
    assert not spans(recording, "op.deliver.lifecycle", "end")
    assert len(spans(recording, "target.remote")) == 1
    assert len(spans(recording, "target.remote", "end")) == 1
    result = _poll_until(kvcr, bool)
    assert len(result) == 1
    for name in (
        "target.main.consume",
        "op.completion_queued",
        "op.completion_returned",
    ):
        [event] = payloads(recording, name)
        assert int(event["op_handle"]) == handle
    returned = payloads(recording, "op.completion_returned")[0]
    assert int(returned["status"]) == int(expected)
    assert (int(returned["request_hi"]), int(returned["request_lo"])) == _nvtx.identity(
        request
    )
    assert int(received["target_incarnation_hi"]) != 0
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
    [returned] = payloads(recording, "op.completion_returned")
    assert int(returned["completed_blocks"]) == 1 + remote_success
    assert int(returned["status"]) == int(
        _nvtx.Status.SUCCESS if remote_success else _nvtx.Status.PARTIAL
    )
    assert int(returned["reason"]) == (
        _nvtx.Reason.NONE if remote_success else _nvtx.Reason.REMOTE_FAILURE
    )
    assert (
        int(returned["request_hi"]),
        int(returned["request_lo"]),
    ) == _nvtx.identity(request)
    assert list(kvcr.poll_completed()) == []
    kvcr.discard_hint(request)


def test_timeout_quarantine_and_late_quiescence_have_distinct_events(recording):
    test_kvcr_remote_target.test_kvcr_deliver_timeout_probes_source_before_finishing(
        False
    )
    for name in ("target.cancel_requested", "target.quarantined", "target.quiesced"):
        assert len(payloads(recording, name)) == 1
    handle = int(payloads(recording, "target.quarantined")[0]["op_handle"])
    [returned] = [
        p
        for p in payloads(recording, "op.completion_returned")
        if int(p["op_handle"]) == handle
    ]
    assert int(returned["status"]) == int(_nvtx.Status.FAILED)
    assert int(returned["reason"]) == int(_nvtx.Reason.DEADLINE)


def test_cancellation_cause_survives_grace_deadline(recording):
    kvcr, agent, control = make_target(eager=True)
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
    wait_for_payload(kvcr, recording, "target.quiesced", ended="target.remote")
    assert len(spans(recording, "target.remote", "end")) == 1
    assert len(payloads(recording, "op.completion_returned")) == 1
    kvcr.close()
    assert reason == int(_nvtx.Reason.CANCELLED)


def test_negative_fill_handle_and_copy_do_not_duplicate_completion(recording):
    test_kvcr_remote_target.test_remote_fetch_timeout_keeps_slot_until_source_is_terminal(
        "guard"
    )
    [queued] = payloads(recording, "target.queued")
    assert int(queued["op_handle"]) < 0
    assert int(queued["local_fill"]) == 1
    assert len(payloads(recording, "target.quiesced")) == 1
    [unresolved] = payloads(recording, "target.shutdown_unresolved")
    assert int(unresolved["status"]) == _nvtx.Status.UNRESOLVED
    assert int(unresolved["reason"]) == _nvtx.Reason.SHUTDOWN
    events = [name for kind, name, *_ in recording.events if kind in ("mark", "end")]
    assert events.index("target.shutdown_unresolved") < events.index("target.quiesced")
    assert events.index("target.quiesced") < events.index("target.remote")
    assert not recording.stack


def test_missing_hint_has_specific_reason(recording):
    kvcr, _, _ = make_target()
    handle = kvcr.deliver({BlockKey(b"k"): [_mem_descriptor()]})
    result = dict(kvcr.poll_completed())
    assert not result[handle][BlockKey(b"k")].success
    assert int(payloads(recording, "op.completion_returned")[0]["reason"]) == int(
        _nvtx.Reason.HINT_UNAVAILABLE
    )


@pytest.mark.parametrize("failure", ["attributes", "push", "mark", "pop", "end"])
def test_target_annotations_cannot_break_empty_or_missing_hint_delivery(
    recording, failure
):
    kvcr, _, _ = make_target()
    recording.fail = failure
    empty = kvcr.deliver({})
    missing = kvcr.deliver({BlockKey(b"k"): [_mem_descriptor()]})
    results = list(kvcr.poll_completed())
    assert [handle for handle, _ in results] == [empty, missing]
    assert results[0][1] == {}
    assert not results[1][1][BlockKey(b"k")].success
    assert not list(kvcr.poll_completed())
    assert not kvcr._core._nvtx_operations
    assert len(recording.ended) == (0 if failure == "attributes" else 2)
    assert not recording.ranges
    assert not recording.stack


@pytest.mark.parametrize(
    "success,indices,expected",
    [(True, (0,), True), (False, (), False), (True, (9,), False)],
)
def test_off_skips_annotation_work_but_preserves_results(
    recording, monkeypatch, success, indices, expected
):
    accesses, reasons = [], _nvtx.Reason

    class Traces(dict):
        def pop(self, *args):
            accesses.append("completion lookup")
            return super().pop(*args)

    class Reasons:
        def __getattr__(self, name):
            accesses.append(name)
            return getattr(reasons, name)

    monkeypatch.setenv("KVCR_NVTX_LEVEL", "off")
    kvcr, agent, control = make_target()
    kvcr._core._nvtx_operations = Traces()
    empty = kvcr.deliver({})
    kvcr.submit_hint(_router_hint("tcp://source:1"), "req")
    key = BlockKey(b"k")
    handle = kvcr.deliver({key: [_mem_descriptor()]}, "req")
    _wait_until(lambda: bool(control.sent))
    monkeypatch.setattr(_nvtx, "Reason", Reasons())
    agent.notifs["source"] = [
        _write_done_notification(handle, success=success, completed_indices=indices)
    ]
    results = _poll_until(kvcr, lambda done: len(done) == 2)
    assert [h for h, _ in results] == [empty, handle]
    assert results[0][1] == {}
    assert results[1][1][key].success is expected
    assert not list(kvcr.poll_completed())
    assert accesses == []
    assert not recording.events
