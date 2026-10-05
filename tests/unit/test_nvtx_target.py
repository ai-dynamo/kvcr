# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Target events must retain asynchronous and caller completion boundaries."""

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
from test_nvtx import payloads

from kvcr import _nvtx
from kvcr.config import RemoteFWDramOptions
from kvcr.types import BlockKey, OpEntryResult, OpEntryStatus

recording = test_nvtx.recording


def target():
    agent, control = FakeNixlAgent(), FakeBytesControl()
    kvcr = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        control,
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
    )
    return kvcr, agent, control


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
    assert not recording.stack
    kvcr.discard_hint(request)
    assert len(payloads(recording, "hint.discarded")) == 1


def test_joined_completion_waits_for_all_tiers(recording):
    kvcr, _, _ = target()
    core = kvcr._core
    trace = core._nvtx.lifecycle(op_handle=777, requested_blocks=2)
    core._nvtx_operations[777] = trace
    keys = (BlockKey(b"a"), BlockKey(b"b"))
    core._joined_completions[777] = (set(keys), {})
    core._complete(777, {keys[0]: OpEntryResult(OpEntryStatus.SUCCESS)})
    assert not payloads(recording, "op.completion_queued")
    core._complete(777, {keys[1]: OpEntryResult(OpEntryStatus.FAILED)})
    assert len(payloads(recording, "op.completion_queued")) == 1
    assert len(list(kvcr.poll_completed())) == 1
    event = payloads(recording, "op.completion_returned")[0]
    assert int(event["status"]) == int(_nvtx.Status.PARTIAL)
    assert int(event["completed_blocks"]) == 1


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


def test_negative_fill_handle_and_copy_do_not_duplicate_completion(recording):
    test_kvcr_remote_target.test_remote_fetch_timeout_keeps_slot_until_source_is_terminal(
        "guard"
    )
    queued = payloads(recording, "target.queued")
    assert len(queued) == 1
    assert int(queued[0]["op_handle"]) < 0
    assert int(queued[0]["local_fill"]) == 1
    assert len(payloads(recording, "target.quiesced")) == 1


def test_missing_hint_has_specific_reason_and_failed_dispatch_cleans_up(
    recording, monkeypatch
):
    kvcr, _, _ = target()
    handle = kvcr.deliver({BlockKey(b"k"): [_mem_descriptor()]})
    result = dict(kvcr.poll_completed())
    assert not result[handle][BlockKey(b"k")].success
    assert int(payloads(recording, "op.completion_returned")[0]["reason"]) == int(
        _nvtx.Reason.HINT_UNAVAILABLE
    )

    def reject(*a, **kw):
        raise ValueError("test dispatch error")

    monkeypatch.setattr(kvcr._core._remote_fw_dram, "deliver", reject)
    with pytest.raises(ValueError, match="test dispatch error"):
        kvcr.deliver({BlockKey(b"k"): [_mem_descriptor()]})
    assert not kvcr._core._nvtx_operations
    assert not recording.stack


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
