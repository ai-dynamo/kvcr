# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Observe NIXL state and resource release without changing their ordering."""

import pytest
import test_nvtx
from test_nvtx import payloads, spans
from test_progress import _mem, _transfer_progress, _TransferAgent

from kvcr import _nvtx

recording = test_nvtx.recording


def test_zero_selected_source_is_failed_without_posting(recording):
    source, agent, pinning = test_nvtx.make_source(recording)
    pinning.complete(0, missing_indices=(0,))
    event = test_nvtx.wait_for_payload(
        source, recording, "source.write.completed", ended="source.write"
    )
    assert int(event["status"]) == 4
    assert int(event["reason"]) == 0
    assert int(event["requested_blocks"]) == 1
    assert int(event["completed_blocks"]) == int(event["completed_bytes"]) == 0
    assert not agent.xfers
    assert len(spans(recording, "source.write")) == 1
    assert len(spans(recording, "source.write", "end")) == 1


@pytest.mark.parametrize(
    "success,requested,completed,status,reason",
    [
        (True, 0, 0, 2, 1),
        (True, 2, 0, 4, 0),
        (True, 2, 1, 3, 0),
        (True, 2, 2, 2, 1),
        (False, 2, 2, 4, 0),
    ],
)
def test_completion_status_and_counts(
    recording, success, requested, completed, status, reason
):
    tracer = _nvtx.create_tracer()
    tracer.begin(requested).finish(
        "success" if success else "failed", completed_blocks=completed
    )
    tracer.lifecycle(
        requested_blocks=requested, selected_bytes=completed * 16
    ).complete_source(success, completed)
    pin = payloads(recording, "source.pin.completed")[0]
    source = payloads(recording, "source.write.completed")[0]
    for event in (pin, source):
        assert int(event["status"]) == status
        assert int(event["reason"]) == reason
    assert int(pin["completed_blocks"]) == completed
    assert int(source["completed_blocks"]) == (completed if success else 0)
    assert int(source["completed_bytes"]) == (completed * 16 if success else 0)


@pytest.mark.parametrize("level", ["low", "medium"])
@pytest.mark.parametrize("value", [None, "", "req-α-😀\x00x", "😀" * 100])
def test_context_mapping_preserves_identity_and_bounded_utf8(
    recording, monkeypatch, value, level
):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    trace = _nvtx.create_tracer().lifecycle(request_id=value)
    trace.mark("op.deliver")
    context = payloads(recording, "request.context")[0]
    event = payloads(recording, "op.deliver")[0]
    assert int(context["request_known"]) == (value is not None)
    assert (int(event["request_hi"]), int(event["request_lo"])) == _nvtx.identity(value)
    display = level == "medium" and value is not None
    assert bool(context["request_display_known"]) == display
    encoded = value.encode("utf-8") if display else b""
    assert context["value_utf8"].tobytes()[: int(context["length"])] == encoded[:256]
    assert bool(context["truncated"]) == (len(encoded) > 256)
    assert int(event["session_known"]) == int(event["parent_session_known"]) == 0


def test_request_churn_does_not_evict_worker_identities(recording, monkeypatch):
    identify, calls = _nvtx.identity, []

    def counted(value):
        calls.append(value)
        return identify(value)

    monkeypatch.setattr(_nvtx, "identity", counted)
    tracer = _nvtx.create_tracer()
    tracer.lifecycle(target_agent="worker", target_incarnation="boot")
    for index in range(130):
        tracer.lifecycle(request_id=f"request-{index}")
    tracer.lifecycle(target_agent="worker", target_incarnation="boot")
    assert calls.count("worker") == calls.count("boot") == 1


def test_source_deadline_and_unresolved_close_are_not_native_completion(recording):
    source, agent, pinning = test_nvtx.make_source(recording)
    agent.state = "PROC"
    pinning.complete(0)
    backend = source._core._remote_fw_dram
    test_nvtx.wait_for_payload(source, recording, "nixl.write.posted")
    source._core._clock = lambda: float("inf")
    test_nvtx.wait_for_payload(source, recording, "source.write.cancel_requested")
    assert not payloads(recording, "nixl.done_observed")
    assert not payloads(recording, "source.write.completed")
    for name in ("source.write", "nixl.write"):
        assert len(spans(recording, name)) == 1
        assert not spans(recording, name, "end")
    op = next(
        o
        for o in source._core._progress._in_flight_ops.values()
        if o.op_id[0] == "source"
    )
    assert not op.close(source._core._progress)
    assert len(payloads(recording, "source.write.shutdown_unresolved")) == 1
    for name in ("source.write", "nixl.write"):
        assert not spans(recording, name, "end")
    agent.state = "DONE"
    final = test_nvtx.wait_for_payload(
        source, recording, "source.write.completed", ended="source.write"
    )
    assert int(final["reason"]) == int(_nvtx.Reason.DEADLINE)
    assert not backend._pending_pin_ops
    for name in ("source.write", "nixl.write"):
        assert len(spans(recording, name, "end")) == 1


@pytest.mark.parametrize("level", ["low", "medium"])
def test_source_path_connects_pin_to_transfer(recording, monkeypatch, level):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    source, agent, pinning = test_nvtx.make_source(recording)
    native_submit = agent.transfer
    open_at_submit = []

    def submit_with_started_ranges(handle):
        open_at_submit.append({name for name, _ in recording.ranges.values()})
        return native_submit(handle)

    monkeypatch.setattr(agent, "transfer", submit_with_started_ranges)
    agent.state = "DONE"
    pinning.complete(0)
    test_nvtx._poll_until(source, lambda _: pinning.unpins == ["pin"])
    assert open_at_submit == [{"source.write", "nixl.write"}]
    pin = payloads(recording, "source.pin.completed")[0]
    submitted = payloads(recording, "nixl.write.posted")
    assert len(submitted) == 1
    assert int(submitted[0]["source_op_id"]) == int(pin["source_op_id"])
    assert int(submitted[0]["instance_hi"]) == int(pin["instance_hi"])
    assert int(submitted[0]["op_handle"]) == 9
    assert int(submitted[0]["selected_blocks"]) == 1
    if level == "medium":
        assert int(submitted[0]["selected_bytes"]) > 0
    else:
        assert int(submitted[0]["selected_bytes"]) == -1
    assert len(payloads(recording, "source.write.completed")) == 1
    for name in ("source.write", "nixl.write"):
        assert (
            spans(recording, name)[0][3]
            == spans(recording, name, "end")[0][3]
            == source._core._progress._thread.ident
        )


@pytest.mark.parametrize(
    "cause,reason", [("timeout", 5), ("route", 17), ("timeout+route", 5)]
)
def test_pending_pin_failure_preserves_source_cause(recording, cause, reason):
    source, agent, pinning = test_nvtx.make_source(recording)
    backend = source._core._remote_fw_dram
    if "route" in cause:
        pending = next(iter(backend._source_pin_ops.values()))
        backend._route_generation[pending.route[0]] = pending.route[1] + 1
    if "timeout" in cause:
        source._core._clock = lambda: float("inf")
    else:
        pinning.complete(0)
    event = test_nvtx.wait_for_payload(
        source, recording, "source.write.completed", ended="source.write"
    )
    assert int(event["status"]) == 4
    assert int(event["reason"]) == reason
    assert int(event["completed_blocks"]) == int(event["completed_bytes"]) == 0
    assert not agent.xfers
    assert pinning.cancelled == ([0] if "timeout" in cause else [])
    assert pinning.unpins == ([] if "timeout" in cause else ["pin"])
    assert test_nvtx._decode_notif(agent.sent_notifs[0][1])["success"] is False
    assert not payloads(recording, "nixl.write.posted")
    assert len(spans(recording, "source.write", "end")) == 1


def test_route_refusal_preserves_cause_at_source_completion(recording, monkeypatch):
    from kvcr.remote_fw_dram import _SourceWriteOp

    source, agent, pinning = test_nvtx.make_source(recording)
    backend = source._core._remote_fw_dram
    submit = source._core._progress.submit

    def replace_route(op):
        if isinstance(op, _SourceWriteOp):
            backend._route_generation[op.route[0]] = op.route[1] + 1
        submit(op)

    monkeypatch.setattr(source._core._progress, "submit", replace_route)
    agent.state = "DONE"
    pinning.complete(0)
    test_nvtx.wait_for_payload(
        source, recording, "source.write.completed", ended="source.write"
    )
    assert len(payloads(recording, "source.write.refused")) == 1
    assert not payloads(recording, "nixl.write.posted")
    assert int(payloads(recording, "source.write.completed")[0]["reason"]) == int(
        _nvtx.Reason.ROUTE_CHANGED
    )


def setup_transfer():
    agent = _TransferAgent()
    progress = _transfer_progress(agent)
    trace = _nvtx.create_tracer().lifecycle(
        op_handle=-9, source_op_id=2, target_agent="target-α", requested_blocks=2
    )
    return agent, progress, trace


def submit(progress, trace):
    return progress.submit_transfer(
        "WRITE",
        [_mem(0)],
        [_mem(1, owner="remote-agent")],
        remote_side_agent="remote-agent",
        trace=trace,
    )


@pytest.mark.parametrize("post", ["PROC", "DONE"])
def test_native_done_and_release_are_distinct_once(recording, post):
    agent, progress, trace = setup_transfer()
    agent.transfer_result = post
    transfer, accepted = submit(progress, trace)
    assert accepted
    assert len(spans(recording, "nixl.write")) == 1
    assert len(payloads(recording, "nixl.done_observed")) == (post == "DONE")
    agent.release_failures = 1
    assert progress.poll_transfer(transfer, require_completion=True) is None
    assert transfer in progress._active_transfers
    assert len(payloads(recording, "nixl.done_observed")) == 1
    assert not spans(recording, "nixl.write", "end")
    assert not payloads(recording, "nixl.write.released")
    assert progress.poll_transfer(transfer, require_completion=True) == (True, None)
    assert transfer not in progress._active_transfers
    done = payloads(recording, "nixl.done_observed")[0]
    released = payloads(recording, "nixl.write.released")[0]
    assert int(done["transfer_id"]) == int(released["transfer_id"]) == transfer
    assert int(done["trace_id"]) == int(released["trace_id"])
    assert int(released["op_handle"]) == -9
    assert len(spans(recording, "nixl.write", "end")) == 1
    assert not recording.ranges
    assert not recording.stack


@pytest.mark.parametrize("post", ["ERR", "unexpected", "exception"])
def test_rejected_or_ambiguous_post_retains_native_handle(recording, post):
    agent, progress, trace = setup_transfer()
    agent.transfer_result = post
    agent.submit_exception = post == "exception"
    transfer, accepted = submit(progress, trace)
    assert not accepted
    posted = payloads(recording, "nixl.write.posted")[0]
    assert int(posted["status"]) == int(
        _nvtx.Status.REJECTED if post == "ERR" else _nvtx.Status.AMBIGUOUS
    )
    agent.state = "PROC"
    assert progress.poll_transfer(transfer, require_completion=True) is None
    assert transfer in progress._active_transfers
    assert not payloads(recording, "nixl.write.released")
    agent.state = "DONE"
    result = progress.poll_transfer(transfer, require_completion=True)
    assert result == (post != "ERR", None)
    assert len(payloads(recording, "nixl.done_observed")) == 1
    assert len(payloads(recording, "nixl.write.released")) == 1


def test_poll_error_and_later_native_done_are_both_observed(recording):
    agent, progress, trace = setup_transfer()
    transfer, _ = submit(progress, trace)
    agent.check_exception = True
    for _ in range(3):
        assert progress.poll_transfer(transfer, require_completion=True) is None
    assert len(payloads(recording, "nixl.write.error")) == 1
    assert transfer in progress._active_transfers
    agent.check_exception = False
    assert progress.poll_transfer(transfer, require_completion=True) == (False, None)
    assert len(payloads(recording, "nixl.done_observed")) == 1
    assert not recording.stack


def test_creation_failure_closes_scope_and_reports_rejection(recording):
    agent, progress, trace = setup_transfer()
    agent.make_prepped_xfer = lambda *a, **kw: None
    with pytest.raises(RuntimeError, match="creation returned None"):
        submit(progress, trace)
    assert not progress._active_transfers
    assert len(payloads(recording, "nixl.write.rejected")) == 1
    assert not recording.stack


@pytest.mark.parametrize("level,count", [("low", 0), ("medium", 1)])
@pytest.mark.parametrize("failure", ["false", "exception", "missing"])
def test_release_retry_detail_level(recording, monkeypatch, level, count, failure):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    agent, progress, trace = setup_transfer()
    transfer, _ = submit(progress, trace)
    release = agent.release_xfer_handle

    def fail(_handle):
        if failure == "exception":
            raise RuntimeError("release failed")
        return False

    agent.release_xfer_handle = None if failure == "missing" else fail
    assert progress.poll_transfer(transfer, require_completion=True) is None
    assert transfer in progress._active_transfers
    assert not spans(recording, "nixl.write", "end")
    assert len(payloads(recording, "nixl.write.release_retry")) == count
    agent.release_xfer_handle = release
    assert progress.poll_transfer(transfer, require_completion=True) == (True, None)
    assert len(spans(recording, "nixl.write", "end")) == 1
