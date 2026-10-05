# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Observe NIXL state and resource release without changing their ordering."""

import pytest
import test_nvtx
from test_nvtx import payloads
from test_progress import _mem, _transfer_progress, _TransferAgent

from kvcr import _nvtx

recording = test_nvtx.recording


@pytest.mark.parametrize("value", [None, "", "req-α-😀\x00x", "😀" * 100])
def test_context_mapping_preserves_identity_and_bounded_utf8(recording, value):
    trace = _nvtx.create_tracer().lifecycle(request_id=value)
    trace.mark("op.deliver")
    context = payloads(recording, "request.context")[0]
    event = payloads(recording, "op.deliver")[0]
    assert int(context["request_known"]) == (value is not None)
    assert (int(event["request_hi"]), int(event["request_lo"])) == _nvtx.identity(value)
    encoded = b"" if value is None else value.encode("utf-8")
    assert context["value_utf8"].tobytes()[: int(context["length"])] == encoded[:256]
    assert bool(context["truncated"]) == (len(encoded) > 256)
    assert int(event["session_known"]) == int(event["parent_session_known"]) == 0


def test_source_deadline_and_unresolved_close_are_not_native_completion(recording):
    source, agent, pinning = test_nvtx.make_source(recording)
    agent.state = "PROC"
    pinning.complete(0)
    backend = source._core._remote_fw_dram
    test_nvtx._poll_until(
        source, lambda _: bool(payloads(recording, "nixl.write.posted"))
    )
    source._core._clock = lambda: float("inf")
    test_nvtx._poll_until(
        source, lambda _: bool(payloads(recording, "source.write.cancel_requested"))
    )
    assert not payloads(recording, "nixl.done_observed")
    assert not payloads(recording, "source.write.completed")
    op = next(
        o
        for o in source._core._progress._in_flight_ops.values()
        if o.op_id[0] == "source"
    )
    assert not op.close(source._core._progress)
    assert len(payloads(recording, "source.write.shutdown_unresolved")) == 1
    agent.state = "DONE"
    test_nvtx._poll_until(
        source, lambda _: bool(payloads(recording, "source.write.completed"))
    )
    final = payloads(recording, "source.write.completed")[0]
    assert int(final["reason"]) == int(_nvtx.Reason.DEADLINE)
    assert not backend._pending_pin_ops


def test_source_path_connects_pin_to_transfer(recording):
    source, agent, pinning = test_nvtx.make_source(recording)
    agent.state = "DONE"
    pinning.complete(0)
    test_nvtx._poll_until(source, lambda _: pinning.unpins == ["pin"])
    pin = payloads(recording, "source.pin.completed")[0]
    submitted = payloads(recording, "nixl.write.posted")
    assert len(submitted) == 1
    assert int(submitted[0]["source_op_id"]) == int(pin["source_op_id"])
    assert int(submitted[0]["instance_hi"]) == int(pin["instance_hi"])
    assert int(submitted[0]["op_handle"]) == 9
    assert int(submitted[0]["selected_blocks"]) == 1
    assert int(submitted[0]["selected_bytes"]) > 0
    assert len(payloads(recording, "source.write.completed")) == 1


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
    test_nvtx._poll_until(
        source, lambda _: bool(payloads(recording, "source.write.completed"))
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
    assert len(payloads(recording, "nixl.done_observed")) == (post == "DONE")
    agent.release_failures = 1
    assert progress.poll_transfer(transfer, require_completion=True) is None
    assert transfer in progress._active_transfers
    assert len(payloads(recording, "nixl.done_observed")) == 1
    assert not payloads(recording, "nixl.write.released")
    assert progress.poll_transfer(transfer, require_completion=True) == (True, None)
    assert transfer not in progress._active_transfers
    done = payloads(recording, "nixl.done_observed")[0]
    released = payloads(recording, "nixl.write.released")[0]
    assert int(done["transfer_id"]) == int(released["transfer_id"]) == transfer
    assert int(done["trace_id"]) == int(released["trace_id"])
    assert int(released["op_handle"]) == -9
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


@pytest.mark.parametrize("failure", ["attributes", "push", "mark", "pop"])
def test_annotation_failure_cannot_change_nixl_lifecycle(recording, failure):
    agent, progress, trace = setup_transfer()
    recording.fail = failure
    transfer, accepted = submit(progress, trace)
    assert accepted
    assert progress.poll_transfer(transfer, require_completion=True) == (True, None)
    assert agent.events[-1] == f"release-transfer:{transfer}"
    assert not recording.stack


@pytest.mark.parametrize("level,count", [("low", 0), ("medium", 1)])
def test_release_retry_detail_level(recording, monkeypatch, level, count):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    agent, progress, trace = setup_transfer()
    transfer, _ = submit(progress, trace)
    agent.release_failures = 1
    assert progress.poll_transfer(transfer, require_completion=True) is None
    assert len(payloads(recording, "nixl.write.release_retry")) == count
    assert progress.poll_transfer(transfer, require_completion=True) == (True, None)
