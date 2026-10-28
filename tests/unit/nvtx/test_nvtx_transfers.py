# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Observe NIXL state and resource release without changing their ordering."""

import ctypes
import threading

import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _buffer_region,
    _decode_notif,
    _mem_descriptor,
    _new_kvcr,
    _poll_until,
    _start_write_message,
    _wait_until,
)
from _nvtx_test_utils import make_source, payloads, spans, wait_for_payload
from test_progress import _mem, _transfer_progress, _TransferAgent

from kvcr import _nvtx
from kvcr.config import LocalDramOptions
from kvcr.types import BlockKey


def test_zero_selected_source_is_failed_without_posting(recording):
    source, agent, pinning = make_source()
    pinning.complete(0, missing_indices=(0,))
    event = wait_for_payload(
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
@pytest.mark.parametrize(
    "value,digest",
    [
        (None, "00000000000000000000000000000000"),
        ("", "cae66941d9efbd404e4d88758ea67670"),
        ("req-α-😀\x00x", "a5a647304e8f1459a3254869eb1714f3"),
        # These share a display prefix, but their full-string hashes differ.
        ("😀" * 100, "d10f2d89f1ec5db3c77d648439d6dfc3"),
        ("😀" * 100 + "x", "fcbe41c4b94439e7b75a2dec6d779afc"),
    ],
)
def test_context_mapping_preserves_identity_and_bounded_utf8(
    recording, monkeypatch, value, digest, level
):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    trace = _nvtx.create_tracer().lifecycle(request_id=value)
    trace.mark("op.deliver")
    context = payloads(recording, "request.context")[0]
    event = payloads(recording, "op.deliver")[0]
    assert int(context["request_known"]) == (value is not None)
    assert f"{int(event['request_hi']):016x}{int(event['request_lo']):016x}" == digest
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
    source, agent, pinning = make_source()
    agent.state = "PROC"
    pinning.complete(0)
    backend = source._core._remote_fw_dram
    wait_for_payload(source, recording, "nixl.write.posted")
    source._core._clock = lambda: float("inf")
    wait_for_payload(source, recording, "source.write.cancel_requested")
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
    final = wait_for_payload(
        source, recording, "source.write.completed", ended="source.write"
    )
    assert int(final["reason"]) == int(_nvtx.Reason.DEADLINE)
    assert not backend._sources._pending_pin_ops
    for name in ("source.write", "nixl.write"):
        assert len(spans(recording, name, "end")) == 1


@pytest.mark.parametrize("level", ["low", "medium"])
def test_source_path_connects_pin_to_transfer(recording, monkeypatch, level):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    source, agent, pinning = make_source()
    pending = next(iter(source._core._remote_fw_dram._sources._source_pin_ops.values()))
    state = pending.nvtx_state
    native_submit = agent.transfer
    open_at_submit = []

    def submit_with_started_ranges(handle):
        op = source._core._progress._in_flight_ops[pending.op_id]
        assert op.nvtx_state is state
        assert state.trace_initialized and state.trace is not None
        open_at_submit.append({name for name, _ in recording.ranges.values()})
        return native_submit(handle)

    monkeypatch.setattr(agent, "transfer", submit_with_started_ranges)
    agent.state = "DONE"
    pinning.complete(0)
    _poll_until(source, lambda _: pinning.unpins == ["pin"])
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
    source, agent, pinning = make_source()
    backend = source._core._remote_fw_dram
    if "route" in cause:
        pending = next(iter(backend._sources._source_pin_ops.values()))
        backend._route_generation[pending.route[0]] = pending.route[1] + 1
    if "timeout" in cause:
        source._core._clock = lambda: float("inf")
    else:
        pinning.complete(0)
    event = wait_for_payload(
        source, recording, "source.write.completed", ended="source.write"
    )
    assert int(event["status"]) == 4
    assert int(event["reason"]) == reason
    assert int(event["completed_blocks"]) == int(event["completed_bytes"]) == 0
    assert not agent.xfers
    assert pinning.cancelled == ([0] if "timeout" in cause else [])
    assert pinning.unpins == ([] if "timeout" in cause else ["pin"])
    assert _decode_notif(agent.sent_notifs[0][1])["success"] is False
    assert not payloads(recording, "nixl.write.posted")
    assert len(spans(recording, "source.write", "end")) == 1


def test_route_refusal_preserves_cause_at_source_completion(recording, monkeypatch):
    from kvcr.backends.remote_fw_dram import _SourceWriteOp

    source, agent, pinning = make_source()
    backend = source._core._remote_fw_dram
    submit = source._core._progress.submit

    def replace_route(op):
        if isinstance(op, _SourceWriteOp):
            backend._route_generation[op.route[0]] = op.route[1] + 1
        submit(op)

    monkeypatch.setattr(source._core._progress, "submit", replace_route)
    agent.state = "DONE"
    pinning.complete(0)
    wait_for_payload(source, recording, "source.write.completed", ended="source.write")
    assert len(payloads(recording, "source.write.refused")) == 1
    assert not payloads(recording, "nixl.write.posted")
    assert int(payloads(recording, "source.write.completed")[0]["reason"]) == int(
        _nvtx.Reason.ROUTE_CHANGED
    )


def test_failed_source_trace_initialization_is_not_retried(recording, monkeypatch):
    source, agent, pinning = make_source()
    calls, polls = [], []
    tracer = source._core._remote_fw_dram._nvtx
    monkeypatch.setattr(
        tracer, "source", lambda *args: calls.append(threading.get_ident())
    )
    monkeypatch.setattr(
        agent, "check_xfer_state", lambda handle: polls.append(handle) or agent.state
    )
    agent.state = "PROC"
    pinning.complete(0)
    _poll_until(source, lambda _: len(polls) >= 2)
    agent.state = "DONE"
    _poll_until(source, lambda _: pinning.unpins == ["pin"])
    assert calls == [source._core._progress._thread.ident]
    assert len(agent.released_xfers) == 1
    assert not payloads(recording, "source.write.completed")


@pytest.mark.parametrize("level", ["off", "low", "medium"])
def test_local_source_reuses_nvtx_state_on_both_write_paths(
    recording, monkeypatch, level
):
    from kvcr.backends.remote_fw_dram import _SourceWriteOp
    from kvcr.backends.remote_sources import _SourcePinOp

    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    pins, writes = {}, {}
    for cls, states in ((_SourcePinOp, pins), (_SourceWriteOp, writes)):
        initialize = cls.__init__

        def capture(op, *args, initialize=initialize, states=states, **kwargs):
            initialize(op, *args, **kwargs)
            states[op.op_id] = op.nvtx_state

        monkeypatch.setattr(cls, "__init__", capture)
    memory = ctypes.create_string_buffer(16)
    agent, control = FakeNixlAgent(), FakeBytesControl()
    source = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        control,
        name="source",
        framework_regions=[_buffer_region(memory, 16)],
        local_dram=LocalDramOptions([("", ctypes.addressof(memory), 16)]),
    )
    agent.state = "DONE"
    key = BlockKey(b"local")
    source.deposit({key: [_mem_descriptor(end_point_name="source")]})
    _poll_until(source, bool)
    control.incoming.append(_start_write_message(12, key))
    _wait_until(lambda: len(writes) == 1)  # Inline progress needs no caller polling.
    _poll_until(source, lambda _: not source._core._local_dram_sources_by_op)
    monkeypatch.setattr(
        source._core._remote_fw_dram, "_try_local_source_write", lambda *args: False
    )
    control.incoming.append(_start_write_message(13, key))
    _poll_until(
        source,
        lambda _: len(writes) == 2 and not source._core._local_dram_sources_by_op,
    )
    assert len(agent.released_xfers) == 3
    assert pins.keys() == writes.keys() and len(pins) == 2
    for op_id, state in pins.items():
        assert writes[op_id] is state
        assert (state is None) == (level == "off")


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
    [done] = payloads(recording, "nixl.done_observed")
    [released] = payloads(recording, "nixl.write.released")
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
