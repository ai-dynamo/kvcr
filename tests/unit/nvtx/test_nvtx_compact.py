# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compact annotations retain identity without doing optional work at low."""

import threading
from types import SimpleNamespace

import pytest
from _nvtx_test_utils import payloads

from kvcr import _nvtx


def test_compact_fields_preserve_types_and_values(recording):
    tracer = _nvtx.create_tracer()
    fields = dict(op_handle=-(2**63), transfer_id=2**64 - 1, value_utf8=255, status=4)
    tracer._emit("typed", 7, fields, method=recording.mark)
    payload = recording.events[0][2]
    assert set(payload.dtype.names) == {
        "schema_version",
        "instance_hi",
        "instance_lo",
        "trace_id",
        "status",
        "reason",
        "op_handle",
        "transfer_id",
        "value_utf8",
    }
    for name, dtype in (
        ("schema_version", "u2"),
        ("trace_id", "u8"),
        ("status", "u1"),
        ("op_handle", "i8"),
        ("transfer_id", "u8"),
        ("value_utf8", ("u1", (256,))),
    ):
        assert payload.dtype[name] == tracer.numpy.dtype(dtype)
    assert int(payload["op_handle"]) == -(2**63)
    assert int(payload["transfer_id"]) == 2**64 - 1
    assert int(payload["status"]) == 4
    assert payload["value_utf8"].tobytes() == b"\xff" * 256
    with pytest.raises(KeyError):
        tracer._emit("typed", 8, dict(unknown_field=1), method=recording.mark)
    assert len(recording.events) == 1


def test_immutable_context_is_not_repeated_on_transitions(recording):
    trace = _nvtx.create_tracer().lifecycle(
        request_id="req-α",
        target_agent="target",
        target_incarnation="inc",
        op_handle=41,
        requested_blocks=7,
    )
    trace.mark("source.write.completed", status=2, completed_blocks=7)
    trace.mark("nixl.done_observed", transfer_id=8)
    contexts = payloads(recording, "request.context")
    assert len(contexts) == 1
    assert int(contexts[0]["op_handle"]) == 41
    assert int(contexts[0]["requested_blocks"]) == 7
    for _, name, raw, _ in recording.events:
        if name == "request.context":
            continue
        assert "request_hi" not in raw.dtype.names
        assert "target_agent_hi" not in raw.dtype.names
        assert "requested_blocks" not in raw.dtype.names
    result = payloads(recording, "source.write.completed")[0]
    assert int(result["op_handle"]) == 41
    assert int(result["completed_blocks"]) == 7


@pytest.mark.parametrize("level,scanned", [("low", False), ("medium", True)])
def test_low_avoids_descriptor_diagnostics(recording, monkeypatch, level, scanned):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    calls = []
    monkeypatch.setattr(_nvtx, "tier", lambda refs: calls.append("tier") or 1)
    monkeypatch.setattr(
        _nvtx, "memory_kind", lambda refs, regions: calls.append("memory") or 1
    )
    kvcr = SimpleNamespace(
        _descriptor_bytes=lambda refs: calls.append("bytes") or 16,
        _memory_regions=({}, {}),
    )
    op = SimpleNamespace(
        src_descriptors=((object(),),),
        dst_descriptors=((object(),),),
        route=("target", 1),
        op_handle=41,
        op_id=("source", 9),
        nvtx_state=SimpleNamespace(target_incarnation="inc", requested_blocks=1),
        source_keys=(b"key",),
    )
    trace = _nvtx.create_tracer().source(op, kvcr)
    assert trace is not None
    assert bool(calls) == scanned
    context = payloads(recording, "request.context")[0]
    assert int(context["selected_bytes"]) == (16 if scanned else -1)


def test_repeated_events_reuse_attributes_and_preserve_snapshots(recording):
    trace = _nvtx.create_tracer().lifecycle(op_handle=4)
    trace.mark("nixl.done_observed", transfer_id=1)
    recording.fail = "attributes"
    trace.mark("nixl.done_observed", transfer_id=2)
    events = payloads(recording, "nixl.done_observed")
    assert [int(e["transfer_id"]) for e in events] == [1, 2]


@pytest.mark.parametrize("context", [False, True], ids=["compact", "context"])
def test_reentrant_emission_does_not_overwrite_outer_payload(recording, context):
    tracer = _nvtx.create_tracer()
    name = "request.context" if context else "nixl.done_observed"
    field = "requested_blocks" if context else "transfer_id"
    original = recording.mark
    entering = False

    def reenter(attrs):
        nonlocal entering
        if not entering:
            entering = True
            tracer._emit(
                name,
                22,
                {} if context else {field: 22},
                context=context,
                method=tracer.domain.mark,
            )
            entering = False
        original(attrs)

    recording.mark = reenter
    tracer._emit(name, 11, {field: 11}, context=context, method=tracer.domain.mark)
    assert [(int(p["trace_id"]), int(p[field])) for _, _, p, _ in recording.events] == [
        (22, -1 if context else 22),
        (11, 11),
    ]


@pytest.mark.parametrize("context", [False, True], ids=["compact", "context"])
def test_concurrent_emission_owns_separate_payload_storage(recording, context):
    tracer = _nvtx.create_tracer()
    barrier = threading.Barrier(2)
    original = recording.mark

    def overlap(attrs):
        barrier.wait(timeout=5)
        original(attrs)

    def annotate(value):
        if context:
            tracer.lifecycle(**({"requested_blocks": value} if value == 11 else {}))
        else:
            trace = tracer.begin(value)
            trace.registered(value)
            trace.finish("success", completed_blocks=value)

    recording.mark = overlap
    threads = [threading.Thread(target=annotate, args=(value,)) for value in (11, 22)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    events = payloads(
        recording, "request.context" if context else "source.pin.completed"
    )
    assert len(events) == len({int(p["trace_id"]) for p in events}) == 2
    assert {int(p["requested_blocks"]) for p in events} == {11, -1 if context else 22}
    if not context:
        assert all(
            p["pin_request_id"] == p["completed_blocks"] == p["requested_blocks"]
            for p in events
        )


def test_reused_context_clears_optional_values_without_rebinding(recording):
    tracer = _nvtx.create_tracer()
    tracer._emit(
        "request.context",
        11,
        dict(
            requested_blocks=16,
            requested_bytes=65536,
            hint_trace_id=123,
            local_blocks=1,
            remote_blocks=2,
            g3_blocks=3,
            completed_blocks=8,
            completed_bytes=32768,
            request_hi=12,
            request_known=1,
            pin_id=21,
            pin_request_known=1,
            pin_request_id=22,
            request_display_known=1,
            length=256,
            truncated=1,
            value_utf8=99,
            status=_nvtx.Status.FAILED,
            reason=_nvtx.Reason.CALLBACK_ERROR,
        ),
        context=True,
        method=tracer.domain.mark,
    )
    # Rebinding would fail; the existing attributes and storage must be reused.
    recording.fail = "attributes"
    tracer._emit("request.context", 12, {}, context=True, method=tracer.domain.mark)
    first, second = [e[2] for e in recording.events]
    assert [int(p["trace_id"]) for p in (first, second)] == [11, 12]
    assert tuple(
        int(first[name])
        for name in ("hint_trace_id", "local_blocks", "remote_blocks", "g3_blocks")
    ) == (123, 1, 2, 3)
    assert int(first["requested_bytes"]) == 65536
    assert first["value_utf8"].tobytes() == b"c" * 256
    assert int(second["schema_version"]) == 3
    assert int(second["instance_hi"]) == tracer.instance_hi
    assert int(second["instance_lo"]) == tracer.instance_lo
    for fields, expected in (
        (
            "requested_blocks requested_bytes selected_blocks selected_bytes "
            "completed_blocks completed_bytes local_blocks remote_blocks g3_blocks",
            -1,
        ),
        (
            "request_hi request_known pin_id pin_request_known pin_request_id "
            "request_display_known length truncated hint_trace_id",
            0,
        ),
    ):
        assert all(int(second[field]) == expected for field in fields.split())
    assert second["value_utf8"].tobytes() == bytes(256)
    assert int(second["status"]) == _nvtx.Status.PENDING
    assert int(second["reason"]) == _nvtx.Reason.NONE


def test_context_recovers_after_partial_assignment_failure(recording):
    tracer = _nvtx.create_tracer()
    with pytest.raises((TypeError, ValueError)):
        tracer._emit(
            "request.context",
            11,
            dict(request_known=1, value_utf8=99, requested_blocks="not-an-integer"),
            context=True,
            method=tracer.domain.mark,
        )
    recording.fail = "attributes"
    tracer._emit("request.context", 12, {}, context=True, method=tracer.domain.mark)
    [(_, _, payload, _)] = recording.events
    assert int(payload["trace_id"]) == 12
    assert int(payload["request_known"]) == 0
    assert int(payload["requested_blocks"]) == -1
    assert payload["value_utf8"].tobytes() == bytes(256)


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
