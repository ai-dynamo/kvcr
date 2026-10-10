# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compact annotations retain identity without doing optional work at low."""

import threading
from types import SimpleNamespace

import pytest
import test_nvtx
from test_nvtx import payloads

from kvcr import _nvtx

recording = test_nvtx.recording


def test_registered_handle_is_not_passed_to_string_only_attributes_factory(
    recording, monkeypatch
):
    # Domain.get_event_attributes takes a string and registers it internally;
    # get_registered_string returns a separate opaque handle type.
    original = recording.get_event_attributes
    monkeypatch.setattr(
        recording, "get_registered_string", lambda name: SimpleNamespace(name=name)
    )

    def attributes(**kwargs):
        if not isinstance(kwargs["message"], str):
            raise TypeError("expected str, not RegisteredString")
        return original(**kwargs)

    monkeypatch.setattr(recording, "get_event_attributes", attributes)
    trace = _nvtx.create_tracer().lifecycle(op_handle=41)
    trace.mark("source.write.completed", completed_blocks=1, status=2)
    assert len(payloads(recording, "request.context")) == 1
    assert len(payloads(recording, "source.write.completed")) == 1


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


@pytest.mark.parametrize("level,display", [("low", False), ("medium", True)])
def test_request_digest_is_always_available_but_display_is_optional(
    recording, monkeypatch, level, display
):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    value = "req-α-😀"
    _nvtx.create_tracer().lifecycle(request_id=value)
    context = payloads(recording, "request.context")[0]
    assert int(context["request_known"]) == 1
    assert (int(context["request_hi"]), int(context["request_lo"])) == _nvtx.identity(
        value
    )
    assert bool(context["request_display_known"]) == display
    if display:
        assert (
            context["value_utf8"].tobytes()[: int(context["length"])] == value.encode()
        )


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
        target_incarnation="inc",
        op_handle=41,
        op_id=("source", 9),
        requested_blocks=1,
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


def test_reentrant_emission_does_not_overwrite_outer_payload(recording):
    tracer = _nvtx.create_tracer()
    outer, inner = tracer.lifecycle(), tracer.lifecycle()
    original = recording.mark
    entering = False

    def reenter(attrs):
        nonlocal entering
        if not entering:
            entering = True
            inner.mark("nixl.done_observed", transfer_id=22)
            entering = False
        original(attrs)

    recording.mark = reenter
    outer.mark("nixl.done_observed", transfer_id=11)
    events = payloads(recording, "nixl.done_observed")
    assert [int(e["transfer_id"]) for e in events] == [22, 11]
    assert [int(e["trace_id"]) for e in events] == [inner.trace_id, outer.trace_id]


def test_concurrent_emission_owns_separate_payload_storage(recording):
    tracer = _nvtx.create_tracer()
    first, second = tracer.lifecycle(), tracer.lifecycle()
    barrier = threading.Barrier(2)
    original = recording.mark

    def overlap(attrs):
        barrier.wait(timeout=5)
        original(attrs)

    recording.mark = overlap
    threads = [
        threading.Thread(
            target=trace.mark,
            args=("nixl.done_observed",),
            kwargs={"transfer_id": value},
        )
        for trace, value in ((first, 11), (second, 22))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    events = payloads(recording, "nixl.done_observed")
    assert {(int(e["trace_id"]), int(e["transfer_id"])) for e in events} == {
        (first.trace_id, 11),
        (second.trace_id, 22),
    }
