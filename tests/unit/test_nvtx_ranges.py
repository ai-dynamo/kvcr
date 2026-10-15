# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Asynchronous ranges follow KVCR ownership, not synchronous call stacks."""

import threading

import pytest
import test_nvtx
from test_nvtx import payloads
from test_nvtx_transfers import setup_transfer, submit

from kvcr import _nvtx

recording = test_nvtx.recording


def spans(domain, name, kind="start"):
    return [event for event in domain.events if event[0] == kind and event[1] == name]


def test_shared_physical_pin_has_one_range_and_multiple_waiters(recording):
    source, agent, pinning = test_nvtx.make_source(recording, (9, 10))
    assert len(spans(recording, "source.pin.wait")) == 1
    assert len(payloads(recording, "source.pin.waiter")) == 2
    assert not spans(recording, "source.pin.wait", "end")
    agent.state = "DONE"
    pinning.complete(0)
    test_nvtx._poll_until(source, lambda _: pinning.unpins == ["pin"])
    assert len(spans(recording, "source.pin.wait", "end")) == 1
    assert 0 in recording.ended  # A zero range ID must not be mistaken for failure.


def test_overlapping_pin_ranges_can_end_on_another_thread(recording):
    tracer = _nvtx.create_tracer()
    first, second = tracer.begin(1), tracer.begin(2)
    assert len(recording.ranges) == 2
    worker = threading.Thread(
        target=first.finish, args=("success",), kwargs={"completed_blocks": 1}
    )
    worker.start()
    worker.join(timeout=5)
    second.finish("failed")
    first.finish("cancelled")
    starts = spans(recording, "source.pin.wait")
    ends = spans(recording, "source.pin.wait", "end")
    assert len(starts) == len(ends) == 2
    assert starts[0][3] != ends[0][3]
    assert not recording.ranges


def test_native_range_survives_done_until_release_succeeds(recording):
    agent, progress, trace = setup_transfer()
    agent.transfer_result = "DONE"
    transfer, submitted = submit(progress, trace)
    assert submitted
    assert len(spans(recording, "nixl.write")) == 1
    agent.release_failures = 1
    assert progress.poll_transfer(transfer) is None
    assert not spans(recording, "nixl.write", "end")
    assert progress.poll_transfer(transfer) is not None
    assert len(spans(recording, "nixl.write", "end")) == 1
    assert not recording.ranges


def test_source_timeout_does_not_end_native_ownership_range(recording):
    source, agent, pinning = test_nvtx.make_source(recording)
    agent.state = "PROC"
    pinning.complete(0)
    test_nvtx._poll_until(
        source, lambda _: bool(payloads(recording, "nixl.write.posted"))
    )
    source._core._clock = lambda: float("inf")
    test_nvtx._poll_until(
        source, lambda _: bool(payloads(recording, "source.write.cancel_requested"))
    )
    try:
        assert len(spans(recording, "source.write")) == 1
        assert len(spans(recording, "nixl.write")) == 1
        assert not spans(recording, "nixl.write", "end")
        assert not spans(recording, "source.write", "end")
    finally:
        agent.state = "DONE"
    test_nvtx._poll_until(
        source, lambda _: bool(payloads(recording, "source.write.completed"))
    )
    assert len(spans(recording, "nixl.write", "end")) == 1
    assert len(spans(recording, "source.write", "end")) == 1


@pytest.mark.parametrize("failure", ["empty", "close"])
def test_terminal_source_paths_close_their_ranges(recording, failure):
    source, agent, pinning = test_nvtx.make_source(recording)
    if failure == "empty":
        pinning.complete(0, missing_indices=(0,))
    else:
        agent.state = "PROC"
        pinning.complete(0)
    test_nvtx._poll_until(
        source,
        lambda _: (
            bool(payloads(recording, "source.write.completed"))
            if failure == "empty"
            else bool(payloads(recording, "nixl.write.posted"))
        ),
    )
    if failure != "empty":
        agent.state = "DONE"
        source.close()
    assert len(spans(recording, "source.write")) == 1
    assert len(spans(recording, "source.write", "end")) == 1


@pytest.mark.parametrize(
    "restart,fail_start", [(False, False), (True, False), (True, True)]
)
def test_end_can_claim_inflight_start_without_blocking(recording, restart, fail_start):
    """A slow native start must not block completion or overwrite a later start."""
    tracer = _nvtx.create_tracer()
    trace = tracer.lifecycle()
    entered, release, ended = (threading.Event() for _ in range(3))
    original = recording.start_range
    calls = 0

    def delayed_start(attrs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release native start")
            if fail_start:
                raise RuntimeError("native start failed")
        return original(attrs)

    def finish_and_restart():
        trace.end("nixl.write")
        if restart:
            trace.start("nixl.write", transfer_id=22)
        ended.set()

    recording.start_range = delayed_start
    starting = threading.Thread(target=trace.start, args=("nixl.write",))
    finishing = threading.Thread(target=finish_and_restart)
    starting.start()
    try:
        assert entered.wait(5)
        finishing.start()
        completed_while_start_blocked = ended.wait(1)
    finally:
        release.set()
        starting.join(timeout=5)
        if finishing.ident is not None:
            finishing.join(timeout=5)
    assert not starting.is_alive() and not finishing.is_alive()
    assert completed_while_start_blocked, "end waited for the native start call"
    if restart:
        # The replacement starts first and receives native ID zero. A late
        # return or failure from the old start must not remove this new range.
        assert list(recording.ranges) == [0]
        assert int(recording.ranges[0][1]["transfer_id"]) == 22
        trace.end("nixl.write")
    assert not recording.ranges
    assert len(recording.ended) == (1 + int(restart) - int(fail_start))


def test_duplicate_start_during_native_call_has_one_range(recording):
    trace = _nvtx.create_tracer().lifecycle()
    original = recording.start_range

    def reenter(attrs):
        trace.start("nixl.write")
        return original(attrs)

    recording.start_range = reenter
    trace.start("nixl.write")
    trace.end("nixl.write")
    trace.end("nixl.write")
    assert len(spans(recording, "nixl.write")) == 1
    assert recording.ended == [0]


def test_reentrant_end_closes_native_start_once(recording):
    trace = _nvtx.create_tracer().lifecycle()
    original = recording.start_range

    def reenter(attrs):
        trace.end("nixl.write")
        return original(attrs)

    recording.start_range = reenter
    trace.start("nixl.write")
    trace.end("nixl.write")
    assert recording.ended == [0]
    assert not recording.ranges
