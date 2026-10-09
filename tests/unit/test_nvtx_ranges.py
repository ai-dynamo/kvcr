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
