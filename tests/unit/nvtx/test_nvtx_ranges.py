# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Asynchronous ranges follow KVCR ownership, not synchronous call stacks."""

import threading

import pytest
from _nvtx_test_utils import make_source, spans, wait_for_payload

from kvcr import _nvtx


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


def test_source_close_ends_range_after_native_completion(recording):
    source, agent, pinning = make_source()
    agent.state = "PROC"
    pinning.complete(0)
    wait_for_payload(source, recording, "nixl.write.posted")
    agent.state = "DONE"
    source.close()
    assert len(spans(recording, "source.write")) == 1
    assert len(spans(recording, "source.write", "end")) == 1


def test_started_lifecycle_can_end_once_on_competing_threads(recording):
    trace = _nvtx.create_tracer().lifecycle()
    trace.start("source.write")
    # KVCR starts each range before publishing its work to another thread.
    assert len(recording.ranges) == 1
    barrier = threading.Barrier(2)

    def complete():
        barrier.wait(timeout=5)
        trace.end("source.write")

    workers = [threading.Thread(target=complete) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()
    assert recording.ended == [0]
    assert (
        spans(recording, "source.write")[0][3]
        != spans(recording, "source.write", "end")[0][3]
    )
    assert not recording.ranges


@pytest.mark.parametrize("failure", [None, "mark", "end"])
def test_source_completion_ends_once_despite_annotation_failure(recording, failure):
    trace = _nvtx.create_tracer().lifecycle()
    trace.start("source.write")
    recording.fail = failure
    trace.complete_source(True, 0)
    recording.fail = None
    trace.complete_source(True, 0)
    assert recording.ended == [0]
    assert not recording.ranges
    assert len(spans(recording, "source.write.completed", "mark")) == (
        failure != "mark"
    )
