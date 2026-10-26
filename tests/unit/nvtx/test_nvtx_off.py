# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Disabled tracing must skip diagnostic work without skipping validation."""

import pytest
from _kvcr_test_utils import (
    _mem_descriptor,
    _poll_until,
    _router_hint,
    _wait_until,
    _write_done_notification,
)
from test_nvtx_target import target

from kvcr import _nvtx
from kvcr.types import BlockKey


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
    kvcr, agent, control = target()
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
