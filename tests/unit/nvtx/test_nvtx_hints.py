# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batch dispatch preserves request-scoped hint tracing and rejection semantics."""

import msgspec
import pytest
from _kvcr_test_utils import (
    _mem_descriptor,
    _poll_until,
    _record_hint_results,
    _router_hint,
    _wait_until,
    _write_done_notification,
)
from _nvtx_test_utils import make_target, payloads

from kvcr import _nvtx
from kvcr.hints.results import HintStatus
from kvcr.types import BlockKey


@pytest.mark.parametrize("level", ["off", "low"])
@pytest.mark.parametrize("scenario", ["replacement", "invalid", "unsupported"])
def test_batch_hint_traces(recording, monkeypatch, level, scenario):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    kvcr, agent, control = make_target()
    results = _record_hint_results(monkeypatch, kvcr)
    hint = _router_hint("tcp://source:1", [1])
    second = _router_hint(
        "tcp://source:1", [] if scenario == "invalid" else [2]
    ).actions[0]
    action_type = "unsupported" if scenario == "unsupported" else second.action_type
    hint.actions.append(
        msgspec.structs.replace(second, action_id="a2", action_type=action_type)
    )
    kvcr.submit_hint(hint, "request")

    statuses = {result.action_id: result.status for result in results}
    assert statuses == {
        "a1": HintStatus.ACCEPTED,
        "a2": HintStatus.ACCEPTED if scenario == "replacement" else HintStatus.REJECTED,
    }
    if scenario == "replacement":
        for expected in (1, 2):
            handle = kvcr.deliver({BlockKey(b"k"): [_mem_descriptor()]}, "request")
            _wait_until(lambda: len(control.sent) == expected)
            agent.notifs["source"] = [_write_done_notification(handle)]
            assert handle in dict(_poll_until(kvcr, bool))
        assert len(payloads(recording, "hint.used")) == (0 if level == "off" else 2)
        assert not kvcr._core._remote_fw_dram._request_hints["request"].failed
        kvcr.submit_hint(_router_hint("tcp://other:2", [3]), "request")
        assert [result.status for result in results] == [HintStatus.ACCEPTED] * 3
        failed = kvcr.deliver({BlockKey(b"k"): [_mem_descriptor()]}, "request")
        assert not dict(kvcr.poll_completed())[failed][BlockKey(b"k")].success
        assert len(control.sent) == 2
    stored = kvcr._core._remote_fw_dram._request_hints["request"]
    assert stored.block_hashes == {2 if scenario == "replacement" else 1}
    assert stored.failed == (scenario == "replacement")
    kvcr.discard_hint("request")
    if level == "off":
        assert stored.trace is None
        assert not recording.events
        return

    submitted = payloads(recording, "hint.submitted")
    replaced = payloads(recording, "hint.replaced")
    rejected = payloads(recording, "hint.rejected")
    used = payloads(recording, "hint.used")
    assert len(submitted) == 1 + (scenario == "replacement")
    assert len({int(event["trace_id"]) for event in submitted}) == len(submitted)
    assert len(replaced) == (scenario == "replacement")
    assert len(rejected) == (scenario == "replacement")
    assert len(used) == (2 if scenario == "replacement" else 0)
    assert len({int(event["op_handle"]) for event in used}) == len(used)
    for event in replaced:
        assert int(event["trace_id"]) == int(submitted[0]["trace_id"])
    for event in used + rejected:
        assert int(event["trace_id"]) == int(submitted[-1]["trace_id"])
    if rejected:
        assert int(rejected[0]["reason"]) == _nvtx.Reason.HINT_CONFLICT
        assert int(rejected[0]["status"]) == _nvtx.Status.REJECTED
    [discarded] = payloads(recording, "hint.discarded")
    assert (
        int(discarded["trace_id"])
        == int(submitted[-1]["trace_id"])
        == stored.trace.trace_id
    )
