# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batch dispatch preserves request-scoped hint tracing and rejection semantics."""

import msgspec
import pytest
from _kvcr_test_utils import _record_hint_results, _router_hint
from _nvtx_test_utils import make_target, payloads

from kvcr import _nvtx
from kvcr.hints.results import HintStatus


@pytest.mark.parametrize("level", ["off", "low", "medium"])
@pytest.mark.parametrize(
    "scenario", ["replacement", "conflict", "invalid", "unsupported"]
)
def test_batch_hint_traces(recording, monkeypatch, level, scenario):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    kvcr, _, _ = make_target()
    results = _record_hint_results(monkeypatch, kvcr)
    hint = _router_hint("tcp://source:1", [1])
    source = "tcp://other:2" if scenario == "conflict" else "tcp://source:1"
    second = _router_hint(source, [] if scenario == "invalid" else [2]).actions[0]
    action_type = "unsupported" if scenario == "unsupported" else second.action_type
    hint.actions.append(
        msgspec.structs.replace(second, action_id="a2", action_type=action_type)
    )
    kvcr.submit_hint(hint, "request")

    statuses = {result.action_id: result.status for result in results}
    assert statuses == {
        "a1": HintStatus.ACCEPTED,
        "a2": HintStatus.REJECTED
        if scenario in ("invalid", "unsupported")
        else HintStatus.ACCEPTED,
    }
    stored = kvcr._core._remote_fw_dram._request_hints["request"]
    assert stored.block_hashes == {2 if scenario == "replacement" else 1}
    assert stored.failed == (scenario == "conflict")
    kvcr.discard_hint("request")
    if level == "off":
        assert stored.trace is None
        assert not recording.events
        return

    submitted = payloads(recording, "hint.submitted")
    replaced = payloads(recording, "hint.replaced")
    rejected = payloads(recording, "hint.rejected")
    assert len(submitted) == 1 + (scenario == "replacement")
    assert len({int(event["trace_id"]) for event in submitted}) == len(submitted)
    assert len(replaced) == (scenario == "replacement")
    assert len(rejected) == (scenario == "conflict")
    for event in replaced + rejected:
        assert int(event["trace_id"]) == int(submitted[0]["trace_id"])
    if rejected:
        assert int(rejected[0]["reason"]) == _nvtx.Reason.HINT_CONFLICT
        assert int(rejected[0]["status"]) == _nvtx.Status.REJECTED
    [discarded] = payloads(recording, "hint.discarded")
    assert (
        int(discarded["trace_id"])
        == int(submitted[-1]["trace_id"])
        == stored.trace.trace_id
    )
