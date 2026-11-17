# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batch dispatch preserves request-scoped hint tracing and rejection semantics."""

import msgspec
import pytest
from _kvcr_test_utils import (
    _mem_descriptor,
    _poll_until,
    _router_hint,
    _wait_until,
    _write_done_notification,
)
from _nvtx_test_utils import make_target, payloads, reason_of

from kvcr import _nvtx
from kvcr.types import BlockKey


@pytest.mark.parametrize("level", ["off", "low"])
def test_batch_hint_replacement_conflict_and_discard(recording, monkeypatch, level):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    kvcr, agent, control = make_target()
    hint = _router_hint("tcp://source:1", [1])
    second = _router_hint("tcp://source:1", [2]).actions[0]
    hint.actions.append(msgspec.structs.replace(second, action_id="a2"))
    kvcr.submit_hint(hint, "request")
    handle = kvcr.deliver({BlockKey(b"k"): [_mem_descriptor()]}, "request")
    _wait_until(lambda: bool(control.sent))
    agent.notifs["source"] = [_write_done_notification(handle)]
    assert handle in dict(_poll_until(kvcr, bool))
    kvcr.submit_hint(_router_hint("tcp://other:2", [3]), "request")
    kvcr.discard_hint("request")
    if level == "off":
        assert not recording.events
        return

    first, replacement = payloads(recording, "hint.submitted")
    [replaced] = payloads(recording, "hint.replaced")
    [rejected] = payloads(recording, "hint.rejected")
    [used] = payloads(recording, "hint.used")
    [discarded] = payloads(recording, "hint.discarded")
    assert int(first["trace_id"]) != int(replacement["trace_id"])
    assert int(replaced["trace_id"]) == int(first["trace_id"])
    for event in (used, rejected, discarded):
        assert int(event["trace_id"]) == int(replacement["trace_id"])
    assert int(used["op_handle"]) == handle
    assert reason_of(rejected) == _nvtx.Reason.HINT_CONFLICT
    assert int(rejected["status"]) == _nvtx.Status.REJECTED
