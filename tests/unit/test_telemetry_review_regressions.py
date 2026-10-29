# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Capture boundary and started-target error regressions from independent review."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kvcr import compact_telemetry as telemetry
from kvcr import operation_capture as capture
from kvcr.remote_ops import _TargetPullOp, _TargetPullState


@pytest.mark.parametrize("headroom", [100, 200])
def test_journal_output_respects_reader_limit(tmp_path, headroom):
    output = telemetry.BufferedWriter(
        tmp_path, source_archive_sha256="test", start_thread=False
    )
    admitted = output.journal(
        dict(event="source_get", key=b"x" * (telemetry.BUFFER_BYTES - headroom))
    )
    output.close()
    footer = json.loads(output.path.read_text().splitlines()[-1])
    # A staged record may be rejected at final framing, but cannot produce a
    # valid footer while exceeding the advertised reader bound.
    frames = list(telemetry.iter_key_journal(output.key_path))
    if footer["valid"]:
        assert admitted and output.dropped == 0
        assert next(row for row in frames if row["event"] == "source_get")[
            "key"
        ] == b"x" * (telemetry.BUFFER_BYTES - headroom)
        assert telemetry.validate_capture(tmp_path)["valid"]
    else:
        assert output.dropped > 0 and footer["dropped"] > 0
        assert frames[-1]["valid"] is False
        with pytest.raises(telemetry.CaptureInvalid):
            telemetry.validate_capture(tmp_path)


def target_operation(output, clock):
    backend = SimpleNamespace(
        _kvcr=SimpleNamespace(
            nixl_agent_name="target",
            _clock=lambda: clock[0],
            config=SimpleNamespace(operation_timeout_ms=1000, abandon_timeout_ms=3000),
        ),
        _dangling_ops=SimpleNamespace(
            incarnation="inc",
            sources={},
            probe=Mock(return_value=True),
            poll_target=Mock(return_value=False),
        ),
        _options=SimpleNamespace(no_pinning_mode=False),
        _record_progress_duration=Mock(),
        _record_progress_counter=Mock(),
        _invalidate_control_peer=Mock(),
        _send_control=Mock(return_value=True),
    )
    op = _TargetPullOp(
        op_id=("target", 1),
        keys=frozenset((b"a",)),
        started_at=0.0,
        deadline=1.0,
        state=_TargetPullState.START_WRITE,
        local_fill=False,
        remote_ctrl_ep="tcp://source:1",
        _backend=backend,
        ordered_keys=(b"a",),
        dst_descriptors=((),),
    )
    op._capture = capture.Context(output, backend, "target", (b"a",), "caller")
    assert op.progress(None, None) == (False, True)
    assert op.state is _TargetPullState.WAITING_WRITE_DONE
    return op


@pytest.mark.parametrize(
    "event,expected",
    [
        ({"terminal": True, "success": False}, "peer_write_failed"),
        ({"terminal": True, "success": False, "cancelled": True}, "peer_write_failed"),
        (
            {"terminal": True, "success": True, "completed_indices": [99]},
            "invalid_completed_indices",
        ),
    ],
)
def test_started_target_failure_has_truthful_reason(tmp_path, event, expected):
    output = telemetry.BufferedWriter(
        tmp_path, source_archive_sha256="test", start_thread=False
    )
    op = target_operation(output, [0.0])
    try:
        assert op.progress(None, event) == (True, True)
    finally:
        output.close()
    terminal = next(
        row
        for row in map(json.loads, output.path.read_text().splitlines())
        if row["event"] == "target_terminal"
    )
    assert terminal["status"] == "failed"
    assert terminal["reason"] == expected and expected in terminal["reasons"]
    assert telemetry.validate_capture(tmp_path)["valid"]


@pytest.mark.parametrize(
    "quiescence",
    [
        {"terminal": True, "success": True, "completed_indices": [0]},
        {"terminal": True, "success": False},
        {"terminal": True, "success": False, "cancelled": True},
    ],
)
def test_target_timeout_reason_survives_terminal_cleanup(tmp_path, quiescence):
    output = telemetry.BufferedWriter(
        tmp_path, source_archive_sha256="test", start_thread=False
    )
    clock = [0.0]
    op = target_operation(output, clock)
    clock[0] = 1.5
    try:
        assert op.progress(None, None) == (False, True)
        assert op.state is _TargetPullState.WAITING_TERMINAL
        assert not op._capture.done
        assert op.progress(None, quiescence) == (True, True)
        assert not op.success
    finally:
        output.close()
    terminal = next(
        row
        for row in map(json.loads, output.path.read_text().splitlines())
        if row["event"] == "target_terminal"
    )
    assert terminal["status"] == "failed"
    assert terminal["reason"] == "target_deadline_expired"
    assert "target_deadline_expired" in terminal["reasons"]
    assert telemetry.validate_capture(tmp_path)["valid"]


@pytest.mark.parametrize(
    "quiescence",
    [
        {"terminal": True, "success": True, "completed_indices": [0]},
        {"terminal": True, "success": False},
        {"terminal": True, "success": False, "cancelled": True},
    ],
)
def test_nonterminal_peer_failure_reason_survives_cleanup(tmp_path, quiescence):
    output = telemetry.BufferedWriter(
        tmp_path, source_archive_sha256="test", start_thread=False
    )
    op = target_operation(output, [0.0])
    try:
        failure = {"terminal": False, "success": False, "cancelled": True}
        assert op.progress(None, failure) == (False, True)
        assert op.state is _TargetPullState.WAITING_TERMINAL
        assert op._capture.reasons == ["peer_write_failed"]
        assert not op._capture.done
        assert op.progress(None, quiescence) == (True, True)
        assert not op.success
    finally:
        output.close()
    terminal = next(
        row
        for row in map(json.loads, output.path.read_text().splitlines())
        if row["event"] == "target_terminal"
    )
    assert terminal["status"] == "failed"
    assert terminal["reason"] == "peer_write_failed"
    assert terminal["reasons"] == ["peer_write_failed"]
    assert telemetry.validate_capture(tmp_path)["valid"]
