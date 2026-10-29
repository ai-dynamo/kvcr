# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Typed caller correlation across peers and forbidden callback evidence."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _mem_descriptor,
    _new_kvcr,
    _op_entries,
    _poll_until,
    _start_write_message,
    _wait_until,
)
from test_kvcr_remote_target import _ConstantHashAdapter, _router_hint
from test_no_pinning_mode import source_factory as source_factory

from kvcr import compact_telemetry as telemetry
from kvcr import operation_capture as capture
from kvcr.config import RemoteFWDramOptions
from kvcr.remote_ops import _CONTROL_DECODER, _StartWrite


def test_two_peer_typed_caller_and_operation_correlation(
    source_factory, tmp_path, monkeypatch
):
    output = telemetry.BufferedWriter(
        tmp_path, source_archive_sha256="test", start_thread=False
    )
    monkeypatch.setattr(telemetry, "writer", lambda: output)
    source, source_agent, source_control, pinning, _ = source_factory(recovered=True)
    target_agent = FakeNixlAgent(metadata=b"target-md")
    target_control = FakeBytesControl("tcp://target:1")
    target = _new_kvcr(
        target_agent,
        FakePrimaryPinning(),
        target_control,
        key_adapter=_ConstantHashAdapter(),
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
    )
    try:
        target.submit_hint(_router_hint("tcp://source:1"), request_id="caller-42")
        handle = target.deliver({b"a": [_mem_descriptor()]}, request_id="caller-42")
        _wait_until(lambda: bool(target_control.sent))
        message = next(
            raw
            for _, raw in target_control.sent
            if isinstance(_CONTROL_DECODER.decode(raw), _StartWrite)
        )
        decoded = _CONTROL_DECODER.decode(message)
        source_control.incoming.append(message)
        _wait_until(lambda: bool(source_agent.xfers))
        _wait_until(lambda: source._core._local_dram._export_readers() == 0)
        target_agent.notifs["source"] = [source_agent.xfers[-1][5]]
        assert _poll_until(target, bool) == [(handle, _op_entries({b"a": True}))]
    finally:
        source.close()
        target.close()
        output.close()
    rows = [json.loads(line) for line in output.path.read_text().splitlines()]
    target_terminal = next(row for row in rows if row["event"] == "target_terminal")
    target_clock = target_terminal["timing_ns"]
    assert target_terminal["start_monotonic_seconds"] is not None
    assert (
        target_clock["target_request_ns"] / 1e9
        <= target_terminal["start_monotonic_seconds"]
        <= target_clock["target_end_ns"] / 1e9
    )
    assert decoded.request_id == "caller-42"
    source_terminal = next(
        row for row in rows if row["event"] == "source_nixl_terminal"
    )
    source_get = next(
        row
        for row in telemetry.iter_key_journal(output.key_path)
        if row["event"] == "source_get"
    )
    assert target_terminal["request_id"] == "caller-42"
    assert decoded.telemetry_request_key == target_terminal["operation_key"]
    for row in (source_terminal, source_get):
        assert row["request_id"] == "caller-42"
        assert row["request_correlation"] == target_terminal["operation_key"]
    assert source_get["operation_key"] == source_terminal["operation_key"]
    assert not pinning.searches
    assert telemetry.validate_capture(tmp_path, ["source", "target"])["valid"]


def test_old_typed_start_write_defaults_missing_correlation():
    raw = _start_write_message(12, b"a", target_agent="target")
    fields = msgspec.msgpack.decode(raw)
    assert "request_id" not in fields and "telemetry_request_key" not in fields
    decoded = _CONTROL_DECODER.decode(raw)
    assert isinstance(decoded, _StartWrite)
    assert decoded.request_id is None
    assert decoded.telemetry_request_key is None
    encoded = msgspec.msgpack.encode(decoded)
    roundtrip = msgspec.msgpack.decode(encoded)
    assert "request_id" not in roundtrip and "telemetry_request_key" not in roundtrip
    assert _CONTROL_DECODER.decode(encoded) == decoded


def test_recovered_mapping_retains_source_request_id(tmp_path, monkeypatch):
    output = telemetry.BufferedWriter(
        tmp_path, source_archive_sha256="test", start_thread=False
    )
    monkeypatch.setattr(telemetry, "writer", lambda: output)
    backend = SimpleNamespace(
        _kvcr=SimpleNamespace(nixl_agent_name="source"),
        _dangling_ops=SimpleNamespace(incarnation="inc"),
        _options=SimpleNamespace(no_pinning_mode=True),
    )

    @capture.request("source")
    def reject(backend, progress, payload):
        capture.reason("malformed_start_write")

    try:
        reject(
            backend,
            None,
            dict(
                keys=None,
                request_id="malformed-caller",
                telemetry_request_key="target-operation",
            ),
        )
    finally:
        output.close()
    terminal = next(
        row
        for row in map(json.loads, output.path.read_text().splitlines())
        if row["event"] == "source_nixl_terminal"
    )
    assert terminal["request_id"] == "malformed-caller"
    assert terminal["request_correlation"] == "target-operation"
    assert telemetry.validate_capture(tmp_path)["valid"]


def test_forbidden_source_callback_invalidates_capture(
    source_factory, tmp_path, monkeypatch
):
    output = telemetry.BufferedWriter(
        tmp_path, source_archive_sha256="test", start_thread=False
    )
    monkeypatch.setattr(telemetry, "writer", lambda: output)
    source, _, _, _, _ = source_factory(recovered=True)
    callback = Mock(side_effect=AssertionError("forbidden framework callback ran"))
    monkeypatch.setattr(source._core, "_request_pin_callback", callback)
    try:
        assert source._core._remote_fw_dram._pin_framework_keys((b"absent",)) is None
        callback.assert_not_called()
    finally:
        source.close()
        output.close()
    rows = [json.loads(line) for line in output.path.read_text().splitlines()]
    error = next(row for row in rows if row["event"] == "capture_error")
    assert error["reason"] == "source_callback_violation:request_pin"
    assert "source_callback_violation:request_pin" in rows[-1]["instrumentation_errors"]
    assert rows[-1]["valid"] is False
    with pytest.raises(telemetry.CaptureInvalid):
        telemetry.validate_capture(tmp_path, ["source"])


@pytest.mark.parametrize("metrics_enabled", [False, True])
def test_real_source_body_clock_and_core_boundary(
    tmp_path, monkeypatch, metrics_enabled
):
    import ctypes

    from kvcr.config import KVCRConfig, LocalDramOptions
    from kvcr.core import _BlockRecord
    from kvcr.local_dram import _LocalDramResidency, _LocalDramState
    from kvcr.recovery_journal import install_recovery_records
    from kvcr.types import RegionDescriptor

    output = telemetry.BufferedWriter(
        tmp_path, source_archive_sha256="test", start_thread=False
    )
    monkeypatch.setattr(telemetry, "writer", lambda: output)
    memory = ctypes.create_string_buffer(b"a" * 16)
    address = ctypes.addressof(memory)
    agent, control = FakeNixlAgent(), FakeBytesControl()
    source = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        control,
        KVCRConfig(
            nixl_agent_name="source",
            pool_layouts=[("", 16)],
            enable_telemetry=metrics_enabled,
        ),
        name="source",
        remote_options=RemoteFWDramOptions(no_pinning_mode=True),
        framework_regions=[RegionDescriptor(addr=address, size=16)],
        local_dram=LocalDramOptions([("", address, 16)]),
    )
    original_timer = source._core._timer
    install_recovery_records(
        source._core,
        {
            b"a": _BlockRecord(
                local_dram=_LocalDramResidency([("", 0)], _LocalDramState.READY)
            )
        },
    )
    backend = source._core._remote_fw_dram
    original_duration = backend._record_progress_duration
    metrics = []

    def observed_duration(scope, started_at, result):
        original_duration(scope, started_at, result)
        if scope == "source_write":
            metrics.extend(
                row
                for row in backend._progress_metrics
                if row[0] == "histogram" and row[3] == (scope, result)
            )

    monkeypatch.setattr(backend, "_record_progress_duration", observed_duration)
    agent.state = "DONE"
    try:
        control.incoming.append(_start_write_message(42, b"a", target_agent="target"))
        _wait_until(lambda: bool(agent.released_xfers))
        assert source._core._timer is original_timer
    finally:
        source.close()
        output.close()
    terminal = next(
        row
        for row in map(json.loads, output.path.read_text().splitlines())
        if row["event"] == "source_nixl_terminal"
    )
    body_start = terminal["start_monotonic_seconds"]
    assert body_start is not None
    clock = terminal["timing_ns"]
    assert (
        clock["source_receive_ns"] / 1e9 <= body_start <= clock["claim_begin_ns"] / 1e9
    )
    assert (
        clock["first_observed_done_ns"]
        <= clock["native_handle_released_ns"]
        <= clock["core_source_write_ns"]
        <= clock["cleanup_done_ns"]
    )
    assert terminal["source_operation_seconds"] == pytest.approx(
        clock["source_end_ns"] / 1e9 - body_start
    )
    assert bool(metrics) is metrics_enabled
    if metrics_enabled:
        assert len(metrics) == 1
        assert metrics[0][2] == pytest.approx(
            clock["core_source_write_ns"] / 1e9 - body_start, abs=1e-8
        )
    assert telemetry.validate_capture(tmp_path, ["source"])["valid"]
