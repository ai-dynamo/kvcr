# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optional Guard tracing preserves READY claims through native release."""

import threading

import pytest
from _kvcr_test_utils import _wait_until
from _nvtx_test_utils import payloads, spans
from test_no_pinning_mode import _request
from test_no_pinning_mode import source_factory as source_factory

from kvcr import _nvtx


@pytest.mark.parametrize("level", ["off", "low", "medium"])
def test_guard_trace_preserves_native_lifetime(
    recording, monkeypatch, level, source_factory
):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    if level == "off":
        monkeypatch.setattr(
            _nvtx, "_load_backend", lambda: pytest.fail("disabled NVTX loaded backend")
        )
    source, agent, control, pinning, _ = source_factory()
    allow_release = threading.Event()
    release = agent.release_xfer_handle
    agent.release_xfer_handle = lambda handle: (
        release(handle) if allow_release.is_set() else False
    )
    try:
        before = len(agent.xfers)
        _request(control, [b"missing", b"a"], sender_incarnation="target-inc")
        _wait_until(lambda: len(agent.xfers) == before + 1)
        assert source._core._local_dram._export_readers() == 1
        op = next(iter(source._core._progress._in_flight_ops.values()))
        assert (op.nvtx_state is None) == (level == "off")
        if level != "off":
            assert op.nvtx_state.target_incarnation == "target-inc"
            assert op.nvtx_state.requested_blocks == 2
        allow_release.set()
        _wait_until(lambda: source._core._local_dram._export_readers() == 0)
    finally:
        allow_release.set()
        source.close()
    assert not pinning.searches and not payloads(recording, "source.pin.context")
    assert len(spans(recording, "nixl.write", "end")) == (level != "off")
    assert not recording.ranges and not recording.stack
