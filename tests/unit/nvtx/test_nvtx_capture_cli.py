# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Check capture arguments and require fresh transferred bytes each iteration."""

import ctypes
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from _kvcr_test_utils import _op_entries

from kvcr.hints.fetch import KV_FETCH
from kvcr.hints.parser import parse_hints


def test_capture_hint_is_accepted_by_dispatcher(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "examples"))
    from nvtx_pin_capture import remote_hint

    hints, rejected = parse_hints(
        remote_hint("tcp://source:1", 4), {KV_FETCH.action_type: KV_FETCH}
    )
    assert not rejected and len(hints) == 1
    assert hints[0].block_hashes == {0, 1, 2, 3}


def run_target(monkeypatch, scenario, copy_calls):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "examples"))
    import nvtx_remote_capture as capture

    before_deliver, reports = [], []

    def create_worker(name, framework, control, region, **kwargs):
        results = {}

        def deliver(descriptors, request):
            before_deliver.append(ctypes.string_at(region.addr, 16))
            handle = len(before_deliver)
            entries = {
                key: scenario == "success" or (scenario == "partial" and i == 0)
                for i, key in enumerate(descriptors)
            }
            if handle in copy_calls:
                for i, success in enumerate(entries.values()):
                    if success:
                        ctypes.memset(region.addr + i * region.size, 73, region.size)
                if not any(entries.values()):
                    # A failed completion need not prove destination quiescence.
                    ctypes.memset(region.addr, 91, 16)
            results[handle] = _op_entries(entries)
            return handle

        return SimpleNamespace(
            submit_hint=lambda *args: None,
            deliver=deliver,
            poll_completed=lambda: results,
            discard_hint=lambda *args: None,
            close=lambda: None,
        )

    monkeypatch.setattr(capture, "control_channel", lambda: None)
    monkeypatch.setattr(capture, "create_worker", create_worker)
    args = SimpleNamespace(
        blocks=2,
        block_bytes=8,
        memory="dram",
        pin_delay_ms=0,
        poll_delay_ms=0,
        scenario=scenario,
        warmup=1,
        iterations=2,
    )
    capture.worker("target", args, SimpleNamespace(send=reports.append), "source")
    return before_deliver, reports


@pytest.mark.parametrize("scenario", ["success", "partial"])
@pytest.mark.parametrize("copy_calls", [{1}, {1, 2, 3}])
def test_capture_checks_fresh_bytes_each_iteration(monkeypatch, scenario, copy_calls):
    if copy_calls == {1}:
        with pytest.raises(AssertionError):
            run_target(monkeypatch, scenario, copy_calls)
    else:
        before_deliver, reports = run_target(monkeypatch, scenario, copy_calls)
        assert len(before_deliver) == 3
        assert len(reports[0]["latency_ms"]) == 2


@pytest.mark.parametrize("scenario", ["zero", "failure", "timeout"])
def test_capture_preserves_buffers_after_failed_results(monkeypatch, scenario):
    before_deliver, reports = run_target(monkeypatch, scenario, {1})
    assert before_deliver == [bytes(16), bytes([91]) * 16, bytes([91]) * 16]
    assert len(reports[0]["latency_ms"]) == 2


def test_partial_capture_requires_two_blocks():
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[3] / "examples/nvtx_remote_capture.py"),
            "--scenario",
            "partial",
            "--blocks",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 2
    assert "partial requires at least two blocks" in result.stderr
