# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Check capture arguments and require fresh transferred bytes each iteration."""

import ctypes
import subprocess
import sys
from itertools import count
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


def run_target(monkeypatch, scenario, copy_calls, damage=None):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "examples"))
    import nvtx_remote_capture as capture

    before_deliver, reports, source_bytes = [], [], []
    clock = SimpleNamespace(**vars(capture.time))
    clock.monotonic = count().__next__
    monkeypatch.setattr(capture, "time", clock)

    def create_worker(name, framework, control, region, **kwargs):
        results = {}
        if name == "nvtx-source":
            source_bytes.append(ctypes.string_at(region.addr, 16))
            requests = [
                framework.request_pin(framework.keys)
                for _ in range(2 if damage == "missing_request" else 3)
            ]
            handles = (
                requests
                if scenario == "timeout"
                else [pin[0] for _, pin in framework.poll_pin_results() if pin]
            )
            if damage == "missing_handle":
                handles.pop()
            elif damage == "duplicate_handle":
                handles[-1] = handles[0]
            callback = (
                framework.cancel_pin_request
                if scenario == "timeout"
                else framework.release_pin
            )
            for handle in handles:
                callback(handle)

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
                        block = 1 - i if damage == "swapped_blocks" else i
                        data = source_bytes[0][
                            block * region.size : (block + 1) * region.size
                        ]
                        ctypes.memmove(region.addr + i * region.size, data, region.size)
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
            _core=SimpleNamespace(_framework_pin_keys={}),
        )

    monkeypatch.setattr(
        capture, "control_channel", lambda: SimpleNamespace(endpoint="source")
    )
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
    capture.worker(
        "source",
        args,
        SimpleNamespace(send=lambda _: None, poll=lambda: True, recv=lambda: None),
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


def test_capture_rejects_swapped_blocks(monkeypatch):
    with pytest.raises(AssertionError):
        run_target(monkeypatch, "success", {1, 2, 3}, "swapped_blocks")


@pytest.mark.parametrize("scenario", ["success", "partial", "zero", "timeout"])
@pytest.mark.parametrize(
    "damage", ["missing_handle", "duplicate_handle", "missing_request"]
)
def test_capture_checks_pin_lifecycle(monkeypatch, scenario, damage):
    with pytest.raises(AssertionError):
        run_target(monkeypatch, scenario, {1, 2, 3}, damage)


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
