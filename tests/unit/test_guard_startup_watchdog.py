# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Startup is unarmed until READY; the native lifetime fences stay active."""

import queue
import threading
from types import SimpleNamespace as A

from kvcr import dangling_ops as dangling
from kvcr import local_dram as local_module
from kvcr.dangling_ops import _DanglingOps, _SourceWriteStatus
from kvcr.local_dram import _LocalDram, _LocalDramResidency, _LocalDramState
from kvcr.progress import _TransferRef


def setup_watchdog(monkeypatch, guard=True):
    clock = [0.0]
    monkeypatch.setattr(dangling, "time", A(monotonic=lambda: clock[0]))
    monkeypatch.setattr(
        local_module, "time", A(monotonic_ns=lambda: round(clock[0] * 1e9))
    )
    local = _LocalDram.__new__(_LocalDram)
    local._kvcr = A(_no_pinning_mode=guard)
    local._export_lock = threading.Lock()
    local._exports = {}
    local._export_sequence = 0
    local._export_held = 0
    local._export_released = queue.SimpleQueue()
    local._capture_writer = None
    local._first_guard_ready_at = None
    local._descriptor = lambda label, slot: _TransferRef("source", slot, label, False)
    local._journal = lambda *args: None
    notices = []
    backend = A(
        _options=A(no_pinning_mode=guard),
        _kvcr=A(
            _local_dram=local,
            config=A(operation_timeout_ms=5000, abandon_timeout_ms=10000),
            _clock=lambda: clock[0],
        ),
        _progress_outbound=[],
        _guard_event=lambda *args: None,
        _send_write_done=lambda *args, **kw: notices.append(kw),
    )
    backend._release_guard_claims = lambda op: local.release_export_claims(
        op.guard_claims
    )
    return _DanglingOps(backend), local, clock, notices


def publish(local, clock, at):
    clock[0] = at
    residency = _LocalDramResidency([("", 0)], _LocalDramState.READY)
    local._publish_export(b"a", residency)
    return residency


def test_empty_guard_startup_gap_does_not_poison_fresh_claim(monkeypatch):
    watchdog, local, clock, _ = setup_watchdog(monkeypatch)
    watchdog.begin_poll()
    clock[0] = 6.0
    assert watchdog.check_source_progress()
    watchdog.begin_poll()
    residency = publish(local, clock, 6.1)
    assert local._first_guard_ready_at == 6.1
    assert watchdog.check_source_progress()
    selected, busy = local.claim_export_subset((b"a",), (("",),), ())
    assert not busy and len(selected) == 1 and residency.export_readers == 1
    local.release_export_claims(tuple(claim for _, claim in selected))
    assert residency.export_readers == 0 and not watchdog._source_stalled
    assert watchdog._guard_source_armed_at == 6.1


def test_first_ready_time_sticky_after_emptying_exports(monkeypatch):
    watchdog, local, clock, _ = setup_watchdog(monkeypatch)
    watchdog.begin_poll()
    publish(local, clock, 1.0)
    assert watchdog.check_source_progress()
    with local._export_lock:
        local._withdraw_locked(b"a")
    publish(local, clock, 2.0)
    assert local._first_guard_ready_at == 1.0
    with local._export_lock:
        local._withdraw_locked(b"a")
    clock[0] = 7.0
    assert not watchdog.check_source_progress()
    assert watchdog._source_stalled and len(watchdog._backend._progress_outbound) == 1
    clock[0] = 7.1
    watchdog.begin_poll()
    assert not watchdog.check_source_progress()


def test_first_ready_before_unobserved_gap_arms_at_publication(monkeypatch):
    watchdog, local, clock, _ = setup_watchdog(monkeypatch)
    watchdog.begin_poll()
    publish(local, clock, 1.0)
    clock[0] = 6.1
    assert not watchdog.check_source_progress()
    assert watchdog._guard_source_armed_at == 1.0


def test_native_acceptance_without_ready_marker_still_fenced(monkeypatch):
    watchdog, local, clock, _ = setup_watchdog(monkeypatch)
    watchdog.begin_poll()
    watchdog.source_writes[("target", 1)] = _SourceWriteStatus(submitted=True)
    clock[0] = 6.0
    assert not watchdog.check_source_progress() and watchdog._source_stalled


def test_legacy_framework_source_empty_gap_still_latches(monkeypatch):
    watchdog, _, clock, _ = setup_watchdog(monkeypatch, guard=False)
    watchdog.begin_poll()
    clock[0] = 6.0
    assert not watchdog.check_source_progress()
    assert "timeout: 5000 ms" in str(watchdog._backend._progress_outbound[0])


def test_postactivation_stall_retains_native_claim_until_done(monkeypatch):
    watchdog, local, clock, notices = setup_watchdog(monkeypatch)
    watchdog.begin_poll()
    residency = publish(local, clock, 1.0)
    assert watchdog.check_source_progress()
    watchdog.begin_poll()
    selected, busy = local.claim_export_subset((b"a",), (("",),), ())
    claims = tuple(claim for _, claim in selected)
    op = A(
        guard_only=True,
        guard_claims=claims,
        route=("target", 0),
        remote_agent="target",
        op_handle=1,
        deadline=100.0,
        transfer_id=7,
    )
    watchdog.source_writes[("target", 1)] = _SourceWriteStatus(submitted=True)
    incarnation = watchdog.incarnation
    clock[0] = 7.0
    assert not watchdog.check_source_progress()
    result = [None]
    calls = []

    def poll(handle, *, require_completion):
        calls.append((handle, require_completion))
        return result[0]

    progress = A(poll_transfer=poll)
    assert watchdog.poll_source(progress, op, cancelling=True) is None
    assert residency.export_readers == 1 and ("target", 1) in watchdog.source_writes
    assert calls == [(7, True)] and notices == [{"terminal": False}]
    assert watchdog.incarnation == incarnation
    result[0] = (True, None)
    assert watchdog.poll_source(progress, op, cancelling=True) == (True, None)
    watchdog.finish_source(op)
    assert residency.export_readers == 0 and not watchdog.source_writes
    assert not watchdog.check_source_progress()


def test_ready_published_during_begin_poll_keeps_activation_baseline(monkeypatch):
    watchdog, local, clock, _ = setup_watchdog(monkeypatch)
    calls = [0]

    def interleaved_clock():
        calls[0] += 1
        if calls[0] == 1:
            return 0.0  # Poll entry, before first READY exists.
        if calls[0] == 2:
            publish(local, clock, 6.1)
        return clock[0]

    monkeypatch.setattr(dangling, "time", A(monotonic=interleaved_clock))
    watchdog.begin_poll()
    assert watchdog._last_progress_at >= 6.1
    clock[0] = 6.2
    assert watchdog.check_source_progress() and not watchdog._source_stalled
