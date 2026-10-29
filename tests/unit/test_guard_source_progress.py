# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""No-pinning source liveness starts at the first export, not model loading."""

from collections import deque
from types import SimpleNamespace

import pytest

from kvcr.dangling_ops import _DanglingOps, _SourceWriteStatus


@pytest.fixture
def monitor(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("kvcr.dangling_ops.time.monotonic", lambda: clock[0])
    local = SimpleNamespace(_first_guard_ready_at=None)
    backend = SimpleNamespace(
        _options=SimpleNamespace(no_pinning_mode=True),
        _kvcr=SimpleNamespace(
            _local_dram=local,
            config=SimpleNamespace(operation_timeout_ms=1000),
        ),
        _progress_outbound=deque(),
    )
    result = _DanglingOps(backend)
    result._last_progress_at = 1.0
    return result, backend, clock


def test_empty_guard_does_not_consume_source_watchdog_during_startup(monitor):
    progress, _, _ = monitor
    assert progress.check_source_progress()
    assert not progress._source_stalled


def test_first_ready_publication_starts_watchdog(monitor):
    progress, backend, clock = monitor
    backend._kvcr._local_dram._first_guard_ready_at = 99.5
    assert progress.check_source_progress()
    assert progress._last_progress_at == 99.5
    clock[0] = 100.6
    assert not progress.check_source_progress()
    assert len(backend._progress_outbound) == 1


def test_eviction_does_not_rearm_or_forgive_a_source_stall(monitor):
    progress, backend, clock = monitor
    backend._kvcr._local_dram._first_guard_ready_at = 99.5
    assert progress.check_source_progress()
    backend._kvcr._local_dram._first_guard_ready_at = None
    clock[0] = 100.6
    assert not progress.check_source_progress()
    backend._kvcr._local_dram._first_guard_ready_at = 100.6
    assert not progress.check_source_progress()


def test_submitted_native_work_does_not_forgive_missing_publication(monitor):
    progress, _, _ = monitor
    progress.source_writes[("target", 1)] = _SourceWriteStatus(submitted=True)
    assert not progress.check_source_progress()


def test_framework_pinning_mode_keeps_original_watchdog(monitor):
    progress, backend, _ = monitor
    backend._options.no_pinning_mode = False
    assert not progress.check_source_progress()
