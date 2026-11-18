# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import random

from clusterbench.metrics import (
    LogHist,
    Telemetry,
    Window,
    thread_delta,
    thread_snapshot,
)


def test_loghist_quantiles_are_close_and_merge_exactly():
    rng = random.Random(0)
    values = [rng.lognormvariate(-7, 1.5) for _ in range(20_000)]
    whole, a, b = LogHist(), LogHist(), LogHist()
    for index, value in enumerate(values):
        whole.add(value)
        (a if index % 2 else b).add(value)
    a.merge(b)
    ordered = sorted(values)
    for q in (0.5, 0.9, 0.99):
        exact = ordered[int(q * len(ordered))]
        assert abs(whole.quantile(q) - exact) / exact < 0.1
        assert a.quantile(q) == whole.quantile(q)
    assert a.n == whole.n and abs(a.total - whole.total) < 1e-9
    again = LogHist.from_json(whole.to_json())
    assert again.quantile(0.99) == whole.quantile(0.99) and again.max == whole.max
    assert LogHist().quantile(0.5) == 0.0


def test_window_accumulates_calls_and_kvcr_telemetry():
    window = Window()
    window.call("query", 0.002, calls=4, items=4)
    window.call("query", 0.001, calls=1, items=1)
    stats = Telemetry()
    assert stats.is_empty()
    stats.increase_counter("kvcr_transfer_blocks", 5, ("remote_deliver",))
    stats.observe_histogram("kvcr_duration_seconds", 0.01, ("transfer", "success"))
    stats.observe_histogram("kvcr_duration_seconds", 0.02, ("transfer", "success"))
    stats.set_gauge("kvcr_state", 3, ("connected_remotes",))
    window.merge_telemetry(stats)
    window.merge_telemetry(None)
    report = window.to_json()
    assert report["calls"]["query"] == 5 and report["items"]["query"] == 5
    assert abs(report["secs"]["query"] - 0.003) < 1e-12
    assert report["kvcr_counters"]["kvcr_transfer_blocks|remote_deliver"] == 5
    assert report["kvcr_hists"]["kvcr_duration_seconds|transfer|success"]["n"] == 2
    assert report["gauges"]["kvcr_state|connected_remotes"] == 3
    window.reset()
    assert not window.to_json()["calls"]


def test_thread_delta_attributes_cpu_to_the_engine_thread():
    before = thread_snapshot()
    total = 0
    for i in range(3_000_000):
        total += i
    roles = thread_delta(before, thread_snapshot())
    assert roles["engine"]["cpu_s"] > 0
    assert roles["engine"]["threads"] == 1


def test_peer_operations_kvcr_timed_past_its_timeout_are_counted():
    window = Window()
    stats = Telemetry()
    name = "kvcr_duration_seconds"
    for seconds in (0.2, 1.01, 1.2):
        stats.observe_histogram(name, seconds, ("remote_deliver", "failed"))
    stats.observe_histogram(name, 1.5, ("remote_deliver", "partial"))
    stats.observe_histogram(name, 0.5, ("remote_deliver", "partial"))
    stats.observe_histogram(name, 3.0, ("remote_deliver", "success"))
    stats.observe_histogram(name, 3.0, ("source_write", "failed"))
    window.merge_telemetry(stats)
    assert "kvcr_peer_ops_past_timeout" not in window.counters  # no timeout known
    window.late_after = 1.0
    window.merge_telemetry(stats)
    # Only peer transfers that did not fully succeed, at or past the timeout.
    assert window.counters["kvcr_peer_ops_past_timeout"] == 3
