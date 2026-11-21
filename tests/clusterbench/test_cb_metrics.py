# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import random
import threading

from clusterbench.metrics import LogHist, Window, thread_delta


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
    # KVCR's telemetry as vLLM's stats object holds it: kinds, then values
    # per label tuple.
    kinds = {
        "kvcr_transfer_blocks": "counter",
        "kvcr_duration_seconds": "histogram",
        "kvcr_state": "gauge",
    }
    values = {
        "kvcr_transfer_blocks": {("remote_deliver",): 5},
        "kvcr_duration_seconds": {("transfer", "success"): [0.01, 0.02]},
        "kvcr_state": {("connected_remotes",): 3},
    }
    window.merge_stats(kinds, values)
    report = window.to_json()
    assert report["calls"]["query"] == 5 and report["items"]["query"] == 5
    assert abs(report["secs"]["query"] - 0.003) < 1e-12
    assert report["kvcr_counters"]["kvcr_transfer_blocks|remote_deliver"] == 5
    assert report["kvcr_hists"]["kvcr_duration_seconds|transfer|success"]["n"] == 2
    assert report["gauges"]["kvcr_state|connected_remotes"] == 3
    window.reset()
    assert not window.to_json()["calls"]


def test_thread_delta_is_what_each_role_used_between_two_snapshots():
    engine = threading.main_thread().native_id
    # Ids no Python thread has, as NIXL's and UCX's threads appear.
    before = {engine: (1.0, 0.25), -1: (4.0, 0.5)}
    after = {engine: (1.5, 0.375), -1: (4.25, 0.5), -2: (0.125, 0.0625)}
    assert thread_delta(before, after) == {
        "engine": {"cpu_s": 0.5, "delay_s": 0.125, "threads": 1},
        # One thread from before and one that started in between.
        "native": {"cpu_s": 0.375, "delay_s": 0.0625, "threads": 2},
    }
