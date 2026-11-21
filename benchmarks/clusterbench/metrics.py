# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Measurement: mergeable histograms, the per-worker window, /proc samplers."""

from __future__ import annotations

import gc
import math
import os
import threading
import time
from collections import defaultdict

_SUB = 8  # buckets per octave
_BASE = 1e-6  # first bucket boundary, seconds
_BUCKETS = _SUB * 40


class LogHist:
    """Log-bucketed histogram of seconds, 8 buckets per octave from 1 us.

    Buckets are fixed, so histograms from different workers merge exactly;
    quantiles are accurate to about 9%.
    """

    __slots__ = ("counts", "n", "total", "max")

    def __init__(self) -> None:
        self.counts: dict[int, int] = {}
        self.n = 0
        self.total = 0.0
        self.max = 0.0

    def add(self, value: float) -> None:
        if value <= _BASE:
            index = 0
        else:
            index = min(_BUCKETS, int(math.log2(value / _BASE) * _SUB) + 1)
        counts = self.counts
        counts[index] = counts.get(index, 0) + 1
        self.n += 1
        self.total += value
        if value > self.max:
            self.max = value

    def merge(self, other: "LogHist") -> None:
        counts = self.counts
        for index, count in other.counts.items():
            counts[index] = counts.get(index, 0) + count
        self.n += other.n
        self.total += other.total
        self.max = max(self.max, other.max)

    def quantile(self, q: float) -> float:
        if not self.n:
            return 0.0
        target = q * self.n
        seen = 0
        for index in sorted(self.counts):
            seen += self.counts[index]
            if seen >= target:
                return min(self.max, _BASE * 2 ** (index / _SUB))
        return self.max

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "sum": self.total,
            "max": self.max,
            "b": {str(k): v for k, v in self.counts.items()},
        }

    @classmethod
    def from_json(cls, data: dict) -> "LogHist":
        hist = cls()
        hist.n, hist.total, hist.max = data["n"], data["sum"], data["max"]
        hist.counts = {int(k): v for k, v in data["b"].items()}
        return hist

    def summary(self) -> dict[str, float]:
        return {
            "n": self.n,
            "mean_ms": self.mean * 1e3,
            "p50_ms": self.quantile(0.5) * 1e3,
            "p90_ms": self.quantile(0.9) * 1e3,
            "p99_ms": self.quantile(0.99) * 1e3,
            "max_ms": self.max * 1e3,
        }


class Window:
    """Everything a worker accumulates over the measurement window."""

    def __init__(self) -> None:
        self._gc_started = 0.0
        self._gc_on = False
        self.reset()

    def reset(self) -> None:
        self.started = time.perf_counter()
        self.ended = 0.0
        # Engine-thread time inside KVCR calls: seconds, calls, items (keys).
        self.secs: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.items: dict[str, int] = defaultdict(int)
        self.counters: dict[str, float] = defaultdict(float)
        self.hists: dict[str, LogHist] = defaultdict(LogHist)
        self.kvcr_counters: dict[str, float] = defaultdict(float)
        self.kvcr_hists: dict[str, LogHist] = defaultdict(LogHist)
        self.gauges: dict[str, float] = {}
        self.gc: dict[str, list[float]] = defaultdict(lambda: [0, 0.0, 0.0])

    def call(self, name: str, seconds: float, calls: int = 1, items: int = 1) -> None:
        self.secs[name] += seconds
        self.calls[name] += calls
        self.items[name] += items

    def watch_gc(self) -> None:
        if not self._gc_on:
            gc.callbacks.append(self._on_gc)
            self._gc_on = True

    def _on_gc(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._gc_started = time.perf_counter()
            return
        pause = time.perf_counter() - self._gc_started
        name = threading.current_thread().name
        entry = self.gc[f"{name}:gen{info['generation']}"]
        entry[0] += 1
        entry[1] += pause
        entry[2] = max(entry[2], pause)

    def merge_stats(self, types: dict, values: dict) -> None:
        """KVCR's telemetry since its last ``get_stats``: per metric name its
        kind, and per label tuple a count, a gauge value or a list of samples."""
        for name, by_labels in values.items():
            kind = types[name]
            for labels, value in by_labels.items():
                key = "|".join((name, *labels))
                if kind == "counter":
                    self.kvcr_counters[key] += value
                elif kind == "gauge":
                    self.gauges[key] = value
                else:
                    hist = self.kvcr_hists[key]
                    for sample in value:
                        hist.add(sample)

    def to_json(self) -> dict:
        ended = self.ended or time.perf_counter()
        return {
            "window_s": ended - self.started,
            "secs": dict(self.secs),
            "calls": dict(self.calls),
            "items": dict(self.items),
            "counters": dict(self.counters),
            "hists": {k: v.to_json() for k, v in self.hists.items()},
            "kvcr_counters": dict(self.kvcr_counters),
            "kvcr_hists": {k: v.to_json() for k, v in self.kvcr_hists.items()},
            "gauges": dict(self.gauges),
            "gc": {
                k: {"count": c, "total_ms": t * 1e3, "max_ms": m * 1e3}
                for k, (c, t, m) in self.gc.items()
            },
        }


_TICK = os.sysconf("SC_CLK_TCK")
_PAGE = os.sysconf("SC_PAGE_SIZE")


def thread_names() -> dict[int, str]:
    """Native thread id -> role, for threads Python knows about."""
    names = {}
    for thread in threading.enumerate():
        if thread.native_id is None:
            continue
        if thread is threading.main_thread():
            names[thread.native_id] = "engine"
        elif thread.name == "kvcr-progress":
            names[thread.native_id] = "progress"
        else:
            names[thread.native_id] = f"py:{thread.name}"
    return names


def thread_snapshot() -> dict[int, tuple[float, float]]:
    """Per thread: CPU seconds and seconds spent runnable but not running."""
    snapshot = {}
    try:
        tids = os.listdir("/proc/self/task")
    except OSError:
        return snapshot
    for tid in tids:
        try:
            with open(f"/proc/self/task/{tid}/stat") as f:
                fields = f.read().rpartition(") ")[2].split()
            cpu = (int(fields[11]) + int(fields[12])) / _TICK
            with open(f"/proc/self/task/{tid}/schedstat") as f:
                delay = int(f.read().split()[1]) / 1e9
        except (OSError, IndexError, ValueError):
            continue
        snapshot[int(tid)] = (cpu, delay)
    return snapshot


def thread_delta(
    before: dict[int, tuple[float, float]], after: dict[int, tuple[float, float]]
) -> dict[str, dict[str, float]]:
    """CPU and run-queue delay by role over an interval; native threads pooled."""
    names = thread_names()
    roles: dict[str, dict[str, float]] = {}
    for tid, (cpu, delay) in after.items():
        cpu0, delay0 = before.get(tid, (0.0, 0.0))
        role = names.get(tid, "native")
        entry = roles.setdefault(role, {"cpu_s": 0.0, "delay_s": 0.0, "threads": 0})
        entry["cpu_s"] += cpu - cpu0
        entry["delay_s"] += delay - delay0
        entry["threads"] += 1
    return roles


def rss_bytes() -> int:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * _PAGE
    except (OSError, IndexError, ValueError):
        return 0
