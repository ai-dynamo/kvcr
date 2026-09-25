# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Parse vLLM reset-interval KV transfer metrics from one worker's log.

Each Phase 1 worker runs one data-parallel rank, so every metric line in a log
belongs to that worker. Values are interval samples; a window sums the samples
whose timestamps fall inside [start, end]. Nothing is subtracted between lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import ANSI

METRIC = re.compile(
    r"(vllm:[^=]+)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(?:, |$)"
)


def metric(name: str, *labels: str) -> str:
    """Build the log form of a labelled vLLM metric, e.g. ('source_write',)."""
    inner = ", ".join(repr(label) for label in labels)
    if len(labels) == 1:
        inner += ","
    return f"vllm:{name}:({inner})"


def duration_count(scope: str, result: str = "success") -> str:
    return metric("kvcr_duration_seconds", scope, result) + "_count"


TRANSFER_BLOCKS = "kvcr_transfer_blocks"
TRANSFER_BYTES = "kvcr_transfer_bytes"
TIER_READ_BYTES = metric("kv_offload_tiering_read_bytes", "1:kvcr")
TIER_WRITE_BYTES = metric("kv_offload_tiering_write_bytes", "1:kvcr")
STATE = "kvcr_state"

SOURCE_WRITE = "source_write"
REMOTE_DELIVER = "remote_deliver"
LOCAL_FILL = "local_fill"
LOCAL_DELIVER = "local_deliver"
TRANSFER = "transfer"


def parse_timestamp(stamp: str) -> datetime | None:
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True)
class Snapshot:
    timestamp: datetime
    line: int
    metrics: dict[str, int | float]


def parse_log(text: str) -> tuple[list[Snapshot], list[str]]:
    """Return interval snapshots and parse errors for complete log lines."""
    snapshots: list[Snapshot] = []
    errors: list[str] = []
    for number, raw in enumerate(text.split("\n")[:-1], 1):
        line = ANSI.sub("", raw).strip()
        if "KV Transfer metrics:" not in line:
            continue
        stamp = parse_timestamp(line.split()[0])
        if stamp is None:
            errors.append(f"line {number}: metric line without ISO timestamp")
            continue
        payload = line.split("KV Transfer metrics:", 1)[1].strip()
        pairs = METRIC.findall(payload)
        values: dict[str, int | float] = {}
        for name, value in pairs:
            values[name] = (
                float(value) if any(c in value for c in ".eE") else int(value)
            )
        if not values or len(values) != payload.count("="):
            errors.append(f"line {number}: incomplete metric parsing")
            continue
        snapshots.append(Snapshot(stamp, number, values))
    return snapshots, errors


@dataclass
class Window:
    """Summed interval samples between two instants for one worker."""

    totals: dict[str, int | float]
    samples: int
    first_after_end: bool
    peer_failures: list[dict[str, Any]]
    parse_errors: list[str]

    def get(self, name: str, default: int | float = 0) -> int | float:
        return self.totals.get(name, default)

    def blocks(self, scope: str) -> int | float:
        return self.get(metric(TRANSFER_BLOCKS, scope))

    def bytes(self, scope: str) -> int | float:
        return self.get(metric(TRANSFER_BYTES, scope))

    def successes(self, scope: str) -> int | float:
        return self.get(duration_count(scope))

    def gauge(self, resource: str) -> int | float | None:
        value = self.totals.get(metric(STATE, resource))
        return value


def window(
    snapshots: list[Snapshot],
    errors: list[str],
    start: datetime,
    end: datetime | None = None,
) -> Window:
    """Sum samples stamped at or after start (and at or before end).

    first_after_end reports whether a sample past `end` exists, which shows the
    interval containing the request has been flushed by the engine.
    """
    totals: dict[str, int | float] = {}
    failures = []
    count = 0
    flushed = False
    for snapshot in snapshots:
        if snapshot.timestamp < start:
            continue
        if end is not None and snapshot.timestamp > end:
            flushed = True
            continue
        count += 1
        for name, value in snapshot.metrics.items():
            if name.startswith(f"vllm:{STATE}:"):
                totals[name] = value  # gauge: keep the latest sample
            else:
                totals[name] = totals.get(name, 0) + value
            if (
                value > 0
                and name.startswith("vllm:kvcr_")
                and ("'failed'" in name or "'partial'" in name)
            ):
                failures.append({"metric": name, "value": value, "line": snapshot.line})
    return Window(totals, count, flushed, failures, list(errors))


def read_window(log: Path, start: datetime, end: datetime | None = None) -> Window:
    snapshots, errors = parse_log(log.read_text(errors="replace"))
    return window(snapshots, errors, start, end)


def latest_gauge(log: Path, resource: str) -> int | float | None:
    snapshots, _ = parse_log(log.read_text(errors="replace"))
    name = metric(STATE, resource)
    for snapshot in reversed(snapshots):
        if name in snapshot.metrics:
            return snapshot.metrics[name]
    return None
