# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Parse vLLM reset-interval KV transfer metrics from one worker's log.

Each Phase 1 worker runs one data-parallel rank, so every metric line in a log
belongs to that worker. Values are interval samples; a window sums the samples
whose timestamps fall inside [start, end]. Nothing is subtracted between lines.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .common import ANSI

# "ConnectorName={...}" segments of a MultiConnector metrics line; the dict
# body has no nested braces, so a greedy match up to the closing brace works.
CONNECTOR_DICT = re.compile(r"(\b[A-Za-z]+Connector)=(\{[^}]*\})")

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


VLLM_STAMP = re.compile(r"\b(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\b")


def parse_timestamp(stamp: str) -> datetime | None:
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def line_timestamp(line: str) -> datetime | None:
    """Timestamp of a metrics line in either log format.

    Dynamo's tracing formatter starts the line with an ISO instant
    (``2026-10-03T21:41:22.673433Z ...``); vLLM's own logger, which Dynamo
    ea23e51+ lets write the KV Transfer metrics line, prefixes
    ``(MainProcess pid=N) INFO 10-08 20:54:47 [...]`` with no year or zone.
    The latter is taken as UTC of the current year (the harness runs with
    TZ=UTC containers); a run spanning New Year would need the launch date.
    """
    stamp = parse_timestamp(line.split()[0])
    if stamp is not None:
        return stamp
    head = line.split("KV Transfer metrics:", 1)[0]
    found = VLLM_STAMP.search(head)
    if found is None:
        return None
    month, day, hour, minute, second = (int(x) for x in found.groups())
    now = datetime.now(timezone.utc)
    year = now.year
    candidate = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    if candidate - now > timedelta(days=1):  # log from late last year
        candidate = candidate.replace(year=year - 1)
    return candidate


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
        stamp = line_timestamp(line)
        if stamp is None:
            errors.append(f"line {number}: metric line without a parseable timestamp")
            continue
        payload = line.split("KV Transfer metrics:", 1)[1].strip()
        values: dict[str, int | float] = {}
        if CONNECTOR_DICT.search(payload):
            # MultiConnector: one Python dict repr per connector,
            # "OffloadingConnector={...}, NixlConnector={...}". Merge them; the
            # KVCR tier lives in the OffloadingConnector dict.
            parsed = 0
            for found in CONNECTOR_DICT.finditer(payload):
                try:
                    block = ast.literal_eval(found.group(2))
                except (ValueError, SyntaxError):
                    continue
                if isinstance(block, dict):
                    parsed += 1
                    for name, value in block.items():
                        if isinstance(value, (int, float)) and not isinstance(
                            value, bool
                        ):
                            values[str(name)] = value
            if not parsed:
                errors.append(f"line {number}: unparsable per-connector metrics")
                continue
        else:
            pairs = METRIC.findall(payload)
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
    if errors:
        totals["_parse_errors"] = len(errors)
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
