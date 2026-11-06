# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Parse per-interval KVCR telemetry from one worker's log.

vLLM logs ``KV Transfer metrics:`` lines (reset-interval samples of the
connector and KVCR counters). SGLang's KVCR HiCache backend logs
``KVCR L3 native stats {json}`` lines every few seconds with the same KVCR
counters under JSON tuple keys; they are mapped onto the vLLM metric names so
the cases read both engines through one window API.

Each Phase 1 worker runs one data-parallel rank, so every metric line in a log
belongs to that worker. Values are interval samples; a window sums the samples
whose timestamps fall inside [start, end]. Nothing is subtracted between lines.
"""

from __future__ import annotations

import ast
import json
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
SGLANG_STATS = "KVCR L3 native stats "


def sglang_metrics(
    payload: dict, page_bytes: dict[str, float] | None = None
) -> dict[str, int | float]:
    """Map one SGLang ``KVCR L3 native stats`` payload onto vLLM metric names.

    The payload's ``metrics`` keys are JSON tuples ``[kind, name, labels,
    field]`` from the backend's TelemetryStats; counters and gauges keep their
    value, histograms keep ``count`` and ``sum`` (as ``_count`` / ``_sum``).
    SGLang has no vLLM tiering counters, so the tier read/write bytes the cases
    check are derived: writes are the KVCR ``local_fill`` bytes; reads are the
    delivered blocks times the bytes per block. KVCR counts bytes for
    ``local_fill`` and ``source_write`` but only blocks for ``remote_deliver``
    and ``local_deliver``, so the per-block size is taken from this worker's own
    ``local_fill`` samples (``page_bytes`` carries it between samples of one log;
    every KVCR page of a model has the same size).
    """
    out: dict[str, int | float] = {}
    for key, value in payload.get("metrics", {}).items():
        if key == "dropped_observations":
            out["_dropped_observations"] = value
            continue
        try:
            kind, name, labels, field = json.loads(key)
        except (ValueError, TypeError):
            continue
        base = metric(str(name), *[str(label) for label in labels])
        if kind == "histogram":
            if field in ("count", "sum"):
                out[f"{base}_{field}"] = value
        elif field == "value":
            out[base] = value
    fill_blocks = out.get(metric(TRANSFER_BLOCKS, LOCAL_FILL), 0)
    fill_bytes = out.get(metric(TRANSFER_BYTES, LOCAL_FILL), 0)
    if page_bytes is not None and fill_blocks and fill_bytes:
        page_bytes["value"] = fill_bytes / fill_blocks
    per_block = (page_bytes or {}).get("value")
    read_bytes: float = 0
    for scope in (REMOTE_DELIVER, LOCAL_DELIVER):
        explicit = out.get(metric(TRANSFER_BYTES, scope))
        if explicit is not None:
            read_bytes += explicit
        elif per_block is not None:
            read_bytes += out.get(metric(TRANSFER_BLOCKS, scope), 0) * per_block
    out[TIER_READ_BYTES] = int(read_bytes)
    out[TIER_WRITE_BYTES] = fill_bytes
    return out


def parse_timestamp(stamp: str) -> datetime | None:
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None


ISO_INSTANT = re.compile(
    r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)$"
)


def precise_stamp(line: str) -> datetime | None:
    """Timezone-aware ISO instant at the start of a Dynamo-formatted line.

    Only a full instant with a zone counts: a bare date or a naive time at the
    start of some other line must not become a comparison partner for the
    aware request timestamps.
    """
    head = line.split(maxsplit=1)[0] if line else ""
    if not ISO_INSTANT.match(head):
        return None
    stamp = parse_timestamp(head)
    return stamp if stamp is not None and stamp.tzinfo is not None else None


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
    # vLLM's own logger stamps to the second; Dynamo's formatter to the
    # microsecond. For a second-precision sample, preceded_by is the latest
    # microsecond-precision stamp seen earlier in the log (a request's
    # handle_payload line, typically): it tells whether the sample was flushed
    # after a request that started in the same second.
    second_precision: bool = False
    preceded_by: datetime | None = None


def parse_log(text: str) -> tuple[list[Snapshot], list[str]]:
    """Return interval snapshots and parse errors for complete log lines."""
    snapshots: list[Snapshot] = []
    errors: list[str] = []
    last_precise: datetime | None = None
    page_bytes: dict[str, float] = {}  # SGLang: bytes per KVCR page, from local_fill
    for number, raw in enumerate(text.split("\n")[:-1], 1):
        line = ANSI.sub("", raw).strip()
        precise = precise_stamp(line)
        if precise is not None and (last_precise is None or precise > last_precise):
            last_precise = precise
        if SGLANG_STATS in line:
            # SGLang: one JSON payload per interval, stamped with an epoch.
            try:
                payload = json.loads(line.split(SGLANG_STATS, 1)[1].strip())
                stamp = datetime.fromtimestamp(float(payload["epoch"]), tz=timezone.utc)
            except (ValueError, KeyError, TypeError):
                errors.append(f"line {number}: unparsable SGLang KVCR stats payload")
                continue
            snapshots.append(
                Snapshot(stamp, number, sglang_metrics(payload, page_bytes))
            )
            continue
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
        snapshots.append(Snapshot(stamp, number, values, precise is None, last_precise))
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


def flushed_after_start(snapshot: Snapshot, start: datetime) -> bool:
    """Whether a second-precision sample stamped in start's second was flushed
    after start.

    vLLM's logger stamps metric lines to the second, so a sample flushed later
    in the second a request started in parses as earlier than the request. It
    counts for the request if a microsecond-stamped line (Dynamo's request
    handling) stamped at or after start precedes it in the log; a sample flushed
    before the request has no such line ahead of it and belongs to the previous
    request.
    """
    return (
        snapshot.second_precision
        and snapshot.timestamp == start.replace(microsecond=0)
        and snapshot.preceded_by is not None
        and snapshot.preceded_by >= start
    )


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
        if snapshot.timestamp < start and not flushed_after_start(snapshot, start):
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
