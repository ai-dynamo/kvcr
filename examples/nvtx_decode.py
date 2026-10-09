# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Validate KVCR events exported by nsys export --include-json=true.

Pass a two-process capture SQLite file and the scenario/number of operations
(including warmup). This checks lifecycle joins and context encoding, not GPU
kernel timing. The workload separately verifies transferred bytes.
"""

import argparse
import json
import sqlite3
from collections import Counter
from hashlib import blake2b

_IDENTITY = ("instance_hi", "instance_lo", "trace_id")
_METADATA = ("start", "end", "thread", "end_thread", "name", "event_type", "range_id")


def reconstruct_context(events):
    """Two passes allow a pin context to follow its callback's range start."""
    contexts = {}
    for event in events:
        if event.get("schema_version") == 3 and event["name"] in (
            "request.context",
            "source.pin.context",
        ):
            key = tuple(event[field] for field in _IDENTITY)
            assert key not in contexts, "duplicate lifecycle context"
            contexts[key] = {
                field: value
                for field, value in event.items()
                if field not in _METADATA and field != "_payload_fields"
            }
    result = []
    for event in events:
        version = event.get("schema_version")
        assert version in (None, 1, 2, 3), f"unsupported payload schema {version}"
        if version != 3:
            result.append(event)
            continue
        key = tuple(event[field] for field in _IDENTITY)
        assert key in contexts, "missing lifecycle context"
        fields = event.get(
            "_payload_fields", tuple(field for field in event if field not in _METADATA)
        )
        emitted = {field: event[field] for field in fields}
        result.append(
            dict(
                contexts[key],
                **emitted,
                **{field: event[field] for field in _METADATA if field in event},
                _payload_fields=fields,
            )
        )
    return result


def scalar(value):
    if isinstance(value, dict):
        return scalar(next(iter(value.values())))
    if isinstance(value, list):
        return [scalar(item) for item in value]
    return value


def decode(path):
    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(NVTX_EVENTS)")
        }
        extra = ",".join(
            "e." + column if column in columns else "NULL"
            for column in ("endGlobalTid", "eventType", "rangeId")
        )
        rows = connection.execute(
            f"SELECT e.start,e.end,e.globalTid,s.value,e.jsonText,{extra} "
            "FROM NVTX_EVENTS e JOIN StringIds s ON s.id=e.textId "
            "WHERE e.jsonText IS NOT NULL ORDER BY e.start"
        ).fetchall()
    events = []
    for start, end, thread, name, raw, end_thread, event_type, range_id in rows:
        payload = {key: scalar(value) for key, value in json.loads(raw).items()}
        if "schema_version" in payload and "instance_hi" in payload:
            events.append(
                dict(
                    start=start,
                    end=end,
                    thread=thread,
                    end_thread=(end_thread if end_thread is not None else thread)
                    if end is not None
                    else None,
                    name=name,
                    event_type=event_type,
                    range_id=range_id,
                    **payload,
                )
            )
    return reconstruct_context(events)


def require_range(named, name, reference):
    spans = [
        event
        for event in named.get(name, [])
        if all(event[field] == reference[field] for field in _IDENTITY)
    ]
    assert len(spans) == 1, f"missing or duplicate {name} range"
    span = spans[0]
    assert span["event_type"] == 60, f"{name} is not a start/end range"
    assert span["range_id"] is not None, f"missing {name} range ID"
    assert span["end"] is not None and span["end"] >= span["start"], (
        f"unclosed {name} range"
    )
    assert span["end_thread"] is not None, f"missing {name} end thread"
    assert span["thread"] >> 24 == span["end_thread"] >> 24, f"{name} crossed processes"
    return span


def operation_key(event):
    return tuple(
        event[key]
        for key in (
            "target_agent_hi",
            "target_agent_lo",
            "target_incarnation_hi",
            "target_incarnation_lo",
            "op_handle",
        )
    )


def validate(events, scenario, operations):
    if scenario == "off":
        assert not events, "off mode emitted KVCR payloads"
        return dict(events=0)
    events = reconstruct_context(events)
    named = {}
    for event in events:
        named.setdefault(event["name"], []).append(event)
        if (
            event["name"] == "request.context"
            and event["request_known"]
            and event.get("request_display_known", 1)
        ):
            values = event["value_utf8"]
            while values and isinstance(values[0], list):
                values = values[0]
            encoded = bytes(values[: event["length"]])
            if not event["truncated"]:
                text = encoded.decode("utf-8")
                digest = blake2b(encoded, digest_size=16).digest()
                assert int.from_bytes(digest[:8], "big") == event["request_hi"]
                assert int.from_bytes(digest[8:], "big") == event["request_lo"]
                assert text.startswith("capture-α-😀-")
        if "session_known" in event:
            assert event["session_known"] == event["parent_session_known"] == 0
    returned = named.get("op.completion_returned", [])
    assert len(returned) == operations, (len(returned), operations)
    expected = {"success": 2, "partial": 3, "zero": 4, "failure": 4, "timeout": 4}[
        scenario
    ]
    gaps = []
    for result in returned:
        assert result["status"] == expected, result
        if result["request_known"]:
            contexts = [
                event
                for event in named.get("request.context", [])
                if all(
                    event[field] == result[field]
                    for field in ("instance_hi", "instance_lo", "trace_id")
                )
            ]
            assert len(contexts) == 1, "missing or duplicate request context"
            assert all(
                contexts[0][field] == result[field]
                for field in ("request_known", "request_hi", "request_lo")
            ), "mismatched request context"
        key = operation_key(result)
        assert any(key[:2]) and any(key[2:4]), "missing cross-worker identity"

        def related(name):
            return [e for e in named.get(name, []) if operation_key(e) == key]

        queued = related("op.completion_queued")
        target = related("target.queued")
        received = related("target.write_done.received")
        assert len(queued) == len(target) == len(received) == 1
        # The caller operation and remote branch have separate local traces.
        # Only the branch's hint_trace_id identifies the consumed submission.
        branch = target[0]
        assert all(
            branch[field] == result[field] for field in ("instance_hi", "instance_lo")
        ), "hint branch and caller use different tracer instances"
        hint_id = branch.get("hint_trace_id", 0)
        assert hint_id, "missing hint trace identity"

        def hint_events(name):
            return [
                event
                for event in named.get(name, [])
                if (event["instance_hi"], event["instance_lo"], event["trace_id"])
                == (branch["instance_hi"], branch["instance_lo"], hint_id)
            ]

        submitted = hint_events("hint.submitted")
        used = [
            event
            for event in hint_events("hint.used")
            if event["op_handle"] == result["op_handle"]
        ]
        assert len(submitted) == len(used) == 1, "missing or duplicate hint association"
        assert all(
            event[field] == result[field]
            for event in (branch, submitted[0], used[0])
            for field in ("request_known", "request_hi", "request_lo")
        ), "mismatched hint request identity"
        assert submitted[0]["start"] <= used[0]["start"] <= branch["start"], (
            "hint used outside its submission lifetime"
        )
        assert not any(
            event["start"] <= used[0]["start"]
            for name in ("hint.replaced", "hint.rejected", "hint.discarded")
            for event in hint_events(name)
        ), "hint invalidated before use"
        assert received[0]["start"] <= queued[0]["start"] <= result["start"]
        compact = result.get("schema_version") == 3
        if compact:
            caller_span = require_range(named, "op.deliver.lifecycle", result)
            target_span = require_range(named, "target.remote", branch)
            assert (
                caller_span["start"]
                <= branch["start"]
                <= result["start"]
                <= caller_span["end"]
            )
            assert (
                target_span["start"]
                <= branch["start"]
                <= received[0]["start"]
                <= target_span["end"]
            )
            assert caller_span["thread"] == caller_span["end_thread"]
            assert target_span["thread"] != target_span["end_thread"], (
                "target ownership never crossed threads"
            )
        gaps.append((result["start"] - received[0]["start"]) / 1e6)
        if scenario == "zero":
            completed = related("source.write.completed")
            assert len(completed) == 1, "missing zero-result source completion"
            assert completed[0]["status"] == 4
            assert (
                completed[0]["completed_blocks"] == completed[0]["completed_bytes"] == 0
            )
            assert not related("nixl.write.posted"), (
                "zero-result operation posted a transfer"
            )
        if scenario in ("success", "partial"):
            posted, done, released = (
                related(name)
                for name in (
                    "nixl.write.posted",
                    "nixl.done_observed",
                    "nixl.write.released",
                )
            )
            assert len(posted) == len(done) == len(released) == 1
            assert all(
                event[field] == posted[0][field]
                for event in (done[0], released[0])
                for field in (
                    "instance_hi",
                    "instance_lo",
                    "trace_id",
                    "source_op_id",
                    "transfer_id",
                )
            ), "mismatched source lifecycle"
            assert posted[0]["start"] <= done[0]["start"] <= released[0]["start"]
            assert (posted[0]["instance_hi"], posted[0]["instance_lo"]) != (
                result["instance_hi"],
                result["instance_lo"],
            ), "source and target share a tracer instance"
            assert posted[0]["thread"] >> 24 != result["thread"] >> 24
            waiters = [
                e
                for e in named["source.pin.waiter"]
                if e["instance_hi"] == posted[0]["instance_hi"]
                and e["instance_lo"] == posted[0]["instance_lo"]
                and e["source_op_id"] == posted[0]["source_op_id"]
            ]
            assert len(waiters) == 1
            pins = [
                e
                for e in named["source.pin.completed"]
                if e["instance_hi"] == waiters[0]["instance_hi"]
                and e["instance_lo"] == waiters[0]["instance_lo"]
                and e["pin_id"] == waiters[0]["pin_id"]
            ]
            assert len(pins) == 1
            assert pins[0]["start"] <= posted[0]["start"]
            if compact:
                native_span = require_range(named, "nixl.write", posted[0])
                source_span = require_range(named, "source.write", posted[0])
                pin_span = require_range(named, "source.pin.wait", pins[0])
                assert native_span["transfer_id"] == posted[0]["transfer_id"]
                assert (
                    native_span["start"]
                    <= posted[0]["start"]
                    <= released[0]["start"]
                    <= native_span["end"]
                )
                assert (
                    source_span["start"]
                    <= native_span["start"]
                    <= native_span["end"]
                    <= source_span["end"]
                )
                assert (
                    pin_span["start"]
                    <= pins[0]["start"]
                    <= pin_span["end"]
                    <= posted[0]["start"]
                )
    counts = Counter(event["name"] for event in events)
    return dict(
        events=len(events),
        event_counts=dict(counts),
        worker_instances=len({(e["instance_hi"], e["instance_lo"]) for e in events}),
        receipt_to_caller_ms=gaps,
        range_counts=dict(
            Counter(event["name"] for event in events if event.get("event_type") == 60)
        ),
        cross_thread_ranges=sum(
            event.get("end_thread") is not None
            and event["thread"] != event["end_thread"]
            for event in events
            if event.get("event_type") == 60
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite")
    parser.add_argument(
        "--scenario",
        choices=("success", "partial", "zero", "failure", "timeout", "off"),
        default="success",
    )
    parser.add_argument("--operations", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(decode(args.sqlite), args.scenario, args.operations), indent=2
        )
    )
