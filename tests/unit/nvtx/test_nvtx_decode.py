# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The capture checker must reject missing or misassociated request mappings."""

import json
import runpy
import sqlite3
from hashlib import blake2b
from pathlib import Path

import pytest

decoder = runpy.run_path(str(Path(__file__).parents[3] / "examples/nvtx_decode.py"))
validate, decode, require_range, reconstruct_context = (
    decoder[name]
    for name in ("validate", "decode", "require_range", "reconstruct_context")
)


@pytest.mark.parametrize("range_id", [None, 0, 7])
def test_range_validation_requires_a_native_id_and_accepts_zero(range_id):
    event = dict(
        instance_hi=1,
        instance_lo=2,
        trace_id=3,
        event_type=60,
        range_id=range_id,
        start=10,
        end=20,
        thread=1 << 24,
        end_thread=1 << 24,
    )
    if range_id is None:
        with pytest.raises(AssertionError, match="range ID"):
            require_range({"target.remote": [event]}, "target.remote", event)
    else:
        span = require_range({"target.remote": [event]}, "target.remote", event)
        assert span["range_id"] == range_id


def compact_sqlite(path, damage=None):
    # Context intentionally follows the range start. Association overrides are
    # dynamic: a hint context may be used with several operation handles.
    prefix = dict(schema_version=3, instance_hi=1, instance_lo=2, trace_id=3)
    context = dict(prefix, op_handle=0, target_agent_hi=4, request_known=0)
    rows = [
        (10, 50, 1, 2, 60, 7, 1, json.dumps(dict(prefix, op_handle=41))),
        (20, None, 1, None, 34, None, 2, json.dumps(context)),
        (30, None, 1, None, 34, None, 3, json.dumps(dict(prefix, op_handle=42))),
    ]
    if damage == "missing":
        rows.pop(1)
    elif damage == "duplicate":
        rows.append(rows[1])
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE StringIds(id INTEGER,value TEXT)")
        connection.executemany(
            "INSERT INTO StringIds VALUES(?,?)",
            [(1, "target.remote"), (2, "request.context"), (3, "hint.used")],
        )
        connection.execute(
            "CREATE TABLE NVTX_EVENTS(start,end,globalTid,endGlobalTid,eventType,"
            "rangeId,textId,jsonText)"
        )
        connection.executemany("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?,?,?,?)", rows)


def test_decoder_reconstructs_late_context_and_keeps_dynamic_associations(tmp_path):
    path = tmp_path / "capture.sqlite"
    compact_sqlite(path)
    events = decode(path)
    assert [event["op_handle"] for event in events] == [41, 0, 42]
    assert all(event["target_agent_hi"] == 4 for event in events)
    assert events[0]["end_thread"] == 2
    assert events[0]["event_type"] == 60
    assert events[0]["range_id"] == 7


@pytest.mark.parametrize("damage", ["missing", "duplicate"])
def test_decoder_rejects_unresolvable_compact_context(tmp_path, damage):
    path = tmp_path / "capture.sqlite"
    compact_sqlite(path, damage)
    with pytest.raises(AssertionError, match="context"):
        decode(path)


@pytest.mark.parametrize("product", [None, "other", "NVIDIA Nsight Systems"])
def test_decoder_handles_exports_without_nvtx_events(tmp_path, product):
    path = tmp_path / "empty.sqlite"
    with sqlite3.connect(path) as connection:
        if product is not None:
            connection.execute("CREATE TABLE META_DATA_EXPORT(name TEXT,value TEXT)")
            connection.execute(
                "INSERT INTO META_DATA_EXPORT VALUES('EXPORT_PRODUCT_NAME',?)",
                (product,),
            )
    if product is None:
        with pytest.raises(sqlite3.OperationalError):
            decode(path)
    elif product != "NVIDIA Nsight Systems":
        with pytest.raises(AssertionError, match="Nsight Systems export"):
            decode(path)
    else:
        events = decode(path)
        assert validate(events, "off", 1) == {"events": 0}
        with pytest.raises(AssertionError):
            validate(events, "success", 1)


def capture_events():
    encoded = "capture-α-😀-0".encode()
    digest = blake2b(encoded, digest_size=16).digest()
    common = dict(
        schema_version=3,
        request_display_known=1,
        selected_bytes=-1,
        instance_hi=1,
        instance_lo=2,
        trace_id=3,
        target_agent_hi=4,
        target_agent_lo=5,
        target_incarnation_hi=6,
        target_incarnation_lo=7,
        op_handle=8,
        status=4,
        requested_blocks=2,
        requested_bytes=32,
        completed_blocks=0,
        completed_bytes=0,
        request_known=1,
        request_hi=int.from_bytes(digest[:8], "big"),
        request_lo=int.from_bytes(digest[8:], "big"),
        session_known=0,
        parent_session_known=0,
        thread=1 << 24,
    )
    events = [
        dict(common, name=name, start=30 + index * 10)
        for index, name in enumerate(
            (
                "target.queued",
                "target.write_done.received",
                "op.completion_queued",
                "op.completion_returned",
            )
        )
    ]
    events[0].update(trace_id=9, hint_trace_id=20)
    events[1]["trace_id"] = 9
    hint = dict(
        common,
        trace_id=20,
        target_agent_hi=0,
        target_agent_lo=0,
        target_incarnation_hi=0,
        target_incarnation_lo=0,
    )
    events.extend(
        [
            dict(hint, name="hint.submitted", op_handle=0, start=10),
            dict(hint, name="hint.used", start=20),
        ]
    )
    result = next(e for e in events if e["name"] == "op.completion_returned")
    branch = next(e for e in events if e["name"] == "target.queued")
    source = dict(
        result,
        instance_hi=11,
        instance_lo=12,
        trace_id=30,
        source_op_id=2,
        request_known=0,
        request_display_known=0,
        thread=2 << 24,
    )
    pin = dict(source, trace_id=50, pin_id=50)
    events.extend(
        [
            dict(source, name="source.write.completed", start=35),
            dict(pin, name="source.pin.waiter", start=31),
            dict(pin, name="source.pin.completed", start=32),
        ]
    )
    for name, reference, start, end, end_thread in (
        ("op.deliver.lifecycle", result, 25, 65, result["thread"]),
        ("target.remote", branch, 28, 45, branch["thread"] + 1),
        ("source.write", source, 33, 36, source["thread"]),
        ("source.pin.wait", pin, 29, 33, pin["thread"]),
    ):
        events.append(
            dict(
                reference,
                name=name,
                start=start,
                end=end,
                end_thread=end_thread,
                event_type=60,
                range_id=0,
            )
        )
    contexts = {}
    mark = dict(end=None, end_thread=None, event_type=34, range_id=None)
    for event in events:
        for field, value in mark.items():
            event.setdefault(field, value)
        contexts.setdefault(
            (event["instance_hi"], event["instance_lo"], event["trace_id"]), event
        )
    events.extend(
        dict(
            context,
            value_utf8=list(encoded),
            length=len(encoded),
            truncated=0,
            name="source.pin.context"
            if context["trace_id"] == 50
            else "request.context",
            start=0,
            **mark,
        )
        for context in contexts.values()
    )
    return reconstruct_context(events)


@pytest.mark.parametrize(
    "damage",
    [
        "submitted_removed",
        "used_removed",
        "wrong_hint",
        "wrong_instance",
        "wrong_branch_instance",
        "wrong_request",
        "wrong_operation",
        "replaced",
        "rejected",
        "discarded",
    ],
)
def test_decoder_rejects_broken_hint_association(damage):
    events = capture_events()
    submitted = next(e for e in events if e["name"] == "hint.submitted")
    used = next(e for e in events if e["name"] == "hint.used")
    target = next(e for e in events if e["name"] == "target.queued")
    if damage == "submitted_removed":
        events.remove(submitted)
    elif damage == "used_removed":
        events.remove(used)
    elif damage == "wrong_hint":
        target["hint_trace_id"] += 1
    elif damage == "wrong_instance":
        used["instance_lo"] += 1
    elif damage == "wrong_branch_instance":
        for event in (submitted, used, target):
            event["instance_lo"] += 1
    elif damage == "wrong_request":
        used["request_hi"] += 1
    elif damage == "wrong_operation":
        used["op_handle"] += 1
    else:
        events.append(dict(submitted, name="hint." + damage, start=15))
    with pytest.raises(AssertionError, match="hint"):
        validate(events, "failure", 1)


@pytest.mark.parametrize("schema", [None, 1, 2, 4])
def test_decoder_rejects_unsupported_schema(schema):
    events = capture_events()
    for event in events:
        event["schema_version"] = schema
    with pytest.raises(AssertionError, match="unsupported payload schema"):
        reconstruct_context(events)


def test_decoder_rejects_incomplete_hint_identity():
    events = capture_events()
    submitted = next(event for event in events if event["name"] == "hint.submitted")
    malformed = dict(submitted, instance_hi=999, instance_lo=999)
    del malformed["trace_id"]
    events.append(malformed)
    with pytest.raises(KeyError, match="trace_id"):
        validate(events, "failure", 1)


@pytest.mark.parametrize("replacement", [False, True])
def test_decoder_accepts_associated_unicode_context(replacement):
    events = capture_events()
    submitted = next(e for e in events if e["name"] == "hint.submitted")
    if replacement:
        events.extend(
            [
                dict(submitted, trace_id=19, start=1),
                dict(submitted, trace_id=19, name="hint.replaced", start=5),
            ]
        )
    validate(events, "failure", 1)


@pytest.mark.parametrize("scenario", ["zero", "failure", "timeout"])
@pytest.mark.parametrize("count", [-1, 0])
def test_decoder_accepts_closed_source_ranges_on_failed_results(scenario, count):
    events = capture_events()
    pin = next(e for e in events if e["name"] == "source.pin.completed")
    pin["status"] = 5 if scenario == "timeout" else 4
    pin.update(completed_blocks=count, completed_bytes=count)
    validate(events, scenario, 1)


def test_decoder_rejects_duplicate_returned_operation():
    events = capture_events()
    validate(events, "failure", 1)
    result = next(e for e in events if e["name"] == "op.completion_returned")
    events.append(dict(result))
    with pytest.raises(AssertionError, match="duplicate returned operation"):
        validate(events, "failure", 2)


@pytest.mark.parametrize("identity", ["trace", "instance", "duplicate"])
def test_decoder_requires_distinct_caller_lifecycles(identity):
    events, second = capture_events(), capture_events()
    fields = ("source_op_id", "pin_id")
    fields += (
        ("instance_lo", "target_incarnation_lo")
        if identity == "instance"
        else ("trace_id", "hint_trace_id", "op_handle")
    )
    for event in second:
        for field in fields:
            if event.get(field):
                event[field] += 100
    if identity == "duplicate":
        next(e for e in second if e["name"] == "op.completion_returned")["trace_id"] = 3
    events.extend(second)
    if identity == "duplicate":
        with pytest.raises(AssertionError, match="duplicate caller lifecycle"):
            validate(events, "failure", 2)
    else:
        validate(events, "failure", 2)


@pytest.mark.parametrize("scenario", ["zero", "failure", "timeout"])
@pytest.mark.parametrize("name", ["source.write", "source.pin.wait"])
def test_decoder_rejects_missing_source_ranges_on_failed_results(scenario, name):
    events = [e for e in capture_events() if e["name"] != name]
    with pytest.raises(AssertionError, match=name):
        validate(events, scenario, 1)


def successful_events(scenario="success"):
    events = capture_events()
    completed = 2 if scenario == "success" else 1
    for event in events:
        event.update(
            status=2 if scenario == "success" else 3,
            completed_blocks=completed,
            completed_bytes=16 * completed,
        )
    source = next(e for e in events if e["name"] == "source.write.completed")
    source.update(transfer_id=40, selected_bytes=16 * completed)
    events.extend(
        dict(source, name=name, start=33 + index)
        for index, name in enumerate(
            ("nixl.write.posted", "nixl.done_observed", "nixl.write.released")
        )
    )
    span = next(e for e in events if e["name"] == "source.write")
    events.append(dict(span, name="nixl.write", end=35, transfer_id=40, range_id=1))
    return events


@pytest.mark.parametrize("scenario", ["success", "partial"])
@pytest.mark.parametrize(
    "field", [None, "instance_hi", "instance_lo", "trace_id", "source_op_id"]
)
def test_decoder_matches_compact_native_and_logical_source(scenario, field):
    events = successful_events(scenario=scenario)
    if field is None:
        validate(reconstruct_context(events), scenario, 1)
        return
    for event in events:
        if event["name"].startswith("nixl."):
            event[field] += 1
    # Keep the disconnected native lifecycle's context and pin joins valid.
    if field != "source_op_id":
        context = next(
            e for e in events if e["name"] == "request.context" and e["trace_id"] == 30
        )
        events.append(dict(context, **{field: context[field] + 1}))
    if field != "trace_id":
        names = ("source.pin.waiter",)
        if field != "source_op_id":
            names += ("source.pin.completed", "source.pin.context")
        events.extend(
            [dict(e, **{field: e[field] + 1}) for e in events if e["name"] in names]
        )
    with pytest.raises(AssertionError, match="mismatched source lifecycle"):
        validate(reconstruct_context(events), scenario, 1)


@pytest.mark.parametrize("name", ["source.write", "nixl.write"])
def test_decoder_rejects_compact_range_with_wrong_source_operation(name):
    events = successful_events()
    next(e for e in events if e["name"] == name)["source_op_id"] += 1
    with pytest.raises(AssertionError, match="mismatched source lifecycle"):
        validate(events, "success", 1)


@pytest.mark.parametrize("name", ["nixl.done_observed", "nixl.write.released"])
@pytest.mark.parametrize(
    "field", ["instance_hi", "instance_lo", "trace_id", "source_op_id", "transfer_id"]
)
def test_decoder_rejects_mismatched_source_lifecycle(name, field):
    events = successful_events()
    next(e for e in events if e["name"] == name)[field] += 1
    with pytest.raises(AssertionError, match="source lifecycle"):
        validate(events, "success", 1)


def test_decoder_accepts_distinct_source_uuid_with_matching_high_half():
    events = successful_events()
    for event in events:
        if event["thread"] == 2 << 24:
            event["instance_hi"] = 1
    validate(events, "success", 1)


@pytest.mark.parametrize("damage", ["removed", "wrong_trace", "wrong_instance"])
def test_decoder_rejects_missing_context_association(damage):
    events = capture_events()
    context = next(
        e for e in events if e["name"] == "request.context" and e["trace_id"] == 3
    )
    if damage == "removed":
        events.remove(context)
    elif damage == "wrong_trace":
        context["trace_id"] += 1
    else:
        context["instance_lo"] += 1
    with pytest.raises(AssertionError, match="request context"):
        validate(events, "failure", 1)


@pytest.mark.parametrize("name", ["op.completion_queued", "target.write_done.received"])
@pytest.mark.parametrize("field", ["instance_hi", "instance_lo", "trace_id"])
def test_decoder_rejects_mismatched_target_lifecycle(name, field):
    events = capture_events()
    next(e for e in events if e["name"] == name)[field] += 1
    with pytest.raises(AssertionError, match="mismatched target lifecycle"):
        validate(events, "failure", 1)


@pytest.mark.parametrize("scenario", ["success", "partial", "zero"])
@pytest.mark.parametrize(
    "name",
    [
        "source.pin.completed",
        "source.write.completed",
        "target.write_done.received",
        "op.completion_queued",
        "op.completion_returned",
    ],
)
@pytest.mark.parametrize("field", ["status", "completed_blocks", "completed_bytes"])
def test_decoder_rejects_contradictory_completion(scenario, name, field):
    events = (
        capture_events() if scenario == "zero" else successful_events(scenario=scenario)
    )
    next(e for e in events if e["name"] == name)[field] = (
        2 if scenario == "zero" else 4 if field == "status" else 0
    )
    with pytest.raises(AssertionError):
        validate(events, scenario, 1)


@pytest.mark.parametrize("scenario", ["success", "partial"])
@pytest.mark.parametrize("detail", ["low", "medium"])
def test_decoder_accepts_unknown_byte_counts(scenario, detail):
    events = successful_events(scenario=scenario)
    for event in events:
        if detail == "low":
            event.update(
                requested_bytes=-1,
                selected_bytes=-1,
                completed_bytes=-1,
                request_display_known=0,
            )
        elif event["trace_id"] in (3, 50):
            event["completed_bytes"] = -1  # Caller/pin bytes may be unknown.
    validate(events, scenario, 1)
