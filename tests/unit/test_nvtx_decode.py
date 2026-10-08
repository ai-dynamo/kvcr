# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The capture checker must reject missing or misassociated request mappings."""

import runpy
from hashlib import blake2b
from pathlib import Path

import pytest

validate = runpy.run_path(str(Path(__file__).parents[2] / "examples/nvtx_decode.py"))[
    "validate"
]


def capture_events():
    encoded = "capture-α-😀-0".encode()
    digest = blake2b(encoded, digest_size=16).digest()
    common = dict(
        instance_hi=1,
        instance_lo=2,
        trace_id=3,
        target_agent_hi=4,
        target_agent_lo=5,
        target_incarnation_hi=6,
        target_incarnation_lo=7,
        op_handle=8,
        status=4,
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
    events.append(
        dict(
            common,
            name="request.context",
            value_utf8=list(encoded),
            length=len(encoded),
            truncated=0,
        )
    )
    return events


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


def test_decoder_accepts_replacement_of_an_older_hint():
    events = capture_events()
    submitted = next(e for e in events if e["name"] == "hint.submitted")
    events.extend(
        [
            dict(submitted, trace_id=19, start=1),
            dict(submitted, trace_id=19, name="hint.replaced", start=5),
        ]
    )
    validate(events, "failure", 1)


def test_decoder_accepts_associated_unicode_context():
    validate(capture_events(), "failure", 1)


def successful_events():
    events = capture_events()
    result = next(e for e in events if e["name"] == "op.completion_returned")
    result["status"] = 2
    source = dict(
        result,
        instance_hi=11,
        instance_lo=12,
        trace_id=30,
        source_op_id=2,
        transfer_id=40,
        thread=2 << 24,
    )
    events.extend(
        dict(source, name=name, start=33 + index)
        for index, name in enumerate(
            ("nixl.write.posted", "nixl.done_observed", "nixl.write.released")
        )
    )
    events.extend(
        [
            dict(source, name="source.pin.waiter", pin_id=50, start=31),
            dict(source, name="source.pin.completed", pin_id=50, start=32),
        ]
    )
    return events


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
    if damage == "removed":
        events.pop()
    elif damage == "wrong_trace":
        events[-1]["trace_id"] += 1
    else:
        events[-1]["instance_lo"] += 1
    with pytest.raises(AssertionError, match="request context"):
        validate(events, "failure", 1)
