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
    )
    events = [
        dict(common, name=name, start=index)
        for index, name in enumerate(
            (
                "target.queued",
                "target.write_done.received",
                "op.completion_queued",
                "op.completion_returned",
            )
        )
    ]
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


def test_decoder_accepts_associated_unicode_context():
    validate(capture_events(), "failure", 1)


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
