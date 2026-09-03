# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from collections.abc import Callable

import pytest
from _kvcr_test_utils import _router_hint

from kvcr.hint_parser import _LOGGED_HINT_ISSUES, _parse_kv_hint


def test_parse_kv_hint_extracts_first_fetch_action() -> None:
    fetch_hint = _router_hint("tcp://source:1234", (123, 456, 123))
    payload = {
        "protocol_version": "0.1",
        "message_id": "test-message",
        "actions": [
            {
                "action_id": "ignored",
                "action_type": "kv.future_action",
                "action_version": "9.0",
                "payload": {"opaque": True},
            },
            fetch_hint["actions"][0],
        ],
    }

    assert _parse_kv_hint(payload) == (
        "tcp://source:1234",
        frozenset({123, 456}),
        "copy",
    )


@pytest.mark.parametrize(
    ("mutate_hint", "warning_text"),
    [
        (
            lambda kv_hint: kv_hint.__setitem__("protocol_version", "9.0"),
            "KV hint protocol_version mismatch",
        ),
        (
            lambda kv_hint: kv_hint["actions"][0].__setitem__(
                "action_version", "2.0"
            ),
            "kv.fetch action_version mismatch",
        ),
    ],
)
def test_parse_kv_hint_warns_but_reads_mismatched_versions(
    caplog: pytest.LogCaptureFixture,
    mutate_hint: Callable[[dict[str, object]], None],
    warning_text: str,
) -> None:
    """Treat envelope/action version mismatches as advisory, not fatal."""
    _LOGGED_HINT_ISSUES.clear()
    kv_hint = _router_hint("tcp://source:1234", (123,))
    mutate_hint(kv_hint)

    with caplog.at_level(logging.WARNING):
        parsed = _parse_kv_hint(kv_hint)

    assert parsed == ("tcp://source:1234", frozenset({123}), "copy")
    assert warning_text in caplog.text


@pytest.mark.parametrize(
    "kv_hint",
    [
        None,
        {},
        {"protocol_version": "0.1", "actions": "bad"},
        {
            "protocol_version": "0.1",
            "actions": [
                {
                    "action_id": "a1",
                    "action_type": "kv.future_action",
                    "action_version": "1.0",
                    "payload": {},
                }
            ],
        },
        {
            "protocol_version": "0.1",
            "actions": [
                {
                    "action_id": "a1",
                    "action_type": "kv.fetch",
                    "action_version": "1.0",
                    "payload": "bad",
                }
            ],
        },
    ],
)
def test_parse_kv_hint_rejects_invalid_protocol(kv_hint: object) -> None:
    with pytest.raises(ValueError, match="invalid router hint"):
        _parse_kv_hint(kv_hint)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fetch_payload",
    [
        {"source_control_endpoint": "", "block_hashes": [1]},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": []},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": [True]},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": [-1]},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": [1 << 64]},
        {
            "source_control_endpoint": "tcp://source:1",
            "block_hashes": [1],
            "mode": "invalid",
        },
        {"block_hashes": [1], "no_retain": "yes"},
    ],
)
def test_parse_kv_hint_rejects_invalid_fetch_payload(fetch_payload: object) -> None:
    """Reject malformed payloads on a matching ``kv.fetch`` action."""
    kv_hint = _router_hint("tcp://source:1")
    kv_hint["actions"] = [
        {
            "action_id": "a1",
            "action_type": "kv.fetch",
            "action_version": "1.0",
            "payload": fetch_payload,
        }
    ]

    with pytest.raises(ValueError, match="invalid router hint"):
        _parse_kv_hint(kv_hint)
