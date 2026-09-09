# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from typing import cast

import pytest
from _kvcr_test_utils import _router_hint

from kvcr.hint_parser import _LOGGED_HINT_ISSUES, _KVFetchHint


def test_kv_fetch_hint_parses_first_fetch_action() -> None:
    fetch_hint = _router_hint("tcp://source:1234", (123, 456, 123))
    later_fetch_hint = _router_hint("tcp://other:1234", (789,))
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
            later_fetch_hint["actions"][0],
        ],
    }

    assert _KVFetchHint.from_hint(payload) == _KVFetchHint(
        source_endpoint="tcp://source:1234",
        block_hashes=frozenset({123, 456}),
    )


def test_kv_fetch_hint_warns_once_per_mismatched_version_field(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _LOGGED_HINT_ISSUES.clear()
    kv_hint = _router_hint("tcp://source:1234", (123,))
    action = cast(list[dict[str, object]], kv_hint["actions"])[0]
    kv_hint["protocol_version"] = "9.0"
    action["action_version"] = "2.0"

    with caplog.at_level(logging.WARNING):
        parsed = _KVFetchHint.from_hint(kv_hint)
        kv_hint["protocol_version"] = "10.0"
        action["action_version"] = "3.0"
        _KVFetchHint.from_hint(kv_hint)

    assert parsed == _KVFetchHint("tcp://source:1234", frozenset({123}))
    assert caplog.text.count("KV hint protocol_version mismatch") == 1
    assert caplog.text.count("kv.fetch action_version mismatch") == 1


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
def test_kv_fetch_hint_rejects_invalid_envelope(kv_hint: object) -> None:
    with pytest.raises(ValueError, match="invalid router hint"):
        _KVFetchHint.from_hint(kv_hint)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fetch_payload",
    [
        {"block_hashes": [1]},
        {"source_control_endpoint": "", "block_hashes": [1]},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": []},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": [True]},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": [-1]},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": [1 << 64]},
        {
            "source_control_endpoint": "tcp://source:1",
            "block_hashes": [1],
            "mode": "move",
        },
        {
            "source_control_endpoint": "tcp://source:1",
            "block_hashes": [1],
            "no_retain": True,
        },
    ],
)
def test_kv_fetch_hint_rejects_invalid_payload(fetch_payload: object) -> None:
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

    with pytest.raises(ValueError):
        _KVFetchHint.from_hint(kv_hint)
