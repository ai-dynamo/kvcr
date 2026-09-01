# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from kvcr.kv_hints import KvSourceLocationsHint, extract_kv_hint


def _router_hint(block_hashes=None):
    return {
        "source_control_endpoint": "tcp://source:1234",
        "block_hashes": block_hashes or [123, b"abc"],
    }


def test_extract_kv_hint_returns_source_locations_hint():
    hint = KvSourceLocationsHint("tcp://source:1234", frozenset({123, b"abc"}))

    assert extract_kv_hint({"router_hint": _router_hint()}) == hint


def test_extract_kv_hint_returns_none_without_router_hint():
    assert extract_kv_hint(None) is None
    assert extract_kv_hint({}) is None


def test_extract_kv_hint_rejects_invalid_payloads():
    assert extract_kv_hint({"router_hint": {}}) is None
    assert extract_kv_hint(
        {
            "router_hint": {
                "source_control_endpoint": "tcp://source:1234",
                "block_hashes": [],
            }
        }
    ) is None
    assert extract_kv_hint(
        {
            "router_hint": {
                "source_control_endpoint": "tcp://source:1234",
                "block_hashes": [True],
            }
        }
    ) is None
