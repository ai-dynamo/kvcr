# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from kvcr.hint_parser import _parse_kv_hint


def test_parse_kv_hint_extracts_kvcr_fields() -> None:
    payload = {
        "source_control_endpoint": "tcp://source:1234",
        "block_hashes": [123, 456, 123],
        "framework_hint": {"kept_out_of_kvcr": True},
    }

    assert _parse_kv_hint(payload) == (
        "tcp://source:1234",
        frozenset({123, 456}),
        "copy",
    )

    assert _parse_kv_hint(
        {"block_hashes": [123], "mode": "move", "no_retain": True}
    ) == (None, frozenset({123}), "move")


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"source_control_endpoint": "", "block_hashes": [1]},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": []},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": [True]},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": [-1]},
        {
            "source_control_endpoint": "tcp://source:1",
            "block_hashes": [1 << 64],
        },
        {
            "source_control_endpoint": "tcp://source:1",
            "block_hashes": [1],
            "mode": "invalid",
        },
        {"block_hashes": [1], "no_retain": "yes"},
    ],
)
def test_parse_kv_hint_rejects_invalid_protocol(payload: object) -> None:
    with pytest.raises(ValueError, match="invalid router hint"):
        _parse_kv_hint(payload)  # type: ignore[arg-type]
