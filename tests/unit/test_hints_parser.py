# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from typing import ClassVar

import msgspec
import pytest
from _kvcr_test_utils import _HintAction, _HintEnvelope, _router_hint

from kvcr.hints.fetch import KV_FETCH, _KVFetchHint
from kvcr.hints.parser import _LOGGED_HINT_ISSUES, HintType, parse_hints
from kvcr.hints.results import HintStatus


def _no_handler(core, dispatcher):
    """Handler factory for parser-only types: parsing must never build a handler."""
    raise AssertionError("parser tests build no handlers")


def _types(*extra: HintType) -> dict[str, HintType]:
    """``kv.fetch`` plus ``extra``, keyed by action type like the dispatcher's."""
    return {t.action_type: t for t in (KV_FETCH, *extra)}


def _fetch_hint(action_id, source, hashes, **extra) -> _HintAction:
    payload = {"source_control_endpoint": source, "block_hashes": list(hashes)}
    payload.update(extra)
    return _HintAction(action_id, "kv.fetch", "1.0", payload)


def test_parse_hints_returns_every_valid_action_in_order():
    hint = _HintEnvelope(
        "0.1",
        "m",
        [
            _fetch_hint("a", "tcp://a:1", (1, 2, 1)),
            _HintAction("x", "kv.future_action", "1.0", {}),
            _fetch_hint("bad", "", (3,)),
            _fetch_hint("b", "tcp://b:1", (4,)),
        ],
    )
    hints, rejected = parse_hints(hint, _types())
    assert hints == [
        _KVFetchHint("a", "tcp://a:1", frozenset({1, 2})),
        _KVFetchHint("b", "tcp://b:1", frozenset({4})),
    ]
    assert [(r.action_id, r.status, r.reason) for r in rejected] == [
        ("x", HintStatus.REJECTED, "unsupported_action_type"),
        ("bad", HintStatus.REJECTED, "invalid router hint"),
    ]
    assert all(r.message_id == "m" for r in rejected)
    assert [r.action_type for r in rejected] == ["kv.future_action", "kv.fetch"]


@pytest.mark.parametrize("as_mapping", [False, True])
def test_parse_hints_accepts_struct_and_mapping(as_mapping):
    hint = _router_hint("tcp://source:1234", (123, 456, 123))
    hint = msgspec.to_builtins(hint) if as_mapping else hint
    hints, rejected = parse_hints(hint, _types())
    assert len(hints) == 1 and rejected == []
    assert hints[0].source_endpoint == "tcp://source:1234"
    assert hints[0].block_hashes == frozenset({123, 456})


def test_parse_hints_non_mapping_action_and_payload():
    hint = _HintEnvelope(
        "0.1",
        7,
        [None, _HintAction("a", "kv.fetch", "1.0", "bad")],
    )
    hints, rejected = parse_hints(hint, _types())
    assert hints == []
    assert [(r.action_id, r.action_type, r.reason, r.message_id) for r in rejected] == [
        (None, "unknown", "invalid_hint", None),
        ("a", "kv.fetch", "invalid_payload", None),
    ]


@pytest.mark.parametrize("hint", [None, {}, _HintEnvelope("0.1", "m", "bad")])
def test_parse_hints_rejects_invalid_envelope(hint):
    with pytest.raises(ValueError, match="invalid router hint"):
        parse_hints(hint, _types())


def test_parse_hints_warns_once_per_mismatched_version_field(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _LOGGED_HINT_ISSUES.clear()
    kv_hint = _router_hint("tcp://source:1234", (123,))
    with caplog.at_level(logging.WARNING):
        for protocol, version in (("9.0", "2.0"), ("10.0", "3.0")):
            changed = msgspec.structs.replace(
                kv_hint,
                protocol_version=protocol,
                actions=[
                    msgspec.structs.replace(kv_hint.actions[0], action_version=version)
                ],
            )
            hints, _ = parse_hints(changed, _types())

    assert len(hints) == 1
    assert caplog.text.count("KV hint protocol_version mismatch") == 1
    assert caplog.text.count("kv.fetch action_version mismatch") == 1


def test_rejection_dedupe_is_bounded_for_unregistered_types(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _LOGGED_HINT_ISSUES.clear()
    hint = _HintEnvelope(
        "0.1",
        "m",
        [_HintAction(str(i), f"kv.future_{i}", "1.0", {}) for i in range(100)],
    )
    with caplog.at_level(logging.WARNING):
        parse_hints(hint, _types())
        _, rejected = parse_hints(hint, _types())
    # Every hint is still reported with its own type; only the log is shared.
    assert [r.action_type for r in rejected] == [f"kv.future_{i}" for i in range(100)]
    assert len(_LOGGED_HINT_ISSUES) == 1
    assert caplog.text.count("unsupported_action_type") == 1


@pytest.mark.parametrize("action_id", [None, 7, ["a"]])
def test_non_string_action_id_is_rejected(action_id) -> None:
    hint = _HintEnvelope(
        "0.1",
        "m",
        [
            _HintAction(action_id, "kv.fetch", "1.0", {}),
            _fetch_hint("ok", "tcp://a:1", (1,)),
        ],
    )
    hints, rejected = parse_hints(hint, _types())
    assert [h.action_id for h in hints] == ["ok"]
    assert [(r.action_id, r.action_type, r.reason) for r in rejected] == [
        (None, "unknown", "invalid_hint")
    ]


def test_parser_exception_rejects_only_its_hint() -> None:
    def parse(action_id, payload):
        raise TypeError("parser bug")

    types = _types(HintType("kv.broken", "1.0", parse, _no_handler))
    hint = _HintEnvelope(
        "0.1",
        "m",
        [
            _HintAction("x", "kv.broken", "1.0", {}),
            _fetch_hint("ok", "tcp://a:1", (1,)),
        ],
    )
    hints, rejected = parse_hints(hint, types)
    assert [h.action_id for h in hints] == ["ok"]
    assert [(r.action_id, r.reason) for r in rejected] == [("x", "invalid_hint")]


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
def test_kv_fetch_invalid_payload_is_rejected(fetch_payload: object) -> None:
    kv_hint = _router_hint("tcp://source:1")
    kv_hint.actions[0] = msgspec.structs.replace(
        kv_hint.actions[0], payload=fetch_payload
    )
    hints, rejected = parse_hints(kv_hint, _types())
    assert hints == []
    assert len(rejected) == 1 and rejected[0].status is HintStatus.REJECTED


def test_supplied_hint_type_is_parsed():
    class _Stub(msgspec.Struct, frozen=True):
        ACTION_TYPE: ClassVar[str] = "kv.stub"
        action_id: str | None
        n: int

    def parse(action_id, payload):
        if "n" not in payload:
            raise ValueError("missing n")
        return _Stub(action_id, payload["n"])

    types = _types(HintType("kv.stub", "1.0", parse, _no_handler))
    hint = _HintEnvelope(
        "0.1",
        "m",
        [
            _HintAction("s", "kv.stub", "1.0", {"n": 3}),
            _HintAction("t", "kv.stub", "1.0", {}),
        ],
    )
    hints, rejected = parse_hints(hint, types)
    assert hints == [_Stub("s", 3)]
    assert [(r.action_id, r.reason) for r in rejected] == [("t", "missing n")]
