# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""kv.fetch hint handler tests: the engine-destination inline path."""

import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _HintAction,
    _HintEnvelope,
    _new_kvcr,
    _record_hint_results,
    _router_hint,
)

from kvcr.config import KVCRConfig, RemoteFWDramOptions
from kvcr.hints.dispatcher import _LOGGED_HANDLER_ERRORS
from kvcr.hints.parser import _LOGGED_HINT_ISSUES
from kvcr.hints.results import HintStatus
from kvcr.types import BlockKey, CacheTier, QueryStatus


@pytest.fixture(autouse=True)
def _reset_once_logs():
    # Once-per-issue warnings are process-global; reset so tests that count
    # warnings do not depend on run order.
    _LOGGED_HANDLER_ERRORS.clear()
    _LOGGED_HINT_ISSUES.clear()


def _take(results) -> list[tuple[str | None, HintStatus, str | None]]:
    """Statuses reported since the previous take."""
    taken = [(r.action_id, r.status, r.reason) for r in results]
    results.clear()
    return taken


class _PrefixAdapter:
    def decode(self, key):
        return int.from_bytes(key[:8], "big")


def _key(block_hash: int) -> BlockKey:
    return BlockKey(block_hash.to_bytes(8, "big"))


def _fetch_kvcr(monkeypatch):
    kvcr = _new_kvcr(
        FakeNixlAgent(),
        FakePrimaryPinning(),
        FakeBytesControl(),
        KVCRConfig(nixl_agent_name="target", pool_layouts=[("", 16)]),
        key_adapter=_PrefixAdapter(),
        remote_options=RemoteFWDramOptions(
            eager_ctrl_connect=False, opportunistic_query=False
        ),
    )
    return kvcr, _record_hint_results(monkeypatch, kvcr)


def _fetch(action_id: str, source: str, hashes) -> _HintAction:
    return _HintAction(
        action_id, "kv.fetch", "1.0",
        {"source_control_endpoint": source, "block_hashes": list(hashes)},
    )  # fmt: skip


def _fetch_envelope(*hints: _HintAction) -> _HintEnvelope:
    return _HintEnvelope("0.1", "m1", list(hints))


_FETCHABLE = (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
_MISS = (QueryStatus.MISS, None)


class _CountingLock:
    def __init__(self, inner) -> None:
        self._inner = inner
        self.entries = 0

    def __enter__(self):
        self.entries += 1
        return self._inner.__enter__()

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    def acquire(self, *args, **kwargs):
        self.entries += 1
        return self._inner.acquire(*args, **kwargs)

    def release(self):
        return self._inner.release()


def test_gpu_fetch_hint_takes_inline_path_without_lock(monkeypatch):
    kvcr, results = _fetch_kvcr(monkeypatch)
    lock = _CountingLock(kvcr._core._state_lock)
    kvcr._core._state_lock = lock

    kvcr.submit_hint(_router_hint("tcp://a:1", (1,)), "req")

    assert lock.entries == 0
    assert kvcr.query((_key(1),), "req") == [_FETCHABLE]
    assert [(r.status, r.request_id) for r in results] == [(HintStatus.ACCEPTED, "req")]


@pytest.mark.parametrize("one_envelope", [True, False], ids=["envelope", "calls"])
@pytest.mark.parametrize(
    ("second_source", "expected"),
    [("tcp://a:1", [_MISS, _FETCHABLE]), ("tcp://b:1", [_MISS, _MISS])],
    ids=["same_source_replaces", "other_source_fails"],
)
def test_second_fetch_for_a_request(monkeypatch, one_envelope, second_source, expected):
    kvcr, results = _fetch_kvcr(monkeypatch)
    first, second = _fetch("a1", "tcp://a:1", [1]), _fetch("a2", second_source, [2])
    for hints in [(first, second)] if one_envelope else [(first,), (second,)]:
        kvcr.submit_hint(_fetch_envelope(*hints), "req")
    # The store's rules apply within an envelope as across calls, and it reports
    # neither the replacement nor the failure.
    assert _take(results) == [
        ("a1", HintStatus.ACCEPTED, None),
        ("a2", HintStatus.ACCEPTED, None),
    ]
    assert kvcr.query((_key(1), _key(2)), "req") == expected


def test_mixed_valid_and_unknown_hints(monkeypatch):
    kvcr, results = _fetch_kvcr(monkeypatch)
    kvcr.submit_hint(
        _fetch_envelope(
            _fetch("a1", "tcp://a:1", [1]),
            _HintAction("f1", "kv.future_action", "1.0", {}),
        ),
        "req",
    )
    assert kvcr.query((_key(1),), "req") == [_FETCHABLE]
    assert set(_take(results)) == {
        ("a1", HintStatus.ACCEPTED, None),
        ("f1", HintStatus.REJECTED, "unsupported_action_type"),
    }


def test_all_invalid_envelope_does_not_raise_from_submit_hint(monkeypatch):
    kvcr, results = _fetch_kvcr(monkeypatch)
    bad = _HintAction("a1", "kv.fetch", "1.0", {"block_hashes": []})
    kvcr.submit_hint(_fetch_envelope(bad), "req")
    assert [r.status for r in results] == [HintStatus.REJECTED]


def test_fetch_hint_without_request_id_is_accepted(monkeypatch):
    kvcr, results = _fetch_kvcr(monkeypatch)
    kvcr.submit_hint(_router_hint("tcp://a:1", (1,)))
    assert [(r.status, r.request_id) for r in results] == [(HintStatus.ACCEPTED, None)]
    assert not kvcr._core._remote_fw_dram._request_hints
