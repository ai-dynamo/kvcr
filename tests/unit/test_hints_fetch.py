# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Request advice and independent prefetch/move hint execution."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _decode_control_message,
    _HintAction,
    _HintEnvelope,
    _invalidate,
    _new_kvcr,
    _poll_until,
    _record_hint_results,
    _router_hint,
    _wait_until,
)
from test_kvcr_remote_target import _Pair

from kvcr.backends.remote_ops import _FetchGroup, _KeysToFetch, _TargetPullOp
from kvcr.config import KVCRConfig, RemoteFWDramOptions
from kvcr.hints.dispatcher import _LOGGED_HANDLER_ERRORS
from kvcr.hints.parser import _LOGGED_HINT_ISSUES
from kvcr.hints.results import HintStatus
from kvcr.types import BlockKey, CacheTier, OpEntryStatus, QueryStatus


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
    def encode_to_router(self, key):
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


def _fetch(action_id: str, source: str, hashes, action_type="kv.fetch") -> _HintAction:
    return _HintAction(
        action_id, action_type, "1.0",
        {"source_control_endpoint": source, "block_hashes": list(hashes)},
    )  # fmt: skip


def _independent(action_id: str, destination: str, hashes, action_type) -> _HintAction:
    return _HintAction(
        action_id, action_type, "1.0",
        {"destination_control_endpoint": destination, "block_hashes": list(hashes)},
    )  # fmt: skip


def _envelope(*hints: _HintAction) -> _HintEnvelope:
    return _HintEnvelope("0.1", "m1", list(hints))


_FETCHABLE = (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
_MISS = (QueryStatus.MISS, None)


def test_fetch_hint_takes_inline_path_without_lock(monkeypatch):
    kvcr, results = _fetch_kvcr(monkeypatch)
    with monkeypatch.context() as patch:
        lock = MagicMock()
        patch.setattr(kvcr._core, "_state_lock", lock)
        kvcr.submit_hint(_router_hint("tcp://a:1", (1,)), "req")
        assert lock.mock_calls == []

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
        kvcr.submit_hint(_envelope(*hints), "req")
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
        _envelope(
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
    kvcr.submit_hint(_envelope(bad), "req")
    assert [r.status for r in results] == [HintStatus.REJECTED]


def test_fetch_hint_without_request_id_is_accepted(monkeypatch):
    kvcr, results = _fetch_kvcr(monkeypatch)
    envelope = _router_hint("tcp://a:1", (1,))
    envelope.actions[0].payload.update(scope="independent", mode="move")
    kvcr.submit_hint(envelope)
    assert [(r.status, r.request_id) for r in results] == [(HintStatus.ACCEPTED, None)]
    assert not kvcr._core._remote_fw_dram._request_hints
    assert not kvcr._core._independent_hint_handler._received


_KEY = BlockKey(b"full-key-with-group")
_TAIL_KEY = BlockKey(b"tail-piece-key")


def _group(**changes):
    return msgspec.structs.replace(
        _FetchGroup(keys=(_KEY,), block_hashes=(123,), key_layout=("",), sizes=(16,)),
        **changes,
    )


@pytest.fixture
def pair(monkeypatch):
    """Two peers with reverse lookup; the source holds _KEY and both record
    hint results: admission at the source, the transfer at the target."""
    pair = _Pair()
    pair.source.kvcr._core._key_adapter = SimpleNamespace(
        decode_from_router={123: _KEY, 124: _TAIL_KEY}.__getitem__
    )
    pair.source.deposit(_KEY)
    pair.source_reports, pair.target_reports = (
        _record_hint_results(monkeypatch, peer.kvcr)
        for peer in (pair.source, pair.target)
    )
    return pair


def _submit(pair, action, carrier=None, hashes=(123,)):
    """Independent hints reach the source and name the target as destination."""
    hint = _independent("action", pair.target.control.endpoint, hashes, action)
    pair.source.kvcr.submit_hint(_envelope(hint), carrier)


def _idle(pair):
    pair.until(pair.idle)
    for peer in (pair.source, pair.target):
        core = peer.kvcr._core
        handler = core._independent_hint_handler
        assert not handler._received and not handler._pending_operations
        assert not core._framework_pin_keys and not core._local_dram_sources_by_op
        for record in core._block_record_map.values():
            if record.local_dram is not None:
                assert (
                    record.local_dram.claim_count
                    == record.local_dram.source_readers
                    == 0
                )
    assert not pair.target.results


@pytest.mark.parametrize("operation", ["fetch", "deliver"])
@pytest.mark.parametrize("succeeds", [False, True])
def test_evict_on_fetch_filters_hashes_and_accepts_late_completion(
    pair, operation, succeeds
):
    source, target = pair.source, pair.target
    source.deposit(_TAIL_KEY, 1)
    target.kvcr._core._key_adapter = SimpleNamespace(
        encode_to_router={_KEY: 123, _TAIL_KEY: 124}.__getitem__
    )
    remote = target.kvcr._core._remote_fw_dram
    remote._options = msgspec.structs.replace(remote._options, opportunistic_query=True)
    hint = _fetch("action", source.control.endpoint, [123], "kv.evict_on_fetch")
    target.kvcr.submit_hint(_envelope(hint), "req")
    blocks = {_KEY: [target.ref()], _TAIL_KEY: [target.ref(1)]}
    writes = len(source.agent.xfers)
    handle = getattr(target.kvcr, operation)(blocks, "req")
    pair.until(lambda: len(source.agent.xfers) > writes)
    target.kvcr._core._clock = lambda: 1.0
    source.agent.state = "DONE" if succeeds else "ERR"
    if not succeeds:
        pair.until(lambda: bool(source.agent.sent_notifs))
        source.agent.state = "DONE"  # Native cleanup settles after reporting failure.
    pair.until(lambda: handle in target.results and pair.idle())
    results = target.results.pop(handle)
    assert results[_KEY].success is succeeds and not results[_TAIL_KEY].success
    kept = (QueryStatus.HIT, CacheTier.LOCAL_G2)
    assert source.kvcr.query(blocks) == [_MISS if succeeds else kept, kept]
    claim = results[_KEY].release_handle
    if claim is not None:
        assert target.kvcr.release([claim]) == [(claim, True)]
    _idle(pair)


def test_waiting_fetch_upgrades_to_eviction_with_completion_grace(pair):
    """A capacity waiter adopts new advice and accepts completion past copy expiry."""
    source, target = pair.source, pair.target
    occupied = (b"occupied-first", b"occupied-second")
    for index, key in enumerate(occupied):
        target.deposit(key, index)
    core = target.kvcr._core
    core._key_adapter = SimpleNamespace(encode_to_router=lambda key: 123)
    pools = core._local_dram._pools
    pools.acquire_sources(occupied[:1])
    pools._capacity_eviction_key = occupied[0]
    pools.retire_sources(occupied[:1])
    target.kvcr.submit_hint(_router_hint(source.control.endpoint), "req")
    writes = len(source.agent.xfers)
    handle = target.kvcr.fetch([_KEY], "req")
    assert core._local_dram._capacity_waiters and len(source.agent.xfers) == writes
    hint = _fetch("upgrade", source.control.endpoint, [123], "kv.evict_on_fetch")
    target.kvcr.submit_hint(_envelope(hint), "req")
    pools.release_sources(occupied[:1])
    pair.until(lambda: len(source.agent.xfers) > writes)
    core._clock = lambda: 1.1
    source.agent.state = "DONE"
    pair.until(lambda: handle in target.results and pair.idle())
    result = target.results.pop(handle)[_KEY]
    assert result.success and source.kvcr.query([_KEY]) == [_MISS]
    claim = result.release_handle
    assert claim is not None and target.kvcr.release([claim]) == [(claim, True)]
    _idle(pair)


@pytest.mark.parametrize("block_hash", [123, 124])
def test_fetch_callback_cannot_upgrade_copy_to_eviction(pair, monkeypatch, block_hash):
    source, target = pair.source, pair.target
    for index in range(2):
        target.deposit(BlockKey(bytes([index])), index)
    core = target.kvcr._core
    core._key_adapter = SimpleNamespace(encode_to_router=lambda key: 123)
    target.kvcr.submit_hint(_router_hint(source.control.endpoint), "req")
    hint = _fetch("b", source.control.endpoint, [block_hash], "kv.evict_on_fetch")

    def upgrade(event):
        if event.removed:
            target.kvcr.submit_hint(_envelope(hint), "req")

    monkeypatch.setattr(core, "_inventory_sink_callback", upgrade)
    handle = target.kvcr.fetch([_KEY], "req")
    source.agent.state = "DONE"
    pair.until(lambda: handle in target.results and pair.idle())
    result = target.results.pop(handle)[_KEY]
    assert result.success is (block_hash == 123)
    messages = [_decode_control_message(raw) for _, raw in target.control.sent]
    assert not any(m.get("move", False) for m in messages)
    assert source.kvcr.query([_KEY]) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    if result.release_handle is not None:
        target.kvcr.release([result.release_handle])
    _idle(pair)


@pytest.mark.parametrize(
    "source_pieces,pieces", [(2, (0,)), (1, (0,)), (2, (0, 1)), (2, (1, 0))]
)
def test_evict_on_fetch_delivery_removes_only_complete_keys(source_pieces, pieces):
    pair = _Pair()
    source, target = pair.source, pair.target
    source.agent.state = "DONE"
    layout = ["", ":tail"]
    refs = [source.ref(), source.ref(1, ":tail")][:source_pieces]
    stored = source.kvcr.deposit({_KEY: refs}, expected_layout=layout)
    assert dict(_poll_until(source.kvcr, bool))[stored][_KEY].success
    target.kvcr._core._key_adapter = SimpleNamespace(encode_to_router=lambda key: 123)
    hint = _fetch("action", source.control.endpoint, [123], "kv.evict_on_fetch")
    target.kvcr.submit_hint(_envelope(hint), "req")
    destinations = [target.ref(i, layout[piece]) for i, piece in enumerate(pieces)]
    handle = target.kvcr.deliver({_KEY: destinations}, "req")
    pair.until(lambda: handle in target.results and pair.idle())
    assert target.results.pop(handle)[_KEY].success
    pair.until(lambda: (source.kvcr.query([_KEY]) == [_MISS]) is (len(pieces) == 2))
    _idle(pair)


@pytest.mark.parametrize("action_type", ["kv.fetch", "kv.evict_on_fetch"])
@pytest.mark.parametrize(
    "case", ["ordinary", "drop", "protected", "pending", "completed"]
)
def test_no_retain_consumes_next_deposit(pair, action_type, case):
    """Discard cancels pending and completed marks; otherwise hinted keys drop once."""
    source, target = pair.source, pair.target
    target.deposit(_TAIL_KEY, 1)
    target.kvcr._core._key_adapter = SimpleNamespace(
        encode_to_router={_KEY: 123, _TAIL_KEY: 124}.__getitem__
    )
    hint = _fetch("action", source.control.endpoint, [123], action_type)
    hint.payload["no_retain"] = case != "ordinary"
    target.kvcr.submit_hint(_envelope(hint), "req")
    blocks = {_KEY: [target.ref()], _TAIL_KEY: [target.ref(1)]}
    handle = target.kvcr.deliver(blocks, "req")
    if case == "pending":
        target.kvcr.discard_hint("req")
        target.kvcr.submit_hint(_envelope(hint), "req")  # Old delivery stays cancelled.
    else:
        hint.payload["no_retain"] = False
        target.kvcr.submit_hint(_envelope(hint), "req")  # Replacement is not discard.
    source.agent.state = target.agent.state = "DONE"
    pair.until(lambda: handle in target.results and pair.idle())
    assert all(result.success for result in target.results.pop(handle).values())
    if case == "completed":
        target.kvcr.discard_hint("req")
    transfers = len(target.agent.xfers)
    deposited = target.kvcr.deposit(blocks, no_evict=case == "protected")
    target.kvcr.discard_hint("req")  # Does not change a submitted deposit's outcome.
    results = dict(_poll_until(target.kvcr, bool))[deposited]
    assert results[_KEY].success is (case != "drop")
    assert results[_TAIL_KEY].success
    if case == "drop":
        assert results[_KEY].status is OpEntryStatus.DROPPED
        assert len(target.agent.xfers) == transfers
        assert target.kvcr.query([_KEY]) == [_MISS]
    if case == "protected":
        claims = [result.release_handle for result in results.values()]
        assert target.kvcr.release(claims) == [(claim, True) for claim in claims]
    target.deposit(_KEY)  # The marker was consumed; a later deposit is ordinary.
    _idle(pair)


@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize("declare", [False, True])
@pytest.mark.parametrize("whole", [False, True])
def test_no_retain_partial_delivery_and_deposit(pair, success, declare, whole):
    """Discard preserves another request's mark, including on incomplete keys."""
    target, core = pair.target, pair.target.kvcr._core
    core._key_adapter = SimpleNamespace(encode_to_router=lambda key: 123)
    layout = ["", ":tail"]
    target.agent.state = "DONE"
    first = target.kvcr.deposit({_KEY: [target.ref()]}, expected_layout=layout)
    assert dict(_poll_until(target.kvcr, bool))[first][_KEY].success
    hint = _fetch("action", pair.source.control.endpoint, [123])
    hint.payload["no_retain"] = True
    destination = target.ref(label="" if success else ":absent")
    for request_id in ("first", "second"):
        target.kvcr.submit_hint(_envelope(hint), request_id)
        delivered = target.kvcr.deliver({_KEY: [destination]}, request_id)
        assert dict(_poll_until(target.kvcr, bool))[delivered][_KEY].success is success
    target.kvcr.discard_hint("first")
    assert (_KEY in core._no_retain) is success
    blocks = {_KEY: [target.ref(1, ":tail")]}
    if whole:
        blocks[_KEY].insert(0, target.ref())
    second = target.kvcr.deposit(blocks, expected_layout=layout if declare else None)
    assert dict(_poll_until(target.kvcr, bool))[second][_KEY].status is (
        OpEntryStatus.DROPPED if success else OpEntryStatus.SUCCESS
    )
    assert not core._no_retain
    if success:
        assert core._block_record_map[_KEY].local_dram.pending_labels == {":tail"}
    later = target.kvcr.deposit(blocks, expected_layout=layout if declare else None)
    assert dict(_poll_until(target.kvcr, bool))[later][_KEY].success
    assert core._block_record_map[_KEY].local_dram.ready_indices(layout) is not None
    target.kvcr.discard_hint("second")
    _idle(pair)


@pytest.mark.parametrize(
    "reason",
    [
        "source_has_no_managed_dram",
        "key_adapter_unavailable",
        "self_destination",
        "destination-less hints are not currently supported",
        "no_retain is not supported by this hint",
    ],  # fmt: skip
)
@pytest.mark.parametrize("action_type", ["kv.prefetch", "kv.move"])
def test_independent_hints_are_rejected_at_the_source_before_resolution(
    monkeypatch, pair, reason, action_type
):
    kvcr, results = pair.source.kvcr, pair.source_reports
    destination = pair.target.control.endpoint
    if reason == "source_has_no_managed_dram":
        kvcr, results = _fetch_kvcr(monkeypatch)
    elif reason == "self_destination":
        destination = pair.source.control.endpoint
    elif reason == "key_adapter_unavailable":
        kvcr._core._key_adapter = None
    hint = _independent("a1", destination, [1], action_type)
    if reason.startswith("no_retain"):
        hint.payload["no_retain"] = False
    if reason.startswith("destination-less"):
        hint.payload["source_control_endpoint"] = hint.payload.pop(
            "destination_control_endpoint"
        )
    kvcr.submit_hint(_envelope(hint), "req")
    assert [
        (r.action_type, r.status, r.reason, r.request_id, r.message_id) for r in results
    ] == [(action_type, HintStatus.REJECTED, reason, "req", "m1")]
    remote = kvcr._core._remote_fw_dram
    assert not remote._request_hints and not remote._control.sent
    assert kvcr._core._outstanding_operations == 0 and not kvcr.poll_completed()


@pytest.mark.parametrize("incarnation", ["current", None, ""])
def test_request_returning_to_its_sender_fails_by_incarnation(pair, incarnation):
    """An alias of the node's own endpoint passes the string check; a move must
    still not invalidate the node's own copies."""
    hint = _independent("action", "tcp://alias-of-source:1", (123,), "kv.move")
    pair.source.kvcr.submit_hint(_envelope(hint))
    _wait_until(lambda: pair.source.control.sent)
    [(endpoint, raw)] = pair.source.control.sent
    payload = msgspec.msgpack.decode(raw)
    payload["sender_control_endpoint"] = endpoint  # an alias in either direction
    if incarnation != "current":
        payload["sender_incarnation"] = incarnation
    raw = msgspec.msgpack.encode(payload)
    pair.source.control.incoming.append(raw)  # delivered back to the sender
    _poll_until(pair.source.kvcr, lambda _: len(pair.source_reports) == 2)
    assert endpoint == "tcp://alias-of-source:1"
    assert [r.status for r in pair.source_reports] == [
        HintStatus.ACCEPTED,
        HintStatus.FAILED,
    ]
    assert pair.source_reports[-1].failures == {
        123: "self_source" if incarnation else "invalid_request"
    }
    assert pair.source.kvcr.query((_KEY,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]


@pytest.mark.parametrize(
    "changes,reason",
    [
        (None, "destination_has_no_managed_dram"),
        ({"sender_control_endpoint": None}, "invalid_request"),
        ({"groups": ()}, "invalid_request"),
        ({"groups": (_group(keys=(), block_hashes=()),)}, "invalid_request"),
        (
            {"groups": (_group(keys=(_KEY, _KEY), block_hashes=(123, 124)),)},
            "invalid_request",
        ),
        (
            {"groups": (_group(keys=(_KEY, _TAIL_KEY), block_hashes=(123, 123)),)},
            "invalid_request",
        ),
        ({"groups": (_group(block_hashes=(123, 124)),)}, "invalid_request"),
        ({"groups": (_group(), _group(block_hashes=(124,)))}, "invalid_request"),
        ({"groups": (_group(), _group(keys=(_TAIL_KEY,)))}, "invalid_request"),
        ({"groups": (_group(sizes=(32,)),)}, "incompatible_layout"),
        ({"groups": (_group(key_layout=(":*",)),)}, "incompatible_layout"),
    ],
)
def test_destination_rejects_unusable_requests(pair, changes, reason, monkeypatch):
    kvcr, reports = (
        _fetch_kvcr(monkeypatch)
        if changes is None
        else (pair.target.kvcr, pair.target_reports)
    )
    request = _KeysToFetch(
        action_id="action", message_id="m1", move=True, groups=(_group(),),
        sender_control_endpoint=pair.source.control.endpoint, sender_incarnation="peer",
    )  # fmt: skip
    request = msgspec.structs.replace(request, **(changes or {}))
    fetch = Mock()
    monkeypatch.setattr(kvcr._core, "fetch", fetch)
    kvcr._core._remote_fw_dram._control.incoming.append(msgspec.msgpack.encode(request))
    _poll_until(kvcr, lambda _: bool(reports))
    [result] = reports
    assert result.status is HintStatus.FAILED
    if reason != "incompatible_layout":
        assert result.reason == reason
    hashes = (h for group in request.groups for h in group.block_hashes)
    assert result.failures == (dict.fromkeys(hashes, reason) or None)
    fetch.assert_not_called()
    _idle(pair)


def test_all_failed_resolution_runs_under_the_owner_lock(pair, monkeypatch):
    source = pair.source.kvcr._core
    owner = threading.get_ident()

    def decode(block_hash):
        assert threading.get_ident() == owner and source._state_lock._is_owned()
        raise KeyError(block_hash)

    monkeypatch.setattr(source._key_adapter, "decode_from_router", decode)
    _submit(pair, "kv.prefetch", hashes=(999,))
    [result] = pair.source_reports
    assert result.status is HintStatus.FAILED and result.failures == {999: "not_found"}
    assert not pair.source.control.sent and not pair.target_reports
    _idle(pair)


@pytest.mark.parametrize("action", ["kv.prefetch", "kv.move"])
@pytest.mark.parametrize("carrier", [None, "unrelated-request"])
@pytest.mark.parametrize("cached", [False, True])
def test_independent_transfer_preserves_carrier_and_releases_claims(
    pair, action, carrier, cached
):
    """Mixed resolution preserves carrier advice and releases each hint's claims."""
    if cached:
        pair.target.deposit(_KEY)
    remote = pair.source.kvcr._core._remote_fw_dram
    if carrier:
        remote.submit_hint("tcp://unrelated:9", frozenset({456}), carrier)
    advice = dict(remote._request_hints)
    before = len(pair.source.agent.xfers)
    _submit(pair, action, carrier, (123, 999))
    assert remote._request_hints == advice
    if carrier:
        pair.source.kvcr.discard_hint(carrier)
    pair.source.agent.state = "DONE"
    pair.until(lambda: pair.target_reports)
    [accepted], [result] = pair.source_reports, pair.target_reports
    assert accepted.status is HintStatus.ACCEPTED and accepted.request_id == carrier
    assert accepted.failures == {999: "not_found"}
    assert result.status is HintStatus.COMPLETED and result.request_id is None
    assert (result.action_id, result.action_type, result.message_id) == (
        "action",
        action,
        "m1",
    )
    assert result.failures is None
    assert not remote._request_hints
    core = pair.target.kvcr._core
    assert core._block_record_map[_KEY].local_dram.claim_count == 0
    assert len(pair.source.agent.xfers) - before == (0 if cached else 1)
    expected = (
        (QueryStatus.MISS, None)
        if action == "kv.move"
        else (QueryStatus.HIT, CacheTier.LOCAL_G2)
    )
    pair.until(lambda: pair.source.kvcr.query((_KEY,)) == [expected])
    _idle(pair)
    assert len(pair.source_reports) == len(pair.target_reports) == 1


@pytest.mark.parametrize("at_submit", [False, True])
def test_nonfatal_error_reports_once_and_allows_later_work(
    pair, at_submit, monkeypatch
):
    owner, name = (
        (pair.source.kvcr._core._progress, "submit")
        if at_submit
        else (pair.target.kvcr._core, "fetch")
    )
    original = getattr(owner, name)
    monkeypatch.setattr(owner, name, Mock(side_effect=RuntimeError("nonfatal")))
    _submit(pair, "kv.prefetch")
    if at_submit:
        # The queue failure raised before ACCEPTED: admission reported once.
        assert _take(pair.source_reports) == [
            ("action", HintStatus.REJECTED, "handler_error")
        ]
    else:
        pair.until(lambda: pair.target_reports)
        assert pair.target_reports.pop().failures == {123: "operation_failed"}
    _idle(pair)
    monkeypatch.setattr(owner, name, original)
    _submit(pair, "kv.prefetch")
    pair.source.agent.state = "DONE"
    pair.until(lambda: pair.target_reports)
    assert pair.target_reports[-1].status is HintStatus.COMPLETED
    _idle(pair)


def test_fatal_fetch_submit_error_escapes_the_same_poll(pair, monkeypatch):
    """Fail after real destination reservation, after poll's initial fatal check."""
    core = pair.target.kvcr._core
    progress = core._progress
    submit = progress.submit
    failure = RuntimeError("fatal fetch submit")

    def fail_reserved(item):
        if isinstance(item, _TargetPullOp):
            assert core._block_record_map[_KEY].local_dram.remote_indices is not None
            progress._failure = failure
        return submit(item)

    monkeypatch.setattr(progress, "submit", fail_reserved)
    _submit(pair, "kv.prefetch")
    with pytest.raises(RuntimeError, match="fatal fetch submit") as error:
        pair.until(lambda: pair.target_reports)
    assert error.value is failure and not pair.target_reports
    with pytest.raises(RuntimeError, match="fatal fetch submit"):
        pair.target.kvcr.close()
    assert core.is_quiescent()


@pytest.mark.parametrize("vanished", [False, True])
def test_prefetch_keeps_named_g2_source_after_capacity_wait(
    pair, vanished, monkeypatch
):
    """A pending eviction delays admission; resumption preserves source and policy."""
    occupied = (b"occupied-first", b"occupied-second")
    for index, key in enumerate(occupied):
        pair.target.deposit(key, index)
    core = pair.target.kvcr._core
    pools = core._local_dram._pools
    pools.acquire_sources(occupied[:1])
    pools._capacity_eviction_key = occupied[0]
    pools.retire_sources(occupied[:1])
    pins = Mock(wraps=pair.source.kvcr._core._request_pin_callback)
    monkeypatch.setattr(pair.source.kvcr._core, "_request_pin_callback", pins)
    _submit(pair, "kv.prefetch")
    pair.until(lambda: bool(core._local_dram._capacity_waiters))
    assert not pair.target_reports
    assert all(
        _decode_control_message(raw)["type"] != "start_write"
        for _, raw in pair.target.control.sent
    )
    if vanished:
        _invalidate(pair.source.kvcr, (_KEY,))
    pools.release_sources(occupied[:1])
    pair.source.agent.state = "DONE"
    pair.until(lambda: pair.target_reports)
    assert pair.target_reports[-1].failures == (
        {123: "operation_failed"} if vanished else None
    )
    requests = [_decode_control_message(raw) for _, raw in pair.target.control.sent]
    writes = [request for request in requests if request["type"] == "start_write"]
    assert len(writes) == 1 and writes[0]["g2_only"] is True
    assert writes[0]["source_control_endpoint"] == pair.source.control.endpoint
    pins.assert_not_called()
    _idle(pair)


@pytest.mark.parametrize(
    "actions",
    [
        ("kv.prefetch", "kv.move"),
        ("kv.move", "kv.prefetch"),
        ("kv.prefetch", "kv.prefetch"),
    ],
)
def test_overlapping_actions_share_one_transfer(pair, actions):
    """Joining an existing fill still completes each hint and releases its claim."""
    core = pair.target.kvcr._core
    before = len(pair.source.agent.xfers)
    for action in actions:
        _submit(pair, action)
    pair.until(lambda: len(core._independent_hint_handler._pending_operations) == 2)
    pair.source.agent.state = "DONE"
    pair.until(lambda: len(pair.target_reports) == 2)
    assert all(r.status is HintStatus.COMPLETED for r in pair.target_reports)
    assert sorted(r.action_type for r in pair.target_reports) == sorted(actions)
    assert len(pair.source.agent.xfers) - before == 1
    _idle(pair)


@pytest.mark.parametrize("action", ["kv.prefetch", "kv.move"])
@pytest.mark.parametrize("failure", [None, "vanished", "layout", "fetch"])
def test_layout_groups_share_one_message_and_final_result(
    pair, action, failure, monkeypatch
):
    """A failed group cannot finish the hint while another group is transferring."""
    pair.source.agent.state = "DONE"
    handle = pair.source.kvcr.deposit(
        {_TAIL_KEY: [pair.source.ref(1, ":tail")]}, expected_layout=[":tail"]
    )
    assert dict(_poll_until(pair.source.kvcr, bool))[handle][_TAIL_KEY].success
    core = pair.target.kvcr._core
    handler, fetch = core._independent_hint_handler, core.fetch
    requested = handler.requested
    received = []

    def fetch_request(request):
        received.append(request)
        groups = sorted(request.groups, key=lambda group: _TAIL_KEY not in group.keys)
        request = msgspec.structs.replace(request, groups=tuple(groups))
        if failure == "vanished":
            _invalidate(pair.source.kvcr, (_TAIL_KEY,))
        elif failure == "layout":
            request = msgspec.structs.replace(
                request,
                groups=tuple(
                    msgspec.structs.replace(g, sizes=(32,))
                    if _TAIL_KEY in g.keys
                    else g
                    for g in request.groups
                ),
            )
        requested(request)

    def fetch_group(keys, **kwargs):
        if failure == "fetch" and _TAIL_KEY in keys:
            raise RuntimeError("one layout failed")
        return fetch(keys, **kwargs)

    monkeypatch.setattr(handler, "requested", fetch_request)
    monkeypatch.setattr(core, "fetch", fetch_group)
    before = len(pair.source.agent.xfers)
    pair.source.agent.state = "PROC"
    _submit(pair, action, hashes=(123, 124))
    pair.until(lambda: len(pair.source.agent.xfers) > before)
    assert not pair.target_reports  # Includes synchronous layout/fetch rejection.
    [request] = received
    assert sorted(g.key_layout for g in request.groups) == [("",), (":tail",)]
    messages = [_decode_control_message(raw) for _, raw in pair.source.control.sent]
    assert sum(m["type"] == "keys_to_fetch" for m in messages) == 1
    assert len(pair.source_reports) == 1
    pair.source.agent.state = "DONE"
    pair.until(lambda: pair.target_reports)
    [result] = pair.target_reports
    reason = "incompatible_layout" if failure == "layout" else "operation_failed"
    assert result.status is (HintStatus.FAILED if failure else HintStatus.COMPLETED)
    assert result.failures == ({124: reason} if failure else None)
    _idle(pair)
    assert len(pair.target_reports) == 1
