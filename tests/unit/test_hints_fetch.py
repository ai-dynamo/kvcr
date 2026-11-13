# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Request advice and independent prefetch/move hint execution."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import msgspec
import pytest
import zmq
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
    free_port,
)
from test_kvcr_remote_target import _Pair

from kvcr import ZmqPeerControlChannel
from kvcr.backends.remote_ops import (
    _FetchGroup,
    _KeysToFetch,
    _TargetMetadataAck,
    _TargetPullOp,
)
from kvcr.config import KVCRConfig, RemoteFWDramOptions
from kvcr.hints.dispatcher import _LOGGED_HANDLER_ERRORS
from kvcr.hints.parser import _LOGGED_HINT_ISSUES
from kvcr.hints.results import HintResult, HintStatus
from kvcr.types import BlockKey, CacheTier, InventoryEvent, OpEntryStatus, QueryStatus


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


class _TestAdapter:
    def encode_to_router(self, key):
        return {_KEY: 123, _TAIL_KEY: 124}.get(key, int.from_bytes(key[:8], "big"))


def _key(block_hash: int) -> BlockKey:
    return BlockKey(block_hash.to_bytes(8, "big"))


def _fetch_kvcr(monkeypatch, control=None):
    kvcr = _new_kvcr(
        FakeNixlAgent(),
        FakePrimaryPinning(),
        control if control is not None else FakeBytesControl(),
        KVCRConfig(nixl_agent_name="target", pool_layouts=[("", 16)]),
        key_adapter=_TestAdapter(),
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


def _send(peer, envelope, carrier):
    if carrier is None:
        peer.control.incoming.append(msgspec.msgpack.encode(envelope))
    else:
        peer.kvcr.submit_hint(envelope, carrier)


_FETCHABLE = (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
_MISS = (QueryStatus.MISS, None)


@pytest.mark.parametrize(
    "no_retain,expected", [(None, {1}), (False, set()), (True, {1, 2})]
)
def test_fetch_hint_without_managed_dram(monkeypatch, no_retain, expected):
    kvcr, results = _fetch_kvcr(monkeypatch)
    assert kvcr._core._local_dram is None
    kvcr._core._no_retain.add(1)
    hint = _router_hint("tcp://a:1", (1, 2))
    if no_retain is not None:
        hint.actions[0].payload["no_retain"] = no_retain
    with monkeypatch.context() as patch:
        lock = MagicMock()
        patch.setattr(kvcr._core, "_state_lock", lock)
        kvcr.submit_hint(hint, "req")
        assert lock.mock_calls == []

    assert kvcr._core._no_retain == expected
    assert kvcr.query((_key(1),), "req") == [_FETCHABLE]
    assert [(r.status, r.request_id) for r in results] == [(HintStatus.ACCEPTED, "req")]


@pytest.mark.parametrize("selected", [False, True])
def test_fetch_hint_takes_inline_path_without_lock(monkeypatch, selected):
    kvcr, results = _fetch_kvcr(monkeypatch)
    core, key = kvcr._core, _key(1)
    remote_query = core._remote_fw_dram.query
    envelope = _envelope(
        _fetch("a1", "tcp://a:1", [1]),
        _HintAction("f1", "kv.future_action", "1.0", {}),
    )

    def complete_during_query(key, request_id):
        available = remote_query(key, request_id)
        core._remote_delivery_keys[request_id].discard(key)
        return available

    with monkeypatch.context() as patch:
        lock = MagicMock()
        patch.setattr(core, "_state_lock", lock)
        patch.setattr(core._remote_fw_dram, "query", complete_during_query)
        kvcr.submit_hint(envelope, "req")
        if selected:
            core._remote_delivery_keys["req"] = {key}
        assert kvcr.query((key,), "req") == [_FETCHABLE]
        assert lock.mock_calls == []
    assert not core._remote_delivery_keys["req"]
    assert [(r.action_id, r.status, r.reason, r.request_id) for r in results] == [
        ("f1", HintStatus.REJECTED, "unsupported_action_type", "req"),
        ("a1", HintStatus.ACCEPTED, None, "req"),
    ]


def test_no_retain_without_key_adapter_records_and_warns(monkeypatch, caplog):
    """Rules are hashes and are recorded anyway; only enforcement needs the
    adapter, so both hints succeed and KVCR warns."""
    kvcr, results = _fetch_kvcr(monkeypatch)
    core = kvcr._core
    core._key_adapter = None
    hint = _fetch("a", "tcp://a:1", [1, 2])
    hint.payload["no_retain"] = True
    action = _HintAction("nr", "kv.no_retain", "1.0", {
        "block_hashes": [2, 3], "no_retain": True,
    })  # fmt: skip
    kvcr.submit_hint(_envelope(hint, action), "req")
    assert {1, 2} <= core._no_retain
    assert "req" in core._remote_fw_dram._request_hints
    _wait_until(lambda: len(results) == 2)
    assert core._no_retain == {1, 2, 3}
    assert "not enforced without a key adapter" in caplog.text
    action.payload.update(block_hashes=[1], no_retain=False)
    kvcr.submit_hint(_envelope(action))
    _wait_until(lambda: len(results) == 3)
    assert core._no_retain == {2, 3}
    assert [r.status for r in results] == [
        HintStatus.ACCEPTED, HintStatus.COMPLETED, HintStatus.COMPLETED,
    ]  # fmt: skip


@pytest.mark.parametrize("one_envelope", [True, False], ids=["envelope", "calls"])
@pytest.mark.parametrize(
    ("second_source", "expected"),
    [("tcp://a:1", [_MISS, _FETCHABLE]), ("tcp://b:1", [_MISS, _MISS])],
    ids=["same_source_replaces", "other_source_fails"],
)
def test_second_fetch_for_a_request(monkeypatch, one_envelope, second_source, expected):
    kvcr, results = _fetch_kvcr(monkeypatch)
    first, second = _fetch("a1", "tcp://a:1", [1]), _fetch("a2", second_source, [2])
    second.payload["no_retain"] = True  # Applied whether the store replaces or fails.
    for hints in [(first, second)] if one_envelope else [(first,), (second,)]:
        kvcr.submit_hint(_envelope(*hints), "req")
    # The store's rules apply within an envelope as across calls, and it reports
    # neither the replacement nor the failure.
    assert _take(results) == [
        ("a1", HintStatus.ACCEPTED, None),
        ("a2", HintStatus.ACCEPTED, None),
    ]
    assert kvcr.query((_key(1), _key(2)), "req") == expected
    assert kvcr._core._no_retain == {2}


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
    assert kvcr._core._hints_dispatcher._inbox.empty()
    assert not kvcr._core._independent_hint_handler._pending_operations


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
    pair = _Pair(key_adapter=_TestAdapter())
    pair.source.deposit(_KEY)
    pair.source_reports, pair.target_reports = (
        _record_hint_results(monkeypatch, peer.kvcr)
        for peer in (pair.source, pair.target)
    )
    return pair


def _submit(pair, action, carrier=None, hashes=(123,)):
    """Submit to the source and await admission without owner polling."""
    hint = _independent("action", pair.target.control.endpoint, hashes, action)
    before = len(pair.source_reports)
    pair.source.kvcr.submit_hint(_envelope(hint), carrier)
    _wait_until(lambda: len(pair.source_reports) > before)


def _idle(pair, *, poll=True):
    pair.until(pair.idle, poll=poll)
    for peer in (pair.source, pair.target):
        core = peer.kvcr._core
        assert not core._local_dram._pools._public_claims
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
@pytest.mark.parametrize("no_retain", [None, False, True])
def test_evict_on_fetch_filters_hashes_and_handles_transfer_failure(
    pair, operation, succeeds, no_retain
):
    source, target = pair.source, pair.target
    source.deposit(_TAIL_KEY, 1)
    marked = target.kvcr._core._no_retain
    marked.update(() if no_retain else (123,))
    remote = target.kvcr._core._remote_fw_dram
    remote._options = msgspec.structs.replace(remote._options, opportunistic_query=True)
    hint = _fetch("action", source.control.endpoint, [123], "kv.evict_on_fetch")
    if no_retain is not None:
        hint.payload["no_retain"] = no_retain
    target.kvcr.submit_hint(_envelope(hint), "req")
    expected = set() if no_retain is False else {123}
    assert marked == expected
    blocks = {_KEY: [target.ref()], _TAIL_KEY: [target.ref(1)]}
    writes = len(source.agent.xfers)
    handle = getattr(target.kvcr, operation)(blocks, "req")
    pair.until(lambda: len(source.agent.xfers) > writes)
    if operation == "fetch":
        hint.payload["no_retain"] = not no_retain
        target.kvcr.submit_hint(_envelope(hint), "req")
        expected = {123} if hint.payload["no_retain"] else set()
    else:
        target.kvcr.discard_hint("req")
    assert marked == expected
    target.kvcr._core._clock = lambda: 0.9
    source.agent.state = "DONE" if succeeds else "ERR"
    if not succeeds:
        pair.until(lambda: bool(source.agent.sent_notifs))
        source.agent.state = "DONE"  # Native cleanup settles after reporting failure.
    pair.until(lambda: handle in target.results and pair.idle())
    results = target.results.pop(handle)
    assert results[_KEY].success is succeeds and not results[_TAIL_KEY].success
    assert marked == expected
    if operation == "deliver":
        # Never attempted, whatever happened to the block that was.
        assert results[_TAIL_KEY].reason == "hint_unavailable"
    kept = (QueryStatus.HIT, CacheTier.LOCAL_G2)
    assert source.kvcr.query(blocks) == [_MISS if succeeds else kept, kept]
    claim = results[_KEY].release_handle
    if claim is not None:
        assert target.kvcr.release([claim]) == [(claim, True)]
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


@pytest.mark.parametrize("move", [False, True], ids=["evict_on_fetch", "move"])
@pytest.mark.parametrize("ack", ["lost", "delayed_owner", "partial"])
def test_source_eviction_waits_for_destination_acceptance(pair, monkeypatch, move, ack):
    source, target = pair.source, pair.target
    source.deposit(_TAIL_KEY, 1)
    keys = (_KEY, _TAIL_KEY)
    core = target.kvcr._core
    endpoint = source.control.endpoint
    core._key_adapter = SimpleNamespace(
        encode_to_router={_KEY: 123, _TAIL_KEY: 124}.__getitem__
    )
    hint = _fetch("action", endpoint, (123, 124), "kv.evict_on_fetch")
    target.kvcr.submit_hint(_envelope(hint), "req")
    held, send = [], target.control.send

    def hold_ack(endpoint, raw):
        message = _decode_control_message(raw)
        if message["type"] == "write_accepted":
            held.append(message)
            return True
        return send(endpoint, raw)

    finish = core._remote_fw_dram._finish_target_pull

    def finish_at_owner(op):
        assert op.success  # Native success is already queued for the owner.
        if ack == "delayed_owner":
            core._clock = lambda: op.deadline + 1
            core._local_dram._expire_pending_ops(core._clock())
        elif ack == "partial":
            core._local_dram.discard_fill((_TAIL_KEY,))
        finish(op)

    monkeypatch.setattr(target.control, "send", hold_ack)
    monkeypatch.setattr(core._remote_fw_dram, "_finish_target_pull", finish_at_owner)
    handle = core.fetch(
        keys, "req", source_endpoint=endpoint if move else None, move=move
    )
    source.agent.state = "DONE"
    pair.until(lambda: handle in target.results and held)
    results = target.results.pop(handle)
    accepted = keys[:1] if ack == "partial" else keys
    assert {key for key, result in results.items() if result.success} == set(accepted)
    [message] = held
    assert set(message["keys"]) == set(accepted)
    for key in keys:
        residency = source.kvcr._core._block_record_map[key].local_dram
        assert residency is not None and residency.source_readers == 0
    if ack == "lost":
        source.kvcr._core._clock = lambda: 100
    else:
        # Stale destination/source incarnations must not consume the one ACK slot.
        source.control.incoming.extend(
            msgspec.msgpack.encode(dict(message, **{field: "stale", "keys": ()}))
            for field in ("sender_incarnation", "source_incarnation")
        )
        source.control.incoming.append(msgspec.msgpack.encode(message))
    claims = [results[key].release_handle for key in accepted]
    assert target.kvcr.release(claims) == [(claim, True) for claim in claims]
    _idle(pair)
    expected = [
        _MISS
        if key in accepted and ack != "lost"
        else (QueryStatus.HIT, CacheTier.LOCAL_G2)
        for key in keys
    ]
    assert source.kvcr.query(keys) == expected
    # Duplicate/late ACKs cannot remove keys rejected by the owner or reopened.
    source.control.incoming.append(msgspec.msgpack.encode(dict(message, keys=keys)))
    pair.until(lambda: not source.control.incoming)
    assert source.kvcr.query(keys) == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_no_retain_applies_on_hint_submission(pair, enabled):
    target, core = pair.target, pair.target.kvcr._core
    core._key_adapter = SimpleNamespace(encode_to_router=Mock(return_value=123))
    hint = _fetch("action", pair.source.control.endpoint, [123, 999])
    core._no_retain.update((123, 999))
    for request, setting in (("a", enabled), ("b", not enabled)):
        hint.payload["no_retain"] = setting
        target.kvcr.submit_hint(_envelope(hint), request)
        assert core._no_retain == ({123, 999} if setting else set())
    core._remote_fw_dram._fail_request_hint("a")
    target.kvcr.discard_hint("b")
    assert core._no_retain == (set() if enabled else {123, 999})
    core._key_adapter.encode_to_router.assert_not_called()
    _idle(pair)


@pytest.mark.parametrize(
    "existing,protected", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize(
    "carrier", [None, "unrelated-request"], ids=["direct", "piggyback"]
)
def test_independent_no_retain_controls_full_and_partial_deposits(
    pair, existing, protected, carrier, monkeypatch
):
    target, core = pair.target, pair.target.kvcr._core
    kvcr, layout = target.kvcr, ["", ":tail"]
    handlers = {
        name: core._hints_dispatcher._handlers[name].submit
        for name in ("kv.no_retain", "kv.invalidate")
    }

    def on_progress(hint, *args):
        assert threading.get_ident() == core._progress._thread.ident
        return handlers[hint.ACTION_TYPE](hint, *args)

    def submit(hint, status=HintStatus.COMPLETED):
        before = len(pair.target_reports)
        _send(target, _envelope(hint), carrier)
        _wait_until(lambda: len(pair.target_reports) == before + 1)
        assert pair.target_reports[-1].status is status
        assert pair.target_reports[-1].request_id == carrier

    for name in handlers:
        handler = core._hints_dispatcher._handlers[name]
        monkeypatch.setattr(handler, "submit", on_progress)
    target.agent.state = "DONE"
    if existing:
        handle = kvcr.deposit({_KEY: [target.ref()]}, expected_layout=layout)
        assert dict(_poll_until(kvcr, bool))[handle][_KEY].success
    core._key_adapter = SimpleNamespace(
        encode_to_router=lambda key: 123, decode_from_router={123: _KEY}.__getitem__
    )
    action = _HintAction("nr", "kv.no_retain", "1.0", {
        "block_hashes": [123, 999, 123], "no_retain": True,
    })  # fmt: skip
    submit(action)
    claims = []
    for refs in ([target.ref(1, ":tail")], [target.ref(), target.ref(1, ":tail")]):
        handle = kvcr.deposit(
            {_KEY: refs}, expected_layout=layout, no_evict=protected and not claims
        )
        entry = dict(_poll_until(kvcr, bool))[handle][_KEY]
        expected = OpEntryStatus.SUCCESS if protected else OpEntryStatus.DROPPED
        assert entry.status is expected and core._no_retain == {123, 999}
        if entry.release_handle is not None:
            claims.append(entry.release_handle)
    invalidate = _HintAction("i", "kv.invalidate", "1.0", {"block_hashes": [123, 999]})
    submit(invalidate, HintStatus.FAILED)
    report = pair.target_reports.pop()
    assert report.status is HintStatus.FAILED and report.failures == {
        123: "protected" if protected else "not_found",
        999: "not_found",
    }
    kvcr.release(claims)
    kvcr.discard_hint("unrelated-request")
    assert core._no_retain == {123, 999} and bool(core._block_record_map) is existing
    action.payload.update(block_hashes=[123], no_retain=False)
    submit(action)
    handle = kvcr.deposit({_KEY: [target.ref(), target.ref(1, ":tail")]})
    assert dict(_poll_until(kvcr, bool))[handle][_KEY].success
    assert core._no_retain == {999}
    submit(invalidate)  # Complete and unprotected: removal is queued.
    assert pair.target_reports[-1].failures == {999: "not_found"}
    _wait_until(lambda: not core._block_record_map)
    assert all(r.status is HintStatus.COMPLETED for r in pair.target_reports)
    _idle(pair, poll=False)


def _control_progress(peer, phase):
    endpoint = f"tcp://{phase}:9"
    peer.control.incoming.append(
        msgspec.msgpack.encode(_TargetMetadataAck(sender_control_endpoint=endpoint))
    )
    _wait_until(
        lambda: endpoint in peer.kvcr._core._remote_fw_dram._metadata_acked_sources
    )


@pytest.mark.parametrize("shutdown", [False, True])
def test_control_hint_backpressure_preserves_peer_control_and_advice(
    pair, shutdown, monkeypatch
):
    core = pair.source.kvcr._core
    destination = pair.target.kvcr._core
    pending = destination._independent_hint_handler._background_items
    dispatcher = core._hints_dispatcher
    inbox = dispatcher._inbox
    remote = core._remote_fw_dram
    resolve_hashes = core.resolve_hashes

    def resolve(hashes):
        assert threading.get_ident() == core._progress._thread.ident
        assert not core._state_lock._is_owned()
        return resolve_hashes(hashes)

    monkeypatch.setattr(core, "resolve_hashes", resolve)
    remote.submit_hint("tcp://other:9", frozenset({456}), "req")
    advice = dict(remote._request_hints)
    envelope = _envelope(
        _fetch("fetch", "tcp://replacement:9", [123]),
        _fetch("evict", "tcp://replacement:9", [123], "kv.evict_on_fetch"),
        _independent("good", pair.target.control.endpoint, [123], "kv.move"),
        _HintAction("set", "kv.no_retain", "1.0", {
            "block_hashes": [999], "no_retain": True,
        }),
    )  # fmt: skip
    clear = _envelope(
        _HintAction("clear", "kv.no_retain", "1.0", {
            "block_hashes": [999], "no_retain": False,
        }),
    )  # fmt: skip
    prefix = [
        ("fetch", HintStatus.REJECTED, "request_scoped_hint"),
        ("evict", HintStatus.REJECTED, "request_scoped_hint"),
        ("good", HintStatus.ACCEPTED, None),
    ]
    overflow = _envelope(
        _independent("overflow", pair.target.control.endpoint, [999], "kv.prefetch")
    )
    with core._state_lock, destination._state_lock:
        with monkeypatch.context() as patch:
            patch.setattr(dispatcher, "poll", lambda: None)
            pair.source.control.incoming.extend(
                [msgspec.msgpack.encode(envelope), msgspec.msgpack.encode(clear)]
                + [msgspec.msgpack.encode(overflow)] * (inbox.maxsize - 1)
            )
            _control_progress(pair.source, "full")
            assert inbox.full() and len(pair.source_reports) == 2
            # Only the preexisting advice's eager metadata handshake is allowed.
            assert all(
                endpoint == "tcp://other:9"
                and _decode_control_message(raw)["type"] == "target_metadata"
                for endpoint, raw in pair.source.control.sent
            )
        _wait_until(lambda: dispatcher._pending is not None)
        pair.until(
            lambda: any(isinstance(item, _KeysToFetch) for item in pending), poll=False
        )
        for peer in (pair.source, pair.target):
            _control_progress(peer, "admission")
        assert not pair.target_reports and not pair.target.events
        assert not core._no_retain
        assert [
            (r.action_id, r.status, r.reason) for r in pair.source_reports
        ] == prefix
        if shutdown:
            apply = Mock(
                side_effect=AssertionError("queued hint executed during close")
            )
            monkeypatch.setattr(dispatcher._handlers["kv.no_retain"], "submit", apply)
            monkeypatch.setattr(destination, "fetch", apply)
            for peer in (pair.source, pair.target):
                peer.kvcr.close()
                assert peer.kvcr._core.is_quiescent()
            apply.assert_not_called()
            assert inbox.empty() and dispatcher._pending is None and not pending
            assert not pair.target_reports
            assert _take(pair.source_reports) == prefix
            return
    _wait_until(lambda: len(pair.source_reports) == inbox.maxsize + 3)
    assert all(r.failures == {999: "not_found"} for r in pair.source_reports[5:])
    assert not core._no_retain and dispatcher._pending is None
    assert remote._request_hints == advice
    reports = _take(pair.source_reports)
    assert reports[:5] == prefix + [
        ("set", HintStatus.COMPLETED, None),
        ("clear", HintStatus.COMPLETED, None),
    ]
    assert set(reports[5:]) == {("overflow", HintStatus.FAILED, None)}
    pair.until(lambda: pair.target.local.raw[:16] == b"a" * 16, poll=False)
    with destination._state_lock:
        pair.source.agent.state = "DONE"
        pair.until(
            lambda: any(isinstance(item, _TargetPullOp) for item in pending), poll=False
        )
        _control_progress(pair.target, "finalization")
        assert not pair.target_reports and not pair.target.events
    pair.until(lambda: pair.target_reports, poll=False)
    pair.until(lambda: pair.source.kvcr.query((_KEY,)) == [_MISS], poll=False)
    assert pair.target_reports[-1].status is HintStatus.COMPLETED
    assert pair.target.events == [InventoryEvent((_KEY,), CacheTier.LOCAL_G2, False)]
    assert sum(
        _decode_control_message(raw)["type"] == "keys_to_fetch"
        for _, raw in pair.source.control.sent
    ) == 1
    _idle(pair, poll=False)


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
    request_id = None if reason == "source_has_no_managed_dram" else "req"
    if reason == "source_has_no_managed_dram":
        control = ZmqPeerControlChannel("127.0.0.1", free_port(), "127.0.0.1")
        kvcr, results = _fetch_kvcr(monkeypatch, control)
    elif reason == "self_destination":
        destination = pair.source.control.endpoint
    elif reason == "key_adapter_unavailable":
        kvcr._core._key_adapter = None
    hint = _independent("a1", destination, [(1 << 64) - 1], action_type)
    if reason.startswith("no_retain"):
        hint.payload["no_retain"] = False
    if reason.startswith("destination-less"):
        hint.payload["source_control_endpoint"] = hint.payload.pop(
            "destination_control_endpoint"
        )
    remote = kvcr._core._remote_fw_dram
    send = Mock(wraps=remote._control.send)
    monkeypatch.setattr(remote._control, "send", send)
    if request_id is None:
        with zmq.Context.instance().socket(zmq.PUSH) as sender:
            sender.linger = 0
            sender.connect(control.endpoint)
            sender.send(msgspec.msgpack.encode(_envelope(hint)), zmq.DONTWAIT)
            _wait_until(lambda: results)
    else:
        kvcr.submit_hint(_envelope(hint), request_id)
        _wait_until(lambda: results)
    assert [
        (r.action_type, r.status, r.reason, r.request_id, r.message_id) for r in results
    ] == [(action_type, HintStatus.REJECTED, reason, request_id, "m1")]
    assert not remote._request_hints
    send.assert_not_called()
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


@pytest.mark.parametrize("action", ["kv.prefetch", "kv.move"])
@pytest.mark.parametrize(
    "carrier", [None, "unrelated-request"], ids=["direct", "piggyback"]
)
@pytest.mark.parametrize("cached", [False, True])
def test_independent_transfer_preserves_carrier_and_releases_claims(
    pair, action, carrier, cached, monkeypatch
):
    """Idle execution preserves carrier advice, cache hits, inventory, and claims."""
    target = pair.target.kvcr
    if cached:
        pair.target.deposit(_KEY)
        remote_query = Mock(wraps=target._core._remote_fw_dram.query)
        monkeypatch.setattr(target._core._remote_fw_dram, "query", remote_query)
        target.submit_hint(_router_hint(pair.source.control.endpoint, (123,)), "local")
        assert target.query((_KEY,), "local") == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
        remote_query.assert_not_called()
        assert not target._core._remote_delivery_keys["local"]
        target.discard_hint("local")
        pair.target_reports.clear()
    pair.source.events.clear()
    pair.target.events.clear()
    remote = pair.source.kvcr._core._remote_fw_dram
    if carrier:
        remote.submit_hint("tcp://unrelated:9", frozenset({456}), carrier)
    advice = dict(remote._request_hints)
    before = len(pair.source.agent.xfers)
    envelope = _envelope(
        _independent("action", pair.target.control.endpoint, [123, 999], action)
    )
    if cached:
        framework_handle = target.fetch((_KEY,))
    # Both ingress paths must resolve and forward while the owner holds its lock.
    with pair.source.kvcr._core._state_lock:
        _send(pair.source, envelope, carrier)
        _wait_until(lambda: pair.source_reports)
        _wait_until(lambda: any(
            _decode_control_message(raw)["type"] == "keys_to_fetch"
            for _, raw in pair.source.control.sent
        ))
    assert remote._request_hints == advice
    if carrier:
        pair.source.kvcr.discard_hint(carrier)
    if not cached:
        # The fake writes bytes at submission; completion still needs DONE.
        pair.until(lambda: pair.target.local.raw[:16] == b"a" * 16, poll=False)
        assert target.query((_KEY,)) == [(QueryStatus.FETCHING, CacheTier.LOCAL_G2)]
        assert not pair.target_reports and not pair.target.events
        framework_handle = target.fetch((_KEY,))
    pools = target._core._local_dram._pools
    reclaim = Mock(wraps=pools._reclaimable_slots)
    monkeypatch.setattr(target._core, "_capacity_needed_callback", Mock())
    monkeypatch.setattr(pools, "_reclaimable_slots", reclaim)
    pair.source.agent.state = "DONE"
    pair.until(lambda: pair.target_reports, poll=False)
    expected = _MISS if action == "kv.move" else (QueryStatus.HIT, CacheTier.LOCAL_G2)
    pair.until(lambda: pair.source.kvcr.query((_KEY,)) == [expected], poll=False)
    pair.until(pair.idle, poll=False)
    reclaim.assert_not_called()
    assert target._core._block_record_map[_KEY].local_dram.claim_count == 1
    [(handle, entries)] = target.poll_completed()
    reclaim.assert_called()
    assert handle == framework_handle and entries[_KEY].success
    claim = entries[_KEY].release_handle
    assert target.poll_completed() == []
    assert target.release([claim]) == [(claim, True)]
    _idle(pair, poll=False)
    [accepted], [result] = pair.source_reports, pair.target_reports
    assert accepted.status is HintStatus.ACCEPTED and accepted.request_id == carrier
    assert accepted.failures == {999: "not_found"}
    assert result == HintResult(
        "action", action, HintStatus.COMPLETED, message_id="m1"
    )
    assert not remote._request_hints
    assert pair.target.kvcr.query((_KEY,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert len(pair.source.agent.xfers) - before == (0 if cached else 1)
    stored = InventoryEvent((_KEY,), CacheTier.LOCAL_G2, False)
    removed = InventoryEvent((_KEY,), CacheTier.LOCAL_G2, True)
    assert pair.target.events == ([] if cached else [stored])
    assert pair.source.events == ([removed] if action == "kv.move" else [])
    known = ({(("",), (16,)): {_KEY: 123}}, {})
    assert pair.source.kvcr._core.resolve_hashes((123,)) == (
        ({}, {123: "not_found"}) if action == "kv.move" else known
    )
    assert target._core.resolve_hashes((123,)) == known


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


def test_fatal_background_fetch_failure_propagates_to_owner(pair, monkeypatch):
    """A fatal failure after destination reservation is visible to the owner."""
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


@pytest.mark.parametrize("outcome", ["ready", "vanished", "timeout", "newcomer"])
def test_prefetch_keeps_named_g2_source_after_capacity_wait(pair, outcome, monkeypatch):
    """A pending eviction delays admission; resumption preserves source and policy."""
    occupied = (b"occupied-first", b"occupied-second")
    claim = pair.target.deposit(occupied[0], protected=True)
    other_claim = pair.target.deposit(occupied[1], 1, protected=outcome == "newcomer")
    core = pair.target.kvcr._core
    pools = core._local_dram._pools
    pools._capacity_eviction_key = occupied[0]
    pools.retire_sources(occupied[:1])
    pins = Mock(wraps=pair.source.kvcr._core._request_pin_callback)
    monkeypatch.setattr(pair.source.kvcr._core, "_request_pin_callback", pins)
    _submit(pair, "kv.prefetch")
    pair.until(lambda: bool(core._local_dram._capacity_waiters), poll=False)
    with monkeypatch.context() as patch:
        maintenance = Mock(wraps=core._local_dram.poll_main)
        patch.setattr(core._local_dram, "poll_main", maintenance)
        _control_progress(pair.target, "blocked")
        _control_progress(pair.target, "still-blocked")
        maintenance.assert_not_called()
    assert not pair.target_reports
    assert all(
        _decode_control_message(raw)["type"] != "start_write"
        for _, raw in pair.target.control.sent
    )
    if outcome == "timeout":
        deadline = next(iter(core._local_dram._pending_residency_ops.values())).deadline
        core._clock = lambda: deadline + 1
        pair.until(lambda: pair.target_reports, poll=False)
        assert pair.target_reports[-1].status is HintStatus.FAILED
        assert pair.target.kvcr.release([claim]) == [(claim, True)]
        _idle(pair, poll=False)
        return
    if outcome == "vanished":
        _invalidate(pair.source.kvcr, (_KEY,))
    start_fill, fill_threads = core._start_local_fill, []

    def resumed_fill(*args, **kwargs):
        fill_threads.append(threading.get_ident())
        assert threading.get_ident() == core._progress._thread.ident
        assert core._state_lock._is_owned()
        return start_fill(*args, **kwargs)

    monkeypatch.setattr(core, "_start_local_fill", resumed_fill)
    # Releasing on the caller must hand the independent waiter back to progress.
    with core._state_lock:
        assert pair.target.kvcr.release([claim]) == [(claim, True)]
        if outcome == "newcomer":
            newcomer = BlockKey(b"newcomer")
            deposit = pair.target.kvcr.deposit(
                {newcomer: [pair.target.ref(1)]}, no_evict=True
            )
    pair.source.agent.state = "DONE"
    pair.until(lambda: pair.target_reports, poll=False)
    assert fill_threads == [core._progress._thread.ident]
    assert pair.target_reports[-1].failures == (
        {123: "operation_failed"} if outcome == "vanished" else None
    )
    requests = [_decode_control_message(raw) for _, raw in pair.target.control.sent]
    writes = [request for request in requests if request["type"] == "start_write"]
    assert len(writes) == 1 and writes[0]["g2_only"] is True
    assert writes[0]["source_control_endpoint"] == pair.source.control.endpoint
    pins.assert_not_called()
    if outcome == "newcomer":
        result = dict(_poll_until(pair.target.kvcr, bool))[deposit][newcomer]
        assert not result.success and newcomer not in core._block_record_map
        assert pair.target.kvcr.release([other_claim]) == [(other_claim, True)]
    _idle(pair, poll=False)


@pytest.mark.parametrize("producer", ["remote", "remote_finished", "deposit_timeout"])
@pytest.mark.parametrize("carrier", [None, "request"], ids=["direct", "piggyback"])
@pytest.mark.parametrize("action", ["kv.prefetch", "kv.move"])
def test_hint_joining_framework_work_finishes_without_owner_poll(
    pair, producer, carrier, action
):
    target, source = pair.target, pair.source
    core, now = target.kvcr._core, 0.0
    core._clock = lambda: now
    timed_out = producer == "deposit_timeout"
    before = len(source.agent.xfers)
    if timed_out:
        handle = target.kvcr.deposit({_KEY: [target.ref()]})
        _wait_until(lambda: target.agent.transfers)
    else:
        target.kvcr.submit_hint(_router_hint(source.control.endpoint, (123,)), "req")
        advice = core._remote_fw_dram._request_hints["req"]
        pair.target_reports.clear()
        handle = target.kvcr.fetch((_KEY,), "req")
        pair.until(lambda: len(source.agent.xfers) > before, poll=False)
        if producer == "remote_finished":
            source.agent.state = "DONE"
            pair.until(lambda: not core._progress._in_flight_ops, poll=False)
    hint = _independent("joined", target.control.endpoint, [123], action)
    _send(source, _envelope(hint), carrier)
    if producer != "remote_finished":
        pair.until(
            lambda: core._independent_hint_handler._pending_operations, poll=False
        )
    if timed_out:
        now = 1.5
    else:
        source.agent.state = "DONE"
    pair.until(lambda: pair.target_reports, poll=False)
    assert pair.target_reports[-1].status is (
        HintStatus.FAILED if timed_out else HintStatus.COMPLETED
    )
    if timed_out:
        assert target.agent.state == "PROC" and not target.agent.released_xfers
        assert core._block_record_map[_KEY].local_dram.writers == [1]
        assert not core._local_dram._pools._public_claims and not target.events
        target.agent.state = "DONE"
    else:
        assert core._remote_fw_dram._request_hints["req"] is advice
        assert target.events == [InventoryEvent((_KEY,), CacheTier.LOCAL_G2, False)]
    pair.until(lambda: handle in dict(core._completion_queue), poll=False)
    entries = dict(target.kvcr.poll_completed())[handle]
    assert entries[_KEY].success is not timed_out
    if not timed_out:
        assert core._remote_fw_dram._request_hints["req"] is not advice
        claim = entries[_KEY].release_handle
        assert target.kvcr.release([claim]) == [(claim, True)]
    _idle(pair, poll=False)


@pytest.mark.parametrize(
    "actions,succeeds",
    [
        (("kv.prefetch", "kv.move"), True),
        (("kv.move", "kv.prefetch"), True),
        (("kv.prefetch", "kv.prefetch"), True),
        (("kv.move",), False),
    ],
)
def test_joined_transfer_and_failed_move_cleanup(pair, actions, succeeds):
    """Joined copies/moves share completion; a failed move reopens its source."""
    core = pair.target.kvcr._core
    source = pair.source.kvcr._core
    pending = core._independent_hint_handler._pending_operations
    before = len(pair.source.agent.xfers)
    for action in actions:
        _submit(pair, action)
    pair.until(lambda: len(pending) == len(actions), poll=False)
    pair.until(lambda: len(pair.source.agent.xfers) > before, poll=False)
    if not succeeds:
        assert source.resolve_hashes((123,)) == ({}, {123: "source_not_transferable"})
    pair.source.agent.state = "DONE" if succeeds else "ERR"
    pair.until(lambda: len(pair.target_reports) == len(actions), poll=False)
    status = HintStatus.COMPLETED if succeeds else HintStatus.FAILED
    assert all(r.status is status for r in pair.target_reports)
    assert sorted(r.action_type for r in pair.target_reports) == sorted(actions)
    assert len(pair.source.agent.xfers) - before == 1
    pair.source.agent.state = "DONE"
    _idle(pair, poll=False)
    if not succeeds:
        assert source.resolve_hashes((123,)) == ({(("",), (16,)): {_KEY: 123}}, {})
        assert core.resolve_hashes((123,)) == ({}, {123: "not_found"})


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
    pins = Mock(side_effect=AssertionError("independent hint used framework memory"))
    monkeypatch.setattr(pair.source.kvcr._core, "_request_pin_callback", pins)
    free = len(core._local_dram._pools._free_slots[""])
    pair.target.events.clear()
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
    pair.until(lambda: len(pair.source.agent.xfers) > before, poll=False)
    assert not pair.target_reports  # Includes synchronous layout/fetch rejection.
    pair.source.agent.state = "DONE"
    pair.until(lambda: pair.target_reports, poll=False)
    _idle(pair, poll=False)
    [request] = received
    assert sorted(g.key_layout for g in request.groups) == [("",), (":tail",)]
    messages = [_decode_control_message(raw) for _, raw in pair.source.control.sent]
    assert sum(m["type"] == "keys_to_fetch" for m in messages) == 1
    assert len(pair.source_reports) == 1
    [result] = pair.target_reports
    reason = "incompatible_layout" if failure == "layout" else "operation_failed"
    assert result.status is (HintStatus.FAILED if failure else HintStatus.COMPLETED)
    assert result.failures == ({124: reason} if failure else None)
    stored = (_KEY,) if failure else (_KEY, _TAIL_KEY)
    assert sorted(pair.target.events, key=lambda event: event.keys) == [
        InventoryEvent((key,), CacheTier.LOCAL_G2, False) for key in stored
    ]
    assert len(core._local_dram._pools._free_slots[""]) == free - len(stored)
    if failure:
        assert pair.target.kvcr.query((_TAIL_KEY,)) == [_MISS]
        assert core.resolve_hashes((124,)) == ({}, {124: "not_found"})
    pins.assert_not_called()
