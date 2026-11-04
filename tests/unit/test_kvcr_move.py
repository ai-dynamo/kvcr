# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Move transfers, early source withdrawal, and cached-key invalidation."""

import ctypes
import os
import time
from collections import Counter
from dataclasses import replace

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakePrimaryPinning,
    _buffer_region,
    _ConstantHashAdapter,
    _decode_control_message,
    _has_outstanding_operations,
    _HintAction,
    _HintEnvelope,
    _invalidate,
    _mem_descriptor,
    _new_kvcr,
    _poll_until,
    _wait_until,
)
from test_g3 import _FakeG3Agent, _MoveLocalToG3Policy
from test_kvcr_remote_target import _CopyingWriteAgent

from kvcr.backends.remote_ops import _Invalidate
from kvcr.config import G3Options, KVCRConfig, LocalDramOptions, RemoteFWDramOptions
from kvcr.policy import FIFOPolicy
from kvcr.types import BlockKey, CacheTier, PlacementAction, QueryStatus


class _KeepPolicy(FIFOPolicy):
    def decide_eviction(self, meta, source):
        return PlacementAction.KEEP, None


class _IndividualWriteAgent(_CopyingWriteAgent):
    def __init__(self):
        super().__init__(metadata=b"source-md")
        self.states = {}

    def check_xfer_state(self, handle):
        return self.states.get(handle, self.state)


class _IndividualG3WriteAgent(_FakeG3Agent):
    def __init__(self):
        super().__init__()
        self.states = {}

    def transfer(self, handle):
        if self._xfer_backends[handle] == ("MOCK",):
            return super().transfer(handle)
        return _CopyingWriteAgent.transfer(self, handle)

    def check_xfer_state(self, handle):
        return self.states.get(handle, self.state)


class _Pair:
    """Real buffers with manually forwarded control messages and completions."""

    def __init__(
        self,
        *,
        slots=1,
        labels=("",),
        source_agent=None,
        target_labels=None,
        opportunistic_query=False,
        block_size=16,
        source_g3=None,
        source_policy=None,
    ):
        self.labels = labels
        target_labels = labels if target_labels is None else target_labels
        self.results = [{}, {}]
        self.control_offsets = [0, 0]
        self.notif_offsets = [0, 0]
        self.delivered_native = set()
        self.source_now = 0.0
        self.source_events = []
        self.target_events = []
        self.source_agent = source_agent or _CopyingWriteAgent(metadata=b"source-md")
        self.target_agent = _CopyingWriteAgent(metadata=b"target-md")
        self.source_control = FakeBytesControl("tcp://source:1")
        self.target_control = FakeBytesControl("tcp://target:1")
        self.source_pinning = FakePrimaryPinning()
        self.source_primary = {}
        self.target_primary = {}
        self.source_local = {}
        self.target_local = {}
        for index, label in enumerate(labels):
            self.source_primary[label] = ctypes.create_string_buffer(
                (slots + 1) * block_size
            )
            self.source_primary[label].raw = bytes([97 + index]) * (
                (slots + 1) * block_size
            )
            self.source_local[label] = ctypes.create_string_buffer(slots * block_size)
            if label in target_labels:
                self.target_primary[label] = ctypes.create_string_buffer(
                    (slots + 1) * block_size
                )
                self.target_local[label] = ctypes.create_string_buffer(
                    slots * block_size
                )
        config = KVCRConfig(
            nixl_agent_name="source",
            pool_layouts=[(label, block_size) for label in labels],
        )

        def dram(memories):
            return LocalDramOptions(
                [
                    (label, ctypes.addressof(memory), len(memory))
                    for label, memory in memories.items()
                ]
            )

        self.source = _new_kvcr(
            self.source_agent,
            self.source_pinning,
            self.source_control,
            config,
            name="source",
            local_dram=dram(self.source_local),
            g3=source_g3,
            framework_regions=[
                _buffer_region(memory, block_size, label)
                for label, memory in self.source_primary.items()
            ],
            policy=source_policy or _KeepPolicy(),
            key_adapter=_ConstantHashAdapter(),
            inventory_sink=self.source_events.append,
        )
        self.source._core._clock = lambda: self.source_now
        self.target = _new_kvcr(
            self.target_agent,
            FakePrimaryPinning(),
            self.target_control,
            KVCRConfig(
                nixl_agent_name="target",
                pool_layouts=[(label, block_size) for label in target_labels],
            ),
            local_dram=dram(self.target_local),
            framework_regions=[
                _buffer_region(memory, block_size, label)
                for label, memory in self.target_primary.items()
            ],
            key_adapter=_ConstantHashAdapter(),
            remote_options=RemoteFWDramOptions(
                eager_ctrl_connect=False, opportunistic_query=opportunistic_query
            ),
            inventory_sink=self.target_events.append,
        )
        self.target._core._clock = lambda: 0.0

    def descriptors(self, index=0, *, labels=None, endpoint="source"):
        return [
            _mem_descriptor(index, label=label, end_point_name=endpoint)
            for label in self.labels
            if labels is None or label in labels
        ]

    def deposit(self, key, *, index=0, no_evict=True):
        self.source_agent.state = "DONE"
        handle = self.source.deposit(
            {key: self.descriptors(index)},
            no_evict=no_evict,
            expected_layout=list(self.labels),
        )
        result = dict(_poll_until(self.source, bool))[handle][key]
        self.source_agent.state = "PROC"
        return result

    def start(self, keys, *, move=True, destinations=None):
        self.advise("move" if move else "copy")
        if destinations is not None:
            return self.target.deliver(destinations, "req")
        return self.target.fetch(keys, "req", expected_layout=list(self.labels))

    def advise(self, mode="copy", *, hashes=(123,), source="tcp://source:1"):
        self.target.submit_hint(
            _HintEnvelope(
                "0.1",
                "move-message",
                [
                    _HintAction(
                        "move-action",
                        "kv.fetch",
                        "1.0",
                        {
                            "mode": mode,
                            "source_control_endpoint": source,
                            "block_hashes": list(hashes),
                        },
                    )
                ],
            ),
            "req",
        )

    def pump(self, *, target_requests=True, source_poll=True):
        for index, (worker, control, agent, other_control, other_agent) in enumerate(
            (
                (
                    self.source,
                    self.source_control,
                    self.source_agent,
                    self.target_control,
                    self.target_agent,
                ),
                (
                    self.target,
                    self.target_control,
                    self.target_agent,
                    self.source_control,
                    self.source_agent,
                ),
            )
        ):
            if index or source_poll:
                self.results[index].update(worker.poll_completed())
            sent = control.sent[self.control_offsets[index] :]
            self.control_offsets[index] += len(sent)
            for _, raw in sent:
                if index == 1 and not target_requests:
                    continue
                other_control.incoming.append(raw)
            notifications = agent.sent_notifs[self.notif_offsets[index] :]
            self.notif_offsets[index] += len(notifications)
            other_agent.notifs.setdefault("peer", []).extend(
                raw for _, raw in notifications
            )
            for transfer, data in enumerate(agent.xfers, 1):
                native_id = index, transfer
                if (
                    native_id in self.delivered_native
                    or transfer not in agent.released_xfers
                ):
                    continue
                self.delivered_native.add(native_id)
                if data[5]:
                    other_agent.notifs.setdefault("peer", []).append(data[5])
        time.sleep(0.001)

    def until(self, predicate, **kwargs):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.pump(**kwargs)
            if predicate():
                return
        raise AssertionError("Move did not reach the expected state")

    def result(self, handle):
        self.until(lambda: handle in self.results[1])
        return self.results[1].pop(handle)

    def copy_done(self):
        self.source_agent.state = "DONE"


def _requests(pair, worker="target", message_type=None):
    messages = [
        _decode_control_message(raw)
        for _, raw in getattr(pair, f"{worker}_control").sent
    ]
    return [
        item
        for item in messages
        if message_type is None or item["type"] == message_type
    ]


def _invalidations(pair):
    """Cached keys the destination asked the source to drop, per message."""
    return [tuple(item["keys"]) for item in _requests(pair, message_type="invalidate")]


def _inventory_keys(events, *, removed):
    assert all(event.tier is CacheTier.LOCAL_G2 for event in events)
    return Counter(
        key for event in events if event.removed is removed for key in event.keys
    )


def _cache_at_target(pair, key, *, index=0):
    pair.target_agent.state = "DONE"
    deposited = pair.target.deposit(
        {key: pair.descriptors(index, endpoint="target")},
        expected_layout=list(pair.labels),
    )
    result = dict(_poll_until(pair.target, bool))[deposited][key]
    assert result.success and result.release_handle is None


@pytest.mark.parametrize("moving", [False, True], ids=["copy", "move"])
@pytest.mark.parametrize("cached_count", [0, 1, 2])
@pytest.mark.parametrize("protected", [False, True])
def test_copy_and_move_with_destination_cache_hits(moving, cached_count, protected):
    pair = _Pair(slots=2)
    keys = (BlockKey(b"first"), BlockKey(b"second"))
    pair.source_primary[""].raw = b"a" * 16 + b"b" * 16 + b"c" * 16
    pair.target_primary[""].raw = pair.source_primary[""].raw
    retentions = []
    for index, key in enumerate(keys):
        result = pair.deposit(key, index=index, no_evict=protected)
        if result.release_handle is not None:
            retentions.append(result.release_handle)
        if index < cached_count:
            _cache_at_target(pair, key, index=index)
    pair.source_events.clear()
    pair.target_events.clear()
    before = len(pair.source_agent.xfers)
    retry = moving and cached_count == 1 and not protected
    if retry:
        # The target's metadata acknowledgment outlived the source's route cache.
        pair.target._core._remote_fw_dram._metadata_acked_sources.add("tcp://source:1")
    handle = pair.start(keys, move=moving)
    if cached_count == 2:
        results = dict(pair.target.poll_completed())[handle]
    else:
        pair.until(lambda: len(pair.source_agent.xfers) == before + 1)
        if moving and not protected:
            # Moving keys stop serving at admission; cached keys are dropped
            # while the physical transfer is still running.
            pair.until(
                lambda: pair.source.query(keys) == [(QueryStatus.MISS, None)] * 2
            )
            pair.until(
                lambda: all(
                    key not in pair.source._core._block_record_map
                    for key in keys[:cached_count]
                )
            )
            assert _inventory_keys(pair.source_events, removed=True) == Counter(
                keys[:cached_count]
            )
            assert pair.source._core._invalidation_handler.has_pending()
        elif moving:
            # A no_evict guarantee keeps protected copies readable throughout.
            for _ in range(3):
                pair.pump()
            assert (
                pair.source.query(keys) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)] * 2
            )
            assert not pair.source._core._invalidation_handler.has_pending()
        pair.copy_done()
        results = pair.result(handle)
    assert all(result.success for result in results.values())
    assert pair.target_local[""].raw == b"a" * 16 + b"b" * 16
    expected = (
        (QueryStatus.MISS, None)
        if moving and not protected
        else (QueryStatus.HIT, CacheTier.LOCAL_G2)
    )
    pair.until(lambda: pair.source.query(keys) == [expected, expected])
    pair.until(lambda: not pair.source._core._invalidation_handler.has_pending())
    requests = _requests(pair, message_type="start_write")
    assert len(requests) == int(cached_count < 2) + int(retry)
    if retry:
        assert "target_agent_metadata" not in requests[0]
        assert requests[1]["target_agent_metadata"] == b"target-md"
        for field in ("keys", "dst_descriptors", "move"):
            assert requests[0][field] == requests[1][field]
    for request in requests:
        assert len(request["keys"]) == len(keys) - cached_count
        assert request.get("move", False) is moving
        assert all(request["dst_descriptors"])
    standalone = _requests(pair, message_type="invalidate")
    assert len(standalone) == int(moving and cached_count > 0)
    if standalone:
        assert standalone[0]["keys"] == list(keys[:cached_count])
        assert standalone[0]["key_layout"] == list(pair.labels)
    assert len(pair.source_agent.xfers) == before + int(cached_count < 2)
    assert _inventory_keys(pair.source_events, removed=True) == Counter(
        keys if moving and not protected else ()
    )
    assert _inventory_keys(pair.target_events, removed=False) == Counter(
        keys[cached_count:]
    )
    assert not _inventory_keys(pair.target_events, removed=True)
    handles = tuple(result.release_handle for result in results.values())
    assert all(handle is not None for handle in handles)
    assert pair.target.release(handles) == [(handle, True) for handle in handles]
    assert pair.source.release(retentions) == [(handle, True) for handle in retentions]


@pytest.mark.parametrize("replace_advice", [False, True])
def test_failed_move_preserves_source_and_successful_cached_result(replace_advice):
    pair = _Pair(slots=2)
    cached, physical = BlockKey(b"cached"), BlockKey(b"physical")
    pair.target_primary[""].raw = pair.source_primary[""].raw
    pair.deposit(cached, no_evict=False)
    pair.deposit(physical, index=1, no_evict=False)
    _cache_at_target(pair, cached)
    before = len(pair.source_agent.xfers)
    moved = pair.start((cached, physical))
    pair.until(lambda: len(pair.source_agent.xfers) == before + 1)
    if replace_advice:
        pair.target.discard_hint("req")
        pair.advise(source="tcp://replacement:1")
        replacement = pair.target._core._remote_fw_dram._request_hints["req"]
    pair.source_agent.state = "ERR"
    results = pair.result(moved)
    assert results[cached].success and not results[physical].success
    assert results[physical].release_handle is None
    pair.source_agent.state = "DONE"
    pair.until(lambda: not _has_outstanding_operations(pair.source))
    pair.until(lambda: not pair.source._core._invalidation_handler.has_pending())
    assert _invalidations(pair) == [(cached,)]
    assert not _requests(pair, message_type="move_done")
    assert pair.source.query((cached, physical)) == [
        (QueryStatus.MISS, None),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]
    assert pair.source_local[""].raw == b"a" * 32
    assert pair.target.release((results[cached].release_handle,)) == [
        (results[cached].release_handle, True)
    ]
    if replace_advice:
        assert pair.target._core._remote_fw_dram._request_hints["req"] == replacement


@pytest.mark.parametrize("moving", [False, True], ids=["copy", "move"])
@pytest.mark.parametrize(
    "local_piece,advice_change",
    [
        ("ready", None),
        ("inflight", None),
        ("inflight", "discard"),
        ("inflight", "replace"),
        ("inflight", "conflict"),
        ("fails", None),
    ],
)
def test_partial_move_settles_local_writers_before_source_removal(
    moving, local_piece, advice_change, monkeypatch
):
    pair = _Pair(labels=("a", "b"))
    key = BlockKey(b"partial")
    pair.deposit(key, no_evict=False)
    pair.target_primary["a"].raw = b"a" * 32
    pair.target_agent.state = "DONE" if local_piece == "ready" else "PROC"
    deposited = pair.target.deposit(
        {key: pair.descriptors(labels=("a",), endpoint="target")},
        expected_layout=["a", "b"],
    )
    _wait_until(lambda: bool(pair.target_agent.xfers))
    if local_piece == "ready":
        assert dict(_poll_until(pair.target, bool))[deposited][key].success
    reported = []
    poll_completed = pair.target.poll_completed

    def observe_completions():
        for item in poll_completed():
            reported.append(item[0])
            yield item

    monkeypatch.setattr(pair.target, "poll_completed", observe_completions)
    handle = pair.start((key,), move=moving)
    if advice_change in ("discard", "replace"):
        pair.target.discard_hint("req")
    if advice_change in ("replace", "conflict"):
        pair.advise(hashes=(999,), source="tcp://replacement:1")
    future_hint = pair.target._core._remote_fw_dram._request_hints.get("req")
    if moving and local_piece != "ready":
        # Source success will remove the whole key, so a local piece must first
        # succeed or become part of the same remote transfer.
        for _ in range(3):
            pair.pump()
        assert not _requests(pair, message_type="start_write")
        assert handle not in pair.results[1]
        assert pair.source.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
        if local_piece == "fails":
            states = iter(("ERR", "DONE"))
            pair.target_agent.check_xfer_state = lambda _: next(states, "DONE")
        else:
            pair.target_agent.state = "DONE"
        pair.until(lambda: deposited in pair.results[1])
        assert pair.results[1].pop(deposited)[key].success is (local_piece != "fails")
    pair.until(lambda: bool(_requests(pair, message_type="start_write")))
    request = _requests(pair, message_type="start_write")[0]
    assert request["source_control_endpoint"] == "tcp://source:1"
    assert request["key_layout"] == ["a", "b"]
    assert request.get("move", False) is moving
    assert [ref["label"] for ref in request["dst_descriptors"][0]] == (
        ["a", "b"] if moving and local_piece == "fails" else ["b"]
    )
    if local_piece == "fails" and not moving:
        # Ordinary copying can still fill its missing pieces independently.
        states = iter(("ERR", "DONE"))
        pair.target_agent.check_xfer_state = lambda _: next(states, "DONE")
        pair.until(lambda: deposited in pair.results[1])
        assert not pair.results[1].pop(deposited)[key].success
        pair.copy_done()
        pair.until(lambda: len(_requests(pair, message_type="start_write")) == 2)
        second = _requests(pair, message_type="start_write")[1]
        assert [ref["label"] for ref in second["dst_descriptors"][0]] == ["a"]
    else:
        pair.copy_done()
        if local_piece == "inflight" and not moving:
            for _ in range(3):
                pair.pump()
            assert handle not in pair.results[1]
            assert not _invalidations(pair)
            assert pair.source.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
        pair.target_agent.state = "DONE"
    result = pair.result(handle)[key]
    assert result.success
    expected = (
        (QueryStatus.MISS, None) if moving else (QueryStatus.HIT, CacheTier.LOCAL_G2)
    )
    pair.until(lambda: pair.source.query((key,)) == [expected])
    pair.until(lambda: not pair.source._core._invalidation_handler.has_pending())
    assert not _invalidations(pair) and not _requests(pair, message_type="move_done")
    assert _inventory_keys(pair.target_events, removed=False) == Counter((key,))
    assert pair.source._core._local_dram._pools.telemetry_state()[
        "local_g2_free_slots"
    ] == (2 if moving else 0)
    assert pair.target_local["a"].raw == b"a" * 16
    assert pair.target_local["b"].raw == b"b" * 16
    assert pair.target.release((result.release_handle,)) == [
        (result.release_handle, True)
    ]
    pair.until(
        lambda: (
            not _has_outstanding_operations(pair.source)
            and not _has_outstanding_operations(pair.target)
        )
    )
    assert not pair.target._core._local_dram._pending_residency_ops
    assert reported.count(handle) == 1
    if advice_change is not None:
        assert (
            pair.target._core._remote_fw_dram._request_hints.get("req") == future_hint
        )


@pytest.mark.parametrize(
    "cached,slow_source", [(False, False), (False, True), (True, False)]
)
def test_move_withdraws_before_transfer_then_drains_existing_readers(
    cached, slow_source, monkeypatch
):
    pair = _Pair()
    key = BlockKey(b"draining")
    pair.deposit(key, no_evict=False)
    if cached:
        pair.target_primary[""].raw = pair.source_primary[""].raw
        _cache_at_target(pair, key)
    read = pair.source.fetch((key,))
    guard = dict(pair.source.poll_completed())[read][key].release_handle
    core = pair.source._core
    pools = core._local_dram._pools
    residency = core._block_record_map[key].local_dram
    if slow_source:
        monkeypatch.setattr(
            core._remote_fw_dram,
            "_try_local_source_write",
            lambda *args, **kwargs: False,
        )
    before = len(pair.source_agent.xfers)
    moved = pair.start((key,))
    pair.until(lambda: core._invalidation_handler.has_pending())
    if not cached:
        pair.until(lambda: len(pair.source_agent.xfers) == before + 1)
    assert pair.source.query((key,)) == [(QueryStatus.MISS, None)]
    assert core._invalidation_handler.has_pending()
    assert (
        pools.acquire_readers((key,), (residency.refs,), residency.key_layout) is False
    )
    rejected = [
        pair.source.fetch((key,)),
        pair.source.deliver({key: pair.descriptors()}),
        pair.source.deposit({key: pair.descriptors()}, no_evict=False),
        pair.source.deposit({key: pair.descriptors()}, no_evict=True),
    ]
    completed = dict(pair.source.poll_completed())
    assert all(not completed[handle][key].success for handle in rejected)
    assert len(pair.source_agent.xfers) == before + int(not cached)
    assert _invalidations(pair) == ([(key,)] if cached else [])
    with core._state_lock:
        pair.copy_done()
        pair.until(lambda: moved in pair.results[1], source_poll=False)
        result = pair.results[1].pop(moved)[key]
        assert result.success and pair.target_local[""].raw == b"a" * 16
        assert core._block_record_map[key].local_dram is residency
    assert pair.source.query((key,)) == [(QueryStatus.MISS, None)]
    assert core._block_record_map[key].local_dram is residency
    assert pair.target.release((result.release_handle,)) == [
        (result.release_handle, True)
    ]
    _invalidate(pair.target, (key,))
    pair.until(lambda: not pair.target._core._invalidation_handler.has_pending())
    pair.source_pinning.prefix_length = 0
    before = len(pair.source_agent.xfers)
    copied = pair.result(pair.start((key,), move=False))[key]
    assert not copied.success and len(pair.source_agent.xfers) == before
    assert pair.source_local[""].raw == b"a" * 16
    dropped = []
    original_drop = pools._drop

    def drop_without_reader_lock(key):
        assert pools._source_state.lock.acquire(blocking=False)
        pools._source_state.lock.release()
        original_drop(key)
        dropped.append(key)

    monkeypatch.setattr(pools, "_drop", drop_without_reader_lock)
    assert pair.source.release((guard,)) == [(guard, True)]
    pair.until(lambda: bool(dropped))
    assert dropped == [key] and key not in core._block_record_map
    assert pools.telemetry_state()["local_g2_free_slots"] == 1


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "advice",
    [
        "missing",
        "discarded",
        "conflicting",
        "unmatched",
        "copy",
        "no_adapter",
        "known_missing",
    ],
)
def test_fetch_preserves_copy_semantics_without_usable_move_advice(cached, advice):
    pair = _Pair()
    key = BlockKey(b"copy-fallback")
    pair.target_primary[""].raw = pair.source_primary[""].raw
    pair.deposit(key, no_evict=False)
    if cached:
        _cache_at_target(pair, key)
    if advice == "discarded":
        pair.advise("move")
        pair.target.discard_hint("req")
    elif advice == "conflicting":
        pair.advise("move")
        pair.advise("move", source="tcp://different-source:1")
    elif advice == "unmatched":
        pair.advise("move", hashes=(999,))
    elif advice == "copy":
        pair.advise("copy")
    elif advice == "no_adapter":
        pair.advise("move")
        pair.target._core._remote_fw_dram._key_adapter = None
    elif advice == "known_missing":
        pair.advise("move")
        remote = pair.target._core._remote_fw_dram
        remote._request_hints["req"] = replace(
            remote._request_hints["req"], missing_keys=frozenset((key,))
        )
    handle = pair.target.fetch((key,), "req")
    pair.copy_done()
    result = pair.result(handle)[key]
    assert result.success is (cached or advice == "copy")
    for _ in range(3):
        pair.pump()
    assert not _invalidations(pair)
    assert len(_requests(pair, message_type="start_write")) == int(
        not cached and advice == "copy"
    )
    assert pair.source.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    if result.success:
        assert result.release_handle is not None
        assert pair.target.release((result.release_handle,)) == [
            (result.release_handle, True)
        ]
    else:
        assert result.release_handle is None


@pytest.mark.parametrize(
    "cached,opportunistic", [(True, False), (False, False), (False, True)]
)
def test_fetch_moves_only_hash_matched_keys(cached, opportunistic, monkeypatch):
    pair = _Pair(slots=2, opportunistic_query=opportunistic)
    moved, copied = BlockKey(b"moved"), BlockKey(b"copied")
    pair.source_primary[""].raw = b"a" * 16 + b"b" * 16 + b"c" * 16
    pair.target_primary[""].raw = pair.source_primary[""].raw
    for index, key in enumerate((moved, copied)):
        pair.deposit(key, index=index, no_evict=False)
        if cached:
            _cache_at_target(pair, key, index=index)
    monkeypatch.setattr(
        pair.target._core._remote_fw_dram._key_adapter,
        "encode_to_router",
        lambda key: 123 if key == moved else 456,
    )
    pair.advise("move", hashes=(123,))
    handle = pair.target.fetch((moved, copied), "req")
    pair.copy_done()
    results = pair.result(handle)
    assert results[moved].success
    assert results[copied].success is cached
    pair.until(lambda: pair.source.query((moved,)) == [(QueryStatus.MISS, None)])
    pair.until(lambda: not pair.source._core._invalidation_handler.has_pending())
    assert _invalidations(pair) == ([(moved,)] if cached else [])
    for request in _requests(pair, message_type="start_write"):
        assert request["keys"] == [moved] and request["move"]
    assert pair.source.query((moved, copied)) == [
        (QueryStatus.MISS, None),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]
    for key, result in results.items():
        if not result.success:
            continue
        address = (
            ctypes.addressof(pair.target_local[""])
            + 16 * result.descriptors[0].element_index
        )
        assert ctypes.string_at(address, 16) == (b"a" if key == moved else b"b") * 16
        assert pair.target.release((result.release_handle,)) == [
            (result.release_handle, True)
        ]


@pytest.mark.parametrize("succeeds", [False, True])
def test_move_joining_a_copy_invalidates_only_after_complete_residency(succeeds):
    pair = _Pair()
    key = BlockKey(b"joined-copy")
    pair.deposit(key, no_evict=False)
    before = len(pair.source_agent.xfers)
    copied = pair.start((key,), move=False)
    pair.until(lambda: len(pair.source_agent.xfers) == before + 1)
    moved = pair.start((key,))
    for _ in range(3):
        pair.pump()
    assert not _invalidations(pair)
    requests = _requests(pair, message_type="start_write")
    assert len(requests) == 1 and not requests[0].get("move", False)
    assert pair.source.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]

    pair.source_agent.state = "DONE" if succeeds else "ERR"
    pair.until(lambda: copied in pair.results[1] and moved in pair.results[1])
    results = [pair.results[1].pop(handle)[key] for handle in (copied, moved)]
    assert all(result.success is succeeds for result in results)
    pair.source_agent.state = "DONE"
    pair.until(lambda: not _has_outstanding_operations(pair.source))
    if succeeds:
        pair.until(lambda: key not in pair.source._core._block_record_map)
        assert _invalidations(pair) == [(key,)]
    else:
        assert not _invalidations(pair)
        assert pair.source.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert not _requests(pair, message_type="move_done")
    assert len(pair.source_agent.xfers) == before + 1
    assert _inventory_keys(pair.target_events, removed=False) == Counter(
        (key,) if succeeds else ()
    )
    handles = tuple(result.release_handle for result in results if result.success)
    assert pair.target.release(handles) == [(handle, True) for handle in handles]


def test_fetch_ignores_self_move_advice_for_local_hit():
    pair = _Pair()
    key = BlockKey(b"self-source")
    _cache_at_target(pair, key)
    pair.advise("move", source=pair.target_control.endpoint)
    handle = pair.target.fetch((key,), "req", expected_layout=[""])
    result = dict(pair.target.poll_completed())[handle][key]
    assert result.success and result.release_handle is not None
    for _ in range(3):
        pair.pump()
    assert not _requests(pair)
    assert pair.target.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert pair.target.release((result.release_handle,)) == [
        (result.release_handle, True)
    ]


def test_deliver_copies_under_move_advice():
    pair = _Pair()
    key = BlockKey(b"delivery")
    pair.deposit(key, no_evict=False)
    handle = pair.start((key,), destinations={key: pair.descriptors(endpoint="target")})
    pair.copy_done()
    result = pair.result(handle)[key]
    assert result.success and result.release_handle is None
    assert pair.target_primary[""].raw[:16] == b"a" * 16
    assert pair.source.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert not _invalidations(pair)


@pytest.mark.parametrize("after_allowance", [False, True])
def test_move_notification_uses_extended_destination_allowance(after_allowance):
    pair = _Pair()
    key = BlockKey(b"late")
    pair.deposit(key, no_evict=False)
    held = []
    real_get_new_notifs = pair.target_agent.get_new_notifs

    def hold_notifs():
        for raw in real_get_new_notifs().values():
            held.extend(raw)
        return {}

    pair.target_agent.get_new_notifs = hold_notifs
    before = len(pair.source_agent.xfers)
    handle = pair.start((key,))
    pair.until(lambda: len(pair.source_agent.xfers) == before + 1)
    source_writes = [
        op
        for op in pair.source._core._progress._in_flight_ops.values()
        if op.op_id[0] == "source"
    ]
    assert source_writes and all(
        op.deadline <= pair.source._core.config.operation_timeout_ms / 1000
        for op in source_writes
    )
    pair.copy_done()
    # Source cleanup finishes even while the destination cannot see write_done.
    pair.until(lambda: key not in pair.source._core._block_record_map)
    _wait_until(lambda: bool(held))
    assert handle not in pair.results[1]
    config = pair.target._core.config
    elapsed = (
        config.abandon_timeout_ms / 1000 + 0.1
        if after_allowance
        else config.operation_timeout_ms / 1000 + 0.1
    )
    pair.target._core._clock = lambda: elapsed
    if not after_allowance:
        for _ in range(3):
            pair.pump()
        assert handle not in pair.results[1]
    pair.target_agent.get_new_notifs = real_get_new_notifs
    pair.target_agent.notifs.setdefault("peer", []).extend(held)
    result = pair.result(handle)[key]
    assert result.success is not after_allowance
    assert (result.release_handle is None) is after_allowance
    pair.until(
        lambda: (
            not _has_outstanding_operations(pair.source)
            and not _has_outstanding_operations(pair.target)
        )
    )
    assert not _invalidations(pair) and not _requests(pair, message_type="move_done")
    assert pair.source.query((key,)) == [(QueryStatus.MISS, None)]
    assert pair.source_local[""].raw == b"a" * 16
    assert pair.target.query((key,)) == [
        (QueryStatus.MISS, None)
        if after_allowance
        else (QueryStatus.HIT, CacheTier.LOCAL_G2)
    ]
    assert _inventory_keys(pair.source_events, removed=True) == Counter((key,))
    assert _inventory_keys(pair.target_events, removed=False) == Counter(
        () if after_allowance else (key,)
    )
    if result.release_handle is not None:
        assert pair.target.release((result.release_handle,)) == [
            (result.release_handle, True)
        ]


def test_mixed_move_reports_missing_source_against_original_key_indices():
    pair = _Pair(slots=3)
    cached, missing, physical = map(BlockKey, (b"cached", b"missing", b"physical"))
    pair.source_primary[""].raw = b"a" * 16 + b"b" * 16 + b"c" * 16 + b"d" * 16
    pair.target_primary[""].raw = pair.source_primary[""].raw
    pair.deposit(cached, no_evict=False)
    pair.deposit(physical, index=2, no_evict=False)
    pair.source_pinning.prefix_length = 0
    _cache_at_target(pair, cached)
    handle = pair.start((cached, missing, physical))
    pair.copy_done()
    results = pair.result(handle)
    assert results[cached].success and results[physical].success
    assert not results[missing].success and results[missing].release_handle is None
    address = (
        ctypes.addressof(pair.target_local[""])
        + results[physical].descriptors[0].element_index * 16
    )
    assert ctypes.string_at(address, 16) == b"c" * 16
    requests = _requests(pair, message_type="start_write")
    assert len(requests) == 1 and requests[0]["keys"] == [missing, physical]
    assert requests[0]["move"]
    assert _invalidations(pair) == [(cached,)]
    pair.until(
        lambda: pair.source.query((cached, physical)) == [(QueryStatus.MISS, None)] * 2
    )
    pair.until(lambda: not pair.source._core._invalidation_handler.has_pending())
    assert _inventory_keys(pair.source_events, removed=True) == Counter(
        (cached, physical)
    )
    handles = (results[cached].release_handle, results[physical].release_handle)
    assert pair.target.release(handles) == [(handle, True) for handle in handles]


@pytest.mark.parametrize("moving", [False, True], ids=["copy", "move"])
@pytest.mark.parametrize("writer_running", [False, True], ids=["missing", "writing"])
def test_move_rejects_busy_source_without_blocking_its_producer(moving, writer_running):
    pair = _Pair(slots=2, labels=("a", "b"), source_agent=_IndividualWriteAgent())
    ready, busy = BlockKey(b"ready"), BlockKey(b"busy")
    pair.deposit(ready, index=1, no_evict=False)
    pair.source_agent.state = "DONE"
    partial = pair.source.deposit(
        {busy: pair.descriptors(labels=("a",))}, expected_layout=["a", "b"]
    )
    assert dict(_poll_until(pair.source, bool))[partial][busy].success
    pair.source_agent.state = "PROC"
    writer = None
    if writer_running:
        before = len(pair.source_agent.xfers)
        writer = pair.source.deposit(
            {busy: pair.descriptors(labels=("b",))}, expected_layout=["a", "b"]
        )
        pair.until(lambda: len(pair.source_agent.xfers) == before + 1)
    before = len(pair.source_agent.xfers)
    handle = pair.start((ready, busy), move=moving)
    core = pair.source._core
    if moving:
        pair.until(lambda: len(pair.source_agent.xfers) == before + 1)
        pair.source_agent.states[before + 1] = "DONE"
        results = pair.result(handle)
        assert results[ready].success and not results[busy].success
        assert results[busy].release_handle is None
        pair.until(lambda: not core._invalidation_handler.has_pending())
    else:
        pair.until(lambda: bool(core._remote_fw_dram._sources._source_pin_ops))
        for _ in range(3):
            pair.pump()
        assert handle not in pair.results[1]
        assert len(pair.source_agent.xfers) == before
    assert not core._block_record_map[busy].pending_invalidation
    assert busy not in _inventory_keys(pair.source_events, removed=True)
    # A rejected move must leave the producer free to finish missing pieces.
    followup = pair.source.deposit(
        {busy: pair.descriptors(labels=("b",))}, expected_layout=["a", "b"]
    )
    pair.source_agent.states.clear()
    pair.copy_done()
    pair.until(lambda: followup in pair.results[0])
    assert pair.results[0][followup][busy].success
    if writer is not None:
        assert pair.results[0][writer][busy].success
    if not moving:
        results = pair.result(handle)
        assert all(entry.success for entry in results.values())
    assert pair.source_now == pair.target._core._clock() == 0
    read = pair.source.fetch((busy,), expected_layout=["a", "b"])
    result = dict(pair.source.poll_completed())[read][busy]
    assert result.success
    for descriptor in result.descriptors:
        address = ctypes.addressof(pair.source_local[descriptor.label])
        assert (
            ctypes.string_at(address + descriptor.element_index * 16, 16)
            == (pair.source_primary[descriptor.label].raw[:16])
        )
    assert pair.source.release((result.release_handle,)) == [
        (result.release_handle, True)
    ]
    claims = tuple(entry.release_handle for entry in results.values() if entry.success)
    assert pair.target.release(claims) == [(claim, True) for claim in claims]
    assert _inventory_keys(pair.source_events, removed=True) == Counter(
        (ready,) if moving else ()
    )


@pytest.mark.parametrize("matching_layout", [False, True])
@pytest.mark.parametrize("declared_layout", [False, True])
@pytest.mark.parametrize("no_evict", [False, True])
@pytest.mark.parametrize("destination_cached", [False, True])
def test_move_checks_layout_of_capacity_waiting_deposit(
    tmp_path, matching_layout, declared_layout, no_evict, destination_cached
):
    page = os.sysconf("SC_PAGE_SIZE")
    pair = _Pair(
        labels=("a", "b"),
        target_labels=("a",),
        block_size=page,
        source_agent=_IndividualG3WriteAgent(),
        source_policy=_MoveLocalToG3Policy(),
        source_g3=G3Options(
            paths=(tmp_path / "source.g3",),
            capacity_bytes_per_file=page * 8,
            backend="MOCK",
        ),
    )
    key, occupant = BlockKey(b"waiting-deposit"), BlockKey(b"occupant")
    pair.source_primary["a"].raw = b"a" * page + b"A" * page
    pair.source_primary["b"].raw = b"b" * page + b"B" * page
    pair.target_primary["a"].raw = pair.source_primary["a"].raw
    pair.deposit(occupant, no_evict=False)
    source_layout = ["a"] if matching_layout else ["a", "b"]
    deposited = pair.source.deposit(
        {key: pair.descriptors(1, labels=source_layout)},
        expected_layout=source_layout if declared_layout else None,
        no_evict=no_evict,
    )
    source = pair.source._core
    assert source._block_record_map[key].local_dram is None
    assert len(source._local_dram._capacity_waiters) == 1
    pair.until(lambda: len(pair.source_agent.xfers) == 2)

    def poll_framework_a():
        for request, (handle, blocks) in pair.source_pinning.poll_pin_results():
            yield (
                request,
                (
                    handle,
                    {key: pair.descriptors(1, labels=("a",)) for key in blocks},
                ),
            )

    source._poll_pin_results_callback = poll_framework_a
    if destination_cached:
        pair.target_agent.state = "DONE"
        cached = pair.target.deposit(
            {key: pair.descriptors(1, labels=("a",), endpoint="target")},
            expected_layout=["a"],
        )
        assert dict(_poll_until(pair.target, bool))[cached][key].success
    pair.advise("move")
    moved = pair.target.fetch((key,), "req", expected_layout=["a"])
    if destination_cached:
        target_result = dict(pair.target.poll_completed())[moved][key]
        pair.until(lambda: source._invalidation_handler.has_pending())
        assert not pair.source_pinning.searches
    else:
        # A capacity-waiting deposit is already admitted; a move must not fence it.
        target_result = pair.result(moved)[key]
        assert not source._invalidation_handler.has_pending()
        assert not pair.source_pinning.searches
        assert len(pair.source_agent.xfers) == 2
    assert target_result.success is destination_cached
    if destination_cached:
        assert target_result.release_handle is not None
        assert pair.target_local["a"].raw == b"A" * page
    else:
        assert target_result.release_handle is None
    # The source has accepted the deposit but has no residency to check yet.
    assert source._block_record_map[key].local_dram is None
    assert deposited not in pair.results[0]

    pair.copy_done()
    pair.until(lambda: deposited in pair.results[0])
    source_result = pair.results[0][deposited][key]
    assert source_result.success
    assert (source_result.release_handle is not None) is no_evict
    for label in source_layout:
        assert pair.source_local[label].raw == pair.source_primary[label].raw[page:]
    pair.until(lambda: not source._invalidation_handler.has_pending())
    preserved = not destination_cached or not matching_layout or no_evict
    assert pair.source.query((key,)) == [
        (QueryStatus.HIT, CacheTier.LOCAL_G2) if preserved else (QueryStatus.MISS, None)
    ]
    if preserved:
        assert source._block_record_map[key].local_dram.key_layout == tuple(
            source_layout
        )
    else:
        # An admitted deposit allocated after withdrawal; dropping clears its entry.
        assert key not in source._local_dram._pools._source_state.index
    if not any((matching_layout, no_evict, declared_layout, destination_cached)):
        # Preserving an unprotected copy must leave its slots available to policy.
        replacement = BlockKey(b"capacity-reuse")
        assert pair.deposit(replacement, no_evict=False).success
        for label in source_layout:
            assert pair.source_local[label].raw == pair.source_primary[label].raw[:page]
    if target_result.release_handle is not None:
        assert pair.target.release((target_result.release_handle,)) == [
            (target_result.release_handle, True)
        ]
    if no_evict:
        assert pair.source.release((source_result.release_handle,)) == [
            (source_result.release_handle, True)
        ]
    pair.until(
        lambda: (
            not _has_outstanding_operations(pair.source)
            and not _has_outstanding_operations(pair.target)
        )
    )
    assert moved not in pair.results[1]
    for core in (source, pair.target._core):
        pools = core._local_dram._pools
        assert not pools._public_claims
        assert not pools._source_state.active
        assert not core._local_dram_sources_by_op
        assert all(
            not record.in_flight_ops for record in core._block_record_map.values()
        )


def test_waiting_move_preserves_later_invalidation_of_rejected_key(
    tmp_path, monkeypatch
):
    page = os.sysconf("SC_PAGE_SIZE")
    pair = _Pair(
        slots=2,
        labels=("a", "b"),
        target_labels=("a",),
        block_size=page,
        source_agent=_IndividualG3WriteAgent(),
        source_policy=_MoveLocalToG3Policy(),
        source_g3=G3Options(
            paths=(tmp_path / "source.g3",),
            capacity_bytes_per_file=page * 12,
            backend="MOCK",
        ),
    )
    occupant, waiting, mismatched = map(
        BlockKey, (b"occupant", b"waiting", b"mismatched")
    )
    pair.deposit(occupant, no_evict=False)
    pair.source.deposit(
        {waiting: pair.descriptors(1, labels=("a",))}, expected_layout=["a"]
    )
    pair.until(lambda: len(pair.source_agent.xfers) == 2)
    deposited = pair.source.deposit(
        {mismatched: pair.descriptors(2)}, expected_layout=["a", "b"]
    )
    core = pair.source._core
    pair.until(lambda: len(pair.source_agent.xfers) == 3)
    assert core._block_record_map[mismatched].local_dram is None

    def poll_framework_full_layout():
        for request, (pin, blocks) in pair.source_pinning.poll_pin_results():
            yield request, (pin, {key: pair.descriptors(2) for key in blocks})

    core._poll_pin_results_callback = poll_framework_full_layout
    sources = core._remote_fw_dram._sources

    def admit_source(op):
        # Exercise retry ownership after admission; public move now rejects
        # this fixture's busy writer before reaching source selection.
        core._add_block_dependencies(op, new_operation=True)
        op.moving = core._invalidation_handler.submit(op.keys, key_layout=op.key_layout)
        sources._source_pin_ops[op.op_id] = op
        sources._resume_source_pin(op.op_id, op)

    monkeypatch.setattr(sources, "start_source_pin", admit_source)
    pair.advise("move")
    moved = pair.target.fetch((mismatched, waiting), "req", expected_layout=["a"])
    pair.until(
        lambda: (
            bool(sources._source_pin_ops)
            and all(op.keys == {waiting} for op in sources._source_pin_ops.values())
        )
    )
    # The move rejected this layout while waiting for the other key's writer.
    # Finish its admitted deposit, then independently invalidate the new copy.
    pair.source_agent.states[3] = "DONE"
    pair.until(lambda: len(pair.source_agent.xfers) == 4)
    pair.source_agent.states[4] = "DONE"
    pair.until(lambda: deposited in pair.results[0])
    assert pair.results[0][deposited][mismatched].success
    pair.until(lambda: not core._block_record_map[mismatched].pending_invalidation)
    read = pair.source.fetch((mismatched,), expected_layout=["a", "b"])
    guard = dict(pair.source.poll_completed())[read][mismatched].release_handle
    assert guard is not None
    pair.source_control.incoming.append(
        msgspec.msgpack.encode(_Invalidate(keys=(mismatched,), key_layout=("a", "b")))
    )
    pair.until(lambda: core._block_record_map[mismatched].pending_invalidation)

    pair.source_agent.states[2] = "DONE"
    pair.until(lambda: not sources._source_pin_ops)
    preserved = core._block_record_map[mismatched].pending_invalidation
    assert pair.source.release((guard,)) == [(guard, True)]
    pair.source_agent.states.clear()
    pair.copy_done()
    result = pair.result(moved)
    assert not result[mismatched].success and result[waiting].success
    claim = result[waiting].release_handle
    assert pair.target.release((claim,)) == [(claim, True)]
    assert preserved, "Old move cancelled a newer invalidation"
    pair.until(lambda: mismatched not in core._block_record_map)


def test_move_source_close_after_immediate_native_completion():
    pair = _Pair()
    key = BlockKey(b"immediate-move")
    assert pair.deposit(key, no_evict=False).success
    core = pair.source._core
    pools = core._local_dram._pools
    residency = core._block_record_map[key].local_dram
    pair.source_agent.state = "DONE"
    before = len(pair.source_agent.xfers)
    handle = pair.start((key,))
    # The owner admits the move and the write completes natively at once;
    # close() must settle it without another application poll.
    pair.until(lambda: len(pair.source_agent.xfers) == before + 1)
    assert residency.claim_count == 1
    assert core._invalidation_handler.has_pending()
    pair.source.close()
    assert not pools._source_state.active
    assert not core._local_dram_sources_by_op
    assert not core._invalidation_handler.has_pending()
    pair.until(lambda: handle in pair.results[1], source_poll=False)
    result = pair.results[1].pop(handle)[key]
    assert result.success and pair.target_local[""].raw == b"a" * 16
    assert pair.target.release((result.release_handle,)) == [
        (result.release_handle, True)
    ]
