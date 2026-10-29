# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Move reuses fetch while the source independently removes eligible G2 copies."""

import ctypes
import time

import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakePrimaryPinning,
    _buffer_region,
    _ConstantHashAdapter,
    _decode_control_message,
    _HintAction,
    _HintEnvelope,
    _invalidate,
    _mem_descriptor,
    _new_kvcr,
    _poll_until,
    _wait_until,
)
from test_kvcr_remote_target import _CopyingWriteAgent

from kvcr.config import KVCRConfig, LocalDramOptions, RemoteFWDramOptions
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
        )
        self.source._core._clock = lambda: self.source_now
        self.invalidation_calls = []
        submit_invalidation = self.source._core._invalidation_handler.submit

        def record_invalidation(keys):
            self.invalidation_calls.append(tuple(keys))
            return submit_invalidation(keys)

        self.source._core._invalidation_handler.submit = record_invalidation
        self.cached_requests = []
        source_cached = self.source._core._move_handler.source_cached

        def record_cached(request):
            source_cached(request)
            self.cached_requests.append(tuple(request.keys))

        self.source._core._move_handler.source_cached = record_cached
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

    def start(self, keys, *, mode="move", destinations=None, explicit_move=False):
        self.advise(mode)
        if destinations is not None:
            return self.target.deliver(destinations, "req")
        if explicit_move:
            return self.target.move(keys, "req", expected_layout=list(self.labels))
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


def _assert_no_invalidation_messages(pair):
    assert not any(
        item["type"].startswith("invalidate")
        for worker in ("source", "target")
        for item in _requests(pair, worker)
    )


def _cache_at_target(pair, key, *, index=0):
    pair.target_agent.state = "DONE"
    deposited = pair.target.deposit(
        {key: pair.descriptors(index, endpoint="target")},
        expected_layout=list(pair.labels),
    )
    result = dict(_poll_until(pair.target, bool))[deposited][key]
    assert result.success and result.release_handle is None


@pytest.mark.parametrize("mode", ["copy", "hint-move", "explicit-move"])
@pytest.mark.parametrize("cached_count", [0, 1, 2])
@pytest.mark.parametrize("protected", [False, True])
def test_copy_and_move_with_destination_cache_hits(mode, cached_count, protected):
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
    before = len(pair.source_agent.xfers)
    moving = mode != "copy"
    handle = pair.start(
        keys,
        mode="move" if mode == "hint-move" else "copy",
        explicit_move=mode == "explicit-move",
    )
    if cached_count == 2:
        # A fully cached move completes before the source receives its command.
        results = dict(pair.target.poll_completed())[handle]
        assert not pair.invalidation_calls
    else:
        pair.copy_done()
        results = pair.result(handle)
    assert all(result.success for result in results.values())
    assert pair.target_local[""].raw == b"a" * 16 + b"b" * 16
    if moving:
        pair.until(
            lambda: (
                {key for batch in pair.invalidation_calls for key in batch} == set(keys)
            )
        )
        pair.until(lambda: not pair.source._core._invalidation_handler.has_pending())
    else:
        pair.until(
            lambda: not pair.source._core._local_dram._pools._source_state.active
        )
        assert not pair.invalidation_calls
    requests = _requests(pair, message_type="start_write")
    assert len(requests) == int(moving or cached_count < 2)
    if requests:
        request = requests[0]
        assert set(request.get("move_keys", ())) == (set(keys) if moving else set())
        assert (
            sum(bool(refs) for refs in request["dst_descriptors"]) == 2 - cached_count
        )
    assert len(pair.source_agent.xfers) == before + int(cached_count < 2)
    expected = (
        (QueryStatus.MISS, None)
        if moving and not protected
        else (QueryStatus.HIT, CacheTier.LOCAL_G2)
    )
    assert pair.source.query(keys) == [expected, expected]
    handles = tuple(result.release_handle for result in results.values())
    assert all(handle is not None for handle in handles)
    assert pair.target.release(handles) == [(handle, True) for handle in handles]
    assert pair.source.release(retentions) == [(handle, True) for handle in retentions]
    _assert_no_invalidation_messages(pair)


def test_failed_move_preserves_source_and_successful_cached_result():
    pair = _Pair(slots=2)
    cached, physical = BlockKey(b"cached"), BlockKey(b"physical")
    pair.target_primary[""].raw = pair.source_primary[""].raw
    pair.deposit(cached, no_evict=False)
    pair.deposit(physical, index=1, no_evict=False)
    _cache_at_target(pair, cached)
    before = len(pair.source_agent.xfers)
    moved = pair.start((cached, physical), explicit_move=True)
    pair.until(lambda: len(pair.source_agent.xfers) == before + 1)
    pair.source_agent.state = "ERR"
    results = pair.result(moved)
    assert results[cached].success and not results[physical].success
    assert results[physical].release_handle is None
    pair.source_agent.state = "DONE"
    pair.until(lambda: not pair.source._core._move_handler._source_transfers)
    pair.until(
        lambda: cached in {key for batch in pair.invalidation_calls for key in batch}
    )
    pair.until(lambda: not pair.source._core._invalidation_handler.has_pending())
    assert all(physical not in batch for batch in pair.invalidation_calls)
    assert pair.source.query((cached, physical)) == [
        (QueryStatus.MISS, None),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]
    assert pair.source_local[""].raw == b"a" * 32
    assert pair.target.release((results[cached].release_handle,)) == [
        (results[cached].release_handle, True)
    ]


@pytest.mark.parametrize("mode", ["copy", "hint-move", "explicit-move"])
@pytest.mark.parametrize("inflight", [False, True])
def test_partial_move_waits_for_local_piece_then_removes_whole_source(
    mode, inflight, monkeypatch
):
    pair = _Pair(labels=("a", "b"))
    key = BlockKey(b"partial")
    pair.deposit(key, no_evict=False)
    pair.target_primary["a"].raw = b"a" * 32
    pair.target_agent.state = "PROC" if inflight else "DONE"
    deposited = pair.target.deposit(
        {key: pair.descriptors(labels=("a",), endpoint="target")},
        expected_layout=["a", "b"],
    )
    _wait_until(lambda: bool(pair.target_agent.xfers))
    if not inflight:
        assert dict(_poll_until(pair.target, bool))[deposited][key].success
    iterations = 0
    progress = pair.target._core._progress
    original_iteration = progress._run_one_iteration

    def observe_iteration():
        nonlocal iterations
        work = original_iteration()
        iterations += 1
        return work

    monkeypatch.setattr(progress, "_run_one_iteration", observe_iteration)
    moving = mode != "copy"
    handle = pair.start(
        (key,),
        mode="move" if mode == "hint-move" else "copy",
        explicit_move=mode == "explicit-move",
    )
    if moving and inflight:
        after_submit = iterations
        pair.until(lambda: iterations >= after_submit + 2)
        assert not _requests(pair, message_type="start_write")
        assert handle not in pair.results[1]
        pair.target_agent.state = "DONE"
    pair.until(lambda: bool(_requests(pair, message_type="start_write")))
    request = _requests(pair, message_type="start_write")[0]
    assert request["key_layout"] == ["a", "b"]
    assert [ref["label"] for ref in request["dst_descriptors"][0]] == ["b"]
    pair.target_agent.state = "DONE"
    pair.copy_done()
    result = pair.result(handle)[key]
    assert result.success
    if moving:
        pair.until(lambda: bool(pair.invalidation_calls))
        pair.until(lambda: not pair.source._core._invalidation_handler.has_pending())
    assert pair.source.query((key,)) == [
        (QueryStatus.MISS, None) if moving else (QueryStatus.HIT, CacheTier.LOCAL_G2)
    ]
    assert pair.source._core._local_dram._pools.telemetry_state()[
        "local_g2_free_slots"
    ] == (2 if moving else 0)
    assert pair.target_local["a"].raw == b"a" * 16
    assert pair.target_local["b"].raw == b"b" * 16
    assert pair.target.release((result.release_handle,)) == [
        (result.release_handle, True)
    ]


@pytest.mark.parametrize("slow_source", [False, True])
def test_move_completes_before_cleanup_then_drains_readers_and_rejects_new_ones(
    slow_source, monkeypatch
):
    pair = _Pair()
    key = BlockKey(b"draining")
    pair.deposit(key, no_evict=False)
    read = pair.source.fetch((key,))
    guard = dict(pair.source.poll_completed())[read][key].release_handle
    core = pair.source._core
    pools = core._local_dram._pools
    residency = core._block_record_map[key].local_dram
    with core._state_lock:
        moved = pair.start((key,))
        pair.copy_done()
        pair.until(lambda: moved in pair.results[1], source_poll=False)
        result = pair.results[1].pop(moved)[key]
        assert result.success and pair.target_local[""].raw == b"a" * 16
        assert not pair.invalidation_calls
        assert residency.source_readers == 1
    pair.until(lambda: core._invalidation_handler.has_pending())
    assert pair.source.query((key,)) == [(QueryStatus.MISS, None)]
    assert core._block_record_map[key].local_dram is residency
    assert pools.acquire_readers((key,), (residency.refs,), residency.key_layout) == ()
    assert pair.target.release((result.release_handle,)) == [
        (result.release_handle, True)
    ]
    _invalidate(pair.target, (key,))
    pair.until(lambda: not pair.target._core._invalidation_handler.has_pending())
    pair.source_pinning.prefix_length = 0
    if slow_source:
        monkeypatch.setattr(
            core._remote_fw_dram, "_try_local_source_write", lambda *args: False
        )
    before = len(pair.source_agent.xfers)
    copied = pair.result(pair.start((key,), mode="copy"))[key]
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


def test_move_hint_selects_only_matching_keys_in_mixed_request():
    pair = _Pair(slots=3, opportunistic_query=True)
    cached, ordinary, moved = map(BlockKey, (b"cached", b"ordinary", b"moved"))
    pair.target_primary[""].raw = pair.source_primary[""].raw
    for index, key in enumerate((cached, ordinary, moved)):
        pair.deposit(key, index=index, no_evict=False)
    _cache_at_target(pair, cached)

    class Adapter:
        def decode(self, key):
            return 999 if key == ordinary else 123

    pair.target._core._remote_fw_dram._key_adapter = Adapter()
    handle = pair.start((cached, ordinary, moved))
    pair.copy_done()
    results = pair.result(handle)
    assert all(result.success for result in results.values())
    requests = _requests(pair, message_type="start_write")
    assert len(requests) == 1
    assert set(requests[0]["move_keys"]) == {cached, moved}
    pair.until(
        lambda: (
            {key for batch in pair.invalidation_calls for key in batch}
            == {cached, moved}
        )
    )
    pair.until(lambda: not pair.source._core._invalidation_handler.has_pending())
    assert pair.source.query((cached, ordinary, moved)) == [
        (QueryStatus.MISS, None),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
        (QueryStatus.MISS, None),
    ]
    handles = tuple(result.release_handle for result in results.values())
    assert pair.target.release(handles) == [(handle, True) for handle in handles]


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("advice", ["missing", "discarded", "conflicting", "unmatched"])
def test_explicit_move_requires_valid_source_advice_even_for_local_hit(cached, advice):
    pair = _Pair()
    key = BlockKey(b"explicit-source-required")
    retained = pair.deposit(key).release_handle
    if cached:
        copied = pair.start((key,), mode="copy")
        pair.copy_done()
        result = pair.result(copied)[key]
        pair.target.release((result.release_handle,))
    pair.target.discard_hint("req")
    if advice == "discarded":
        pair.advise()
        pair.target.discard_hint("req")
    elif advice == "conflicting":
        pair.advise()
        pair.advise(source="tcp://different-source:1")
    elif advice == "unmatched":
        pair.advise(hashes=(999,))
    handle = pair.target.move((key,), "req")
    result = dict(pair.target.poll_completed())[handle][key]
    assert not result.success and result.release_handle is None
    assert pair.source.release((retained,)) == [(retained, True)]
    assert not pair.target._core._move_handler._pending
    assert not pair.source._core._invalidation_handler._pending


def test_move_invalidation_waits_for_other_native_transfer_cleanup():
    agent = _IndividualWriteAgent()
    pair = _Pair(source_agent=agent)
    key = BlockKey(b"concurrent-transfer")
    pair.deposit(key, no_evict=False)
    before = len(agent.xfers)
    delivery = pair.start(
        (key,),
        mode="copy",
        destinations={key: pair.descriptors(endpoint="target")},
    )
    pair.until(lambda: len(agent.xfers) == before + 1)
    move = pair.start((key,))
    pair.until(lambda: len(agent.xfers) == before + 2)
    agent.states[before + 2] = "DONE"

    moved = pair.result(move)[key]
    pair.until(lambda: bool(pair.invalidation_calls))

    assert moved.success
    assert delivery not in pair.results[1]
    assert pair.source._core._block_record_map[key].local_dram.source_readers == 1
    assert pair.source._core._invalidation_handler._pending
    assert pair.source.query((key,)) == [(QueryStatus.MISS, None)]
    assert pair.source_local[""].raw == b"a" * 16
    agent.states[before + 1] = "DONE"
    assert pair.result(delivery)[key].success
    pair.until(lambda: not pair.source._core._invalidation_handler.has_pending())
    assert pair.source.query((key,)) == [(QueryStatus.MISS, None)]
    assert pair.target.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert pair.target_local[""].raw == b"a" * 16
    assert pair.target_primary[""].raw[:16] == b"a" * 16
    assert pair.target.release((moved.release_handle,)) == [
        (moved.release_handle, True)
    ]
    _assert_no_invalidation_messages(pair)


def test_mixed_move_reports_missing_source_against_original_key_indices():
    pair = _Pair(slots=3)
    cached, missing, physical = map(BlockKey, (b"cached", b"missing", b"physical"))
    pair.source_primary[""].raw = b"a" * 16 + b"b" * 16 + b"c" * 16 + b"d" * 16
    pair.target_primary[""].raw = pair.source_primary[""].raw
    pair.deposit(cached, no_evict=False)
    pair.deposit(physical, index=2, no_evict=False)
    pair.source_pinning.prefix_length = 0
    _cache_at_target(pair, cached)
    handle = pair.start((cached, missing, physical), explicit_move=True)
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
    assert len(requests) == 1 and requests[0]["keys"] == [missing, physical, cached]
    pair.until(
        lambda: (
            {key for batch in pair.invalidation_calls for key in batch}
            == {cached, physical}
        )
    )
    pair.until(lambda: not pair.source._core._invalidation_handler.has_pending())
    assert pair.source.query((cached, physical)) == [(QueryStatus.MISS, None)] * 2
    handles = (results[cached].release_handle, results[physical].release_handle)
    assert pair.target.release(handles) == [(handle, True) for handle in handles]
