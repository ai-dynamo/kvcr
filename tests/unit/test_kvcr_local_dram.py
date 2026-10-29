# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KVCR local-DRAM, capacity, and policy tests."""

import ctypes
import gc
import hashlib
import heapq
import logging
from contextlib import closing, suppress
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _buffer_region,
    _key_history,
    _mem_descriptor,
    _new_kvcr,
    _new_local_kvcr,
    _op_entries,
    _poll_until,
    _RecordingFIFOPolicy,
    _recovered_record,
    _wait_until,
)

from kvcr import core as core_module
from kvcr import progress as progress_module
from kvcr.config import KVCRConfig, LocalDramOptions
from kvcr.core import _BlockRecord
from kvcr.local_dram import (
    _layout_indices,
    _LocalCopyOp,
    _LocalDramResidency,
    _LocalDramState,
)
from kvcr.policy import FIFOPolicy, LRUPolicy
from kvcr.policy_runtime import _EvictionQueue, _PolicyInvoker
from kvcr.recovery_journal import (
    RecoveryMirrorError,
    _attach_journal,
    _RecoveryMirror,
    install_recovery_records,
)
from kvcr.types import (
    BlockKey,
    BlockMeta,
    CacheTier,
    InventoryEvent,
    OpEntryStatus,
    PlacementAction,
    QueryStatus,
    RegionDescriptor,
)


def test_expired_copy_logs_without_submitting(caplog):
    caplog.set_level(logging.DEBUG, logger="kvcr.local_dram")
    copy = SimpleNamespace(transfer_id=None, clock=lambda: 1, deadline=1, op_id=7)
    progress = Mock()
    assert _LocalCopyOp.progress(copy, progress, None) == (True, True)
    progress.submit_transfer.assert_not_called()
    assert "local_copy_expired_before_submit op=7" in caplog.text


def _two_pool_kvcr(
    agent, pools, source, config=None, capacity_needed_callback=None, *, labeled=False
):
    config = config or KVCRConfig(
        nixl_agent_name="target", pool_layouts=[("full", 8), ("swa", 8)]
    )
    return _new_kvcr(
        agent,
        FakePrimaryPinning(),
        FakeBytesControl(),
        config,
        local_dram=LocalDramOptions(
            [
                (name, ctypes.addressof(pool), len(pool))
                for (name, _), pool in zip(config.pool_layouts, pools, strict=True)
            ]
        ),
        capacity_needed_callback=capacity_needed_callback,
        framework_regions=[
            RegionDescriptor(
                addr=ctypes.addressof(source),
                size=config.pool_layouts[0][1],
                label="full",
            ),
            RegionDescriptor(
                addr=ctypes.addressof(source) + config.pool_layouts[0][1],
                size=config.pool_layouts[1][1],
                label="swa:*" if labeled else "swa",
                count=(len(source) - config.pool_layouts[0][1])
                // config.pool_layouts[1][1],
            ),
        ],
    )


@pytest.mark.parametrize("history", [False, True])
def test_local_deposit_deduplicates_and_evicts_fifo(monkeypatch, caplog, history):
    monkeypatch.setenv("KVCR_KEY_HISTORY", "1" if history else "0")
    caplog.set_level(logging.DEBUG, logger="kvcr.core")
    block_size = 16
    primary = ctypes.create_string_buffer(block_size * 3)
    local = ctypes.create_string_buffer(block_size * 2)
    primary.raw = b"a" * block_size + b"b" * block_size + b"c" * block_size
    events: list[InventoryEvent] = []
    agent = FakeNixlAgent()
    policy = _RecordingFIFOPolicy()
    kvcr = _new_local_kvcr(
        agent,
        local,
        2,
        events.append,
        policy=policy,
        framework_regions=[_buffer_region(primary, len(local) // 2)],
    )
    keys = tuple(BlockKey(f"k{index}".encode()) for index in range(3))
    if not history:

        def forbidden(*args, **kwargs):
            pytest.fail("disabled history performed diagnostic work")

        monkeypatch.setattr(kvcr._core, "_log_key_history", forbidden)
        monkeypatch.setattr(core_module, "hashlib", SimpleNamespace(sha256=forbidden))

    first = kvcr.deposit(
        {key: [_mem_descriptor(index)] for index, key in enumerate(keys[:2])},
        hints={"priority": "test"},
    )
    _wait_until(lambda: len(agent.transfers) == 1)
    assert list(kvcr.poll_completed()) == []
    assert policy.ingested == []

    agent.state = "DONE"
    completed = _poll_until(kvcr, bool)
    assert completed == [(first, _op_entries({keys[0]: True, keys[1]: True}))]
    assert completed[0][1][keys[0]] is completed[0][1][keys[1]]
    assert local.raw == b"a" * block_size + b"b" * block_size
    assert events == [InventoryEvent(keys[:2], CacheTier.LOCAL_G2, False)]
    assert [(meta.block_key, source) for meta, source in policy.ingested] == [
        (key, CacheTier.FW_G2) for key in keys[:2]
    ]
    assert all(
        meta.resident_tiers == frozenset((CacheTier.LOCAL_G2,))
        for meta, *_ in policy.ingested
    )

    ready_duplicate = kvcr.deposit({keys[0]: [_mem_descriptor()]})
    assert list(kvcr.poll_completed()) == [
        (ready_duplicate, _op_entries({keys[0]: True}))
    ]
    assert len(agent.transfers) == 1
    assert len(events) == 1
    assert len(policy.ingested) == 2

    replacement = kvcr.deposit({keys[2]: [_mem_descriptor(2)]})
    completed = _poll_until(kvcr, lambda results: bool(results))
    assert completed == [(replacement, _op_entries({keys[2]: True}))]
    assert [meta.block_key for meta, _ in policy.scored] == list(keys)
    assert all(source is CacheTier.LOCAL_G2 for _, source in policy.scored)
    assert [meta.block_key for meta, _ in policy.decided] == [keys[0]]
    assert [meta.block_key for meta, *_ in policy.ingested] == list(keys)
    assert [meta.block_key for meta in policy.removed] == [keys[0]]
    assert CacheTier.LOCAL_G2 not in policy.removed[0].resident_tiers
    assert local.raw == b"c" * block_size + b"b" * block_size
    assert events[1:] == [
        InventoryEvent((keys[0],), CacheTier.LOCAL_G2, True),
        InventoryEvent((keys[2],), CacheTier.LOCAL_G2, False),
    ]
    records = _key_history(caplog)
    assert bool(records) is history
    if history:
        assert [r["sequence"] for r in records] == [
            str(i) for i in range(1, len(records) + 1)
        ]
        removal = next(r for r in records if r["event"] == "remove")
        assert removal["reason"] == "capacity_eviction"
        assert removal["key_sha256"] == hashlib.sha256(keys[0]).hexdigest()
        assert {r["event"] for r in records} >= {"ready", "inventory_accepted"}


def test_parallel_deposits_preserve_committed_pieces_and_claims(monkeypatch) -> None:
    pools = [ctypes.create_string_buffer(8), ctypes.create_string_buffer(24)]
    source = ctypes.create_string_buffer(
        b"_" * 8 + b"a" * 8 + b"b" * 8 + b"c" * 8 + b"b" * 8
    )
    agent = FakeNixlAgent()
    states = {}

    def check_state(handle):
        state = states.get(handle, "PROC")
        if state == "ERR":
            states[handle] = "DONE"
        return state

    monkeypatch.setattr(agent, "check_xfer_state", check_state)
    kvcr = _two_pool_kvcr(agent, pools, source, labeled=True)
    journal = Mock()
    journal.publish.return_value = True
    _attach_journal(kvcr._core._local_dram, journal)
    events = []
    kvcr._core._inventory_sink_callback = events.append
    key = BlockKey(b"layered")
    layout = ["swa:a", "swa:b", "swa:c"]
    first = kvcr.deposit({key: [_mem_descriptor(0, "swa:a")]}, expected_layout=layout)
    second = kvcr.deposit({key: [_mem_descriptor(1, "swa:b")]})
    duplicate = kvcr.deposit({key: [_mem_descriptor(3, "swa:b")]}, no_evict=True)
    _wait_until(lambda: len(agent.transfers) == 2)
    assert kvcr.poll_completed() == []
    assert kvcr._core._local_dram.telemetry_state()["local_g2_allocated_slots"] == 3
    assert kvcr.query([key]) == [(QueryStatus.FETCHING, CacheTier.LOCAL_G2)]

    # b finishes while a is still copying; the duplicate reuses b without a DMA.
    states[agent.transfers[1]] = "DONE"
    completed = dict(_poll_until(kvcr, lambda done: len(done) == 2))
    assert completed[second][key].success and completed[duplicate][key].success
    claim = completed[duplicate][key].release_handle
    assert claim is not None
    assert len(agent.transfers) == 2
    assert kvcr.query([key]) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert events == []
    kvcr.align_sequence([key])
    journal.publish.assert_not_called()
    assert kvcr.release([claim]) == [(claim, True)]
    # The earlier writer protects the key even after the public claim is released.
    assert kvcr._core._local_dram.telemetry_state()["local_g2_evictable_slots"] == 0
    duplicate = kvcr.deposit({key: [_mem_descriptor(3, "swa:b")]}, no_evict=True)
    claim = dict(kvcr.poll_completed())[duplicate][key].release_handle
    assert claim is not None

    pieces = [_mem_descriptor(i, f"swa:{part}") for i, part in enumerate("abc")]
    third = kvcr.deposit({key: pieces})
    _wait_until(lambda: len(agent.transfers) == 3)
    assert agent.xfers[2][3][0] == agent.xfers[0][3][0]
    assert len(agent.xfers[2][3]) == 2  # b is already committed; only a and c copy.
    assert kvcr._core._local_dram.telemetry_state()["local_g2_allocated_slots"] == 3
    assert kvcr.poll_completed() == []
    source[32:40] = b"\0" * 8
    delivery = kvcr.deliver({key: [_mem_descriptor(3, "swa:b")]})
    _wait_until(lambda: len(agent.transfers) == 4)
    delivery_handle = agent.transfers[3]
    _wait_until(lambda: source.raw[32:40] == b"b" * 8)

    states[agent.transfers[2]] = "DONE"
    assert dict(_poll_until(kvcr, bool))[third][key].success
    assert events == [InventoryEvent((key,), CacheTier.LOCAL_G2, False)]
    assert journal.publish.call_count == 1
    mirror = _RecoveryMirror(("full", "swa"))
    for call in journal.publish.call_args_list:
        mirror.apply(*call.args)
    recovered = mirror.take_records()[key].local_dram
    assert recovered.slots == tuple(zip(layout, range(3)))
    assert not recovered.pending_labels
    published = journal.publish.call_count
    states[agent.transfers[0]] = "ERR"
    assert not dict(_poll_until(kvcr, bool))[first][key].success
    assert journal.publish.call_count == published
    assert kvcr._core._local_dram.telemetry_state()["local_g2_allocated_slots"] == 3
    # Complete the older delivery after all reserved pieces become ready.
    states[delivery_handle] = "DONE"
    assert dict(_poll_until(kvcr, bool))[delivery][key].success
    source[8:32] = b"\0" * 24
    delivery = kvcr.deliver({key: pieces})
    _wait_until(lambda: len(agent.transfers) == 5)
    states[agent.transfers[4]] = "DONE"
    assert dict(_poll_until(kvcr, bool))[delivery][key].success
    assert source.raw[8:32] == b"a" * 8 + b"b" * 8 + b"c" * 8
    assert kvcr._core._local_dram.telemetry_state()["local_g2_evictable_slots"] == 0
    assert kvcr.release([claim]) == [(claim, True)]
    assert kvcr._core._local_dram.telemetry_state()["local_g2_evictable_slots"] == 3
    for conflicting in (
        list(reversed(layout)),
        ["swa:a", "swa:b"],
        layout + ["swa:d"],
    ):
        rejected = kvcr.deposit({key: pieces[:1]}, expected_layout=conflicting)
        assert not dict(kvcr.poll_completed())[rejected][key].success
    rejected = kvcr.deposit({key: [_mem_descriptor(0, "swa:unknown")]})
    assert not dict(kvcr.poll_completed())[rejected][key].success
    assert len(agent.transfers) == 5


@pytest.mark.parametrize("timeout", [False, True])
def test_failed_labelled_deposit_keeps_previous_parts(monkeypatch, timeout) -> None:
    pools = [ctypes.create_string_buffer(8), ctypes.create_string_buffer(16)]
    source = ctypes.create_string_buffer(b"_" * 8 + b"a" * 8 + b"b" * 8)
    agent = FakeNixlAgent()
    agent.state = "DONE"
    kvcr = _two_pool_kvcr(agent, pools, source, labeled=True)
    key = BlockKey(b"layered")
    first = kvcr.deposit(
        {key: [_mem_descriptor(0, "swa:a")]}, expected_layout=["swa:a", "swa:b"]
    )
    assert dict(_poll_until(kvcr, bool))[first][key].success
    now = 0.0
    kvcr._core._clock = lambda: now
    with monkeypatch.context() as patch:
        if timeout:
            patch.setattr(agent, "check_xfer_state", lambda handle: "PEND")
        else:
            patch.setattr(
                agent,
                "make_prepped_xfer",
                Mock(side_effect=RuntimeError("submission failed")),
            )
        failed = kvcr.deposit({key: [_mem_descriptor(1, "swa:b")]})
        if timeout:
            _wait_until(lambda: len(agent.transfers) == 2)
            now = 5.0
        assert not dict(_poll_until(kvcr, bool))[failed][key].success
        if timeout:
            assert agent.transfers[-1] not in agent.released_xfers
            assert kvcr._core._local_dram.telemetry_state()["local_g2_free_slots"] == 1
    agent.state = "DONE"
    if timeout:
        assert _poll_until(kvcr, lambda _: len(agent.released_xfers) == 2) == []
    assert kvcr._core._local_dram.telemetry_state()["local_g2_evictable_slots"] == 0
    delivery = kvcr.deliver({key: [_mem_descriptor(1, "swa:a")]})
    assert dict(_poll_until(kvcr, bool))[delivery][key].success
    assert source.raw[16:24] == b"a" * 8
    source[16:24] = b"b" * 8
    retry = kvcr.deposit({key: [_mem_descriptor(1, "swa:b")]})
    assert dict(_poll_until(kvcr, bool))[retry][key].success
    assert pools[1].raw == b"a" * 8 + b"b" * 8


@pytest.mark.parametrize("case", ["pending", "retry"])
def test_local_reads_wait_for_later_piece_and_release_claims(monkeypatch, case):
    pools = [ctypes.create_string_buffer(8), ctypes.create_string_buffer(16)]
    source = ctypes.create_string_buffer(b"_" * 8 + b"a" * 8 + b"b" * 8)
    agent = FakeNixlAgent()
    agent.state = "DONE"
    kvcr = _two_pool_kvcr(agent, pools, source, labeled=True)
    now = 0.0
    kvcr._core._clock = lambda: now
    key = BlockKey(b"partial")
    pieces = [_mem_descriptor(0, "swa:a"), _mem_descriptor(1, "swa:b")]
    first = kvcr.deposit({key: pieces[:1]}, expected_layout=["swa:a", "swa:b"])
    assert dict(_poll_until(kvcr, bool))[first][key].success
    initial_transfers = len(agent.transfers)
    assert kvcr._core._local_dram.telemetry_state()["local_g2_evictable_slots"] == 0
    assert kvcr.query([key]) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    other = BlockKey(b"other")
    blocked = kvcr.deposit({other: pieces})
    assert not dict(_poll_until(kvcr, bool))[blocked][other].success
    agent.state = "PROC"
    if case == "pending":
        deposit = kvcr.deposit({key: [_mem_descriptor(1, "swa:b")]})
        _wait_until(lambda: len(agent.transfers) == initial_transfers + 1)
    delivery = kvcr.deliver({key: [_mem_descriptor(0, "swa:b")]})
    assert kvcr.poll_completed() == []
    assert len(agent.transfers) == initial_transfers + int(case == "pending")
    assert source.raw[8:16] == b"a" * 8
    assert kvcr._core._local_dram.telemetry_state()["local_g2_evictable_slots"] == 0
    fetch = kvcr.fetch([key], expected_layout=["swa:a", "swa:b"])
    assert kvcr.poll_completed() == []
    now = 0.75
    try:
        if case == "retry":
            with monkeypatch.context() as patch:
                patch.setattr(
                    agent, "make_prepped_xfer", Mock(side_effect=RuntimeError)
                )
                failed = kvcr.deposit({key: pieces[1:]})
                completed = dict(_poll_until(kvcr, bool))
                assert set(completed) == {failed}
                assert not completed[failed][key].success
        if case != "pending":
            deposit = kvcr.deposit({key: [_mem_descriptor(1, "swa:b")]})
    finally:
        agent.state = "DONE"
    completed = dict(_poll_until(kvcr, lambda done: len(done) == 3))
    assert completed[deposit][key].success and completed[delivery][key].success
    fetched = completed[fetch][key]
    assert fetched.success
    assert fetched.descriptors == [
        _mem_descriptor(0, "swa:a"),
        _mem_descriptor(1, "swa:b"),
    ]
    assert source.raw[8:16] == b"b" * 8
    assert kvcr._core._local_dram.telemetry_state()["local_g2_evictable_slots"] == 0
    assert kvcr.release([fetched.release_handle]) == [(fetched.release_handle, True)]
    assert kvcr._core._local_dram.telemetry_state()["local_g2_evictable_slots"] == 2
    replacement = kvcr.deposit({other: pieces})
    assert dict(_poll_until(kvcr, bool))[replacement][other].success
    assert kvcr.query([key]) == [(QueryStatus.MISS, None)]


def test_local_dram_rejects_overlapping_pools() -> None:
    memory = ctypes.create_string_buffer(16)
    address = ctypes.addressof(memory)

    with pytest.raises(ValueError, match="overlap"):
        _new_kvcr(
            FakeNixlAgent(),
            FakePrimaryPinning(),
            FakeBytesControl(),
            KVCRConfig(
                nixl_agent_name="target",
                pool_layouts=[("full", 8), ("swa", 8)],
            ),
            local_dram=LocalDramOptions(
                [("full", address, 16), ("swa", address + 8, 8)]
            ),
        )


@pytest.mark.parametrize("labeled", [False, True])
def test_multi_pool_residency_moves_and_evicts_as_one_key(labeled, monkeypatch) -> None:
    full = ctypes.create_string_buffer(16)
    swa = ctypes.create_string_buffer(16)
    payload = b"a" * 16 + b"b" * 8 + b"c" * 8
    source = ctypes.create_string_buffer(payload, 32)
    agent = FakeNixlAgent()
    kvcr = _two_pool_kvcr(
        agent,
        (full, swa),
        source,
        KVCRConfig(
            nixl_agent_name="target",
            pool_layouts=[("full", 16), ("swa", 8)],
        ),
        labeled=labeled,
    )
    layout = ["full", "swa:a", "swa:b"] if labeled else ["full", "swa", "swa"]
    descriptors = [
        _mem_descriptor(index, label) for index, label in zip((0, 0, 1), layout)
    ]
    first, second = BlockKey(b"first"), BlockKey(b"second")

    with pytest.raises(ValueError, match="framework registration"):
        kvcr.deposit({first: [_mem_descriptor(label="unknown")]})
    operation = kvcr.deposit({first: descriptors})
    _wait_until(lambda: bool(agent.transfers))
    local_dram = kvcr._core._local_dram
    residency = kvcr._core._block_record_map[first].local_dram
    assert local_dram._size_bytes(residency) == 32
    pool_counts = residency.pool_counts
    assert dict(pool_counts) == {"full": 1, "swa": 2}
    duplicate = kvcr.deposit({first: descriptors})
    assert kvcr._core._local_dram.telemetry_state()["local_g2_allocated_slots"] == 3
    invalid_layout = ["swa:missing"] if labeled else layout[1:]
    wrong_layout = kvcr.fetch((first,), expected_layout=invalid_layout)
    early = dict(kvcr.poll_completed())
    agent.state = "DONE"
    assert set(early) == {wrong_layout}
    assert not early[wrong_layout][first].success
    done = dict(_poll_until(kvcr, lambda done: len(done) == 2))
    assert done[operation][first].success and done[duplicate][first].success
    assert len(agent.transfers) == 1
    assert full.raw + swa.raw == payload
    # A ready block keeps nothing for the cyclic GC to scan.
    assert residency.pending_labels is None and residency.writers is None
    assert not gc.is_tracked(residency)
    assert not gc.is_tracked(kvcr._core._block_record_map[first])
    duplicate = kvcr.deposit({first: descriptors[1:] + descriptors[:1]})
    assert dict(kvcr.poll_completed())[duplicate][first].success is labeled
    assert len(agent.transfers) == 1
    wrong_layout = kvcr.fetch((first,), expected_layout=invalid_layout)
    assert not dict(kvcr.poll_completed())[wrong_layout][first].success
    claim = kvcr.fetch((first,), expected_layout=layout)
    result = dict(_poll_until(kvcr, lambda done: claim in dict(done)))[claim][first]
    assert result.success and result.release_handle is not None
    assert result.descriptors == descriptors
    now = 0.0
    kvcr._core._clock = lambda: now
    wrong = kvcr.deliver(
        {first: [_mem_descriptor(label="swa:missing" if labeled else "swa")]}
    )
    assert not dict(kvcr.poll_completed())[wrong][first].success

    assert residency.pool_counts is pool_counts
    # Ready blocks already have their label index, but references stay incremental.
    assert residency.indices_by_label == (
        {"full": 0, "swa:a": 1, "swa:b": 2} if labeled else {"full": 0, "swa": None}
    )
    assert not residency.refs
    make_ref = Mock(wraps=local_dram._descriptor)
    monkeypatch.setattr(local_dram, "_descriptor", make_ref)
    agent.state = "PROC"
    partial = kvcr.deliver(
        {
            first: [
                _mem_descriptor(i, label)
                for i, label in enumerate(reversed(layout[1:]))
            ]
        }
    )
    if labeled:
        _wait_until(lambda: len(agent.transfers) == 2)
        assert make_ref.call_count == 2  # Only the requested parts are constructed.
        # Releasing the fetch cannot make any part evictable during a partial copy.
        assert kvcr._core._block_record_map[first].local_dram.claim_count == 2
        kvcr.release([result.release_handle])
        blocked = kvcr.deposit({second: descriptors})
        assert dict(kvcr.poll_completed())[blocked][second].success is False
        agent.state = "DONE"
        assert dict(_poll_until(kvcr, bool))[partial][first].success
        assert source.raw == b"a" * 16 + b"c" * 8 + b"b" * 8
    else:
        agent.state = "DONE"
        assert dict(_poll_until(kvcr, bool))[partial][first].success
        assert source.raw == payload  # Repeated pools retain their original order.
        kvcr.release([result.release_handle])
    assert kvcr._core._local_dram.telemetry_state()["local_g2_evictable_slots"] == 3

    source.raw = b"\x00" * len(source)
    matching = kvcr.deliver({first: descriptors})
    assert dict(_poll_until(kvcr, bool))[matching][first].success
    assert source.raw == payload
    assert make_ref.call_count == 3  # The full copy reuses those earlier parts.
    assert not gc.is_tracked(residency.refs)
    operation = kvcr.deposit({second: descriptors})
    _poll_until(kvcr, lambda done: operation in dict(done))
    assert kvcr.query((first, second)) == [
        (QueryStatus.MISS, None),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]


@pytest.mark.parametrize(
    "requested,expected",
    [
        (["", "swa:a", "swa", "swa"], [0, 1, 2, 3]),
        (["swa:a", ""], [1, 0]),
        (["swa", "swa"], [2, 3]),
        (["swa", "swa", ""], None),
        (["swa"], None),
        (["missing"], None),
        (["", ""], None),
        ([], None),
    ],
)
def test_layout_matching_with_and_without_index(requested, expected):
    for index in (None, {"": 0, "swa:a": 1, "swa": None}):
        assert (
            _layout_indices(
                ("", "swa:a", "swa", "swa"),
                requested,
                indices_by_label=index,
            )
            == expected
        )


def test_failed_group_reservation_does_not_evict_a_partial_group() -> None:
    pools = [ctypes.create_string_buffer(8), ctypes.create_string_buffer(8)]
    source = ctypes.create_string_buffer(16)
    agent = FakeNixlAgent()
    kvcr = _two_pool_kvcr(agent, pools, source)
    full, swa, grouped = (BlockKey(name) for name in (b"full", b"swa", b"grouped"))
    descriptors = [
        _mem_descriptor(label="full"),
        _mem_descriptor(0, label="swa"),
    ]

    kvcr.deposit({full: [descriptors[0]], swa: [descriptors[1]]})
    # Both batches reject full; only swa should be claimed after the fill.
    fetch = kvcr.fetch((full, swa), expected_layout=["swa"])
    duplicate = kvcr.deposit({full: [descriptors[1]], swa: [descriptors[1]]})
    assert list(kvcr.poll_completed()) == []
    agent.state = "DONE"
    completed = dict(_poll_until(kvcr, lambda results: len(results) == 3))
    assert completed[duplicate] == _op_entries({full: False, swa: True})
    assert completed[fetch][full].status is OpEntryStatus.FAILED
    claim = completed[fetch][swa]
    assert claim.success

    transfers = len(agent.transfers)
    operation = kvcr.deposit(
        {grouped: descriptors[:1]}, expected_layout=["full", "swa"]
    )
    result = dict(_poll_until(kvcr, lambda done: operation in dict(done)))[operation]
    assert result[grouped].status is OpEntryStatus.FAILED
    assert len(agent.transfers) == transfers
    assert kvcr.query((full, swa)) == [
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]
    retry = kvcr.deposit({grouped: [descriptors[0]]})
    assert dict(_poll_until(kvcr, lambda done: retry in dict(done)))[retry][
        grouped
    ].success
    kvcr.release([claim.release_handle])


def test_group_allocation_evicts_enough_whole_keys(monkeypatch) -> None:
    full = tuple(BlockKey(f"full{index}".encode()) for index in range(32))
    pools = [
        ctypes.create_string_buffer((len(full) + 1) * 8),
        ctypes.create_string_buffer(16),
    ]
    source = ctypes.create_string_buffer(24)
    agent = FakeNixlAgent()
    agent.state = "DONE"
    kvcr = _two_pool_kvcr(agent, pools, source, labeled=True)
    swa0, swa1, grouped = (BlockKey(name) for name in (b"swa0", b"swa1", b"grouped"))
    descriptors = [
        _mem_descriptor(label="full"),
        _mem_descriptor(0, label="swa:a"),
        _mem_descriptor(1, label="swa:b"),
    ]

    operation = kvcr.deposit(
        {
            **{key: [descriptors[0]] for key in full},
            swa0: [descriptors[1]],
            swa1: [descriptors[2]],
        }
    )
    _poll_until(kvcr, lambda done: operation in dict(done))
    pops = Mock(wraps=heapq.heappop)
    monkeypatch.setattr(heapq, "heappop", pops)
    operation = kvcr.deposit({grouped: descriptors})
    result = dict(_poll_until(kvcr, lambda done: operation in dict(done)))[operation]

    assert result[grouped].success
    # The older full-only rows must be scanned once while making room in swa.
    assert pops.call_count <= len(full) + 2
    assert kvcr.query(full) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)] * len(full)
    assert kvcr.query((swa0, swa1, grouped)) == [
        (QueryStatus.MISS, None),
        (QueryStatus.MISS, None),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]
    assert kvcr._core._local_dram.telemetry_state()["local_g2_evictable_slots"] == (
        len(full) + 3
    )
    retry = kvcr.deposit({swa0: [descriptors[0]]})
    assert dict(_poll_until(kvcr, lambda done: retry in dict(done)))[retry][
        swa0
    ].success
    assert kvcr.query(full[:2]) == [
        (QueryStatus.MISS, None),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]


def test_eviction_heap_compaction_preserves_active_candidates() -> None:
    queue = _EvictionQueue()
    first, churned, excluded = (
        BlockKey(name) for name in (b"first", b"churned", b"excluded")
    )
    queue.insert(first, 0)
    queue.insert(churned, 1)
    queue.insert(excluded, 1)
    assert all(queue._live[item[2]] is item for item in queue._heap)

    protected = frozenset((excluded,))
    with closing(queue.candidates(protected)) as candidates:
        assert next(candidates) == first
        for _ in range(100):
            queue.remove(churned)
            queue.insert(churned, 1)
        assert len(queue._heap) <= 2 * len(queue)
        assert next(candidates) == churned
        assert not queue.insert(churned, 2)
        assert list(candidates) == []

    assert queue.score(churned) == 2
    assert len(queue._heap) == 3
    assert list(queue.candidates(frozenset())) == [first, excluded, churned]


def test_lru_float_score_uses_small_tail_first_bias() -> None:
    policy = _PolicyInvoker(LRUPolicy())

    def score(timestamp, position):
        return policy.eviction_score(
            BlockMeta(BlockKey(b"key"), 16, 0, timestamp, frozenset(), position),
            CacheTier.LOCAL_G2,
        )

    # Position must remain visible at a realistic clock value.
    timestamp = 1_000_000.0
    assert isinstance(score(timestamp, 0), float)
    assert score(timestamp, -1) == score(timestamp, 0)
    assert score(timestamp, 2) < score(timestamp, 1) < score(timestamp, 0)
    assert score(timestamp, 0) < score(timestamp + 0.001, 4096)
    assert score(None, 1) < score(None, 0) < score(0, 4096)


@pytest.mark.parametrize(
    ("policy", "use_current_time", "keep_claim", "evicted_index"),
    # use_current_time: None skips alignment; False uses newest access; True uses now.
    [
        (FIFOPolicy(), None, False, 1),
        (LRUPolicy(), None, False, 0),
        (None, None, False, 0),
        (None, False, False, 1),
        (None, True, False, 2),
        (None, False, True, 0),
        (FIFOPolicy(), False, False, 1),
    ],
    ids=[
        "fifo",
        "lru",
        "default-lru",
        "aligned",
        "current-time",
        "claimed",
        "aligned-fifo",
    ],
)
def test_builtin_policy_eviction_order(
    policy: FIFOPolicy | None,
    use_current_time: bool | None,
    keep_claim: bool,
    evicted_index: int,
) -> None:
    block_size = 16
    primary = ctypes.create_string_buffer(block_size * 4)
    local = ctypes.create_string_buffer(block_size * 3)
    agent = FakeNixlAgent()
    agent.state = "DONE"
    kvcr = _new_local_kvcr(
        agent,
        local,
        3,
        policy=policy,
        framework_regions=[_buffer_region(primary, len(local) // 3)],
    )
    now = 0.0
    kvcr._core._clock = lambda: now
    keys = [BlockKey(f"k{index}".encode()) for index in range(4)]

    kvcr.deposit({key: [_mem_descriptor(index)] for index, key in enumerate(keys[:2])})
    _poll_until(kvcr, lambda results: bool(results))

    now = 1.0
    first_fetch = kvcr.fetch((keys[0],))
    first_claim = dict(kvcr.poll_completed())[first_fetch][keys[0]].release_handle
    now = 2.0
    second_fetch = kvcr.fetch((keys[1],))
    second_claim = dict(kvcr.poll_completed())[second_fetch][keys[1]].release_handle
    assert first_claim is not None and second_claim is not None
    kvcr.release((first_claim,) if keep_claim else (second_claim, first_claim))

    now = 3.0
    kvcr.deposit({keys[2]: [_mem_descriptor(2)]})
    _poll_until(kvcr, bool)
    now = 4.0
    if use_current_time is not None:
        kvcr.align_sequence(keys[:2], use_current_time=use_current_time)
    kvcr.deposit({keys[3]: [_mem_descriptor(3)]})
    _poll_until(kvcr, bool)

    statuses = kvcr.query(keys)
    assert statuses.pop(evicted_index) == (QueryStatus.MISS, None)
    assert statuses == [(QueryStatus.HIT, CacheTier.LOCAL_G2)] * 3
    if keep_claim:
        kvcr.release((second_claim,))


@pytest.mark.parametrize(
    ("access_first", "use_current_time", "expected_time"),
    [(False, False, None), (True, False, 3.0), (False, True, 5.0), (True, True, 5.0)],
)
def test_align_sequence_updates_policy_and_journal(
    access_first: bool, use_current_time: bool, expected_time: float | None
) -> None:
    events = []

    class RecordingLRUPolicy(LRUPolicy):
        def on_align_sequence(self, blocks, use_current_time):
            events.append(("align", tuple(blocks), use_current_time))

        def eviction_score(self, meta, source):
            events.append(("score", meta, source))
            return super().eviction_score(meta, source)

    primary = ctypes.create_string_buffer(16)
    local = ctypes.create_string_buffer(48)
    agent = FakeNixlAgent()
    kvcr = _new_local_kvcr(
        agent,
        local,
        3,
        policy=RecordingLRUPolicy(),
        framework_regions=[_buffer_region(primary, len(local) // 3)],
    )
    records = _g2_recovered(first=0, last=1)
    first, last = records
    records[first].last_access = 3.0 if access_first else None
    records[first].access_count = int(access_first)
    install_recovery_records(kvcr._core, records)
    missing = BlockKey(b"missing")
    filling = BlockKey(b"filling")
    kvcr.deposit({filling: [_mem_descriptor()]})
    _wait_until(lambda: bool(agent.transfers))
    filling_record = kvcr._core._block_record_map[filling]
    assert filling_record.local_dram.state is _LocalDramState.FILLING
    events.clear()
    journal = Mock()
    journal.publish.return_value = True
    _attach_journal(kvcr._core._local_dram, journal)

    kvcr._core._clock = lambda: 5.0
    kvcr.align_sequence([])
    kvcr.align_sequence([missing, filling])
    ordered_keys = [missing, first, first, last, filling]
    kvcr.align_sequence(ordered_keys, use_current_time=use_current_time)

    assert [event[0] for event in events] == ["align", "score", "score"]
    _, blocks, callback_flag = events[0]
    assert callback_flag is use_current_time
    assert [
        (meta.block_key, meta.position, meta.last_access, meta.access_count)
        for meta in blocks
    ] == [
        (first, 1, expected_time, int(access_first)),
        (last, 3, expected_time, 0),
    ]
    assert set(kvcr._core._block_record_map) == {first, last, filling}
    assert filling_record.position == -1
    assert filling_record.last_access is None
    assert kvcr._core._local_dram.telemetry_state()["local_g2_evictable_slots"] == 2
    assert {event[1].block_key: event[1] for event in events[1:]} == {
        meta.block_key: meta for meta in blocks
    }
    mirror = _RecoveryMirror(("",))
    for call in journal.publish.call_args_list:
        mirror.apply(*call.args)
    assert {key: record.position for key, record in mirror.take_records().items()} == {
        first: 1,
        last: 3,
    }
    # Timestamp-only updates do not need another recovery record.
    kvcr._core._clock = lambda: 6.0
    kvcr.align_sequence(ordered_keys, use_current_time=True)
    assert journal.publish.call_count == 2
    agent.state = "DONE"


def test_local_deposit_applies_optional_admission() -> None:
    block_size = 16
    primary = ctypes.create_string_buffer(block_size * 2)
    local = ctypes.create_string_buffer(block_size * 2)
    keys = (BlockKey(b"drop"), BlockKey(b"policy-error"))
    policy = FIFOPolicy()
    policy.decide_ingest = Mock(
        side_effect=[
            (PlacementAction.DROP, None),
            RuntimeError("policy failed"),
        ]
    )
    agent = FakeNixlAgent()
    kvcr = _new_local_kvcr(
        agent,
        local,
        2,
        policy=policy,
        framework_regions=[_buffer_region(primary, len(local) // 2)],
    )

    op_handle = kvcr.deposit(
        {key: [_mem_descriptor(index)] for index, key in enumerate(keys)},
        hints={"source": "framework"},
    )
    _wait_until(lambda: bool(agent.transfers))
    agent.state = "DONE"

    results = dict(_poll_until(kvcr, lambda results: bool(results)))[op_handle]
    assert results[keys[0]].status is OpEntryStatus.DROPPED
    assert results[keys[1]].success
    assert kvcr.query(keys) == [
        (QueryStatus.MISS, None),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]
    assert [
        (args[0].block_key, *args[1:])
        for args, _ in policy.decide_ingest.call_args_list
    ] == [(key, CacheTier.FW_G2, False, None, {"source": "framework"}) for key in keys]
    assert all(
        not args[0].resident_tiers for args, _ in policy.decide_ingest.call_args_list
    )


def test_policy_lifecycle_hook_failures_are_logged(caplog) -> None:
    block_size = 16
    primary = ctypes.create_string_buffer(block_size * 2)
    local = ctypes.create_string_buffer(block_size)
    agent = FakeNixlAgent()
    agent.state = "DONE"
    policy = FIFOPolicy()
    policy.on_ingest = Mock(side_effect=RuntimeError("ingest hook failed"))
    policy.on_remove = Mock(side_effect=RuntimeError("remove hook failed"))
    policy.on_align_sequence = Mock(side_effect=RuntimeError("alignment hook failed"))
    kvcr = _new_local_kvcr(
        agent,
        local,
        1,
        policy=policy,
        framework_regions=[_buffer_region(primary, len(local) // 1)],
    )
    keys = (BlockKey(b"k0"), BlockKey(b"k1"))

    with caplog.at_level(logging.WARNING, logger="kvcr.policy_runtime"):
        for index, key in enumerate(keys):
            op_handle = kvcr.deposit({key: [_mem_descriptor(index)]})
            assert _poll_until(kvcr, lambda results: bool(results)) == [
                (op_handle, _op_entries({key: True}))
            ]
            kvcr.align_sequence([key])

    assert policy.on_ingest.call_count == 2
    policy.on_remove.assert_called_once()
    warnings = [record.getMessage() for record in caplog.records]
    assert warnings.count("KVCR on_ingest failed") == 2
    assert warnings.count("KVCR on_remove failed") == 1
    assert warnings.count("KVCR on_align_sequence failed") == 2


def test_local_claims_fetch_deliver_release_and_capacity() -> None:
    block_size = 16
    primary = ctypes.create_string_buffer(block_size * 2)
    local = ctypes.create_string_buffer(block_size)
    destination = primary
    primary.raw = b"a" * block_size + b"b" * block_size

    agent = FakeNixlAgent()
    policy = _RecordingFIFOPolicy()
    capacity_requests: list[list[tuple[str, int]]] = []
    kvcr = _new_local_kvcr(
        agent,
        local,
        1,
        capacity_low_watermark_percent=100,
        capacity_needed_callback=capacity_requests.append,
        policy=policy,
        local_dram_backend="LOCAL",
        framework_regions=[
            _buffer_region(primary, len(local) // 1),
        ],
    )
    now = 0.0
    kvcr._core._clock = lambda: now
    first_key, second_key = BlockKey(b"k0"), BlockKey(b"k1")

    deposit = kvcr.deposit({first_key: [_mem_descriptor()]}, no_evict=True)
    _wait_until(lambda: bool(agent.transfers))
    assert capacity_requests == [[("", 1)]]
    with pytest.raises(ValueError, match="expected layout"):
        kvcr.fetch((first_key, second_key), expected_layout=["unknown"])
    fetch = kvcr.fetch((first_key,), expected_layout=[""])
    assert kvcr.query((first_key,)) == [(QueryStatus.FETCHING, CacheTier.LOCAL_G2)]
    assert list(kvcr.poll_completed()) == []

    agent.state = "DONE"
    completed = dict(_poll_until(kvcr, lambda results: len(results) == 2))
    deposit_result = completed[deposit][first_key]
    assert deposit_result.success
    assert deposit_result.descriptors is None
    deposit_claim = deposit_result.release_handle
    assert deposit_claim is not None

    assert kvcr.query((first_key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    fetch_result = completed[fetch][first_key]
    assert fetch_result.descriptors == [_mem_descriptor()]
    fetch_claim = fetch_result.release_handle
    assert fetch_claim is not None
    assert policy.scored == []

    ready_fetch = kvcr.fetch((first_key,))
    ready_fetch_result = dict(kvcr.poll_completed())[ready_fetch][first_key]
    assert ready_fetch_result.descriptors == [_mem_descriptor()]
    ready_fetch_claim = ready_fetch_result.release_handle
    assert ready_fetch_claim is not None

    ctypes.memset(ctypes.addressof(destination), 0, block_size)
    agent.state = "PROC"
    deliver = kvcr.deliver(
        {
            first_key: [_mem_descriptor()],
            second_key: [_mem_descriptor(1)],
        }
    )
    _wait_until(lambda: len(agent.transfers) == 2)
    assert kvcr.release((deposit_claim, fetch_claim, ready_fetch_claim)) == [
        (deposit_claim, True),
        (fetch_claim, True),
        (ready_fetch_claim, True),
    ]
    assert kvcr.release((fetch_claim,)) == [(fetch_claim, False)]

    blocked = kvcr.deposit({second_key: [_mem_descriptor(1)]})
    assert list(kvcr.poll_completed()) == [(blocked, _op_entries({second_key: False}))]

    agent.state = "DONE"
    now = 0.25
    assert _poll_until(kvcr, lambda results: bool(results)) == [
        (deliver, _op_entries({first_key: True, second_key: False}))
    ]
    assert destination.raw[:block_size] == b"a" * block_size
    assert agent.xfer_backends and all(
        backends == ["LOCAL"] for backends in agent.xfer_backends
    )
    assert [
        (meta.access_count, meta.last_access)
        for meta, _ in policy.scored
        if meta.block_key == first_key
    ] == [(3, 0.25)]
    assert [meta.block_key for meta, *_ in policy.ingested] == [first_key]
    assert policy.removed == []

    replacement = kvcr.deposit({second_key: [_mem_descriptor(1)]})
    assert capacity_requests == [[("", 1)], [("", 1)]]
    assert _poll_until(kvcr, lambda results: bool(results)) == [
        (replacement, _op_entries({second_key: True}))
    ]
    assert local.raw == b"b" * block_size


def test_capacity_pressure_is_pool_local() -> None:
    pools = [ctypes.create_string_buffer(8), ctypes.create_string_buffer(16)]
    source = ctypes.create_string_buffer(24)
    capacity_requests: list[list[tuple[str, int]]] = []
    agent = FakeNixlAgent()
    agent.state = "DONE"
    kvcr = _two_pool_kvcr(
        agent,
        pools,
        source,
        KVCRConfig(
            nixl_agent_name="target",
            pool_layouts=[("full", 8), ("swa", 8)],
            capacity_low_watermark_percent=100,
        ),
        capacity_needed_callback=capacity_requests.append,
    )
    descriptors = [
        _mem_descriptor(label="full"),
        _mem_descriptor(0, label="swa"),
        _mem_descriptor(1, label="swa"),
    ]

    kvcr._core._update_capacity_pressure({"full": 1, "swa": 2})
    kvcr._core._update_capacity_pressure({"full": 0, "swa": 1})
    kvcr._core._update_capacity_pressure({"full": 0, "swa": 0})
    assert capacity_requests == [[("full", 1), ("swa", 2)]]
    kvcr._core._update_capacity_pressure({"full": 1, "swa": 2})
    kvcr._core._update_capacity_pressure({"full": 0, "swa": 1})
    assert capacity_requests == [[("full", 1), ("swa", 2)]] * 2
    kvcr._core._update_capacity_pressure({"full": 1, "swa": 2})
    capacity_requests.clear()

    full = kvcr.deposit({BlockKey(b"full"): descriptors[:1]}, no_evict=True)
    _poll_until(kvcr, lambda done: full in dict(done))
    assert capacity_requests == [[("full", 1)]]

    swa_key = BlockKey(b"swa")
    swa = kvcr.deposit({swa_key: descriptors[1:]})
    _poll_until(kvcr, lambda done: swa in dict(done))
    assert capacity_requests == [[("full", 1)], [("swa", 2)]]
    capacity_requests.clear()

    claim = kvcr.fetch((swa_key,), expected_layout=["swa", "swa"])
    _poll_until(kvcr, lambda done: claim in dict(done))
    assert capacity_requests == [[("swa", 2)]]


@pytest.mark.parametrize(
    ("kind", "failure", "abandon", "duplicate"),
    [
        ("deposit", "submission", False, False),
        ("deposit", "timeout", False, False),
        ("deposit", "timeout", True, False),
        ("deposit", "sticky_error", True, False),
        ("deliver", "timeout", True, False),
        ("deposit", "timeout", True, True),
        ("deposit", "discard", True, True),
    ],
)
def test_local_copy_timeout_retains_memory_until_completion(
    kind: str, failure: str, abandon: bool, duplicate: bool, caplog, request
) -> None:
    caplog.set_level(logging.DEBUG, logger="kvcr.local_dram")
    states = {}

    class PendingAgent(FakeNixlAgent):
        error_polls = 0

        def transfer(self, handle):
            result = super().transfer(handle)
            return "ERR" if failure == "submission" else result

        def check_xfer_state(self, handle):
            state = states.get(handle, self.state)
            if failure == "sticky_error" and state != "DONE":
                self.error_polls += 1
                raise RuntimeError("cached native error")
            return state

    now = 0.0
    block_size = 16
    primary = ctypes.create_string_buffer(
        b"a" * block_size + b"b" * block_size, block_size * 2
    )
    local = ctypes.create_string_buffer(block_size)
    agent = PendingAgent()
    # Fake-only teardown: real cached NIXL errors cannot transition to DONE.
    request.addfinalizer(lambda: setattr(agent, "state", "DONE"))
    errors = []
    kvcr = _new_local_kvcr(
        agent,
        local,
        1,
        framework_regions=[_buffer_region(primary, block_size)],
        on_resilience_event=errors.append,
    )
    kvcr._core._clock = lambda: now
    key, other = BlockKey(b"k0"), BlockKey(b"other")

    if kind == "deliver":
        agent.state = "DONE"
        deposit = kvcr.deposit({key: [_mem_descriptor()]})
        assert dict(_poll_until(kvcr, bool))[deposit][key].success
    agent.state = "PEND"
    released = list(agent.released_xfers)
    op_handle = getattr(kvcr, kind)({key: [_mem_descriptor()]})
    _wait_until(lambda: len(agent.transfers) > len(released))
    native_handle = agent.transfers[-1]
    expired = (
        kvcr.deposit({key: [_mem_descriptor()]})
        if duplicate and failure == "timeout"
        else None
    )
    if failure in ("timeout", "sticky_error"):
        now = 2.0
    progress = kvcr._core._progress
    copy = next(iter(progress._in_flight_ops.values()))
    if failure != "discard":
        _wait_until(lambda: copy.failure_latched)

    assert list(kvcr.poll_completed()) == (
        [(expired, _op_entries({key: False}))] if expired is not None else []
    )
    residency = kvcr._core._block_record_map[key].local_dram
    assert residency is not None
    assert agent.released_xfers == released
    if duplicate:
        now = 4.5 if failure == "timeout" else 0.0
        second = kvcr.deposit({key: [_mem_descriptor()]})
        assert kvcr.poll_completed() == []
        assert len(agent.transfers) == 1
    if failure == "discard":
        kvcr._core._local_dram.discard_fill([key])
        assert dict(kvcr.poll_completed()) == {
            op_handle: _op_entries({key: False}),
            second: _op_entries({key: False}),
        }
    elif abandon:
        now = 5.0
        assert _poll_until(kvcr, bool) == [(op_handle, _op_entries({key: False}))]
        assert [error.state for error in errors] == ["uncertain"]
        if kind == "deposit":
            assert errors[0].source_blocks == {key: [_mem_descriptor()]}
            assert errors[0].source_owners == {key: "framework"}
        else:
            assert errors[0].destination_regions == [_mem_descriptor()]
            assert errors[0].destination_owner == "framework"
    if abandon:
        assert agent.released_xfers == released
        blocked = kvcr.deposit({other: [_mem_descriptor(1)]})
        assert dict(kvcr.poll_completed())[blocked][other].success is False

    if failure == "sticky_error":
        checked = agent.error_polls
        now = 50.0
        _wait_until(lambda: agent.error_polls >= checked + 2)
        assert kvcr.poll_completed() == []
        assert [error.state for error in errors] == ["uncertain"]
        assert copy.transfer_id in progress._active_transfers
        assert agent.released_xfers == released
        assert key in kvcr._core._block_record_map
        return

    if duplicate and failure == "timeout":
        states[native_handle] = "DONE"
        # The attachment retries only after the failed native writer has drained.
        assert _poll_until(kvcr, lambda _: len(agent.transfers) == 2) == []
        assert native_handle in agent.released_xfers
        residency = kvcr._core._block_record_map[key].local_dram
        assert agent.transfers[1] not in agent.released_xfers
        blocked = kvcr.deposit({other: [_mem_descriptor(1)]})
        assert dict(kvcr.poll_completed())[blocked][other].success is False

    agent.state = "DONE"
    expected = [] if abandon else [(op_handle, _op_entries({key: False}))]
    if duplicate and failure == "timeout":
        expected = [(second, _op_entries({key: True}))]
    assert _poll_until(kvcr, lambda _: residency.claim_count == 0) == expected
    assert [error.state for error in errors] == (
        ["uncertain", "quiesced"] if abandon and failure == "timeout" else []
    )
    if errors:
        assert errors[0].op_handle == errors[1].op_handle
    retained = kind == "deliver" or (duplicate and failure == "timeout")
    assert (key in kvcr._core._block_record_map) is retained
    assert kvcr.query([key]) == [
        (QueryStatus.HIT, CacheTier.LOCAL_G2) if retained else (QueryStatus.MISS, None)
    ]
    if duplicate:
        replacement = kvcr.deposit({other: [_mem_descriptor(1)]})
        assert _poll_until(kvcr, bool) == [(replacement, _op_entries({other: True}))]
        assert agent.xfers[-1][3] == agent.xfers[0][3]
        assert local.raw == b"b" * block_size

    for event, expected_failure in [
        ("submit_failed", "submission"),
        ("deadline_expired", "timeout"),
    ]:
        assert caplog.text.count(f"local_copy_{event}") == (failure == expected_failure)


def test_close_retains_pending_local_copy_resources(monkeypatch) -> None:
    monkeypatch.setattr(progress_module, "_OP_CLEANUP_TIMEOUT_SECONDS", 0.01)
    primary = ctypes.create_string_buffer(16)
    local = ctypes.create_string_buffer(16)
    agent = FakeNixlAgent()
    agent.state = "PEND"
    kvcr = _new_local_kvcr(
        agent, local, 1, framework_regions=[_buffer_region(primary, 16)]
    )
    kvcr._core._clock = lambda: 0.0
    key = BlockKey(b"k0")
    progress = kvcr._core._progress
    try:
        kvcr.deposit({key: [_mem_descriptor()]})
        _wait_until(lambda: bool(agent.transfers))
        copy = next(iter(progress._in_flight_ops.values()))
        _wait_until(lambda: copy.transfer_id is not None)
        transfer_id = copy.transfer_id
        registrations = progress._memory_registrations.copy()
        prepared = {
            owner: entries.copy() for owner, entries in progress._prepared.items()
        }
        residency = kvcr._core._block_record_map[key].local_dram

        with pytest.raises(RuntimeError, match="active transfers"):
            kvcr.close()
        assert not progress._thread.is_alive()
        assert not kvcr._core.is_quiescent() and not kvcr._core._closed
        assert transfer_id in progress._active_transfers
        assert not agent.released_xfers and not agent.deregistered
        assert progress._memory_registrations == registrations
        assert progress._prepared == prepared
        assert kvcr._core._block_record_map[key].local_dram is residency
    finally:
        # Fake-only cleanup: the stopped thread cannot observe late completion.
        agent.state = "DONE"
        with suppress(RuntimeError):
            progress.close()
        progress._close_progress_ops()
        progress._close_nixl()
        with suppress(RuntimeError):
            kvcr.close()  # Re-raises the saved failure after completing teardown.


def test_local_initialize_failure_completes_without_failing_progress() -> None:
    class FailingInitializeAgent(FakeNixlAgent):
        def initialize_xfer(self, *args, **kwargs):
            raise RuntimeError("invalid descriptors")

    primary = ctypes.create_string_buffer(16)
    local = ctypes.create_string_buffer(16)
    agent = FailingInitializeAgent()
    policy = _RecordingFIFOPolicy()
    kvcr = _new_local_kvcr(
        agent,
        local,
        1,
        policy=policy,
        framework_regions=[_buffer_region(primary, len(local) // 1)],
    )
    key = BlockKey(b"k0")

    op_handle = kvcr.deposit({key: [_mem_descriptor()]})

    assert _poll_until(kvcr, lambda results: bool(results)) == [
        (op_handle, _op_entries({key: False}))
    ]
    kvcr._core._progress.raise_if_failed()
    assert kvcr._core._progress._active_transfers == {}
    assert policy.ingested == []
    assert policy.removed == []


def _g2_recovered(**slots: int) -> dict[BlockKey, _BlockRecord]:
    return {
        BlockKey(name.encode()): _BlockRecord(
            local_dram=_LocalDramResidency([("", slot)], _LocalDramState.READY)
        )
        for name, slot in slots.items()
    }


def test_a_recovered_pool_deposits_into_free_rows_then_evicts_to_admit_more() -> None:
    """New data fills the rows recovery left free; once full, a recovered row goes."""
    block_size = 16
    primary = ctypes.create_string_buffer(block_size * 3)
    primary.raw = b"x" * block_size + b"y" * block_size + b"n" * block_size
    local = ctypes.create_string_buffer(block_size * 4)
    local[0:block_size] = b"s" * block_size
    local[2 * block_size : 3 * block_size] = b"f" * block_size
    agent = FakeNixlAgent()
    agent.state = "DONE"
    kvcr = _new_local_kvcr(
        agent, local, 4, framework_regions=[_buffer_region(primary, len(local) // 4)]
    )
    local_dram = kvcr._core._local_dram
    assert local_dram is not None

    records = _g2_recovered(first=2, second=0)
    records[BlockKey(b"first")].local_dram = _LocalDramResidency(
        [(":first", 2)], _LocalDramState.READY
    )
    residency = records[BlockKey(b"first")].local_dram
    residency._cached_size = 8  # Cached with a different pool size.
    install_recovery_records(kvcr._core, records)
    assert residency.indices_by_label == {":first": 0}
    assert local_dram._size_bytes(residency) == 16

    fresh = (BlockKey(b"fresh0"), BlockKey(b"fresh1"))
    operation = kvcr.deposit(
        {key: [_mem_descriptor(index)] for index, key in enumerate(fresh)}
    )
    completed = dict(_poll_until(kvcr, lambda results: bool(results)))

    assert completed[operation] == _op_entries({fresh[0]: True, fresh[1]: True})
    # Nothing recovered was evicted or overwritten.
    first, second = BlockKey(b"first"), BlockKey(b"second")
    keys = (first, second, *fresh)
    assert kvcr.query(keys) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)] * 4
    assert local.raw[:block_size] == b"s" * block_size
    assert local.raw[2 * block_size : 3 * block_size] == b"f" * block_size

    # Every row is now occupied, so the deposit below can only land by evicting
    # a recovered row -- a pool recovered full has to stay writable.
    assert not local_dram._free_slots[""]
    extra = BlockKey(b"extra")
    operation = kvcr.deposit({extra: [_mem_descriptor(2)]})
    completed = dict(_poll_until(kvcr, lambda results: bool(results)))

    assert completed[operation] == _op_entries({extra: True})
    # One recovered block gave up its row to the new bytes; the rest kept theirs.
    assert kvcr.query((first, second, *fresh, extra)) == [
        (QueryStatus.MISS, None),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]
    assert local.raw[:block_size] == b"s" * block_size
    assert local.raw[2 * block_size : 3 * block_size] == b"n" * block_size


def test_installing_records_into_a_core_that_holds_some_is_refused() -> None:
    """It replaces the table wholesale and rebuilds the free list from it."""
    local = ctypes.create_string_buffer(64)
    kvcr = _new_local_kvcr(FakeNixlAgent(), local, 4)
    kvcr._core._block_record_map[BlockKey(b"held")] = _BlockRecord(
        local_dram=_LocalDramResidency([("", 1)], _LocalDramState.READY)
    )

    with pytest.raises(RecoveryMirrorError, match="holds none"):
        install_recovery_records(kvcr._core, _g2_recovered(first=2, second=0))


@pytest.mark.parametrize(
    "records",
    [
        _g2_recovered(first=0, second=0),
        _g2_recovered(first=0, second=4),
        {
            BlockKey(key): _recovered_record(g2=[(label, 0)])
            for key, label in ((b"first", ":first"), (b"second", ":second"))
        },
        {BlockKey(b"first"): _recovered_record(g2=[(":same", 0), (":same", 1)])},
        {BlockKey(b"first"): _recovered_record(g2=[(":", 0)])},
        {BlockKey(b"first"): _recovered_record(g2=[(":*", 0)])},
        {
            BlockKey(b"first"): _BlockRecord(
                local_dram=_LocalDramResidency(
                    [(":a", 0), (":b", 1)],
                    _LocalDramState.READY,
                    pending_labels={":b"},
                )
            )
        },
        {
            BlockKey(b"first"): _BlockRecord(
                local_dram=_LocalDramResidency([("", 0)], _LocalDramState.FILLING)
            )
        },
    ],
    ids=[
        "duplicate-row",
        "row-out-of-range",
        "labeled-alias",
        "duplicate-label",
        "empty-label",
        "wildcard-label",
        "partial-key",
        "row-never-settled",
    ],
)
def test_adopt_recovery_slots_rejects_invalid_or_unsettled_rows(
    records: dict[BlockKey, _BlockRecord],
) -> None:
    """Duplicate, out-of-range, or still-filling recovered rows are refused."""
    local = ctypes.create_string_buffer(64)
    kvcr = _new_local_kvcr(FakeNixlAgent(), local, 4)
    local_dram = kvcr._core._local_dram
    assert local_dram is not None

    with pytest.raises(ValueError, match="invalid local DRAM recovery slots"):
        local_dram.adopt_recovery_slots(records)


def test_recovered_history_bounds_records_and_reports_callback_failure(
    monkeypatch, caplog
):
    monkeypatch.setenv("KVCR_KEY_HISTORY", "1")
    caplog.set_level(logging.DEBUG, logger="kvcr.core")
    local = ctypes.create_string_buffer(16 * 33)
    kvcr = _new_local_kvcr(FakeNixlAgent(), local, 33)
    records = {
        BlockKey(str(i).encode()): _BlockRecord(
            local_dram=_LocalDramResidency([("", i)], _LocalDramState.READY)
        )
        for i in range(33)
    }
    kvcr._core.adopt_recovery_records(records)
    history = _key_history(caplog)
    assert [r["batch_offset"] for r in history] == ["0", "32"]
    hashes = [r["key_sha256"].split(",") for r in history]
    assert [len(batch) for batch in hashes] == [32, 1]
    assert {h for batch in hashes for h in batch} == {
        hashlib.sha256(key).hexdigest() for key in records
    }
    key = next(iter(records))
    assert not kvcr._core._publish_inventory((key,), CacheTier.LOCAL_G2, removed=True)
    assert _key_history(caplog)[-1]["event"] == "inventory_no_callback"

    def fail(event):
        raise RuntimeError("callback failed")

    kvcr._core._inventory_sink_callback = fail
    assert not kvcr._core._publish_inventory((key,), CacheTier.LOCAL_G2, removed=True)
    assert _key_history(caplog)[-1]["event"] == "inventory_failed"
