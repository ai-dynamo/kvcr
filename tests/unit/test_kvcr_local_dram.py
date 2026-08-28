# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KVCR local-DRAM, capacity, and policy tests."""

import ctypes
import logging
from unittest.mock import Mock

import pytest
from _kvcr_test_utils import (
    FakeNixlAgent,
    _mem_descriptor,
    _new_local_kvcr,
    _op_entries,
    _poll_until,
    _RecordingFIFOPolicy,
    _wait_until,
)

from kvcr.core import _BlockRecord
from kvcr.local_dram import _LocalDramResidency, _LocalDramState
from kvcr.policy import FIFOPolicy, LRUPolicy
from kvcr.recovery_journal import (
    RecoveryMirrorError,
    install_recovery_records,
)
from kvcr.types import (
    BlockKey,
    CacheTier,
    InventoryEvent,
    OpEntryStatus,
    PlacementAction,
    QueryStatus,
)


def test_local_deposit_deduplicates_and_evicts_fifo() -> None:
    block_size = 16
    primary = ctypes.create_string_buffer(block_size * 3)
    local = ctypes.create_string_buffer(block_size * 2)
    primary_addr = ctypes.addressof(primary)
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
    )
    keys = tuple(BlockKey(f"k{index}".encode()) for index in range(3))

    first = kvcr.deposit(
        {
            key: _mem_descriptor(primary_addr + index * block_size)
            for index, key in enumerate(keys[:2])
        },
        hints={"priority": "test"},
    )
    _wait_until(lambda: len(agent.transfers) == 1)
    filling_duplicate = kvcr.deposit({keys[0]: _mem_descriptor(primary_addr)})
    assert list(kvcr.poll_completed()) == []
    assert len(agent.transfers) == 1
    assert policy.ingested == []

    agent.state = "DONE"
    completed = _poll_until(kvcr, lambda results: len(results) == 2)
    assert dict(completed) == {
        first: _op_entries({keys[0]: True, keys[1]: True}),
        filling_duplicate: _op_entries({keys[0]: True}),
    }
    assert local.raw == b"a" * block_size + b"b" * block_size
    assert events == [InventoryEvent(keys[:2], CacheTier.LOCAL_G2, False)]
    assert [(meta.block_key, source) for meta, source in policy.ingested] == [
        (key, CacheTier.FW_G2) for key in keys[:2]
    ]
    assert all(
        meta.resident_tiers == frozenset((CacheTier.LOCAL_G2,))
        for meta, *_ in policy.ingested
    )

    ready_duplicate = kvcr.deposit({keys[0]: _mem_descriptor(primary_addr)})
    assert list(kvcr.poll_completed()) == [
        (ready_duplicate, _op_entries({keys[0]: True}))
    ]
    assert len(agent.transfers) == 1
    assert len(events) == 1
    assert len(policy.ingested) == 2

    replacement = kvcr.deposit(
        {keys[2]: _mem_descriptor(primary_addr + 2 * block_size)}
    )
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


@pytest.mark.parametrize(
    ("policy", "evicted_index"),
    [(FIFOPolicy(), 1), (LRUPolicy(), 0), (None, 0)],
    ids=["fifo", "lru", "default-lru"],
)
def test_builtin_policy_eviction_order(
    policy: FIFOPolicy | None, evicted_index: int
) -> None:
    block_size = 16
    primary = ctypes.create_string_buffer(block_size * 3)
    local = ctypes.create_string_buffer(block_size * 2)
    primary_addr = ctypes.addressof(primary)
    agent = FakeNixlAgent()
    agent.state = "DONE"
    kvcr = _new_local_kvcr(agent, local, 2, policy=policy)
    now = 0.0
    kvcr._core._clock = lambda: now
    keys = tuple(BlockKey(f"k{index}".encode()) for index in range(3))

    kvcr.deposit(
        {
            key: _mem_descriptor(primary_addr + index * block_size)
            for index, key in enumerate(keys[:2])
        }
    )
    _poll_until(kvcr, lambda results: bool(results))

    now = 1.0
    first_fetch = kvcr.fetch((keys[0],))
    first_claim = dict(kvcr.poll_completed())[first_fetch][keys[0]].release_handle
    now = 2.0
    second_fetch = kvcr.fetch((keys[1],))
    second_claim = dict(kvcr.poll_completed())[second_fetch][keys[1]].release_handle
    assert first_claim is not None and second_claim is not None
    kvcr.release((second_claim, first_claim))

    kvcr.deposit({keys[2]: _mem_descriptor(primary_addr + 2 * block_size)})
    _poll_until(kvcr, lambda results: bool(results))
    statuses = kvcr.query(keys)
    assert statuses.pop(evicted_index) == (QueryStatus.MISS, None)
    assert statuses == [
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]


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
    kvcr = _new_local_kvcr(agent, local, 2, policy=policy)

    op_handle = kvcr.deposit(
        {
            key: _mem_descriptor(ctypes.addressof(primary) + index * block_size)
            for index, key in enumerate(keys)
        },
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
    kvcr = _new_local_kvcr(agent, local, 1, policy=policy)
    keys = (BlockKey(b"k0"), BlockKey(b"k1"))

    with caplog.at_level(logging.WARNING, logger="kvcr.policy_runtime"):
        for index, key in enumerate(keys):
            op_handle = kvcr.deposit(
                {key: _mem_descriptor(ctypes.addressof(primary) + index * block_size)}
            )
            assert _poll_until(kvcr, lambda results: bool(results)) == [
                (op_handle, _op_entries({key: True}))
            ]

    assert policy.on_ingest.call_count == 2
    policy.on_remove.assert_called_once()
    warnings = [record.getMessage() for record in caplog.records]
    assert warnings.count("KVCR on_ingest failed") == 2
    assert warnings.count("KVCR on_remove failed") == 1


def test_local_claims_fetch_deliver_release_and_capacity() -> None:
    block_size = 16
    primary = ctypes.create_string_buffer(block_size * 2)
    local = ctypes.create_string_buffer(block_size)
    destination = ctypes.create_string_buffer(block_size * 2)
    primary_addr = ctypes.addressof(primary)
    primary.raw = b"a" * block_size + b"b" * block_size
    agent = FakeNixlAgent()
    policy = _RecordingFIFOPolicy()
    capacity_requests: list[int] = []
    kvcr = _new_local_kvcr(
        agent,
        local,
        1,
        capacity_low_watermark_percent=100,
        capacity_needed_callback=capacity_requests.append,
        policy=policy,
    )
    now = 0.0
    kvcr._core._clock = lambda: now
    first_key, second_key = BlockKey(b"k0"), BlockKey(b"k1")

    deposit = kvcr.deposit({first_key: _mem_descriptor(primary_addr)}, no_evict=True)
    _wait_until(lambda: bool(agent.transfers))
    assert capacity_requests == [1]
    fetch = kvcr.fetch((first_key,))
    assert kvcr.query((first_key,)) == [(QueryStatus.FETCHING, CacheTier.LOCAL_G2)]
    assert list(kvcr.poll_completed()) == []

    agent.state = "DONE"
    completed = dict(_poll_until(kvcr, lambda results: len(results) == 2))
    deposit_result = completed[deposit][first_key]
    assert deposit_result.success
    assert deposit_result.descriptor is not None
    deposit_claim = deposit_result.release_handle
    assert deposit_claim is not None

    assert kvcr.query((first_key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    fetch_result = completed[fetch][first_key]
    fetch_claim = fetch_result.release_handle
    assert fetch_result.descriptor == deposit_result.descriptor
    assert fetch_claim is not None

    ready_fetch = kvcr.fetch((first_key,))
    ready_fetch_result = dict(kvcr.poll_completed())[ready_fetch][first_key]
    ready_fetch_claim = ready_fetch_result.release_handle
    assert ready_fetch_claim is not None

    agent.state = "PROC"
    deliver = kvcr.deliver(
        {
            first_key: _mem_descriptor(ctypes.addressof(destination)),
            second_key: _mem_descriptor(ctypes.addressof(destination) + block_size),
        }
    )
    _wait_until(lambda: len(agent.transfers) == 2)
    assert kvcr.release((deposit_claim, fetch_claim, ready_fetch_claim)) == [
        (deposit_claim, True),
        (fetch_claim, True),
        (ready_fetch_claim, True),
    ]
    assert kvcr.release((fetch_claim,)) == [(fetch_claim, False)]

    blocked = kvcr.deposit({second_key: _mem_descriptor(primary_addr + block_size)})
    assert list(kvcr.poll_completed()) == [(blocked, _op_entries({second_key: False}))]

    agent.state = "DONE"
    now = 0.25
    assert _poll_until(kvcr, lambda results: bool(results)) == [
        (deliver, _op_entries({first_key: True, second_key: False}))
    ]
    assert destination.raw[:block_size] == b"a" * block_size
    assert [
        (meta.access_count, meta.last_access)
        for meta, _ in policy.scored
        if meta.block_key == first_key
    ] == [(0, 0.0), (3, 0.25)]
    assert [meta.block_key for meta, *_ in policy.ingested] == [first_key]
    assert policy.removed == []

    replacement = kvcr.deposit({second_key: _mem_descriptor(primary_addr + block_size)})
    assert capacity_requests == [1, 1]
    assert _poll_until(kvcr, lambda results: bool(results)) == [
        (replacement, _op_entries({second_key: True}))
    ]
    assert local.raw == b"b" * block_size


def test_capacity_needed_is_edge_triggered() -> None:
    local = ctypes.create_string_buffer(10)
    capacity_requests: list[int] = []
    kvcr = _new_local_kvcr(
        FakeNixlAgent(),
        local,
        10,
        capacity_low_watermark_percent=20,
        capacity_needed_callback=capacity_requests.append,
    )

    kvcr._core._update_capacity_pressure(2)
    kvcr._core._update_capacity_pressure(1)
    kvcr._core._update_capacity_pressure(0)
    assert capacity_requests == [2]

    kvcr._core._update_capacity_pressure(2)
    kvcr._core._update_capacity_pressure(1)
    assert capacity_requests == [2, 2]


@pytest.mark.parametrize(
    ("failure", "terminal_state", "success"),
    [
        ("submission", None, False),
        ("timeout", None, False),
        ("timeout", "DONE", True),
    ],
)
def test_local_deposit_waits_for_safe_release(
    failure: str, terminal_state: str | None, success: bool
) -> None:
    class DelayedReleaseAgent(FakeNixlAgent):
        def __init__(self):
            super().__init__()
            self.allow_release = False
            self.release_attempts = 0

        def transfer(self, handle):
            result = super().transfer(handle)
            return "ERR" if failure == "submission" else result

        def release_xfer_handle(self, handle):
            self.release_attempts += 1
            if not self.allow_release:
                return False
            super().release_xfer_handle(handle)

    now = 0.0
    block_size = 16
    primary = ctypes.create_string_buffer(block_size)
    local = ctypes.create_string_buffer(block_size)
    agent = DelayedReleaseAgent()
    kvcr = _new_local_kvcr(agent, local, 1)
    kvcr._core._clock = lambda: now
    key = BlockKey(b"k0")

    op_handle = kvcr.deposit(
        {key: _mem_descriptor(ctypes.addressof(primary), block_size)}
    )
    _wait_until(lambda: bool(agent.transfers))
    if failure == "timeout":
        now = 2.0
    _wait_until(lambda: agent.release_attempts > 0)

    assert list(kvcr.poll_completed()) == []
    assert kvcr._core._block_record_map[key].local_dram is not None

    if terminal_state is not None:
        agent.state = terminal_state
    agent.allow_release = True
    assert _poll_until(kvcr, lambda results: bool(results)) == [
        (op_handle, _op_entries({key: success}))
    ]
    assert agent.released_xfers == [1]
    assert (key in kvcr._core._block_record_map) is success


def test_local_initialize_failure_completes_without_failing_progress() -> None:
    class FailingInitializeAgent(FakeNixlAgent):
        def initialize_xfer(self, *args, **kwargs):
            raise RuntimeError("invalid descriptors")

    primary = ctypes.create_string_buffer(16)
    local = ctypes.create_string_buffer(16)
    agent = FailingInitializeAgent()
    policy = _RecordingFIFOPolicy()
    kvcr = _new_local_kvcr(agent, local, 1, policy=policy)
    key = BlockKey(b"k0")

    op_handle = kvcr.deposit(
        {key: _mem_descriptor(ctypes.addressof(primary), len(primary))}
    )

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
            local_dram=_LocalDramResidency(slot, _LocalDramState.READY)
        )
        for name, slot in slots.items()
    }


def test_adopt_recovery_slots_rebuilds_ready_allocator_state() -> None:
    local = ctypes.create_string_buffer(64)
    kvcr = _new_local_kvcr(FakeNixlAgent(), local, 4)
    first, second = BlockKey(b"first"), BlockKey(b"second")
    records = _g2_recovered(first=2, second=0)

    assert kvcr._core._local_dram is not None
    local_dram = kvcr._core._local_dram
    kvcr._core._block_record_map.update(records)
    local_dram.adopt_recovery_slots(records)

    assert list(local_dram._free_slots) == [1, 3]
    del first, second


def test_a_pool_recovered_full_still_admits_a_new_deposit() -> None:
    """A cache worth recovering is usually a full one, so it has to stay writable."""
    local = ctypes.create_string_buffer(32)
    primary = ctypes.create_string_buffer(16)
    primary.raw = b"n" * 16
    agent = FakeNixlAgent()
    kvcr = _new_local_kvcr(agent, local, 2)
    local_dram = kvcr._core._local_dram
    assert local_dram is not None

    recovered = install_recovery_records(kvcr._core, _g2_recovered(first=0, second=1))

    # The returned keys are the only route from a recovered block to a router.
    assert set(recovered) == {BlockKey(b"first"), BlockKey(b"second")}

    # Every row the pool has came back occupied, so the deposit below can only
    # land by evicting one of them.
    assert not local_dram._free_slots

    fresh = BlockKey(b"fresh")
    operation = kvcr.deposit({fresh: _mem_descriptor(ctypes.addressof(primary))})
    agent.state = "DONE"
    completed = dict(_poll_until(kvcr, lambda results: bool(results)))

    assert completed[operation] == _op_entries({fresh: True})
    assert kvcr._core._block_record_map[fresh].local_dram is not None
    # One recovered block gave up its row; the other kept it.
    assert len(kvcr._core._block_record_map) == 2


def test_installing_records_into_a_core_that_holds_some_is_refused() -> None:
    """It replaces the table wholesale and rebuilds the free list from it."""
    local = ctypes.create_string_buffer(64)
    kvcr = _new_local_kvcr(FakeNixlAgent(), local, 4)
    kvcr._core._block_record_map[BlockKey(b"held")] = _BlockRecord(
        local_dram=_LocalDramResidency(1, _LocalDramState.READY)
    )

    with pytest.raises(RecoveryMirrorError, match="holds none"):
        install_recovery_records(kvcr._core, _g2_recovered(first=2, second=0))


@pytest.mark.parametrize("slots", [(0, 0), (0, 4)])
def test_adopt_recovery_slots_rejects_invalid_slots(slots: tuple[int, int]) -> None:
    local = ctypes.create_string_buffer(64)
    kvcr = _new_local_kvcr(FakeNixlAgent(), local, 4)
    local_dram = kvcr._core._local_dram
    assert local_dram is not None

    with pytest.raises(ValueError, match="invalid local DRAM recovery slots"):
        local_dram.adopt_recovery_slots(_g2_recovered(first=slots[0], second=slots[1]))


def test_adopt_recovery_slots_rejects_a_row_that_never_settled() -> None:
    local = ctypes.create_string_buffer(64)
    kvcr = _new_local_kvcr(FakeNixlAgent(), local, 4)
    local_dram = kvcr._core._local_dram
    assert local_dram is not None
    unsettled = {
        BlockKey(b"first"): _BlockRecord(
            local_dram=_LocalDramResidency(0, _LocalDramState.FILLING)
        )
    }

    with pytest.raises(ValueError, match="invalid local DRAM recovery slots"):
        local_dram.adopt_recovery_slots(unsettled)
