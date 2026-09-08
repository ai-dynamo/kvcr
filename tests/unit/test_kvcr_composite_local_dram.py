# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Regression tests for composite local-DRAM capacity handling."""

import ctypes
from unittest.mock import Mock

import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _mem_descriptor,
    _new_kvcr,
    _op_entries,
    _poll_until,
)

from kvcr.config import KVCRConfig, LocalDramOptions
from kvcr.policy import FIFOPolicy
from kvcr.types import BlockKey, CacheTier, PlacementAction, QueryStatus


def _new_composite_kvcr(agent, pools, *, policy=None):
    return _new_kvcr(
        agent,
        FakePrimaryPinning(),
        FakeBytesControl(),
        config=KVCRConfig(
            nixl_agent_name="target",
            pool_layouts=[(name, 8) for name, _buffer in pools],
            inventory_report_interval_ms=0,
        ),
        local_dram=LocalDramOptions(
            [(name, ctypes.addressof(buffer), len(buffer)) for name, buffer in pools]
        ),
        policy=policy,
    )


class _KeepOneVictimPolicy(FIFOPolicy):
    def __init__(self, kept_key: BlockKey) -> None:
        self.kept_key = kept_key
        self.decided: list[BlockKey] = []

    def decide_eviction(self, meta, source):
        self.decided.append(meta.block_key)
        if meta.block_key == self.kept_key:
            return (PlacementAction.KEEP, None)
        return super().decide_eviction(meta, source)


def test_failed_composite_plan_does_not_apply_partial_evictions() -> None:
    primary = ctypes.create_string_buffer(b"a" * 8 + b"b" * 8 + b"n" * 16)
    pools = (
        ("a", ctypes.create_string_buffer(8)),
        ("b", ctypes.create_string_buffer(8)),
    )
    old_a, old_b, new = (
        BlockKey(b"old-a"),
        BlockKey(b"old-b"),
        BlockKey(b"new"),
    )
    policy = _KeepOneVictimPolicy(old_b)
    agent = FakeNixlAgent()
    agent.state = "DONE"
    kvcr = _new_composite_kvcr(agent, pools, policy=policy)
    address = ctypes.addressof(primary)

    initial = kvcr.deposit(
        {
            old_a: [_mem_descriptor(address, 8, "a")],
            old_b: [_mem_descriptor(address + 8, 8, "b")],
        }
    )
    assert dict(_poll_until(kvcr, bool))[initial] == _op_entries(
        {old_a: True, old_b: True}
    )

    replacement = kvcr.deposit(
        {
            new: [
                _mem_descriptor(address + 16, 8, "a"),
                _mem_descriptor(address + 24, 8, "b"),
            ]
        }
    )
    assert list(kvcr.poll_completed()) == [(replacement, _op_entries({new: False}))]
    assert policy.decided == [old_a, old_b]
    assert kvcr.query((old_a, old_b, new)) == [
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
        (QueryStatus.MISS, None),
    ]


def test_blocked_pool_waiter_does_not_hold_up_an_independent_pool() -> None:
    primary = ctypes.create_string_buffer(b"A" * 8 + b"B" * 8 + b"a" * 8 + b"b" * 8)
    pools = (
        ("a", ctypes.create_string_buffer(8)),
        ("b", ctypes.create_string_buffer(8)),
    )
    old_a, old_b, new_a, new_b = (
        BlockKey(b"old-a"),
        BlockKey(b"old-b"),
        BlockKey(b"new-a"),
        BlockKey(b"new-b"),
    )
    agent = FakeNixlAgent()
    agent.state = "DONE"
    kvcr = _new_composite_kvcr(agent, pools)
    local = kvcr._core._local_dram
    assert local is not None
    address = ctypes.addressof(primary)

    old_a_op = kvcr.deposit({old_a: [_mem_descriptor(address, 8, "a")]})
    assert dict(_poll_until(kvcr, bool))[old_a_op] == _op_entries({old_a: True})
    old_b_op = kvcr.deposit(
        {old_b: [_mem_descriptor(address + 8, 8, "b")]}, no_evict=True
    )
    old_b_result = dict(_poll_until(kvcr, bool))[old_b_op][old_b]
    assert old_b_result.release_handle is not None

    # Model an asynchronous eviction from pool A. While it is pending, both
    # full-pool admissions become waiters in FIFO order.
    local._capacity_eviction_key = old_a
    wait_a = kvcr.deposit({new_a: [_mem_descriptor(address + 16, 8, "a")]})
    wait_b = kvcr.deposit({new_b: [_mem_descriptor(address + 24, 8, "b")]})
    assert [waiter.key for waiter in local._capacity_waiters] == [new_a, new_b]

    # Releasing B invokes the waiter pump. A remains blocked, but B can now
    # reserve its independent pool and finish.
    local.retire_sources((old_b,))
    assert kvcr.release((old_b_result.release_handle,)) == [
        (old_b_result.release_handle, True)
    ]
    completed = dict(_poll_until(kvcr, lambda done: wait_b in dict(done)))
    assert completed[wait_b] == _op_entries({new_b: True})
    assert wait_a not in completed
    assert [waiter.key for waiter in local._capacity_waiters] == [new_a]

    # Settle the synthetic pending eviction so this test leaves no pending op.
    local.abandon_capacity_eviction(old_a)
    local._resume_capacity_waiters()
    assert dict(_poll_until(kvcr, lambda done: wait_a in dict(done)))[wait_a] == (
        _op_entries({new_a: True})
    )


def test_discarding_retry_survives_an_unrelated_waiter_resume() -> None:
    pools = (
        ("a", ctypes.create_string_buffer(8)),
        ("b", ctypes.create_string_buffer(8)),
    )
    kvcr = _new_composite_kvcr(FakeNixlAgent(), pools)
    local = kvcr._core._local_dram
    assert local is not None
    key = BlockKey(b"retry")
    kvcr._core._block_record(key)
    destinations, pending = local.reserve_fill(
        (key,),
        sources={key: CacheTier.REMOTE_G2},
        required_local=True,
        deadline=kvcr._core._operation_deadline(),
        expected_layout=("a",),
    )
    assert set(destinations) == {key}
    assert not pending
    local.discard_fill((key,))

    kvcr._core._remote_fw_dram.query = Mock(return_value=True)
    retry = kvcr.fetch((key,), request_id="remote", expected_layout=["a"])
    assert [waiter.key for waiter in local._capacity_waiters] == [key]

    # Any fill completion or retired claim can invoke this pump. The retry must
    # remain queued while the discarded fill still owns its extent.
    local._resume_capacity_waiters()
    assert list(kvcr.poll_completed()) == []
    assert [waiter.key for waiter in local._capacity_waiters] == [key]

    # Once the old transfer reaches a safe terminal state, its extent is freed
    # and the retry is attempted. No real remote hint exists, so that attempt
    # fails normally instead of being failed by the unrelated resume above.
    local.complete_fill((key,), success=False)
    assert list(kvcr.poll_completed()) == [(retry, _op_entries({key: False}))]


def test_local_pool_region_rejects_partial_trailing_slot() -> None:
    local = ctypes.create_string_buffer(9)

    with pytest.raises(ValueError, match="must contain complete blocks"):
        _new_kvcr(
            FakeNixlAgent(),
            FakePrimaryPinning(),
            FakeBytesControl(),
            config=KVCRConfig(
                nixl_agent_name="target",
                pool_layouts=[("a", 8)],
                inventory_report_interval_ms=0,
            ),
            local_dram=LocalDramOptions([("a", ctypes.addressof(local), len(local))]),
        )
