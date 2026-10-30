# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Internal removal preserves existing readers and application retention."""

import ctypes

import pytest
from _kvcr_test_utils import (
    _buffer_region,
    _invalidate,
    _mem_descriptor,
    _new_local_kvcr,
    _poll_until,
    _wait_until,
)
from test_kvcr_move import _Pair
from test_kvcr_remote_target import _CopyingWriteAgent

from kvcr.policy import FIFOPolicy
from kvcr.types import BlockKey, CacheTier, InventoryEvent, QueryStatus


class _Local:
    def __init__(self, *, policy=None, slots=1):
        self.agent = _CopyingWriteAgent()
        self.local = ctypes.create_string_buffer(16 * slots)
        self.framework = ctypes.create_string_buffer(b"a" * 16 + b"b" * 16, 32)
        self.events = []
        self.kvcr = _new_local_kvcr(
            self.agent,
            self.local,
            slots,
            inventory_sink=self.events.append,
            framework_regions=[_buffer_region(self.framework, 16)],
            policy=policy,
        )

    def deposit(self, key, *, index=0, no_evict=False):
        self.agent.state = "DONE"
        handle = self.kvcr.deposit({key: [_mem_descriptor(index)]}, no_evict=no_evict)
        result = dict(_poll_until(self.kvcr, bool))[handle][key]
        self.agent.state = "PROC"
        assert result.success
        return result


def test_invalidate_drains_existing_readers_and_rejects_new_reads():
    fixture = _Local()
    kvcr = fixture.kvcr
    key = BlockKey(b"active-reader")
    fixture.deposit(key)
    fetch = kvcr.fetch((key,))
    reader = dict(kvcr.poll_completed())[fetch][key].release_handle
    assert reader is not None

    _invalidate(kvcr, (key, key))

    assert kvcr.query((key,)) == [(QueryStatus.MISS, None)]
    another_fetch = kvcr.fetch((key,))
    another_delivery = kvcr.deliver({key: [_mem_descriptor(1)]})
    completed = dict(kvcr.poll_completed())
    assert kvcr._core._invalidation_handler.has_pending()
    for handle in (another_fetch, another_delivery):
        assert not completed[handle][key].success
        assert completed[handle][key].release_handle is None
    assert kvcr._core._block_record_map[key].local_dram.claim_count == 1
    assert fixture.local.raw == b"a" * 16
    assert fixture.framework.raw[16:] == b"b" * 16
    assert not any(event.removed for event in fixture.events)
    pools = kvcr._core._local_dram._pools
    assert pools.telemetry_state()["local_g2_free_slots"] == 0

    assert kvcr.release((reader,)) == [(reader, True)]
    assert not kvcr.poll_completed()
    assert not kvcr._core._invalidation_handler.has_pending()
    assert pools.telemetry_state()["local_g2_free_slots"] == 1
    assert pools.telemetry_state()["local_g2_evictable_slots"] == 0
    assert fixture.events.count(InventoryEvent((key,), CacheTier.LOCAL_G2, True)) == 1
    # Freed slots can be reused without resurrecting the removed key.
    fixture.deposit(BlockKey(b"replacement"), index=1)
    assert kvcr.query((key,)) == [(QueryStatus.MISS, None)]
    assert fixture.local.raw == b"b" * 16


def test_existing_retention_skips_invalidation():
    fixture = _Local()
    kvcr = fixture.kvcr
    key = BlockKey(b"retained")
    retention = fixture.deposit(key, no_evict=True).release_handle
    fetch = kvcr.fetch((key,))
    reader = dict(kvcr.poll_completed())[fetch][key].release_handle
    _invalidate(kvcr, (key,))
    assert not kvcr.poll_completed()
    handler = kvcr._core._invalidation_handler
    assert retention is not None
    assert not handler.has_pending()
    assert not kvcr._core._block_record_map[key].pending_invalidation
    assert kvcr.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    reopened = kvcr.fetch((key,))
    result = dict(kvcr.poll_completed())[reopened][key]
    assert result.success and result.release_handle is not None
    assert kvcr.release((result.release_handle,)) == [(result.release_handle, True)]
    assert kvcr.release((reader,)) == [(reader, True)]
    _invalidate(kvcr, (key,))
    assert not kvcr.poll_completed()
    assert not handler.has_pending()
    assert not any(event.removed for event in fixture.events)
    assert fixture.local.raw == b"a" * 16

    # Releasing retention does not revive skipped removal requests.
    assert kvcr.release((retention,)) == [(retention, True)]
    assert not kvcr.poll_completed()
    assert kvcr.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    _invalidate(kvcr, (key,))
    assert not kvcr.poll_completed()
    assert kvcr.query((key,)) == [(QueryStatus.MISS, None)]
    assert fixture.events.count(InventoryEvent((key,), CacheTier.LOCAL_G2, True)) == 1


@pytest.mark.parametrize("no_evict", [False, True])
def test_pending_invalidation_rejects_only_affected_deposit_keys(no_evict):
    fixture = _Local(slots=2)
    kvcr = fixture.kvcr
    key, other = BlockKey(b"removing"), BlockKey(b"unrelated")
    fixture.deposit(key)
    fetch = kvcr.fetch((key,))
    reader = dict(kvcr.poll_completed())[fetch][key].release_handle
    _invalidate(kvcr, (key,))

    fixture.agent.state = "DONE"
    deposit = kvcr.deposit(
        {key: [_mem_descriptor()], other: [_mem_descriptor(1)]}, no_evict=no_evict
    )
    results = dict(_poll_until(kvcr, bool))[deposit]
    assert not results[key].success and results[key].release_handle is None
    assert results[other].success
    assert (results[other].release_handle is not None) is no_evict
    assert len(fixture.agent.transfers) == 2
    record = kvcr._core._block_record_map[key]
    assert record.pending_invalidation
    assert record.local_dram.retention_handles is None
    assert record.local_dram.claim_count == 1
    assert kvcr.query((key, other)) == [
        (QueryStatus.MISS, None),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]
    assert fixture.local.raw == b"a" * 16 + b"b" * 16
    assert not any(event.removed for event in fixture.events)

    assert kvcr.release((reader,)) == [(reader, True)]
    assert not kvcr.poll_completed()
    assert not kvcr._core._invalidation_handler.has_pending()
    assert key not in kvcr._core._block_record_map
    assert fixture.events.count(InventoryEvent((key,), CacheTier.LOCAL_G2, True)) == 1
    retained = results[other].release_handle
    if retained is not None:
        assert kvcr.release((retained,)) == [(retained, True)]
    fixture.deposit(key)
    assert kvcr.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]


@pytest.mark.parametrize(
    "no_evict,duplicate,succeeds",
    [
        (False, False, True),
        (False, True, True),
        (True, False, True),
        (True, True, True),
        (True, False, False),
    ],
)
def test_inflight_deposits_obey_invalidation_order(no_evict, duplicate, succeeds):
    fixture = _Local()
    kvcr = fixture.kvcr
    key = BlockKey(b"copying")
    blocks = {key: [_mem_descriptor()]}
    writer = kvcr.deposit(blocks, no_evict=no_evict and not duplicate)
    _wait_until(lambda: bool(fixture.agent.transfers))
    deposit = kvcr.deposit(blocks, no_evict=no_evict) if duplicate else writer
    _invalidate(kvcr, (key,))
    assert not kvcr.poll_completed()
    assert kvcr._core._invalidation_handler.has_pending()
    assert kvcr.query((key,)) == [(QueryStatus.MISS, None)]
    residency = kvcr._core._block_record_map[key].local_dram
    assert residency.claim_count == 1

    if not succeeds:
        states = iter(("ERR", "DONE"))
        fixture.agent.check_xfer_state = lambda _: next(states, "DONE")
    else:
        fixture.agent.state = "DONE"
    completed = dict(_poll_until(kvcr, lambda results: len(results) == 1 + duplicate))
    assert completed[writer][key].success is succeeds
    result = completed[deposit][key]
    assert result.success is succeeds
    assert len(fixture.agent.transfers) == 1
    if no_evict and succeeds:
        retention = result.release_handle
        assert retention is not None
        assert not kvcr._core._invalidation_handler.has_pending()
        assert kvcr.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
        assert kvcr.release((retention,)) == [(retention, True)]
        assert not kvcr.poll_completed()
        assert kvcr.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
        _invalidate(kvcr, (key,))
    elif not succeeds:
        assert result.release_handle is None
        assert not kvcr._core._local_dram._pools._public_claims
        # Failed protected operations must not block later source removal.
        fixture.deposit(key)
        _invalidate(kvcr, (key,))
    else:
        assert result.release_handle is None
    assert not kvcr.poll_completed()
    assert residency.retention_handles is None
    assert not kvcr._core._local_dram._pools._public_claims
    assert not kvcr._core._invalidation_handler.has_pending()
    assert key not in kvcr._core._block_record_map
    assert fixture.events.count(InventoryEvent((key,), CacheTier.LOCAL_G2, True)) == 1


def test_pending_invalidation_waits_for_readers_before_preserving_retention():
    fixture = _Local()
    kvcr = fixture.kvcr
    key = BlockKey(b"retaining-with-reader")
    deposit = kvcr.deposit({key: [_mem_descriptor()]}, no_evict=True)
    fetch = kvcr.fetch((key,))
    _invalidate(kvcr, (key,))
    fixture.agent.state = "DONE"
    results = dict(_poll_until(kvcr, lambda results: len(results) == 2))
    assert results[deposit][key].success and results[fetch][key].success
    retention, reader = (
        results[handle][key].release_handle for handle in (deposit, fetch)
    )
    assert retention is not None and reader is not None
    handler = kvcr._core._invalidation_handler
    assert handler.has_pending()
    assert kvcr.query((key,)) == [(QueryStatus.MISS, None)]
    assert fixture.local.raw == b"a" * 16

    assert kvcr.release((reader,)) == [(reader, True)]
    assert not kvcr.poll_completed()
    assert not handler.has_pending()
    assert kvcr.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert kvcr.release((retention,)) == [(retention, True)]
    assert not kvcr.poll_completed()
    assert kvcr.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    _invalidate(kvcr, (key,))
    assert not kvcr.poll_completed()
    assert key not in kvcr._core._block_record_map


def test_fetch_admitted_before_allocation_invalidation_keeps_its_claim(monkeypatch):
    pair = _Pair()
    key = BlockKey(b"invalidate-during-allocation")
    pair.deposit(key, no_evict=False)
    target = pair.target
    decide_ingest = target._core._policy.decide_ingest

    def invalidate_during_allocation(*args, **kwargs):
        _invalidate(target, (key,))
        return decide_ingest(*args, **kwargs)

    monkeypatch.setattr(
        target._core._policy, "decide_ingest", invalidate_during_allocation
    )
    fetch = pair.start((key,), mode="copy")
    assert target._core._block_record_map[key].pending_invalidation
    assert target.query((key,), "req") == [(QueryStatus.MISS, None)]
    for later in (
        target.fetch((key,)),
        target.deliver({key: pair.descriptors(endpoint="target")}, "req"),
    ):
        assert not pair.result(later)[key].success

    pair.copy_done()
    result = pair.result(fetch)[key]
    assert result.success and result.release_handle is not None
    assert pair.target_local[""].raw == b"a" * 16
    assert pair.target_primary[""].raw == bytes(len(pair.target_primary[""]))
    assert target.query((key,)) == [(QueryStatus.MISS, None)]
    assert target._core._invalidation_handler.has_pending()
    assert target.release((result.release_handle,)) == [(result.release_handle, True)]
    pair.until(lambda: key not in target._core._block_record_map)
    assert not target._core._invalidation_handler.has_pending()
    assert pair.source.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]


def test_removal_callback_can_redeposit_without_a_stale_removal_event():
    class RedepositPolicy(FIFOPolicy):
        callback = None

        def on_remove(self, meta):
            super().on_remove(meta)
            callback, self.callback = self.callback, None
            if callback is not None:
                callback(meta)

    policy = RedepositPolicy()
    fixture = _Local(policy=policy)
    kvcr = fixture.kvcr
    key = BlockKey(b"replaced-during-removal")
    fixture.deposit(key)
    fixture.events.clear()
    replacement = []

    def redeposit(meta):
        assert meta.block_key == key
        assert kvcr.query((key,)) == [(QueryStatus.MISS, None)]
        # deposit() polls recursively; the new residency must survive that poll.
        replacement.append(fixture.deposit(key, index=1, no_evict=True))

    policy.callback = redeposit
    _invalidate(kvcr, (key,))
    assert not kvcr.poll_completed()

    assert not kvcr._core._invalidation_handler.has_pending()
    assert len(replacement) == 1 and replacement[0].success
    retention = replacement[0].release_handle
    assert retention is not None
    assert kvcr.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert fixture.local.raw == b"b" * 16
    assert fixture.events == [InventoryEvent((key,), CacheTier.LOCAL_G2, False)]
    assert kvcr.release((retention,)) == [(retention, True)]
