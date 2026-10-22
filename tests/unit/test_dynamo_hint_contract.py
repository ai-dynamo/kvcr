# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dynamo's cross-role hint uses the existing request-driven copy contract."""

import ctypes
from pathlib import Path

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _buffer_region,
    _decode_control_message,
    _has_outstanding_operations,
    _HintEnvelope,
    _mem_descriptor,
    _new_kvcr,
    _poll_until,
    _wait_until,
)

from kvcr.config import LocalDramOptions, RemoteFWDramOptions
from kvcr.hint_parser import _KVFetchHint
from kvcr.types import BlockKey, CacheTier, InventoryEvent, QueryStatus


@pytest.fixture
def dynamo_hint():
    path = Path(__file__).parents[1] / "fixtures" / "dynamo_decode_to_prefill_hint.json"
    return msgspec.json.decode(path.read_bytes())


class _U64KeyAdapter:
    def decode(self, key: BlockKey) -> int:
        return int.from_bytes(key[:8], "big", signed=False)


@pytest.mark.parametrize("as_mapping", [False, True])
def test_dynamo_envelope_accepts_unsigned_root_prefix(dynamo_hint, as_mapping):
    hint = (
        dynamo_hint if as_mapping else msgspec.convert(dynamo_hint, type=_HintEnvelope)
    )
    parsed = _KVFetchHint.from_hint(hint)

    assert parsed.source_endpoint == "tcp://decode-worker:9000"
    assert parsed.block_hashes == frozenset((0, 1 << 63, (1 << 64) - 2, (1 << 64) - 1))


@pytest.mark.parametrize("local_prefix_blocks", [1, 2])
def test_dynamo_root_hint_fetches_only_missing_suffix_and_keeps_source_copy(
    dynamo_hint, local_prefix_blocks
):
    payload = dynamo_hint["actions"][0]["payload"]
    hashes = payload["block_hashes"].copy()
    keys = tuple(BlockKey(value.to_bytes(8, "big")) for value in hashes)
    local_keys, missing_keys = keys[:local_prefix_blocks], keys[local_prefix_blocks:]
    block_size = 16
    primary = ctypes.create_string_buffer(b"p" * block_size * len(keys))
    source_local = ctypes.create_string_buffer(block_size * len(keys))
    target_local = ctypes.create_string_buffer(block_size * len(keys))
    source_agent, target_agent = FakeNixlAgent(), FakeNixlAgent()
    source_control = FakeBytesControl(payload["source_control_endpoint"])
    target_control = FakeBytesControl("tcp://prefill-worker:9000")
    source_pinning = FakePrimaryPinning()
    source_events: list[InventoryEvent] = []
    target_events: list[InventoryEvent] = []
    source = _new_kvcr(
        source_agent,
        source_pinning,
        source_control,
        name="source",
        local_dram=LocalDramOptions(
            [("", ctypes.addressof(source_local), len(source_local))]
        ),
        framework_regions=[_buffer_region(primary, block_size)],
        inventory_sink=source_events.append,
    )
    target = _new_kvcr(
        target_agent,
        FakePrimaryPinning(),
        target_control,
        key_adapter=_U64KeyAdapter(),
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
        local_dram=LocalDramOptions(
            [("", ctypes.addressof(target_local), len(target_local))]
        ),
        framework_regions=[_buffer_region(primary, block_size)],
        inventory_sink=target_events.append,
    )
    for kvcr, agent, resident_keys, endpoint in (
        (source, source_agent, keys, "source"),
        (target, target_agent, local_keys, "target"),
    ):
        agent.state = "DONE"
        deposit = kvcr.deposit(
            {
                key: [_mem_descriptor(index, end_point_name=endpoint)]
                for index, key in enumerate(resident_keys)
            }
        )
        assert all(
            result.success for result in dict(_poll_until(kvcr, bool))[deposit].values()
        )
        agent.state = "PROC"

    target.submit_hint(dynamo_hint, request_id="turn-2-prefill")
    assert target.query(keys, "turn-2-prefill") == [
        *((QueryStatus.HIT, CacheTier.LOCAL_G2),) * len(local_keys),
        *((QueryStatus.FETCHABLE, CacheTier.REMOTE_G2),) * len(missing_keys),
    ]
    assert target.query((BlockKey((42).to_bytes(8, "big")),), "turn-2-prefill") == [
        (QueryStatus.MISS, None)
    ]
    assert target_control.sent == []  # A hint alone does not copy cache blocks.
    fetch = target.fetch(keys, request_id="turn-2-prefill")
    _wait_until(lambda: bool(target_control.sent))
    ((endpoint, raw_request),) = target_control.sent
    request = _decode_control_message(raw_request)
    assert endpoint == payload["source_control_endpoint"]
    assert request["type"] == "start_write"
    assert request["keys"] == list(missing_keys)
    assert payload["block_hashes"] == hashes  # The full root chain stays intact.
    assert target.query(keys, "turn-2-prefill") == [
        *((QueryStatus.HIT, CacheTier.LOCAL_G2),) * len(local_keys),
        *((QueryStatus.FETCHING, CacheTier.LOCAL_G2),) * len(missing_keys),
    ]
    assert target_events == [InventoryEvent(local_keys, CacheTier.LOCAL_G2, False)]

    source_control.incoming.append(raw_request)
    _poll_until(source, lambda _: len(source_agent.xfers) == 2)
    assert source_pinning.searches == []  # All source blocks are already in DRAM.
    assert len(source_agent.xfers[-1][1]) == len(missing_keys)
    notification = source_agent.xfers[-1][5]
    assert notification is not None
    assert all(
        source._core._block_record_map[key].local_dram.claim_count == 1
        for key in missing_keys
    )
    assert list(target.poll_completed()) == []

    source_agent.state = "DONE"
    _poll_until(source, lambda _: not _has_outstanding_operations(source))
    assert all(
        source._core._block_record_map[key].local_dram.claim_count == 0
        for key in missing_keys
    )
    # Source completion alone cannot publish destination residency.
    assert list(target.poll_completed()) == []
    assert target_events == [InventoryEvent(local_keys, CacheTier.LOCAL_G2, False)]
    target_agent.notifs["source"] = [notification]
    results = dict(_poll_until(target, bool))[fetch]
    assert set(results) == set(keys)
    assert all(result.success for result in results.values())
    handles = tuple(result.release_handle for result in results.values())
    assert all(handle is not None for handle in handles)
    assert target.release(handles) == [(handle, True) for handle in handles]
    target.discard_hint("turn-2-prefill")
    assert target.query(keys) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)] * len(keys)
    assert target_events == [
        InventoryEvent(local_keys, CacheTier.LOCAL_G2, False),
        InventoryEvent(missing_keys, CacheTier.LOCAL_G2, False),
    ]
    assert source.query(keys) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)] * len(keys)
    assert source_events == [InventoryEvent(keys, CacheTier.LOCAL_G2, False)]
