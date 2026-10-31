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
    _record_hint_results,
    _wait_until,
    _write_done_notification,
)

from kvcr.config import LocalDramOptions, RemoteFWDramOptions
from kvcr.hints.fetch import KV_FETCH
from kvcr.hints.parser import parse_hints
from kvcr.hints.results import HintStatus
from kvcr.types import BlockKey, CacheTier, InventoryEvent, QueryStatus


@pytest.fixture
def dynamo_hint():
    path = Path(__file__).parents[1] / "fixtures" / "dynamo_decode_to_prefill_hint.json"
    return msgspec.json.decode(path.read_bytes())


class _U64KeyAdapter:
    def decode(self, key: BlockKey) -> int:
        return int.from_bytes(key[:8], "big", signed=False)


class _VllmHintAction(msgspec.Struct, frozen=True):
    action_id: str
    action_type: str
    action_version: str
    payload: dict[str, object]


class _VllmHintEnvelope(msgspec.Struct, frozen=True):
    protocol_version: str
    message_id: str
    actions: list[_VllmHintAction]


def test_structural_vllm_envelope_preserves_action_and_message_ids(
    dynamo_hint, monkeypatch
):
    kvcr = _new_kvcr(
        FakeNixlAgent(),
        FakePrimaryPinning(),
        FakeBytesControl(),
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
    )
    results = _record_hint_results(monkeypatch, kvcr)
    envelope = msgspec.convert(dynamo_hint, type=_VllmHintEnvelope)
    kvcr.submit_hint(envelope, request_id="turn-2")
    assert len(results) == 1
    (result,) = results
    assert result.status is HintStatus.ACCEPTED
    assert (
        result.action_id,
        result.action_type,
        result.message_id,
        result.request_id,
    ) == (
        envelope.actions[0].action_id,
        "kv.fetch",
        envelope.message_id,
        "turn-2",
    )


def test_dynamo_envelope_accepts_unsigned_root_prefix(dynamo_hint):
    hint = msgspec.convert(dynamo_hint, type=_HintEnvelope)
    hints, rejected = parse_hints(hint, {KV_FETCH.action_type: KV_FETCH})
    assert rejected == []
    (parsed,) = hints

    assert parsed.source_endpoint == "tcp://decode-worker:9000"
    assert parsed.block_hashes == frozenset((0, 1 << 63, (1 << 64) - 2, (1 << 64) - 1))


def test_dynamo_mapping_must_be_decoded_to_typed_envelope(dynamo_hint):
    with pytest.raises(ValueError, match="invalid router hint"):
        parse_hints(dynamo_hint, {KV_FETCH.action_type: KV_FETCH})


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
    fetch = None
    try:
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
                result.success
                for result in dict(_poll_until(kvcr, bool))[deposit].values()
            )
            agent.state = "PROC"

        target.submit_hint(
            msgspec.convert(dynamo_hint, type=_HintEnvelope),
            request_id="turn-2-prefill",
        )
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

        residencies = {
            key: source._core._block_record_map[key].local_dram for key in missing_keys
        }
        slots = {key: residency.slots for key, residency in residencies.items()}
        source_control.incoming.append(raw_request)
        _poll_until(source, lambda _: len(source_agent.xfers) == 2)
        assert source_pinning.searches == []  # All source blocks are already in DRAM.
        assert len(source_agent.xfers[-1][1]) == len(missing_keys)
        notification = source_agent.xfers[-1][5]
        assert notification is not None
        pools = source._core._local_dram._pools
        assert pools.source_operations == 1
        assert all(
            source._core._block_record_map[key].local_dram is residencies[key]
            and residencies[key].slots == slots[key]
            and residencies[key].source_readers == 1
            and pools._source_state.active[key] is residencies[key]
            for key in missing_keys
        )
        assert list(target.poll_completed()) == []

        source_agent.state = "DONE"
        _poll_until(
            source,
            lambda _: not _has_outstanding_operations(source)
            and all(
                residency.source_readers == 0 for residency in residencies.values()
            ),
        )
        assert pools.source_operations == 0
        assert all(
            source._core._block_record_map[key].local_dram is residencies[key]
            and residencies[key].slots == slots[key]
            and residencies[key].claim_count == 0
            and key not in pools._source_state.active
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
    finally:
        # Failed assertions must still retire the fake native work before the
        # autouse fixture closes these instances or the next test replaces fakes.
        source_agent.state = target_agent.state = "DONE"
        _poll_until(
            source,
            lambda _: not source_control.incoming
            and not _has_outstanding_operations(source)
            and not source._core._progress._active_transfers
            and not source._core._progress._in_flight_ops,
        )
        if fetch is not None and _has_outstanding_operations(target):
            # The source is now quiescent, so a failed terminal acknowledgement
            # can safely retire any unfinished destination reservation.
            target_agent.notifs.setdefault("source", []).append(
                _write_done_notification(fetch, success=False)
            )
        _poll_until(target, lambda _: not _has_outstanding_operations(target))
