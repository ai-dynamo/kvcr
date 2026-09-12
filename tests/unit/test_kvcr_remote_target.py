# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KVCR remote framework-DRAM target-side tests."""

import ctypes
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import DEFAULT, Mock

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    FakeTelemetryStats,
    _block_op_ids,
    _ConstantHashAdapter,
    _decode_control_message,
    _decode_notif,
    _has_outstanding_operations,
    _mem_descriptor,
    _new_kvcr,
    _op_entries,
    _poll_until,
    _RecordingFIFOPolicy,
    _router_hint,
    _wait_until,
    _write_done_notification,
)

from kvcr import TRANSFER_BLOCKS_METRIC, TRANSFER_BYTES_METRIC
from kvcr.config import KVCRConfig, LocalDramOptions, RemoteFWDramOptions
from kvcr.types import (
    BlockKey,
    CacheTier,
    InventoryEvent,
    OpEntryResult,
    OpEntryStatus,
    PlacementAction,
    QueryStatus,
)


def _make_block_key(block_hash: bytes, group_idx: int) -> BlockKey:
    return BlockKey(block_hash + group_idx.to_bytes(4, "big", signed=False))


def test_submit_hint_filters_unlisted_hash():
    target = _new_kvcr(
        FakeNixlAgent(),
        FakePrimaryPinning(),
        FakeBytesControl("tcp://target:1"),
        key_adapter=_ConstantHashAdapter(),
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
    )
    target.submit_hint(_router_hint("tcp://source:1", (999,)), request_id="req")

    assert target.query((BlockKey(b"k"),), "req") == [(QueryStatus.MISS, None)]


def test_kvcr_opportunistic_query_accepts_key_outside_hint():
    agent = FakeNixlAgent()
    control = FakeBytesControl("tcp://target:1")
    target = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        control,
        key_adapter=_ConstantHashAdapter(),
        remote_options=RemoteFWDramOptions(
            eager_ctrl_connect=False,
            opportunistic_query=True,
        ),
    )
    requested_key = BlockKey(b"k1")
    target.submit_hint(_router_hint("tcp://source:1", (999,)), request_id="req")

    assert target.query((requested_key,), "req") == [
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
    ]
    op_handle = target.deliver({requested_key: [_mem_descriptor()]}, request_id="req")

    assert list(target.poll_completed()) == []
    _wait_until(lambda: bool(control.sent))
    assert _decode_control_message(control.sent[0][1])["keys"] == [requested_key]
    target.discard_hint("req")
    assert target.query((requested_key,), "req") == [(QueryStatus.MISS, None)]
    agent.notifs["source"] = [_write_done_notification(op_handle)]
    _poll_until(target, bool)


def test_remote_fetch_uses_local_then_framework_sources() -> None:
    block_size = 16
    source_primary = ctypes.create_string_buffer(b"b" * block_size)
    source_local = ctypes.create_string_buffer(block_size * 2)
    target_local = ctypes.create_string_buffer(block_size * 2)
    source_agent = FakeNixlAgent(metadata=b"source-md")
    target_agent = FakeNixlAgent(metadata=b"target-md")
    source_pinning = FakePrimaryPinning()
    source_control = FakeBytesControl("tcp://source:1")
    target_control = FakeBytesControl("tcp://target:1")
    events: list[InventoryEvent] = []
    keys = (BlockKey(b"k0"), BlockKey(b"k1"))
    source_policy = _RecordingFIFOPolicy()
    policy = _RecordingFIFOPolicy()
    policy.decide_ingest = Mock(return_value=(PlacementAction.DROP, None))
    source = _new_kvcr(
        source_agent,
        source_pinning,
        source_control,
        name="source",
        local_dram=LocalDramOptions(
            [("", ctypes.addressof(source_local), len(source_local))]
        ),
        policy=source_policy,
    )
    source_now = 0.0
    source._core._clock = lambda: source_now
    target = _new_kvcr(
        target_agent,
        FakePrimaryPinning(),
        target_control,
        key_adapter=_ConstantHashAdapter(),
        remote_options=RemoteFWDramOptions(
            eager_ctrl_connect=False,
        ),
        local_dram=LocalDramOptions(
            [("", ctypes.addressof(target_local), len(target_local))]
        ),
        inventory_sink=events.append,
        policy=policy,
    )

    source_deposit = source.deposit(
        {keys[0]: [_mem_descriptor(ctypes.addressof(source_primary), block_size)]}
    )
    _wait_until(lambda: bool(source_agent.transfers))
    source_agent.state = "DONE"
    assert _poll_until(source, lambda completed: bool(completed)) == [
        (source_deposit, _op_entries({keys[0]: True}))
    ]
    source_agent.state = "PROC"

    target.submit_hint(_router_hint("tcp://source:1"), request_id="req")
    assert target.query(keys, "req") == [
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2),
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2),
    ]
    assert target_control.sent == []
    policy.decide_ingest.assert_not_called()
    fetch = target.fetch(keys, request_id="req", hints={"source": "framework"})
    assert target.query(keys, "req") == [
        (QueryStatus.FETCHING, CacheTier.LOCAL_G2),
        (QueryStatus.FETCHING, CacheTier.LOCAL_G2),
    ]

    _wait_until(
        lambda: any(
            _decode_control_message(message)["type"] == "start_write"
            for _, message in target_control.sent
        )
    )
    source_control.incoming.extend(message for _, message in target_control.sent)
    assert _poll_until(source, lambda _: len(source_agent.xfers) == 2) == []
    assert source_pinning.searches == [(keys[1],)]
    source_xfer = source_agent.xfers[1]
    assert source_xfer[1] == [
        (ctypes.addressof(source_local), block_size, 0),
        (0, block_size, 0),
    ]
    assert source._core._block_record_map[keys[0]].local_dram.claim_count == 1

    notification = source_xfer[5]
    assert notification is not None
    source_now = 0.25
    source_agent.state = "DONE"
    assert _poll_until(source, lambda _: not _has_outstanding_operations(source)) == []
    assert source._core._block_record_map[keys[0]].local_dram.claim_count == 0
    assert source_pinning.unpins == ["pin"]
    assert [
        (meta.access_count, meta.last_access)
        for meta, _ in source_policy.scored
        if meta.block_key == keys[0]
    ] == [(0, 0.0), (1, 0.25)]

    target_agent.notifs["source"] = [notification]
    completed = dict(_poll_until(target, lambda results: bool(results)))
    assert set(completed) == {fetch}
    assert all(result.success for result in completed[fetch].values())
    assert all(
        result.descriptors is not None and result.release_handle is not None
        for result in completed[fetch].values()
    )
    assert target.query(keys, "req") == [
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]
    assert [
        (args[0].block_key, *args[1:])
        for args, _ in policy.decide_ingest.call_args_list
    ] == [
        (key, CacheTier.REMOTE_G2, True, None, {"source": "framework"}) for key in keys
    ]
    assert [(meta.block_key, source) for meta, source in policy.ingested] == [
        (key, CacheTier.REMOTE_G2) for key in keys
    ]
    assert events == [InventoryEvent(keys, CacheTier.LOCAL_G2, False)]
    assert all(
        success
        for _, success in target.release(
            tuple(
                result.release_handle
                for result in completed[fetch].values()
                if result.release_handle is not None
            )
        )
    )


@pytest.mark.parametrize(
    ("layout", "expected_layout", "success"),
    [
        ([("full", 16), ("swa", 8)], ["swa", "full"], False),
        ([("full", 16), ("swa", 8)], ["full", "swa"], True),
        ([("", 16), ("", 16)], ["", ""], True),
    ],
)
def test_remote_fetch_preserves_block_layout_and_bytes(
    layout: list[tuple[str, int]],
    expected_layout: list[str],
    success: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class CopyingWriteAgent(FakeNixlAgent):
        def transfer(self, handle):
            self.transfers.append(handle)
            _, sources, _, destinations, _, _ = self.xfers[handle - 1]
            for (src, src_size, _), (dst, dst_size, _) in zip(
                sources, destinations, strict=True
            ):
                ctypes.memmove(dst, src, min(src_size, dst_size))
            return "PROC"

    payloads = [bytes([index + 1]) * size for index, (_, size) in enumerate(layout)]
    source_primary = [ctypes.create_string_buffer(data, len(data)) for data in payloads]
    pool_data = dict.fromkeys(dict(layout), b"")
    for (name, _), data in zip(layout, payloads, strict=True):
        pool_data[name] += data
    source_local = {
        name: ctypes.create_string_buffer(len(data)) for name, data in pool_data.items()
    }
    target_local = {
        name: ctypes.create_string_buffer(len(data)) for name, data in pool_data.items()
    }
    source_agent = CopyingWriteAgent(metadata=b"source-md")
    target_agent = FakeNixlAgent(metadata=b"target-md")
    source_control = FakeBytesControl("tcp://source:1")
    target_control = FakeBytesControl("tcp://target:1")
    config = KVCRConfig(
        nixl_agent_name="unused", pool_layouts=list(dict(layout).items())
    )

    def dram(memories: dict[str, ctypes.Array]) -> LocalDramOptions:
        return LocalDramOptions(
            [
                (name, ctypes.addressof(memory), len(memory))
                for name, memory in memories.items()
            ]
        )

    source = _new_kvcr(
        source_agent,
        FakePrimaryPinning(),
        source_control,
        config,
        name="source",
        local_dram=dram(source_local),
    )
    target = _new_kvcr(
        target_agent,
        FakePrimaryPinning(),
        target_control,
        config,
        key_adapter=_ConstantHashAdapter(),
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
        local_dram=dram(target_local),
    )
    key = BlockKey(b"multi-pool")
    descriptors = [
        _mem_descriptor(ctypes.addressof(memory), size, info=name)
        for (name, size), memory in zip(layout, source_primary, strict=True)
    ]
    source_agent.state = "DONE"
    deposit = source.deposit({key: descriptors})
    assert dict(_poll_until(source, bool))[deposit][key].success
    source_agent.state = "PROC"

    target.submit_hint(_router_hint("tcp://source:1"), request_id="req")
    fetch = target.fetch((key,), "req", expected_layout=expected_layout)
    _wait_until(lambda: bool(target_control.sent))
    source_control.incoming.extend(message for _, message in target_control.sent)
    if success:
        _poll_until(source, lambda _: len(source_agent.xfers) == 2)
        source_xfer = source_agent.xfers[1]
        assert len(source_xfer[1]) == len(source_xfer[3]) == 2
        notification = source_xfer[5]
        source_agent.state = "DONE"
        _poll_until(source, lambda _: not _has_outstanding_operations(source))
    else:
        _poll_until(source, lambda _: bool(source_agent.sent_notifs))
        notification = source_agent.sent_notifs[0][1]
        assert "start_write layout mismatch" in caplog.text
    target_agent.notifs["source"] = [notification]
    result = dict(_poll_until(target, bool))[fetch][key]
    assert result.success is success
    if success:
        assert [
            (descriptor.info, descriptor.size)
            for descriptor in result.descriptors or ()
        ] == layout
        assert {name: memory.raw for name, memory in target_local.items()} == pool_data


def test_remote_staging_commits_available_keys() -> None:
    block_size = 16
    local = ctypes.create_string_buffer(block_size * 3)
    agent = FakeNixlAgent()
    control = FakeBytesControl()
    events: list[InventoryEvent] = []
    keys = tuple(BlockKey(f"k{i}".encode()) for i in range(3))
    target = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        control,
        key_adapter=_ConstantHashAdapter(),
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
        local_dram=LocalDramOptions([("", ctypes.addressof(local), len(local))]),
        inventory_sink=events.append,
    )
    target.submit_hint(_router_hint("tcp://source:1"), request_id="req")
    fetch = target.fetch(keys, request_id="req")
    _wait_until(lambda: bool(control.sent))
    message = _decode_control_message(control.sent[0][1])
    agent.notifs["source"] = [
        _write_done_notification(message["op_handle"], completed_indices=(0, 2))
    ]

    completed = _poll_until(target, lambda results: bool(results))
    assert len(completed) == 1 and completed[0][0] == fetch
    results = completed[0][1]
    available = (keys[0], keys[2])
    release_handles = []
    for key in keys:
        if key in available:
            assert results[key].success
            assert results[key].descriptors is not None
            assert results[key].release_handle is not None
            release_handles.append(results[key].release_handle)
        else:
            assert results[key] == OpEntryResult(OpEntryStatus.FAILED)
    assert target.query(keys, "req") == [
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
        (QueryStatus.MISS, None),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]
    assert events == [InventoryEvent(available, CacheTier.LOCAL_G2, False)]
    assert target.release(release_handles) == [
        (handle, True) for handle in release_handles
    ]


@pytest.mark.parametrize(
    ("completed_indices", "success"),
    [((-1,), True), ((2,), True), ((True,), True), ((0, 0), True), ((0, 1), "false")],
)
def test_remote_completion_rejects_invalid_notification(
    completed_indices, success
) -> None:
    agent = FakeNixlAgent()
    control = FakeBytesControl()
    target = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        control,
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
    )
    keys = (BlockKey(b"k0"), BlockKey(b"k1"))
    target.submit_hint(_router_hint("tcp://source:1"), request_id="req")
    op_handle = target.deliver(
        {key: [_mem_descriptor()] for key in keys}, request_id="req"
    )
    _wait_until(lambda: bool(control.sent))
    agent.notifs["source"] = [
        _write_done_notification(
            op_handle, completed_indices=completed_indices, success=success
        )
    ]
    assert _poll_until(target, bool) == [
        (op_handle, _op_entries(dict.fromkeys(keys, False)))
    ]
    assert not _has_outstanding_operations(target)


@pytest.mark.parametrize(
    "resolution", ["queued", "late", "notification", "probe", "guard", "unknown"]
)
def test_remote_fetch_timeout_keeps_slot_until_source_is_terminal(
    resolution,
) -> None:
    errors = []
    now = 0.0
    block_size = 16
    local = ctypes.create_string_buffer(block_size)
    agent = FakeNixlAgent()
    control = FakeBytesControl()
    key, replacement = BlockKey(b"k0"), BlockKey(b"k1")
    target = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        control,
        KVCRConfig(
            nixl_agent_name="target",
            pool_layouts=[("", 16)],
            operation_timeout_ms=1000,
            abandon_timeout_ms=2000,
        ),
        on_resilience_event=errors.append,
        key_adapter=_ConstantHashAdapter(),
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
        local_dram=LocalDramOptions([("", ctypes.addressof(local), len(local))]),
    )
    target._core._clock = lambda: now
    target.submit_hint(_router_hint("tcp://source:1"), request_id="req")
    fetch = target.fetch((key,), request_id="req")
    _wait_until(lambda: bool(control.sent))
    message = _decode_control_message(control.sent[0][1])

    progress = target._core._progress
    _wait_until(lambda: ("target", message["op_handle"]) in progress._in_flight_ops)
    operation = progress._in_flight_ops[("target", message["op_handle"])]
    assert not operation.close(progress)
    with pytest.raises(RuntimeError, match="unresolved operations"):
        progress._close_nixl()
    assert agent.deregistered == []
    if resolution != "unknown":
        control.incoming.append(
            msgspec.msgpack.encode(
                {
                    "type": "target_metadata_ack",
                    "sender_control_endpoint": "tcp://source:1",
                    "sender_incarnation": "source",
                    "op_handle": message["op_handle"],
                }
            )
        )
        _wait_until(lambda: operation.source_incarnation == "source")
    if resolution == "queued":
        # Progress accepts success before expiry; main consumes it after expiry.
        agent.notifs["source"] = [_write_done_notification(message["op_handle"])]
        _wait_until(lambda: not target._core._progress._completed.empty())

    now = 1.5  # Past T, but still inside the destination's grace period.
    assert _poll_until(target, lambda completed: bool(completed)) == [
        (fetch, _op_entries({key: False}))
    ]
    if resolution != "queued":
        assert target.query((key,), "req") == [
            (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
        ]
        assert _has_outstanding_operations(target)
        assert not operation.close(progress)
        if resolution != "late":
            now = 2.0
            assert _poll_until(target, lambda _: bool(errors)) == []
            assert [error.state for error in errors] == ["uncertain"]
            control.incoming.append(
                _probe_ack(
                    message["op_handle"],
                    sender_incarnation="new",
                    terminal=True,
                    dead_incarnation="source" if resolution == "unknown" else None,
                )
            )
            _wait_until(lambda: not control.incoming)
            now = 100.0  # Time and a replacement incarnation cannot free the slot.
            assert list(target.poll_completed()) == []
        blocked = target.deposit({replacement: [_mem_descriptor(size=block_size)]})
        assert list(target.poll_completed()) == [
            (blocked, _op_entries({replacement: False}))
        ]

        if resolution in ("probe", "guard"):
            fields = {"terminal": True}
            if resolution == "guard":
                fields.update(sender_incarnation="guard", dead_incarnation="source")
            control.incoming.append(_probe_ack(message["op_handle"], **fields))
        else:
            agent.notifs["source"] = [_write_done_notification(message["op_handle"])]
        assert (
            _poll_until(target, lambda _: not _has_outstanding_operations(target)) == []
        )
    if resolution not in ("queued", "late"):
        assert [error.state for error in errors] == ["uncertain", "quiesced"]
        assert all(error.op_handle == message["op_handle"] for error in errors)
        assert errors[0].destination_regions == errors[1].destination_regions
    assert not _has_outstanding_operations(target)
    assert key not in target._core._block_record_map
    assert operation.close(progress)

    # The terminal completion makes the single slot reusable.
    primary = ctypes.create_string_buffer(b"a" * block_size, block_size)
    agent.state = "DONE"
    deposit = target.deposit(
        {replacement: [_mem_descriptor(ctypes.addressof(primary), block_size)]}
    )
    assert _poll_until(target, bool) == [(deposit, _op_entries({replacement: True}))]
    assert local.raw == primary.raw


def test_kvcr_deliver_propagates_source_pin_miss():
    target_agent = FakeNixlAgent(metadata=b"target-md")
    source_agent = FakeNixlAgent(metadata=b"source-md")
    target_control = FakeBytesControl("tcp://target:1")
    source_control = FakeBytesControl("tcp://source:1")
    source_pinning = FakePrimaryPinning(prefix_length=0)
    target = _new_kvcr(
        target_agent,
        FakePrimaryPinning(),
        target_control,
        KVCRConfig(nixl_agent_name="target", pool_layouts=[("", 16)]),
        key_adapter=_ConstantHashAdapter(),
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
    )
    source = _new_kvcr(
        source_agent,
        source_pinning,
        source_control,
        KVCRConfig(nixl_agent_name="source", pool_layouts=[("", 16)]),
        name="source",
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
    )
    key = _make_block_key(b"k0", 0)
    other_group = _make_block_key(b"k0", 1)

    target.submit_hint(_router_hint("tcp://source:1"), request_id="req")
    op_handle = target.deliver(
        {key: [_mem_descriptor()]},
        request_id="req",
    )
    _wait_until(lambda: bool(target_control.sent))
    source_control.incoming.extend(message for _, message in target_control.sent)

    assert _poll_until(source, lambda _: bool(source_agent.sent_notifs)) == []
    assert source_pinning.searches == [(key,)]
    assert source_agent.xfers == []
    assert not any(
        _decode_control_message(message).get("type") == "write_done"
        for _, message in source_control.sent
    )
    assert len(source_agent.sent_notifs) == 1
    assert source_agent.sent_notifs[0][0] == b"remote-1"
    assert _decode_notif(source_agent.sent_notifs[0][1]) == {
        "type": "write_done",
        "op_handle": op_handle,
        "success": True,
        "completed_indices": [],
    }

    target_agent.notifs["source"] = [source_agent.sent_notifs[0][1]]
    assert _poll_until(target, lambda completed: bool(completed)) == [
        (op_handle, _op_entries({key: False}))
    ]
    assert target.query((key, other_group), "req") == [
        (QueryStatus.MISS, None),
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2),
    ]
    assert _poll_until(source, lambda _: not _has_outstanding_operations(source)) == []
    assert not source._core._remote_fw_dram._dangling_ops.source_writes

    target_control.sent = []
    retry_handle = target.deliver(
        {key: [_mem_descriptor()]},
        request_id="req",
    )
    assert list(target.poll_completed()) == [(retry_handle, _op_entries({key: False}))]
    assert target_control.sent == []

    other_destination = [_mem_descriptor(addr=256)]
    mixed_handle = target.deliver(
        {key: [_mem_descriptor()], other_group: other_destination}, request_id="req"
    )
    _wait_until(lambda: bool(target_control.sent))
    message = _decode_control_message(target_control.sent[0][1])
    assert message["keys"] == [other_group]
    assert message["dst_descriptors"] == msgspec.to_builtins([other_destination])
    target_agent.notifs["source"] = [_write_done_notification(mixed_handle)]
    assert _poll_until(target, bool) == [
        (mixed_handle, _op_entries({key: False, other_group: True}))
    ]
    assert target.query((key,), "req") == [(QueryStatus.MISS, None)]
    assert not _has_outstanding_operations(target)

    target.submit_hint(_router_hint("tcp://source:1"), request_id="other")
    assert target.query((key,), "other") == [
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
    ]
    target.submit_hint(_router_hint("tcp://source:1"), request_id="req")
    assert target.query((key,), "req") == [(QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)]


def _probe_ack(handle, source="tcp://source:1", **fields):
    return msgspec.msgpack.encode(
        {
            "type": "write_probe_ack",
            "sender_control_endpoint": source,
            "sender_incarnation": "source",
            "op_handle": handle,
            "terminal": False,
            **fields,
        }
    )


def _acked_deliver(control, kvcr, source, key):
    """Drive a deliver whose start_write carries no metadata, and return it."""
    kvcr.submit_hint(_router_hint(source), request_id="metadata")
    _wait_until(lambda: len(control.sent) == 1)
    ack = {
        "type": "target_metadata_ack",
        "sender_control_endpoint": source,
        "target_agent": "source",
        "sender_incarnation": "source",
    }
    control.incoming.append(msgspec.msgpack.encode(ack))
    _wait_until(lambda: not control.incoming)
    control.sent = []

    kvcr.submit_hint(_router_hint(source), request_id="load")
    op_handle = kvcr.deliver({key: [_mem_descriptor()]}, request_id="load")
    _wait_until(lambda: len(control.sent) == 1)
    sent = _decode_control_message(control.sent[-1][1])
    assert sent["type"] == "start_write"
    assert "target_agent_metadata" not in sent
    return op_handle, sent["op_handle"]


def test_only_a_refusal_from_this_operation_s_source_finishes_it():
    """A deliver has no other way out, and no other sender may take it."""
    control = FakeBytesControl()
    kvcr = _new_kvcr(
        FakeNixlAgent(metadata=b"target-md"),
        FakePrimaryPinning(),
        control,
        KVCRConfig(nixl_agent_name="target", pool_layouts=[("", 16)]),
    )
    now = [0.0]
    kvcr._core._clock = lambda: now[0]
    source = "tcp://source:1"
    key = BlockKey(b"k0")
    handle, op_handle = _acked_deliver(control, kvcr, source, key)

    def refuse(sender: str) -> None:
        control.incoming.append(
            msgspec.msgpack.encode(
                {
                    "type": "write_refused",
                    "sender_control_endpoint": sender,
                    "op_handle": op_handle,
                }
            )
        )
        _wait_until(lambda: not control.incoming)

    refuse("tcp://stranger:9999")
    completed = []
    for _ in range(50):
        completed.extend(kvcr.poll_completed())
        time.sleep(0.002)
    assert completed == []

    refuse(source)
    assert _poll_until(kvcr, lambda done: bool(done)) == [
        (handle, _op_entries({key: False}))
    ]


def test_kvcr_metadata_ack_retry_lifecycle():
    agent = FakeNixlAgent(metadata=b"target-md")
    pinning = FakePrimaryPinning()
    control = FakeBytesControl()
    kvcr = _new_kvcr(
        agent,
        pinning,
        control,
        KVCRConfig(nixl_agent_name="target", pool_layouts=[("", 16)]),
    )
    now = [0.0]
    kvcr._core._clock = lambda: now[0]
    source = "tcp://source:1"

    kvcr.submit_hint(_router_hint(source), request_id="metadata")
    _wait_until(lambda: len(control.sent) == 1)
    kvcr.submit_hint(_router_hint(source), request_id="duplicate")
    time.sleep(0.005)
    assert [
        _decode_control_message(message)["type"] for _, message in control.sent
    ] == ["target_metadata"]
    now[0] = 0.1
    kvcr.submit_hint(_router_hint(source), request_id="retry")
    _wait_until(lambda: len(control.sent) == 2)
    assert len(control.sent) == 2
    control.sent = []
    control.incoming.append(
        msgspec.msgpack.encode(
            {"type": "target_metadata_ack", "sender_control_endpoint": source}
        )
    )
    _wait_until(lambda: not control.incoming)

    kvcr.submit_hint(_router_hint(source), request_id="acked")
    time.sleep(0.005)
    assert control.sent == []

    key = BlockKey(b"k0")
    kvcr.submit_hint(_router_hint(source), request_id="load")
    control.send_result = False
    op_handle = kvcr.deliver({key: [_mem_descriptor()]}, request_id="load")
    assert _poll_until(kvcr, lambda completed: bool(completed)) == [
        (op_handle, _op_entries({key: False}))
    ]
    acked_start_write = _decode_control_message(control.sent[-1][1])
    assert acked_start_write["type"] == "start_write"
    assert "target_agent_metadata" not in acked_start_write

    control.send_result = True
    control.sent = []
    kvcr.submit_hint(_router_hint(source), request_id="reconnect")
    _wait_until(lambda: bool(control.sent))
    assert [
        _decode_control_message(message)["type"] for _, message in control.sent
    ] == ["target_metadata"]


@pytest.mark.parametrize("source_responsive", [False, True])
def test_kvcr_deliver_timeout_probes_source_before_finishing(source_responsive):
    errors = []
    now = 0.0
    agent = FakeNixlAgent(metadata=b"target-md")
    control = FakeBytesControl()
    kvcr = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        control,
        KVCRConfig(
            nixl_agent_name="target",
            pool_layouts=[("", 16)],
            operation_timeout_ms=1000,
            abandon_timeout_ms=7000,
        ),
        on_resilience_event=errors.append,
    )
    kvcr._core._clock = lambda: now
    key, source = BlockKey(b"k0"), "tcp://source:1"
    handle, _ = _acked_deliver(control, kvcr, source, key)
    control.sent.clear()
    control.send = Mock(wraps=control.send, side_effect=[False, DEFAULT])
    now = 1.0
    _wait_until(lambda: control.send.call_count == 2)
    control.send.side_effect = None
    assert list(kvcr.poll_completed()) == []
    if source_responsive:
        agent.notifs["source"] = [
            _write_done_notification(handle, success=False, terminal=False)
        ]
        control.incoming.append(_probe_ack(handle, source))
        _wait_until(lambda: not control.incoming and not agent.notifs)
        assert list(kvcr.poll_completed()) == []
    now = 6.0
    time.sleep(0.01)
    assert list(kvcr.poll_completed()) == []
    now = 7.0
    assert _poll_until(kvcr, bool) == [(handle, _op_entries({key: False}))]
    assert control.send.call_count == 3  # Retry at T, then cleanup at abandonment.
    assert [error.state for error in errors] == ["uncertain"]
    assert errors[0].op_handle == handle

    # An abandoned operation must not blacklist the endpoint for fresh work.
    kvcr.submit_hint(_router_hint(source), request_id="retry")
    retry = kvcr.deliver({key: [_mem_descriptor()]}, request_id="retry")
    _wait_until(
        lambda: any(
            _decode_control_message(raw).get("op_handle") == retry
            for _, raw in control.sent
        )
    )
    agent.notifs["source"] = [_write_done_notification(retry)]
    assert _poll_until(kvcr, bool) == [(retry, _op_entries({key: True}))]

    control.incoming.append(
        _probe_ack(handle, source, sender_incarnation="guard", terminal=True)
    )
    _wait_until(lambda: not control.incoming)
    now = 100.0
    assert list(kvcr.poll_completed()) == []
    assert _has_outstanding_operations(kvcr)
    assert len(errors) == 1
    control.incoming.append(
        _probe_ack(
            handle, source, sender_incarnation="guard", dead_incarnation="source"
        )
    )
    assert _poll_until(kvcr, lambda _: len(errors) == 2) == []
    assert not _has_outstanding_operations(kvcr)
    assert [error.state for error in errors] == ["uncertain", "quiesced"]
    assert errors[0].destination_regions == errors[1].destination_regions


@pytest.mark.parametrize(
    "outcome",
    [
        "cancelled",
        "cancelled_same_poll",
        "failed_first",
        "failed_last",
        "terminal",
        "log",
        "raise",
        "shutdown",
        "shutdown_probe",
    ],
)
def test_remote_write_cancellation_and_late_completion(
    outcome,
    caplog,
):
    caplog.set_level(logging.INFO, logger="kvcr.core")
    shutting_down = outcome.startswith("shutdown")
    errors = []

    def on_resilience_event(error):
        errors.append(error)
        raise error

    now = 0.0
    descriptor = _mem_descriptor()
    agent, control = FakeNixlAgent(), FakeBytesControl()
    kvcr = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        control,
        KVCRConfig(
            nixl_agent_name="target", pool_layouts=[("", 16)], operation_timeout_ms=1000
        ),
        on_resilience_event=on_resilience_event if outcome == "raise" else None,
    )
    kvcr._core._clock = lambda: now
    key, source = BlockKey(b"k0"), "tcp://source:1"
    handle, _ = _acked_deliver(control, kvcr, source, key)
    if outcome in ("cancelled", "cancelled_same_poll", "failed_first", "failed_last"):
        # The source can cancel before the target's own timeout.
        now = 0.5
        cancelled = _write_done_notification(
            handle, success=False, terminal=outcome in ("failed_first", "failed_last")
        )
        notifications = [_write_done_notification(handle)]
        if outcome == "cancelled":
            agent.notifs["source"] = [cancelled]
            _wait_until(lambda: not agent.notifs)
            assert list(kvcr.poll_completed()) == []
        elif outcome == "failed_first":
            notifications.insert(0, cancelled)
        else:
            notifications.append(cancelled)
        agent.notifs["source"] = notifications
        assert _poll_until(kvcr, bool) == [(handle, _op_entries({key: False}))]
        return
    now = 1.0
    _wait_until(
        lambda: any(
            _decode_control_message(raw).get("type") == "write_probe"
            for _, raw in control.sent
        )
    )
    if outcome == "terminal":
        # The Guard's reply and a completed write may be observed in one poll.
        # The destination is still held, so this is not a late write.
        def guard_reply():
            control.recv = lambda: []
            agent.notifs["source"] = [
                _write_done_notification(handle, success=False),
                _write_done_notification(handle),
                _write_done_notification(handle, success=False),
            ]
            return [
                _probe_ack(
                    handle,
                    source,
                    sender_incarnation="guard",
                    dead_incarnation="source",
                ),
            ]

        control.recv = guard_reply
        assert _poll_until(kvcr, bool) == [(handle, _op_entries({key: False}))]
        return
    now = 5.0
    if outcome == "raise":
        with pytest.raises(RuntimeError):
            _poll_until(kvcr, bool)
        assert [error.state for error in errors] == ["uncertain"]
    assert _poll_until(kvcr, bool) == [(handle, _op_entries({key: False}))]
    kvcr.submit_hint(_router_hint(source), request_id="retry")
    other_handle = kvcr.deliver({key: [_mem_descriptor()]}, request_id="retry")
    _wait_until(
        lambda: any(
            _decode_control_message(raw).get("op_handle") == other_handle
            for _, raw in control.sent
        )
    )
    notifications = [
        _write_done_notification(handle, success=False, terminal=False),
        _write_done_notification(handle),
        _write_done_notification(handle, success=False, terminal=False),
        _write_done_notification(other_handle),
    ]
    if shutting_down:
        # Drain framework jobs, then resolve the old quarantine during close.
        agent.notifs["source"] = [_write_done_notification(other_handle)]
        assert _poll_until(kvcr, bool) == [(other_handle, _op_entries({key: True}))]
        with ThreadPoolExecutor(max_workers=1) as executor:
            closing = executor.submit(kvcr.close)
            _wait_until(lambda: kvcr._core._progress._startup_stage == "cleanup")
            if outcome == "shutdown_probe":

                def guard_reply(_endpoint, message):
                    if _decode_control_message(message)["type"] == "write_probe":
                        control.incoming.append(
                            _probe_ack(
                                handle,
                                source,
                                sender_incarnation="guard",
                                dead_incarnation="source",
                            )
                        )
                    return True

                control.send = guard_reply
                now = 6.0
            else:
                agent.notifs["source"] = notifications
            closing.result(timeout=6)
        assert kvcr._core.is_quiescent()
    else:
        agent.notifs["source"] = notifications
    if outcome == "raise":
        with pytest.raises(RuntimeError):
            _poll_until(kvcr, bool)
    # Even a raising handler must not kill progress or lose another completion.
    if not shutting_down:
        assert _poll_until(kvcr, bool) == [(other_handle, _op_entries({key: True}))]
    if outcome == "log" or shutting_down:
        errors = [
            record.args[0]
            for record in caplog.records
            if record.args and getattr(record.args[0], "op_handle", None) == handle
        ]
    assert [error.state for error in errors] == ["uncertain", "quiesced"]
    for error in errors:
        assert error.op_handle == handle
        assert error.destination_regions == [descriptor]
        assert error.source_blocks is None
    kvcr.close()


def test_kvcr_target_ignores_unknown_op_handle_notification():
    """A write_done for an op_handle we never issued is dropped, not applied."""
    agent = FakeNixlAgent(metadata=b"target-md")
    control = FakeBytesControl()
    kvcr = _new_kvcr(agent, FakePrimaryPinning(), control)
    agent.notifs["source"] = [
        b"KVCR:"
        + msgspec.msgpack.encode(
            {"type": "write_done", "op_handle": 999, "success": True}
        )
    ]

    time.sleep(0.01)
    assert list(kvcr.poll_completed()) == []


@pytest.mark.parametrize("failure_source", ["control", "notification"])
def test_kvcr_receive_failure_is_logged(kvcr_caplog, failure_source):
    class FailingControl(FakeBytesControl):
        def recv(self) -> list[bytes]:
            raise RuntimeError("receive failed")

    class FailingNotificationAgent(FakeNixlAgent):
        def get_new_notifs(self):
            raise RuntimeError("notification receive failed")

    agent = (
        FailingNotificationAgent(metadata=b"target-md")
        if failure_source == "notification"
        else FakeNixlAgent(metadata=b"target-md")
    )
    control = FailingControl() if failure_source == "control" else FakeBytesControl()
    _new_kvcr(agent, FakePrimaryPinning(), control)
    expected = f"{failure_source} receive failed"
    _wait_until(
        lambda: any(expected in record.getMessage() for record in kvcr_caplog.records)
    )

    assert any(
        expected in record.getMessage()
        for record in kvcr_caplog.records
        if record.levelno == logging.WARNING
    )


def test_kvcr_deliver_fails_closed_on_mixed_hint_sources():
    agent = FakeNixlAgent(metadata=b"target-md")
    control = FakeBytesControl()
    kvcr = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        control,
        KVCRConfig(nixl_agent_name="target", pool_layouts=[("", 16)]),
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
    )
    key_a = _make_block_key(b"block-A", 0)
    key_b = _make_block_key(b"block-B", 0)
    kvcr.submit_hint(_router_hint("tcp://source-A:1"), request_id="req")
    kvcr.submit_hint(_router_hint("tcp://source-B:1"), request_id="req")

    op_handle = kvcr.deliver(
        {key_a: [_mem_descriptor()], key_b: [_mem_descriptor()]}, request_id="req"
    )

    assert list(kvcr.poll_completed()) == [
        (op_handle, _op_entries({key_a: False, key_b: False}))
    ]
    assert control.sent == []


def test_kvcr_deliver_fails_closed_when_hint_source_missing():
    agent = FakeNixlAgent(metadata=b"target-md")
    control = FakeBytesControl()
    kvcr = _new_kvcr(agent, FakePrimaryPinning(), control)
    key = _make_block_key(b"orphan", 0)

    op_handle = kvcr.deliver({key: [_mem_descriptor()]}, request_id="req")

    assert list(kvcr.poll_completed()) == [(op_handle, _op_entries({key: False}))]
    assert control.sent == []


def test_kvcr_request_scoped_sources_do_not_overwrite():
    agent = FakeNixlAgent(metadata=b"target-md")
    control = FakeBytesControl()
    kvcr = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        control,
        KVCRConfig(nixl_agent_name="target", pool_layouts=[("", 16)]),
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
    )
    key = _make_block_key(b"shared-block", 0)

    kvcr.submit_hint(_router_hint("tcp://source-A:1"), request_id="req-a")
    kvcr.submit_hint(_router_hint("tcp://source-B:1"), request_id="req-b")
    op_a = kvcr.deliver({key: [_mem_descriptor()]}, request_id="req-a")
    op_b = kvcr.deliver({key: [_mem_descriptor()]}, request_id="req-b")

    _wait_until(lambda: len(control.sent) == 2)
    assert [endpoint for endpoint, _ in control.sent] == [
        "tcp://source-A:1",
        "tcp://source-B:1",
    ]
    assert {("target", op_a), ("target", op_b)} <= _block_op_ids(kvcr)
    agent.notifs["source"] = [
        _write_done_notification(op_a),
        _write_done_notification(op_b),
    ]
    _poll_until(kvcr, lambda _: not _has_outstanding_operations(kvcr))


@pytest.mark.parametrize(
    ("eager_ctrl_connect", "missing_indices"),
    [
        (False, ()),
        (True, ()),
        (False, (1,)),
        (False, (0,)),
    ],
)
def test_remote_framework_dram_transfers_available_keys(
    eager_ctrl_connect: bool,
    missing_indices: tuple[int, ...],
) -> None:
    target_agent = FakeNixlAgent(metadata=b"target-md")
    source_agent = FakeNixlAgent(metadata=b"source-md")
    source_pinning = FakePrimaryPinning(missing_indices=missing_indices)
    target_control = FakeBytesControl("tcp://target:1")
    source_control = FakeBytesControl("tcp://source:1")
    target = _new_kvcr(
        target_agent,
        FakePrimaryPinning(),
        target_control,
        KVCRConfig(
            nixl_agent_name="target",
            pool_layouts=[("", 16)],
            enable_telemetry=True,
        ),
        key_adapter=_ConstantHashAdapter(),
        remote_options=RemoteFWDramOptions(
            eager_ctrl_connect=eager_ctrl_connect,
        ),
    )
    source = _new_kvcr(
        source_agent,
        source_pinning,
        source_control,
        KVCRConfig(
            nixl_agent_name="source",
            pool_layouts=[("", 16)],
            enable_telemetry=True,
        ),
        name="source",
        remote_options=RemoteFWDramOptions(backend="REMOTE"),
    )
    source._core._clock = lambda: 0.0
    keys = tuple(_make_block_key(b"k0", index) for index in range(3))
    completed_indices = [
        index for index in range(len(keys)) if index not in missing_indices
    ]

    target.submit_hint(_router_hint("tcp://source:1"), request_id="req")
    assert target.query(keys, "req") == [
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2),
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2),
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2),
    ]

    if eager_ctrl_connect:
        _wait_until(lambda: len(target_control.sent) == 1)
        source_control.incoming.extend(message for _, message in target_control.sent)
        target_control.sent = []

    op_handle = target.deliver(
        {
            key: [_mem_descriptor(addr=8192 + index * 16)]
            for index, key in enumerate(keys)
        },
        request_id="req",
    )
    _wait_until(lambda: bool(target_control.sent))
    source_control.incoming.extend(message for _, message in target_control.sent)

    assert _poll_until(source, lambda _: bool(source_agent.xfers)) == []
    assert source_agent.xfer_backends == [["REMOTE"]]
    assert source_agent.xfers[0][3] == [
        (8192 + index * 16, 16, 0) for index in completed_indices
    ]
    notification = source_agent.xfers[0][5]
    assert notification is not None

    source_agent.state = "DONE"
    assert _poll_until(source, lambda _: not _has_outstanding_operations(source)) == []
    assert source_pinning.unpins == ["pin"]

    target_agent.notifs["source"] = [notification]
    assert _poll_until(target, lambda completed: bool(completed)) == [
        (
            op_handle,
            _op_entries(
                {key: index in completed_indices for index, key in enumerate(keys)}
            ),
        )
    ]
    assert target.query(keys, "req") == [
        (QueryStatus.MISS, None)
        if index in missing_indices
        else (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
        for index in range(len(keys))
    ]

    source_stats = source.get_stats()
    target_stats = target.get_stats()
    assert isinstance(source_stats, FakeTelemetryStats)
    assert isinstance(target_stats, FakeTelemetryStats)
    assert (
        "counter",
        TRANSFER_BYTES_METRIC,
        32,
        ("source_write",),
    ) in source_stats.records
    assert (
        "counter",
        TRANSFER_BLOCKS_METRIC,
        len(completed_indices),
        ("remote_deliver",),
    ) in target_stats.records
