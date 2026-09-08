# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Focused tests for composite remote framework-DRAM transfers."""

import ctypes
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _ConstantHashAdapter,
    _decode_control_message,
    _decode_notif,
    _has_outstanding_operations,
    _mem_descriptor,
    _new_kvcr,
    _op_entries,
    _poll_until,
    _router_hint,
    _wait_until,
)

from kvcr.config import FrameworkDramInput, KVCRConfig, LocalDramOptions
from kvcr.remote_fw_dram import (
    _message_descriptor_counts,
    _RemoteFWDram,
    _RequestHint,
    _SourcePinOp,
    _SourceWriteState,
)
from kvcr.types import BlockKey, MemDescriptor


def _descriptor(addr: int, size: int, pool: str) -> MemDescriptor:
    return MemDescriptor("worker", "DRAM", addr, size, 0, pool)


def test_target_flattens_bundles_and_sends_logical_boundaries() -> None:
    backend = object.__new__(_RemoteFWDram)
    progress = Mock()
    kvcr = SimpleNamespace(
        _timer=lambda: 0.0,
        _record_duration=Mock(),
        _add_block_dependencies=Mock(),
        _progress=progress,
    )
    backend._kvcr = kvcr
    backend._request_hints = {"req": _RequestHint("tcp://source:1", frozenset(), 0.0)}
    keys = (BlockKey(b"k0"), BlockKey(b"k1"))
    bundles = {
        keys[0]: (
            _descriptor(100, 8, "pool-a"),
            _descriptor(108, 16, "pool-b"),
        ),
        keys[1]: (_descriptor(200, 8, "pool-a"),),
    }

    assert backend._start_target_pull(bundles, "req", 10.0, 7, local_fill=False)

    op = progress.submit.call_args.args[0]
    assert op.ordered_keys == keys
    assert op.descriptor_counts == (2, 1)
    assert op.dst_descriptors == (*bundles[keys[0]], *bundles[keys[1]])


def test_wire_descriptor_counts_are_backward_compatible_and_strict() -> None:
    assert _message_descriptor_counts({}, key_count=2, descriptor_count=2) == (1, 1)
    assert _message_descriptor_counts(
        {"descriptor_counts": [2, 1]}, key_count=2, descriptor_count=3
    ) == (2, 1)

    for counts in ([1], [2, 0], [2, 2], [True, 2]):
        with pytest.raises(TypeError, match="descriptor counts"):
            _message_descriptor_counts(
                {"descriptor_counts": counts},
                key_count=2,
                descriptor_count=3,
            )
    with pytest.raises(TypeError, match="missing descriptor counts"):
        _message_descriptor_counts({}, key_count=2, descriptor_count=3)


@pytest.mark.parametrize("mismatch", ["count", "layout", "size"])
def test_source_transfers_only_complete_matching_key_prefix(mismatch: str) -> None:
    backend = object.__new__(_RemoteFWDram)
    keys = tuple(BlockKey(f"k{index}".encode()) for index in range(3))
    sources = {
        keys[0]: (
            _descriptor(100, 8, "pool-a"),
            _descriptor(108, 16, "pool-b"),
        ),
        keys[1]: (
            _descriptor(200, 8, "pool-a"),
            _descriptor(208, 16, "pool-b"),
        ),
        keys[2]: (
            _descriptor(300, 8, "pool-a"),
            _descriptor(308, 16, "pool-b"),
        ),
    }
    destinations = [
        _descriptor(1000, 8, "pool-a"),
        _descriptor(1008, 16, "pool-b"),
        _descriptor(2000, 8, "pool-a"),
        _descriptor(2008, 16, "pool-b"),
        _descriptor(3000, 8, "pool-a"),
        _descriptor(3008, 16, "pool-b"),
    ]
    descriptor_counts = (2, 2, 2)
    if mismatch == "count":
        sources[keys[1]] = sources[keys[1]][:1]
    elif mismatch == "layout":
        destinations[3] = _descriptor(2008, 16, "pool-c")
    else:
        destinations[3] = _descriptor(2008, 32, "pool-b")

    progress = Mock()
    release_local = Mock()
    kvcr = SimpleNamespace(
        _claim_local_dram_sources=Mock(return_value=sources),
        _block_record_map={},
        _framework_pin_keys={},
        _release_local_dram_sources=release_local,
        _remove_block_dependencies=Mock(),
        _add_block_dependencies=Mock(),
        _progress=progress,
    )
    backend._kvcr = kvcr
    backend._source_pin_ops = {}
    backend._fw_pins_by_op = {}
    backend._route_generation = {}
    backend._release_framework_pins = Mock()
    op_id = ("source", 1)
    waiting = _SourcePinOp(
        op_id=op_id,
        keys=set(keys),
        started_at=0.0,
        deadline=10.0,
        remote_agent=b"peer",
        op_handle=11,
        ordered_keys=keys,
        dst_descriptors=tuple(destinations),
        descriptor_counts=descriptor_counts,
    )
    backend._source_pin_ops[op_id] = waiting

    backend._submit_prepared_source_write(op_id, waiting)

    submitted = progress.submit.call_args.args[0]
    assert submitted.state is _SourceWriteState.READY_TO_WRITE
    assert submitted.completed_count == 1
    assert submitted.completed_descriptor_count == 2
    assert submitted.src_descriptors == sources[keys[0]]
    assert submitted.dst_descriptors[:2] == tuple(destinations[:2])
    release_local.assert_called_once_with(op_id, {keys[1], keys[2]})

    transfer_progress = Mock()
    transfer_progress.submit_transfer.return_value = (9, True)
    transfer_progress.poll_transfer.return_value = None
    kvcr._clock = lambda: 0.0
    kvcr._timer = lambda: 0.0
    backend._options = SimpleNamespace(backend="UCX")
    backend._telemetry_enabled = False
    backend._record_progress_duration = Mock()
    backend._send_write_done = Mock()

    assert submitted.progress(transfer_progress, None) == (False, True)
    transfer_call = transfer_progress.submit_transfer.call_args
    assert transfer_call.args[1] == sources[keys[0]]
    assert transfer_call.args[2] == tuple(destinations[:2])


def test_framework_pin_residency_keeps_the_descriptor_bundle() -> None:
    backend = object.__new__(_RemoteFWDram)
    key = BlockKey(b"k")
    record = SimpleNamespace(fw_mem=None)
    kvcr = SimpleNamespace(
        _framework_pin_keys={},
        _normalize_descriptors=lambda descriptors: tuple(descriptors),
        _block_record=lambda requested: record if requested == key else None,
    )
    backend._kvcr = kvcr
    backend._try_release_pin = Mock(return_value=True)
    descriptors = [
        _descriptor(100, 8, "pool-a"),
        _descriptor(108, 16, "pool-b"),
    ]

    assert backend._install_framework_pin((key,), ("pin", {key: descriptors})) == "pin"
    assert record.fw_mem.descriptors == tuple(descriptors)
    assert kvcr._framework_pin_keys == {"pin": {key}}


def test_composite_remote_delivery_round_trips_logical_key_boundaries(caplog) -> None:
    source_primary_a = ctypes.create_string_buffer(b"a" * 8)
    source_primary_b = ctypes.create_string_buffer(b"b" * 16)
    source_local_a = ctypes.create_string_buffer(8)
    source_local_b = ctypes.create_string_buffer(16)
    target_a = ctypes.create_string_buffer(8)
    target_b = ctypes.create_string_buffer(16)
    source_agent = FakeNixlAgent(metadata=b"source-md")
    target_agent = FakeNixlAgent(metadata=b"target-md")
    source_control = FakeBytesControl("tcp://source:1")
    target_control = FakeBytesControl("tcp://target:1")
    config = KVCRConfig(
        nixl_agent_name="ignored",
        pool_layouts=[("pool-a", 8), ("pool-b", 16)],
        enable_telemetry=True,
        inventory_report_interval_ms=0,
    )
    caplog.set_level(logging.INFO, logger="kvcr.core")
    source = _new_kvcr(
        source_agent,
        FakePrimaryPinning(),
        source_control,
        config,
        name="source",
        local_dram=LocalDramOptions(
            [
                ("pool-a", ctypes.addressof(source_local_a), len(source_local_a)),
                ("pool-b", ctypes.addressof(source_local_b), len(source_local_b)),
            ]
        ),
    )
    target = _new_kvcr(
        target_agent,
        FakePrimaryPinning(),
        target_control,
        config,
        name="target",
        key_adapter=_ConstantHashAdapter(),
        framework_dram_regions=(
            FrameworkDramInput(ctypes.addressof(target_a), len(target_a)),
            FrameworkDramInput(ctypes.addressof(target_b), len(target_b)),
        ),
    )
    key = BlockKey(b"composite")

    deposit = source.deposit(
        {
            key: [
                _mem_descriptor(ctypes.addressof(source_primary_a), 8, info="pool-a"),
                _mem_descriptor(ctypes.addressof(source_primary_b), 16, info="pool-b"),
            ]
        }
    )
    source_agent.state = "DONE"
    assert _poll_until(source, lambda completed: bool(completed)) == [
        (deposit, _op_entries({key: True}))
    ]
    source_agent.state = "PROC"

    target.submit_hint(_router_hint("tcp://source:1"), request_id="req")
    deliver = target.deliver(
        {
            key: [
                _mem_descriptor(ctypes.addressof(target_a), 8, info="pool-a"),
                _mem_descriptor(ctypes.addressof(target_b), 16, info="pool-b"),
            ]
        },
        request_id="req",
    )
    _wait_until(lambda: bool(target_control.sent))
    message = _decode_control_message(target_control.sent[-1][1])
    assert message["keys"] == [key]
    assert message["descriptor_counts"] == [2]
    source_control.incoming.append(target_control.sent[-1][1])

    assert _poll_until(source, lambda _: len(source_agent.xfers) == 2) == []
    source_write = source_agent.xfers[-1]
    assert source_write[1] == [
        (ctypes.addressof(source_local_a), 8, 0),
        (ctypes.addressof(source_local_b), 16, 0),
    ]
    assert source_write[3] == [
        (ctypes.addressof(target_a), 8, 0),
        (ctypes.addressof(target_b), 16, 0),
    ]
    notification = source_write[5]
    assert _decode_notif(notification)["completed_count"] == 1

    source_agent.state = "DONE"
    assert _poll_until(source, lambda _: not _has_outstanding_operations(source)) == []
    target_agent.notifs["source"] = [notification]
    assert _poll_until(target, lambda completed: bool(completed)) == [
        (deliver, _op_entries({key: True}))
    ]
    messages = [record.getMessage() for record in caplog.records]
    assert (
        f"KVCR native transfer scope=source_write op={message['op_handle']} "
        "result=success blocks=1/1 bytes=24/24"
    ) in messages
    assert (
        f"KVCR native transfer scope=remote_deliver op={deliver} "
        "result=success blocks=1/1 bytes=24/24"
    ) in messages
