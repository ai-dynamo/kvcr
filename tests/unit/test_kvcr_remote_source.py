# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KVCR remote framework-DRAM source-side tests."""

import ctypes
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    FakeTelemetryStats,
    PendingPrimaryPinning,
    _buffer_region,
    _decode_control_message,
    _decode_notif,
    _has_outstanding_operations,
    _invalidate,
    _mem_descriptor,
    _new_kvcr,
    _op_entries,
    _poll_until,
    _recovered_record,
    _start_write_message,
    _wait_until,
)

from kvcr import DURATION_METRIC, TRANSFER_BLOCKS_METRIC, TRANSFER_BYTES_METRIC
from kvcr.backends.local_dram_pools import _LocalDramResidency, _LocalDramState
from kvcr.backends.remote_fw_dram import _RemoteFWDram
from kvcr.backends.remote_ops import _CONTROL_DECODER, _TargetMetadata
from kvcr.backends.remote_sources import _FwMemResidency, _RemoteSources, _SourcePinOp
from kvcr.config import KVCRConfig, LocalDramOptions
from kvcr.runtime.core import _BlockRecord, _KVCRCore
from kvcr.runtime.progress import _STOP, _TransferRef
from kvcr.service.recovery_journal import install_recovery_records
from kvcr.types import BlockKey, PinHandle, PinRequestId, RegionDescriptor


@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO])
@pytest.mark.parametrize("custom_argument", [False, True])
@pytest.mark.parametrize(
    "result,reason",
    [
        ((None, {}), "invalid framework pin result"),
        (("pin", {}), "request_pin returned incomplete descriptors"),
        (("pin", {b"key": None}), "request_pin returned no descriptors"),
        (("pin", {b"key": []}), "descriptor normalization or installation failed"),
    ],
)
def test_framework_pin_rejection_diagnostic(
    result, reason, level, custom_argument, caplog
):
    caplog.set_level(level, logger="kvcr.runtime.core")
    remote = Mock()
    argument = "private callback data"
    if custom_argument:
        argument = MagicMock()
        argument.__eq__.return_value = True
        argument.__str__.return_value = "private callback data"
    remote._kvcr._normalize_descriptors.side_effect = ValueError(argument)
    assert _RemoteSources._install_framework_pin(remote, [b"key"], result) is None
    remote._discard_pin_result.assert_called_once_with(result, [b"key"])
    assert (reason in caplog.text) == (level == logging.DEBUG)
    assert "private callback data" not in caplog.text
    if custom_argument:
        argument.__eq__.assert_not_called()
        argument.__str__.assert_not_called()


def _write_probe_message(op_handle: int, incarnation=None) -> bytes:
    return msgspec.msgpack.encode(
        {
            "type": "write_probe",
            "source_incarnation": incarnation,
            "op_handle": op_handle,
            "target_agent": "target",
            "sender_control_endpoint": "tcp://target:1",
            "source_control_endpoint": "tcp://source:1",
        }
    )


@pytest.mark.parametrize("transfer_state,recovered", [("PROC", False), ("DONE", True)])
def test_local_source_starts_inline_and_holds_its_slot(transfer_state, recovered):
    memory = ctypes.create_string_buffer(b"a" * 16 + b"b" * 16 + b"c" * 16, 48)
    descriptors = [
        _mem_descriptor(i, label=label, end_point_name="source")
        for i, label in enumerate(("", ":b", ":c"))
    ]
    agent, pinning, control = FakeNixlAgent(), FakePrimaryPinning(), FakeBytesControl()
    callbacks = []
    source = _new_kvcr(
        agent,
        pinning,
        control,
        KVCRConfig(
            nixl_agent_name="source",
            pool_layouts=[("", 16)],
            capacity_low_watermark_percent=100,
        ),
        name="source",
        framework_regions=[
            RegionDescriptor(
                addr=ctypes.addressof(memory), size=16, count=3, label=label
            )
            for label in ("", ":*")
        ],
        local_dram=LocalDramOptions([("", ctypes.addressof(memory), len(memory))]),
        capacity_needed_callback=lambda request: callbacks.append(
            (threading.get_ident(), request)
        ),
    )
    key = BlockKey(b"local")
    try:
        agent.state = "DONE"
        if recovered:
            install_recovery_records(
                source._core,
                {
                    key: _BlockRecord(
                        local_dram=_LocalDramResidency(
                            [("", 0), (":b", 1), (":c", 2)],
                            _LocalDramState.READY,
                            refs=tuple(
                                _TransferRef("old-agent", i, label, framework=False)
                                for i, label in enumerate(("", ":b", ":c"))
                            ),
                        )
                    )
                },
            )
        else:
            deposit = source.deposit(
                {key: [descriptors[0], descriptors[2]]},
                expected_layout=["", ":b", ":c"],
            )
            assert _poll_until(source, bool) == [(deposit, _op_entries({key: True}))]
        prior_transfers = len(agent.xfers)
        residency = source._core._block_record_map[key].local_dram
        assert residency.indices_by_label == {"": 0, ":b": 1, ":c": 2}
        request = msgspec.msgpack.decode(_start_write_message(12, key))
        request.update(
            target_regions=(
                {
                    label: RegionDescriptor(
                        addr=4096,
                        size=16,
                        mem_type="VRAM",
                        device_Id=7,
                        count=2,
                        label=label,
                    )
                    for label in ("", ":*")
                },
                {},
            ),
            dst_descriptors=[
                [_TransferRef("target", i, label) for i, label in enumerate((":c", ""))]
            ],
        )
        if not recovered:
            # Ready pieces still read through the caller until the block is complete.
            control.incoming.extend(
                [
                    msgspec.msgpack.encode(dict(request, op_handle=11)),
                    _write_probe_message(99),
                ]
            )
            _wait_until(lambda: bool(control.sent))
            assert len(agent.xfers) == prior_transfers
            _poll_until(
                source,
                lambda _: (
                    len(agent.released_xfers) == prior_transfers + 1
                    and not _has_outstanding_operations(source)
                ),
            )
            assert [
                ctypes.string_at(addr, size) for addr, size, _ in agent.xfers[-1][1]
            ] == [b"c" * 16, b"a" * 16]
            deposit = source.deposit({key: descriptors[1:2]})
            assert _poll_until(source, bool) == [(deposit, _op_entries({key: True}))]
            prior_transfers = len(agent.xfers)
        callbacks.clear()
        agent.state = transfer_state
        with (
            # A resident-source request must not re-enter the submission queue.
            patch.object(source._core._progress, "submit", side_effect=AssertionError),
            source._core._state_lock,
        ):
            control.incoming.append(msgspec.msgpack.encode(request))
            _wait_until(lambda: len(agent.xfers) == prior_transfers + 1)
        if transfer_state == "PROC":
            assert residency.source_readers == 1

        # The peer request must be served without another caller-side poll.
        assert residency.claim_count == 0
        assert [
            ctypes.string_at(addr, size) for addr, size, _ in agent.xfers[-1][1]
        ] == [
            b"c" * 16,
            b"a" * 16,
        ]
        assert agent.xfers[-1][3] == [(4096, 16, 7), (4112, 16, 7)]
        assert pinning.searches == []
        assert not source._core._remote_fw_dram._sources._source_pin_ops
        assert callbacks == []
        if transfer_state == "PROC":
            assert list(source.poll_completed()) == []
            assert callbacks == [(threading.get_ident(), [("", 3)])]

        agent.state = "DONE"
        _wait_until(lambda: len(agent.released_xfers) == prior_transfers + 1)
        assert _poll_until(source, lambda _: residency.source_readers == 0) == []
        assert not source._core._local_dram_sources_by_op

        # The caller-queued path reuses the same parts without pinning the framework.
        request["op_handle"] = 13
        with patch.object(
            source._core._remote_fw_dram, "_try_local_source_write", return_value=False
        ):
            control.incoming.append(msgspec.msgpack.encode(request))
            _poll_until(
                source,
                lambda _: (
                    len(agent.released_xfers) == prior_transfers + 2
                    and residency.claim_count == 0
                ),
            )
        assert agent.xfers[-1][1] == agent.xfers[-2][1]
        assert pinning.searches == []
    finally:
        agent.state = "DONE"
        source.close()


@pytest.mark.parametrize("ready_source", [False, True])
def test_source_invalidation_preserves_admitted_reads(ready_source, request):
    primary = ctypes.create_string_buffer(b"a" * 16, 16)
    local = ctypes.create_string_buffer(16)
    agent, pinning, control = (
        FakeNixlAgent(),
        PendingPrimaryPinning(),
        FakeBytesControl(),
    )
    request.addfinalizer(lambda: setattr(agent, "state", "DONE"))
    source = _new_kvcr(
        agent,
        pinning,
        control,
        name="source",
        framework_regions=[_buffer_region(primary, 16)],
        local_dram=LocalDramOptions([("", ctypes.addressof(local), 16)]),
    )
    core = source._core
    key, other = BlockKey(b"removing"), BlockKey(b"other")
    if ready_source:
        agent.state = "DONE"
        deposit = source.deposit({key: [_mem_descriptor(end_point_name="source")]})
        assert dict(_poll_until(source, bool))[deposit][key].success
        agent.state = "PROC"

    control.incoming.append(_start_write_message(1, key))
    if ready_source:
        _wait_until(lambda: len(agent.xfers) == 2)
        assert core._block_record_map[key].local_dram.source_readers == 1
    else:
        # This read is admitted before it has any local source claim.
        _poll_until(source, lambda _: pinning.searches == [(key,)])
        deposit = source.deposit({key: [_mem_descriptor(end_point_name="source")]})
        _wait_until(lambda: len(agent.xfers) == 1)
        assert not core._local_dram_sources_by_op
    _invalidate(source, (key,))
    assert core._block_record_map[key].pending_invalidation

    control.incoming.append(_start_write_message(2, key))
    mixed = msgspec.msgpack.decode(_start_write_message(3, key))
    mixed.update(
        keys=[key, other],
        dst_descriptors=[[_TransferRef("target", index)] for index in range(2)],
    )
    control.incoming.append(msgspec.msgpack.encode(mixed))
    expected_searches = [] if ready_source else [(key,)]
    expected_searches.append((other,))
    _poll_until(
        source,
        lambda _: pinning.searches == expected_searches and bool(agent.sent_notifs),
    )
    assert [_decode_notif(payload) for _, payload in agent.sent_notifs] == [
        {"type": "write_done", "op_handle": 2, "success": False}
    ]
    # The rejected key neither joins its older pin request nor asks for a new pin.
    assert pinning.searches == expected_searches
    pinning.complete(len(expected_searches) - 1, "pin-other")
    agent.state = "DONE"
    completed = dict(
        _poll_until(
            source,
            lambda completed: (
                (ready_source or any(handle == deposit for handle, _ in completed))
                and any(
                    xfer[5] and _decode_notif(xfer[5])["op_handle"] == 3
                    for xfer in agent.xfers
                )
            ),
        )
    )
    if not ready_source:
        assert completed[deposit][key].success
        assert core._block_record_map[key].pending_invalidation
        # A missing framework result resumes the older read against G2, even
        # though it acquires its first local source claim after invalidation.
        pinning.complete(0, "pin-missing", missing_indices=(0,))
    _poll_until(source, lambda _: not _has_outstanding_operations(source))

    notifications = {
        _decode_notif(xfer[5])["op_handle"]: _decode_notif(xfer[5])
        for xfer in agent.xfers
        if xfer[5]
    }
    assert notifications[1]["success"]
    assert notifications[1]["completed_indices"] == [0]
    assert notifications[3]["success"]
    assert notifications[3]["completed_indices"] == [1]
    assert len(agent.xfers) == 3  # Deposit, earlier read, admitted mixed-key entry.
    assert not core._invalidation_handler.has_pending()
    assert key not in core._block_record_map
    assert not core._local_dram_sources_by_op
    assert not core._remote_fw_dram._sources._source_pin_ops
    assert not core._remote_fw_dram._dangling_ops.source_writes


@pytest.mark.parametrize(
    "first_piece_ready,mixed", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("timeout", [False, True])
def test_local_source_waits_for_declared_pieces(
    first_piece_ready, timeout, mixed, caplog, request
) -> None:
    primary = ctypes.create_string_buffer(b"a" * 16 + b"b" * 16, 32)
    local = ctypes.create_string_buffer(32)
    agent, pinning, control = FakeNixlAgent(), FakePrimaryPinning(), FakeBytesControl()
    request.addfinalizer(lambda: setattr(agent, "state", "DONE"))
    source = _new_kvcr(
        agent,
        pinning,
        control,
        name="source",
        framework_regions=[
            _buffer_region(primary, 16),
            _buffer_region(primary, 16, ":*"),
        ],
        local_dram=LocalDramOptions([("", ctypes.addressof(local), 32)]),
    )
    now = 0.0
    source._core._clock = lambda: now
    backend = source._core._remote_fw_dram
    key = BlockKey(b"partial")
    layout = [":a", ":b"]
    first = source.deposit(
        {key: [_mem_descriptor(0, label=":a", end_point_name="source")]},
        expected_layout=layout,
    )
    _wait_until(lambda: len(agent.xfers) == 1)
    if first_piece_ready:
        agent.state = "DONE"
        assert dict(_poll_until(source, bool))[first][key].success
        agent.state = "PROC"

    message = msgspec.msgpack.decode(_start_write_message(12, key))
    rejected = (BlockKey(b"wrong-before"), BlockKey(b"wrong-after")) if mixed else ()
    keys = [rejected[0], key, rejected[1]] if mixed else [key]
    message.update(
        keys=keys,
        key_layout=layout,
        target_regions=({":*": _buffer_region(primary, 16, ":*")}, {}),
        dst_descriptors=[
            [_TransferRef("target", i, label) for i, label in enumerate(layout)]
            for _ in keys
        ],
    )
    control.incoming.append(msgspec.msgpack.encode(message))
    _poll_until(source, lambda _: bool(backend._sources._source_pin_ops))
    assert len(agent.xfers) == 1
    residency = source._core._block_record_map[key].local_dram
    producer_claims = int(not first_piece_ready)
    if mixed:
        _poll_until(
            source,
            lambda _: pinning.searches and not backend._sources._pending_pin_ops,
        )
        assert pinning.unpins == ["pin"]
        for rejected_key in rejected:
            assert rejected_key not in source._core._block_record_map
        for _ in range(3):
            assert source.poll_completed() == []
    assert residency.claim_count == producer_claims + 1

    if timeout:
        now = 2.0
        _poll_until(source, lambda _: bool(agent.sent_notifs))
        notification = agent.sent_notifs[-1][1]
        _poll_until(source, lambda _: not source._core._local_dram_sources_by_op)
        assert residency.claim_count == producer_claims
        assert len(agent.xfers) == 1
        agent.state = "DONE"
        _poll_until(source, lambda _: not _has_outstanding_operations(source))
    else:
        second = source.deposit(
            {key: [_mem_descriptor(1, label=":b", end_point_name="source")]}
        )
        _wait_until(lambda: len(agent.xfers) == 2)
        agent.state = "DONE"
        completed = dict(
            _poll_until(source, lambda _: not _has_outstanding_operations(source))
        )
        assert completed[second][key].success
        assert len(agent.xfers) == 3
        assert [
            ctypes.string_at(addr, size) for addr, size, _ in agent.xfers[-1][1]
        ] == [b"a" * 16, b"b" * 16]
        notification = agent.xfers[-1][5]
        assert _decode_notif(notification)["completed_indices"] == [1 if mixed else 0]
    assert _decode_notif(notification)["success"] is not timeout
    assert not backend._sources._source_pin_ops
    assert not source._core._local_dram_sources_by_op
    assert residency.claim_count == 0
    assert pinning.searches == ([rejected] if mixed else [])
    assert [
        record.args[1]
        for record in caplog.records
        if "start_write layout mismatch" in record.message
    ] == list(rejected)


@pytest.mark.parametrize(
    "outcome", ["write", "framework", "deadline", "cancel", "route", "shutdown"]
)
def test_local_source_retries_after_reader_contention(outcome):
    memory = ctypes.create_string_buffer(16)
    descriptor = _mem_descriptor(end_point_name="source")
    agent, pinning, control = FakeNixlAgent(), FakePrimaryPinning(), FakeBytesControl()
    source = _new_kvcr(
        agent,
        pinning,
        control,
        name="source",
        local_dram=LocalDramOptions([("", ctypes.addressof(memory), len(memory))]),
        framework_regions=[RegionDescriptor(addr=ctypes.addressof(memory), size=16)],
    )
    now = 0.0
    source._core._clock = lambda: now
    local = source._core._local_dram._pools
    backend = source._core._remote_fw_dram
    key = BlockKey(b"local")
    requested_key = BlockKey(b"framework-only") if outcome == "framework" else key
    try:
        agent.state = "DONE"
        deposit = source.deposit({key: [descriptor]})
        assert _poll_until(source, bool) == [(deposit, _op_entries({key: True}))]
        agent.state = "PROC"
        with local._source_state.lock:
            control.incoming.extend(
                [
                    _start_write_message(12, requested_key, target_agent="target"),
                    _write_probe_message(99),
                ]
            )
            _wait_until(
                lambda: any(
                    _decode_control_message(raw).get("type") == "write_probe_ack"
                    for _, raw in control.sent
                )
            )
            _wait_until(lambda: bool(backend._pending_local_sources))
            assert len(agent.xfers) == 1
            assert source._core._block_record_map[key].local_dram.claim_count == 0
            if outcome == "deadline":
                now = 2.0
            elif outcome == "cancel":
                control.incoming.append(_write_probe_message(12))
            elif outcome == "route":
                backend._route_generation["target"] = 1
            elif outcome == "shutdown":
                source._core._progress._submissions.put(_STOP)
            if outcome not in ("write", "framework"):
                _wait_until(lambda: bool(agent.sent_notifs))
                assert _decode_notif(agent.sent_notifs[0][1]) == {
                    "type": "write_done",
                    "op_handle": 12,
                    "success": False,
                }
                assert len(agent.xfers) == 1

        # Ready local sources retry without another caller-side poll.
        if outcome == "write":
            _wait_until(lambda: len(agent.xfers) == 2)
            assert source._core._block_record_map[key].local_dram.source_readers == 1
            assert local.telemetry_state()["local_g2_evictable_slots"] == 0
        elif outcome == "framework":
            assert _poll_until(source, lambda _: len(agent.xfers) == 2) == []
        assert pinning.searches == (
            [(requested_key,)] if outcome == "framework" else []
        )
        agent.state = "DONE"
        _poll_until(source, lambda _: not _has_outstanding_operations(source))
        assert len(agent.xfers) == (2 if outcome in ("write", "framework") else 1)
        assert pinning.unpins == (["pin"] if outcome == "framework" else [])
        assert source._core._outstanding_operations == 0
        assert source._core._block_record_map[key].local_dram.claim_count == 0
    finally:
        agent.state = "DONE"
        source.close()
    assert not backend._pending_local_sources
    if outcome not in ("write", "framework"):
        assert (
            sum(_decode_notif(raw)["op_handle"] == 12 for _, raw in agent.sent_notifs)
            == 1
        )


@pytest.mark.parametrize("pin_before_deadline", [True, False])
def test_kvcr_start_write_respects_framework_pin_deadline(
    pin_before_deadline: bool,
    caplog,
) -> None:
    """Source writes must not start after their framework-pin deadline."""
    caplog.set_level(logging.DEBUG, logger="kvcr.runtime.core")
    now = 0.0
    deadline_captured = threading.Event()

    def clock() -> float:
        captured = now
        deadline_captured.set()
        return captured

    source_agent = FakeNixlAgent(metadata=b"source-md")
    pinning = PendingPrimaryPinning()
    control = FakeBytesControl()
    source = _new_kvcr(
        source_agent,
        pinning,
        control,
        KVCRConfig(
            nixl_agent_name="source",
            pool_layouts=[("", 16)],
            operation_timeout_ms=10_000,
            abandon_timeout_ms=20_000,
        ),
        name="source",
    )
    source._core._clock = clock
    key = BlockKey(b"k0")
    control.incoming.append(_start_write_message(9, key, remaining_timeout_ms=100))
    if pin_before_deadline:
        assert (
            _poll_until(
                source,
                lambda _: bool(source._core._remote_fw_dram._sources._pending_pin_ops),
            )
            == []
        )
    else:
        assert deadline_captured.wait(timeout=1)

    now = 2.0
    assert _poll_until(source, lambda _: bool(source_agent.sent_notifs)) == []

    assert source_agent.xfers == []
    assert pinning.searches == ([(key,)] if pin_before_deadline else [])
    assert pinning.cancelled == ([PinRequestId(0)] if pin_before_deadline else [])
    assert caplog.text.count("framework_pin_timeout") == pin_before_deadline
    assert _decode_notif(source_agent.sent_notifs[0][1]) == {
        "type": "write_done",
        "op_handle": 9,
        "success": False,
    }


@pytest.mark.parametrize("incarnation", [None, "matching", "other"])
@pytest.mark.parametrize("before_start", [False, True])
def test_write_probe_fences_write_waiting_on_framework_pin(
    incarnation, before_start
) -> None:
    agent = FakeNixlAgent(metadata=b"source-md")
    pinning = PendingPrimaryPinning()
    control = FakeBytesControl("tcp://source:1")
    source = _new_kvcr(agent, pinning, control, name="source")
    start = _start_write_message(9, BlockKey(b"k0"), target_agent="target")
    if not before_start:
        control.incoming.append(start)
        _poll_until(source, lambda _: bool(pinning.pending))

    if incarnation == "matching":
        incarnation = source._core._remote_fw_dram._dangling_ops.incarnation
    control.incoming.append(_write_probe_message(9, incarnation))
    _wait_until(lambda: bool(control.sent))
    assert _decode_control_message(control.sent[-1][1])["terminal"] is (
        incarnation != "other"
    )

    if before_start:
        if incarnation != "other":
            for metadata in (b"target-md", b"reconnected-target-md"):
                payload = msgspec.msgpack.decode(start)
                payload["target_agent_metadata"] = metadata
                control.incoming.append(msgspec.msgpack.encode(payload))
                _wait_until(lambda: not control.incoming)
                time.sleep(0.01)
                assert list(source.poll_completed()) == []
                assert pinning.searches == []
            assert agent.xfers == []
            return
        control.incoming.append(start)
        _poll_until(source, lambda _: bool(pinning.pending))
    pinning.complete(0)
    if incarnation == "other":
        assert _poll_until(source, lambda _: bool(agent.xfers)) == []
        agent.state = "DONE"
    assert _poll_until(source, lambda _: bool(pinning.unpins)) == []
    assert len(agent.xfers) == (1 if incarnation == "other" else 0)


@pytest.mark.parametrize("log_level", [logging.INFO, logging.DEBUG])
def test_stalled_source_refuses_queued_and_future_writes(caplog, log_level) -> None:
    caplog.set_level(log_level, logger="kvcr.runtime.core")
    errors = []

    def on_resilience_event(error):
        errors.append(error)
        raise error

    agent = FakeNixlAgent(metadata=b"source-md")
    pinning = PendingPrimaryPinning()
    control = FakeBytesControl()
    source = _new_kvcr(
        agent, pinning, control, name="source", on_resilience_event=on_resilience_event
    )

    stalled_for = 0.0
    with patch(
        "kvcr.backends.dangling_ops.time",
        SimpleNamespace(monotonic=lambda: time.monotonic() + stalled_for),
    ):
        for handle in (1, 2):
            control.incoming.append(
                _start_write_message(handle, BlockKey(b"k0"), target_agent="target")
            )
            if handle == 1:
                _poll_until(source, lambda _: bool(pinning.searches))
                stalled_for = 2.0
                pinning.complete(0)
                with pytest.raises(RuntimeError, match="source progress stalled"):
                    _poll_until(source, lambda _: bool(errors))
            assert _poll_until(source, lambda _: len(agent.sent_notifs) == handle) == []
            assert not _decode_notif(agent.sent_notifs[-1][1])["success"]
    assert agent.xfers == []
    assert pinning.searches == [(BlockKey(b"k0"),)]
    assert pinning.unpins == ["pin"]
    assert len(errors) == 1  # A raising callback must not stop native cleanup.
    assert "timeout: 1000 ms" in str(errors[0])
    assert "new source writes disabled until restart" in str(errors[0])
    refusals = [m for m in caplog.messages if "source_transfer_refused " in m]
    assert bool(refusals) is (log_level == logging.DEBUG)
    if refusals:
        assert "reason=source_stalled" in refusals[0]


def test_kvcr_close_cleans_pending_pin_operations():
    agent = FakeNixlAgent(metadata=b"source-md")
    pinning = PendingPrimaryPinning()
    control = FakeBytesControl()
    key = BlockKey(b"k0")
    control.incoming.append(_start_write_message(1, key))

    source = _new_kvcr(agent, pinning, control, name="source")
    assert (
        _poll_until(
            source,
            lambda _: bool(source._core._remote_fw_dram._sources._pending_pin_ops),
        )
        == []
    )
    assert source._core._remote_fw_dram._sources._pending_pin_ops

    # The second request is accepted but has not reached main-thread pinning.
    control.incoming.append(_start_write_message(2, key))
    _wait_until(
        lambda: len(source._core._remote_fw_dram._dangling_ops.source_writes) == 2
    )
    notifications_at_close = []
    control.close = lambda: notifications_at_close.extend(
        _decode_notif(raw) for _, raw in agent.sent_notifs
    )
    source.close()

    assert notifications_at_close == [
        {"type": "write_done", "op_handle": handle, "success": False}
        for handle in (1, 2)
    ]
    assert agent.xfers == []
    assert pinning.cancelled == [PinRequestId(0)]
    assert not source._core._remote_fw_dram._sources._source_pin_ops
    assert not source._core._remote_fw_dram._sources._pending_pin_ops
    assert not source._core._framework_pin_keys


@pytest.mark.parametrize(
    "field,value",
    [
        ("remaining_timeout_ms", msgspec.UNSET),
        ("remaining_timeout_ms", True),
        ("remaining_timeout_ms", 0),
        ("remaining_timeout_ms", float("nan")),
        ("remaining_timeout_ms", float("inf")),
        ("key_layout", "yes"),
        ("keys", [b"k0", b"k0"]),
        ("keys", []),
        ("keys", ["k0"]),
        ("dst_descriptors", []),
        ("dst_descriptors", "invalid"),
        ("framework", 1),
        ("label", 1),
        ("element_index", -1),
        ("end_point_name", "other"),
    ],
)
def test_kvcr_malformed_start_write_notifies_failure(kvcr_caplog, field, value):
    source_agent = FakeNixlAgent(metadata=b"source-md")
    control = FakeBytesControl()
    pinning = FakePrimaryPinning()
    _new_kvcr(source_agent, pinning, control, name="source")
    request = msgspec.msgpack.decode(
        _start_write_message(6, BlockKey(b"k0"), target_agent="target")
    )
    if value is msgspec.UNSET:
        request.pop(field)
    elif field in ("framework", "label", "element_index", "end_point_name"):
        descriptors = request["dst_descriptors"][0]
        descriptors.append({**descriptors[0], field: value})
    else:
        request[field] = value
        if field == "keys":
            request["dst_descriptors"] *= len(value)
    control.incoming.append(msgspec.msgpack.encode(request))
    _wait_until(lambda: bool(source_agent.sent_notifs))

    assert not pinning.searches
    assert not source_agent.xfers
    assert _decode_notif(source_agent.sent_notifs[0][1]) == {
        "type": "write_done",
        "op_handle": 6,
        "success": False,
    }
    assert any(
        "malformed start_write" in record.getMessage()
        or "start_write setup failed" in record.getMessage()
        for record in kvcr_caplog.records
        if record.levelno == logging.WARNING
    )


@pytest.mark.parametrize(
    "field,cached,with_metadata",
    [
        ("sender_incarnation", False, False),
        ("sender_incarnation", True, False),
        ("target_regions", True, True),
    ],
)
def test_malformed_peer_preserves_route(kvcr_caplog, field, cached, with_metadata):
    agent, control, pinning = FakeNixlAgent(), FakeBytesControl(), FakePrimaryPinning()
    source = _new_kvcr(agent, pinning, control, name="source")
    tier = source._core._remote_fw_dram
    request = msgspec.msgpack.decode(
        _start_write_message(6, b"k0", target_agent="target")
    )
    if cached:
        control.incoming.append(
            msgspec.msgpack.encode({**request, "type": "target_metadata"})
        )
        _wait_until(lambda: "target" in tier._remote_agents_by_target)
        cached_route = tier._remote_agents_by_target["target"]
    request.update(
        sender_control_endpoint="tcp://target:1",
        source_control_endpoint="tcp://source:1",
    )
    request[field] = 1
    if not with_metadata:
        del request["target_agent_metadata"]
    control.incoming.append(msgspec.msgpack.encode(request))
    _wait_until(lambda: bool(agent.sent_notifs or control.sent))

    assert not pinning.searches
    assert not agent.xfers
    if cached and not with_metadata:
        assert not control.sent
        assert _decode_notif(agent.sent_notifs[0][1]) == {
            "type": "write_done",
            "op_handle": 6,
            "success": False,
        }
    else:
        assert not agent.sent_notifs
        endpoint, raw = control.sent[0]
        reply = _decode_control_message(raw)
        assert endpoint == "tcp://target:1"
        assert reply["type"] == "write_refused"
        assert reply["op_handle"] == 6
        assert reply["sender_control_endpoint"] == "tcp://source:1"
        assert reply.get("reason") == (None if with_metadata else "metadata_required")
    if cached:
        assert tier._remote_agents_by_target["target"] == cached_route
        assert agent.remote_agents == [b"target-md"]
    assert field in kvcr_caplog.text


def test_kvcr_notification_send_failure_is_logged(kvcr_caplog):
    class FailingNotificationAgent(FakeNixlAgent):
        def send_notif(self, agent_name, notif_msg):
            raise RuntimeError("notification failed")

    source_agent = FailingNotificationAgent(metadata=b"source-md")
    control = FakeBytesControl()
    kvcr = _new_kvcr(
        source_agent,
        FakePrimaryPinning(prefix_length=0),
        control,
        name="source",
    )
    control.incoming.append(
        _start_write_message(7, BlockKey(b"k0"), target_agent="target")
    )
    assert (
        _poll_until(
            kvcr,
            lambda _: any(
                "write_done notification failed" in record.getMessage()
                for record in kvcr_caplog.records
            ),
        )
        == []
    )

    assert any(
        "write_done notification failed" in record.getMessage()
        for record in kvcr_caplog.records
        if record.levelno == logging.WARNING
    )


@pytest.mark.parametrize("failure", ["initialize", "error", "exception", "async"])
@pytest.mark.parametrize("local_source", [False, True], ids=["framework", "local"])
def test_kvcr_source_transfer_error_notifies_failure_and_cleans_up(
    failure: str,
    local_source: bool,
) -> None:
    """Every source-transfer failure reports failure and releases its state."""

    class FailingTransferAgent(FakeNixlAgent):
        def initialize_xfer(self, *args, **kwargs):
            if failure == "initialize":
                raise RuntimeError("invalid descriptors")
            return super().initialize_xfer(*args, **kwargs)

        def transfer(self, handle):
            self.transfers.append(handle)
            if failure == "exception":
                raise RuntimeError("ambiguous submission")
            return "PROC" if failure == "async" else "ERR"

    source_agent = FailingTransferAgent(metadata=b"source-md")
    pinning = FakePrimaryPinning()
    control = FakeBytesControl()
    key = BlockKey(b"k0")
    memory = ctypes.create_string_buffer(16)
    kvcr = _new_kvcr(
        source_agent,
        pinning,
        control,
        name="source",
        local_dram=(
            LocalDramOptions([("", ctypes.addressof(memory), len(memory))])
            if local_source
            else None
        ),
    )
    if local_source:
        install_recovery_records(kvcr._core, {key: _recovered_record(g2=[("", 0)])})
        residency = kvcr._core._block_record_map[key].local_dram
    kvcr._core._clock = lambda: 0.0  # Failure must come from ERR, not a timeout.
    control.incoming.append(_start_write_message(5, key))

    if failure == "async":
        assert _poll_until(kvcr, lambda _: bool(source_agent.xfers)) == []
        assert source_agent.sent_notifs == []
        source_agent.state = "ERR"
    if failure != "initialize":
        assert _poll_until(kvcr, lambda _: bool(source_agent.sent_notifs)) == []
        assert _decode_notif(source_agent.sent_notifs[0][1])["terminal"] is False
        assert pinning.unpins == []
        assert source_agent.released_xfers == []
        if local_source:
            assert residency.source_readers == 1
            assert kvcr._core._local_dram._pools.source_operations == 1
        source_agent.state = "DONE"
    completed = _poll_until(
        kvcr,
        lambda _: source_agent.sent_notifs and not _has_outstanding_operations(kvcr),
    )
    assert completed == []

    assert pinning.unpins == ([] if local_source else ["pin"])
    if local_source:
        assert residency.source_readers == 0
        assert kvcr._core._local_dram._pools.source_operations == 0
    assert source_agent.transfers == ([] if failure == "initialize" else [1])
    agent_name, notif = source_agent.sent_notifs[-1]
    assert agent_name == "remote-1"
    assert _decode_notif(notif) == {
        "type": "write_done",
        "op_handle": 5,
        "success": False,
    }
    assert source_agent.released_xfers == ([] if failure == "initialize" else [1])
    assert not _has_outstanding_operations(kvcr)


def test_kvcr_source_ignores_malformed_control_messages():
    """Malformed control payloads never crash the scheduler or cause effects."""
    source_agent = FakeNixlAgent(metadata=b"source-md")
    pinning = FakePrimaryPinning()
    control = FakeBytesControl()
    control.incoming.append(b"\xff\xff\xff not msgpack")
    control.incoming.append(msgspec.msgpack.encode(b"top-level-not-dict"))
    control.incoming.append(msgspec.msgpack.encode({"type": "unknown"}))
    control.incoming.append(_write_probe_message(True))
    for op_handle in (float("inf"), True):
        control.incoming.append(_start_write_message(op_handle, b"invalid"))
    key = BlockKey(b"valid")
    control.incoming.append(_start_write_message(1, key))
    kvcr = _new_kvcr(source_agent, pinning, control, name="source")

    assert _poll_until(kvcr, lambda _: bool(source_agent.xfers)) == []
    assert pinning.searches == [(key,)]
    assert len(source_agent.xfers) == 1
    assert source_agent.sent_notifs == []
    assert pinning.unpins == []
    source_agent.state = "DONE"
    assert _poll_until(kvcr, lambda _: bool(pinning.unpins)) == []
    assert pinning.unpins == ["pin"]


@pytest.mark.parametrize(
    ("terminal_state", "abandon", "raises"),
    [
        ("DONE", False, False),
        ("DONE", True, False),
        ("DONE", True, True),
        ("shutdown", True, False),
    ],
)
def test_kvcr_source_timeout_releases_pins_on_completion_or_abandonment(
    terminal_state,
    abandon,
    raises,
):
    now = 0.0
    agent, pinning, control = FakeNixlAgent(), FakePrimaryPinning(), FakeBytesControl()
    errors = []

    def on_resilience_event(error):
        if not errors:
            assert pinning.unpins == []
        errors.append(error)
        if raises:
            raise RuntimeError("callback failed")

    kvcr = _new_kvcr(
        agent, pinning, control, name="source", on_resilience_event=on_resilience_event
    )
    kvcr._core._clock = lambda: now
    key = BlockKey(b"k0")
    try:
        control.incoming.append(_start_write_message(12, key, target_agent="target"))
        assert _poll_until(kvcr, lambda _: bool(agent.xfers)) == []
        source_handle = next(iter(kvcr._core._progress._in_flight_ops))[1]
        now = 1.5  # A late first poll must not restart the abandonment deadline.
        _wait_until(lambda: bool(agent.sent_notifs))
        assert _decode_notif(agent.sent_notifs[0][1]).get("terminal", True) is False
        # Releasing a PROC handle is not evidence that the native write stopped.
        assert agent.released_xfers == []
        assert pinning.unpins == []
        if abandon:
            now = 5.0
            if raises:
                with pytest.raises(RuntimeError, match="callback failed"):
                    _poll_until(kvcr, lambda _: bool(errors))
                assert pinning.unpins == []
            assert _poll_until(kvcr, lambda _: bool(pinning.unpins)) == []
            assert [error.state for error in errors] == ["uncertain"]
            assert agent.released_xfers == []
            control.incoming.append(_write_probe_message(12))
            _wait_until(lambda: bool(control.sent))
            assert _decode_control_message(control.sent[-1][1])["terminal"] is False
        if terminal_state == "shutdown":
            kvcr._core._progress._submissions.put(_STOP)
            _wait_until(lambda: kvcr._core._progress._startup_stage == "cleanup")
            assert agent.released_xfers == []
        agent.state = "DONE"
        if terminal_state == "shutdown":
            kvcr.close()
            assert not kvcr._core._framework_pin_keys
            assert not kvcr._core._local_dram_sources_by_op
        else:
            if raises:
                with pytest.raises(RuntimeError, match="callback failed"):
                    _poll_until(kvcr, lambda _: len(errors) == 2)
            assert (
                _poll_until(kvcr, lambda _: not _has_outstanding_operations(kvcr)) == []
            )
        assert pinning.unpins == ["pin"]
        assert agent.released_xfers == [1]
        assert [error.state for error in errors] == (
            ["uncertain", "quiesced"] if abandon else []
        )
        for error in errors:
            assert error.op_handle == source_handle
            assert error.source_blocks == {
                key: [_mem_descriptor(end_point_name="source")]
            }
            assert error.source_owners == {key: "framework"}
            assert error.destination_regions is None
            assert error.destination_owner is None
    finally:
        agent.state = "DONE"
        kvcr.close()


@pytest.mark.parametrize("local_source", [False, True], ids=["framework", "local"])
def test_source_lifecycles_distinguish_targets_reusing_the_same_handle(local_source):
    now = 0.0
    agent, control, errors, done = FakeNixlAgent(), FakeBytesControl(), [], set()
    agent.check_xfer_state = lambda handle: "DONE" if handle in done else "PROC"
    memory = ctypes.create_string_buffer(16)
    source = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        control,
        name="source",
        on_resilience_event=errors.append,
        framework_regions=[_buffer_region(memory, 16)],
        local_dram=(
            LocalDramOptions([("", ctypes.addressof(memory), len(memory))])
            if local_source
            else None
        ),
    )
    source._core._clock = lambda: now
    key = BlockKey(b"shared")
    if local_source:
        install_recovery_records(source._core, {key: _recovered_record(g2=[("", 0)])})
        residency = source._core._block_record_map[key].local_dram
    try:
        for target in ("target-a", "target-b"):
            control.incoming.append(_start_write_message(12, key, target_agent=target))
        _poll_until(source, lambda _: len(agent.xfers) == 2)
        if local_source:
            assert residency.source_readers == 2
            assert source._core._local_dram._pools.source_operations == 2
        now = 5.0
        _poll_until(source, lambda _: len(errors) == 2)
        pending = {error.op_handle for error in errors}
        assert len(pending) == 2
        assert [error.state for error in errors] == ["uncertain", "uncertain"]
        assert (
            errors[0].source_blocks
            == errors[1].source_blocks
            == {key: [_mem_descriptor(end_point_name="source")]}
        )
        owner = "kvcr" if local_source else "framework"
        assert errors[0].source_owners == errors[1].source_owners == {key: owner}
        for native_handle in (1, 2):
            done.add(native_handle)
            _poll_until(source, lambda _: len(errors) == 2 + native_handle)
            event = errors[-1]
            assert event.state == "quiesced"
            assert event.source_blocks == errors[0].source_blocks
            assert event.source_owners == errors[0].source_owners
            pending.remove(event.op_handle)
            assert len(pending) == 2 - native_handle
            if local_source:
                assert residency.source_readers == 2 - native_handle
                assert (
                    source._core._local_dram._pools.source_operations
                    == 2 - native_handle
                )
                if native_handle == 1:
                    replacement = BlockKey(b"replacement")
                    blocked = source.deposit(
                        {replacement: [_mem_descriptor(end_point_name="source")]}
                    )
                    assert source.poll_completed() == [
                        (blocked, _op_entries({replacement: False}))
                    ]
                    assert len(agent.xfers) == 2
        assert not _has_outstanding_operations(source)
    finally:
        done.update((1, 2))
        source.close()


@pytest.mark.parametrize("mixed", [False, True])
def test_abandoned_source_keeps_local_slot_claimed_until_quiescence(mixed):
    now = 0.0
    memory = ctypes.create_string_buffer(16)
    descriptor = _mem_descriptor(end_point_name="source")
    agent, control, errors = FakeNixlAgent(), FakeBytesControl(), []
    source = _new_kvcr(
        agent,
        FakePrimaryPinning(missing_indices=(0,)),
        control,
        name="source",
        on_resilience_event=errors.append,
        local_dram=LocalDramOptions([("", ctypes.addressof(memory), len(memory))]),
        framework_regions=[RegionDescriptor(addr=ctypes.addressof(memory), size=16)],
    )
    source._core._clock = lambda: now
    key, replacement = BlockKey(b"k0"), BlockKey(b"k1")
    missing, framework_hit = BlockKey(b"missing"), BlockKey(b"framework-hit")
    expected_sources = {key: [descriptor]}
    expected_owners = {key: "kvcr"}
    if mixed:
        expected_sources[framework_hit] = [descriptor]
        expected_owners[framework_hit] = "framework"
    try:
        agent.state = "DONE"
        deposit = source.deposit({key: [descriptor]})
        assert _poll_until(source, bool) == [(deposit, _op_entries({key: True}))]
        agent.state = "PROC"
        payload = msgspec.msgpack.decode(
            _start_write_message(12, key, target_agent="target")
        )
        payload["keys"] = [key, missing, framework_hit] if mixed else [key]
        payload["dst_descriptors"] = [
            [_TransferRef("target", index)] for index in range(len(payload["keys"]))
        ]
        control.incoming.append(msgspec.msgpack.encode(payload))
        _poll_until(source, lambda _: len(agent.xfers) == 2)
        now = 5.0
        assert _poll_until(source, lambda _: bool(errors)) == []
        assert [error.state for error in errors] == ["uncertain"]
        assert errors[0].source_blocks == expected_sources
        blocked = source.deposit({replacement: [descriptor]})
        assert list(source.poll_completed()) == [
            (blocked, _op_entries({replacement: False}))
        ]
        residency = source._core._block_record_map[key].local_dram
        assert residency.claim_count + residency.source_readers == 1
        agent.state = "DONE"
        assert _poll_until(source, lambda _: len(errors) == 2) == []
        assert [error.state for error in errors] == ["uncertain", "quiesced"]
        assert residency.claim_count + residency.source_readers == 0
        assert errors[1].source_blocks == expected_sources
        for error in errors:
            assert error.source_owners == expected_owners
            assert error.destination_regions is error.destination_owner is None
        deposit = source.deposit({replacement: [descriptor]})
        assert _poll_until(source, bool) == [
            (deposit, _op_entries({replacement: True}))
        ]
    finally:
        agent.state = "DONE"
        source.close()


@pytest.mark.parametrize("failure", [False, None, 1, RuntimeError("release failed")])
def test_kvcr_pin_release_failure_is_logged_and_retried(kvcr_caplog, failure):
    class FailingPinRelease(FakePrimaryPinning):
        def release_pin(self, pin_handle):
            self.unpins.append(pin_handle)
            if isinstance(failure, Exception):
                raise failure
            return failure

    source_agent = FakeNixlAgent(metadata=b"source-md")
    pinning = FailingPinRelease()
    control = FakeBytesControl()
    kvcr = _new_kvcr(source_agent, pinning, control, name="source")
    control.incoming.append(_start_write_message(14, BlockKey(b"k0")))
    assert _poll_until(kvcr, lambda _: bool(source_agent.xfers)) == []
    source_agent.state = "DONE"
    assert (
        _poll_until(
            kvcr,
            lambda _: pinning.unpins == ["pin"],
        )
        == []
    )

    assert pinning.unpins == ["pin"]
    assert "pin" in kvcr._core._framework_pin_keys
    assert not kvcr._core._block_record_map
    warnings = [record.getMessage() for record in kvcr_caplog.records]
    assert any("release_pin failed" in message for message in warnings)

    failure = True
    kvcr.poll_completed()
    assert pinning.unpins == ["pin", "pin"]
    assert not kvcr._core._framework_pin_keys


@pytest.mark.parametrize("expires", [False, True])
def test_framework_reacquisition_waits_for_pin_release(expires: bool) -> None:
    now = 0.0
    agent = FakeNixlAgent()
    agent.state = "DONE"
    pinning = FakePrimaryPinning()
    pinning.release_pin = Mock(return_value=False)
    control = FakeBytesControl()
    source = _new_kvcr(agent, pinning, control, name="source")
    source._core._clock = lambda: now
    backend = source._core._remote_fw_dram
    key = BlockKey(b"k0")

    control.incoming.append(_start_write_message(1, key))
    _poll_until(source, lambda _: pinning.release_pin.called)
    control.incoming.append(_start_write_message(2, key))
    _poll_until(
        source, lambda _: backend._sources._source_pin_ops or len(agent.xfers) > 1
    )
    assert pinning.searches == [(key,)]
    assert len(agent.xfers) == 1

    if expires:
        now = 2.0
        _poll_until(source, lambda _: bool(agent.sent_notifs))
        assert _decode_notif(agent.sent_notifs[-1][1]) == {
            "type": "write_done",
            "op_handle": 2,
            "success": False,
        }
        assert pinning.searches == [(key,)]

    pinning.release_pin.return_value = True
    _poll_until(source, lambda _: not _has_outstanding_operations(source))
    assert pinning.searches == ([(key,)] if expires else [(key,), (key,)])
    assert len(agent.xfers) == (1 if expires else 2)
    assert not source._core._framework_pin_keys


@pytest.mark.parametrize("late", [False, True], ids=["invalid-result", "late-result"])
def test_discarded_framework_pin_release_is_retried(late: bool) -> None:
    now = 0.0
    agent = FakeNixlAgent()
    pinning = PendingPrimaryPinning()
    pinning.release_pin = Mock(return_value=False)
    control = FakeBytesControl()
    source = _new_kvcr(agent, pinning, control, name="source")
    source._core._clock = lambda: now
    key = BlockKey(b"k0")
    control.incoming.append(_start_write_message(1, key))
    _poll_until(source, lambda _: pinning.searches)
    if late:
        now = 2.0
        _poll_until(source, lambda _: pinning.cancelled)
    pinning.complete(0, missing_indices=() if late else (0,))
    _poll_until(source, lambda _: pinning.release_pin.called)
    assert "pin" in source._core._framework_pin_keys
    assert agent.xfers == []

    pinning.release_pin.return_value = True
    source.poll_completed()
    assert pinning.release_pin.call_count >= 2
    assert not source._core._framework_pin_keys


def test_framework_pin_poll_failures_are_logged_without_escaping(kvcr_caplog):
    class FailingPinPoll(FakePrimaryPinning):
        def __init__(self):
            super().__init__()
            self.poll_attempts = 0

        def poll_pin_results(self):
            self.poll_attempts += 1
            raise RuntimeError("poll failed")

    pinning = FailingPinPoll()
    kvcr = _new_kvcr(
        FakeNixlAgent(),
        pinning,
        FakeBytesControl(),
    )

    assert list(kvcr.poll_completed()) == []
    kvcr.close()

    assert pinning.poll_attempts == 2
    warnings = [record.getMessage() for record in kvcr_caplog.records]
    assert any("framework pin result polling failed" in message for message in warnings)


def test_kvcr_source_poll_failure_waits_for_quiescence_and_is_logged(kvcr_caplog):
    class RaisingAgent(FakeNixlAgent):
        def check_xfer_state(self, handle):
            if self.state != "DONE":
                raise RuntimeError("boom")
            return self.state

    source_agent = RaisingAgent(metadata=b"source-md")
    pinning = FakePrimaryPinning()
    control = FakeBytesControl()
    key = BlockKey(b"k0")
    kvcr = _new_kvcr(source_agent, pinning, control, name="source")

    control.incoming.append(_start_write_message(5, key))
    _poll_until(
        kvcr,
        lambda _: any(
            "transfer progress failed" in rec.getMessage()
            for rec in kvcr_caplog.records
        ),
    )
    assert source_agent.released_xfers == []
    assert pinning.unpins == []
    source_agent.state = "DONE"
    assert _poll_until(kvcr, lambda _: not _has_outstanding_operations(kvcr)) == []
    assert source_agent.released_xfers == [1]
    assert pinning.unpins == ["pin"]
    assert _decode_notif(source_agent.sent_notifs[-1][1]) == {
        "type": "write_done",
        "op_handle": 5,
        "success": False,
    }
    warnings = [rec for rec in kvcr_caplog.records if rec.levelno == logging.WARNING]
    assert any("transfer progress failed" in rec.getMessage() for rec in warnings)


@pytest.mark.parametrize(
    ("shared_key_hit", "second_has_uncovered_key"),
    [(True, True), (False, True), (False, False)],
)
def test_pending_pin_waiters_share_partial_results_and_request_uncovered_keys(
    shared_key_hit: bool,
    second_has_uncovered_key: bool,
) -> None:
    agent = FakeNixlAgent(metadata=b"source-md")
    pinning = PendingPrimaryPinning()
    control = FakeBytesControl()
    keys = (BlockKey(b"k0"), BlockKey(b"k1"), BlockKey(b"k2"))
    source = _new_kvcr(agent, pinning, control, name="source")
    second_keys = keys[1:] if second_has_uncovered_key else keys[1:2]
    for op_handle, op_keys in ((1, keys[:2]), (2, second_keys)):
        control.incoming.append(
            msgspec.msgpack.encode(
                {
                    "type": "start_write",
                    "op_handle": op_handle,
                    "remaining_timeout_ms": 1000,
                    "target_agent_metadata": b"target-md",
                    "target_regions": (
                        {"": RegionDescriptor(addr=128, size=16, count=1024)},
                        {},
                    ),
                    "keys": list(op_keys),
                    "dst_descriptors": [
                        [_TransferRef("target", index)] for index in range(len(op_keys))
                    ],
                }
            )
        )

    expected_searches = [keys[:2]]
    if second_has_uncovered_key:
        expected_searches.append(keys[2:])
    assert (
        _poll_until(source, lambda _: len(pinning.searches) == len(expected_searches))
        == []
    )
    assert pinning.searches == expected_searches

    pinning.complete(
        0,
        "pin-ab",
        missing_indices=() if shared_key_hit else (1,),
    )
    if not shared_key_hit:
        agent.state = "DONE"
    if second_has_uncovered_key:
        pinning.complete(1, "pin-c")
    else:
        wait = source._core._remote_fw_dram._sources._pending_pin_ops[0]
        second_op = ("source", 2)
        first_op = ("source", 1)
        # Exercise the legal order where the missing-only waiter resolves first.
        with patch.object(wait, "op_ids", (second_op, first_op)):
            assert (
                _poll_until(source, lambda _: not _has_outstanding_operations(source))
                == []
            )

    if shared_key_hit:
        assert _poll_until(source, lambda _: len(agent.xfers) == 2) == []
        assert [xfer[2] for xfer in agent.xfers] == [[0, 1], [0, 1]]
    else:
        assert (
            _poll_until(source, lambda _: not _has_outstanding_operations(source)) == []
        )
        assert [xfer[2] for xfer in agent.xfers] == [[0]] * (
            2 if second_has_uncovered_key else 1
        )
        expected_unpins = (
            {"pin-ab", "pin-c"} if second_has_uncovered_key else {"pin-ab"}
        )
        assert set(pinning.unpins) == expected_unpins
    assert pinning.searches == expected_searches
    agent.state = "DONE"
    _poll_until(source, lambda _: not _has_outstanding_operations(source))


def test_source_telemetry_precedes_release_and_is_not_duplicated() -> None:
    """Telemetry is captured once before a transfer handle is released."""

    class LifecycleAgent(FakeNixlAgent):
        def __init__(self):
            super().__init__(metadata=b"source-md")
            self.lifecycle: list[tuple[str, int]] = []
            self.fail_release = True

        def get_xfer_telemetry(self, handle):
            self.lifecycle.append(("telemetry", handle))
            return super().get_xfer_telemetry(handle)

        def release_xfer_handle(self, handle):
            self.lifecycle.append(("release", handle))
            if self.fail_release:
                self.fail_release = False
                raise RuntimeError("busy")
            super().release_xfer_handle(handle)

    agent = LifecycleAgent()
    agent.state = "DONE"
    pinning = FakePrimaryPinning()
    control = FakeBytesControl()
    control.incoming.append(_start_write_message(15, BlockKey(b"k0")))
    source = _new_kvcr(
        agent,
        pinning,
        control,
        KVCRConfig(
            nixl_agent_name="source",
            pool_layouts=[("", 16)],
            enable_telemetry=True,
        ),
        name="source",
    )

    assert _poll_until(source, lambda _: pinning.unpins == ["pin"]) == []
    assert agent.lifecycle == [
        ("telemetry", 1),
        ("release", 1),
        ("release", 1),
    ]
    assert agent.telemetry_handles == [1]

    stats = source.get_stats()
    assert isinstance(stats, FakeTelemetryStats)
    transfer_durations = [
        record
        for record in stats.records
        if record[0] == "histogram"
        and record[1] == DURATION_METRIC
        and record[3] == ("transfer", "success")
    ]
    assert len(transfer_durations) == 1
    assert (
        "counter",
        TRANSFER_BYTES_METRIC,
        32,
        ("source_write",),
    ) in stats.records
    assert (
        "counter",
        TRANSFER_BLOCKS_METRIC,
        1,
        ("source_write",),
    ) in stats.records


@pytest.mark.parametrize(
    ("parts", "selected"),
    [("abc", (0, 1, 2)), ("ca", (2, 0)), ("x", ())],
    ids=["whole", "partial", "unknown"],
)
def test_a_resumed_write_holds_a_pin_another_operation_acquired(
    parts, selected
) -> None:
    """fw_mem belongs to the block, so two writes can want the same pin."""
    backend = object.__new__(_RemoteFWDram)
    kvcr = object.__new__(_KVCRCore)
    kvcr._block_record_map = {}
    kvcr._framework_pin_keys = {}
    kvcr._local_dram_sources_by_op = {}
    kvcr._progress = Mock()
    kvcr._add_block_dependencies = Mock()
    kvcr._remove_block_dependencies = Mock()
    kvcr._claim_local_dram_sources = Mock(return_value={})
    kvcr._release_local_dram_sources = Mock()
    backend._kvcr = kvcr
    backend._sources = _RemoteSources(backend, kvcr)
    released: list[PinHandle] = []
    backend._sources._release_framework_pins = released.extend

    key = BlockKey(b"shared")
    borrowed = PinHandle("pinned-by-the-other-operation")
    sources = [
        _TransferRef("target", i, "pool:" + label) for i, label in enumerate("abc")
    ]
    destinations = tuple(_TransferRef("target", 0, "pool:" + label) for label in parts)
    kvcr._block_record_map[key] = _BlockRecord(
        fw_mem=_FwMemResidency(sources, borrowed)
    )

    # This operation acquired a pin of its own for a key it no longer needs.
    stale = PinHandle("acquired-here")
    waiting = _SourcePinOp(
        started_at=0.0,
        deadline=10.0,
        remote_agent="peer",
        op_handle=1,
        ordered_keys=(key,),
        dst_descriptors=(destinations,),
        op_id=("source", 1),
        keys=frozenset((key,)),
        framework_pins={stale},
    )
    backend._sources._source_pin_ops[("source", 1)] = waiting

    backend._sources._submit_prepared_source_write(("source", 1), waiting)

    submitted = kvcr._progress.submit.call_args.args[0]
    assert submitted.src_descriptors == (
        (tuple(sources[i] for i in selected),) if selected else ()
    )
    assert submitted.dst_descriptors == ((destinations,) if selected else ())
    assert submitted.completed_indices == ((0,) if selected else ())
    expected_pins = {borrowed} if selected else set()
    assert submitted.framework_pins == expected_pins
    assert backend._sources._fw_pins_by_op[submitted.op_id] == expected_pins
    # And the one it acquired but does not read through is handed back.
    assert released == [stale]


def test_a_replacement_reusing_its_predecessors_name_refreshes_the_route() -> None:
    """Busy routes survive refresh requests and can be replaced after draining."""
    agent, pinning, control = FakeNixlAgent(), FakePrimaryPinning(), FakeBytesControl()
    removed = []
    agent.remove_remote_agent = removed.append
    source = _new_kvcr(agent, pinning, control, name="source")
    try:
        progress = source._core._progress
        tier = source._core._remote_fw_dram
        payload = _CONTROL_DECODER.decode(
            _start_write_message(1, b"key", target_agent="worker-a")
        )
        payload.target_agent_metadata = b"gen-1"
        payload.sender_control_endpoint = "tcp://target:1"
        payload.source_control_endpoint = "tcp://source:1"
        first = tier._remote_agent(progress, payload)
        assert first == ("worker-a", "remote-1")
        assert tier._remote_agent(progress, payload) == first
        assert agent.remote_agents == [b"gen-1"]

        # Reject malformed replacement geometry or names before changing the route.
        for catalog in (
            {"": RegionDescriptor(addr=128, size=0)},
            {"wrong-name": RegionDescriptor(addr=128, size=16)},
        ):
            with pytest.raises(ValueError):
                tier._remote_agent(
                    progress,
                    msgspec.structs.replace(payload, target_regions=(catalog, {})),
                )
        assert agent.remote_agents == [b"gen-1"]
        assert removed == []

        control.incoming.append(msgspec.msgpack.encode(payload))
        _poll_until(source, lambda _: len(agent.xfers) == 1)
        old_catalog = progress._catalogs[first[1]]
        old_prepared = progress._prepared[first[1]]
        old_transfers = progress._active_transfers.copy()
        payload.op_handle = 2
        payload.target_agent_metadata = b"gen-2"
        control.sent.clear()
        control.incoming.extend(
            [
                msgspec.msgpack.encode(
                    {"type": "target_metadata", **msgspec.structs.asdict(payload)}
                ),
                msgspec.msgpack.encode(payload),
                _write_probe_message(99),
            ]
        )
        _wait_until(
            lambda: (
                progress._failure is not None
                or any(
                    _decode_control_message(raw)["type"] == "write_probe_ack"
                    for _, raw in control.sent
                )
            )
        )
        progress.raise_if_failed()
        assert [_decode_control_message(raw)["type"] for _, raw in control.sent] == [
            "write_refused",
            "write_probe_ack",
        ]
        assert _decode_control_message(control.sent[0][1])["op_handle"] == 2
        assert progress._thread.is_alive()
        assert progress._active_transfers == old_transfers
        assert progress._catalogs[first[1]] is old_catalog
        assert progress._prepared[first[1]] is old_prepared
        assert (
            tier._remote_agent(progress, _TargetMetadata(target_agent="worker-a"))
            == first
        )
        assert tier._route_generation == {}
        assert agent.remote_agents == [b"gen-1"]
        assert removed == agent.released_xfers == []

        agent.state = "DONE"
        _poll_until(source, lambda _: not _has_outstanding_operations(source))
        # Native bindings may return str directly rather than bytes.
        add_remote_agent = agent.add_remote_agent
        agent.add_remote_agent = lambda metadata: add_remote_agent(metadata).decode()
        control.incoming.append(msgspec.msgpack.encode(payload))
        _poll_until(
            source,
            lambda _: len(agent.xfers) == 2 and not _has_outstanding_operations(source),
        )
        assert agent.remote_agents == [b"gen-1", b"gen-2"]
        assert removed == [first[1]]
        assert first[1] not in progress._catalogs
        assert agent.xfers[-1][4] == "remote-2"
        # The bump is what fences queued predecessor operations off the new route.
        assert tier._route_generation == {"worker-a": 1}

        # Geometry can change without changing the allocation metadata.
        payload.target_regions = (
            {"": RegionDescriptor(addr=128, size=16, count=2)},
            {},
        )
        replaced = tier._remote_agent(progress, payload)
        assert tier._route_generation == {"worker-a": 2}
        assert progress._catalogs[replaced[1]][1][0][""].count == 2
        # A payload carrying no metadata still reuses whatever route is cached.
        assert (
            tier._remote_agent(progress, _TargetMetadata(target_agent="worker-a"))
            == replaced
        )
        assert agent.remote_agents == [b"gen-1", b"gen-2", b"gen-2"]
    finally:
        agent.state = "DONE"


def test_route_setup_cleans_failed_agents_and_preserves_existing_routes() -> None:
    """Cache only prepared routes without disturbing an established peer."""
    agent = FakeNixlAgent()
    removed = []
    agent.remove_remote_agent = removed.append
    progress = SimpleNamespace(
        nixl_agent=agent,
        prepare_memory=Mock(),
        release_prepared=Mock(),
        _prepared={},
        _active_transfers={},
        _stop_requested=False,
        _disconnected_remote_agents=set(),
    )
    tier = SimpleNamespace(
        _kvcr=SimpleNamespace(_timer=time.monotonic),
        _record_progress_duration=lambda *_args: None,
        _remote_agents_by_target={},
        _route_generation={},
    )
    _RemoteFWDram._remote_agent(
        tier,
        progress,
        _TargetMetadata(target_agent="worker-a", target_agent_metadata=b"gen-1"),
    )
    progress.prepare_memory.side_effect = [RuntimeError("prep failed"), None]
    payload = _TargetMetadata(target_agent="worker-b", target_agent_metadata=b"gen-3")
    with pytest.raises(RuntimeError, match="prep failed"):
        _RemoteFWDram._remote_agent(tier, progress, payload)
    assert "worker-b" not in tier._remote_agents_by_target
    assert removed == ["remote-2"]
    progress.release_prepared.assert_called_with("remote-2")
    assert not progress._stop_requested
    assert _RemoteFWDram._remote_agent(tier, progress, payload) == (
        "worker-b",
        "remote-3",
    )
    assert agent.remote_agents == [b"gen-1", b"gen-3", b"gen-3"]

    # A repeated native name still belongs to the established peer.
    agent.add_remote_agent = lambda _: "remote-1"
    progress.prepare_memory.reset_mock(side_effect=True)
    progress.release_prepared.reset_mock()
    with pytest.raises(RuntimeError, match="already initialized"):
        _RemoteFWDram._remote_agent(
            tier,
            progress,
            _TargetMetadata(target_agent="worker-c", target_agent_metadata=b"gen-4"),
        )
    assert "worker-c" not in tier._remote_agents_by_target
    progress.prepare_memory.assert_not_called()
    progress.release_prepared.assert_not_called()
    assert removed == ["remote-2"]
    assert _RemoteFWDram._remote_agent(
        tier, progress, _TargetMetadata(target_agent="worker-a")
    ) == ("worker-a", "remote-1")

    # After a typed disconnect, reload only if NIXL dropped native metadata.
    payload = _TargetMetadata(target_agent="worker-a", target_agent_metadata=b"gen-1")
    agent.add_remote_agent = Mock(return_value="remote-1")
    agent.check_remote_metadata = Mock(return_value=True)
    progress._disconnected_remote_agents.add("remote-1")
    assert _RemoteFWDram._remote_agent(tier, progress, payload) == (
        "worker-a",
        "remote-1",
    )
    agent.check_remote_metadata.return_value = False
    progress._disconnected_remote_agents.add("remote-1")
    progress.release_prepared.side_effect = RuntimeError("active transfers")
    with pytest.raises(RuntimeError, match="active transfers"):
        _RemoteFWDram._remote_agent(tier, progress, payload)
    assert progress._disconnected_remote_agents == {"remote-1"}
    assert tier._route_generation == {}
    progress.release_prepared.side_effect = None
    assert _RemoteFWDram._remote_agent(tier, progress, payload) == (
        "worker-a",
        "remote-1",
    )
    assert not progress._disconnected_remote_agents
    agent.add_remote_agent.assert_called_once_with(b"gen-1")
    assert removed == ["remote-2"]
    assert tier._route_generation == {"worker-a": 1}


@pytest.mark.parametrize(
    "message_type", ["target_metadata", "start_write", "malformed"]
)
def test_route_setup_propagates_cleanup_errors(message_type) -> None:
    """Let progress handle fatal cleanup errors from each control path."""

    class StickyAgent(FakeNixlAgent):
        def remove_remote_agent(self, handle: str) -> None:
            raise RuntimeError("route busy")

    sticky = StickyAgent()
    control = FakeBytesControl()
    source = _new_kvcr(sticky, FakePrimaryPinning(), control, name="source")
    progress = source._core._progress
    tier = source._core._remote_fw_dram
    tier._remote_agent(
        progress,
        _TargetMetadata(target_agent="worker-a", target_agent_metadata=b"gen-1"),
    )
    payload = msgspec.msgpack.decode(
        _start_write_message(1, b"key", target_agent="worker-a")
    )
    payload["target_agent_metadata"] = b"gen-2"
    if message_type == "target_metadata":
        payload["type"] = message_type
    elif message_type == "malformed":
        del payload["keys"]

    control.incoming.append(msgspec.msgpack.encode(payload))
    _wait_until(lambda: not progress._thread.is_alive())
    with pytest.raises(RuntimeError, match="cleanup failed") as error:
        source.close()
    assert progress._failure is error.value
    assert str(error.value.__cause__) == "route busy"
    assert sticky.remote_agents == [b"gen-1"]
    assert tier._route_generation == {}


@pytest.mark.parametrize(
    "incarnation,with_metadata,stopping,reason",
    [
        ("old", False, False, "metadata_required"),
        ("old", True, False, "source_restarted"),
        ("unknown", False, False, "metadata_required"),
        ("unknown", True, False, "source_restarted"),
        ("guard", False, True, None),
        ("guard", True, False, None),
    ],
)
def test_start_write_refusal_names_why(incarnation, with_metadata, stopping, reason):
    """Missing metadata, or a quote of some other process, is refused over
    control with a reason before anything is submitted; a stopping source
    refuses without one, and our own quote with metadata is served."""
    agent = FakeNixlAgent(metadata=b"guard-md")
    control = FakeBytesControl("tcp://guard:1")
    source = _new_kvcr(agent, FakePrimaryPinning(), control, name="guard")
    tier = source._core._remote_fw_dram
    tier._dangling_ops.incarnation = "guard"
    tier._dangling_ops.dead_incarnations.add("old")
    payload = msgspec.msgpack.decode(
        _start_write_message(17, b"k", target_agent="target")
    )
    payload.update(
        sender_control_endpoint="tcp://target:1",
        source_control_endpoint="tcp://guard:1",
        source_incarnation=incarnation,
    )
    if not with_metadata:
        del payload["target_agent_metadata"]
    progress = source._core._progress
    delivered = []

    def recv():  # Stop and the request land in one poll, on the progress thread.
        if delivered:
            return []
        delivered.append(True)
        progress._stop_requested = stopping
        return [msgspec.msgpack.encode(payload)]

    def refusals():
        return [
            message
            for _, raw in control.sent
            if (message := _decode_control_message(raw))["type"] == "write_refused"
        ]

    control.recv = recv
    if stopping:
        _wait_until(lambda: not progress._thread.is_alive())
    elif reason:
        _wait_until(lambda: bool(refusals()))  # A metadata ACK may go out first.
    else:
        _wait_until(lambda: bool(tier._dangling_ops.source_writes))

    refusals = refusals()
    assert [(m.get("reason"), m["sender_incarnation"]) for m in refusals] == (
        [(reason, "guard")] if reason or stopping else []
    )
    assert not agent.sent_notifs
    assert bool(tier._dangling_ops.source_writes) is (reason is None and not stopping)
