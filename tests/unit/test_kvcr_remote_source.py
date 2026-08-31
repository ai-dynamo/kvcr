# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KVCR remote framework-DRAM source-side tests."""

import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    FakeTelemetryStats,
    PendingPrimaryPinning,
    _decode_notif,
    _has_outstanding_operations,
    _mem_descriptor,
    _new_kvcr,
    _poll_until,
    _start_write_message,
    _wait_until,
)

from kvcr import DURATION_METRIC, TRANSFER_BLOCKS_METRIC, TRANSFER_BYTES_METRIC
from kvcr.config import KVCRConfig
from kvcr.core import _BlockRecord, _KVCRCore
from kvcr.remote_fw_dram import _FwMemResidency, _RemoteFWDram, _SourcePinOp
from kvcr.types import BlockKey, PinHandle, PinRequestId


@pytest.mark.parametrize("pin_before_deadline", [True, False])
def test_kvcr_start_write_respects_framework_pin_deadline(
    pin_before_deadline: bool,
) -> None:
    """Source writes must not start after their framework-pin deadline."""
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
        KVCRConfig(nixl_agent_name="source", operation_timeout_ms=10_000),
        name="source",
    )
    source._core._clock = clock
    key = BlockKey(b"k0")
    control.incoming.append(_start_write_message(9, key, remaining_timeout_ms=100))
    if pin_before_deadline:
        assert (
            _poll_until(
                source, lambda _: bool(source._core._remote_fw_dram._pending_pin_ops)
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
    assert _decode_notif(source_agent.sent_notifs[0][1]) == {
        "type": "write_done",
        "op_handle": 9,
        "success": False,
    }


def test_kvcr_close_cleans_pending_pin_operations():
    agent = FakeNixlAgent(metadata=b"source-md")
    pinning = PendingPrimaryPinning()
    control = FakeBytesControl()
    key = BlockKey(b"k0")
    control.incoming.append(_start_write_message(1, key))

    source = _new_kvcr(agent, pinning, control, name="source")
    assert (
        _poll_until(
            source, lambda _: bool(source._core._remote_fw_dram._pending_pin_ops)
        )
        == []
    )
    assert source._core._remote_fw_dram._pending_pin_ops

    source.close()

    assert pinning.cancelled == [PinRequestId(0)]
    assert not source._core._remote_fw_dram._source_pin_ops
    assert not source._core._remote_fw_dram._pending_pin_ops
    assert not source._core._framework_pin_keys


def test_kvcr_malformed_start_write_notifies_failure(kvcr_caplog):
    source_agent = FakeNixlAgent(metadata=b"source-md")
    control = FakeBytesControl()
    _new_kvcr(source_agent, FakePrimaryPinning(), control, name="source")
    control.incoming.append(
        msgspec.msgpack.encode(
            {
                "type": "start_write",
                "op_handle": 6,
                "target_agent": "target",
                "target_agent_metadata": b"target-md",
                "keys": [BlockKey(b"k0")],
                "dst_descriptors": [],
            }
        )
    )
    _wait_until(lambda: bool(source_agent.sent_notifs))

    assert _decode_notif(source_agent.sent_notifs[0][1]) == {
        "type": "write_done",
        "op_handle": 6,
        "success": False,
    }
    assert any(
        "malformed start_write" in record.getMessage()
        for record in kvcr_caplog.records
        if record.levelno == logging.WARNING
    )


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


@pytest.mark.parametrize("failure", ["initialize", "error", "exception"])
def test_kvcr_source_transfer_error_notifies_failure_and_cleans_up(
    failure: str,
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
            return "ERR"

    source_agent = FailingTransferAgent(metadata=b"source-md")
    pinning = FakePrimaryPinning()
    control = FakeBytesControl()
    key = BlockKey(b"k0")
    control.incoming.append(_start_write_message(5, key))
    kvcr = _new_kvcr(source_agent, pinning, control, name="source")

    assert _poll_until(kvcr, lambda _: pinning.unpins == ["pin"]) == []

    assert source_agent.transfers == ([] if failure == "initialize" else [1])
    assert len(source_agent.sent_notifs) == 1
    agent_name, notif = source_agent.sent_notifs[0]
    assert agent_name == b"remote-1"
    assert _decode_notif(notif) == {
        "type": "write_done",
        "op_handle": 5,
        "success": False,
    }
    assert source_agent.released_xfers == ([] if failure == "initialize" else [1])
    assert not _has_outstanding_operations(kvcr)


def test_kvcr_source_async_transfer_error_notifies_failure():
    source_agent = FakeNixlAgent(metadata=b"source-md")
    pinning = FakePrimaryPinning()
    control = FakeBytesControl()
    key = BlockKey(b"k0")
    control.incoming.append(_start_write_message(7, key))
    kvcr = _new_kvcr(source_agent, pinning, control, name="source")

    assert _poll_until(kvcr, lambda _: bool(source_agent.xfers)) == []
    assert source_agent.sent_notifs == []

    source_agent.state = "ERR"
    assert (
        _poll_until(
            kvcr,
            lambda _: (
                bool(source_agent.sent_notifs) and not _has_outstanding_operations(kvcr)
            ),
        )
        == []
    )

    assert len(source_agent.sent_notifs) == 1
    agent_name, notif = source_agent.sent_notifs[0]
    assert agent_name == b"remote-1"
    assert _decode_notif(notif) == {
        "type": "write_done",
        "op_handle": 7,
        "success": False,
    }
    assert source_agent.released_xfers == [1]
    assert pinning.unpins == ["pin"]


def test_kvcr_source_ignores_malformed_control_messages():
    """Malformed control payloads never crash the scheduler or cause effects."""
    source_agent = FakeNixlAgent(metadata=b"source-md")
    pinning = FakePrimaryPinning()
    control = FakeBytesControl()
    control.incoming.append(b"\xff\xff\xff not msgpack")
    control.incoming.append(msgspec.msgpack.encode(b"top-level-not-dict"))
    control.incoming.append(msgspec.msgpack.encode({"type": "unknown"}))
    kvcr = _new_kvcr(source_agent, pinning, control, name="source")

    time.sleep(0.01)
    assert list(kvcr.poll_completed()) == []
    kvcr._core._progress.raise_if_failed()
    assert source_agent.xfers == []
    assert source_agent.sent_notifs == []
    assert pinning.unpins == []


@pytest.mark.parametrize(
    "terminal_state",
    [None, "ERR", "DONE"],
    ids=["cancelled", "failed", "completed"],
)
def test_kvcr_source_timeout_holds_pins_until_safe_release(
    terminal_state: str | None,
) -> None:
    class DelayedReleaseAgent(FakeNixlAgent):
        def __init__(self):
            super().__init__(metadata=b"source-md")
            self.release_attempts = 0
            self.allow_release = False

        def release_xfer_handle(self, handle):
            self.release_attempts += 1
            if self.release_attempts == 1:
                raise RuntimeError("busy")
            if not self.allow_release:
                return False
            super().release_xfer_handle(handle)

    now = 0.0
    source_agent = DelayedReleaseAgent()
    pinning = FakePrimaryPinning()
    control = FakeBytesControl()
    key = BlockKey(b"k0")
    kvcr = _new_kvcr(
        source_agent,
        pinning,
        control,
        KVCRConfig(
            nixl_agent_name="source",
            operation_timeout_ms=1000,
            enable_telemetry=True,
        ),
        name="source",
    )
    kvcr._core._clock = lambda: now
    control.incoming.append(_start_write_message(12, key))

    assert _poll_until(kvcr, lambda _: bool(source_agent.xfers)) == []
    now = 2.0
    _wait_until(lambda: source_agent.release_attempts > 0)
    assert source_agent.released_xfers == []
    assert pinning.unpins == []
    assert _has_outstanding_operations(kvcr)

    if terminal_state is not None:
        source_agent.state = terminal_state
    source_agent.allow_release = True
    assert _poll_until(kvcr, lambda _: not _has_outstanding_operations(kvcr)) == []
    assert not kvcr._core._remote_fw_dram._source_pin_ops
    assert pinning.unpins == ["pin"]
    assert source_agent.released_xfers == [1]
    if terminal_state != "DONE":
        assert _decode_notif(source_agent.sent_notifs[0][1]) == {
            "type": "write_done",
            "op_handle": 12,
            "success": False,
        }
    else:
        assert source_agent.sent_notifs == []
        assert source_agent.telemetry_handles == [1]


def test_kvcr_pin_release_failure_is_logged_without_escaping(kvcr_caplog):
    class FailingPinRelease(FakePrimaryPinning):
        def release_pin(self, pin_handle):
            self.unpins.append(pin_handle)
            raise RuntimeError("release failed")

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
    warnings = [record.getMessage() for record in kvcr_caplog.records]
    assert any("release_pin failed" in message for message in warnings)
    # Shutdown reports the pin instead; drop it so the fixture can close.
    kvcr._core._framework_pin_keys.clear()


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


def test_kvcr_source_poll_failure_is_terminal_and_logged(kvcr_caplog):
    class RaisingAgent(FakeNixlAgent):
        def check_xfer_state(self, handle):
            raise RuntimeError("boom")

    source_agent = RaisingAgent(metadata=b"source-md")
    pinning = FakePrimaryPinning()
    control = FakeBytesControl()
    key = BlockKey(b"k0")
    kvcr = _new_kvcr(source_agent, pinning, control, name="source")

    control.incoming.append(_start_write_message(5, key))
    assert (
        _poll_until(
            kvcr,
            lambda _: (
                bool(source_agent.sent_notifs) and not _has_outstanding_operations(kvcr)
            ),
        )
        == []
    )
    assert source_agent.released_xfers == [1]
    assert pinning.unpins == ["pin"]
    assert _decode_notif(source_agent.sent_notifs[0][1]) == {
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
                    "keys": list(op_keys),
                    "dst_descriptors": [
                        _mem_descriptor(addr=128 + index * 16).__dict__
                        for index in range(len(op_keys))
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
        wait = source._core._remote_fw_dram._pending_pin_ops[0]
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
        assert [xfer[2] for xfer in agent.xfers] == [[0]]
        expected_unpins = (
            {"pin-ab", "pin-c"} if second_has_uncovered_key else {"pin-ab"}
        )
        assert set(pinning.unpins) == expected_unpins
    assert pinning.searches == expected_searches


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
        KVCRConfig(nixl_agent_name="source", enable_telemetry=True),
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


def test_a_resumed_write_holds_a_pin_another_operation_acquired() -> None:
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
    backend._source_pin_ops = {}
    backend._fw_pins_by_op = {}
    released: list[PinHandle] = []
    backend._release_framework_pins = released.extend

    key = BlockKey(b"shared")
    borrowed = PinHandle("pinned-by-the-other-operation")
    kvcr._block_record_map[key] = _BlockRecord(
        fw_mem=_FwMemResidency(_mem_descriptor(), borrowed)
    )

    # This operation acquired a pin of its own for a key it no longer needs.
    stale = PinHandle("acquired-here")
    waiting = _SourcePinOp(
        started_at=0.0,
        deadline=10.0,
        remote_agent=b"peer",
        op_handle=1,
        ordered_keys=(key,),
        dst_descriptors=(_mem_descriptor(),),
        op_id=("source", 1),
        keys={key},
        framework_pins={stale},
    )
    backend._source_pin_ops[("source", 1)] = waiting

    backend._submit_prepared_source_write(("source", 1), waiting)

    submitted = kvcr._progress.submit.call_args.args[0]
    assert borrowed in submitted.framework_pins, (
        "the resumed write reads through this pin but does not hold it"
    )
    assert backend._fw_pins_by_op[submitted.op_id] == {borrowed}
    # And the one it acquired but does not read through is handed back.
    assert released == [stale]


def test_a_replacement_reusing_its_predecessors_name_refreshes_the_route() -> None:
    """Same peer name with new metadata re-adds the NIXL route; identical
    metadata keeps reusing the cached one."""
    agent = FakeNixlAgent()
    progress = SimpleNamespace(nixl_agent=agent)
    tier = SimpleNamespace(
        _kvcr=SimpleNamespace(_timer=time.monotonic),
        _record_progress_duration=lambda *_args: None,
        _remote_agents_by_target={},
    )
    payload = {"target_agent": "worker-a", "target_agent_metadata": b"gen-1"}

    first = _RemoteFWDram._remote_agent(tier, progress, payload)

    assert _RemoteFWDram._remote_agent(tier, progress, payload) == first
    assert agent.remote_agents == [b"gen-1"]

    replaced = _RemoteFWDram._remote_agent(
        tier,
        progress,
        {"target_agent": "worker-a", "target_agent_metadata": b"gen-2"},
    )

    assert agent.remote_agents == [b"gen-1", b"gen-2"]
    assert replaced[1] != first[1]
    # A payload carrying no metadata still reuses whatever route is cached.
    named_only = {"target_agent": "worker-a"}
    assert _RemoteFWDram._remote_agent(tier, progress, named_only) == replaced
