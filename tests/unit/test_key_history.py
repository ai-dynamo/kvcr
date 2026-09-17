"""Opt-in per-key lifecycle history with fake NIXL byte copies."""

import ctypes
import hashlib
import json
import logging

from _kvcr_test_utils import (
    FakeNixlAgent,
    _mem_descriptor,
    _new_local_kvcr,
    _poll_until,
)

from kvcr.core import _BlockRecord
from kvcr.local_dram import _LocalDramResidency, _LocalDramState
from kvcr.types import BlockKey, CacheTier


def _history(caplog):
    return [
        json.loads(record.message.split("KVCR_KEY_HISTORY ", 1)[1])
        for record in caplog.records
        if "KVCR_KEY_HISTORY " in record.message
    ]


def test_history_captures_ready_remove_and_inventory(monkeypatch, caplog):
    monkeypatch.setenv("KVCR_KEY_HISTORY", "1")
    caplog.set_level(logging.INFO, logger="kvcr.core")
    primary = ctypes.create_string_buffer(b"a" * 32, 32)
    local = ctypes.create_string_buffer(16)
    agent = FakeNixlAgent()
    agent.state = "DONE"
    kvcr = _new_local_kvcr(agent, local, 1, inventory_sink=lambda event: None)
    first, second = BlockKey(b"first"), BlockKey(b"second")

    first_op = kvcr.deposit({first: [_mem_descriptor(ctypes.addressof(primary), 16)]})
    _poll_until(kvcr, lambda done: first_op in dict(done))
    second_op = kvcr.deposit(
        {second: [_mem_descriptor(ctypes.addressof(primary) + 16, 16)]}
    )
    _poll_until(kvcr, lambda done: second_op in dict(done))

    events = _history(caplog)
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert any(event["event"] == "ready" for event in events)
    removal = next(event for event in events if event["event"] == "remove")
    assert removal["reason"] == "capacity_eviction"
    assert removal["key_sha256"] == [hashlib.sha256(first).hexdigest()]
    assert any(
        event["event"] == "inventory_callback"
        and event["tier"] == CacheTier.LOCAL_G2.value
        and event["outcome"] == "accepted"
        for event in events
    )


def test_callback_outcomes_and_disabled_history(monkeypatch, caplog):
    monkeypatch.setenv("KVCR_KEY_HISTORY", "1")
    caplog.set_level(logging.INFO, logger="kvcr.core")
    local = ctypes.create_string_buffer(16)
    kvcr = _new_local_kvcr(FakeNixlAgent(), local, 1)
    key = BlockKey(b"callback")
    assert not kvcr._core._publish_inventory((key,), CacheTier.LOCAL_G2, removed=False)
    assert _history(caplog)[0]["outcome"] == "no_callback"

    caplog.clear()
    monkeypatch.delenv("KVCR_KEY_HISTORY", raising=False)
    quiet = _new_local_kvcr(FakeNixlAgent(), local, 1)

    def forbidden():
        raise AssertionError("disabled history consumed keys")
        yield

    quiet._core._log_key_history("ready", forbidden())
    assert _history(caplog) == []


def test_recovered_history_is_bounded_and_hashed(monkeypatch, caplog):
    monkeypatch.setenv("KVCR_KEY_HISTORY", "1")
    caplog.set_level(logging.INFO, logger="kvcr.core")
    local = ctypes.create_string_buffer(16 * 33)
    kvcr = _new_local_kvcr(FakeNixlAgent(), local, 33)
    records = {
        BlockKey(str(index).encode()): _BlockRecord(
            local_dram=_LocalDramResidency([("", index)], _LocalDramState.READY)
        )
        for index in range(33)
    }

    kvcr._core.adopt_recovery_records(records)

    events = _history(caplog)
    assert [event["event"] for event in events] == [
        "recovered_ready",
        "recovered_ready",
    ]
    assert [len(event["key_sha256"]) for event in events] == [32, 1]
    assert [event["batch_offset"] for event in events] == [0, 32]
    assert {digest for event in events for digest in event["key_sha256"]} == {
        hashlib.sha256(key).hexdigest() for key in records
    }
