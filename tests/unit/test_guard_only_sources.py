# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Option A uses real core/progress/allocator with the suite's byte-address fakes."""

import ctypes
import threading

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _decode_notif,
    _mem_descriptor,
    _new_kvcr,
    _op_entries,
    _poll_until,
    _start_write_message,
    _wait_until,
)

from kvcr.config import KVCRConfig, LocalDramOptions, RemoteFWDramOptions
from kvcr.core import _BlockRecord
from kvcr.local_dram import _LocalDramResidency, _LocalDramState
from kvcr.progress import _TransferRef
from kvcr.recovery_journal import install_recovery_records
from kvcr.types import RegionDescriptor


def _source(slots=3, *, recovered=False):
    memory = ctypes.create_string_buffer(b"a" * 16 + b"b" * 16 + b"c" * 16, 48)
    agent, control, pinning = FakeNixlAgent(), FakeBytesControl(), FakePrimaryPinning()
    options = RemoteFWDramOptions(guard_only_sources=True)
    source = _new_kvcr(
        agent,
        pinning,
        control,
        KVCRConfig(
            nixl_agent_name="source", pool_layouts=[("", 16)], enable_telemetry=True
        ),
        name="source",
        remote_options=options,
        framework_regions=[
            RegionDescriptor(addr=ctypes.addressof(memory), size=16, count=3),
            RegionDescriptor(
                addr=ctypes.addressof(memory), size=16, count=3, label=":*"
            ),
        ],
        local_dram=LocalDramOptions([("", ctypes.addressof(memory), slots * 16)]),
    )
    agent.state = "DONE"
    if recovered:
        install_recovery_records(
            source._core,
            {
                b"a": _BlockRecord(
                    local_dram=_LocalDramResidency([("", 0)], _LocalDramState.READY)
                )
            },
        )
    else:
        deposit = source.deposit({b"a": [_mem_descriptor(0, end_point_name="source")]})
        assert _poll_until(source, bool) == [(deposit, _op_entries({b"a": True}))]
    return source, agent, control, pinning, memory


@pytest.fixture
def source_factory():
    instances = []

    def make(*args, **kwargs):
        instance = _source(*args, **kwargs)
        instances.append(instance)
        return instance

    yield make
    for source, agent, *_ in reversed(instances):
        agent.state = "DONE"
        source.close()


def _request(control, keys, handle=12, **extra):
    payload = msgspec.msgpack.decode(
        _start_write_message(handle, keys[0], target_agent="target")
    )
    payload.update(
        keys=keys,
        dst_descriptors=[[_TransferRef("target", i)] for i in range(len(keys))],
        **extra,
    )
    control.incoming.append(msgspec.msgpack.encode(payload))


def _notifications(agent):
    return [_decode_notif(message) for _, message in agent.sent_notifs]


def test_guard_only_sources_is_explicit_and_default_off():
    assert not RemoteFWDramOptions().guard_only_sources
    assert RemoteFWDramOptions(guard_only_sources=True).guard_only_sources


@pytest.mark.parametrize("recovered", [False, True])
def test_ready_subset_progress_while_caller_state_lock_held(recovered, source_factory):
    source, agent, control, pinning, memory = source_factory(recovered=recovered)
    before = len(agent.xfers)
    with source._core._state_lock:
        _request(control, [b"missing", b"a", b"other"])
        _wait_until(lambda: len(agent.xfers) == before + 1)
        assert agent.xfers[-1][1] == [(ctypes.addressof(memory), 16, 0)]
        notice = _decode_notif(agent.xfers[-1][5])
        assert notice["completed_indices"] == [1]
        assert pinning.searches == []
        assert not source._core._remote_fw_dram._source_pin_ops
    assert source._core._local_dram._export_readers() == 0


def test_zero_hit_and_layout_mismatch_never_create_source_pin(
    monkeypatch, source_factory
):
    source, agent, control, pinning, _ = source_factory()
    from kvcr.remote_fw_dram import _SourcePinOp

    def forbidden(*args, **kwargs):
        raise AssertionError("SourcePinOp created")

    monkeypatch.setattr(_SourcePinOp, "__init__", forbidden)
    before = len(agent.xfers)
    with source._core._state_lock:
        _request(control, [b"a", b"missing"], key_layout=["different"])
        _wait_until(
            lambda: any(n.get("op_handle") == 12 for n in _notifications(agent))
        )
        notice = next(n for n in _notifications(agent) if n.get("op_handle") == 12)
        assert notice["completed_indices"] == []
        assert len(agent.xfers) == before
        assert pinning.searches == []


def test_export_index_contention_is_prompt_miss_without_caller_queue():
    source, agent, control, pinning, _ = _source()
    local = source._core._local_dram
    with local._export_lock:
        _request(control, [b"a"])
        _wait_until(lambda: bool(_notifications(agent)))
        assert _notifications(agent)[-1]["completed_indices"] == []
        assert pinning.searches == []
    source.close()


def test_native_pending_claim_fences_slot_reuse_then_releases_without_main_poll(
    source_factory,
):
    source, agent, control, pinning, _ = source_factory(slots=1)
    local = source._core._local_dram
    before = len(agent.xfers)
    agent.state = "PROC"
    _request(control, [b"a"])
    _wait_until(lambda: len(agent.xfers) == before + 1)
    assert local._export_readers() == 1
    assert local.telemetry_state()["guard_export_readers"] == 1
    blocked = source.deposit(
        {b"replace": [_mem_descriptor(1, end_point_name="source")]}
    )
    assert _poll_until(source, bool) == [
        (blocked, _op_entries({b"replace": False}))
    ]
    agent.state = "DONE"
    with source._core._state_lock:
        _wait_until(lambda: local._export_readers() == 0)
        assert local.telemetry_state()["guard_export_readers"] == 0
    replacement = source.deposit(
        {b"replace": [_mem_descriptor(1, end_point_name="source")]}
    )
    assert _poll_until(source, bool) == [
        (replacement, _op_entries({b"replace": True}))
    ]
    _request(control, [b"a"], handle=13)
    _wait_until(
        lambda: any(n.get("op_handle") == 13 for n in _notifications(agent))
    )
    assert (
        next(n for n in _notifications(agent) if n.get("op_handle") == 13)[
            "completed_indices"
        ]
        == []
    )


def test_mixed_ready_missing_and_per_key_layout_rejection(caplog, source_factory):
    caplog.set_level("INFO", logger="kvcr.core")
    source, agent, control, pinning, _ = source_factory()
    deposits = source.deposit(
        {
            b"wrong": [_mem_descriptor(1, ":b", end_point_name="source")],
            b"other": [_mem_descriptor(2, end_point_name="source")],
        }
    )
    assert _poll_until(source, bool) == [
        (deposits, _op_entries({b"wrong": True, b"other": True}))
    ]
    before = len(agent.xfers)
    with source._core._state_lock:
        _request(control, [b"missing", b"a", b"wrong", b"other"])
        _wait_until(lambda: len(agent.xfers) == before + 1)
        assert _decode_notif(agent.xfers[-1][5])["completed_indices"] == [1, 3]
    assert "requested=4" in caplog.text
    assert pinning.searches == []


@pytest.mark.parametrize("ambiguous_submit", [False, True])
def test_cancel_uncertainty_retains_readers_until_native_quiescence(
    ambiguous_submit, source_factory
):
    source, agent, control, pinning, _ = source_factory(slots=1)
    now = [0.0]
    source._core._clock = lambda: now[0]
    local = source._core._local_dram
    original = agent.transfer

    def transfer(handle):
        original(handle)
        raise RuntimeError("native submit outcome unknown")

    if ambiguous_submit:
        agent.transfer = transfer
    before = len(agent.xfers)
    agent.state = "PROC"
    _request(control, [b"a"])
    _wait_until(lambda: len(agent.xfers) == before + 1)
    with source._core._state_lock:
        now[0] = 100.0
        statuses = source._core._remote_fw_dram._dangling_ops.source_writes
        _wait_until(lambda: any(s.abandoned for s in statuses.values()))
        assert local._export_readers() == 1
        agent.state = "ERR"
        # A generic native error is not proof that source access stopped.
        _wait_until(
            lambda: any(
                t.outcome is False
                for t in source._core._progress._active_transfers.values()
            )
        )
        assert local._export_readers() == 1
        agent.state = "DONE"
        _wait_until(lambda: local._export_readers() == 0)
        assert not source._core._remote_fw_dram._source_pin_ops


def test_eviction_rechecks_reader_acquired_during_policy(monkeypatch):
    source, agent, control, pinning, _ = _source(slots=1)
    local = source._core._local_dram
    reached, resume = threading.Event(), threading.Event()
    decide = source._core._decide_eviction

    def pause(*args, **kwargs):
        result = decide(*args, **kwargs)
        reached.set()
        assert resume.wait(2)
        return result

    monkeypatch.setattr(source._core, "_decide_eviction", pause)
    outcome = []

    def replace():
        outcome.append(
            source.deposit({b"new": [_mem_descriptor(1, end_point_name="source")]})
        )

    worker = threading.Thread(target=replace)
    try:
        agent.state = "PROC"
        before = len(agent.xfers)
        worker.start()
        assert reached.wait(2)
        _request(control, [b"a"])
        _wait_until(lambda: len(agent.xfers) == before + 1)
        resume.set()
        worker.join(2)
        assert not worker.is_alive()
        assert _poll_until(source, bool) == [(outcome[0], _op_entries({b"new": False}))]
        assert source._core._block_record_map[b"a"].local_dram is not None
        assert local._export_readers() == 1 and len(local._free_slots[""]) == 0
    finally:
        resume.set()
        worker.join(2)
        agent.state = "DONE"
        source.close()


def test_close_waits_for_native_quiescence_before_reader_release():
    source, agent, control, pinning, _ = _source()
    local = source._core._local_dram
    errors = []

    def close():
        try:
            source.close()
        except BaseException as error:
            errors.append(error)

    closer = threading.Thread(target=close)
    try:
        agent.state = "PROC"
        before = len(agent.xfers)
        _request(control, [b"a"])
        _wait_until(lambda: len(agent.xfers) == before + 1)
        closer.start()
        _wait_until(lambda: source._core._progress._stop_requested)
        assert local._export_readers() == 1
        agent.state = "DONE"
        closer.join(2)
        assert not closer.is_alive() and errors == []
        assert local._export_readers() == 0
    finally:
        agent.state = "DONE"
        if closer.ident is not None:
            closer.join(3)
        source.close()


def test_native_handle_cleanup_failure_keeps_export_reader():
    source, agent, control, pinning, _ = _source()
    local = source._core._local_dram
    release = agent.release_xfer_handle
    attempted = threading.Event()

    def pending_release(handle):
        attempted.set()
        return False

    try:
        before = len(agent.xfers)
        agent.release_xfer_handle = pending_release
        _request(control, [b"a"])
        _wait_until(lambda: len(agent.xfers) == before + 1)
        assert attempted.wait(1)
        with source._core._state_lock:
            assert local._export_readers() == 1
            agent.release_xfer_handle = release
            _wait_until(lambda: local._export_readers() == 0)
    finally:
        agent.release_xfer_handle = release
        source.close()


def test_source_access_accounting_is_deferred_without_holding_reader(source_factory):
    source, agent, control, pinning, _ = source_factory()
    local = source._core._local_dram
    record = source._core._block_record_map[b"a"]
    before_access = record.access_count
    before = len(agent.xfers)
    with source._core._state_lock:
        _request(control, [b"a", b"missing"])
        _wait_until(lambda: len(agent.xfers) == before + 1)
        _wait_until(lambda: local._export_readers() == 0)
        assert record.access_count == before_access
    _poll_until(source, lambda _: record.access_count == before_access + 1)
