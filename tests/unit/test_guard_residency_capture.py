# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exact-key capture observes transitions without extending the export lock."""

import ctypes
import threading
from types import SimpleNamespace

import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _mem_descriptor,
    _new_kvcr,
    _poll_until,
)
from test_no_pinning_mode import source_factory as source_factory

from kvcr import compact_telemetry as telemetry
from kvcr.config import LocalDramOptions, RemoteFWDramOptions
from kvcr.local_dram import (
    _GuardExport,
    _LocalDram,
    _LocalDramResidency,
    _LocalDramState,
)
from kvcr.progress import _TransferRef
from kvcr.types import RegionDescriptor


@pytest.fixture
def observed():
    local = object.__new__(_LocalDram)
    local._export_lock = threading.Lock()
    local._exports = {}
    local._export_held = 0
    local._export_sequence = 0
    local._first_guard_ready_at = None
    records = []

    def journal(record):
        if record.get("miss_reasons") != ["index_busy"]:
            assert not local._export_lock.locked()
        records.append(record)

    local._capture_writer = SimpleNamespace(journal=journal)
    local._kvcr = SimpleNamespace(
        _no_pinning_mode=True,
        nixl_agent_name="source",
        _remote_fw_dram=SimpleNamespace(
            _dangling_ops=SimpleNamespace(incarnation="generation")
        ),
    )
    local._descriptor = lambda label, slot: _TransferRef(
        "source", slot, label=label, framework=False
    )
    return local, records


def test_ready_and_withdraw_share_lock_order_but_encode_after_unlock(observed):
    local, records = observed
    residency = _LocalDramResidency([("pool", 0)], _LocalDramState.READY)
    local._publish_export(b"key", residency)
    ready = records.pop()
    assert ready["event"] == "ready"
    assert ready["export_sequence"] == 1
    assert ready["generation"] == residency.telemetry_generation
    assert local._first_guard_ready_at == ready["transition_ns"] / 1e9
    with local._export_lock:
        withdrawal = local._withdraw_locked(b"key")
    local._journal_withdrawal(b"key", withdrawal, "capacity_eviction")
    assert records[-1]["event"] == "withdraw"
    assert records[-1]["reason"] == "capacity_eviction"
    assert records[-1]["export_sequence"] == 2
    assert records[-1]["transition_ns"] >= ready["transition_ns"]
    assert records[-1]["generation"] == ready["generation"]


def test_reused_key_has_new_residency_generation():
    first = _LocalDramResidency([("pool", 0)], _LocalDramState.READY)
    second = _LocalDramResidency([("pool", 0)], _LocalDramState.READY)
    assert first.telemetry_generation != second.telemetry_generation


def test_get_journals_ready_and_missing_without_framework_callback(observed):
    local, records = observed
    local._publish_export(
        b"ready", _LocalDramResidency([("pool", 0)], _LocalDramState.READY)
    )
    claims, busy = local.claim_export_subset(
        (b"ready", b"missing"), (("pool",), ("pool",)), ("pool",)
    )
    get = records[-1]
    assert not busy and len(claims) == 1
    assert get["event"] == "source_get"
    assert get["keys"] == (b"ready", b"missing")
    assert get["ready_indices"] == [0]
    assert get["miss_reasons"] == ["ready", "absent"]
    assert get["generations"][0] is not None
    assert get["export_sequence"] == 1
    local.release_export_claims(tuple(claim for _, claim in claims))


def test_busy_claim_is_explicit_without_waiting_on_export_lock(observed):
    local, records = observed
    # A different thread holding this lock must not force a framework handoff.
    with local._export_lock:
        claims, busy = local.claim_export_subset((b"key",), (("pool",),), ("pool",))
    assert busy and claims == ()
    assert records[-1]["miss_reasons"] == ["index_busy"]


def test_capacity_eviction_journal_correlates_with_subsequent_get():
    source_memory = ctypes.create_string_buffer(b"a" * 16 + b"b" * 16, 32)
    guard_memory = ctypes.create_string_buffer(16)
    agent = FakeNixlAgent()
    agent.state = "DONE"
    cache = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        FakeBytesControl(),
        local_dram=LocalDramOptions([("", ctypes.addressof(guard_memory), 16)]),
        remote_options=RemoteFWDramOptions(no_pinning_mode=True),
        framework_regions=[
            RegionDescriptor(addr=ctypes.addressof(source_memory), size=16, count=2)
        ],
    )
    records = []
    local = cache._core._local_dram
    local._capture_writer = SimpleNamespace(journal=records.append)
    for index, key in enumerate((b"a", b"b")):
        cache.deposit({key: [_mem_descriptor(index)]})
        assert _poll_until(cache, bool)
    claims, busy = local.claim_export_subset((b"a", b"b"), (("",), ("",)), ("",))
    assert not busy and [position for position, _ in claims] == [1]
    eviction = next(row for row in records if row["event"] == "withdraw")
    assert eviction["key"] == b"a" and eviction["reason"] == "capacity_eviction"
    first_ready = next(row for row in records if row["event"] == "ready")
    assert eviction["generation"] == first_ready["generation"]
    get = records[-1]
    assert get["miss_reasons"] == ["absent", "ready"]
    assert get["export_sequence"] > eviction["export_sequence"]
    local.release_export_claims(tuple(claim for _, claim in claims))
    cache.close()
    assert records[-1]["reason"] == "shutdown"


@pytest.mark.parametrize(
    "state,requested,declaration,reason",
    [
        ("retiring", ("pool",), ("pool",), "retired"),
        ("ready", ("pool",), ("other",), "key_layout_mismatch"),
        ("ready", ("other",), ("pool",), "missing_label_layout"),
        ("incomplete", ("pool",), ("pool",), "incomplete_label"),
    ],
)
def test_per_key_miss_reason_keeps_generation(
    observed, state, requested, declaration, reason
):
    local, records = observed
    residency = _LocalDramResidency([("pool", 0)], _LocalDramState.READY)
    local._publish_export(b"key", residency)
    if state == "retiring":
        residency.retire_on_release = True
    if state == "incomplete":
        local._exports[b"key"] = _GuardExport(residency, ("pool",), ("pool",), (None,))
    claims, busy = local.claim_export_subset((b"key",), (requested,), declaration)
    assert not busy and not claims and residency.export_readers == 0
    assert records[-1]["miss_reasons"] == [reason]
    assert records[-1]["generations"] == [residency.telemetry_generation]


def test_retirement_reused_key_and_shutdown_have_exact_lineage(
    source_factory, tmp_path, monkeypatch
):
    output = telemetry.BufferedWriter(
        tmp_path, source_archive_sha256="test", start_thread=False
    )
    monkeypatch.setattr(telemetry, "writer", lambda: output)
    source, _, _, _, _ = source_factory()
    local = source._core._local_dram
    journal = output.journal

    def unlocked(record):
        assert not local._export_lock.locked()
        return journal(record)

    monkeypatch.setattr(output, "journal", unlocked)
    try:
        source.deposit({b"b": [_mem_descriptor(1, end_point_name="source")]})
        assert _poll_until(source, bool)
        with source._core._state_lock:
            local.acquire_sources((b"b",))
            local.retire_sources((b"b",))
            local.release_sources((b"b",))
        source.deposit({b"a": [_mem_descriptor(0, end_point_name="source")]})
        assert _poll_until(source, bool)
    finally:
        source.close()
        output.close()
    records = list(telemetry.iter_key_journal(output.key_path))
    assert {row["reason"] for row in records if row["event"] == "withdraw"} == {
        "capacity_eviction",
        "policy_retirement",
        "shutdown",
    }
    generations = {
        row["generation"]
        for row in records
        if row["event"] == "ready" and row["key"] == b"a"
    }
    assert len(generations) == 2
    assert telemetry.validate_capture(tmp_path, ["source"])["valid"]
