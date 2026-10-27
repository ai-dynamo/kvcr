# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Guard-only READY-subset serving and native reader lifetime."""

import ctypes

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

from kvcr.config import LocalDramOptions, RemoteFWDramOptions
from kvcr.core import _BlockRecord
from kvcr.local_dram import _LocalDramResidency, _LocalDramState
from kvcr.progress import _TransferRef
from kvcr.recovery_journal import install_recovery_records
from kvcr.types import RegionDescriptor


@pytest.fixture
def source_factory():
    opened = []

    def make(recovered=False):
        memory = ctypes.create_string_buffer(b"a" * 16 + b"b" * 16)
        address = ctypes.addressof(memory)
        agent, control, pinning = (
            FakeNixlAgent(),
            FakeBytesControl(),
            FakePrimaryPinning(),
        )
        source = _new_kvcr(
            agent,
            pinning,
            control,
            name="source",
            remote_options=RemoteFWDramOptions(guard_only_sources=True),
            framework_regions=[RegionDescriptor(addr=address, size=16, count=2)],
            local_dram=LocalDramOptions([("", address, 16)]),
        )
        opened.append((source, agent, memory))
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
            op = source.deposit({b"a": [_mem_descriptor(0, end_point_name="source")]})
            assert _poll_until(source, bool) == [(op, _op_entries({b"a": True}))]
        return source, agent, control, pinning, address

    yield make
    for source, agent, _ in reversed(opened):
        agent.state = "DONE"
        source.close()


def _request(control, keys, *, handle=12, **extra):
    payload = msgspec.msgpack.decode(
        _start_write_message(handle, keys[0], target_agent="target")
    )
    payload.update(
        keys=keys,
        dst_descriptors=[[_TransferRef("target", i)] for i in range(len(keys))],
        **extra,
    )
    control.incoming.append(msgspec.msgpack.encode(payload))


@pytest.mark.parametrize("recovered", [False, True])
def test_ready_subset_serves_without_framework_handoff(source_factory, recovered):
    assert not RemoteFWDramOptions().guard_only_sources
    source, agent, control, pinning, address = source_factory(recovered)
    before = len(agent.xfers)
    with source._core._state_lock:
        _request(control, [b"missing", b"a"])
        _wait_until(lambda: len(agent.xfers) == before + 1)
        assert agent.xfers[-1][1] == [(address, 16, 0)]
        assert _decode_notif(agent.xfers[-1][5])["completed_indices"] == [1]
        assert not source._core._remote_fw_dram._source_pin_ops
        assert pinning.searches == []
    assert source._core._local_dram._export_readers() == 0


def test_native_read_fences_slot_reuse_until_done(source_factory):
    source, agent, control, _, _ = source_factory()
    local = source._core._local_dram
    agent.state = "PROC"
    before = len(agent.xfers)
    _request(control, [b"a"])
    _wait_until(lambda: len(agent.xfers) == before + 1)
    assert local._export_readers() == 1
    blocks = {b"replace": [_mem_descriptor(1, end_point_name="source")]}
    blocked = source.deposit(blocks)
    assert _poll_until(source, bool) == [(blocked, _op_entries({b"replace": False}))]
    agent.state = "DONE"
    with source._core._state_lock:
        _wait_until(lambda: local._export_readers() == 0)
    replacement = source.deposit(blocks)
    assert _poll_until(source, bool) == [(replacement, _op_entries({b"replace": True}))]
