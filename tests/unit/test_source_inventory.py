# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The router's logical hint must not stand in for a Mamba source checkpoint."""

import ctypes
import time

from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _ConstantHashAdapter,
    _new_kvcr,
    _router_hint,
)

from kvcr.config import KVCRConfig, LocalDramOptions
from kvcr.core import _BlockRecord
from kvcr.local_dram import _LocalDramResidency, _LocalDramState
from kvcr.types import BlockKey


def test_exact_source_inventory_rejects_missing_middle_mamba_checkpoint():
    memory = ctypes.create_string_buffer(16 * 4)
    source_control = FakeBytesControl("tcp://source:1")
    target_control = FakeBytesControl("tcp://target:1")
    config = KVCRConfig(nixl_agent_name="source", pool_layouts=[("mamba", 16)])
    source = _new_kvcr(
        FakeNixlAgent(),
        FakePrimaryPinning(),
        source_control,
        config,
        name="source",
        local_dram=LocalDramOptions([("mamba", ctypes.addressof(memory), len(memory))]),
    )
    target = _new_kvcr(
        FakeNixlAgent(),
        FakePrimaryPinning(),
        target_control,
        config,
        name="target",
        key_adapter=_ConstantHashAdapter(),
    )
    middle, full = BlockKey(b"middle"), BlockKey(b"full")
    # Both keys match the logical router hint. The middle checkpoint has only
    # one of two physical Mamba components, so it must not count as reusable.
    source._core._block_record_map[middle] = _BlockRecord(
        local_dram=_LocalDramResidency([("mamba:conv", 2)], _LocalDramState.READY)
    )
    source._core._block_record_map[full] = _BlockRecord(
        local_dram=_LocalDramResidency(
            [("mamba:conv", 0), ("mamba:ssm", 1)], _LocalDramState.READY
        )
    )
    try:
        target.submit_hint(_router_hint("tcp://source:1"), request_id="scope")
        assert [status.name for status, _ in target.query([middle, full], "scope")] == [
            "FETCHABLE",
            "FETCHABLE",
        ]
        handle = target.probe_remote_source(
            [middle, full], "scope", ["mamba:conv", "mamba:ssm"]
        )
        deadline = time.monotonic() + 2
        entries = None
        while time.monotonic() < deadline and entries is None:
            for control, peer in (
                (target_control, source_control),
                (source_control, target_control),
            ):
                while control.sent:
                    endpoint, message = control.sent.pop(0)
                    if endpoint == peer.endpoint:
                        peer.incoming.append(message)
            source.poll_completed()
            for completed_handle, result in target.poll_completed():
                if completed_handle == handle:
                    entries = result
            time.sleep(0.001)
        assert entries is not None
        assert not entries[middle].success
        assert entries[full].success
    finally:
        target.close()
        source.close()
