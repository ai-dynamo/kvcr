# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Validate only claimed destinations and preserve native source ownership."""

import msgspec
from _kvcr_test_utils import (
    _decode_notif,
    _mem_descriptor,
    _poll_until,
    _start_write_message,
    _wait_until,
)
from test_no_pinning_mode import _request
from test_no_pinning_mode import source_factory as source_factory

from kvcr.progress import _TransferRef


def test_guard_validates_only_claimed_subset_after_claim(source_factory, monkeypatch):
    source, agent, control, pinning, _ = source_factory()
    core = source._core
    validate = core._validate_descriptors
    seen = []

    def checked(descriptors, **kwargs):
        seen.append((core._local_dram._export_readers(), tuple(descriptors)))
        return validate(descriptors, **kwargs)

    monkeypatch.setattr(core, "_validate_descriptors", checked)
    payload = msgspec.msgpack.decode(
        _start_write_message(901, b"a", target_agent="target")
    )
    payload.update(
        keys=[b"missing", b"a"],
        dst_descriptors=[[_TransferRef("target", 1000)], [_TransferRef("target", 1)]],
    )
    before = len(agent.xfers)
    try:
        control.incoming.append(msgspec.msgpack.encode(payload))
        _wait_until(lambda: bool(seen))
        assert seen == [(1, (_TransferRef("target", 1),))]
        _wait_until(lambda: len(agent.xfers) == before + 1)
        assert _decode_notif(agent.xfers[-1][5])["completed_indices"] == [1]
        assert not pinning.searches
    finally:
        source.close()


def test_selected_descriptor_failure_releases_claim_before_submission(
    source_factory, monkeypatch
):
    source, agent, control, pinning, _ = source_factory()
    core = source._core
    observed = []

    def invalid(*args, **kwargs):
        observed.append(core._local_dram._export_readers())
        raise ValueError("selected destination rejected")

    monkeypatch.setattr(core, "_validate_descriptors", invalid)
    before = len(agent.xfers)
    try:
        _request(control, [b"a"], handle=902)
        _wait_until(lambda: bool(observed))
        assert observed == [1]
        _wait_until(lambda: core._local_dram._export_readers() == 0)
        assert len(agent.xfers) == before
        assert not core._remote_fw_dram._dangling_ops.source_writes
        assert not pinning.searches
    finally:
        source.close()


def test_first_ready_time_survives_export_eviction(source_factory):
    source, _, _, _, _ = source_factory()
    local = source._core._local_dram
    first_ready = local._first_guard_ready_at
    assert first_ready is not None
    source.deposit({b"b": [_mem_descriptor(1, end_point_name="source")]})
    assert _poll_until(source, bool)
    assert b"a" not in local._exports and b"b" in local._exports
    assert local._first_guard_ready_at == first_ready
