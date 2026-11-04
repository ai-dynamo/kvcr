# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading

import msgspec
import pytest
from _gpu_test_utils import (
    DOMAIN,
    acknowledge_gpu_peer,
    gpu_envelope,
    gpu_hint,
    new_gpu,
    refs,
    source_message,
)
from _kvcr_test_utils import (
    _decode_control_message,
    _op_entries,
    _poll_until,
    _wait_until,
    _write_done_notification,
)

from kvcr.backends.remote_ops import _TargetMetadataAck, _TargetPullState
from kvcr.types import BlockKey


def test_legacy_gpu_metadata_timeout_has_no_destination_hazard(monkeypatch):
    errors = []
    kvcr, agent, _, control = new_gpu(monkeypatch, on_resilience_event=errors.append)
    clock = [0.0]
    kvcr._core._clock = lambda: clock[0]
    kvcr.submit_hint(gpu_envelope(), request_id="req")
    handle = kvcr.deliver({BlockKey(b"k0"): refs("target", 2)}, "req")
    _wait_until(lambda: bool(control.sent))
    clock[0] = 1.0
    assert _poll_until(kvcr, bool) == [(handle, _op_entries({BlockKey(b"k0"): False}))]
    assert all(
        _decode_control_message(raw)["type"] == "target_metadata"
        for _, raw in control.sent
    )
    assert agent.remote_agents == agent.xfers == errors == []


def test_gpu_source_rejects_metadata_domain_before_native_connection(monkeypatch):
    kvcr, agent, pinning, control = new_gpu(monkeypatch, owner="source")
    message = source_message()
    message.update(
        type="target_metadata", gpu_capability_only=True, compatibility_domain="other"
    )
    control.incoming.append(msgspec.msgpack.encode(message))
    _wait_until(lambda: bool(control.sent))
    reply = _decode_control_message(control.sent[0][1])
    assert reply["gpu_peer_accepted"] is False
    assert reply["compatibility_domain"] == DOMAIN
    assert agent.remote_agents == agent.xfers == pinning.searches == []
    assert not {
        "target_agent_metadata",
        "target_regions",
        "dst_descriptors",
    }.intersection(reply)
    list(kvcr.poll_completed())


def test_gpu_source_capability_reply_exposes_no_native_memory(monkeypatch):
    kvcr, agent, pinning, control = new_gpu(monkeypatch, owner="source")
    contract = gpu_hint()["actions"][0]["payload"]
    control.incoming.append(
        msgspec.msgpack.encode(
            {
                "type": "target_metadata",
                "op_handle": 12,
                "gpu_capability_only": True,
                "sender_control_endpoint": "tcp://target:1",
                **{
                    key: contract[key]
                    for key in (
                        "source_memory_kind",
                        "target_memory_kind",
                        "transport",
                        "compatibility_domain",
                    )
                },
            }
        )
    )
    _wait_until(lambda: bool(control.sent))
    reply = _decode_control_message(control.sent[-1][1])
    assert reply["type"] == "target_metadata_ack"
    assert reply["op_handle"] == 12 and reply["gpu_peer_accepted"] is True
    assert reply["sender_control_endpoint"] == "tcp://source:1"
    assert reply["sender_incarnation"]
    assert not {
        "target_agent_metadata",
        "target_regions",
        "dst_descriptors",
    }.intersection(reply)
    assert agent.remote_agents == agent.xfers == pinning.searches == []
    list(kvcr.poll_completed())


@pytest.mark.parametrize("change", ["operation", "endpoint"])
def test_gpu_metadata_ack_must_match_actual_operation_and_source(monkeypatch, change):
    kvcr, agent, _, control = new_gpu(monkeypatch)
    clock = [0.0]
    kvcr._core._clock = lambda: clock[0]
    kvcr.submit_hint(gpu_envelope(), request_id="req")
    handle = kvcr.deliver({BlockKey(b"k0"): refs("target", 2)}, "req")
    _wait_until(lambda: bool(control.sent))
    contract = gpu_hint()["actions"][0]["payload"]
    control.incoming.append(
        msgspec.msgpack.encode(
            {
                "type": "target_metadata_ack",
                "op_handle": handle + 1 if change == "operation" else handle,
                "sender_control_endpoint": "tcp://other:1"
                if change == "endpoint"
                else "tcp://source:1",
                "sender_incarnation": "source-generation",
                "gpu_peer_accepted": True,
                "gpu_capability_only": True,
                **{
                    key: contract[key]
                    for key in (
                        "source_memory_kind",
                        "target_memory_kind",
                        "transport",
                        "compatibility_domain",
                    )
                },
            }
        )
    )
    _wait_until(lambda: not control.incoming)
    assert list(kvcr.poll_completed()) == []
    clock[0] = 1.0
    assert _poll_until(kvcr, bool) == [(handle, _op_entries({BlockKey(b"k0"): False}))]
    assert all(
        _decode_control_message(raw)["type"] == "target_metadata"
        for _, raw in control.sent
    )
    assert agent.remote_agents == agent.xfers == []


@pytest.mark.parametrize("change", ["refused", "domain", "incarnation"])
def test_gpu_late_invalid_capability_ack_cannot_complete_an_offered_destination(
    monkeypatch, change
):
    errors = []
    kvcr, agent, _, control = new_gpu(monkeypatch, on_resilience_event=errors.append)
    core = kvcr._core
    core._clock = lambda: 0.0
    kvcr.submit_hint(gpu_envelope(), request_id="offered")
    handle = kvcr.deliver({BlockKey(b"k0"): refs("target", 2)}, "offered")
    _wait_until(lambda: bool(control.sent))
    acknowledge_gpu_peer(control, handle)
    backend = core._remote_fw_dram
    op_id = ("target", handle)
    _wait_until(
        lambda: op_id in core._progress._in_flight_ops
        and core._progress._in_flight_ops[op_id].state
        is _TargetPullState.WAITING_WRITE_DONE
    )
    handled = threading.Event()
    original = backend._handle_target_metadata_ack

    def observe_ack(progress, reply):
        try:
            original(progress, reply)
        finally:
            handled.set()

    monkeypatch.setattr(backend, "_handle_target_metadata_ack", observe_ack)
    reply = _TargetMetadataAck(
        op_handle=handle,
        sender_control_endpoint="tcp://source:1",
        sender_incarnation="source-generation",
        gpu_peer_accepted=True,
        gpu_capability_only=True,
        **backend._gpu_contract(),
    )
    if change == "refused":
        reply.gpu_peer_accepted = False
    elif change == "domain":
        reply.compatibility_domain = "other"
    else:
        reply.sender_incarnation = "stale"
    control.incoming.append(msgspec.msgpack.encode(reply))
    try:
        _wait_until(handled.is_set)
        assert list(kvcr.poll_completed()) == []
        assert core._block_record_map[BlockKey(b"k0")].active_op_ids
        assert core._progress._in_flight_ops[op_id].state is (
            _TargetPullState.WAITING_WRITE_DONE
        )
        assert errors == []
    finally:
        agent.notifs["source"] = [_write_done_notification(handle)]
        completed = _poll_until(kvcr, lambda _: core._outstanding_operations == 0)
    assert completed == [(handle, _op_entries({BlockKey(b"k0"): True}))]
    assert not core._block_record_map
    assert errors == []
