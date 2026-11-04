# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
from unittest.mock import Mock

import msgspec
import pytest
from _gpu_test_utils import (
    DOMAIN,
    TLS,
    GPUAgent,
    GPUPinning,
    gpu_envelope,
    gpu_hint,
    new_gpu,
    refs,
    regions,
    source_message,
)
from _kvcr_test_utils import (
    FakeBytesControl,
    _decode_control_message,
    _op_entries,
    _poll_until,
    _use_nixl_agent,
    _wait_until,
)

from kvcr import KVCR, KVCRBindings
from kvcr.config import (
    G3Options,
    KVCRBackendConfigs,
    KVCRConfig,
    KVCRGuardConfig,
    LocalDramOptions,
    RemoteFWGPUOptions,
)
from kvcr.types import BlockKey, CacheTier


@pytest.mark.parametrize(
    "change",
    ["host", "empty", "local", "disk", "guard", "tls", "zero", "device", "control"],
)
def test_gpu_constructor_fails_before_native_registration(
    monkeypatch, tmp_path, change
):
    monkeypatch.setenv("UCX_TLS", "tcp,cuda" if change == "tls" else TLS)
    monkeypatch.setenv("NIXL_UCX_CUDA_IPC_ONLY", "1")
    agent = GPUAgent()
    rs = regions("target")
    options = dict(
        framework_regions=rs,
        remote_fw_gpu=RemoteFWGPUOptions(compatibility_domain=DOMAIN),
    )
    guard = None
    if change == "host":
        options["framework_regions"] = [msgspec.structs.replace(rs[0], mem_type="DRAM")]
    if change == "empty":
        options["framework_regions"] = []
    if change == "zero":
        options["framework_regions"] = [msgspec.structs.replace(rs[0], addr=0)]
    if change == "device":
        options["framework_regions"] = [msgspec.structs.replace(rs[0], device_Id=-1)]
    if change == "local":
        options["local_dram"] = LocalDramOptions([("main", 3000, 32)])
    if change == "disk":
        options["g3"] = G3Options(paths=(tmp_path / "g3",), capacity_bytes_per_file=32)
    if change == "guard":
        guard = KVCRGuardConfig(
            kvcr_service_socket_path="/not-used",
            guard_index=0,
            compatibility_digest="test",
        )
    with _use_nixl_agent(agent), pytest.raises(ValueError):
        KVCR(
            KVCRConfig(
                nixl_agent_name="target", pool_layouts=[("main", 16), ("index", 8)]
            ),
            KVCRBindings(
                Mock(),
                Mock(),
                Mock(),
                framework_control=None if change == "control" else FakeBytesControl(),
            ),
            KVCRBackendConfigs(**options),
            guard,
        )
    assert agent.registrations == agent.xfers == []


def test_gpu_source_preserves_fragment_device_indices_and_pin_until_done(monkeypatch):
    monkeypatch.setattr(
        ctypes, "memmove", Mock(side_effect=AssertionError("host payload copy"))
    )
    pinning = GPUPinning(ready=False)
    kvcr, agent, _, control = new_gpu(monkeypatch, owner="source", pinning=pinning)
    control.incoming.append(msgspec.msgpack.encode(source_message()))
    _poll_until(kvcr, lambda _: bool(pinning.searches))
    assert agent.xfers == []  # Producer readiness has not been published.
    pinning.completed.append(pinning.saved)
    _poll_until(kvcr, lambda _: bool(agent.xfers))
    _, src, _, dst, _, _ = agent.xfers[0]
    assert src == [(1320, 16, 3), (2160, 8, 3), (1064, 16, 3), (2032, 8, 3)]
    assert dst == [(9160, 16, 5), (10080, 8, 5), (9480, 16, 5), (10240, 8, 5)]
    assert set(agent.prep_types) == {"VRAM"}
    assert {kind for _, kind in agent.registrations} == {"VRAM"}
    assert pinning.unpins == []
    record = kvcr._core._block_record_map[BlockKey(b"k0")]
    meta = kvcr._core._block_meta(BlockKey(b"k0"), record, 24)
    assert meta.resident_tiers == frozenset({CacheTier.FW_G1})
    assert meta.size_bytes == 24
    agent.state = "DONE"
    _poll_until(kvcr, lambda _: bool(pinning.unpins))
    assert pinning.unpins == ["pin"]
    assert agent.released_xfers == [1]
    ctypes.memmove.assert_not_called()


@pytest.mark.parametrize(
    "change", ["legacy", "domain", "memory", "transport", "incarnation", "refused"]
)
def test_gpu_metadata_rejection_precedes_destination_offer(monkeypatch, change):
    errors = []
    kvcr, agent, _, control = new_gpu(monkeypatch, on_resilience_event=errors.append)
    kvcr.submit_hint(gpu_envelope(), request_id="req")
    handle = kvcr.deliver({BlockKey(b"k0"): refs("target", 2)}, "req")
    _wait_until(lambda: bool(control.sent))
    response = {
        "type": "target_metadata_ack",
        "op_handle": handle,
        "sender_control_endpoint": "tcp://source:1",
        "sender_incarnation": "legacy",
        "gpu_peer_accepted": True,
        "gpu_capability_only": True,
        **{
            key: gpu_hint()["actions"][0]["payload"][key]
            for key in (
                "source_memory_kind",
                "target_memory_kind",
                "transport",
                "compatibility_domain",
            )
        },
    }
    if change == "legacy":
        for name in (
            "source_memory_kind",
            "target_memory_kind",
            "transport",
            "compatibility_domain",
        ):
            response.pop(name)
    if change == "domain":
        response["compatibility_domain"] = "other"
    if change == "memory":
        response["source_memory_kind"] = "cpu"
    if change == "transport":
        response["transport"] = "legacy"
    if change == "incarnation":
        response.pop("sender_incarnation")
    if change == "refused":
        response["gpu_peer_accepted"] = False
    control.incoming.append(msgspec.msgpack.encode(response))
    assert _poll_until(kvcr, bool) == [(handle, _op_entries({BlockKey(b"k0"): False}))]
    sent = [_decode_control_message(raw) for _, raw in control.sent]
    assert all(row["type"] == "target_metadata" for row in sent)
    assert all(
        not {"target_agent_metadata", "target_regions", "dst_descriptors"}.intersection(
            row
        )
        for row in sent
    )
    assert agent.remote_agents == agent.xfers == errors == []
    assert not kvcr._core._block_record_map
