# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-only protocol/lifetime tests with opaque addresses and fake native agents.

These tests never allocate/read CUDA or host payload buffers. Actual CUDA IPC,
bitwise transfer and no-host-bounce qualification require an allocated GPU run.
"""

import ctypes
from unittest.mock import Mock

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _ConstantHashAdapter,
    _decode_control_message,
    _decode_notif,
    _new_kvcr,
    _op_entries,
    _poll_until,
    _use_nixl_agent,
    _wait_until,
    _write_done_notification,
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
from kvcr.hint_parser import _KVFetchHint, _KVGPUFetchHint
from kvcr.progress import _KVCRProgress
from kvcr.types import BlockKey, CacheTier, MemoryRef, QueryStatus, RegionDescriptor

DOMAIN = "test-host:cuda-ipc:layout-v1"
TLS = "cuda_ipc,posix,self"


def gpu_hint(*, domain=DOMAIN):
    return {
        "protocol_version": "0.1",
        "actions": [
            {
                "action_type": "kv.fetch_gpu",
                "action_version": "1.0",
                "payload": {
                    "source_control_endpoint": "tcp://source:1",
                    "block_hashes": [123],
                    "source_memory_kind": "gpu",
                    "target_memory_kind": "gpu",
                    "transport": "nixl_gpu_direct",
                    "compatibility_domain": domain,
                },
            }
        ],
    }


def refs(owner, index):
    return [
        MemoryRef(end_point_name=owner, label=label, element_index=index)
        for label in ("main", "index")
    ]


def regions(owner):
    source = owner == "source"
    return [
        RegionDescriptor(
            addr=1000 if source else 9000,
            size=16,
            stride=64 if source else 80,
            count=8,
            mem_type="VRAM",
            device_Id=3 if source else 5,
            label="main",
        ),
        RegionDescriptor(
            addr=2000 if source else 10000,
            size=8,
            stride=32 if source else 40,
            count=8,
            mem_type="VRAM",
            device_Id=3 if source else 5,
            label="index",
        ),
    ]


class GPUAgent(FakeNixlAgent):
    def __init__(self):
        super().__init__()
        self.prep_types = []

    def get_backend_mem_types(self, backend):
        assert backend == "UCX"
        return ["DRAM", "VRAM"]

    def prep_xfer_dlist(self, agent_name, descs, *, mem_type, backends):
        self.prep_types.append(mem_type)
        return super().prep_xfer_dlist(
            agent_name, descs, mem_type=mem_type, backends=backends
        )

    def transfer(self, handle):
        # Opaque CUDA addresses must never be dereferenced by the fake backend.
        self.transfers.append(handle)
        return "PROC"


class GPUPinning(FakePrimaryPinning):
    def __init__(self, *, ready=True, missing=()):
        super().__init__()
        self.ready = ready
        self.missing = missing

    def request_pin(self, keys):
        request = super().request_pin(keys)
        _, (_, mapping) = self.completed.pop()
        self.saved = (
            request,
            (
                "pin",
                {
                    key: None if i in self.missing else refs(self.agent_name, (5, 1)[i])
                    for i, key in enumerate(mapping)
                },
            ),
        )
        if self.ready:
            self.completed.append(self.saved)
        return request


def new_gpu(monkeypatch, *, owner="target", agent=None, pinning=None, **kwargs):
    monkeypatch.setenv("UCX_TLS", TLS)
    monkeypatch.setenv("NIXL_UCX_CUDA_IPC_ONLY", "1")
    agent = agent or GPUAgent()
    pinning = pinning or GPUPinning()
    control = FakeBytesControl(f"tcp://{owner}:1")
    kvcr = _new_kvcr(
        agent,
        pinning,
        control,
        KVCRConfig(nixl_agent_name=owner, pool_layouts=[("main", 16), ("index", 8)]),
        name=owner,
        key_adapter=_ConstantHashAdapter(),
        framework_regions=regions(owner),
        remote_gpu_options=RemoteFWGPUOptions(
            compatibility_domain=DOMAIN, eager_ctrl_connect=False
        ),
        **kwargs,
    )
    return kvcr, agent, pinning, control


def source_message():
    return {
        "type": "start_write_gpu",
        "op_handle": 12,
        "remaining_timeout_ms": 1000,
        "target_agent": "target",
        "target_agent_metadata": b"target-md",
        "target_regions": ({r.label: r for r in regions("target")}, {}),
        "sender_control_endpoint": "tcp://target:1",
        "source_control_endpoint": "tcp://source:1",
        "keys": [b"k0", b"k1"],
        "dst_descriptors": [refs("target", 2), refs("target", 6)],
        "source_memory_kind": "gpu",
        "target_memory_kind": "gpu",
        "transport": "nixl_gpu_direct",
        "compatibility_domain": DOMAIN,
    }


def acknowledge_gpu_peer(control, handle):
    contract = gpu_hint()["actions"][0]["payload"]
    control.incoming.append(
        msgspec.msgpack.encode(
            {
                "type": "target_metadata_ack",
                "sender_control_endpoint": "tcp://source:1",
                "sender_incarnation": "source-generation",
                "op_handle": handle,
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
    _wait_until(
        lambda: any(
            _decode_control_message(raw)["type"] == "start_write_gpu"
            for _, raw in control.sent
        )
    )


def test_gpu_hint_is_strict_and_isolated_from_cpu():
    parsed = _KVGPUFetchHint.from_hint(gpu_hint())
    assert (
        parsed.source_endpoint,
        parsed.block_hashes,
        parsed.compatibility_domain,
    ) == ("tcp://source:1", frozenset({123}), DOMAIN)
    with pytest.raises(ValueError):
        _KVFetchHint.from_hint(gpu_hint())
    legacy = gpu_hint()
    legacy["actions"][0]["action_type"] = "kv.fetch"
    with pytest.raises(ValueError):
        _KVGPUFetchHint.from_hint(legacy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_memory_kind", "cpu"),
        ("target_memory_kind", "cpu"),
        ("transport", "nixl"),
        ("compatibility_domain", ""),
        ("block_hashes", [True]),
        ("source_control_endpoint", ""),
    ],
)
def test_gpu_hint_rejects_unsafe_payload(field, value):
    hint = gpu_hint()
    hint["actions"][0]["payload"][field] = value
    with pytest.raises(ValueError):
        _KVGPUFetchHint.from_hint(hint)


@pytest.mark.parametrize(
    "field,value", [("protocol_version", "9"), ("action_version", "9")]
)
def test_gpu_hint_rejects_unknown_version(field, value):
    hint = gpu_hint()
    (hint if field == "protocol_version" else hint["actions"][0])[field] = value
    with pytest.raises(ValueError):
        _KVGPUFetchHint.from_hint(hint)


@pytest.mark.parametrize(
    "tls",
    [
        (),
        ("tcp", "cuda_ipc"),
        ("all",),
        ("cuda_ipc", "cuda_copy"),
        ("cuda_ipc", "gdr_copy"),
        ("posix",),
        ("cuda_ipc", "cuda_ipc"),
    ],
)
def test_gpu_options_exclude_host_payload_fallback(tls):
    with pytest.raises(ValueError):
        RemoteFWGPUOptions(compatibility_domain=DOMAIN, ucx_tls=tls)


@pytest.mark.parametrize(
    "change", ["host", "empty", "local", "disk", "guard", "tls", "zero", "device"]
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
            KVCRBindings(Mock(), Mock(), Mock(), framework_control=FakeBytesControl()),
            KVCRBackendConfigs(**options),
            guard,
        )
    assert agent.registrations == agent.xfers == []


def test_gpu_native_init_rechecks_tls_before_agent(monkeypatch):
    monkeypatch.setenv("UCX_TLS", "tcp,cuda")
    progress = _KVCRProgress(
        Mock(),
        Mock(),
        list,
        Mock(),
        nixl_agent_name="gpu",
        nixl_listen_port=1,
        required_gpu_ucx_tls=("cuda_ipc", "posix", "self"),
    )
    with pytest.raises(RuntimeError, match="UCX_TLS changed"):
        progress._initialize_nixl()
    assert progress._nixl_agent is None


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
    agent.state = "DONE"
    _poll_until(kvcr, lambda _: bool(pinning.unpins))
    assert pinning.unpins == ["pin"]
    assert agent.released_xfers == [1]
    ctypes.memmove.assert_not_called()


@pytest.mark.parametrize("change", ["legacy", "domain", "transport", "host", "managed"])
def test_gpu_source_refuses_wrong_contract_and_host_peer_without_dma(
    monkeypatch, change
):
    kvcr, agent, pinning, control = new_gpu(monkeypatch, owner="source")
    payload = source_message()
    if change == "legacy":
        payload["type"] = "start_write"
    if change == "domain":
        payload["compatibility_domain"] = "other"
    if change == "transport":
        payload["transport"] = "cpu"
    if change == "host":
        payload["target_regions"][0]["main"] = msgspec.structs.replace(
            regions("target")[0], mem_type="DRAM"
        )
    if change == "managed":
        payload["target_regions"][1]["cpu"] = regions("target")[0]
    control.incoming.append(msgspec.msgpack.encode(payload))
    _wait_until(lambda: bool(agent.sent_notifs) or bool(control.sent))
    assert agent.xfers == pinning.searches == []
    assert agent.remote_agents == []


def test_cpu_source_never_executes_new_gpu_message():
    agent, control = FakeNixlAgent(), FakeBytesControl("tcp://source:1")
    kvcr = _new_kvcr(agent, FakePrimaryPinning(), control, name="source")
    control.incoming.append(msgspec.msgpack.encode(source_message()))
    _wait_until(lambda: bool(agent.sent_notifs) or bool(control.sent))
    assert agent.xfers == []
    assert _decode_notif(agent.sent_notifs[-1][1])["success"] is False
    list(kvcr.poll_completed())


def test_gpu_target_emits_distinct_wire_and_rejects_domain_and_storage(monkeypatch):
    kvcr, agent, _, control = new_gpu(monkeypatch)
    with pytest.raises(ValueError, match="domain"):
        kvcr.submit_hint(gpu_hint(domain="wrong"), request_id="bad")
    kvcr.submit_hint(gpu_hint(), request_id="req")
    assert kvcr.query([BlockKey(b"k0")], "req") == [
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G1)
    ]
    for call in (
        lambda: kvcr.deposit({b"k0": refs("target", 2)}),
        lambda: kvcr.fetch([b"k0"]),
    ):
        with pytest.raises(ValueError, match="no KVCR-managed"):
            call()
    handle = kvcr.deliver({BlockKey(b"k0"): refs("target", 2)}, "req")
    _wait_until(lambda: bool(control.sent))
    probe = _decode_control_message(control.sent[-1][1])
    assert probe["type"] == "target_metadata"
    assert not {
        "target_agent_metadata",
        "target_regions",
        "dst_descriptors",
    }.intersection(probe)
    acknowledge_gpu_peer(control, handle)
    message = _decode_control_message(control.sent[-1][1])
    assert message["type"] == "start_write_gpu"
    assert message["compatibility_domain"] == DOMAIN
    assert message["source_memory_kind"] == message["target_memory_kind"] == "gpu"
    assert list(kvcr.poll_completed()) == []
    agent.notifs["source"] = [_write_done_notification(handle)]
    assert _poll_until(kvcr, bool) == [(handle, _op_entries({BlockKey(b"k0"): True}))]


def test_gpu_target_timeout_quarantines_until_actual_terminal(monkeypatch):
    errors = []
    kvcr, agent, _, control = new_gpu(monkeypatch, on_resilience_event=errors.append)
    clock = [0.0]
    kvcr._core._clock = lambda: clock[0]
    kvcr.submit_hint(gpu_hint(), request_id="req")
    handle = kvcr.deliver({BlockKey(b"k0"): refs("target", 2)}, "req")
    _wait_until(lambda: bool(control.sent))
    acknowledge_gpu_peer(control, handle)
    clock[0] = 1.0
    _wait_until(
        lambda: any(
            _decode_control_message(raw)["type"] == "write_probe"
            for _, raw in control.sent
        )
    )
    assert list(kvcr.poll_completed()) == []
    clock[0] = 5.0
    _poll_until(kvcr, lambda _: bool(errors))
    assert list(kvcr.poll_completed()) == []
    assert [e.state for e in errors] == ["uncertain"]
    assert kvcr._core._block_record_map[BlockKey(b"k0")].active_op_ids
    agent.notifs["source"] = [_write_done_notification(handle)]
    assert _poll_until(kvcr, bool) == [(handle, _op_entries({BlockKey(b"k0"): False}))]
    assert [e.state for e in errors] == ["uncertain", "quiesced"]
    assert (
        list(kvcr.poll_completed()) == []
    )  # A late success never validates canceled data.


def test_gpu_partial_pin_preserves_original_target_key_position(monkeypatch):
    kvcr, agent, pinning, control = new_gpu(
        monkeypatch, owner="source", pinning=GPUPinning(missing=(0,))
    )
    agent.state = "DONE"
    control.incoming.append(msgspec.msgpack.encode(source_message()))
    _poll_until(kvcr, lambda _: bool(pinning.unpins))
    _, src, _, dst, _, notification = agent.xfers[0]
    assert src == [(1064, 16, 3), (2032, 8, 3)]
    assert dst == [(9480, 16, 5), (10240, 8, 5)]
    assert _decode_notif(notification)["completed_indices"] == [1]
    assert pinning.unpins == ["pin"]


def test_gpu_source_timeout_keeps_allocation_pinned_until_native_done(monkeypatch):
    errors = []
    kvcr, agent, pinning, control = new_gpu(
        monkeypatch, owner="source", on_resilience_event=errors.append
    )
    clock = [0.0]
    kvcr._core._clock = lambda: clock[0]
    try:
        control.incoming.append(msgspec.msgpack.encode(source_message()))
        _poll_until(kvcr, lambda _: bool(agent.xfers))
        clock[0] = 1.0
        _wait_until(lambda: bool(agent.sent_notifs))
        assert agent.released_xfers == pinning.unpins == []
        clock[0] = 5.0
        _poll_until(kvcr, lambda _: bool(errors))
        assert [e.state for e in errors] == ["uncertain"]
        assert agent.released_xfers == pinning.unpins == []
        assert kvcr._core._framework_pin_keys
        agent.state = "DONE"
        _poll_until(kvcr, lambda _: bool(pinning.unpins))
        assert [e.state for e in errors] == ["uncertain", "quiesced"]
        assert agent.released_xfers == [1]
        assert pinning.unpins == ["pin"]
    finally:
        agent.state = "DONE"
        kvcr.close()


def test_gpu_native_backend_without_vram_support_fails_closed(monkeypatch):
    agent = GPUAgent()
    agent.get_backend_mem_types = lambda _: ["DRAM"]
    with pytest.raises(RuntimeError, match="native UCX VRAM"):
        new_gpu(monkeypatch, agent=agent)
    assert agent.xfers == agent.registrations == []


@pytest.mark.parametrize("value", [None, "0", "true", ""])
def test_gpu_requires_explicit_native_direct_policy_flag(monkeypatch, value):
    monkeypatch.setenv("UCX_TLS", TLS)
    if value is None:
        monkeypatch.delenv("NIXL_UCX_CUDA_IPC_ONLY", raising=False)
    else:
        monkeypatch.setenv("NIXL_UCX_CUDA_IPC_ONLY", value)
    agent = GPUAgent()
    with _use_nixl_agent(agent), pytest.raises(ValueError, match="CUDA_IPC_ONLY=1"):
        KVCR(
            KVCRConfig(
                nixl_agent_name="target", pool_layouts=[("main", 16), ("index", 8)]
            ),
            KVCRBindings(Mock(), Mock(), Mock(), framework_control=FakeBytesControl()),
            KVCRBackendConfigs(
                framework_regions=regions("target"),
                remote_fw_gpu=RemoteFWGPUOptions(compatibility_domain=DOMAIN),
            ),
        )
    assert agent.registrations == agent.xfers == []


def test_gpu_native_init_rechecks_required_native_policy(monkeypatch):
    monkeypatch.setenv("UCX_TLS", TLS)
    monkeypatch.setenv("NIXL_UCX_CUDA_IPC_ONLY", "0")
    progress = _KVCRProgress(
        Mock(),
        Mock(),
        list,
        Mock(),
        nixl_agent_name="gpu",
        nixl_listen_port=1,
        required_gpu_ucx_tls=("cuda_ipc", "posix", "self"),
    )
    with pytest.raises(RuntimeError, match="policy is not enabled"):
        progress._initialize_nixl()
    assert progress._nixl_agent is None


@pytest.mark.parametrize("change", ["legacy", "domain", "memory", "transport"])
def test_gpu_metadata_rejection_precedes_destination_offer(monkeypatch, change):
    errors = []
    kvcr, agent, _, control = new_gpu(monkeypatch, on_resilience_event=errors.append)
    kvcr.submit_hint(gpu_hint(), request_id="req")
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


def test_legacy_gpu_metadata_timeout_has_no_destination_hazard(monkeypatch):
    errors = []
    kvcr, agent, _, control = new_gpu(monkeypatch, on_resilience_event=errors.append)
    clock = [0.0]
    kvcr._core._clock = lambda: clock[0]
    kvcr.submit_hint(gpu_hint(), request_id="req")
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
    kvcr.submit_hint(gpu_hint(), request_id="req")
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
