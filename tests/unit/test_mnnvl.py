# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MNNVL policy and negotiation tests; addresses are opaque, never GPU evidence."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    _ConstantHashAdapter,
    _decode_control_message,
    _new_kvcr,
    _poll_until,
    _use_nixl_agent,
    _wait_until,
)
from test_gpu_direct import (
    DOMAIN,
    GPUAgent,
    GPUPinning,
    gpu_envelope,
    refs,
    regions,
    source_message,
)

from kvcr.backends.remote_fw_dram import _RemoteFWDram
from kvcr.backends.remote_ops import (
    _TargetMetadata,
    _TargetMetadataAck,
    _TargetPullOp,
    _TargetPullState,
)
from kvcr.config import KVCRConfig, RemoteFWGPUOptions
from kvcr.runtime import progress as kvcr_progress
from kvcr.runtime.progress import _KVCRProgress
from kvcr.types import BlockKey

FABRIC = "mnnvl:0123456789abcdef0123456789abcdef:0"
MNNVL_TLS = ("cuda_ipc", "tcp", "self")


def options(**changes):
    return RemoteFWGPUOptions(
        **{
            "compatibility_domain": DOMAIN,
            "gpu_transport": "mnnvl",
            "fabric_domain": FABRIC,
            "ucx_tls": MNNVL_TLS,
            "eager_ctrl_connect": False,
            **changes,
        }
    )


def native_progress():
    return _KVCRProgress(
        Mock(),
        Mock(),
        list,
        Mock(),
        nixl_agent_name="mnnvl",
        nixl_listen_port=1,
        dram_backends=["UCX"],
        required_gpu_ucx_tls=MNNVL_TLS,
        required_gpu_transport="mnnvl",
    )


def new_mnnvl(monkeypatch, *, owner="source"):
    monkeypatch.delenv("NIXL_UCX_CUDA_IPC_ONLY", raising=False)
    agent, pinning = GPUAgent(), GPUPinning()
    agent.plugin_params.update(gpu_direct_transport="", gpu_direct_ucx_tls="")
    control = FakeBytesControl(f"tcp://{owner}:1")
    kvcr = _new_kvcr(
        agent,
        pinning,
        control,
        KVCRConfig(nixl_agent_name=owner, pool_layouts=[("main", 16), ("index", 8)]),
        name=owner,
        key_adapter=_ConstantHashAdapter(),
        framework_regions=regions(owner),
        remote_gpu_options=options(),
    )
    return kvcr, agent, pinning, control


@pytest.mark.parametrize(
    "fabric",
    [
        None,
        "",
        "cluster",
        "mnnvl:" + "0" * 32 + ":1",
        "mnnvl:0123456789ABCDEF0123456789ABCDEF:0",
        "mnnvl:0123456789abcdef0123456789abcdef:00",
        "mnnvl:0123456789abcdef0123456789abcdef:-1",
        "mnnvl:0123456789abcdef0123456789abcdef:4294967296",
    ],
)
def test_mnnvl_requires_canonical_nonzero_fabric_identity(fabric):
    with pytest.raises(ValueError, match="fabric UUID and clique"):
        options(fabric_domain=fabric)


def test_mnnvl_clique_zero_and_u32_max_are_valid():
    assert options().fabric_domain == FABRIC
    assert options(fabric_domain=FABRIC[:-1] + "4294967295").gpu_transport == "mnnvl"


@pytest.mark.parametrize(
    "tls",
    [
        ("cuda_ipc", "posix", "self"),
        ("cuda_ipc", "tcp"),
        ("cuda_ipc", "tcp", "self", "cuda_copy"),
        ("cuda_ipc", "tcp", "self", "gdr_copy"),
        ("cuda_ipc", "rc", "self"),
        ("tcp", "cuda_ipc", "self"),
    ],
)
def test_mnnvl_excludes_other_payload_transports(tls):
    with pytest.raises(ValueError, match="exactly"):
        options(ucx_tls=tls)


def test_cuda_ipc_profile_remains_separate():
    original = RemoteFWGPUOptions(compatibility_domain=DOMAIN)
    assert original.gpu_transport == "cuda_ipc"
    assert original.fabric_domain is None
    assert original.ucx_tls == ("cuda_ipc", "posix", "self")
    with pytest.raises(ValueError, match="does not accept"):
        RemoteFWGPUOptions(compatibility_domain=DOMAIN, fabric_domain=FABRIC)
    with pytest.raises(ValueError, match="transport must"):
        options(gpu_transport="rdma")


def test_mnnvl_native_policy_is_per_backend_not_process(monkeypatch):
    monkeypatch.delenv("NIXL_UCX_CUDA_IPC_ONLY", raising=False)
    monkeypatch.setenv(
        "UCX_TLS", "rc,cuda_copy"
    )  # Ordinary engine's independent policy.
    agent, configured = GPUAgent(), []
    agent.plugin_params.update(gpu_direct_transport="", gpu_direct_ucx_tls="")
    with _use_nixl_agent(agent), monkeypatch.context() as scoped:
        scoped.setattr(
            kvcr_progress, "nixl_agent_config", lambda **v: configured.append(v) or v
        )
        native_progress()._initialize_nixl()
    assert configured[0]["backends"] == []
    assert agent.backend_params["gpu_direct_transport"] == "mnnvl"
    assert agent.backend_params["gpu_direct_ucx_tls"] == "cuda_ipc,tcp,self"
    assert agent.backend_params["ucx_error_handling_mode"] == "none"
    assert os.environ["UCX_TLS"] == "rc,cuda_copy"
    assert "NIXL_UCX_CUDA_IPC_ONLY" not in os.environ
    assert agent.registrations == agent.xfers == []


@pytest.mark.parametrize("failure", ["legacy_global", "old_native", "ignored_policy"])
def test_mnnvl_fails_before_vram_if_native_policy_unavailable(monkeypatch, failure):
    monkeypatch.delenv("NIXL_UCX_CUDA_IPC_ONLY", raising=False)
    agent = GPUAgent()
    if failure == "legacy_global":
        monkeypatch.setenv("NIXL_UCX_CUDA_IPC_ONLY", "1")
    elif failure == "ignored_policy":
        agent.plugin_params.update(gpu_direct_transport="", gpu_direct_ucx_tls="")
        agent.get_backend_params = lambda _: {"ucx_error_handling_mode": "none"}
    with _use_nixl_agent(agent), pytest.raises(RuntimeError):
        native_progress()._initialize_nixl()
    assert ("memory_types", "UCX") not in agent.backend_calls
    assert agent.registrations == agent.xfers == []


@pytest.mark.parametrize(
    "change", ["missing_profile", "ipc", "fabric", "missing_fabric"]
)
def test_mnnvl_source_rejects_incompatible_peer_before_native_connection(
    monkeypatch, change
):
    _, agent, pinning, control = new_mnnvl(monkeypatch)
    message = {**source_message(), "gpu_transport": "mnnvl", "fabric_domain": FABRIC}
    if change == "missing_profile":
        message.pop("gpu_transport")
    elif change == "ipc":
        message["gpu_transport"] = "cuda_ipc"
    elif change == "fabric":
        message["fabric_domain"] = FABRIC[:-1] + "1"
    else:
        message.pop("fabric_domain")
    control.incoming.append(msgspec.msgpack.encode(message))
    _wait_until(lambda: bool(control.sent) or bool(agent.sent_notifs))
    assert agent.remote_agents == agent.xfers == pinning.searches == []


def test_mnnvl_target_rejects_legacy_ack_before_offering_destination(monkeypatch):
    kvcr, agent, _, control = new_mnnvl(monkeypatch, owner="target")
    kvcr.submit_hint(gpu_envelope(), request_id="mnnvl")
    handle = kvcr.deliver({BlockKey(b"k0"): refs("target", 2)}, "mnnvl")
    _wait_until(lambda: bool(control.sent))
    probe = _decode_control_message(control.sent[-1][1])
    assert probe["fabric_domain"] == FABRIC and probe["gpu_transport"] == "mnnvl"
    control.incoming.append(
        msgspec.msgpack.encode(
            {
                "type": "target_metadata_ack",
                "sender_control_endpoint": "tcp://source:1",
                "sender_incarnation": "old",
                "op_handle": handle,
                "gpu_peer_accepted": True,
                "gpu_capability_only": True,
                **{
                    k: probe[k]
                    for k in (
                        "source_memory_kind",
                        "target_memory_kind",
                        "transport",
                        "compatibility_domain",
                    )
                },
            }
        )
    )
    _poll_until(
        kvcr, lambda entries: any(op_handle == handle for op_handle, _ in entries)
    )
    assert agent.remote_agents == agent.xfers == []
    assert all(
        not {"target_regions", "dst_descriptors", "target_agent_metadata"}.intersection(
            _decode_control_message(raw)
        )
        for _, raw in control.sent
    )


def test_restarted_gpu_source_must_repeat_capability_before_new_offer():
    backend = SimpleNamespace(
        _gpu=True,
        _kvcr=SimpleNamespace(_clock=lambda: 1.0),
        _metadata_acked_sources={"peer"},
        _options=SimpleNamespace(metadata_retry_interval_ms=100),
        _send_control=Mock(return_value=True),
    )
    op = _TargetPullOp(
        op_id=("target", 1),
        keys=frozenset({BlockKey(b"k0")}),
        started_at=0.0,
        deadline=2.0,
        state=_TargetPullState.WAITING_WRITE_DONE,
        local_fill=False,
        remote_ctrl_ep="peer",
        _backend=backend,
        source_incarnation="old",
        gpu_peer_verified=True,
    )
    assert op.progress(Mock(), {"retry_metadata": "new"}) == (False, True)
    assert not op.gpu_peer_verified and op.retried
    assert backend._send_control.call_args.args[2] == _TargetMetadata(
        op_handle=1, gpu_capability_only=True
    )


def test_delayed_old_capability_ack_cannot_undo_source_restart():
    backend = object.__new__(_RemoteFWDram)
    backend._gpu = True
    backend._options = options()
    backend._dangling_ops = SimpleNamespace(sources={"peer": "old"})
    backend._metadata_acked_sources = {"peer"}
    backend._metadata_retry_after = {"peer": 2.0}
    op = _TargetPullOp(
        op_id=("target", 1),
        keys=frozenset({BlockKey(b"k0")}),
        started_at=0.0,
        deadline=2.0,
        state=_TargetPullState.START_WRITE,
        local_fill=False,
        remote_ctrl_ep="peer",
        _backend=backend,
        source_incarnation="new",
        retried=True,
        gpu_peer_verified=False,
    )
    progress = SimpleNamespace(_in_flight_ops={op.op_id: op})
    reply = _TargetMetadataAck(
        sender_control_endpoint="peer",
        op_handle=1,
        sender_incarnation="old",
        gpu_peer_accepted=True,
        gpu_capability_only=True,
        **backend._gpu_contract(),
    )
    backend._handle_target_metadata_ack(progress, reply)
    assert op.source_incarnation == "new" and not op.gpu_peer_verified
    assert backend._dangling_ops.sources == {"peer": "old"}
    backend._handle_target_metadata_ack(
        progress, msgspec.structs.replace(reply, sender_incarnation="new")
    )
    assert op.source_incarnation == "new" and op.gpu_peer_verified
    assert backend._dangling_ops.sources == {"peer": "new"}
    assert "peer" not in backend._metadata_acked_sources
