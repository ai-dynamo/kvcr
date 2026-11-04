# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import msgspec
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _ConstantHashAdapter,
    _decode_control_message,
    _HintEnvelope,
    _new_kvcr,
    _wait_until,
)

from kvcr.config import KVCRConfig, RemoteFWGPUOptions
from kvcr.hints.fetch_gpu import KV_FETCH_GPU
from kvcr.hints.parser import parse_hints
from kvcr.types import MemoryRef, RegionDescriptor

DOMAIN = "test-host:cuda-ipc:layout-v1"
TLS = "cuda_ipc,posix,self"


def gpu_hint(*, domain=DOMAIN):
    return {
        "protocol_version": "0.1",
        "message_id": "gpu-message",
        "actions": [
            {
                "action_id": "gpu-fetch",
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


def gpu_envelope(*, domain=DOMAIN):
    return msgspec.convert(gpu_hint(domain=domain), type=_HintEnvelope)


def parse_gpu_hint(hint):
    return parse_hints(
        msgspec.convert(hint, type=_HintEnvelope),
        {KV_FETCH_GPU.action_type: KV_FETCH_GPU},
    )


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
        self.plugin_params = {"ucx_error_handling_mode": "peer", "num_workers": "1"}
        self.backend_params = {}
        self.backend_calls = []
        self.backends = {}

    def get_plugin_params(self, backend):
        assert backend == "UCX"
        self.backend_calls.append(("defaults", backend))
        return self.plugin_params

    def create_backend(self, backend, params):
        assert backend == "UCX"
        self.backend_calls.append(("create", backend, dict(params)))
        self.backend_params = dict(params)
        self.backends[backend] = object()

    def get_backend_params(self, backend):
        assert backend == "UCX"
        self.backend_calls.append(("readback", backend))
        return self.backend_params

    def get_backend_mem_types(self, backend):
        assert backend == "UCX"
        self.backend_calls.append(("memory_types", backend))
        return ["DRAM_SEG", "VRAM_SEG"]

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
    def sent(action_type):
        return any(
            (message := _decode_control_message(raw))["type"] == action_type
            and message.get("op_handle") == handle
            for _, raw in control.sent
        )

    _wait_until(lambda: sent("target_metadata"))
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
    _wait_until(lambda: sent("start_write_gpu"))
