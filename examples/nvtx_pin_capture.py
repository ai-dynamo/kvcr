# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Capture framework pinning through two real KVCR/NIXL agents on Linux.

The framework binding is deliberately small: it returns registered host-memory
blocks after a configurable delay. This validates the pin instrumentation, not
Dynamo/vLLM integration, GPU transfers, or production performance.
"""

import argparse
import ctypes
import json
import socket
import time
from contextlib import ExitStack

from kvcr import KVCR, KVCRBindings
from kvcr.config import KVCRBackendConfigs, KVCRConfig, RemoteFWDramOptions
from kvcr.control_channels import ZmqPeerControlChannel
from kvcr.types import BlockKey, MemoryRef, PinRequestId, RegionDescriptor


def control_channel():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return ZmqPeerControlChannel("127.0.0.1", port, "127.0.0.1")


class Framework:
    def __init__(self, name, keys, delay, fail):
        self.name, self.keys, self.delay, self.fail = name, keys, delay, fail
        self.pending = {}
        self.requests, self.releases, self.cancelled = 0, [], []

    def request_pin(self, keys):
        request = PinRequestId(self.requests)
        self.requests += 1
        self.pending[request] = (time.monotonic() + self.delay, tuple(keys))
        return request

    def poll_pin_results(self):
        ready = []
        for request, (deadline, keys) in list(self.pending.items()):
            if time.monotonic() < deadline:
                continue
            del self.pending[request]
            result = (
                None
                if self.fail
                else (
                    f"pin-{request}",
                    {
                        key: [
                            MemoryRef(
                                end_point_name=self.name,
                                element_index=self.keys.index(key),
                            )
                        ]
                        for key in keys
                    },
                )
            )
            ready.append((request, result))
        return ready

    def release_pin(self, handle):
        self.releases.append(handle)
        return True

    def cancel_pin_request(self, request):
        self.cancelled.append(request)
        self.pending.pop(request, None)


def run(blocks, delay_ms, scenario):
    size = 4096
    keys = tuple(BlockKey(f"block-{i}".encode()) for i in range(blocks))
    expected = b"".join(bytes([i % 255 + 1]) * size for i in range(blocks))
    source_memory = ctypes.create_string_buffer(expected, len(expected))
    target_memory = ctypes.create_string_buffer(len(expected))
    source_framework = Framework(
        "nvtx-source",
        keys,
        20.0 if scenario == "timeout" else delay_ms / 1000,
        scenario == "failure",
    )
    target_framework = Framework("nvtx-target", keys, 0, False)
    channels = [control_channel(), control_channel()]
    with ExitStack() as stack:
        workers = []
        for name, memory, framework, control in zip(
            ("nvtx-source", "nvtx-target"),
            (source_memory, target_memory),
            (source_framework, target_framework),
            channels,
        ):
            worker = KVCR(
                KVCRConfig(
                    nixl_agent_name=name,
                    nixl_listen_port=0,
                    pool_layouts=[("", size)],
                    operation_timeout_ms=10_000,
                    abandon_timeout_ms=20_000,
                ),
                KVCRBindings(
                    framework.request_pin,
                    framework.poll_pin_results,
                    framework.release_pin,
                    cancel_pin_request=framework.cancel_pin_request,
                    framework_control=control,
                ),
                KVCRBackendConfigs(
                    framework_regions=[
                        RegionDescriptor(
                            addr=ctypes.addressof(memory),
                            size=size,
                            count=blocks,
                        )
                    ],
                    remote_fw_dram=RemoteFWDramOptions(eager_ctrl_connect=False),
                ),
            )
            stack.callback(worker.close)
            workers.append(worker)
        source, target = workers
        target.submit_hint(
            {
                "protocol_version": "0.1",
                "actions": [
                    {
                        "action_type": "kv.fetch",
                        "action_version": "1.0",
                        "payload": {
                            "source_control_endpoint": channels[0].endpoint,
                            "block_hashes": list(range(blocks)),
                        },
                    }
                ],
            },
            request_id="nvtx-pin-example",
        )
        operation = target.deliver(
            {
                key: [MemoryRef(end_point_name="nvtx-target", element_index=i)]
                for i, key in enumerate(keys)
            },
            request_id="nvtx-pin-example",
        )
        deadline = time.monotonic() + 30
        results = {}
        while time.monotonic() < deadline:
            source.poll_completed()
            results.update(target.poll_completed())
            if operation in results and not source_framework.pending:
                # Let source-side release follow target notification processing.
                if scenario != "success" or source_framework.releases:
                    break
            time.sleep(0.001)
        assert operation in results, "delivery did not complete"
        assert source_framework.requests == 1, "framework pin path was not exercised"
        success = all(item.success for item in results[operation].values())
        assert success == (scenario == "success"), results
        if success:
            assert target_memory.raw == expected, "transferred bytes differ"
            assert source_framework.releases == ["pin-0"], "pin was not released once"
        if scenario == "timeout":
            assert source_framework.cancelled == [0]
        print(
            json.dumps(
                {
                    "scenario": scenario,
                    "blocks": blocks,
                    "bytes": len(expected),
                    "success": success,
                    "pin_requests": source_framework.requests,
                    "pin_releases": source_framework.releases,
                    "cancelled": source_framework.cancelled,
                    "operation": operation,
                }
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--delay-ms", type=float, default=20)
    parser.add_argument(
        "--scenario", choices=("success", "failure", "timeout"), default="success"
    )
    args = parser.parse_args()
    if args.blocks < 1 or args.delay_ms < 0:
        parser.error("blocks must be positive and delay-ms nonnegative")
    run(args.blocks, args.delay_ms, args.scenario)
