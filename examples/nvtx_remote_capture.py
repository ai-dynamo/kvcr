# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Two-process remote-deliver correctness and instrumentation microbenchmark.

Uses real KVCR and NIXL/UCX with a small synthetic framework binding. Optional
VRAM buffers require a CUDA-enabled PyTorch installation. This is not an LLM
serving benchmark. Nsight can capture the parent and both spawned workers.
"""

import argparse
import ctypes
import json
import multiprocessing as mp
import statistics
import time
from contextlib import closing

from nvtx_pin_capture import Framework, control_channel

from kvcr import KVCR, KVCRBindings
from kvcr.config import KVCRBackendConfigs, KVCRConfig, RemoteFWDramOptions
from kvcr.types import BlockKey, MemoryRef, RegionDescriptor


class Buffer:
    def __init__(self, size, value, memory):
        self.memory = memory
        if memory == "vram":
            import torch

            self.data = torch.full((size,), value, dtype=torch.uint8, device="cuda:0")
            torch.cuda.synchronize()
            self.address = self.data.data_ptr()
        else:
            self.data = ctypes.create_string_buffer(bytes([value]) * size, size)
            self.address = ctypes.addressof(self.data)

    def read(self):
        if self.memory == "vram":
            return self.data.cpu().numpy().tobytes()
        return self.data.raw


class PartialFramework(Framework):
    def poll_pin_results(self):
        results = super().poll_pin_results()
        for _, result in results:
            if result is not None:
                result[1][self.keys[-1]] = None
        return results


class ZeroFramework(Framework):
    def poll_pin_results(self):
        results = super().poll_pin_results()
        for _, result in results:
            if result is not None:
                result[1].update(dict.fromkeys(result[1]))
        return results


def worker(role, args, pipe, endpoint=None):
    name = "nvtx-" + role
    keys = tuple(BlockKey(f"block-{i}".encode()) for i in range(args.blocks))
    size = args.blocks * args.block_bytes
    memory = Buffer(size, 73 if role == "source" else 0, args.memory)
    framework_type = {"partial": PartialFramework, "zero": ZeroFramework}.get(
        args.scenario, Framework
    )
    framework = framework_type(
        name,
        keys,
        max(5, args.pin_delay_ms / 1000)
        if args.scenario == "timeout"
        else args.pin_delay_ms / 1000,
        args.scenario == "failure",
    )
    control = control_channel()
    with closing(
        KVCR(
            KVCRConfig(
                nixl_agent_name=name,
                nixl_listen_port=0,
                pool_layouts=[("", args.block_bytes)],
                operation_timeout_ms=1000 if args.scenario == "timeout" else 10_000,
                abandon_timeout_ms=5000 if args.scenario == "timeout" else 20_000,
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
                        addr=memory.address,
                        size=args.block_bytes,
                        count=args.blocks,
                        mem_type="VRAM" if args.memory == "vram" else "DRAM",
                        device_Id=0,
                    )
                ],
                remote_fw_dram=RemoteFWDramOptions(eager_ctrl_connect=False),
            ),
        )
    ) as kvcr:
        if role == "source":
            pipe.send(control.endpoint)
            while not pipe.poll():
                kvcr.poll_completed()
                time.sleep(0.0001)
            pipe.recv()
            # Drain source main-thread pin release after target completion.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                kvcr.poll_completed()
                if not framework.pending and not kvcr._core._framework_pin_keys:
                    break
                time.sleep(0.0001)
            pipe.send(
                dict(
                    pin_requests=framework.requests,
                    pin_releases=len(framework.releases),
                    pin_cancellations=len(framework.cancelled),
                )
            )
            return
        latencies = []
        for iteration in range(args.warmup + args.iterations):
            request = f"capture-α-😀-{iteration}"
            kvcr.submit_hint(
                {
                    "protocol_version": "0.1",
                    "actions": [
                        {
                            "action_type": "kv.fetch",
                            "action_version": "1.0",
                            "payload": {
                                "source_control_endpoint": endpoint,
                                "block_hashes": list(range(args.blocks)),
                            },
                        }
                    ],
                },
                request,
            )
            started = time.perf_counter_ns()
            handle = kvcr.deliver(
                {
                    key: [MemoryRef(end_point_name=name, element_index=i)]
                    for i, key in enumerate(keys)
                },
                request,
            )
            if args.poll_delay_ms:
                time.sleep(args.poll_delay_ms / 1000)
            deadline = time.monotonic() + 30
            result = {}
            while handle not in result and time.monotonic() < deadline:
                result.update(kvcr.poll_completed())
                if handle not in result:
                    time.sleep(0.0001)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            assert handle in result, "delivery did not complete"
            success_count = sum(entry.success for entry in result[handle].values())
            expected = (
                args.blocks
                if args.scenario == "success"
                else args.blocks - 1
                if args.scenario == "partial"
                else 0
            )
            assert success_count == expected, (success_count, expected)
            if expected:
                assert (
                    memory.read()[: expected * args.block_bytes]
                    == bytes([73]) * expected * args.block_bytes
                )
            kvcr.discard_hint(request)
            if iteration >= args.warmup:
                latencies.append(elapsed)
        pipe.send(
            dict(
                latency_ms=latencies,
                median_ms=statistics.median(latencies),
                mean_ms=statistics.mean(latencies),
                delivery_ops_per_second=1000 / statistics.mean(latencies),
            )
        )


def run(args):
    context = mp.get_context("spawn")
    source_pipe, source_child = context.Pipe()
    source = context.Process(target=worker, args=("source", args, source_child))
    target = None
    try:
        source.start()
        source_child.close()
        assert source_pipe.poll(120), "source startup timed out"
        endpoint = source_pipe.recv()
        target_pipe, target_child = context.Pipe()
        target = context.Process(
            target=worker, args=("target", args, target_child, endpoint)
        )
        target.start()
        target_child.close()
        assert target_pipe.poll(max(120, (args.warmup + args.iterations) * 30)), (
            "target timed out"
        )
        result = target_pipe.recv()
        target.join(30)
        assert target.exitcode == 0, target.exitcode
        source_pipe.send("stop")
        assert source_pipe.poll(30), "source drain timed out"
        result.update(source_pipe.recv())
        source.join(30)
        assert source.exitcode == 0, source.exitcode
        result.update(vars(args), source_pid=source.pid, target_pid=target.pid)
        print(json.dumps(result))
    finally:
        for process in (target, source):
            if process is not None and process.is_alive():
                process.terminate()
                process.join(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", choices=("dram", "vram"), default="dram")
    parser.add_argument(
        "--scenario",
        choices=("success", "partial", "zero", "failure", "timeout"),
        default="success",
    )
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--poll-delay-ms", type=float, default=0)
    parser.add_argument("--pin-delay-ms", type=float, default=0)
    args = parser.parse_args()
    if (
        min(args.blocks, args.block_bytes, args.iterations) < 1
        or min(args.warmup, args.poll_delay_ms, args.pin_delay_ms) < 0
    ):
        parser.error("counts must be positive; warmup and delay must be nonnegative")
    if args.scenario == "partial" and args.blocks < 2:
        parser.error(
            "partial requires at least two blocks; use zero for all-missing results"
        )
    run(args)
