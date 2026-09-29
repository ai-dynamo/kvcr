# SPDX-License-Identifier: Apache-2.0
"""Single-agent UCX path diagnostic; every payload copy goes through NIXL.

Run in a fresh process per configuration. See NIXL_UCX_DIAGNOSTICS.md.
This intentionally does not change KVCR's production backend defaults.
"""

import argparse
import importlib.metadata
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import torch
from nixl import nixl_agent, nixl_agent_config


def run(args):
    torch.cuda.set_device(args.device)
    size = args.spans * args.span_bytes
    source = torch.arange(size, dtype=torch.uint8, device=f"cuda:{args.device}")
    restored = torch.zeros_like(source)
    dram = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    torch.cuda.synchronize()
    name = f"kvcr-self-probe-{uuid.uuid4().hex[:8]}"
    # Delay backend creation so the diagnostic can set UCX backend parameters.
    agent = nixl_agent(name, nixl_agent_config(backends=[], enable_prog_thread=True))
    agent.create_backend(
        "UCX",
        {
            "num_threads": str(args.threads),
            "ucx_error_handling_mode": args.error_mode,
        },
    )
    registrations = []
    results = []
    for tensor, mem_type in ((source, "VRAM"), (restored, "VRAM"), (dram, "DRAM")):
        registrations.append(
            agent.register_memory(
                [(tensor.data_ptr(), size, args.device if tensor.is_cuda else 0, "")],
                mem_type=mem_type,
            )
        )

    def descriptors(tensor, mem_type):
        return agent.get_xfer_descs(
            [
                (
                    tensor.data_ptr() + i * args.span_bytes,
                    args.span_bytes,
                    args.device if tensor.is_cuda else 0,
                )
                for i in range(args.spans)
            ],
            mem_type=mem_type,
        )

    for direction, src, dst, src_type, dst_type in (
        ("D2H", source, dram, "VRAM", "DRAM"),
        ("H2D", dram, restored, "DRAM", "VRAM"),
    ):
        local = descriptors(src, src_type)
        remote = descriptors(dst, dst_type)
        samples = []
        print(f"PROBE_DIRECTION {direction} agent={name} peer={name}", flush=True)
        for iteration in range(args.warmup + args.iterations):
            start = time.perf_counter()
            handle = agent.initialize_xfer(
                "WRITE", local, remote, name, backends=["UCX"]
            )
            agent.transfer(handle)
            deadline = time.monotonic() + args.timeout
            while True:
                state = agent.check_xfer_state(handle)
                if state == "DONE":
                    break
                if state not in ("PROC", "PEND") or time.monotonic() > deadline:
                    # Do not deregister memory still reachable by DMA on failure.
                    # This standalone process exits and its context is torn down.
                    raise RuntimeError(f"NIXL transfer did not complete: {state}")
            elapsed_ms = (time.perf_counter() - start) * 1000
            agent.release_xfer_handle(handle)
            if iteration >= args.warmup:
                samples.append(elapsed_ms)
        torch.cuda.synchronize()
        if not torch.equal(src.cpu(), dst.cpu()):
            raise RuntimeError(f"{direction} byte comparison failed")
        results.append(
            {
                "direction": direction,
                "bytes": size,
                "p50_ms": statistics.median(samples),
                "samples_ms": samples,
                "bytes_verified": True,
            }
        )
    for registration in reversed(registrations):
        agent.deregister_memory(registration)
    return {
        "args": vars(args),
        "gpu": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "nixl": importlib.metadata.version("nixl"),
        "env": {
            key: os.environ.get(key)
            for key in (
                "NIXL_LOG_LEVEL",
                "UCX_LOG_LEVEL",
                "UCX_PROTO_INFO",
                "UCX_TLS",
                "UCX_CUDA_COPY_BW",
                "UCX_MAX_RMA_RAILS",
                "UCX_PROTOS",
            )
        },
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--error-mode", choices=("peer", "none"), default="peer")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--spans", type=int, default=64)
    parser.add_argument("--span-bytes", type=int, default=131072)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--report")
    args = parser.parse_args()
    if min(args.spans, args.span_bytes, args.iterations, args.timeout) <= 0:
        parser.error("sizes, iterations, and timeout must be positive")
    if min(args.threads, args.device, args.warmup) < 0:
        parser.error("threads, device, and warmup must be non-negative")
    report = run(args)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    print("PROBE_RESULT", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
