<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Read one remote-delivery capture

The two-process example uses real KVCR/NIXL UCX with a synthetic framework binding and registered memory. Start with DRAM; `--memory vram` additionally requires compatible CUDA PyTorch and uses GPU 0. It does not run Dynamo or an inference model or validate hints in an LLM serving framework. Keep raw reports outside the checkout.

```bash
uv sync --extra profiling
mkdir -p /tmp/kvcr-nvtx-review
KVCR_NVTX_LEVEL=medium nsys profile --trace=nvtx,cuda \
  --sample=none --cpuctxsw=none \
  --output=/tmp/kvcr-nvtx-review/remote-delayed \
  uv run --extra profiling python examples/nvtx_remote_capture.py \
    --blocks 4 --warmup 1 --iterations 2 \
    --pin-delay-ms 20 --poll-delay-ms 50
```

Open the `.nsys-rep` in Nsight Systems and expand both worker processes and their NVTX lanes. Select `hint.submitted` on the target, then follow its local `trace_id` through `hint.used`. Join `target.queued` to its `request.context` by `(instance_hi, instance_lo, trace_id)` to read `hint_trace_id`. The caller and remote branch have separate trace IDs. Use the target agent/incarnation/handle tuple to find the source write, then source instance/operation ID to find its pin waiter and pin ID. The [event reference](nvtx-events.md#payloads-and-identities) defines each join.

Off disables KVCR annotations; other libraries can still emit NVTX events.

## Other outcomes

Use a different output name for each scenario. `--scenario partial` requires at least two blocks. `zero` returns an accepted pin with every block missing; `failure` returns no pin result. Timeout delays the framework result beyond the operation deadline.

| Scenario | Expected result |
| --- | --- |
| success, low | 4 blocks transferred and returned successfully. |
| partial, low | 3 of 4 blocks; pin/source/caller PARTIAL. |
| zero, low | 0 blocks; pin/source/caller FAILED, no native write posted. |
| failure, low | Framework result unavailable; FAILED, cause unknown. |
| timeout, low | Deadline path and failed caller completion; event count can vary with polling. |
| off | Workload succeeds with no KVCR payload events. |
| delayed, low | Same successful join with visible pin/poll delays. |
| delayed, medium | Same ranges plus optional request display and byte diagnostics. |

## Repeat the overhead comparison

Compare an immutable pre-optimization checkout (`88f2918`) with a candidate checkout identified by its commit SHA. For each of five repetitions, shuffle both variants' `off`, `low`, `medium` runs, plain and under the same Nsight command above; alternate which collection mode runs first. Use `--blocks 16 --block-bytes 4096 --warmup 30 --iterations 300 --pin-delay-ms 0 --poll-delay-ms 0` each time, with the same environment and no competing validation workloads. Save every per-operation JSON latency and report. That is 60 runs, 18,000 measured deliveries and 1,800 warmups. Compare the median of five run means, the range of those means, and each level's delta against its own variant and collection mode's off baseline. Verify each run's bytes and 330 pin requests/releases with zero cancellations; preserve raw runs, versions and source hashes alongside the summary.

Use the same capture harness for both variants, including destination resets.

The JSON reports delivery latency and its reciprocal operation rate. Timing starts at `deliver()` and ends at caller completion; hint creation, byte verification, destination reset, startup and teardown are excluded. Report size includes warmups and other libraries. Successful runs exercise few medium-only events, and run-to-run variation can reverse low/medium ordering. These DRAM measurements do not establish model throughput or a production overhead budget; that budget is a separate reviewer decision.
