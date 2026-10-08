<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Read one remote-delivery capture

The two-process example uses real KVCR/NIXL UCX with a synthetic framework
binding and registered memory. Start with DRAM; `--memory vram` additionally
requires compatible CUDA PyTorch. It does not run Dynamo or an inference model.
Keep raw reports outside the checkout.

```bash
uv sync --extra profiling
mkdir -p /tmp/kvcr-nvtx-review
KVCR_NVTX_LEVEL=low nsys profile --trace=nvtx,cuda \
  --sample=none --cpuctxsw=none \
  --output=/tmp/kvcr-nvtx-review/remote-delayed \
  uv run --extra profiling python examples/nvtx_remote_capture.py \
    --blocks 4 --warmup 1 --iterations 2 \
    --pin-delay-ms 20 --poll-delay-ms 50
nsys export --type=sqlite --include-json=true \
  --output=/tmp/kvcr-nvtx-review/remote-delayed.sqlite \
  /tmp/kvcr-nvtx-review/remote-delayed.nsys-rep
uv run python examples/nvtx_decode.py \
  /tmp/kvcr-nvtx-review/remote-delayed.sqlite --operations 3
```

Open the `.nsys-rep` in Nsight Systems and expand both worker processes and their
NVTX lanes. Select `hint.submitted` on the target, then follow its local
`trace_id` through `hint.used` and `target.queued.hint_trace_id`. The caller and
remote branch have separate trace IDs. Use the target agent/incarnation/handle
tuple to find the source write, then source instance/operation ID to find its
pin waiter and pin ID. The [event reference](nvtx-events.md) and
[producer map](nvtx-context.md) define each join.

## Worked operation

A Linux capture with Python 3.12.12, NIXL 1.5.0, NVTX 0.2.16, NumPy 2.5.3 and
Nsight Systems 2025.3.2 exported 72 KVCR payload events for three deliveries.
Both worker processes were present; the harness verified transferred bytes.
The following is the last measured operation. Times are milliseconds relative
to its first request-context event; instance UUIDs and handle are aliased.
These times are observations from one run, not latency targets.

| Time | Worker / event | Interpretation |
| --- | --- | --- |
| 0.122 | Target: `hint.submitted`, trace 7 | Request `capture-α-😀-2` has a new hint lifecycle. |
| 0.582 | Target: `op.deliver`, trace 8, handle H3 | Synchronous public dispatch begins. |
| 1.053 | Target: `hint.used`, trace 7, H3 | That submission is used for this operation. |
| 1.169 | Target: `target.queued`, trace 9, hint 7 | The remote branch is queued; caller trace 8 is separate. |
| 1.740 | Target: `target.start_write.enqueued` | Control enqueueing, before source processing. |
| 2.991 | Source: `source.pin.registered`, pin 5 | Framework request accepted; waiter links pin 5 to source operation 3 and H3. |
| 23.377 | Source: `source.pin.completed` | Four blocks available after the configured pin delay and polling. |
| 25.032 | Source: `nixl.write.posted`, trace 6 | Native write posted. |
| 25.156 | Source: `nixl.done_observed` | First observed native DONE. |
| 25.312 | Source: `nixl.write.released` | Native handle release succeeded. |
| 25.455 | Source: `source.write.completed` | Logical source success: 4 blocks, 16,384 bytes. |
| 25.501 | Target: `target.write_done.received` | Progress has the remote result. |
| 51.654 | Target: `target.main.consume` | Main-thread polling consumes it. |
| 51.929 | Target: `op.completion_queued`, trace 8 | All operation branches are joined. |
| 52.084 | Target: `op.completion_returned`, trace 8 | Caller receives the result: receipt-to-return gap 26.583 ms. |

The pin registration-to-completion interval is 20.386 ms. It includes framework
and polling delay. The later 26.583 ms gap shows delayed caller polling; it is
not extra DMA time. Absolute clocks from separate machines would require clock
alignment before using a similar cross-worker timing table.

## Other outcomes

Use a different output name for each scenario and pass the matching scenario to
the decoder. `--scenario partial` requires at least two blocks. `zero` returns
an accepted pin with every block missing; `failure` returns no pin result.
Timeout delays the framework result beyond the operation deadline.

| Scenario | Deliveries / payload events in the checked capture | Expected result |
| --- | --- | --- |
| success | 1 / 24 | 4 blocks transferred and returned successfully. |
| partial | 1 / 24 | 3 of 4 blocks; pin/source/caller PARTIAL. |
| zero | 1 / 20 | 0 blocks; pin/source/caller FAILED, no native write posted. |
| failure | 1 / 20 | Framework result unavailable; FAILED, cause unknown. |
| timeout | 1 / 21 | Deadline path and failed caller completion; event count can vary with polling. |
| off | 1 / 0 | Workload succeeds with no KVCR payload events. |
| delayed | 3 / 72 | Same successful join with visible pin/poll delays. |

The decoder checks hint submission/use/request/worker/handle identity, rejects a
hint invalidated before use, and verifies source pin/write and completion joins.
Replacing an older hint does not invalidate the newer submission. Removing hint
or request-context rows from the checked captures makes validation fail.
Native failure, ambiguous posting, retries and quarantine are controlled unit
cases; these captures do not inject hardware faults.

## Repeat the overhead comparison

For each of five repetitions, rotate `off`, `low`, `medium`; run every level
plain and under the same Nsight command above. Use `--blocks 16 --block-bytes
4096 --warmup 20 --iterations 200 --pin-delay-ms 0 --poll-delay-ms 0` each time.
Save the example's per-operation JSON latency values and each report size.
That is 30 runs, 6,000 measured deliveries and 600 warmups. Compare the median
of five run means per configuration, along with the range of run means,
reciprocal delivery rate, and median report bytes. Verify each run's bytes and
220 pin requests/releases; preserve raw runs alongside the summary.

The timed interval starts at `deliver()` and ends at caller completion. Hint
creation, byte verification, startup and teardown are excluded. Report size
includes warmups and other libraries. Successful runs exercise few medium-only
events, and run-to-run variation can reverse low/medium ordering. These DRAM
measurements do not establish model throughput or a production overhead budget;
that budget is a separate reviewer decision.
