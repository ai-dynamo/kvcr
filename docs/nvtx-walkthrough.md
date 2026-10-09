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
KVCR_NVTX_LEVEL=medium nsys profile --trace=nvtx,cuda \
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

Schema 3 emits immutable fields once in `request.context` or
`source.pin.context`. The decoder reconstructs these fields before joining
compact transition payloads. Low retains identities and ranges; medium also
records the bounded request display and optional descriptor byte counts used
in the table below.

## Worked operation

A Linux capture with Python 3.12.12, NIXL 1.5.0, NVTX 0.2.16, NumPy 2.5.3 and
Nsight Systems 2025.3.2 exported 90 KVCR payload events for three deliveries.
Both worker processes were present; the harness verified transferred bytes.
The following is the last measured operation. Times are milliseconds relative
to its `hint.submitted` mark; instance UUIDs and handle are aliased.
These times are observations from one run, not latency targets.

| Time | Worker / event | Interpretation |
| --- | --- | --- |
| 0.000 | Target: `hint.submitted`, trace 7 | Request `capture-α-😀-2` has a new hint lifecycle. |
| 0.332 | Target: `op.deliver.lifecycle`, trace 8, H3 | Caller range begins; it stays open until completion is returned. |
| 0.352 | Target: `op.deliver` | Synchronous dispatch begins. |
| 0.614 | Target: `target.remote`, trace 9 | Independent branch range begins on the main thread. |
| 0.648 | Target: `hint.used`, trace 7, H3 | That submission is associated with this operation. |
| 0.664 | Target: `target.queued`, trace 9, hint 7 | Remote progress is queued; caller trace 8 is separate. |
| 0.852 | Target: `target.start_write.enqueued` | Control enqueueing, before source processing. |
| 1.990 | Source: `source.pin.wait`, pin 5 | Observed pin wait begins before the framework callback. |
| 2.170 | Source: `source.pin.registered` | Pin accepted; waiter links pin 5 to source operation 3 and H3. |
| 22.334 | Source: `source.pin.completed` | Four blocks available; the pin range ends at 22.348 ms. |
| 22.907 | Source: `source.write`, trace 6 | Logical source range begins after pin selection. |
| 23.125 | Source: `nixl.write` | Native handle ownership range begins. |
| 23.213 | Source: `nixl.write.posted` | Native write posted. |
| 23.232 | Source: `nixl.done_observed` | First observed native DONE; ownership range remains open. |
| 23.279 | Source: `nixl.write.released` | Release succeeds; native range ends at 23.287 ms. |
| 23.328 | Source: `source.write.completed` | Source success: 4 blocks, 16,384 bytes; source range ends at 23.333 ms. |
| 24.066 | Target: `target.write_done.received` | Progress has the result; branch range ends on progress at 24.086 ms. |
| 50.885 | Target: `target.main.consume` | Main-thread polling consumes it. |
| 51.031 | Target: `op.completion_queued`, trace 8 | All operation branches are joined. |
| 51.085 | Target: `op.completion_returned`, trace 8 | Caller receives the result; caller range ends at 51.094 ms. |

The pin wait range lasts 20.358 ms. It includes the callback, framework delay
and polling delay. The later receipt-to-return gap of 27.019 ms shows delayed caller polling; it is
not extra DMA time. Absolute clocks from separate machines would require clock
alignment before using a similar cross-worker timing table.

## Other outcomes

Use a different output name for each scenario and pass the matching scenario to
the decoder. `--scenario partial` requires at least two blocks. `zero` returns
an accepted pin with every block missing; `failure` returns no pin result.
Timeout delays the framework result beyond the operation deadline.

| Scenario | Deliveries / payload events in the checked capture | Expected result |
| --- | --- | --- |
| success, low | 1 / 30 | 4 blocks transferred and returned successfully. |
| partial, low | 1 / 30 | 3 of 4 blocks; pin/source/caller PARTIAL. |
| zero, low | 1 / 25 | 0 blocks; pin/source/caller FAILED, no native write posted. |
| failure, low | 1 / 25 | Framework result unavailable; FAILED, cause unknown. |
| timeout, low | 1 / 26 | Deadline path and failed caller completion; event count can vary with polling. |
| off | 1 / 0 | Workload succeeds with no KVCR payload events. |
| delayed, low | 3 / 90 | Same successful join with visible pin/poll delays. |
| delayed, medium | 3 / 90 | Same ranges plus optional request display and byte diagnostics. |

The decoder checks hint submission/use/request/worker/handle identity, rejects a
hint invalidated before use, and verifies source pin/write and completion joins.
It also checks native range IDs, closure, release ordering, and the target
range's cross-thread endpoint. All independent ranges in these eight captures
were closed; unresolved native shutdown is a separate unit case.
Replacing an older hint does not invalidate the newer submission. Removing hint
or request-context rows from the checked captures makes validation fail.
Native failure, ambiguous posting, retries and quarantine are controlled unit
cases; these captures do not inject hardware faults.

## Repeat the overhead comparison

Compare an immutable pre-optimization checkout (`88f2918`) with the updated
stack. For each of five repetitions, shuffle both variants' `off`, `low`,
`medium` runs, plain and under the same Nsight command above; alternate which
collection mode runs first. Use `--blocks 16 --block-bytes 4096 --warmup 30
--iterations 300 --pin-delay-ms 0 --poll-delay-ms 0` each time, with the same
environment and no competing validation workloads.
Save every per-operation JSON latency and report. That is 60 runs, 18,000
measured deliveries and 1,800 warmups. Compare the median of five run means,
the range of those means, and each level's delta against its own variant and
collection mode's off baseline. Verify each run's bytes and 330 pin
requests/releases with zero cancellations; preserve raw runs, versions and
source hashes alongside the summary.

The timed interval starts at `deliver()` and ends at caller completion. Hint
creation, byte verification, startup and teardown are excluded. Report size
includes warmups and other libraries. Successful runs exercise few medium-only
events, and run-to-run variation can reverse low/medium ordering. These DRAM
measurements do not establish model throughput or a production overhead budget;
that budget is a separate reviewer decision.
