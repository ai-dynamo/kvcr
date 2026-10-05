<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVTX framework-pin tracing

This first instrumentation checkpoint traces framework pin requests used by the
remote-memory path. It separates the synchronous `request_pin` callback from the
asynchronous wait for its result. A shared pin has one lifetime and separate
associations to each waiting source operation.

Source NIXL submission, native completion, release and cancellation are also
traced; see the [source lifecycle reference](nvtx-events.md). Target processing,
router hints and caller polling remain separate work in this source checkpoint.

## Enable tracing

On Linux, install the optional Python bindings and structured-payload dependency:

```bash
uv sync --extra profiling
```

`KVCR_NVTX_LEVEL` is read when a KVCR remote-memory backend is constructed:

| Value | Behavior |
| --- | --- |
| `off` | No NVTX/NumPy import, pin trace objects, or payload construction. |
| `low` | Pin lifecycles and source NIXL writes. Default when the profiling dependencies are available. |
| `medium` | Low detail plus waiter-detachment and native release-retry events. |

Without the optional dependencies, tracing is a no-op. An explicit request for
tracing with an unavailable backend warns once. An unsupported level warns once
and disables tracing. There is no `high` level in this checkpoint.

Tracing is independent of KVCR telemetry and is not gated on profiler attachment.
Consequently, low/medium payload construction also costs work outside a profiler.
Use `off` as the baseline when measuring overhead. Annotation failures are kept
out of KVCR's result and resource-ownership paths.

## Capture a real pin and transfer

The example starts two real KVCR/NIXL UCX agents in **one process**, with registered
host memory and a small framework binding that completes pins after a configured
delay. It checks transferred bytes and pin release. It requires Linux and NIXL;
it does not run a model, Dynamo, vLLM, or a GPU transfer.

```bash
KVCR_NVTX_LEVEL=low nsys profile \
  --trace=nvtx,cuda --sample=none --cpuctxsw=none \
  --output=kvcr-pin-success \
  uv run --extra profiling python examples/nvtx_pin_capture.py \
    --blocks 2 --delay-ms 20

nsys export --type=sqlite --include-json=true \
  --output=kvcr-pin-success.sqlite kvcr-pin-success.nsys-rep
```

For controlled failed results and timeouts, use `--scenario failure` and
`--scenario timeout` with distinct report names. Timeout deliberately delays the
pin beyond the 10-second operation deadline. These are correctness examples,
not throughput benchmarks. Keep the reports outside the source checkout.

Extended payloads require a collector/viewer with support for them. The initial
probe used Python 3.12, `nvtx` 0.2.16, NumPy 2.5.3, and Nsight Systems 2025.3.2.
Verify decoding in your environment before relying on a capture. The CUDA runtime
package `nvidia-nvtx` does not supply the Python `nvtx` annotation API.

In the SQLite export, inspect `NVTX_EVENTS.jsonText` (enabled by `--include-json`)
and join its `textId` to `StringIds.id` for the event name. For example:

```sql
SELECT e.start, e.end, s.value AS event, e.jsonText
FROM NVTX_EVENTS AS e JOIN StringIds AS s ON s.id = e.textId
WHERE s.value LIKE 'source.pin.%'
ORDER BY e.start;
```

## Interpretation

All annotations use the `KVCR` domain and `framework_pin` category. Event names
are bounded static strings. IDs and counts are payloads, never registered names.

| Event | Meaning |
| --- | --- |
| `source.pin.framework` | Same-thread push/pop around the framework's `request_pin` callback, including exceptions. |
| `source.pin.registered` | A new physical request was accepted into KVCR's pending-pin state. |
| `source.pin.waiter` | Association between that pin and one source operation/target operation handle. |
| `source.pin.completed` | KVCR observed a usable/failed result, deadline, cancellation, or shutdown. Exactly one terminal observation per pin trace. |
| `source.pin.detached` | One source operation stopped waiting; other waiters may remain. Medium detail only. |

Registration-to-completion measures the observed asynchronous wait, including
delay until KVCR polls the framework. It is not the framework's internal execution
time. A cancellation marker reports KVCR's decision; it does not establish native
transfer quiescence or permission to reuse a buffer. Late results are still
discarded/released by the existing lifecycle and do not emit a second completion.

Schema version 1 uses a fresh structured NumPy payload for every event:

| Field | Interpretation |
| --- | --- |
| `instance_hi`, `instance_lo` | Two uint64 halves of a UUID assigned to this KVCR tracer instance. |
| `pin_id` | Independent uint64 sequence within that instance; distinguishes framework request-ID reuse. |
| `pin_request_known`, `pin_request_id` | Signed int64 framework request ID and availability flag. The flag is zero before the callback returns or when its Python integer is outside int64 range; the independent pin identity and lifecycle events are still recorded. Zero is a valid request ID when the flag is set. |
| `source_op_id`, `op_handle` | Signed int64 source and target operation handles. Zero denotes unavailable context in direct helper calls. The source path carries these even if the pin callback fails before registration. |
| `requested_blocks`, `completed_blocks` | Requested count and count of non-missing entries in an accepted result. `-1` means the completed count is unavailable. |
| `status`, `reason` | Bounded codes below. |
| `fw_dram_utilization_known` | Always zero in this checkpoint: no framework utilization source is exposed by the current bindings. |

Join pin events by `(instance_hi, instance_lo, pin_id)`, not by framework request
ID alone. Operation handles alone are not globally unique across target agents.
This schema is a source-side pin checkpoint, not a cross-worker identity scheme.
No request, session, or parent-session identity is inferred from those handles.

Status codes: `1=pending`, `2=success`, `3=partial`, `4=failed`, `5=timeout`,
`6=cancelled`. Partial reports how many blocks were available, without asserting
why others were missing.

Reason codes: `0=unknown`, `1=none`, `2=callback_error`, `3=duplicate_request`,
`4=invalid_result`, `5=deadline`, `6=cancelled`, `7=shutdown`, `8=no_waiters`.
A framework result of `None` gives no failure cause: it is **not** evidence of
cache pressure. Callback errors identify the stage, not an underlying diagnosis.
Exception strings are not recorded.

## Integration gaps on the reviewed baseline

On `fb4f264`, `submit_hint()` accepts a `request_id`; the hint parser retains only
source endpoint and block hashes. The target pull retains the request ID, but its
`start_write` message does not transport request/session/parent-session context.
`KVCRBindings` exposes pin callbacks, not framework-cache utilization. Capturing
those inputs needs a verified integration source and potentially a separately
scoped API or wire change. KVCR local-cache occupancy is not a substitute.

The pin hooks can also be encountered by remote fetch. They do not provide full
fetch fan-out correlation. No production-overhead acceptance threshold has been
established by this example.

References: [NVTX Python best practices](https://nvidia.github.io/NVTX/python/best_practices.html)
and [extended payload examples](https://nvidia.github.io/NVTX/python/annotation_attributes.html).
