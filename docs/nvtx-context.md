<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVTX identity producers and unavailable inputs

This contract describes the instrumented KVCR API based on `fb4f264`, plus the
immutable Dynamo/vLLM integration snapshots below. It is a map of observed input
seams, not a claim that newer releases have the same integration gaps.

| Identity / value | Producer and meaning | Availability and join |
| --- | --- | --- |
| Inference request ID | The adapter supplies `ReqContext.req_id` to `KVCR.submit_hint()` and `deliver()`. KVCR accepts the supplied string; it does not manufacture an inference ID. | Known on target/caller events; full UTF-8 BLAKE2b digest plus a bounded display mapping. The source wire message has no request ID. Join source to target using the existing agent/incarnation/operation tuple. |
| Framework pin request ID | The framework's `request_pin()` callback returns a Python integer. The inspected vLLM pin adapter separately constructs `kvcr-source:<counter>` for its source operation. | A pin ID is not the inference request ID. Schema 3 pin context records an integer only when it fits int64; otherwise `pin_request_known=0`. Join compact records to context by tracer instance and `trace_id` (equal to its independent `pin_id`), including reused framework IDs. |
| Session ID | Dynamo ingress reads agent/session headers; the inspected handler forwards a session ID to vLLM `generate()`. | The inspected offloading `ReqContext` does not retain it, and KVCR has no session input. `session_known=0` is explicit. |
| Parent session ID | Dynamo `AgentContext` can contain it. | It is not forwarded on the inspected handler-to-offloading path. KVCR does not infer a parent from request or operation IDs; `parent_session_known=0`. |
| Hint lifecycle ID | KVCR assigns a local `trace_id` to each accepted hint submission. Replacement creates another ID; a conflicting source rejects the existing hint. | Schema 2 target `hint_trace_id` identifies the consumed submission within the remote backend's tracer instance. The caller's separate trace ID does not identify the hint. |
| Tracer / source operation / native transfer | KVCR creates a UUID per tracer and local sequences for source operations and native transfers. | These identify local lifecycles, not global requests. Native posting, first DONE, logical completion and handle release are distinct boundaries. |
| Target operation | Target KVCR allocates a signed handle; existing control metadata identifies target agent and incarnation. | Join workers by `(target_agent_hi, target_agent_lo, target_incarnation_hi, target_incarnation_lo, op_handle)`. Local-fill handles can be negative; a handle alone is insufficient. |
| Framework DRAM utilization | The inspected vLLM primary CPU tier owns its cache statistics. | KVCR has no reader callback for them. `fw_dram_utilization_known=0`; registered capacity, pin counts and local G2 occupancy are not substitutes. |

The inspected CPU metric `vllm:kv_offload_cpu_cache_usage_perc` is
`(allocated_chunks - free_chunks - evictable_cache_chunks) / capacity_chunks`,
with zero for zero capacity. It measures the non-evictable fraction, not all
resident cached bytes. A future input seam must name which definition it exposes,
when it samples, and how unknown/zero-capacity states are represented.

## Integration work that needs a separate change

Dynamo's inspected hint producer writes
`sampling_params.extra_args['kv_transfer_params']['kv_hint']`; the inspected
vLLM KVCR adapter reads `ReqContext.kv_hints`. Those fields remain separate on the
examined request path. A compatible carrier must be verified before claiming
Dynamo remote reuse. Session/parent propagation and framework utilization likewise
need explicit framework-to-KVCR seams. They are outside these instrumentation
PRs, which preserve the public API and wire contract.

The existing two-process example uses a synthetic framework binding with real
KVCR/NIXL. It proves the supplied request-to-hint and worker-to-worker joins; it
does not prove Dynamo propagates the inputs above.

## Immutable source references

- [Dynamo hint producer, def3b79](https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/components/src/dynamo/vllm/kv_hints.py#L34-L52).
- [Dynamo session/parent ingress](https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/llm/src/protocols/agents.rs#L10-L103), [session extraction](https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/components/src/dynamo/common/backend/agent_context.py#L20-L29), and [vLLM generation forwarding](https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/components/src/dynamo/vllm/handlers.py#L3426-L3436).
- [vLLM offloading request context, db9527a](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/kv_offload/base.py#L89-L103), [scheduler context construction](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py#L546-L558).
- [Migrated adapter, 4dce1f9: inference delivery](https://github.com/mkhazraee/vllm/blob/4dce1f9ff8a107167018a4a16058e9629883fd51/vllm/v1/kv_offload/tiering/kvcr/manager.py#L465-L481), [hint input](https://github.com/mkhazraee/vllm/blob/4dce1f9ff8a107167018a4a16058e9629883fd51/vllm/v1/kv_offload/tiering/kvcr/manager.py#L576-L586), [source pin request](https://github.com/mkhazraee/vllm/blob/4dce1f9ff8a107167018a4a16058e9629883fd51/vllm/v1/kv_offload/tiering/kvcr/manager.py#L178-L199), and [CPU utilization definition](https://github.com/mkhazraee/vllm/blob/4dce1f9ff8a107167018a4a16058e9629883fd51/vllm/v1/kv_offload/cpu/manager.py#L442-L466).
- [KVCR bindings on the baseline](https://github.com/ai-dynamo/kvcr/blob/fb4f2643fff47cffe0b21266e170f5569605f9b2/src/kvcr/api.py#L56-L77).
