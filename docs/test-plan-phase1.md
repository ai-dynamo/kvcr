<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVCR test plan v7, Phase 1 implementation

This document maps the Phase 1 critical path of the KVCR test plan
(TC-01 → TC-02 → TC-03 same-node → TC-04, Guard mode) to the runnable harness
in [`tests/e2e/run_phase1.py`](../tests/e2e/run_phase1.py). It complements the
peer-reuse smoke test described in [test-plan.md](test-plan.md); it does not
replace it.

The single core criterion is unchanged: with KVCR enabled or disabled, the
model's generated token ID sequence, length, and finish reason do not change.
Path evidence is required in addition to equal tokens.

## Topology and configuration

| Item | Value |
| --- | --- |
| Node | one Linux x86-64 host, two GPUs, aggregated deployment, TP=1 |
| Workers | two independent `dynamo.vllm` processes, DP=1 each, one GPU each |
| Router | `dynamo.frontend --router-mode kv` |
| KVCR (Guard mode) | one `kvcr.kvcr_service` per worker (`--guard-count 1 --pool-sizes-gb 2`); the worker's tier sets `kvcr_service_socket_path` and `compatibility_digest` |
| KVCR (non-Guard) | tier sets `secondary_g2_slots` instead; no service, TC-04 is BLOCKED |
| KV state agent (Guard mode) | one `dynamo.kv_state_agent --max-slots 2`; each worker sets `dynamo_state_agent` with a stable per-slot `cache_owner_ids` entry so the router keeps a dead engine's blocks routable to its Guard (`--no-state-agent` disables) |
| Model | `Qwen/Qwen3-0.6B`, bf16, `--block-size 64`, `--enforce-eager` |
| Decoding | `temperature 0`, `seed 0`, fixed `max_tokens` (default 32) |
| Token capture | `nvext.extra_fields: ["prompt_token_ids", "completion_token_ids", "worker_id", "detailed_finish_reason"]` through the Dynamo frontend; a build without these fields is BLOCKED |
| Determinism | optional `--batch-invariant` sets `VLLM_BATCH_INVARIANT=1`; an unstable reference is an error |

Each worker is pinned by request header `x-dynamo-worker-instance-id`, so
every stage names the worker that served it and the response proves it.

## Sequence

1. **Preflight.** Two usable CUDA devices, free `/dev/shm`, kernel 6.5+ and
   `libatomic` for Guard mode, free ports. Missing prerequisites → BLOCKED.
2. **Reference.** A KVCR-disabled worker on GPU 0 behind its own router
   generates every prompt twice. The launch command is checked for the
   absence of `--kv-transfer-config` and the worker log for the absence of
   "kvcr". Unequal repeats fail the run.
3. **Stack.** Guard services, router, then worker 0 and worker 1 started one
   at a time so each instance ID is learned unambiguously.
4. **Cases.** TC-01 → TC-04. Each case records checks, evidence files, and
   measurements into `result.json`; JUnit mirrors PASS/FAIL/BLOCKED.
5. **Cleanup.** Owned process groups, discovery directory, service sockets,
   and `/dev/shm` pool files are removed in the same run.

## Case mapping

| Case | Plan validations | Harness checks |
| --- | --- | --- |
| TC-01 Integration generation correctness (R1) | tokens/length/finish match reference; reuse has cache-path evidence | cold request on worker 0: tokens equal reference, `cached_tokens == 0`, KVCR `local_fill` blocks > 0 after the request; reuse request: tokens equal reference, `cached_tokens` positive and block-aligned; no failed/partial KVCR metrics |
| TC-02 Local DRAM full flow (R2) | only committed data delivered; miss not a hit; partial hit; release per API contract; tokens match | seed full prompt on worker 0 (commit evidence); **partial**: cold worker 1 requests a prompt sharing only the first half → hit ≤ shared-prefix blocks and < seeded blocks, `remote_deliver == source_write == hit blocks`, bytes equal; **present**: full prompt on worker 1 → remaining blocks arrive via `remote_deliver`; **absent**: never-seen prompt → `cached_tokens == 0`, no `remote_deliver`; **release**: worker 0 `local_g2_allocated_slots > 0` and `pinned_keys == 0` after consumers finished |
| TC-03 Request hints and peer transfer, same-node (R4) | hints isolated per request; data from expected source; transfer counts verifiable | seed on worker 0; two concurrent requests to worker 1, one hinted and one fresh: hinted has a block-aligned hit with `remote_deliver == source_write == hit blocks` and equal bytes; fresh has `cached_tokens == 0`; `transfer`, `source_write`, `remote_deliver` success counts ≥ 1; no failed/partial |
| TC-04 Guard protection, takeover, re-attach (R3) | preserved data available; evidence for Guard serving and re-attach; tokens match; no dual owner | commit two prompts on worker 0, recovery not disabled; SIGKILL worker 0's process group; service logs `KVCR_EVENT guard_promoted` with `recovered_blocks ≥ committed`; worker 1 requests both prompts in turn, each checked for tokens equal reference and hit blocks == `remote_deliver`; replacement worker 0 starts, attaches without a second promotion, serves a fresh prompt (tokens equal reference, commits) and the preserved prompt (tokens equal reference; cached tokens recorded) |

Recorded measurements, not thresholds: Guard promotion delay after the kill,
Guard-served request latency, replacement readiness time, and the cached
tokens the replacement reports for the preserved prefix.

## Evidence layout

```text
<output>/
  result.json            run status, per-case checks, measurements, limitations
  junit.xml              PASS → testcase, FAIL → failure, BLOCKED → skipped
  provenance.json        package versions, image source revisions, harness hash
  reference/             reference requests/responses and reference-tokens.json
  requests/              every case request and response (token IDs included)
  stack/                 launch commands, kv-transfer configs, service/worker/router logs
```

Status is PASS or FAIL only; combinations that cannot run are BLOCKED with
the missing condition. One combination passing does not extend to another.

## Running

Inside the integration image with two visible GPUs:

```bash
python3 tests/e2e/run_phase1.py --output-dir artifacts/phase1/run --mode guard
```

From a host with Docker and the NVIDIA Container Toolkit, using the image
produced by `build_vllm_e2e.sh`:

```bash
bash .github/scripts/run_phase1.sh --gpus 0,1
bash .github/scripts/run_phase1.sh --gpus GPU-…,GPU-… --image <ref> \
  --cpus 32 --cpuset-cpus 0-31 --memory 96g --hf-cache /path/to/hf-cache
```

`--cases TC-01,TC-03` selects cases; `--mode nonguard` runs the non-Guard
compatibility path (TC-04 BLOCKED). The [GPU workflow](../.github/workflows/vllm-e2e.yml)
runs Phase 1 as `two-gpu-phase1` after the peer-reuse smoke test.

## Known limits

- Aggregated only. A single node with two GPUs cannot form a disaggregated
  1P+1D pair with a valid KVCR source and target; that combination is not run.
- Cross-node TC-03 and the non-Guard rerun belong to Phase 3.
- Local-hit path evidence for TC-01 is the KVCR commit after the cold request
  plus the reuse hit count. Remote and Guard-served paths have direct transfer
  telemetry. A same-worker KVCR `local_deliver` is recorded when it occurs but
  is not required, because the GPU prefix cache normally answers first.
- Guard mode needs Linux 6.5+ (`SO_PEERPIDFD`) on the host kernel. Nodes on
  5.15 report BLOCKED for the whole Guard run.
- Byte equality between source write and target tier read is the transfer
  oracle; no independent KV-byte checksum is taken.

## First execution (2026-09-25)

Node `4u8g-gen-0080` on dlcluster, two H100 94 GB, Linux 6.17, Docker 29.6.
Image built with `build_vllm_e2e.sh` from KVCR `6824a5b` (main `317e62f` +
this branch), Dynamo main `6da70ec`, vLLM PR 53624 head `7015687c`. Four runs
(`~/outputs/kvcr-phase1-ci-20260925` holds the evidence); runs 2-4 used the
image with updated harness code mounted from the checkout.

| Run | Result | What changed |
| --- | --- | --- |
| 1 | FAIL before cases | harness scratch paths contained "kvcr"; reference bypass check tripped |
| 2 | TC-01 PASS, TC-02 FAIL, TC-03 PASS, TC-04 FAIL | TC-02 partial hit reported as `remote_deliver partial` plus a source `framework_pin_wait failed` for the one uncovered key; TC-04 see findings |
| 3 | TC-01/02/03 PASS, TC-04 FAIL | partial accepted for the partial sub-case; second takeover prompt added |
| 4 | TC-01/02/03 PASS, TC-04 FAIL (first takeover request only) | KV state agent added; second takeover request served by the Guard: 22 blocks, 161,480,704 bytes, tokens equal reference |

Findings for the KVCR and Dynamo teams:

1. **First hinted request after Guard promotion fails.** Worker 1 had
   exchanged NIXL metadata with the primary at `tcp://127.0.0.1:17771`, so
   its next `start_write` to the same endpoint omits `target_agent_metadata`
   (`_metadata_acked_sources`). The promoted Guard is a fresh NIXL agent and
   refuses: `TypeError: missing target agent metadata` → `write_refused` →
   `remote_deliver failed`, `cached_tokens 0`, recomputation (tokens still
   equal the reference). The refusal invalidates the peer, so the next request
   republishes metadata and succeeds. Reproduced in runs 2, 3, and 4.
2. **Without the KV state agent the Guard is never hinted.** When the engine
   dies, the router removes that worker's blocks; a later request has no
   `hint_wait` at all (run 3). `dynamo.kv_state_agent` with per-slot
   `cache_owner_ids` keeps the blocks and the hint source alive (run 4).
3. Partial-prefix requests are delivered as `remote_deliver` result
   `partial` with one `framework_pin_wait failed` on the source for the block
   that straddles the shared prefix; delivered blocks equal the reported hit.

Measured, not thresholds: Guard promotion 2.0 s after SIGKILL; Guard-served
request latency 0.45 s; replacement worker ready and re-attached in 30 s;
replacement reports 1408 cached tokens for the preserved prefix after handback.
