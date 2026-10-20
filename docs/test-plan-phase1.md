<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVCR test plan v7, Phase 1 implementation

This document maps Phase 1 of the KVCR test plan (TC-01 → TC-02 → TC-03
same-node → TC-04 in Guard mode, plus the performance cases TC-05 and TC-06 that
the plan places in the same round) to the runnable harness
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
| Model | dense `Qwen/Qwen3-0.6B` (CI) or `Qwen/Qwen3-8B`, and hybrid attention `Qwen/Qwen3.5-4B` (Gated DeltaNet + attention), bf16, `--block-size 64`; `--enforce-eager` unless `--no-enforce-eager`; the hybrid model needs `--worker-arg=--mamba-cache-mode=align`, `--keep-hybrid-kv-cache-manager`, and its geometry `--hit-block-tokens 576 --ssm-groups 3 --page-bytes 18874368` so the page and byte checks are exact |
| Decoding | `temperature 0`, `seed 0`, fixed `max_tokens` (default 32) |
| Token capture | `nvext.extra_fields: ["prompt_token_ids", "completion_token_ids", "worker_id", "detailed_finish_reason"]` through the Dynamo frontend; a build without these fields is BLOCKED |
| Determinism | optional `--batch-invariant` sets `VLLM_BATCH_INVARIANT=1`; an unstable reference is an error (required for the dense model, rejected by Qwen3.5's GDN layers) |
| Performance (TC-05/06) | `--perf-repeats 5` distinct prompts per arm, `--perf-max-tokens 128`, `--perf-concurrency 1`; streamed completions timed on the client, token IDs taken from each chunk's `nvext.completion_token_ids`, `cached_tokens` from the final `usage` chunk |

Each worker is pinned by request header `x-dynamo-worker-instance-id`, so
every stage names the worker that served it and the response proves it.

## Sequence

1. **Preflight.** Two usable CUDA devices, free `/dev/shm`, kernel 6.5+ and
   `libatomic` for Guard mode, free ports. Missing prerequisites → BLOCKED.
2. **Reference.** A KVCR-disabled worker on GPU 0 behind its own router
   generates every prompt twice. Performance prompts are first streamed cold
   and then warm (vLLM prefix-cache hit) to record the KVCR-disabled TTFT/TPOT
   baseline; all repeats must agree. The launch command is checked for the
   absence of `--kv-transfer-config` and the worker log for the absence of
   "kvcr". Unequal repeats fail the run and `result.json` records which run
   diverged at which token.
3. **Stack.** Guard services, router, then worker 0 and worker 1 started one
   at a time so each instance ID is learned unambiguously.
4. **Cases.** Default order TC-01, TC-02, TC-03, TC-05, TC-04, TC-06: the
   performance case runs before the Guard case because TC-04 replaces worker 0
   at the same endpoint and the first peer transfer afterwards fails (finding 1
   below), which would pollute the remote-hit sample. Each case records checks,
   evidence files, and measurements into `result.json`; JUnit mirrors
   PASS/FAIL/BLOCKED.
5. **Cleanup.** Owned process groups, discovery directory, service sockets,
   and `/dev/shm` pool files are removed in the same run.

## Case mapping

| Case | Plan validations | Harness checks |
| --- | --- | --- |
| TC-01 Integration generation correctness (R1) | tokens/length/finish match reference; reuse has cache-path evidence | cold request on worker 0: tokens equal reference, `cached_tokens == 0`, KVCR `local_fill` blocks > 0 after the request; reuse request: tokens equal reference, `cached_tokens` positive and block-aligned; no failed/partial KVCR metrics |
| TC-02 Local DRAM full flow (R2) | only committed data delivered; miss not a hit; partial hit; release per API contract; tokens match | seed full prompt on worker 0 (commit evidence); **partial**: cold worker 1 requests a prompt sharing only the first half → hit ≤ shared-prefix blocks and < seeded blocks, `remote_deliver == source_write == hit blocks`, bytes equal; **present**: full prompt on worker 1 → remaining blocks arrive via `remote_deliver`; **absent**: never-seen prompt → `cached_tokens == 0`, no `remote_deliver`; **release**: worker 0 `local_g2_allocated_slots > 0` and `pinned_keys == 0` after consumers finished |
| TC-03 Request hints and peer transfer, same-node (R4) | hints isolated per request; data from expected source; transfer counts verifiable | seed on worker 0; two concurrent requests to worker 1, one hinted and one fresh: hinted has a block-aligned hit with `remote_deliver == source_write == hit blocks` and equal bytes; fresh has `cached_tokens == 0`; `transfer`, `source_write`, `remote_deliver` success counts ≥ 1; no failed/partial |
| TC-04 Guard protection, takeover, re-attach (R3) | preserved data available; evidence for Guard serving and re-attach; tokens match; no dual owner | commit two prompts on worker 0, recovery not disabled; SIGKILL worker 0's process group; service logs `KVCR_EVENT guard_promoted` with `recovered_blocks ≥ committed`; worker 1 requests both prompts in turn, each checked for tokens equal reference and hit blocks == `remote_deliver`; replacement worker 0 starts, attaches without a second promotion, serves a fresh prompt (tokens equal reference, commits) and the preserved prompt (tokens equal reference; cached tokens recorded) |

| TC-05 Cold-path overhead and reuse latency (perf) | TTFT, TPOT, ITL, throughput for KVCR disabled vs cold vs local hit vs remote hit; transfer time; DRAM and GPU usage | five prompts per arm, streamed: KVCR-disabled cold and warm (from the reference phase), KVCR cold on worker 0 (`cached_tokens == 0`, `local_fill` committed), local hit on worker 0 (block-aligned hit), remote hit on worker 1 (`remote_deliver == cached blocks`, source bytes == tier read bytes, no failed/partial); every stream's tokens equal the reference; p50/p95 per arm plus derived cold-path overhead and hit gains; pool gauges and `nvidia-smi` memory before/after |
| TC-06 Guard recovery timing (perf) | engine failure → first correct response; takeover; re-attach; post-recovery TTFT and TPOT | seed and warm five prompts on worker 0; SIGKILL; `t1` = first Guard-served response with correct tokens and a positive hit (polled); remaining prompts served by the Guard; replacement starts, `t2` = model ready and discovered, single owner; post-recovery streams on the replacement compared with pre-failure |

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
compatibility path (TC-04 BLOCKED). `--harness-arg` forwards any
`run_phase1.py` option, e.g. `--harness-arg=--batch-invariant`,
`--harness-arg=--model=Qwen/Qwen3.5-4B`,
`--harness-arg=--worker-arg=--mamba-cache-mode=align`,
`--harness-arg=--keep-hybrid-kv-cache-manager`, `--harness-arg=--no-enforce-eager`.
Hybrid geometry: `--hit-block-tokens 576 --ssm-groups 3 --page-bytes 18874368`
(Qwen3.5-4B; the `hybrid` matrix entry of `scripts/dlcluster/phase1.sbatch`
always passes them) makes every transfer check exact: pages == hit blocks + 3
per hit and bytes == pages * 18 MiB. Without them the checks are lower bounds.
Negative unit tests in `tests/ci/test_phase1.py` pin the verdicts: a zero-hit
replacement, a local hit without Guard delivery, a token mismatch during
recovery, and a delivery missing SSM pages all fail. Verdict rules that are always on: TC-04's replacement
must serve the preserved prefix as a block-aligned hit; TC-06's first correct
Guard response must be covered by `remote_deliver` pages on the target (a local
re-commit does not count) and any token mismatch fails the case.
The [GPU workflow](../.github/workflows/vllm-e2e.yml) runs Phase 1 as
`two-gpu-phase1` after the peer-reuse smoke test, with
`--cases TC-01,TC-02,TC-03,TC-04` (functional gate; TC-05/06 are run separately).

On dlcluster, [`scripts/dlcluster/submit.sh`](../scripts/dlcluster/submit.sh)
bundles the committed HEAD, ships it and
[`phase1.sbatch`](../scripts/dlcluster/phase1.sbatch) to the login node and
submits an exclusive `2GPU & x86_64 & UB24.04` H100 job that reuses or builds
the image and runs the `dense` and `hybrid` configurations; evidence lands in
`~/kvcr-runs/<jobid>/` and `<jobid>.tgz`. `--image local/kvcr-phase1:<sha>`
reuses an image already on the node, `--extra "--harness-arg=..."` applies to
every configuration, `--matrix dense` picks one.

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

## Second execution (2026-09-29)

Node `2u2g-gen-0857` on dlcluster (two H100 NVL 94 GB, Linux 6.17, Docker
29.6), exclusive. Image `local/kvcr-phase1:fc773069…` built from KVCR main
`317e62f` + this branch, Dynamo main `6822bab`, vLLM PR 53624 head `7015687c`.
Harness commits `c039956`–`ea33c94` added TC-05/TC-06, the streaming client,
`--perf-*`, `--worker-arg`, `--keep-hybrid-kv-cache-manager`,
`--no-enforce-eager` and the dlcluster scripts. Evidence:
`~/outputs/kvcr-phase1-ci-20260929/` (`*-result.json`, `*.tgz`, `REPORT.md`).

| Config | TC-01 | TC-02 | TC-03 | TC-04 | TC-05 | TC-06 | Job |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense Qwen3-0.6B, Guard, `--batch-invariant` | PASS | PASS | PASS | FAIL (a) | PASS | PASS | 2329563 |
| hybrid Qwen3.5-0.8B, Guard, CUDA graphs | PASS | FAIL (b) | FAIL (b) | FAIL (b) | FAIL (b) | FAIL (b) | 2329871 |

(a) finding 1 of 2026-09-25, reproduced: only the first hinted request after
Guard promotion misses. (b) one root cause, finding 3 below: no peer transfer
ever runs for the hybrid model.

Performance, p50 of five prompts (1344 input tokens, 128 output tokens):

| Arm | dense TTFT | dense TPOT | hybrid TTFT | hybrid TPOT |
| --- | --- | --- | --- | --- |
| KVCR disabled, cold | 35.1 ms | 14.3 ms | 41.5 ms | 1.62 ms |
| KVCR disabled, vLLM prefix-cache hit | 35.3 ms | 14.4 ms | 23.1 ms | 1.62 ms |
| KVCR cold (store + commit) | 39.4 ms (+4.2) | 15.0 ms (+0.7) | 44.9 ms (+3.5) | 1.78 ms (+0.15) |
| KVCR local hit (GPU prefix cache answers) | 37.6 ms | 15.0 ms | 23.7 ms | 1.73 ms |
| KVCR remote hit, same node | 67.3 ms (+28) | 15.2 ms | 45.9 ms (= miss) | 1.79 ms |
| Guard-served hit after kill | 75.2 ms | 15.5 ms | 44.3 ms (4/5 miss) | 1.78 ms |
| replacement hit from reclaimed pool | 73.0 ms | 15.0 ms | 38.2 ms | 1.74 ms |

Dense ran with `VLLM_BATCH_INVARIANT=1` and `--enforce-eager`, hybrid with CUDA
graphs and no batch invariance; compare within a column only. Guard timeline
(TC-06): promotion 1.0 s after SIGKILL in both; first correct Guard-served
response 4.1 s (dense) / 2.3 s (hybrid); replacement serving after 44.5 s
(dense, 30 s engine start) / 70 s (hybrid, 66 s incl. CUDA graph capture);
TPOT unchanged across recovery. A remote hit moves 123 MB per request
(84 blocks × 1,468,006 B over four successful requests; source bytes equal
target tier read bytes). At 0.6B parameters and 1.3k tokens prefill costs
~3 ms, so no hit shortens TTFT here; this round measures correctness and
overhead, not benefit.

### Sidecar run (2026-09-30, job 2333497, node `2u2g-gen-0859`)

`submit.sh --topology sidecar`: `kvcr_service` per worker in its own
container, host dirs shared at `/run/kvcr` and `/dev/shm/kvcr`, host network,
separate PID namespaces (Dynamo PR 14695 layout); harness attaches with
`--external-services`. Same outcome as single-container: TC-01/02/03/05/06
PASS, TC-04 FAIL on finding 1 only. Guard promotion across containers worked
(`guard_promoted` 1.0 s after kill, first correct Guard response 4.0 s,
replacement serving 44.7 s). TC-05 p50: KVCR cold 40.3 ms (+5.1), remote hit
71.2 ms (+31). Runner fixes from this run: `--docker-arg=-v` form for argparse
(`a7cbdb3`); `--gpus 1` single-GPU mode (`1e6470d`, both workers on one GPU at
0.35 utilization, not yet exercised — the node lacked device-cgroup isolation
and the container saw two GPUs); only Slurm-allocated GPU UUIDs are handed to
docker (`388f12b`). Guard cannot run on aws-iad-cs-002 / eos / cw-pdx-cs-001
(kernels 5.15/5.19: `SO_PEERPIDFD` → `ENOPROTOOPT`, KVCR fails closed).

### Kubernetes runs (2026-09-30, dynamo-aws-dev-02)

`scripts/k8s/phase1-pod.py` renders the sidecar layout as a pod (two
`kvcr_service` native sidecars + main) and, with `--two-node`, two pods on two
nodes: pod `b` runs `tests/e2e/phase1/node_agent.py` and worker 1's sidecar,
pod `a` runs etcd (Dynamo discovery; the file store's inotify watch is
node-local), worker 0's sidecar and the harness with
`--remote-worker 1=http://<run>-b…:9800 --etcd http://<run>-a…:2379`.
Results: single node 5/6 (TC-04 = finding 1, remote hit 114 ms); two nodes
5/6 (cross-node remote hit 244 ms, 183 MB per request); with the fix for
finding 1 (`ai-dynamo/kvcr` PR #52) on `PYTHONPATH` via `--extra-env`, both
layouts 6/6 PASS and TC-06's first Guard-served response drops from 5.7 s to
1.3 s. Do not put "kvcr" in run names, cache paths or hostnames: the
reference phase greps the engine log for it (now scrubbed for known
harness names).

Findings for the KVCR, Dynamo and vLLM teams:

1. **First request to a replaced peer at the same endpoint is refused**
   (dense TC-04; TC-05 remote arm when run after TC-04). Same mechanism as
   2026-09-25: the target skips NIXL metadata for an already-acked endpoint,
   the promoted Guard is a new NIXL agent and raises `missing target agent
   metadata` (`remote_fw_dram.py` `_remote_agent`, line 1748). One miss and
   recomputation per peer after every takeover or replacement; tokens stay
   correct.
2. **`--disable-hybrid-kv-cache-manager` cannot run Qwen3.5.** vLLM's
   `unify_hybrid_kv_cache_specs` fails with "Failed to promote local KV cache
   specs to one unified type" because GDN and attention layers cannot share one
   KV group (job 2329349). The hybrid configuration keeps the manager on.
3. **With multiple KV cache groups no peer transfer happens.** The Dynamo
   plugin call `get_kv_cache_group_metadata` fails on this vLLM
   (`'EngineCoreProc' object has no attribute`) and falls back to
   `cache_config.block_size` (also in dense, harmless there); vLLM's
   `OffloadingEventsTracker` emits placeholder `BlockStored` payloads for
   non-full-attention groups; `dynamo.kv_state_agent` quarantines the ingress
   (`KVCR block size is incompatible`, `CacheOwner residency failed closed`)
   and the frontend logs `KV state-agent recovery query failed` (391× in
   2329871). Without a block index the router never hints, so every
   cross-worker check sees `cached_tokens 0` and `remote_deliver 0`. KVCR's
   local store and commit work for the hybrid model (TC-01 PASS); one
   Guard-served hit did land in TC-06, so the path is intermittent rather than
   absent.
4. **Determinism.** On the KVCR-disabled dense engine the cold-prefill
   generation differs from every later prefix-cache-hit generation (first
   divergence at token 1–33 across five prompts; the hits agree with each
   other); `VLLM_BATCH_INVARIANT=1` makes them identical. Qwen3.5 rejects
   batch-invariant mode (`not supported for GDN_ATTN`) yet its references were
   stable without it (19 prompts). Under `--enforce-eager` Qwen3.5 decodes at
   ~520 ms/token; CUDA graphs give 1.7 ms.

Not covered: disaggregated deployment, more than one TC-05 concurrency level,
a forced GPU-cache eviction so the local-hit arm exercises KVCR's DRAM read,
non-Guard reruns (Phase 3).

### KVCR v0.1.1 release verification (2026-10-08, dynamo-aws-dev-02)

Image built with `KVCR_PINNED_SOURCES=1` from the v0.1.1 `Dockerfile.quick-start`
pins (Dynamo `ea23e51`, vLLM adapter `0856750`, base `nightly-43b4aaea`), pushed as
`ghcr.io/luckyq/kvcr-phase1:ad5011b…` by `scripts/dlcluster/build_image.sh`.
v0.1.1 does not include the fix for finding 1 (`ai-dynamo/kvcr` PR #52), so no
`PYTHONPATH` overlay; the hybrid run keeps the placeholder-skip `events.py`
overlay re-applied onto the new adapter. Runs `v011-dense8b` (Qwen3-8B,
batch-invariant, eager) and `v011-hyb4b` (Qwen3.5-4B, hybrid KV manager, CUDA
graphs): dense TC-01/02/03/05/06 PASS, TC-04 FAIL = finding 1; hybrid TC-01/06
PASS, TC-02/03/05 fail only on dense block arithmetic and the partial-prompt
geometry already seen on 10-03, TC-04 = finding 1. No regression against b827065;
tokens equal the reference in every request. Dense remote hit 160.8 ms vs cold
84.5 ms (1600-token prompt, 236 MB per hit), hybrid remote hit 78.2 ms vs cold
96.0 ms; Guard promoted 1.0 s after the kill in both.

Two harness consequences: Dynamo ea23e51 lets vLLM's logger write the
`KV Transfer metrics` line (`(MainProcess pid=N) INFO MM-DD HH:MM:SS …`), which
the parser now accepts next to the ISO format (`telemetry.line_timestamp`); and
`run_phase1.py` grew `--perf-evict-filler`, `--perf-concurrency-levels` and
`--max-model-len` for the TC-05 DRAM-hit and concurrency arms (run
`v011-dense8b-perf`, see the evidence report).

Hybrid follow-up the same day (harness `1eb9e6a`): `--hit-block-tokens 576`
makes the hit/delivery checks geometry-aware (pages must cover the hit blocks
and equal the source's `source_write` pages and bytes), `--tc02-shared-sections
140 --tc02-sections 200` give TC-02 a 4-block shared prefix, and the reference
phase records `cached_tokens` so TC-02's partial arm is marked not applicable
when the KVCR-disabled engine itself reports no partial hit. For hybrid models
that happens because vLLM `mamba_cache_mode=align` keeps SSM state only at the
last aligned boundary of each scheduler step: the P1/P2 junction has a
checkpoint only if P1 was served as its own request. `--tc02-seed-prefix` does
that (system-prompt warm-up pattern) and TC-02's partial arm then passes on
hybrid with a real cross-worker partial hit (`v011-hyb4b-tc02-seed`: 1728
tokens, 6 pages == 6, 113 MB both sides). Runs `v011-hyb4b-final` +
`v011-hyb4b-tc05` + `v011-hyb4b-tc02-seed`: TC-01/02/03/05/06 PASS, TC-04 =
finding 1, same standing as dense. Hybrid runs should pass
`--hit-block-tokens 576 --tc02-shared-sections 140 --tc02-sections 200
--tc02-seed-prefix`. The reference phase now covers only the requested
cases' prompts; one hybrid run aborted because the bare engine was unstable on
the 3.9k-token TC-02 prompt (GDN rejects batch-invariant mode).

### Verdict tightening (2026-10-19, review of #103)

Harness `27a227a`, v0.1.1 image, dev-02 2×H100. Changes: TC-04 replacement must
hit the preserved prefix; TC-06 first correct Guard response must be covered by
`remote_deliver` on the target, attempts rotate over unseen prompts, any token
mismatch fails; hybrid `--ssm-groups 3` makes pages exact; CI selects TC-01..04.

| run | result | evidence |
| --- | --- | --- |
| dense 8B TC-04 | PASS: replacement `cached_tokens` 1472; Guard-served 23 pages = 23 blocks, 217 MB | `rv-dense8b-tc0406` |
| dense 8B TC-06 | PASS: first attempt Guard-served, 23 pages, t1 1.42 s | `rv2-dense8b-tc06` |
| hybrid 4B TC-04 | PASS: Guard-served 5 pages = 2 blocks + 3 SSM, 94 MB; replacement 1152 | `rv2-hyb4b-tc0406` |
| hybrid 4B TC-06 | PASS: first attempt Guard-served, 5 pages, t1 1.24 s; Guard arm 5/5 hit | `rv3-hyb4b-tc06` |

What the old TC-06 missed: with the kvcr#51 first-request miss, a retry of the
same prompt is a local re-commit on the target (`remote_deliver` 0), not a Guard
hit (`rv-dense8b-tc0406`). The hybrid runs need the state-agent overlay for the
SSM placeholder events; without it the index is quarantined and no hints are sent.

### 2P+1D disaggregated (2026-10-10)

`--topology disagg` (three GPUs; `--two-node` puts P1 on the second node) runs
workers 0/1 as prefill workers with a MultiConnector of the KVCR
OffloadingConnector and NixlConnector kv_producer, worker 2 as the decode
worker (NixlConnector kv_consumer). Requests pin the prefill worker
(`x-dynamo-prefill-instance-id`), identity comes from `prefill_worker_id`.
Dynamo reads `secondary_tiers`/`dynamo_state_agent` only from the top-level
extra config, so the harness mirrors them there (finding A in the report).
Runs `v011-dense8b-2p1d` and `p1-2p1d-2node`: TC-01/02/03/05 PASS, TC-04 = #51
first-request miss, TC-06 PASS single node and 3/5 Guard-served hits across
nodes; 0 token mismatches. Run at `--gpu-memory-utilization 0.35` because the
dead prefill's KV region stays mapped by the decode worker (finding B).
