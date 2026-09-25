<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVCR functionality test plan

The immediate acceptance target is a real, attributable KVCR transfer between
two vLLM DP ranks, supported by the existing core regression suite. A visible
GPU, an HTTP 200, or a positive cache count alone does not meet that target.

This plan covers main at `27e39e06058bb740f030e16c400a5f95965100f6`
plus the CI scripts in this branch. It separates runnable coverage from proposed
tests. Earlier H200 results used older KVCR source and do not validate this
revision. Preparing these scripts does not establish a new GPU pass.

## Run the implemented checks

From the repository root, on Linux with the [development prerequisites](dev-guide.md#prerequisites):

```bash
uv sync --group dev
uv run ruff check src tests
mkdir -p artifacts/unit
uv run pytest tests/unit tests/ci -q --junitxml=artifacts/unit/junit.xml
```

The service/Guard tests require Linux 6.5 or newer and libatomic. Most core tests
use fake NIXL agents; the real Guard transfer test additionally requires a
runnable native NIXL/UCX backend and can otherwise skip. Preserve the skip
reason: a skipped native test is not native coverage.

Build the integration image from a clean, committed Linux amd64 checkout:

```bash
KVCR_E2E_IMAGE=local/kvcr-vllm-e2e KVCR_PUSH=0 \
  bash .github/scripts/build_vllm_e2e.sh
bash .github/scripts/run_vllm_e2e.sh --gpus 0,1 \
  --build-output artifacts/build \
  --output-dir artifacts/e2e/local-unique --timeout 1200
```

Use only GPUs assigned to the run and a new output directory. The wrapper runs
the built image; the build metadata determines its identity. Inside that image,
the same harness can be invoked directly when the matching checkout is mounted
and the working directory is its repository root:

```bash
python3 tests/e2e/run_vllm.py --output-dir /artifacts/e2e \
  --timeout 1200 --check-output-consistency
```

For an existing idle deployment, add `--base-url http://HOST:PORT` and
`--worker-log /path/to/the/same/worker.log`. This verification mode requires
exclusive test traffic and leaves the server lifecycle to its owner.

The [CPU workflow](../.github/workflows/ci.yml) and
[GPU workflow](../.github/workflows/vllm-e2e.yml) are the automation entry points.
GPU CI requires the runner, registry and enable-variable configuration in
[ci.md](ci.md). A disabled or skipped GPU job is not an acceptance pass.

## Implemented P0 acceptance contract

The [E2E harness](../tests/e2e/run_vllm.py) uses the quick-start peer configuration:
`Qwen/Qwen3-0.6B`, TP=1, DP=2, block size 64. It starts its own frontend and worker
unless verification mode is selected, and bounds startup, requests and cleanup.

1. Record the tested KVCR, Dynamo and adapter revisions and image identity.
   Launch mode requires exactly two visible GPUs to execute actual CUDA tensor
   operations, plus at least 8 GiB free shared memory.
2. Probe the worker identity. Submit a fresh UUID prefix to rank 0, wait 15 seconds
   for offload/indexing, then submit the identical prompt to rank 1.
3. Require successful HTTP responses from the same requested worker and correct
   ranks. The cold request must have zero cached tokens; the remote request must
   have a positive multiple of 64 cached tokens.
4. Require fresh rank-0 `transfer=success` and `source_write=success`, plus rank-1
   `remote_deliver=success`. Source and destination block counts must equal the
   remote cached-token count divided by 64; byte counts must match and be positive.
   Sum reset-interval samples with engine attribution. Stale, missing or ambiguous
   evidence cannot establish success. Positive peer failed/partial counters fail.
5. Compare cold and cached completion text with deterministic generation settings.
   This is output consistency, not a semantic golden answer or a KV-byte checksum.
6. Retain requests, responses, provenance, logs, parsed metrics, `result.json` and
   JUnit results on failure as well as success. Stop only owned process groups.

The peer-only configuration has no KVCR-owned persistence tier. Its unused local
cascade failures are recorded separately and do not excuse peer transfer errors.
This test does not require every theoretically reusable prompt block to hit;
it verifies that the reported reuse has matching transfer evidence.

## Existing coverage and its limits

“Existing” below means executable tests are present, not that they passed on a
particular machine. The Python files link to the test definitions named here.

| Contract | Existing test mapping | What remains unproved |
| --- | --- | --- |
| Local deposit, fetch, deliver, release; grouped eviction and pool pressure | [test_kvcr_local_dram.py](../tests/unit/test_kvcr_local_dram.py): `test_local_claims_fetch_deliver_release_and_capacity`, `test_multi_pool_residency_moves_and_evicts_as_one_key`, `test_failed_group_reservation_does_not_evict_a_partial_group` | Native GPU bytes under capacity pressure. |
| Request-scoped hints and ordered layouts | [test_kvcr_remote_target.py](../tests/unit/test_kvcr_remote_target.py): `test_kvcr_request_scoped_sources_do_not_overwrite`, `test_remote_fetch_preserves_block_layout_and_bytes`, `test_remote_staging_commits_available_keys` | The layout byte test uses `ctypes.memmove` in a fake agent, not native transport. |
| Ready local source advances without caller polling (#38) | [test_kvcr_remote_source.py](../tests/unit/test_kvcr_remote_source.py): `test_local_source_starts_without_caller_poll_and_holds_its_slot`, `test_local_source_falls_back_during_fill_publication_without_blocking_progress` | Native DMA, eviction pressure and repeated lock contention together. |
| Replacement peer invalidates the correct incarnation (#41) | [test_kvcr_remote_target.py](../tests/unit/test_kvcr_remote_target.py): `test_only_a_refusal_from_this_operation_s_source_finishes_it`; [source route tests](../tests/unit/test_kvcr_remote_source.py): `test_a_replacement_reusing_its_predecessors_name_refreshes_the_route` | A real replacement followed by a successful fresh transfer. |
| Timeout, late completion and pin lifetime | [test_progress.py](../tests/unit/test_progress.py): `test_progress_does_not_deregister_memory_with_an_active_transfer`; [target tests](../tests/unit/test_kvcr_remote_target.py): `test_remote_fetch_timeout_keeps_slot_until_source_is_terminal`, `test_remote_write_cancellation_and_late_completion`; [source tests](../tests/unit/test_kvcr_remote_source.py): `test_abandoned_source_keeps_local_slot_claimed_until_quiescence` | Native writes that remain active beyond a logical deadline. |
| Coalesced pins and shared fills | [test_kvcr_remote_source.py](../tests/unit/test_kvcr_remote_source.py): `test_pending_pin_waiters_share_partial_results_and_request_uncovered_keys`, `test_a_resumed_write_holds_a_pin_another_operation_acquired`; target timeout test above uses two fetch waiters | Concurrent real transfers, one waiter failing, and full-pool reuse. |
| G3 spill, fill and slot recovery | [test_g3.py](../tests/unit/test_g3.py): `test_g3_spill_deliver_and_fill_reuse_existing_progress`, `test_waiting_g3_fetch_source_is_not_evicted`, `test_g3_recovery_rebuilds_free_slots_and_a_tier_recovered_full_frees_one` | Native storage-plugin failure and persistence guarantees. |
| Guard ownership and journal recovery | [test_guard.py](../tests/unit/test_guard.py), [test_recovery_journal.py](../tests/unit/test_recovery_journal.py), [test_guard_integration.py](../tests/unit/test_guard_integration.py): `test_promoted_guard_serves_real_nixl_transfers` | The latter already kills a real primary, transfers host-memory bytes through UCX and checks replacement handback. It is conditional and does not prove GPU DMA safety during a crash. |
| Adapter integration and evidence correctness | [run_vllm.py](../tests/e2e/run_vllm.py), [test_e2e.py](../tests/ci/test_e2e.py), [test_build_vllm_e2e.py](../tests/ci/test_build_vllm_e2e.py) | Implemented in this branch; requires a new image and GPU run for this source revision. CPU tests of the harness are not a GPU pass. |

## Phase 1 of the KVCR test plan v7

TC-01 → TC-04 (integration correctness, local DRAM flow, same-node peer
transfer, Guard takeover and re-attach) are implemented by
[`tests/e2e/run_phase1.py`](../tests/e2e/run_phase1.py) and described in
[test-plan-phase1.md](test-plan-phase1.md). They compare generated token IDs
against a KVCR-disabled reference and require transfer telemetry per path.

## Bounded next tests, not implemented by this CI change

Keep the initial expansion to the following six cases. Use fixed seeds, bounded
timeouts and captured operation/peer identities; avoid an unrestricted product
of models, layouts, transports and failure timings.

| Priority / case | Setup and action | Required oracle |
| --- | --- | --- |
| P0: native bytes and layout | Two real agents in separate processes; known byte patterns in one unnamed pool, two unequal named pools, and repeated descriptors from one pool. Transfer 1 and 17 blocks through host/GPU destinations; introduce one source/destination layout mismatch. | Every destination byte equals the source pattern; guard regions remain unchanged; descriptor order is preserved. Incompatible layouts fail without publishing a valid residency. |
| P0: deadline versus DMA lifetime | One-slot destination and a retained source claim; delay a native write beyond operation and abandonment deadlines, then let it finish. Attempt a competing allocation before and after terminal proof. | Each request completes once; failure never turns into late success. Claimed slots and registrations remain valid while access is unresolved. Matching `uncertain`/`quiesced` regions are reported where required; reuse succeeds only after quiescence. |
| P0: shared work under capacity pressure | Two requests overlap on one block, with one additional missing block and a tiny pool. Delay one pin result and let one waiter expire while another remains valid. | Acquire only uncovered keys; preserve the surviving waiter's pin/claim; return accurate per-key results exactly once; never evict a held group. After all work drains, resource counts return to baseline. |
| P0: same-endpoint peer replacement (#41) | Establish source incarnation A, replace the source at the same endpoint/name with B, and issue a fresh request after A is refused. Deliver a delayed refusal belonging to A after B is learned. | Unknown handles/wrong senders do not finish the request. Refusing A clears only A's cached incarnation. The next operation relearns B and transfers correct bytes; the delayed refusal cannot erase B. Replacement alone must not free memory from unresolved A operations. |
| P0: progress during caller pause (#38) | Deposit a real local block; pause caller polling while a peer requests it. Separately hold the metadata publication lock while a peer request and probe arrive. | Ready bytes transfer without another caller poll and remain claimed. Lock contention leaves progress responsive and queues acquisition for the caller. Capacity callbacks run on the caller thread; completion releases claims exactly once. |
| P1: recovery boundaries | Run the existing real Guard test without skips; extend it with a crash before residency publication, a journal-overflow case, and replacement handback while a native request is active. | Recover only settled records; fence ownership before serving; preserve bytes across takeover/handback. Journal overflow disables warm recovery explicitly. Old requests are not replayed as new successes; fresh caller retries use the current owner. |

Run native happy-path cases on one dedicated two-GPU node first. Add one
two-node same-stack transfer/replacement lane when cross-node support is a
release requirement. Report the exact backend and hardware instead of claiming
that one transport validates all transports.

## Unsupported features and promotion criteria

- Public `abort()` currently returns `False` with a TODO in
  [core.py](../src/kvcr/core.py). Internal timeout/cancellation tests do not
  establish public selective-abort support. Optional movement callbacks and
  individual-entry streaming completions remain TODOs as well.
- G4/object storage is unsupported. G3 currently supports one pool and one
  block per key; cross-tier fallback is limited. Guards serve recovered G2,
  not G3. Inherited G3 files are not verified; missing files can be recreated
  with zeros. Do not claim disk integrity or safe shared G3 paths; see the
  [documented recovery limitations](dev-guide.md#crash-recovery).
- Make the GPU job a required check only after operations enables it and a
  reviewed run demonstrates the complete P0 acceptance contract on the tested
  image digest. Disabled jobs, missing telemetry and unsupported prerequisites
  must remain distinguishable from passes.
- Promote each planned native case only after its deterministic failure path
  demonstrates that the oracle rejects wrong bytes, premature reuse or stale
  peer identity, and repeated fresh runs pass without hidden skips.
- Preserve known startup/shutdown compatibility errors separately from mechanism
  assertions. A transfer pass does not mean error-free startup or shutdown,
  semantic model correctness, fault tolerance, or a performance target.
