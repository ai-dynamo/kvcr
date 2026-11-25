<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Per-PR GPU CI and integration contract

The active gate is **vLLM: 12 configurations and 42 functional cases** and
**SGLang HiCache: 3 configurations and 9 functional cases**. SGLang Linker is an
extension point, not a registered lane. A target-diagram row is never a passing result.
This document describes the executable contract; historical runs remain in their
Actions artifacts. TC-05/06, nightly/weekly scheduling, cross-node validation and
runner infrastructure changes are outside this per-PR gate.

## Execution and trust boundary

1. CPU `ci.yml` runs lint and `tests/unit tests/ci` on pull requests.
2. The trusted PR mirror pushes `pull-request/<number>` in the upstream repository.
   `docker-build.yml` selects all active required integrations for execution-affecting
   workflow, recipe, dependency, source or E2E changes. It currently selects vLLM
   (`build-vllm`, `e2e-vllm`) and SGLang HiCache (`build-sglang-hicache`,
   `e2e-sglang-hicache`).
   An unauthorized fork push alone does not grant access to GPU runners/secrets.
3. `resolve` checks out recipe and tested source separately, resolves the serving
   base tag to a digest and resolves the nightly policy to one usable exact
   `ai-dynamo`/`ai-dynamo-runtime` version **before Docker caching**.
4. `build-ci-image.yml` validates registered data-only image inputs, builds with the
   shared BuildKit service, publishes to ECR and returns the digest and build identity.
5. The compatibility entry `vllm-e2e.yml` calls `integration-e2e.yml`; the SGLang
   lane calls it directly with its own integration and build outputs. It
   checks the same-run build artifact, seals a selected profile manifest and starts
   independent profile jobs. Each uses only its own image.
6. Each profile verifies installed source/build/overlay identity, prepares one
   immutable local model snapshot, runs a fresh independent KVCR-disabled reference,
   stops it, then runs the candidate and selected cases. The wrapper owns timeouts,
   failure envelopes and a PID-start-time-safe descendant containment backstop.
7. Each integration's summary compares Actions conclusions, source-authorized
   required policy, immutable identities and native evidence. Missing, duplicate,
   stale, failed, blocked or canceled work cannot become full coverage PASS.
   The outer **GPU E2E verdict** requires every active build and integration caller
   registered in `ci_lanes.json`, plus full selection. A passing diagnostic subset
   remains nonqualifying and cannot turn this normal required gate green.

The outer workflow owns cancellation. A newer run for the same mirrored PR/tested
ref supersedes the previous one; different PRs/tested refs have different groups.
Reusable workflows must not copy the outer canceling group. Use the root
**Docker Build** manual dispatch for a fresh diagnostic build. The thin vLLM
workflow has no standalone dispatch because its build artifact belongs to the
same run. Explicit profile subsets are labeled diagnostic and do not qualify the
normal full per-PR gate.

## Qualified vLLM selection

The checkpoint inventory is `tests/e2e/ci_models.json`; vLLM launch geometry and
budgets remain in `ci_profiles.json`. Every model runs the following four profiles:

| Topology | Mode | Worker placement | Required cases |
|---|---|---|---|
| aggregated | Guard | workers 0/1 on GPUs 0/1 | TC-01–04 |
| aggregated | native | workers 0/1 on GPUs 0/1 | TC-01–03 |
| disaggregated 2P+1D | Guard | prefill 0/1 share GPU 0; decode 2 uses GPU 1 | TC-01–04 |
| disaggregated 2P+1D | native | prefill 0/1 share GPU 0; decode 2 uses GPU 1 | TC-01–03 |

| Checkpoint | Immutable Hub revision |
|---|---|
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f` |
| Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` |

Every serving worker is TP=1 and DP=1. The former standalone DP=2 transfer
smoke is retired from the production gate after preserving TC-03's explicit
source/target, freshness, transfer/delivery and output assertions. An image-build
canary exercises actual installed rank-specific initialization and hint port
mapping. This is not continued DP=2 serving qualification. The old harness remains
available only as an explicitly nonqualifying manual diagnostic.
Native means KVCR stays enabled while Guard recovery services are absent. It does
not mean the KVCR-disabled reference. Disaggregated NIXL prefill-to-decode traffic
is distinct from the KVCR peer transfer checked between prefill workers. Hybrid
references, candidates and replacements use `VLLM_SSM_CONV_STATE_LAYOUT=DS` in
disaggregated profiles. Do not restore the obsolete Mamba `align` override.

TC-01 proves cold output, KVCR commit and warm cache reuse; it does **not** force
same-live-worker local KVCR restoration. TC-02 proves seeded full/partial/missing
prefix behavior, actual peer delivery, page/byte reconciliation and release.
TC-03 proves hinted delivery plus isolated cold output under observed overlapping
client requests; this is not a claim of simultaneous GPU forward passes. TC-04
injects worker death, verifies Guard promotion/serving/reattachment, and now requires
actual local KVCR delivery when the replacement reads the preserved prefix.
Independent reference token IDs, lengths and finish reasons remain exact.

The vLLM launcher sets `VLLM_LOGGING_PREFIX=CI_LOG_EPOCH=%(created).9f ` for
every worker role. The marker is engine-neutral so it does not trip the
reference engine's independent KVCR-free log check. The epoch comes from the log record's creation time; nine
decimal places serialize that value and do not imply nanosecond clock accuracy.
The final vLLM evidence gate requires this prefix on every observed KVCR metrics
sample from candidate prefill/aggregated workers and replacements. References
and NIXL-only decoders need no KVCR samples. Missing, malformed or truncated
required telemetry fails the gate. This preserves strict request windows when a
peer source handles control traffic without a precise Dynamo HTTP log entry.
SGLang logging and explicitly diagnostic legacy artifact replay are unchanged.

Same-live-worker eviction/readback remains a qualification gap until a verified
worker-addressed upper-tier reset can be acknowledged while KVCR residency survives.
`cached_tokens` alone never closes that gap. A reset experiment must preserve
secondary KVCR, establish a fresh telemetry window, prove local delivery and exact
output, and exclude KVCR peer delivery without forbidding NIXL decode transfer.

## SGLang HiCache selection

`ci_integrations/sglang_hicache.py` (shared SGLang policy in `sglang.py`) and
`ci_profiles_sglang_hicache.json` define one aggregated native profile per model
family, on the same pinned checkpoints as vLLM. KVCR runs inside each SGLang worker,
so there is no Guard service or supported disaggregated layout: TC-01–03 only.

| Profile | Checkpoint | SGLang launch | Qualification |
|---|---|---|---|
| `sglang-hicache-dense-agg-native` | `qwen3-0-6b` | deterministic inference, fa3 | previous CI pass |
| `sglang-hicache-mamba-agg-native` | `nemotron3-nano-4b-bf16` | deterministic inference, triton | pending |
| `sglang-hicache-gdn-agg-native` | `qwen3-5-0-8b` | deterministic inference, triton | pending |

Workers 0/1 use GPUs 0/1 with `--mem-fraction-static 0.8`, page size 64 and an
8192-token context. Hybrid rows use the vLLM hybrid TC-02 prompt layout (seeded
prefix, 140/200 sections). Upstream SGLang does not yet claim peer reuse for
hybrid (checkpointed SWA/Mamba) pools, so the hybrid rows are expected to show
where TC-02/03 peer transfer stops; they stay `pending` until a run passes.
Telemetry comes from the backend's `KVCR L3 native stats` lines; tier read bytes
are derived from the measured `local_fill` page size and labeled as such.

## Images and dependencies

Image recipes remain engine-specific. Recipe-owned build canaries are passed in
the small `ci-recipe` named build context, separate from tested KVCR source.
A direct local build must provide that context too; the root workflow prepares it. `ci_images.IMAGE_PROFILES` allowlists each
qualified recipe, BuildKit flavor, Python version, base argument and compatibility
overlays. Inputs contain source SHA, recipe SHA, immutable base, requested Dynamo
policy, resolved exact version and overlay identity. Their canonical SHA256 binds
build and execution. `resolve_for_profiles` intersects usable wheels across all
selected image-profile Python runtimes, so Dana can add Python 3.12 beside vLLM's
3.14 without independently resolving a different nightly. Explicit incompatible
exceptions require deliberate policy and qualification, never silent fallback.

The installed `/opt/kvcr-build-metadata.json` uses schema 2, retaining `kvcr_sha`,
base and package fields while adding image profile, recipe SHA, input digest, full
installed inventory and compatibility overlays. Both Dynamo distributions must
match the resolved version. KVCR/Dynamo pip install reports are retained. The
wrapper copies these records into each profile's `image-provenance/` artifact.
This supports exact image replay and audit; floating apt/transitive installation
means a fresh rebuild is not claimed to be hermetic.

The checksum-pinned [Dynamo #18840](https://github.com/ai-dynamo/dynamo/pull/18840)
overlay remains temporary. The vLLM public-channel import adjustment is recorded
separately. Runtime preflight rehashes installed modified files and the image build
imports the real `dynamo.vllm.main` worker entry point. Removal requires a containing
wheel, successful entry-point import and fresh full vLLM matrix. A merged
upstream PR alone does not qualify an installed image.

## Model access and cache

`ci_model_access.prepare_model` performs one supervised revision-pinned Hub fetch
before starting workers and propagates the local path to reference, tokenizer and
all candidates. It preserves standard Hub locks/atomic blob publication, follows
normal blob symlinks and validates readable config plus every indexed weight shard.
The receipt records requested/resolved pin, Hub version, cache path/scope, UID/GID,
writability, disk availability, timing and required inventory, including failures.
This is completion/readability checking, not full multi-GB weight rehashing.

Current runner configuration supplies job-local `/scratch` as an ephemeral
`emptyDir`, not a verified persistent/shared HF cache. `HF_HOME=/scratch/huggingface`
and mutable engine/JIT caches stay job-local. Public checkpoints need no new token.
A future gated model needs a deliberately configured trusted read-only credential;
never log it, install a shared login or grant fork jobs additional permissions.
Do not delete retained model inputs during owned runtime cleanup. Persistent cache
work requires separate Ops evidence and authorization.

## Runtime, cleanup and evidence

Each profile reserves two GPUs on `prod-kvcr-tester-amd-gpu-2-v1`. The inspected ARC
configuration admits two jobs; increasing to eight is independent Ops capacity and
fairness work requiring 16 allocated GPUs plus CPU/RAM/storage/network resources.
No workflow `max-parallel` creates that capacity. Profiles fail independently;
reference and candidate remain sequential within each job. Prior success on L40S
hardware does not qualify every GPU class allowed by a node pool.

The pool also assigns 24GB A10G devices. Nemotron uses a per-worker GPU memory
fraction of 0.44: a fresh run measured about 8GiB of model/runtime memory, exceeding
the former disaggregated 0.32 reservation (about 7.06GiB on that device). Two
colocated prefills reserve at most 88% together, below the existing 90% ceiling.
This changes only the memory budget; A10G execution still requires matrix
qualification and is not implied by this arithmetic or earlier L40S results.

Evidence distinguishes requested logical GPU mapping, observed allocated UUIDs,
worker launch settings and GPU process snapshots. GPU process PIDs may use a
physical-host namespace; snapshots are not proof of zero physical GPU memory.
Port checks apply to declared integration resources and tolerate reusable TCP
TIME_WAIT while rejecting active listeners. They never kill foreign listeners.

Cleanup reports three distinct concepts: reclamation, graceful shutdown and
observation completeness. Owned process/path failures block PASS. Normal forced
stops/nonzero exits are visible warnings pending a measured shutdown fix; TC-04's
intentional death is classified separately. The wrapper preserves ownership-token
and PID-start-time checks for detached processes. Evidence/model inputs are retained.

Repeated SIGINT/TERM no longer repeatedly interrupts harness teardown. `exec python3`
improves signal delivery, but GitHub may kill the step tree after its short cancel
sequence; the wrapper's 180-second ordinary timeout grace is not an Actions cancel
guarantee. Uploads after hard cancellation are best effort. Pod/GPU teardown and
real cancellation during loading/serving still require runtime/Ops qualification.

Artifacts, all scoped by integration/run/attempt:

```
kvcr-inputs-RUN-ATTEMPT
kvcr-build-IMAGE_PROFILE-RUN-ATTEMPT
kvcr-plan-INTEGRATION-RUN-ATTEMPT
kvcr-result-INTEGRATION-RUN-ATTEMPT-PROFILE
kvcr-summary-INTEGRATION-RUN-ATTEMPT
```

Profile artifacts contain the v2 `profile-result.json`, model receipt, build records,
console log and native `run/` evidence: result, JUnit, provenance, per-request files,
reference records, launches, telemetry and cleanup. Semantic proof requirements
are registered in source and recomputed against named checks. Missing expected
telemetry is not observed zero; HiCache-derived byte metrics are labeled derived.
Historical v1 artifacts require explicit replay mode and cannot qualify production.

## Dana's integration steps

Status: steps 1–7 are done for HiCache (dense, Mamba and GDN rows; hybrids
pending). Linker follows on its own branch with a non-gating lane.

1. Rebase/replay only her distinct commits onto the exact reviewed PR208 handoff
   commit. At the 2026-11-25 11:12 UTC refresh, both
   [PR207](https://github.com/ai-dynamo/kvcr/pull/207) and
   [PR213](https://github.com/ai-dynamo/kvcr/pull/213) target
   `doxu/phase1-8-sglang` (`b988101`). PR207 has three distinct commits:
   `a27d187`, `0f45772`, `1fc7106`. PR213 adds amended Linker commit `0061f0e`,
   superseding `04bcb9a`, on top of those three. Preserve the amended fourth delta;
   do not replay both Linker versions or the eight ancestor PR180 commits already
   present here. Preserve
   her engine pins and layered installs without replacing the enhanced harness.
   Resolve overlap in `docker-build.yml`, `run_phase1.py`, and
   `phase1/{cases,stack,telemetry}.py` by composing the engine additions with the
   shared identity, health, delivery and cleanup assertions. Replace duplicate
   workflow plumbing with the interfaces below and update this integration
   contract for the lanes she actually qualifies.
2. Add a qualified SGLang image profile to `ci_images.IMAGE_PROFILES`, preserving
   `Dockerfile.quick-start-sglang` and its engine-specific canary. Add schema-2
   installed metadata/reports and pass source/recipe/input identity build args.
   Preserve the tested fork pin unless a replacement is independently qualified.
3. Add `ci_integrations/sglang_hicache.py` and its catalog, then a distinct
   `sglang_linker.py`/catalog only when its runtime/proofs exist. Register each in
   `ci_integrations.__init__`. Add checkpoint keys to `ci_models.json` when needed.
   HiCache is not a substitute for Linker, and SGLang Guard is unsupported.
4. Implement the adapter methods listed below with exact supported profiles,
   engine-specific flags/environment, native service requirements, independent
   reference disablement and evidence validation. HiCache native can require the
   state agent even though vLLM native does not. Supply engine-specific readback,
   byte geometry and telemetry capability policy; do not copy vLLM assumptions.
5. Resolve a common usable Dynamo version once across selected image profiles.
   Add explicit `build-sglang-hicache` and lane callers, each consuming only its
   own build outputs. Use separate image builds when HiCache/Linker dependencies
   differ. Never use a matrix scalar image output as a multi-image map.
6. Add both callers to `ci_lanes.json` and the outer verdict job's `needs` list. Keep artifact
   namespaces distinct. Remove the old cross-engine cache-lock dependency: the
   inspected jobs have separate scratch volumes. Do not add simultaneous engines
   inside one two-GPU profile or bypass independent references.
7. Start with the qualified dense aggregated native HiCache TC-01–03 lane; qualify
   intended hybrid/disaggregated profiles explicitly. Keep unsupported target rows
   planned/unqualified. Run all required vLLM profiles again alongside the new lane.

PR213's initial Linker lane uses `allow_failure`; that is diagnostic coverage,
not a qualified required lane. Do not carry that waiver into an active root gate.
The refreshed Linker parser leaves `local_fill` page counts unchanged and divides
`source_write`/delivery page-layer counts by a model layer count, retaining their
raw values under `_per_layer:`. Preserve that distinction. Bind the normalization
to the pinned prepared model configuration and correct KV-cache layer geometry,
especially for hybrids; the current unpinned configuration lookup is not that proof.
Preserve this PR's malformed-counter rejection and measured/derived byte labels
when composing its telemetry parser. The current Git runner configuration has
job-local scratch, so cross-engine cache-lock ordering is not required by the
inspected mounts. Verify actual combined execution before removing other ordering
dependencies whose purpose has not been established.

Adapter interface:

| Symbol | Responsibility |
|---|---|
| `CATALOG` | Integration-owned catalog path |
| `IMAGE_PROFILE` | Registered build profile required by this integration |
| `validate_catalog(catalog)` | Reject unsupported tuples/flags/resources |
| `required_profile_ids(catalog=None)` | Source-owned full required functional set |
| `smoke_profiles(catalog)` | Source-owned required smoke inventory, possibly empty |
| `resolve_profile(item, model)` | Effective cases, pin, geometry and launch settings |
| `build_command(profile, output, snapshot, prompt_tag, gpus=None)` | Exact engine launch |
| `environment(profile)` | Engine-specific worker settings |
| `runtime_resources(profile)` | Ports/resources to observe, ownership scope |
| `validate_result(profile, envelope, directory, errors)` | Native launch and semantic proof validation |
| `validate_model_snapshot(profile, path, inventory)` | Engine-specific tokenizer/config prerequisites; returns inventory |

The common Phase-1 evidence layout is `run/result.json`, `run/provenance.json`
and `run/stack/cleanup.json` for functional profiles. The canonical result must
include selected cases, nonempty passing checks and a stable independent-reference
record. The common launcher requires schema-2 cleanup with `status=PASS`,
`reclaimed=true`, and nonempty, complete process outcome/state and owned-path
ledgers for every functional integration. An engine normalizes its own evidence
into this contract and validates capabilities in `validate_result`; it must not
pretend that unavailable engine counters or unsupported cases passed.

Reusable build call inputs: `image_profile`, `source_sha`, `recipe_sha`,
`resolved_inputs_json`; outputs: `image` (digest), `source_sha`, `recipe_sha`,
`image_profile`, `resolved_inputs_sha256`, `build_manifest_artifact`.
Reusable integration call inputs: `integration`, all six build outputs and
`profile_selection` (`all` for the full gate). Image profile and integration are
separate identities. Data is validated against registered recipes; arbitrary
recipes or shell commands are not workflow inputs.

Dana owns SGLang capabilities, profiles, image/canary and evidence normalization.
We own the common execution/verdict contract and preserving vLLM regression.
Maintainers confirm required check names and the separate DCO startup problem.
Ops confirms deployed isolation/capacity and hard-cancel pod/GPU teardown. None of
those confirmations are implied by this PR's YAML or a successful normal run.
