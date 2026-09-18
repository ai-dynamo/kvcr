<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Two-GPU vLLM functionality CI

The [vLLM E2E workflow](../.github/workflows/vllm-e2e.yml) builds the tested KVCR
commit into an integration image, then verifies a real rank-0 to rank-1 KVCR
transfer on `prod-kvcr-tester-amd-gpu-2-v1`. It extends the runner provisioning
work in [PR #23](https://github.com/ai-dynamo/kvcr/pull/23).

The workflow is disabled until operations completes the setup below. A skipped
job is not a successful functionality test. Ordinary lint and CPU unit tests
continue to run independently.

The [test plan](test-plan.md) defines the implemented acceptance checks and
separates them from proposed native-transfer and recovery coverage.

## What the jobs verify

1. Check out `github.sha`. For `pull_request`, this is GitHub's synthetic PR
   merge commit, not the PR head. The build refuses a different HEAD or modified
   source files.
2. Resolve Dynamo `refs/heads/main` and vLLM `refs/pull/53624/head` once at build
   time. Fetch and verify those exact SHAs in the image build. Install KVCR from
   the checked-out source. The vLLM base image remains digest-pinned; the six KVCR
   adapter files come from the requested vLLM PR. This does not rebuild all of
   vLLM from that PR.
3. Publish both `<repository>:<KVCR-merge-SHA>` and a unique
   `<KVCR-merge-SHA>-run-<run-id>-<attempt>` tag. Verify the published image's
   provenance labels. Pass its immutable digest to the GPU job, so a later
   rebuild of the same KVCR commit cannot change the tested image.
4. Verify the three source SHAs baked into the image. Check that exactly two
   CUDA GPUs can execute tensor operations and that `/dev/shm` has at least
   8 GiB free. Launch the Dynamo KV router and vLLM with TP=1, DP=2, and
   `Qwen/Qwen3-0.6B`.
5. Discover the worker ID, create a fresh prefix, seed it on rank 0, wait for
   offload/indexing, and retrieve the identical prompt on rank 1. Require HTTP
   success, the requested worker/rank identities, zero cached tokens for the
   fresh seed, and positive block-aligned cached tokens on rank 1.
6. Require rank-0 `transfer=success` and `source_write=success`, plus rank-1
   `remote_deliver=success`. Match transferred block counts to cached tokens
   divided by the block size (64), match source/destination byte counts, and
   reject positive peer failed/partial counters. Also compare the cold and
   cached deterministic completion text.

vLLM prints reset-interval metrics. The test sums attributed interval samples;
it does not subtract consecutive log lines as though they were cumulative
Prometheus counters. Missing or ambiguous evidence fails the test. The peer-only
configuration omits KVCR-owned local persistence, so unused local-tier cascade
failures are recorded separately from peer delivery failures.

This checks the peer reuse path and output consistency. It does not establish a
semantic golden answer, inspect every KV byte, test local persistence or
recovery, or measure performance. CPU tests exercise lifecycle logic and the CI
evidence checks, but do not replace the GPU job.

## Operations setup

Provision these repository variables through the normal operations review:

| Variable | Value / purpose |
| --- | --- |
| `KVCR_GPU_CI_ENABLED` | Set to `true` only after the remaining setup is ready. |
| `KVCR_E2E_BUILD_RUNNER` | Approved builder label; defaults to `prod-builder-v3`. |
| `KVCR_BUILDKIT_ENDPOINT` | Approved remote BuildKit endpoint reachable by that runner. |
| `ECR_REGISTRY` | Registry hostname, without a scheme or repository path. |
| `KVCR_E2E_REPOSITORY` | Dedicated ECR repository path. |
| `AWS_DEFAULT_REGION` | ECR region. |

The builder runner needs Git, Python 3, Docker/buildx, AWS CLI, and an existing
AWS identity allowed to authenticate and push to the dedicated repository. Its
BuildKit service needs enough CPU, memory, disk, and network access to compile
Dynamo and assemble the large vLLM image. The workflow does not forward registry
credentials or BuildKit secrets to Dockerfile commands. Build steps request a
`nofile` limit of 65,536; the BuildKit service's hard limit must permit this to
avoid file-descriptor exhaustion during parallel Rust compilation. Configure lifecycle
retention for CI images. The SHA tag must permit replacement when upstream
revisions advance; the unique run tag and digest preserve each tested build.

The dedicated GPU runner needs the following:

- Admission for the repository and reviewed PR merge refs, with isolation for
  code executed from PRs. The workflow automatically runs privileged resources
  only for same-repository PRs, after the enable flag is set. Fork PRs remain
  skipped and need the operations-approved contributor testing process. The
  workflow does not use `pull_request_target`.
- Permission to pull the private ECR image **before** the job container starts,
  through the runner's Kubernetes image-pull configuration or credential
  provider. A later `docker login` step cannot solve this bootstrap requirement.
- Exactly two allocated NVIDIA GPUs with a compatible driver, visible inside
  the job container. `nvidia-smi` alone is insufficient; the test performs CUDA
  computation. Consult the [quick-start prerequisites](quick-start.md#prerequisites)
  for the pinned CUDA runtime's driver requirement.
- At least 8 GiB **free** shared memory, suitable memlock limits for two 2 GB
  host-pinned tiers, and sufficient host RAM for the model and runtime. Configure
  `/dev/shm` and limits in the runner's Kubernetes hook/pod setup; the workflow
  does not assume Docker-only `container.options` work there.
- Access to the public model repository. The test uses an ungated model and
  needs no Hugging Face token. An optional managed model cache may reduce startup
  time. Keep each job's discovery files and local ports isolated.

The CI image defaults to root; an Actions Kubernetes hook can override its UID.
Verify that the actual job UID can write the Actions workspace, artifact
directory, and model cache. The ordinary quick-start image continues to default
to the `vllm` user. The GPU test needs registry-pull access only.

Enable the variable only after the operations PR is approved and these settings
are verified. Run one reviewed PR or `workflow_dispatch`, inspect both jobs and
their artifacts, then select the GPU check as a required branch-protection
check if desired. Branch protection and repository variables are not configured
by this change.

## Running the same checks locally

Resolve source inputs without building (a dirty checkout is allowed here):

```bash
bash .github/scripts/build_vllm_e2e.sh --resolve-only
```

Build from a clean committed checkout on a Linux amd64 builder:

```bash
KVCR_E2E_IMAGE=local/kvcr-vllm-e2e \
KVCR_PUSH=0 \
bash .github/scripts/build_vllm_e2e.sh
```

Set `KVCR_BUILDER` to an existing buildx builder if needed. Local mode loads the
image into Docker; `KVCR_PUSH=1` publishes after registry authentication.
`E2E_BUILD_OUTPUT` overrides the default `artifacts/build` directory.

Run that locally loaded image on two explicitly selected, allocated GPUs:

```bash
bash .github/scripts/run_vllm_e2e.sh --gpus 0,1
```

The wrapper reads `artifacts/build/image-identity.json` and
`artifacts/build/source-revisions.json`, verifies that the image and checkout
match, and runs the same E2E harness in a new container. It mounts the committed
test script read-only, supplies shared memory and memlock settings, publishes no host ports,
and removes only its own container. It requires an explicit pair of device
indices or full GPU UUIDs; there is no implicit access to all host GPUs.
Use `--help` for artifact-directory overrides and published-image execution.
On a shared Slurm host, also pass `--cpus`, `--cpuset-cpus`, and `--memory`
within the assigned allocation; Docker containers may not inherit the invoking
Slurm step's CPU and memory cgroup. The wrapper applies the memory limit to
memory plus swap as well. Use the allocation's GPU UUIDs when host indices and
Slurm-visible indices differ.

Building and running are separate commands because the build may happen on a
CPU machine while testing happens on a GPU machine. When using different
machines, publish the image and transfer the source-revisions manifest and
matching Git checkout; use the wrapper's digest-pinned `--image` mode.

Inside the resulting image, with exactly two assigned GPUs, sufficient shared
memory, and the matching checkout mounted for the test script:

```bash
python3 tests/e2e/run_vllm.py \
  --output-dir /artifacts/e2e --timeout 1200 --check-output-consistency
```

The harness starts and stops its own frontend and worker process groups. To
verify an existing idle deployment without changing its processes, supply both
`--base-url http://HOST:PORT` and `--worker-log /path/to/worker.log`. That mode
requires exclusive test traffic and the shared log of the same worker; package
versions recorded on the client do not describe the remote server. Always use
a new, empty output directory.

The CI orchestration and evidence-check tests need Git, Bash, pytest and
Python 3.12. Docker calls are mocked, so they need no Docker daemon or GPUs:

```bash
python3 -m pytest tests/ci -q
```

## Evidence and failure investigation

Both workflow jobs upload artifacts on success or failure for 14 days. The build
artifact includes resolved refs/SHAs, base-image digest, GitHub run identity,
build log, build metadata, and verified image identity. The image itself carries
`/opt/kvcr-build/source-revisions.json` and OCI labels.

The GPU artifact includes image provenance, launch configuration, preflight
results, exact HTTP requests and responses, frontend/worker logs, the bounded
worker-log window, parsed interval metrics, `result.json`, and `junit.xml`.
JUnit represents the full transfer check as one case; readiness, validation,
or cleanup failures fail that case and return a nonzero process exit code.
CPU CI also retains a JUnit report for `tests/unit` and `tests/ci`.
Read the result
and HTTP evidence first, then the worker logs. Missing telemetry, wrong rank,
zero remote cache reuse, or source/destination disagreement must remain failed
checks; do not fall back to a successful `nvidia-smi` result.

Dynamo main and the requested vLLM PR are intentionally moving inputs. A build
compatibility failure identifies the resolved revisions in its artifact; it
must be investigated rather than silently replaced with older revisions.
