<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVCR Quick Start with SGLang HiCache

Build one image containing **Dynamo + SGLang HiCache + KVCR + NIXL**, then run
an automated two-worker remote-cache correctness check. No source edits,
private registry, NVIDIA-internal credentials, or existing custom image are
required. You can give this guide to an AI agent; the agent contract below
defines what it must check before reporting success.

> [!WARNING]
> **Draft integration preview**, not a released compatibility promise. The
> public Docker build and complete small-model GPU acceptance must both pass.
> Prior source-checkout benchmarks do **not** validate this Docker image.
> See [validation status](#validation-status). Do not replace the pins with
> `main` or suppress a failing check.

This enables **HiCache storage only**, not the direct Linker or Dynamo's
ownership-aware state-agent path. It does not promise Guard restart
persistence. See the separate [Linker preview](sglang-linker-preview.md).
The [vLLM quick start](quick-start-vllm.md) uses a different image and procedure.

## Scope and prerequisites

One Linux **x86-64** host, two idle **B200 GPUs allocated to you**, two
independently complete TP1 SGLang workers, and one real Dynamo KV-aware frontend.
The model is pinned Qwen3-0.6B, BF16, `trtllm_mha`, 64-token pages, and the
Python unified radix tree. HiCache host staging is **2 GiB per worker**, separate
from **1 GiB per worker** of KVCR DRAM. HBM capacity is capped at 8192 tokens.

This is a **same-host correctness smoke test**, using loopback endpoints and
`UCX_TLS=tcp,sm,self,cuda_copy`, not an RDMA, cross-host, TP>1, hybrid-model, or performance
qualification. The initial backend/hardware scope is B200; other GPUs need a
separate validation, not silent backend substitution.

You need:

- A CUDA 13.0-compatible NVIDIA driver (580.126.20 or newer recommended),
  Docker Engine with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  Git, Bash, `timeout`, and Python 3.
- Budget at least 32 GiB host RAM, four build CPUs, and 150 GiB free disk for
  sources, Rust intermediates, image layers, model cache and runtime. These are
  planning allowances, not measured minimums.
- Public access to Docker Hub, GitHub, PyPI, crates.io, Debian/Ubuntu mirrors,
  `flashinfer.ai`, and Hugging Face. This public model needs no HF token.
- No other compute processes on the selected GPUs. The verifier refuses busy
  GPUs and never kills foreign processes or acquires a cluster reservation.

The HTTP frontend and KVCR control endpoints bind loopback; Dynamo's native
endpoints and KV-event subscribers use the container's address. The event
publisher therefore binds `*` inside that namespace. No ports are published to the host.
Do not add host networking or share another session's discovery directory.

## 1. Get and record the recipe

Until this PR merges, fetch its public ref explicitly in a new directory:

```bash
set -euo pipefail
git clone https://github.com/ai-dynamo/kvcr.git kvcr-sgl-quick-start
cd kvcr-sgl-quick-start
git fetch origin pull/114/head
git checkout --detach FETCH_HEAD
git diff --exit-code

export QS_EVIDENCE="$(mktemp -d /tmp/kvcr-sgl-evidence.XXXXXX)"
git rev-parse HEAD | tee "$QS_EVIDENCE/recipe-commit.txt"
```

This resolves the PR once. For a repeat, use the **recorded commit**, not a
refreshed PR. The checkout must contain `Dockerfile.quick-start-sglang` and
`examples/sglang-hicache/verify.py`.

| Component | Exact public source |
| --- | --- |
| SGLang HiCache performance leaf, including foundation | `142e34c04b9dc828553178c0b600dc9aa2cb9b42` |
| Dynamo router-hint integration, PR #16117 | `66ddd723d6024f8b7e2d3d8c8d0e0b31f60c92a2` |
| KVCR transfer batching, PR #113 | `bb48f3843a292f514becd3efedf29429e272c0af` |
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` |

The [Dockerfile](../Dockerfile.quick-start-sglang) pins public AMD64 base and
Rust-builder image digests, builds Dynamo's native runtime and Python wheel,
and installs the full pinned SGLang Python implementation and KVCR package.
It does not overlay a few files onto an unknown SGLang release. SGLang's source
is in a public contributor fork while the review stack is pending.

Critical kernel/runtime pins are in
[`constraints.txt`](../examples/sglang-hicache/constraints.txt). They are not a
complete transitive Python/OS lock; keep the package manifest and built image
ID. Do not describe this build as hermetic.

## 2. Build the integration image

```bash
set -euo pipefail
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --progress plain \
  --file Dockerfile.quick-start-sglang --tag kvcr-sgl-quick-start:local . \
  2>&1 | tee "$QS_EVIDENCE/build.log"

export QS_IMAGE_ID="$(docker image inspect kvcr-sgl-quick-start:local --format '{{.Id}}')"
printf '%s\n' "$QS_IMAGE_ID" | tee "$QS_EVIDENCE/image-id.txt"
```

**Stop on build failure.** The final step runs `verify.py --check-image`: native
Dynamo, HiCache backend and hint-adapter imports; critical versions; CUDA
family; source provenance. It records `image-check.json` and
`installed-packages.txt`. This is a **CPU canary**, not GPU acceptance.

No GPU is needed to build. The first Rust/image build can take substantially
longer than launching an existing image. Diagnose downloads, package resolution,
native compilation and imports at their own stage; do not bypass failures with
`--no-deps`, private images, or changed pins.

## 3. Run automated GPU acceptance

Inspect the GPUs first. This example selects `0,1`; substitute only two GPUs
allocated to you. Do not expose all GPUs on a shared host.

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv

export QS_NAME="kvcr-sgl-$(date -u +%Y%m%dT%H%M%SZ)-$$"
docker volume create "$QS_NAME-models"
export QS_CID="$(docker run --detach --init --name "$QS_NAME" \
  --gpus '"device=0,1"' --shm-size 8g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --volume "$QS_NAME-models:/root/.cache/huggingface" \
  --volume "$QS_EVIDENCE:/evidence" \
  "$QS_IMAGE_ID" python3 /opt/kvcr-quick-start/verify.py --output /evidence/run)"
printf '%s\n' "$QS_CID" | tee "$QS_EVIDENCE/container-id.txt"

# Overall bound, including downloads and three worker startups.
if ! timeout 5400 docker wait "$QS_CID" > "$QS_EVIDENCE/container-exit.txt"; then
  docker stop --time 90 "$QS_CID"
  docker logs "$QS_CID" > "$QS_EVIDENCE/container.log" 2>&1
  echo "Timed out or lost Docker access: inspect evidence; do not report PASS."
  exit 1
fi
docker logs "$QS_CID" > "$QS_EVIDENCE/container.log" 2>&1
docker cp "$QS_CID:/opt/kvcr-quick-start/build-check" "$QS_EVIDENCE/build-check"
docker cp "$QS_CID:/opt/kvcr-quick-start/installed-packages.txt" "$QS_EVIDENCE/"
```

The verifier downloads the pinned model and owns all launched processes. It
uses fresh discovery state, workers and cache pools; startup, publication and
request waits are bounded. The frontend generates trusted source hints: the
test pins each request's **destination**, but never fabricates source hints or
bypasses the router. Repeating on the same warm worker cannot pass.

Required gates:

1. A new **2048-token cold source** request has zero cached tokens and completes
   its L3 deposit before the destination request.
2. The response comes from the **fresh second worker**, restoring exactly
   **1984 tokens** (31 full pages), with matching completed HiCache prefetch.
3. Successful peer-delivery operations and **terminal native source-write bytes
   and blocks** match the pinned model's K/V geometry. Failed transfers or lost
   diagnostic observations fail the check.
4. A destination HBM repeat hits; a different-prefix control misses. Neither
   causes additional L3 prefetch.
5. All **16 output token IDs** match cold source, restored destination, HBM
   repeat, and a **fresh KVCR-disabled cold worker on the same destination GPU**.
   Exact input echoes, worker identities, lengths, one terminal completion and
   usage accounting must also agree.
6. Owned processes stop; `result.json` has `passed: true` **and**
   `gpu_acceptance: true`; evidence checksums pass.

The container exits after checking and cleaning up its own processes. It does
not release the underlying GPU allocation. This bounded smoke test does not
prove universal model accuracy or performance. A token mismatch is a failure
to investigate, never a reason to relax equality.

## 4. Check and retain evidence

`docker wait` itself can succeed when the container exits nonzero. Check its
**reported exit code**, the full GPU result, and checksums separately:

```bash
test "$(cat "$QS_EVIDENCE/container-exit.txt")" = 0
python3 - "$QS_EVIDENCE/run/result.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    result = json.load(stream)
assert result.get("passed") is True and result.get("gpu_acceptance") is True, result
print("PASS: two-worker HiCache/KVCR acceptance, including cold destination control")
PY
(cd "$QS_EVIDENCE/run" && sha256sum --check SHA256SUMS)
```

Retain the whole directory, including failures: recipe commit, image ID, build
log, packages, commands, model revision, GPU UUIDs, worker identities, raw SSE,
native logs, trace, acceptance/cleanup receipts and checksums. Retrying requires
a new evidence directory and container identity, not overwriting the failure.

## 5. Remove only this container

After evidence collection, use the recorded **container ID**, not a broad
name filter or `docker system prune`:

```bash
docker stop --time 90 "$QS_CID"
docker rm "$QS_CID"
```

Retain the model volume unless you deliberately want to discard this session's
download. Never delete shared caches, Pods, reservations or another user's work.

## Troubleshooting and AI-agent contract

- **Build/import failure:** keep `build.log`; identify the failed download, pin,
  wheel or system library. Do not switch branches, inject untracked files,
  change kernel pins or skip the canary.
- **Startup failure:** inspect the named worker log and launch receipt. Check
  driver, architecture, backend and memory. Readiness is not inference success.
- **Destination cache miss:** check completed source publication, both worker
  identities, KV-aware frontend/two-tier policy, `router_hint` capability and
  reachable advertised endpoints. A warm-source repeat is not a substitute.
- **Transfer timeout/output mismatch:** preserve evidence and controls. Never
  label fallback recomputation as successful remote reuse.
- **Public download failure:** resolve ordinary network access with the
  operator. Do not retrieve others' credentials or bypass TLS verification.
- **No Docker/compatible GPUs:** report the prerequisite. A source-only run in
  a developer container does not qualify the built image.
- **Definition of done:** successful build, the exact built image exercised,
  every GPU gate passed, and checksums verified. Otherwise state the last
  proven stage and blocker; do not claim production readiness.

CPU-only parser/telemetry tests (install pytest first) can be run separately:

```bash
python3 -m pytest --noconftest tests/unit/test_sglang_quick_start.py -q
```

`--noconftest` isolates these standalone example tests from native KVCR fixtures;
it does not bypass any image or GPU gate.

## Validation status

Validation on 2026-10-21:

| Gate | Result |
| --- | --- |
| Standalone verifier CPU tests | 16 passed, including false-positive and discovery-race checks |
| Shell fences, relative links, lint and whitespace | Passed |
| Import/version canary with CUDA devices hidden | Passed in the existing developer runtime |
| This small-model verifier with pinned source checkouts on two B200 GPUs | Passed: 1984 restored tokens, 31 native blocks / 227540992 bytes, all 16 output IDs equal across source, restore, HBM and independent cold destination |
| Clean build of the public Dockerfile | **Not run**: no Docker daemon in the available validation environments |
| GPU acceptance inside that freshly built public image | **Not run** |

The source-runtime run used a pre-existing developer installation with the
recorded package versions; it did not build the image above. Earlier failed
attempts were preserved: an inherited offline model-download setting, an
initial UCX transport configuration missing CUDA memory support, and a
replacement-worker discovery race in the verifier. The final recipe corrects
the latter two without changing serving code or relaxing acceptance checks.

The HiCache leaf also has prior two-B200 Qwen3-32B-FP8 source-checkout validation.
Neither source-runtime result qualifies the public Docker image.

This PR adds the public build, CPU canary and automated small-model verifier.
A clean public Docker build followed by its full GPU acceptance remains a
required release gate. Keep the preview warning until the immutable built
image and passing evidence are recorded. Linker, cross-host RDMA, Guard
replacement, hybrid cache pools and TP>1 remain outside this guide's claims.
