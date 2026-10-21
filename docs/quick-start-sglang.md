<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVCR with SGLang and Dynamo

> [!WARNING]
> Integration-preview draft. The SGLang integrations and Dynamo companion are
> under review. The commands below are a candidate recipe, not yet a validated
> public quick start. A public, immutable image/build recipe and a complete
> two-worker restore check are required before publishing this guide.

This guide is for operators trying KVCR-backed reuse between two independently
complete SGLang replicas. For KVCR API development and source installation, use
the [developer guide](dev-guide.md). For vLLM, use the existing
[vLLM quick start](quick-start.md).

## Choose one integration

| | HiCache storage | Direct linker |
| --- | --- | --- |
| Device restore path | KVCR DRAM → SGLang host staging → HBM | KVCR DRAM → device pools |
| SGLang host tier | Enabled, separately sized | No HiCache host tier |
| Selector | `--hicache-storage-backend kvcr` | `--unified-cache-external-linker-backend kvcr` |
| DRAM budget option | `local_dram_bytes` (bytes per rank) | `dram_size_gb` (GiB per rank) |
| Advertised address | `control_advertise_host` | `advertise_host` |
| Payload listen-port option | `nixl_listen_port` | `nixl_port` |

Do not enable both modes on one engine. Neither mode makes another model,
layout, dtype, or shard's cached bytes compatible. Do not assume their storage
formats are interchangeable. KVCR serves completed deposits in its own DRAM;
these examples do not expose unpinned SGLang pages as peer sources.

## Prerequisites and version boundary

Use a Linux x86-64 host with two supported NVIDIA GPUs, a compatible driver,
Docker with the NVIDIA Container Toolkit, and enough model-cache disk and
host RAM. Budget SGLang host staging, KVCR DRAM and other processes separately.
The example uses one TP1 engine per GPU. It is a same-host smoke test, not a
cross-host RDMA performance claim.

The image must include compatible SGLang, KVCR, NIXL/UCX and Dynamo revisions,
including [Dynamo PR #16117](https://github.com/ai-dynamo/dynamo/pull/16117).
For the ownership-aware direct linker, also use its companion
[Dynamo PR #16827](https://github.com/ai-dynamo/dynamo/pull/16827).
Use an immutable image identity and retain its source/dependency manifest.
Installing all projects' latest branches is not a compatibility check.

The new SGLang review stack starts with
[shared endpoint contracts](https://github.com/sgl-project/sglang/pull/45646),
then separates [HiCacheStorage](https://github.com/sgl-project/sglang/pull/45647)
from [linker lifetime/inventory contracts](https://github.com/sgl-project/sglang/pull/45648).
Their respective leaves are the
[HiCache performance follow-up](https://github.com/sgl-project/sglang/pull/45649)
and [direct-device linker](https://github.com/sgl-project/sglang/pull/45650).
The original [SGLang PR #43635](https://github.com/sgl-project/sglang/pull/43635)
remains available for provenance. The modes use different leaf branches; do not
assume either leaf contains the other implementation.

Recorded source-checkout acceptance used Qwen3-32B-FP8 revision
`aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df` with two same-host B200 TP1 workers.
The HiCache performance leaf was `142e34c04b9dc828553178c0b600dc9aa2cb9b42`;
the direct-linker leaf was `20cdafc07c42f4c78767e41ad34058f627a8da13`, tested
with Dynamo `a310f33e9b55e90a8da0ef742e0ca4cb3491bec2`.
These results do not validate the small-model Docker commands below, publish a
compatible public image, or qualify Guard replacement/cross-host/TP>1 operation.

For this draft, supply an already-built compatible image explicitly. Do not
substitute the vLLM quick-start image; it does not provide this SGLang stack.

```bash
: "${KVCR_SGL_IMAGE:?Set a compatible immutable SGLang integration image}"
docker volume create kvcr-sgl-model-cache
docker run --detach --name kvcr-sgl-quick-start \
  --gpus all --network host --shm-size 8g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --volume kvcr-sgl-model-cache:/root/.cache/huggingface \
  --env DYN_REQUEST_PLANE=tcp --env DYN_EVENT_PLANE=zmq \
  --env DYN_DISCOVERY_BACKEND=file \
  --env DYN_FILE_KV=/tmp/kvcr-sgl-quick-start \
  --entrypoint /bin/sleep "$KVCR_SGL_IMAGE" infinity
```

These container and discovery names are reserved for this quick-start session.
Check for existing users of the names and ports first; do not remove someone
else's container or reuse an active discovery directory. For multiple sessions,
use distinct names, discovery directories and every port below.

## Start the KV-aware frontend

In a container shell, start the frontend:

```bash
docker exec -it kvcr-sgl-quick-start bash
env -u NATS_SERVER python3 -m dynamo.frontend --router-mode kv --http-port 8000
```

The router creates the trusted request-scoped source hints consumed by the
in-process worker. A direct HTTP request with no router hint can exercise local
L3, but does not establish router-guided remote reuse. Do not accept arbitrary
client hints as trusted router metadata.

## HiCache mode: start two workers

Open two more container shells. In the first use `RANK=0`; in the second use
`RANK=1`. Run this block in each, in the foreground:

```bash
export RANK=0  # Use 1 in the second shell.
export MODEL=Qwen/Qwen3-0.6B
CONTROL_PORT=$((32100 + RANK * 100))
NIXL_PORT=$((33100 + RANK * 100))
EVENT_PORT=$((5557 + RANK))
export DYN_SYSTEM_PORT=$((8081 + RANK))

env -u NATS_SERVER CUDA_VISIBLE_DEVICES="$RANK" python3 -m dynamo.sglang \
  --model-path "$MODEL" --served-model-name "$MODEL" --tp-size 1 \
  --page-size 64 --context-length 8192 --mem-fraction-static 0.5 \
  --disable-cuda-graph --disable-piecewise-cuda-graph \
  --enable-hierarchical-cache --hicache-size 2 \
  --hicache-mem-layout page_first --hicache-io-backend kernel \
  --hicache-storage-backend kvcr \
  --hicache-storage-backend-extra-config "{\"control_advertise_host\":\"127.0.0.1\",\"control_port\":$CONTROL_PORT,\"nixl_listen_port\":$NIXL_PORT,\"local_dram_bytes\":1073741824}" \
  --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$EVENT_PORT\"}"
```

The 2 GiB HiCache host pool and 1 GiB KVCR DRAM budget are independent and
per rank. These illustrative small-model settings are not the long-context
benchmark configuration. Use the same model revision and cache layout on both
workers; pin the model revision in the final reproducible recipe.

Loopback endpoints are valid only because both workers share this host/network
namespace. For a multi-host deployment, advertise reachable worker addresses,
verify the actual payload transport, and account for all rank-offset ports.
The control endpoint's `tcp://` scheme does not prove payload TCP or RDMA.

## Linker mode: alternative worker configuration

> [!IMPORTANT]
> The linker emits `ownership="kvcr"` inventory events. Dynamo's ordinary
> SGLang `KvEventPublisher` path rejects that ownership domain. Router-hint
> forwarding in PR #16117 is not sufficient by itself. PR #16827 implements the
> ownership-aware state-agent ingress and worker lifecycle wiring for opt-in
> TP1 aggregated direct-linker workers. Its `--kvcr-state-agent-owner` option
> requires an explicit valid owner identity; it is not a boolean and must not
> be used with HiCache. The fragment below documents only the SGLang engine
> options, not a complete state-agent launch recipe. The final public guide
> still needs a pinned, runnable owner/discovery configuration and acceptance
> script for this exact setup.

The current linker draft still backs externally restored prefixes up into
destination DRAM. Its bounded-deposit fix prevents an oversized-copy timeout,
but does **not** restore the original no-retain behavior. Suppressing redundant
backup requires successful-restore provenance through post-forward insertion;
it must not advertise nonexistent destination-owned DRAM or omit offload of
newly computed KV. Treat no-retain and multi-source/replication policy as
explicit outstanding work, not an implicit property of this recipe.

Do not run this at the same time as the HiCache example on the same GPUs.
Keep the common model, event and worker setup, but replace **all** hierarchical
cache flags and the HiCache JSON with:

```bash
--enable-unified-cache-external-linker \
--unified-cache-external-linker-backend kvcr \
--hicache-storage-backend-extra-config '{
  "kvcr_service_socket_path": null,
  "dram_size_gb": 1,
  "advertise_host": "127.0.0.1",
  "control_port": 19500,
  "nixl_port": 20500
}'
```

For the second worker use control port `19600` and NIXL port `20600`.
Explicit `null` selects in-process DRAM; this quick start does not provision
Guard memory or promise recovery across engine loss. Keep TP dedup and
topology-specific performance options disabled in the initial smoke test.

## Verify real reuse, not just readiness

Wait for the frontend and **both complete engines** to become ready. An HTTP
200 or a `KVCR ready` line proves neither remote restore nor correct output.
The final acceptance script must check all of the following:

1. A cold source request populates a known, page-aligned prefix. Wait for its
   completed KVCR deposit before attempting to restore it.
2. The destination has no prior HBM, HiCache or local KVCR copy of that prefix.
   Keep the source alive and its deposits resident while testing the target.
3. The destination request receives the intended source hint and completes with
   positive, expected block-aligned restored tokens. Correlate the request,
   full storage-key identity, source endpoint and native transfer completion.
4. Compare deterministic output tokens with a cold control. Check all cache
   pools required by the model, error status, lengths and terminal native bytes.
5. A different-prefix negative control misses as expected. Retain version,
   command/configuration and completion evidence alongside the output.

Sending the same prompt twice through a KV-aware router may select the same
warm worker. That is **not** a fresh-destination peer-restore test. The final
script must verify destination identity instead of relying on router placement.
The direct-engine microbenchmark and native host-byte tests are separate gates;
neither alone validates this public Dynamo recipe.

## Stop only this session

Interrupt the three foreground processes in their own shells and let registered
transfers drain. Preserve their logs before removing the named container:

```bash
docker stop --time 60 kvcr-sgl-quick-start
docker rm kvcr-sgl-quick-start
```

Retain the model-cache volume unless you explicitly want to remove it. This
cleanup does not delete GPU reservations, Kubernetes Pods, shared caches or
resources belonging to another session.
