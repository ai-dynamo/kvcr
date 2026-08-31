# KVCR Quick Start

This guide builds and runs a container containing KV Cache Runner (KVCR),
Dynamo, vLLM, and NIXL. It is for users who want to try the integrated stack
without editing source code in any of those projects.

> [!IMPORTANT]
> This guide uses the public, still-open vLLM
> [KVCR secondary-tier adapter PR #53624](https://github.com/vllm-project/vllm/pull/53624)
> at an immutable commit. Treat this as a pinned public preview until the PR is
> merged and released in vLLM.

For source builds, editable installs, API development, or test workflows, use
the [developer guide](dev-guide.md).

---

## Prerequisites

- A Linux host with at least two supported NVIDIA GPUs;
- NVIDIA driver 580.00.03 or newer for the CUDA 13 runtime;
- Docker Engine with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html);
- network access to GitHub, PyPI, Docker Hub, and the model source; and
- enough free disk space for the multi-gigabyte Dynamo/vLLM image and its build
  layers.

The example below runs two data-parallel ranks on one host, so it requires two
visible GPUs. It uses loopback addresses and a 2 GB host tier per rank. Adjust
the model, memory, and network addresses for the target system.

Run the commands below from the KVCR repository root.

---

## 1. Build the integration image

The repository includes [Dockerfile.quick-start](../Dockerfile.quick-start).
Pin the public adapter source used by this guide, then build the integration
image:

```bash
export KVCR_VLLM_REPO=https://github.com/vllm-project/vllm.git
export KVCR_VLLM_REF=35ab7457aafa89d6849e40d01401c69ffff8e33a

DOCKER_BUILDKIT=1 docker build \
  --build-arg KVCR_VLLM_REPO="$KVCR_VLLM_REPO" \
  --build-arg KVCR_VLLM_REF="$KVCR_VLLM_REF" \
  --file Dockerfile.quick-start \
  --tag kvcr-quick-start:local \
  .
```

No private-repository credentials are required. The first build uses pinned,
tested vLLM and Dynamo revisions and may take several minutes. Continue only if
its final compatibility checks pass.

---

## 2. Start the container

Create a persistent model-cache volume, then start one container with host
networking and all GPUs visible:

```bash
docker volume create kvcr-hf-cache

docker run --detach \
  --name kvcr-quick-start \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --volume kvcr-hf-cache:/home/vllm/.cache/huggingface \
  --entrypoint /bin/sleep \
  kvcr-quick-start:local infinity
```

For a gated model, export `HF_TOKEN` on the host and add `--env HF_TOKEN` to
the `docker run` command. The image defaults to TCP requests, ZMQ events, and
file discovery at `/tmp/dynamo-kvcr-quick-start`; every `docker exec` shell
inherits those settings.

The rest of this guide uses three terminals: one container shell for the
frontend, one for the worker, and one host shell for requests.

---

## 3. Start the Dynamo KV router

Open the frontend terminal and enter the container:

```bash
docker exec -it kvcr-quick-start bash
```

Then start the OpenAI-compatible endpoint on port 8000:

```bash
env -u NATS_SERVER python3 -m dynamo.frontend \
  --router-mode kv \
  --http-port 8000
```

KVCR requires Dynamo's KV-aware router because it supplies the request-scoped
source hints used for remote reuse. A round-robin router or vLLM's
`consistent_hash` router does not produce those hints.

---

## 4. Start a KVCR-enabled worker

Open the worker terminal and enter the same container:

```bash
docker exec -it kvcr-quick-start bash
```

Define the model and the two vLLM configurations. This example downloads a
small model into the persistent cache on first use; `MODEL` may instead name a
compatible local path mounted into the container.

```bash
export MODEL=Qwen/Qwen3-0.6B

export KV_TRANSFER_CONFIG='{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 2000000000,
    "enable_external_pinning": true,
    "self_describing_kv_events": true,
    "secondary_tiers": [
      {
        "type": "kvcr",
        "router_capabilities": ["router_hint"],
        "control_host": "0.0.0.0",
        "control_ports": [17771, 17772],
        "control_advertise_host": "127.0.0.1",
        "operation_timeout_ms": 5000,
        "enable_telemetry": true
      }
    ]
  }
}'

export KV_EVENTS_CONFIG='{
  "publisher": "zmq",
  "topic": "kv-events",
  "endpoint": "tcp://*:20081",
  "enable_kv_cache_events": true
}'
```

Launch two data-parallel ranks:

```bash
env -u NATS_SERVER CUDA_VISIBLE_DEVICES=0,1 python3 -m dynamo.vllm \
  --model "$MODEL" \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --block-size 64 \
  --max-model-len 32000 \
  --enable-prefix-caching \
  --disable-hybrid-kv-cache-manager \
  --enforce-eager \
  --gpu-memory-utilization 0.8 \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --kv-events-config "$KV_EVENTS_CONFIG"
```

The important relationships are:

| Setting | Requirement |
| --- | --- |
| `cpu_bytes_to_use` | Capacity of vLLM's primary host-pinned tier, per the adapter's configured scope |
| `enable_external_pinning` | Lets KVCR safely serve framework-owned host blocks to a peer |
| `self_describing_kv_events` | Supplies block metadata needed by Dynamo's tier-aware index |
| `router_capabilities` | Advertises that the tier accepts `router_hint` plans |
| `control_ports` | Contains one unique port per local DP rank, in rank order |
| `control_advertise_host` | Is reachable by peer workers; loopback is valid only on one host |
| KV events endpoint | Does not overlap the control-port range |

This example leaves `secondary_g2_slots` at its default of zero. It exercises
direct remote framework-DRAM mode: vLLM owns the 2 GB primary host tier, while
KVCR registers and pins those blocks for a peer transfer. It does not allocate
a separate KVCR-owned local G2 pool. Because there is no local KVCR deposit
target, the generic primary-to-secondary cascade may increment
`kv_offload_tiering_cascade_job_failures`; that counter is not a peer-transfer
result. Use the terminal source and destination telemetry in the verification
section below to judge the transfer itself.

---

## 5. Verify a KVCR peer transfer

First make a real inference request and read the registered worker ID shared by
the two DP ranks from the response:

```bash
export MODEL=Qwen/Qwen3-0.6B

WORKER_ID=$(
  curl --fail --silent --show-error \
    http://127.0.0.1:8000/v1/completions \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"worker identity probe\",\"max_tokens\":1,\"temperature\":0,\"nvext\":{\"extra_fields\":[\"worker_id\"]}}" \
  | python3 -c 'import json, sys; response = json.load(sys.stdin); assert response["choices"]; print(response["nvext"]["worker_id"]["decode_worker_id"])'
)
export WORKER_ID
```

Use a fresh prefix spanning several complete blocks. Send it first to DP rank 0,
then send the shared prefix to DP rank 1:

```bash
PREFIX=$(python3 -c 'import uuid; tag = uuid.uuid4().hex; print(" ".join(f"{tag} section{i} cache routing data" for i in range(80)))')

curl --fail --silent --show-error \
  http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -H "x-dynamo-worker-instance-id: $WORKER_ID" \
  -H 'x-dynamo-dp-rank: 0' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"$PREFIX seed source\",\"max_tokens\":8,\"temperature\":0,\"nvext\":{\"extra_fields\":[\"worker_id\"]}}"

# Allow the asynchronous full-block KV events to reach the local router.
sleep 2

curl --fail --silent --show-error \
  http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -H "x-dynamo-worker-instance-id: $WORKER_ID" \
  -H 'x-dynamo-dp-rank: 1' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"$PREFIX retrieve on target\",\"max_tokens\":8,\"temperature\":0,\"nvext\":{\"extra_fields\":[\"worker_id\"]}}"
```

These headers constrain the KV router to a specific DP rank; they do not bypass
it. For the second request, the router can still identify rank 0 as the source
and provide its source hint to rank 1. The two responses should report
`decode_dp_rank` 0 and 1, respectively, under `nvext.worker_id`.

Compare the periodic `KV Transfer metrics` immediately before and after the
second request. The source must report successful `transfer` and `source_write`
operations with positive block and byte deltas. The destination must report a
successful, not partial, `remote_deliver` with the same block delta, and its
`kv_offload_tiering_read_bytes` delta must equal the source byte delta.

The explicit rank selection is only for this deterministic mechanism test. In
a normal deployment, omit the two routing headers. KVCR transfers can occur
when load, availability, or routing constraints cause the KV router to select a
target other than the cache-owning worker; observe them with the same terminal
source and destination metrics.

---

## Troubleshooting

### The image does not build

- Confirm that the host can pull the pinned `vllm/vllm-openai` image and reach
  the public vLLM source, Dynamo's GitHub repository, and PyPI.
- Confirm that `KVCR_VLLM_REPO` is the public vLLM repository and
  `KVCR_VLLM_REF` is the immutable PR #53624 commit shown above.
- Read the final compatibility-check output. It identifies whether the Dynamo
  router build, the vLLM adapter, or KVCR failed.
- Keep the base image, `DYNAMO_REF`, and `KVCR_VLLM_REF` together. Overriding
  only one can produce an importable but incorrect runtime path.
- Do not bypass the import and compatibility canary. A successfully created
  layer is not sufficient if the assembled serving path is inconsistent.

### The KVCR tier does not initialize

- Verify that the package imports as `kvcr`.
- Verify that the vLLM tier adapter expects `"type": "kvcr"`.
- Make `control_ports` a list with exactly one entry per local DP rank.
- Check that every control and KV-events port is unique and available.
- Confirm that the installed NIXL version matches the `nvidia-kvcr` pin.

### Dynamo does not produce router hints

- Use `--router-mode kv`; round-robin and `vllm-router` do not create KVCR
  source hints.
- Keep `self_describing_kv_events` and KV event publication enabled.
- Confirm that the registered worker advertises `router_hint` and its control
  endpoints.
- Use a shared prefix longer than one full vLLM block.
- Confirm that the source still owns or can pin the exact hinted block keys.

A high overlap score alone does not prove hint delivery. Check that the vLLM
adapter passes the source endpoint, request ID, mode, and keys into KVCR.

### Peer control connects but transfer fails

Binding to `0.0.0.0` does not make it a usable destination. Peers must connect
to `control_advertise_host`, and that address must resolve and route from the
destination process.

A successful control acknowledgement does not prove payload correctness, and
a successful NIXL submission does not prove completion. Check the source pin,
memory descriptors, NIXL terminal status, and destination data before exposing
the block to vLLM.

For the full diagnostic checklist, see the
[developer guide](dev-guide.md#troubleshooting).
