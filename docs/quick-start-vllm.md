# KVCR Quick Start with vLLM

This guide builds and runs a container containing KV Cache Runner (KVCR), Dynamo, vLLM, and NIXL. It is for users who want to try the integrated stack without editing source code in any of those projects.

> [!IMPORTANT]
> This is a public preview using vLLM's supported KVCR secondary-tier adapter. The container recipe pins the exact adapter, vLLM base image, and landed Dynamo revision validated together for this guide.

For source builds, editable installs, API development, or test workflows, use the [developer guide](dev-guide.md).

For a proposed Kubernetes deployment with an optional KVCR guard service sidecar, see [Dynamo PR #14695](https://github.com/ai-dynamo/dynamo/pull/14695). That example requires a compatible prebuilt Dynamo vLLM runtime image.

---

## Prerequisites

- A Linux x86-64 host with at least two supported NVIDIA GPUs;
- NVIDIA driver 580.126.20 or newer for the pinned CUDA 13.0.2 runtime;
- Docker Engine with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html);
- Git, curl, and Python 3 for the build and verification commands;
- network access to GitHub, PyPI, Docker Hub, crates.io, Debian package mirrors, and the model source; and
- enough free disk space for the multi-gigabyte Dynamo/vLLM image and its build layers.

The example below runs two data-parallel ranks on one host, so it requires two visible GPUs. It uses loopback addresses, a 2 GB host tier per rank, and an 8 GB shared-memory allocation for both ranks' host allocations plus vLLM IPC. Adjust the model, memory, and network addresses for the target system.

Run the commands below from the KVCR repository root.

---

## 1. Build the integration image

The repository includes [Dockerfile.quick-start-vllm](../Dockerfile.quick-start-vllm). The build fetches pinned adapter source and applies it to the base image. Pass the public repository and exact revision explicitly:

```bash
export KVCR_VLLM_REPO=https://github.com/vllm-project/vllm.git
export KVCR_VLLM_REF=fed7de909f894fbee0c45130376aed2e2c9bf0d3

DOCKER_BUILDKIT=1 docker build \
  --build-arg KVCR_VLLM_REPO="$KVCR_VLLM_REPO" \
  --build-arg KVCR_VLLM_REF="$KVCR_VLLM_REF" \
  --file Dockerfile.quick-start-vllm \
  --tag kvcr-quick-start:local \
  .
```

No private-repository credentials are required. The first build uses pinned, tested vLLM and Dynamo revisions and may take several minutes. Continue only if its final compatibility checks pass.

---

## 2. Start the container

Create a persistent model-cache volume, then start one container with host networking, all GPUs visible, and enough shared memory for both ranks' host allocations:

```bash
docker volume create kvcr-hf-cache

docker run --detach \
  --name kvcr-quick-start \
  --gpus all \
  --network host \
  --shm-size 8g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --volume kvcr-hf-cache:/home/vllm/.cache/huggingface \
  --entrypoint /bin/sleep \
  kvcr-quick-start:local infinity
```

For a gated model, export `HF_TOKEN` on the host and add `--env HF_TOKEN` to the `docker run` command. The image defaults to TCP requests, ZMQ events, and file discovery at `/tmp/dynamo-kvcr-quick-start`; every `docker exec` shell inherits those settings.

The rest of this guide uses three terminals: one container shell for the frontend, one for the worker, and one host shell for requests.

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

This quick-start integration requires Dynamo's KV-aware router because it supplies the request-scoped source hints used for remote reuse. A round-robin router or vLLM's `consistent_hash` router does not produce those hints.

---

## 4. Start a KVCR-enabled worker

Open the worker terminal and enter the same container:

```bash
docker exec -it kvcr-quick-start bash
```

Define the model and the two vLLM configurations. This example downloads a small model into the persistent cache on first use; `MODEL` may instead name a compatible local path mounted into the container.

```bash
export MODEL=Qwen/Qwen3-0.6B

export KV_TRANSFER_CONFIG='{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 2000000000,
    "self_describing_kv_events": true,
    "secondary_tiers": [
      {
        "type": "kvcr",
        "router_capabilities": ["router_hint"],
        "control_host": "0.0.0.0",
        "control_ports": [17771, 17772],
        "control_advertise_host": "127.0.0.1",
        "operation_timeout_ms": 5000,
        "abandon_timeout_ms": 10000,
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

Launch two data-parallel ranks with DEBUG logging to show transfer metrics even when the source rank is idle:

```bash
env -u NATS_SERVER CUDA_VISIBLE_DEVICES=0,1 VLLM_LOGGING_LEVEL=DEBUG \
  python3 -m dynamo.vllm \
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
| `self_describing_kv_events` | Supplies block metadata needed by Dynamo's tier-aware index |
| `router_capabilities` | Advertises that the tier accepts `router_hint` plans |
| `control_ports` | Contains one unique port per local DP rank, in rank order |
| `control_advertise_host` | Is reachable by peer workers; loopback is valid only on one host |
| KV events endpoint | Does not overlap the control-port range |

This configuration intentionally omits `secondary_g2_slots` and `g3`. It tests KVCR peer transfer between the ranks' vLLM-owned host tiers, not KVCR-owned local persistence.

---

## 5. Verify a KVCR peer to peer transfer

In the host request terminal, set `MODEL` to the same value used in Step 4; replace the example below if you chose a different model. Then make an inference request to read the registered worker ID shared by both DP ranks:

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

Use a fresh prefix spanning several complete blocks. Send it first to DP rank 0, then send the shared prefix to DP rank 1:

```bash
PREFIX=$(python3 -c 'import uuid; tag = uuid.uuid4().hex; print(" ".join(f"{tag} section{i} cache routing data" for i in range(80)))')

curl --fail --silent --show-error \
  http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -H "x-dynamo-worker-instance-id: $WORKER_ID" \
  -H 'x-dynamo-dp-rank: 0' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"$PREFIX seed source\",\"max_tokens\":8,\"temperature\":0,\"nvext\":{\"extra_fields\":[\"worker_id\"]}}"

# CPU offload, KV-event publication, and router indexing are asynchronous.
sleep 15

curl --fail --silent --show-error \
  http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -H "x-dynamo-worker-instance-id: $WORKER_ID" \
  -H 'x-dynamo-dp-rank: 1' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"$PREFIX retrieve on target\",\"max_tokens\":8,\"temperature\":0,\"nvext\":{\"extra_fields\":[\"worker_id\"]}}"
```

The two JSON responses include these fields in one validated run:

| Request | `decode_dp_rank` | `decode_worker_id` | `cached_tokens` |
| --- | --- | --- | --- |
| Seed | 0 | 5683645024925323515 | 0 |
| Retrieval | 1 | 5683645024925323515 | 2624 |

Both ranks share a worker ID. The fresh seed should report zero cached tokens; the retrieval must report a positive, block-aligned count. Values vary. If retrieval reports zero, generate a new `PREFIX` before retrying so rank 1 cannot reuse its own cached prefix.

These headers constrain the KV router; they do not bypass it. For the retrieval, the router can still identify rank 0 as the source and provide its source hint to rank 1. The response is useful evidence, but terminal telemetry is the final proof. The relevant post-request `KV Transfer metrics` from that run contained:

```text
source rank 0:
  vllm:kvcr_duration_seconds:('transfer', 'success')_count=1
  vllm:kvcr_duration_seconds:('source_write', 'success')_count=1
  vllm:kvcr_transfer_blocks:('source_write',)=41
  vllm:kvcr_transfer_bytes:('source_write',)=300941312

destination rank 1:
  vllm:kvcr_duration_seconds:('remote_deliver', 'success')_count=1
  vllm:kvcr_transfer_blocks:('remote_deliver',)=41
  vllm:kv_offload_tiering_read_bytes:('1:kvcr',)=300941312
```

The exact counts may differ, but the source and destination block counts must match, as must the source-write and destination-read byte counts. There must be no positive `remote_deliver` `partial` or `failed` count. These values reset after each reporting interval, so inspect the interval or intervals covering the retrieval instead of subtracting two log lines.

Because this peer-only configuration has no KVCR-owned local tier, the same log can contain `kv_offload_tiering_cascade_job_failures` for the unused local deposit path. That counter is separate from peer delivery and does not invalidate the matching successful transfer above.

The explicit rank selection is only for this deterministic mechanism test. In a normal deployment, omit the two routing headers. KVCR transfers can occur when load, availability, or routing constraints cause the KV router to select a target other than the cache-owning worker. Verify transfers using the same telemetry criteria above.

---

## Troubleshooting

- **Build failure:** Check access to the public sources and package indexes, and inspect the final compatibility-check output. Keep the base image, `DYNAMO_REF`, and `KVCR_VLLM_REF` together.
- **Tier initialization failure:** Check that the `kvcr` tier is registered, NIXL matches the KVCR pin, and shared memory covers both host allocations and vLLM IPC.
- **No router hint or cache hit:** Use `--router-mode kv`, enable self-describing KV events, and confirm the worker advertises `router_hint` and its control endpoints. Use a fresh prefix longer than one block.
- **Peer connection or transfer failure:** Ensure control and event ports are unique, with one control port per DP rank, and `control_advertise_host` is reachable by peers. Check terminal transfer status and destination data; a control acknowledgement alone does not prove delivery.

For more checks, see the [developer guide](dev-guide.md#troubleshooting).
