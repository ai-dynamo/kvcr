# KVCR Quick Start

This guide builds and runs a container containing KV Cache Runner (KVCR),
Dynamo, vLLM, and NIXL. It is for users who want to try the integrated stack
without editing source code in any of those projects.

> [!IMPORTANT]
> Public end-to-end availability is pending an upcoming vLLM PR containing the
> KVCR integration. Until that PR is available, this guide is a preview and the
> image build intentionally stops at the public-source placeholders. Once the
> PR is public, users can supply its repository and commit to try KVCR E2E.

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

The repository includes [Dockerfile.quick-start](../Dockerfile.quick-start),
which is prepared to assemble the required integration environment after the
public PR is available:

```bash
export KVCR_VLLM_REPO=PUBLIC_VLLM_PR_REPOSITORY_PENDING
export KVCR_VLLM_REF=PUBLIC_VLLM_PR_COMMIT_PENDING

DOCKER_BUILDKIT=1 docker build \
  --build-arg KVCR_VLLM_REPO="$KVCR_VLLM_REPO" \
  --build-arg KVCR_VLLM_REF="$KVCR_VLLM_REF" \
  --file Dockerfile.quick-start \
  --tag kvcr-quick-start:local \
  .
```

Do not run this command with the placeholder values. After the public vLLM PR
is available, replace both values with its public repository URL and immutable
commit. No private-repository credentials should be required.

The first build pulls the large vLLM runtime and compiles Dynamo's Rust
bindings, so it can take several minutes even on a fast host.

The build starts from the pinned vLLM nightly, fetches and applies the matching
six-file KVCR integration from the public PR, installs the local KVCR checkout,
and builds Dynamo at the revision that supports one KVCR control endpoint per
data-parallel rank. Once published, the public integration revision and base
image must be treated as one compatibility set.

The last build steps import all four components, verify `nixl==1.3.2`, preserve
the base image's validated NCCL 2.30.7, guard its NumPy and protobuf ABI
families, and confirm that Dynamo contains the required `control_ports`
router-hint support. A failure there means the image is not usable; do not
continue to the launch steps.

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

For multiple hosts, replace `127.0.0.1` with an address reachable from every
peer and bind or advertise the KV-events endpoint appropriately. NIXL and its
UCX transport must also be configured for the intended interconnect.

---

## 5. Verify with real requests

Do not treat a liveness response as proof that a backend is ready. Wait for a
real inference request through the Dynamo frontend to return HTTP 200 with a
non-empty `choices` array:

```bash
export MODEL=Qwen/Qwen3-0.6B

curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"Reply with the word ready.\",\"max_tokens\":8}"
```

Then send at least two prompts that share several full blocks. vLLM publishes
stored events for full blocks, so a very short shared prefix may never become
visible to the router:

```bash
PREFIX=$(python3 -c 'print(" ".join(f"section{i} cache routing data" for i in range(80)))')

for QUESTION in \
  'Summarize the design.' \
  'List two failure modes.'
do
  curl -sS http://127.0.0.1:8000/v1/completions \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"$PREFIX $QUESTION\",\"max_tokens\":32,\"temperature\":0}"
done
```

Confirm all of the following before calling the stack healthy:

1. Both DP ranks initialized a KVCR tier and distinct control endpoints.
2. Dynamo consumed self-describing KV events from the workers.
3. Repeated-prefix requests completed correctly through port 8000.
4. A delivered block was exposed to vLLM only after a terminal KVCR completion.

Normal deterministic KV routing prefers the worker that already owns the
prefix, so a healthy run may use local reuse without performing a peer
transfer. To exercise the peer-transfer mechanics, stop only the frontend with
Ctrl-C, restart it in the same terminal, and repeat the shared-prefix requests:

```bash
env -u NATS_SERVER python3 -m dynamo.frontend \
  --router-mode kv \
  --router-temperature 1 \
  --http-port 8000
```

This deliberately trades locality for cross-rank traffic and is for mechanism
validation, not a performance comparison. Wait for the periodic `KV Transfer
metrics` log. A successful transfer reports `transfer` with result `success`
on the source and `remote_deliver` with result `success` on the destination;
their block and byte counts must agree. Verify this telemetry rather than
relying only on the router's overlap score.

---

## Troubleshooting

### The image does not build

- Confirm that the host can pull the pinned `vllm/vllm-openai` image and reach
  the public KVCR-vLLM source, Dynamo's GitHub repository, and PyPI.
- Confirm that the public vLLM PR is available and that `KVCR_VLLM_REPO` and
  `KVCR_VLLM_REF` identify its public repository and immutable commit.
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
