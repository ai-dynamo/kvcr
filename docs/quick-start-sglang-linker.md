<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVCR Quick Start with the SGLang Direct Linker

Build and run Dynamo, SGLang and KVCR without editing their serving code. SGLang's external cache linker hands GPU KV pages to KVCR, which keeps them in DRAM and delivers them layer by layer into another worker's device pools using Dynamo's source hints. There is no HiCache host tier on this path.

For the HiCache storage route, use the [SGLang HiCache quick start](quick-start-sglang.md). For vLLM, use the [vLLM quick start](quick-start-vllm.md). For implementation and API development, use the [developer guide](dev-guide.md).

## Support status

- Dense models on TP1 workers (reference: Qwen3-32B): covered by this guide.
- Rank-replicated hybrid models (DeepSeek-V4-Flash, two TP4 workers over RDMA, no Guard): previously tested with a different Dynamo integration; not covered by the pinned draft in this guide.
- Not covered: KVCR Guard with the linker, remote hints for dense TP>1, CP/PP, and workers spanning several nodes.

## Prerequisites

- A Linux x86-64 host with two available NVIDIA GPUs (Hopper or newer, 80 GB or more each) compatible with the pinned SGLang/CUDA 13 image.
- NVIDIA driver 580.95.05 or newer for the pinned CUDA 13 runtime.
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), Git, and public access to Docker Hub, GitHub, PyPI, `pypi.nvidia.com`, `flashinfer.ai`, crates.io and Hugging Face. These public indexes need no login.
- Allow 64 GiB host RAM and 250 GiB disk for this build and model cache.

This example uses two independent TP1 workers on one host, Qwen3-32B-FP8 with BF16 KV, `flashinfer`, 64-token pages, and the Python radix tree. Each worker has a 4 GiB in-process KVCR DRAM pool. The same-host transport is UCX TCP/shared memory with CUDA-copy support, not an RDMA performance test.

## 1. Build the integration image

Until this PR merges, fetch its public recipe and build the image from [Dockerfile.quick-start-sglang-linker](../Dockerfile.quick-start-sglang-linker):

```bash
set -euo pipefail
git clone https://github.com/ai-dynamo/kvcr.git kvcr-sgl-linker-quick-start
cd kvcr-sgl-linker-quick-start
git fetch origin pull/129/head
git checkout --detach FETCH_HEAD
git rev-parse HEAD
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
  --file Dockerfile.quick-start-sglang-linker --tag kvcr-sgl-linker-quick-start:local .
export QS_IMAGE_ID="$(docker image inspect kvcr-sgl-linker-quick-start:local --format '{{.Id}}')"
```

Keep these revisions together:

| Component | Revision |
| --- | --- |
| SGLang direct linker [#45650](https://github.com/sgl-project/sglang/pull/45650) (includes [#45646](https://github.com/sgl-project/sglang/pull/45646) and [#45648](https://github.com/sgl-project/sglang/pull/45648)) | `2d010d70d080e9ea4a6891bb7219ee9dd833c731` |
| Dynamo linker inventory [#17655](https://github.com/ai-dynamo/dynamo/pull/17655) | `874861a7fdce464a4c28a880886897309172aacd` |
| KVCR `main` | `7186e7c20ba6e4d71ac3aff507fac9bfd83e8a07` |

These are draft public PR heads, not released packages. The linker and HiCache SGLang branches do not contain each other. Continue only if the final import/version checks pass. Packages are recorded in `/opt/sglang-installed-packages.txt`.

## 2. Start the container

Select only your allocated GPUs; this example uses `0,1`. Do not use occupied GPUs or stop another user's processes. No host ports are published.

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv
export QS_NAME="kvcr-sgl-linker-$(date -u +%Y%m%dT%H%M%SZ)-$$"
docker volume create "$QS_NAME-models"
export QS_CID="$(docker run --detach --init --name "$QS_NAME" \
  --gpus '"device=0,1"' --shm-size 8g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --env MODEL=Qwen/Qwen3-32B-FP8 --env MODEL_PATH=/model \
  --volume "$QS_NAME-models:/model" \
  "$QS_IMAGE_ID" sleep infinity)"
docker exec "$QS_CID" python3 -c \
  'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-32B-FP8", revision="aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df", local_dir="/model")'
printf 'Container: %s\n' "$QS_CID"
```

Open separate terminals with `docker exec -it <container-id> bash`: one for the state agent, one for the router, one per worker, and one for requests. Commands in Steps 3-5 run inside the container.

## 3. Start the Dynamo KV router

First start the KV state agent in its own terminal:

```bash
set -o pipefail
env -u NATS_SERVER DYN_NAMESPACE=kvcrqs DYN_SYSTEM_PORT=24004 \
  python3 -m dynamo.kv_state_agent 2>&1 | tee /tmp/state-agent.log
```

In another terminal, start the router:

```bash
set -o pipefail
env -u NATS_SERVER DYN_SYSTEM_PORT=24003 python3 -m dynamo.frontend \
  --http-host 127.0.0.1 --http-port 24000 --namespace kvcrqs \
  --router-mode kv --router-policy-config /opt/sglang-router-policy.yaml \
  2>&1 | tee /tmp/frontend.log
```

## 4. Start two linker workers

Run this block in **each** worker terminal, setting `RANK=0` in the first and `RANK=1` in the second. Keep both running throughout verification.

```bash
set -euo pipefail
export RANK=0                         # Use 1 in the second terminal.
export PROMETHEUS_MULTIPROC_DIR="/tmp/sgl-metrics-$RANK"
mkdir "$PROMETHEUS_MULTIPROC_DIR"
KVCR_CONFIG="$(python3 - <<'PY'
import ipaddress, json, os, socket
rank = int(os.environ['RANK'])
host = socket.gethostbyname(socket.gethostname())
assert not ipaddress.ip_address(host).is_loopback, host
print(json.dumps({
    'kvcr_service_socket_path': None, 'dram_size_gb': 4,
    'advertise_host': host, 'publish_inventory': True,
    'control_port': 24100 + rank * 32, 'nixl_port': 24200 + rank * 32,
    'enable_telemetry': True, 'startup_timeout_seconds': 300,
    'operation_timeout_ms': 20000, 'abandon_timeout_ms': 60000,
}))
PY
)"
env -u NATS_SERVER CUDA_VISIBLE_DEVICES="$RANK" DYN_SYSTEM_PORT="$((24001 + RANK))" \
  python3 -m dynamo.sglang --endpoint dyn://kvcrqs.backend.generate \
  --model-path "$MODEL_PATH" --served-model-name "$MODEL" --tp-size 1 \
  --host 127.0.0.1 --port "$((24010 + RANK))" \
  --kv-cache-dtype bfloat16 --attention-backend flashinfer \
  --context-length 8192 --max-total-tokens 8192 --chunked-prefill-size 8192 \
  --max-running-requests 1 --page-size 64 --mem-fraction-static 0.8 \
  --disable-cuda-graph --disable-piecewise-cuda-graph \
  --incremental-streaming-output --enable-metrics \
  --enable-unified-cache-external-linker \
  --unified-cache-external-linker-backend kvcr \
  --hicache-storage-backend-extra-config "$KVCR_CONFIG" \
  --kv-events-config "{\"publisher\":\"zmq\",\"format\":\"dynamo\",\"topic\":\"\",\"endpoint\":\"tcp://*:$((24020 + RANK))\"}" \
  2>&1 | tee "/tmp/worker-$RANK.log"
```

Linker options are strict; HiCache option names such as `local_dram_bytes`, `control_advertise_host` or `enable_remote_hint` fail startup. `dram_size_gb` is per TP rank. Each TP rank binds `control_port` and `nixl_port` plus its global rank, so keep port ranges apart. `publish_inventory` enables KVCR-owned event publication; `advertise_host` must be reachable and non-loopback. The KV event endpoint must bind a wildcard host (`tcp://*:PORT`); with `tcp://127.0.0.1:PORT` the publisher connects instead and the router gets no events.

## 5. Verify peer reuse

Wait for successful LLM registration in both worker logs and `Adding worker` for both IDs in the frontend log. Then require `ready: true`:

```bash
python3 - <<'PY'
import json, os, urllib.request
url = f"http://127.0.0.1:24000/v1/models/{os.environ['MODEL']}/ready"
with urllib.request.urlopen(url, timeout=15) as response:
    status = json.load(response)
assert status.get('ready') is True, status
print(status)
PY
```

This check sends a fresh prompt of about 4,300 tokens with a random code in the middle to worker 0, then the identical prompt to worker 1, then again to worker 0. It pins only the destination; Dynamo still supplies the source hint. Worker 1 must reuse every complete page before the last prompt token, every output must contain the code, and worker 1's output must match worker 0's warm replay, which uses the same cached prefix (a cold prefill can differ in later greedy tokens). The first request on each worker JIT-compiles kernels and can take several minutes. Do not probe this prompt before the test.

```bash
python3 - <<'PY'
import json, os, random, re, time, urllib.request, uuid
from pathlib import Path
from transformers import AutoTokenizer

workers = []
for rank in (0, 1):
    log = Path(f'/tmp/worker-{rank}.log').read_text()
    ids = set(re.findall(r'WorkerKvQuery endpoint starting for worker (\d+)', log))
    assert len(ids) == 1 and 'Successfully registered LLM with runtime config' in log
    workers.append(int(ids.pop()))
assert workers[0] != workers[1]
frontend = Path('/tmp/frontend.log').read_text()
assert all(f'Adding worker WorkerWithDpRank {{ worker_id: {w}, dp_rank: 0 }}' in frontend for w in workers)
tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'])
code = str(random.randrange(100000, 1000000))
filler = 'The capital of France is Paris. ' * 300
prompt = (f'Document {uuid.uuid4().hex}. {filler}The access code for vault 7 is {code}. '
          f'{filler}Question: what is the access code for vault 7? Answer: The access code for vault 7 is')
tokens = tokenizer.encode(prompt, add_special_tokens=False)
expected = (len(tokens) - 1) // 64 * 64
outputs = []
for rank, worker in enumerate(workers + workers[:1]):
    if rank:
        time.sleep(15)  # Deposit, KV events and router indexing are asynchronous.
    payload = dict(model=os.environ['MODEL'], prompt=tokens, temperature=0,
                   max_tokens=16, stream=True, stream_options={'include_usage': True},
                   nvext={'backend_instance_id': worker, 'dp_rank': 0,
                          'extra_fields': ['worker_id', 'engine_data', 'completion_token_ids']})
    request = urllib.request.Request('http://127.0.0.1:24000/v1/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=900) as response:
        raw = response.read().decode()
    Path(f'/tmp/response-{rank}.sse').write_text(raw)
    events = [json.loads(line[5:]) for line in raw.splitlines()
              if line.startswith('data:') and line[5:].strip() != '[DONE]']
    assert not any('error' in event for event in events)
    ext = [event.get('nvext', {}) for event in events]
    ids = [token for item in ext for token in item.get('completion_token_ids', [])]
    reported = [item['worker_id'] for item in ext if 'worker_id' in item]
    assert reported and all(x['prefill_worker_id'] == x['decode_worker_id'] == worker for x in reported)
    cache = [item.get('engine_data', {}).get('kv_cache_hit', {}) for item in ext]
    hits = [item['reused_tokens'] for item in cache if 'reused_tokens' in item]
    assert hits and max(hits) == (expected if rank else 0), (rank, hits, expected)
    text = tokenizer.decode(ids)
    assert code in text, (rank, code, text)
    outputs.append(ids)
    print(f'request {rank} (worker {rank % 2}): cached_tokens={max(hits)}/{len(tokens)}, output={text!r}')
assert outputs[1] == outputs[2], 'Restored output differs from the warm replay on worker 0'
print('Response checks passed; verify transfer telemetry below before declaring success.')
PY
```

Then confirm a physical peer transfer from the `KVCR telemetry` lines. Each line covers one interval; sum them once:

```bash
python3 - <<'PY'
import json, re
from collections import Counter
from pathlib import Path
for rank in (0, 1):
    totals = Counter()
    for line in Path(f'/tmp/worker-{rank}.log').read_text().splitlines():
        if (match := re.search(r'KVCR telemetry rank=\d+ interval_s=\S+ stats=(\{.*\})', line)):
            totals.update({k: v for k, v in json.loads(match.group(1)).items()
                           if k.endswith(('.counter', '.count'))})
    print(f'worker {rank}:', json.dumps(dict(sorted(totals.items())), indent=2))
PY
```

- Destination (worker 1): `kvcr_transfer_blocks[remote_deliver].counter` equals reused pages times layers (`cached_tokens / 64 * 64` for this 64-layer model, so it equals `cached_tokens`) and there is no `failed` entry.
- Source (worker 0): nonzero `kvcr_transfer_bytes[source_write].counter` and no `failed` entry.

A cache-hit counter alone is not proof of transfer. If any check fails, preserve the logs; retry with fresh workers and a new prompt, not a warm destination. This smoke test does not qualify accuracy, performance, cross-host RDMA, TP>1 or hybrid pools.

## Cross-host and TP>1

- Each worker must run on one node (`nnodes=1`, CP1, PP1); workers can be on different hosts. Set `advertise_host` to a reachable address, open the control/NIXL port ranges, and use shared discovery (for example `DYN_DISCOVERY_BACKEND=etcd`).
- For RDMA, give the container `/dev/infiniband`, `--cap-add IPC_LOCK` and `--ulimit memlock=-1`, and set `UCX_TLS=rc,ud,self,cuda_copy`.
- On hosts with several NICs per NUMA node, pin each TP rank to the NIC marked `PIX` for its GPU in `nvidia-smi topo -m` by setting `UCX_NET_DEVICES=<nic>:1` in that rank's process before KVCR starts. Otherwise the first peer connection can stall and time out.
- Dense models with TP>1 keep local reuse but disable remote hints; scale them with TP1 workers.
- The linker has a rank-replicated TP>1 mode for MLA-style models, but the pinned Dynamo inventory integration currently advertises TP1 only. The following is a historical linker configuration, not a supported router setup for these pins:

```json
{"kvcr_service_socket_path": null, "dram_size_gb": 140, "trailing_capacity_divisor": 16, "advertise_host": "<NODE_A_IP>", "control_port": 19500, "nixl_port": 20500, "replicated_source_rank_routing": true, "shard_replicated": true, "operation_timeout_ms": 20000, "abandon_timeout_ms": 60000, "startup_timeout_seconds": 600}
```

`replicated_source_rank_routing` makes target rank `r` fetch from source rank `r`; `--enable-linker-mla-dedup` makes each rank read only its own pages and all-gather each layer; `shard_replicated` stores one DRAM copy per TP group and requires the other two. The linker tunes Python GC itself (`gc_full_collection_threshold`); `--gc-threshold` does not reach the scheduler processes.

## Cleanup and troubleshooting

On the host, collect the session's evidence, then stop only its container:

```bash
export QS_EVIDENCE="$(mktemp -d /tmp/kvcr-sgl-linker-evidence.XXXXXX)"
for log in state-agent frontend worker-0 worker-1; do
  docker cp "$QS_CID:/tmp/$log.log" "$QS_EVIDENCE/"
done
for rank in 0 1 2; do
  docker cp "$QS_CID:/tmp/response-$rank.sse" "$QS_EVIDENCE/"
done
docker cp "$QS_CID:/opt/sglang-installed-packages.txt" "$QS_EVIDENCE/"
docker stop --time 90 "$QS_CID"
docker rm "$QS_CID"
```

The model volume is kept for reuse; remove it with `docker volume rm "$QS_NAME-models"` when done.

- **Build/import failure:** keep source and kernel pins together; do not skip the canary.
- **Startup failure:** check worker logs, GPU memory, unique ports and `startup_timeout_seconds`; on multi-NIC hosts, pin NICs per rank.
- **Cache miss:** check hint advertisement (`advertise_host`, `control_port`), wildcard KV event endpoints, and that the frontend added both workers. `KVCR remote hints disabled` in a worker log means its layout cannot serve peers.
- **Transfer failure:** failed loads are not recomputed and can stop the worker. Size `dram_size_gb` for the working set and keep generous timeouts. For same-host runs, retain `UCX_TLS=tcp,sm,self,cuda_copy`.
- **Weight or layout change:** restart the workers with matching configurations.
