<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVCR Quick Start with SGLang HiCache

Build and run Dynamo, SGLang and KVCR without editing their serving code. SGLang stages KV through HiCache host memory; KVCR stores it in a separate DRAM pool. A Dynamo state agent indexes KVCR-owned blocks so the router can generate source hints for peer delivery.

> [!IMPORTANT]
> This integration is a preview built from pinned source revisions, including the open SGLang PR below.

For vLLM, use the [vLLM quick start](quick-start-vllm.md). For implementation and API development, use the [developer guide](dev-guide.md).

## Prerequisites

- A Linux x86-64 host with two available NVIDIA GPUs compatible with the pinned SGLang/CUDA 13 image and enough free GPU memory for this example.
- NVIDIA driver 580.126.20 or newer for the pinned CUDA 13 runtime.
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), Git, and public access to Docker Hub, GitHub, PyPI, `pypi.nvidia.com`, `flashinfer.ai`, crates.io and Hugging Face. These public indexes need no login.
- Allow 32 GiB host RAM and 150 GiB disk for this build and model cache.

This example uses two independent TP1 workers on one host, Qwen3-0.6B BF16, `trtllm_mha`, 64-token pages, and the Python radix tree. Each worker has 2 GiB of HiCache staging and a separate 1 GiB KVCR DRAM pool. The same-host transport is UCX TCP/shared memory with CUDA-copy support, not an RDMA performance test.

## 1. Build the integration image

Clone the repository and build the SGLang image from [Dockerfile.quick-start-sglang](../Dockerfile.quick-start-sglang):

```bash
set -euo pipefail
git clone https://github.com/ai-dynamo/kvcr.git kvcr-sgl-quick-start
cd kvcr-sgl-quick-start
git rev-parse HEAD
export QS_TAG="kvcr-sgl-quick-start:$(date -u +%Y%m%dT%H%M%SZ)-$$"
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
  --file Dockerfile.quick-start-sglang --tag "$QS_TAG" .
export QS_IMAGE_ID="$(docker image inspect "$QS_TAG" --format '{{.Id}}')"
```

The build uses pinned [SGLang #45647](https://github.com/sgl-project/sglang/pull/45647) and Dynamo revisions from the Dockerfile, and installs KVCR from the local checkout used as the build context (`.`), including any local changes. Continue only if the final import/version checks pass.

## 2. Start the container

Select two available GPUs; this example uses `0,1`. No host ports are published.

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv
export QS_NAME="kvcr-sgl-$(date -u +%Y%m%dT%H%M%SZ)-$$"
docker volume create "$QS_NAME-models"
export QS_CID="$(docker run --detach --init --name "$QS_NAME" \
  --gpus '"device=0,1"' --shm-size 8g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --env MODEL=Qwen/Qwen3-0.6B --env MODEL_PATH=/model \
  --volume "$QS_NAME-models:/model" \
  "$QS_IMAGE_ID" sleep infinity)"
docker exec "$QS_CID" python3 -c \
  'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca", local_dir="/model")'
printf 'Container: %s\n' "$QS_CID"
```

Open separate terminals with `docker exec -it <container-id> bash`, using the printed ID: one for the state agent, one for the router, one per worker, and one for requests. All inherit the same private discovery directory and TCP/ZMQ settings. The public model needs no HF token. Commands in Steps 3-6 run inside the container.

## 3. Start the KV state agent

```bash
set -o pipefail
env -u NATS_SERVER DYN_NAMESPACE=kvcrqs DYN_SYSTEM_PORT=24004 \
  python3 -m dynamo.kv_state_agent --max-slots 4 \
  2>&1 | tee /tmp/state-agent.log
```

Keep this process running.

## 4. Start the Dynamo KV router

```bash
set -o pipefail
env -u NATS_SERVER DYN_SYSTEM_PORT=24003 python3 -m dynamo.frontend \
  --http-host 127.0.0.1 --http-port 24000 --namespace kvcrqs \
  --router-mode kv --router-policy-config /opt/sglang-router-policy.yaml \
  2>&1 | tee /tmp/frontend.log
```

The KV router supplies source hints; a round-robin router does not demonstrate router-guided peer reuse.

## 5. Start two HiCache workers

Run this block in **each** worker terminal, setting `RANK=0` in the first and `RANK=1` in the second. Keep both running throughout verification.

```bash
set -euo pipefail
export RANK=0                         # Use 1 in the second terminal.
export PROMETHEUS_MULTIPROC_DIR="/tmp/sgl-metrics-$RANK"
mkdir "$PROMETHEUS_MULTIPROC_DIR"
KVCR_CONFIG="$(python3 - <<'PY'
import ipaddress, json, os, socket
rank = int(os.environ['RANK'])
host = os.environ.get('KVCR_ADVERTISE_HOST') or socket.gethostbyname(socket.gethostname())
assert not ipaddress.ip_address(host).is_loopback, 'Set KVCR_ADVERTISE_HOST to a reachable container address'
print(json.dumps({
    'control_advertise_host': host,
    'control_port': 24100 + rank * 32, 'nixl_listen_port': 24200 + rank * 32,
    'local_dram_bytes': 1 << 30,
    'publish_inventory': True,
    'enable_remote_hint': True, 'hicache_storage_pass_prefix_keys': True,
    'align_storage_prefix': True, 'combine_page_reads': True,
    'combine_page_writes': True, 'diagnostic_telemetry': True,
    'verify_transfer_bytes': True,
    'startup_timeout_seconds': 300,
    'operation_timeout_ms': 20000, 'abandon_timeout_ms': 60000,
}))
PY
)"
env -u NATS_SERVER CUDA_VISIBLE_DEVICES="$RANK" DYN_SYSTEM_PORT="$((24001 + RANK))" \
  python3 -m dynamo.sglang --endpoint dyn://kvcrqs.backend.generate \
  --model-path "$MODEL_PATH" --served-model-name "$MODEL" --tp-size 1 \
  --host 127.0.0.1 --port "$((24010 + RANK))" \
  --dtype bfloat16 --attention-backend trtllm_mha \
  --context-length 8192 --max-total-tokens 8192 --chunked-prefill-size 8192 \
  --max-running-requests 1 --page-size 64 --mem-fraction-static 0.5 \
  --disable-cuda-graph --disable-piecewise-cuda-graph \
  --incremental-streaming-output --enable-metrics \
  --enable-hierarchical-cache --hicache-size 2 \
  --hicache-mem-layout page_first --hicache-write-policy write_through \
  --hicache-storage-backend kvcr --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config "$KVCR_CONFIG" \
  --kv-events-config "{\"publisher\":\"zmq\",\"format\":\"dynamo\",\"topic\":\"\",\"endpoint\":\"tcp://*:$((24020 + RANK))\"}" \
  2>&1 | tee "/tmp/worker-$RANK.log"
```

Control, NIXL, metrics and event ports must be distinct. The ownership descriptor requires a reachable non-loopback control address, even in this same-container example. Both terminals derive the container IP automatically; set the same `KVCR_ADVERTISE_HOST` in both if hostname resolution is unsuitable. KV events bind `*` inside that private network namespace.

## 6. Verify peer reuse

Wait for successful LLM registration in both worker logs and `Adding worker` for both IDs in the frontend log. Allow up to 20 minutes per worker for first-use compilation. Then, in the request terminal, require `ready: true`:

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

This check sends a fresh 2048-token prefix to worker 0, then requests the identical prefix on worker 1. Do not probe this prefix before the test. The first request can compile additional attention kernels after readiness.

```bash
python3 - <<'PY'
import json, os, re, time, urllib.request, uuid
from pathlib import Path
from transformers import AutoTokenizer

workers = []
for rank in (0, 1):
    log = Path(f'/tmp/worker-{rank}.log').read_text()
    log = re.sub(r'\x1b\[[0-9;]*m', '', log)
    ids = set(re.findall(r'Backend admission gate created[^\n]*worker_id=([0-9a-f]+)', log))
    assert len(ids) == 1 and 'Successfully registered LLM with runtime config' in log
    workers.append(int(ids.pop(), 16))
assert workers[0] != workers[1]
frontend = Path('/tmp/frontend.log').read_text()
assert all(f'Adding worker WorkerWithDpRank {{ worker_id: {w}, dp_rank: 0 }}' in frontend for w in workers)
tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'])
prompt = f'Document {uuid.uuid4().hex}. ' + 'The capital of France is Paris. ' * 600
tokens = tokenizer.encode(prompt, add_special_tokens=False)[:2048]
assert len(tokens) == 2048
outputs = []
for rank, worker in enumerate(workers):
    if rank:
        time.sleep(15)  # Deposit, KV events and router indexing are asynchronous.
    payload = dict(model=os.environ['MODEL'], prompt=tokens, temperature=0,
                   max_tokens=16, ignore_eos=True, stream=True,
                   stream_options={'include_usage': True},
                   nvext={'backend_instance_id': worker, 'dp_rank': 0,
                          'extra_fields': ['worker_id', 'engine_data', 'completion_token_ids']})
    request = urllib.request.Request('http://127.0.0.1:24000/v1/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = response.read().decode()
    Path(f'/tmp/response-{rank}.sse').write_text(raw)
    events = [json.loads(line[5:]) for line in raw.splitlines()
              if line.startswith('data:') and line[5:].strip() != '[DONE]']
    assert not any('error' in event for event in events)
    ext = [event.get('nvext', {}) for event in events]
    ids = [token for item in ext for token in item.get('completion_token_ids', [])]
    reported = [item['worker_id'] for item in ext if 'worker_id' in item]
    assert reported and all(x['prefill_worker_id'] == x['decode_worker_id'] == worker for x in reported)
    usage = [event['usage'] for event in events if event.get('usage')][-1]
    cache = [item.get('engine_data', {}).get('kv_cache_hit', {}) for item in ext]
    hits = [item['reused_tokens'] for item in cache if 'reused_tokens' in item]
    assert hits and max(hits) == (1984 if rank else 0)
    assert usage['prompt_tokens'] == 2048 and usage['completion_tokens'] == len(ids) == 16
    assert sum(c.get('finish_reason') is not None for e in events for c in e.get('choices', [])) == 1
    outputs.append(ids)
    print(f'worker {rank}: cached_tokens={max(hits)}, output_tokens={len(ids)}')
assert outputs[0] == outputs[1], 'Cold/restored output token IDs differ'
print('Response checks passed; verify terminal transfer logs below before declaring success.')
PY
```

Also inspect `KVCR L3 native stats` and `KVCR L3 operation` in both worker logs:

- Destination: successful peer `deliver` from worker 0's advertised address on port **24100**, **31 blocks**, **62 spans**, and no failed keys; native `remote_deliver` reports success.
- Source: terminal peer-send (`source_write`) totals **31 blocks / 227,540,992 bytes**. Do not confuse this with initial local deposition (`local_fill`), which may include 32 prompt pages; the destination recomputes the final page for logits.
- No failed transfers or dropped observations. Sum each relevant drained reporting interval once; do not subtract two snapshots.
- Destination metrics: **1984** completed HiCache prefetched tokens.

Allow the five-second native telemetry reporting interval to drain, then read those logs and the destination's isolated metrics in the request terminal:

```bash
sleep 6
grep -E 'KVCR L3 (native stats|operation)' /tmp/worker-{0,1}.log
python3 - <<'PY'
from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.multiprocess import MultiProcessCollector
registry = CollectorRegistry()
MultiProcessCollector(registry, path='/tmp/sgl-metrics-1')
rows = generate_latest(registry).decode().splitlines()
total = sum(float(row.rsplit(None, 1)[1]) for row in rows
            if row.split('{', 1)[0] == 'sglang:prefetched_tokens_total')
assert total == 1984, total
print('Completed HiCache prefetched tokens:', total)
PY
```

A cache-hit counter alone is not proof of transfer. If any check fails, preserve the logs and investigate; retry with fresh workers and a new prefix, not a warm destination. This smoke test does not qualify general accuracy, performance, cross-host RDMA, TP>1 or hybrid pools.

## Cleanup and troubleshooting

On the host, collect the session's evidence, then stop only its container:

```bash
export QS_EVIDENCE="$(mktemp -d /tmp/kvcr-sgl-evidence.XXXXXX)"
for log in state-agent frontend worker-0 worker-1; do
  docker cp "$QS_CID:/tmp/$log.log" "$QS_EVIDENCE/"
done
for rank in 0 1; do
  docker cp "$QS_CID:/tmp/response-$rank.sse" "$QS_EVIDENCE/"
done
docker cp "$QS_CID:/opt/sglang-installed-packages.txt" "$QS_EVIDENCE/"
docker stop --time 90 "$QS_CID"
docker rm "$QS_CID"
```

Keep the model volume and any underlying GPU allocation unless you deliberately choose to release them. For a failed run, collect whichever evidence exists.

- **Build/import failure:** use the pinned source and kernel versions together and check the final import/version output.
- **Startup failure:** check worker logs, GPU architecture, driver, memory and unique ports. Model readiness is not a transfer pass.
- **Cache miss:** check source deposit, KV events, router discovery and hints.
- **Transfer failure:** retain `UCX_TLS=tcp,sm,self,cuda_copy`; missing CUDA memory support can prevent NIXL startup even for host-staged data.
