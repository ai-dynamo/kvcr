"""Bare vLLM + CPU offload + self-describing KV events: dump what the offload
tracker emits per KV cache group for a hybrid model. Arg: label."""

import json
import random
import subprocess
import sys
import threading
import time
import urllib.request

import msgspec
import zmq
from vllm.distributed.kv_events import BlockStored, KVEventBatch

label = sys.argv[1]
MODEL = "Qwen/Qwen3.5-4B"
PORT = 8012
EP = 5557
events = []


def sub():
    ctx = zmq.Context()
    s = ctx.socket(zmq.SUB)
    s.connect(f"tcp://127.0.0.1:{EP}")
    s.setsockopt(zmq.SUBSCRIBE, b"")
    dec = msgspec.msgpack.Decoder(KVEventBatch)
    while True:
        topic, seq, payload = s.recv_multipart()
        events.append(dec.decode(payload))


threading.Thread(target=sub, daemon=True).start()
kvt = {
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "spec_name": "CPUOffloadingSpec",
        "cpu_bytes_to_use": 8 << 30,
        "self_describing_kv_events": True,
    },
}
kve = {
    "publisher": "zmq",
    "endpoint": f"tcp://*:{EP}",
    "enable_kv_cache_events": True,
    "topic": "",
}
srv = subprocess.Popen(
    [
        "vllm",
        "serve",
        MODEL,
        "--port",
        str(PORT),
        "--max-model-len",
        "8192",
        "--block-size",
        "64",
        "--enable-prefix-caching",
        "--mamba-cache-mode",
        "align",
        "--gpu-memory-utilization",
        "0.8",
        "--max-num-seqs",
        "8",
        "--kv-transfer-config",
        json.dumps(kvt),
        "--kv-events-config",
        json.dumps(kve),
    ],
    stdout=open(f"/cache/doxu-kvcr/evidence/ssm-events-{label}.log", "w"),
    stderr=subprocess.STDOUT,
)
for _ in range(240):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
        break
    except Exception:
        time.sleep(5)
else:
    print("server not up")
    srv.terminate()
    sys.exit(1)
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
random.seed(7)
vocab = "alpha beta gamma delta river stone cloud paper metal green quick slow north south table chair light dark seven nine".split()
ids = tok(" ".join(random.choice(vocab) for _ in range(6000)))["input_ids"][
    : 6 * 576 + 100
]


def ask(seq):
    body = json.dumps(
        {"model": MODEL, "prompt": seq, "max_tokens": 4, "temperature": 0}
    ).encode()
    json.load(
        urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/completions",
                body,
                {"Content-Type": "application/json"},
            ),
            timeout=600,
        )
    )


ask(ids)
time.sleep(8)
# evict GPU so the CPU tier is the only copy, then re-request to trigger a CPU lookup hit
filler = tok(" ".join(random.choice(vocab) for _ in range(20000)))["input_ids"][:7800]
ask(filler)
time.sleep(3)
ask(ids)
time.sleep(8)
srv.terminate()
summary = {}
for b in events:
    for e in b.events:
        kind = type(e).__name__
        med = getattr(e, "medium", None)
        g = getattr(e, "group_idx", None)
        key = f"{kind} medium={med} group={g}"
        d = summary.setdefault(
            key,
            {
                "n": 0,
                "block_sizes": set(),
                "spec_kinds": set(),
                "hashes": 0,
                "tokens": 0,
            },
        )
        d["n"] += 1
        d["hashes"] += len(e.block_hashes)
        if isinstance(e, BlockStored):
            d["block_sizes"].add(e.block_size)
            d["spec_kinds"].add(getattr(e, "kv_cache_spec_kind", None))
            d["tokens"] += len(e.token_ids)
print(f"=== {label}: {len(events)} batches")
for k in sorted(summary):
    d = summary[k]
    print(
        f"{k:<42} n={d['n']:4d} hashes={d['hashes']:5d} tokens={d['tokens']:6d} block_sizes={sorted(d['block_sizes'])} kinds={sorted(x for x in d['spec_kinds'] if x)}"
    )
