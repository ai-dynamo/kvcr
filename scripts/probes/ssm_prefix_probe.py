import json
import random
import re
import subprocess
import sys
import time
import urllib.request

MODEL = "Qwen/Qwen3.5-4B"
PORT = 8011
BLOCK = 576
srv = subprocess.Popen(
    [
        "vllm",
        "serve",
        MODEL,
        "--port",
        str(PORT),
        "--max-model-len",
        "9216",
        "--mamba-cache-mode",
        "align",
        "--enable-prefix-caching",
        "--block-size",
        "64",
        "--gpu-memory-utilization",
        "0.8",
        "--max-num-seqs",
        "8",
    ],
    stdout=open("/cache/doxu-kvcr/evidence/ssm-prefix-probe-vllm.log", "w"),
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
    sys.exit(1)
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
random.seed(7)
words = open("/usr/share/dict/words").read().split() if False else None
vocab = [
    w
    for w in (
        "alpha beta gamma delta river stone cloud paper metal green quick slow north south table chair light dark seven nine".split()
    )
]
text = " ".join(random.choice(vocab) for _ in range(16000))
ids = tok(text)["input_ids"][: 14 * BLOCK]


def ask(n, max_tokens=1, tag=""):
    body = json.dumps(
        {"model": MODEL, "prompt": ids[:n], "max_tokens": max_tokens, "temperature": 0}
    ).encode()
    r = json.load(
        urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/completions",
                body,
                {"Content-Type": "application/json"},
            ),
            timeout=600,
        )
    )
    u = r["usage"]
    c = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    h = hits()
    print(
        f"{tag:<42} prompt={u['prompt_tokens']:5d} usage.cached={c:5d} metrics.hits={h - ask.last:5d}",
        flush=True,
    )
    ask.last = h
    return c


def hits():
    m = (
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10)
        .read()
        .decode()
    )
    return int(
        float(
            re.search(r"^vllm:prefix_cache_hits_total\{[^}]*\} (\S+)", m, re.M).group(1)
        )
    )


ask.last = 0
ask.last = hits()
X = 100
ask(6 * BLOCK + X, 1, "R1 long 6b+100, cold")
ask(4 * BLOCK + X, 1, "R2 4b+100, strict prefix of R1")
ask(5 * BLOCK + X, 1, "R3 5b+100, strict prefix of R1")
ask(4 * BLOCK + X, 1, "R4 = R2 again")
ask(6 * BLOCK + X, 1, "R5 = R1 again")
ask(3 * BLOCK + X, 1, "R6 3b+100, prefix of R2")
ask(12 * BLOCK + X, 1, "R7 long 12b+100, cold tail")
ask(8 * BLOCK + X, 1, "R8 8b+100, prefix of R7 (chunk bdry?)")
ask(10 * BLOCK + X, 1, "R9 10b+100, prefix of R7")
ask(12 * BLOCK + X, 1, "R10 = R7 again")
srv.terminate()
