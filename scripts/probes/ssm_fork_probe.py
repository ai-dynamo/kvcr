# ruff: noqa: E402, E501  -- throwaway probe, imports after the server is up
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
import random as _r

_r.seed(11)


def tail(n, seed):
    rr = _r.Random(seed)
    return [rr.choice(ids[:4096]) for _ in range(n)]


P = ids[: 4 * BLOCK + 300]  # shared prefix 2604 tok, junction floor = 2304


def ask_ids(seq, tag, max_tokens=1):
    body = json.dumps(
        {"model": MODEL, "prompt": seq, "max_tokens": max_tokens, "temperature": 0}
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
    h = hits()
    print(
        f"{tag:<46} prompt={r['usage']['prompt_tokens']:5d} metrics.hits={h - ask.last:5d}",
        flush=True,
    )
    ask.last = h


ask_ids(P + tail(1500, 1), "S1 P+X (prefix 2604 + 1500), cold")
ask_ids(P + tail(1500, 2), "S2 P+Y first fork (expect 0)")
ask_ids(P + tail(1500, 3), "S3 P+Z second fork (junction ckpt?)")
ask_ids(P + tail(1500, 4), "S4 P+W third fork")
ask_ids(P + tail(1500, 2), "S5 = S2 again (own end ckpt)")
Q = ids[4 * BLOCK + 300 : 4 * BLOCK + 300 + 7 * BLOCK + 100]  # fresh prefix 4132 tok
ask_ids(Q + tail(900, 5), "S6 Q+A cold (prefix 4132 => floor 4032)")
ask_ids(Q[: 3 * BLOCK + 50] + tail(900, 6), "S7 Q[:1778]+B fork inside Q (expect 0)")
ask_ids(
    Q[: 3 * BLOCK + 50] + tail(900, 7), "S8 Q[:1778]+C second fork (junction 1728?)"
)
ask_ids(Q + tail(900, 8), "S9 Q+D fork at 4132 (hit 4032 from S6 end?)")
srv.terminate()
