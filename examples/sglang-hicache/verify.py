# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, same-host Dynamo -> SGLang HiCache -> KVCR acceptance check.

No cluster resources are created or removed. Only this invocation's child
process groups are stopped. A passing result requires real remote transfers,
not just readiness, a cache-hit counter, or matching text.
"""

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PAGE = 64
OUTPUT_TOKENS = 16
PINS = {
    "sglang": "142e34c04b9dc828553178c0b600dc9aa2cb9b42",
    "dynamo": "66ddd723d6024f8b7e2d3d8c8d0e0b31f60c92a2",
    "kvcr": "bb48f3843a292f514becd3efedf29429e272c0af",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def values(value, name):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == name:
                yield child
            yield from values(child, name)
    elif isinstance(value, list):
        for child in value:
            yield from values(child, name)


def grade_response(events, tokens, worker):
    """The pinned API emits completion-ID deltas, not cumulative arrays."""
    output, usage, echoes, workers, hits = [], [], [], set(), []
    terminal = 0
    for event in events:
        require("error" not in event, "Frontend returned an error")
        ext = event.get("nvext", {})
        ids = ext.get("completion_token_ids", [])
        require(
            isinstance(ids, list) and all(type(x) is int for x in ids),
            "Malformed completion token IDs",
        )
        output.extend(ids)
        terminal += sum(
            c.get("finish_reason") is not None for c in event.get("choices", [])
        )
        echoes.extend(values(ext, "prompt_token_ids"))
        workers.update(values(ext, "prefill_worker_id"))
        workers.update(values(ext, "decode_worker_id"))
        hits.extend(values(ext, "reused_tokens"))
        if event.get("usage"):
            usage.append(event["usage"])
    require(
        terminal == 1 and len(output) == OUTPUT_TOKENS,
        "Missing/duplicate completion or incorrect output length",
    )
    require(
        usage and usage[-1]["completion_tokens"] == len(output),
        "Completion usage differs from exact token count",
    )
    require(usage[-1]["prompt_tokens"] == len(tokens), "Prompt usage mismatch")
    require(echoes and all(x == tokens for x in echoes), "Prompt IDs changed")
    require(workers == {worker}, "Missing worker identity or wrong worker")
    if not hits:
        hits = [
            u["prompt_tokens_details"]["cached_tokens"]
            for u in usage
            if "cached_tokens" in u.get("prompt_tokens_details", {})
        ]
    require(
        hits and all(type(x) is int and 0 <= x <= len(tokens) for x in hits),
        "Missing or invalid cache accounting",
    )
    return {"output_ids": output, "cached_tokens": max(hits), "worker": worker}


def native_stats(text):
    """Sum drained intervals exactly once; reject lost observations."""
    totals = {"bytes": 0, "blocks": 0, "success": 0, "failed": 0}
    for raw in re.findall(r"KVCR L3 native stats (\{[^\n]+\})", text):
        for key, value in json.loads(raw)["metrics"].items():
            if key == "dropped_observations":
                require(value == 0, "Native telemetry dropped observations")
                continue
            kind, name, labels, field = json.loads(key)
            if kind == "counter" and labels == ["source_write"]:
                if name == "kvcr_transfer_bytes":
                    totals["bytes"] += value
                if name == "kvcr_transfer_blocks":
                    totals["blocks"] += value
            if kind == "histogram" and name == "kvcr_duration_seconds":
                if labels[0] in ("source_write", "remote_deliver", "transfer"):
                    if field == "count" and labels[-1] in ("success", "failed"):
                        totals[labels[-1]] += value
    return totals


def check_image(output):
    require(platform.machine() == "x86_64", "This recipe is AMD64 only")
    versions = {}
    for line in Path(__file__).with_name("constraints.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, expected = line.split("==")
        actual = metadata.version(name)
        require(actual.split("+", 1)[0] == expected, f"{name}: {actual} != {expected}")
        versions[name] = actual
    modules = {}
    for name in (
        "sglang",
        "kvcr",
        "dynamo.sglang",
        "dynamo.llm",
        "sglang.srt.mem_cache.storage.kvcr.kvcr_store",
        "dynamo.sglang.kv_hints",
    ):
        modules[name] = str(importlib.import_module(name).__file__)
    from dynamo.common.constants import KV_HINT_TRANSFER_CAPABILITY_KEY
    from sglang.srt.mem_cache.storage.kvcr.kvcr_store import KVCRStore

    require(KV_HINT_TRANSFER_CAPABILITY_KEY == "router_hint", "Wrong hint protocol")
    require(callable(KVCRStore.batch_get_v1), "Missing HiCache implementation")
    for name, sha in PINS.items():
        require(
            os.environ.get(f"KVCR_QUICK_START_{name.upper()}_REF") == sha,
            f"Missing/wrong build provenance: {name}",
        )
    import torch

    require(torch.version.cuda == "13.0", "Unexpected CUDA wheel family")
    record = {
        "passed": True,
        "kind": "CPU import/version canary, NOT GPU acceptance",
        "source_pins": PINS,
        "packages": versions,
        "module_paths": modules,
        "base_image": os.environ.get("KVCR_QUICK_START_BASE_IMAGE"),
        "cuda": torch.version.cuda,
        "python": sys.version,
    }
    save(output / "image-check.json", record)
    return record


class Session:
    def __init__(self, output, base_port, startup_timeout):
        self.output, self.port, self.timeout = output, base_port, startup_timeout
        self.children = {}
        self.stopped = []
        self.namespace = "kvcrqs" + uuid.uuid4().hex[:12]
        self.env = dict(os.environ)
        for key in ("NATS_SERVER", "DYN_DISABLE_FRONTEND_NVEXT"):
            self.env.pop(key, None)
        self.env.update(
            DYN_DISCOVERY_BACKEND="file",
            DYN_FILE_KV=str(output / "discovery"),
            DYN_REQUEST_PLANE="tcp",
            DYN_EVENT_PLANE="zmq",
            DYN_SYSTEM_HOST="127.0.0.1",
            DYN_RUNTIME_NUM_WORKER_THREADS="4",
            DYN_RUNTIME_MAX_BLOCKING_THREADS="8",
            TOKENIZERS_PARALLELISM="false",
            SGLANG_RUST_BUILD_MODE="never",
            SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND="python",
            OMP_NUM_THREADS="4",
            UCX_TLS="tcp,sm,self,cuda_copy",
        )
        (output / "discovery").mkdir()

    def spawn(self, name, command, env):
        require(name not in self.children, "Duplicate process name")
        with (self.output / f"{name}.log").open("xb") as log:
            child = subprocess.Popen(
                command,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        # Popen owns/reaps the leader; an unreaped PID cannot be reused.
        self.children[name] = child
        save(
            self.output / f"{name}-launch.json",
            {
                "pid": child.pid,
                "command": command,
                "env": {
                    k: env[k]
                    for k in self.env
                    if k
                    in (
                        "DYN_FILE_KV",
                        "DYN_REQUEST_PLANE",
                        "DYN_EVENT_PLANE",
                        "DYN_DISCOVERY_BACKEND",
                        "UCX_TLS",
                        "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND",
                    )
                },
                "gpu": env.get("CUDA_VISIBLE_DEVICES"),
            },
        )

    def alive(self):
        for name, child in self.children.items():
            require(child.poll() is None, f"{name} exited; inspect its log")

    def stop(self, name):
        child = self.children[name]
        if child.poll() is None:
            require(os.getpgid(child.pid) == child.pid, "Changed child process group")
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                require(os.getpgid(child.pid) == child.pid, "Changed process group")
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=15)
        self.stopped.append(
            {"name": name, "pid": child.pid, "returncode": child.returncode}
        )
        del self.children[name]

    def wait(self, predicate, timeout, description):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.alive()
            try:
                value = predicate()
                if value:
                    return value
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise RuntimeError(f"Timed out: {description}")

    def worker(self, name, rank, gpu, model_path, l3=True):
        metrics = self.output / f"{name}-metrics"
        metrics.mkdir()
        env = dict(
            self.env,
            CUDA_VISIBLE_DEVICES=gpu,
            DYN_SYSTEM_PORT=str(self.port + 1 + rank),
            PROMETHEUS_MULTIPROC_DIR=str(metrics),
        )
        command = [
            sys.executable,
            "-m",
            "dynamo.sglang",
            "--endpoint",
            f"dyn://{self.namespace}.backend.generate",
            "--model-path",
            model_path,
            "--served-model-name",
            MODEL,
            "--tp-size",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port + 10 + rank),
            "--dtype",
            "bfloat16",
            "--attention-backend",
            "trtllm_mha",
            "--context-length",
            "8192",
            "--max-total-tokens",
            "8192",
            "--max-running-requests",
            "1",
            "--page-size",
            str(PAGE),
            "--chunked-prefill-size",
            "8192",
            "--mem-fraction-static",
            "0.5",
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
            "--incremental-streaming-output",
            "--enable-metrics",
            "--enable-hierarchical-cache",
            "--hicache-size",
            "2",
            "--hicache-mem-layout",
            "page_first",
            "--hicache-write-policy",
            "write_through",
            "--kv-events-config",
            json.dumps(
                {
                    "publisher": "zmq",
                    "topic": "",
                    "endpoint": f"tcp://*:{self.port + 20 + rank}",
                }
            ),
        ]
        if l3:
            config = {
                "control_advertise_host": "127.0.0.1",
                "control_port": self.port + 100 + rank * 32,
                "nixl_listen_port": self.port + 200 + rank * 32,
                "local_dram_bytes": 1 << 30,
                "enable_remote_hint": True,
                "align_storage_prefix": True,
                "combine_page_reads": True,
                "combine_page_writes": True,
                "hicache_storage_pass_prefix_keys": True,
                "diagnostic_telemetry": True,
                "startup_timeout_seconds": 300,
                "operation_timeout_ms": 20000,
                "abandon_timeout_ms": 60000,
            }
            command += [
                "--hicache-storage-backend",
                "kvcr",
                "--hicache-storage-prefetch-policy",
                "wait_complete",
                "--hicache-storage-backend-extra-config",
                json.dumps(config),
            ]
        self.spawn(name, command, env)

        def ready():
            get_json(f"http://127.0.0.1:{self.port + 1 + rank}/health")
            text = (self.output / f"{name}.log").read_text()
            if "Successfully registered LLM with runtime config" not in text:
                return None
            found = set(
                re.findall(
                    r"WorkerKvQuery endpoint starting for worker (\d+)",
                    text,
                )
            )
            return int(next(iter(found))) if len(found) == 1 else None

        return self.wait(ready, self.timeout, f"{name} model readiness")

    def frontend(self):
        env = dict(
            self.env,
            DYN_SYSTEM_PORT=str(self.port + 3),
            DYN_REQUEST_TRACE="1",
            DYN_REQUEST_TRACE_OUTPUT_PATH=str(self.output / "trace"),
        )
        self.spawn(
            "frontend",
            [
                sys.executable,
                "-m",
                "dynamo.frontend",
                "--http-host",
                "127.0.0.1",
                "--http-port",
                str(self.port),
                "--namespace",
                self.namespace,
                "--router-mode",
                "kv",
                "--router-policy-config",
                str(Path(__file__).with_name("router-policy.yaml")),
            ],
            env,
        )
        self.wait(
            lambda: any(
                x["id"] == MODEL
                for x in get_json(f"http://127.0.0.1:{self.port}/v1/models").get(
                    "data", []
                )
            ),
            180,
            "frontend model discovery",
        )

    def request(self, worker, tokens, name):
        self.alive()
        # Worker /health can become ready before the frontend has consumed its
        # discovery record (particularly after the cold-control restart).
        # Wait for the pinned router's exact worker-membership acknowledgement;
        # do not issue/retry a generation that could warm the supposedly cold cache.
        self.wait(
            lambda: (
                f"Adding worker WorkerWithDpRank {{ worker_id: {worker}, dp_rank: 0 }}"
                in (self.output / "frontend.log").read_text()
            ),
            60,
            f"frontend discovery of worker {worker}",
        )
        readiness = get_json(f"http://127.0.0.1:{self.port}/v1/models/{MODEL}/ready")
        require(readiness.get("ready") is True, "Frontend model is not ready")
        save(self.output / f"{name}-readiness.json", readiness)
        payload = {
            "model": MODEL,
            "prompt": tokens,
            "temperature": 0,
            "max_tokens": OUTPUT_TOKENS,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
            "nvext": {
                "backend_instance_id": worker,
                "dp_rank": 0,
                "extra_fields": [
                    "worker_id",
                    "engine_data",
                    "timing",
                    "completion_token_ids",
                    "prompt_token_ids",
                ],
            },
        }
        save(self.output / f"{name}-request.json", payload)
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-request-id": name},
        )
        events = []
        with (self.output / f"{name}-response.sse").open("xb") as raw:
            try:
                response = urllib.request.urlopen(request, timeout=180)
            except urllib.error.HTTPError as error:
                save(
                    self.output / f"{name}-http-error.json",
                    {
                        "status": error.code,
                        "body": error.read().decode(errors="replace"),
                    },
                )
                raise
            with response:
                for line in response:
                    raw.write(line)
                    if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                        events.append(json.loads(line[5:]))
        result = grade_response(events, tokens, worker)
        save(self.output / f"{name}-response.json", {**result, "events": events})
        return result

    def metric(self, name, metric):
        from prometheus_client import CollectorRegistry, generate_latest
        from prometheus_client.multiprocess import MultiProcessCollector

        registry = CollectorRegistry()
        MultiProcessCollector(registry, path=str(self.output / f"{name}-metrics"))
        text = generate_latest(registry).decode()
        return sum(
            float(line.rsplit(None, 1)[1])
            for line in text.splitlines()
            if line.split("{", 1)[0] == metric
        )

    def transfer(self):
        return native_stats((self.output / "source.log").read_text())


def run(args, output):
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    require(platform.system() == "Linux", "GPU run requires Linux")
    rows = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    require(len(rows) == 2, "Expose exactly TWO GPUs to this container")
    gpus = [row.split(",")[0].strip() for row in rows]
    require(
        len(set(gpus)) == 2 and all("B200" in row for row in rows),
        "This pinned trtllm_mha recipe is initially scoped to two B200 GPUs",
    )
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    require(not active, "GPUs already have compute processes; do not preempt them")
    # Fail before startup on occupied ports. Binding races still fail in the child.
    for port in range(args.base_port, args.base_port + 234):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    model_path = snapshot_download(MODEL, revision=REVISION)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    marker = uuid.uuid4().hex
    text = f"Document {marker}. " + "The capital of France is Paris. " * 600
    tokens = tokenizer.encode(text, add_special_tokens=False)[:2048]
    negative = tokenizer.encode(
        "Different document: " + marker + ". " + text, add_special_tokens=False
    )[:2048]
    require(
        len(tokens) == len(negative) == 2048 and tokens[:PAGE] != negative[:PAGE],
        "Prompt construction failed",
    )
    expected = ((len(tokens) - 1) // PAGE) * PAGE
    # This model is dense MHA/GQA, BF16, TP1: K and V for every layer.
    expected_bytes = (
        expected
        * config.num_hidden_layers
        * config.num_key_value_heads
        * config.head_dim
        * 2
        * 2
    )
    save(
        output / "plan.json",
        {
            "model": MODEL,
            "revision": REVISION,
            "gpus": rows,
            "expected_restored_tokens": expected,
            "expected_native_bytes": expected_bytes,
            "source_pins": PINS,
            "topology": "same-host, two TP1 replicas, UCX tcp/sm/self/cuda_copy",
            "base_port": args.base_port,
            "input_ids": tokens,
            "negative_ids": negative,
        },
    )
    session = Session(output, args.base_port, args.startup_timeout)
    try:
        source = session.worker("source", 0, gpus[0], model_path)
        target = session.worker("target", 1, gpus[1], model_path)
        require(source != target, "Workers must be independent")
        session.frontend()
        cold = session.request(source, tokens, "cold-source")
        require(cold["cached_tokens"] == 0, "Source was not cold")
        session.wait(
            lambda: (
                session.metric("source", "sglang:backuped_tokens_total") >= expected
            ),
            120,
            "completed source deposit",
        )
        restored = session.request(target, tokens, "remote-restore")
        require(restored["cached_tokens"] == expected, "Remote cache count mismatch")
        session.wait(
            lambda: (
                session.metric("target", "sglang:prefetched_tokens_total") == expected
            ),
            30,
            "completed HiCache prefetch",
        )
        session.wait(
            lambda: session.transfer()["bytes"] >= expected_bytes,
            30,
            "terminal native telemetry",
        )
        native = session.transfer()
        require(
            native["bytes"] == expected_bytes
            and native["blocks"] == expected // PAGE
            and native["success"] > 0
            and native["failed"] == 0,
            f"Native source transfer mismatch: {native}",
        )
        session.wait(
            lambda: native_stats((output / "target.log").read_text())["success"] > 0,
            30,
            "terminal destination telemetry",
        )
        target_native = native_stats((output / "target.log").read_text())
        require(target_native["failed"] == 0, "Native destination transfer failed")
        operations = [
            json.loads(raw)
            for raw in re.findall(
                r"KVCR L3 operation (\{[^\n]+\})", (output / "target.log").read_text()
            )
        ]
        deliveries = [op for op in operations if op.get("direction") == "deliver"]
        require(
            deliveries
            and all(
                op["source"] == f"tcp://127.0.0.1:{args.base_port + 100}"
                and op["source"] != op["endpoint"]
                and not op["failed_keys"]
                for op in deliveries
            ),
            "Missing successful peer delivery",
        )
        require(
            sum(op["requested_blocks"] for op in deliveries) == expected // PAGE,
            "Peer-delivery block count mismatch",
        )
        hbm = session.request(target, tokens, "hbm-repeat")
        miss = session.request(target, negative, "negative-control")
        require(
            hbm["cached_tokens"] == expected and miss["cached_tokens"] == 0,
            "HBM/negative-control cache accounting mismatch",
        )
        require(
            session.metric("target", "sglang:prefetched_tokens_total") == expected,
            "Unexpected extra restore in control requests",
        )
        # Restart ONLY the owned destination with L3 absent, on the SAME GPU.
        session.stop("target")
        control = session.worker("control", 1, gpus[1], model_path, l3=False)
        require(
            control not in (source, target), "Control reused an old worker identity"
        )
        independent = session.request(control, tokens, "destination-cold-no-l3")
        require(independent["cached_tokens"] == 0, "Independent control was not cold")
        require(
            cold["output_ids"]
            == restored["output_ids"]
            == hbm["output_ids"]
            == independent["output_ids"],
            "Cold/restored output token IDs differ",
        )
        save(
            output / "acceptance.json",
            {
                "passed": True,
                "restored_tokens": expected,
                "native": native,
                "destination_native": target_native,
                "delivery_operations": deliveries,
                "exact_output_tokens": OUTPUT_TOKENS,
                "same_destination_cold_control": True,
                "source_worker": source,
                "destination_worker": target,
                "control_worker": control,
            },
        )
    finally:
        errors = []
        for name in reversed(list(session.children)):
            try:
                session.stop(name)
            except Exception as error:
                errors.append(f"{name}: {error}")
        save(output / "cleanup.json", {"stopped": session.stopped, "errors": errors})
        require(not errors, f"Cleanup incomplete: {errors}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New evidence directory"
    )
    parser.add_argument(
        "--check-image", action="store_true", help="CPU import/version gate only"
    )
    parser.add_argument("--base-port", type=int, default=24000)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    args = parser.parse_args()
    require(1024 <= args.base_port <= 65000, "Invalid base port")
    require(0 < args.startup_timeout <= 1800, "Startup limit must be 1..1800 seconds")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "passed": False,
        "gpu_acceptance": not args.check_image,
        "base_image": os.environ.get("KVCR_QUICK_START_BASE_IMAGE"),
        "source_pins": PINS,
    }

    def interrupted(signum, frame):
        raise RuntimeError(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        check_image(output)
        if not args.check_image:
            run(args, output)
        result["passed"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        save(output / "result.json", result)
        # Evidence files only: no model-cache traversal, secrets, or mutable mmap files.
        files = sorted(
            p
            for p in output.rglob("*")
            if p.is_file() and (p.suffix in (".json", ".log", ".sse", ".jsonl", ".gz"))
        )
        with (output / "SHA256SUMS").open("x") as manifest:
            for path in files:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                manifest.write(f"{digest}  {path.relative_to(output)}\n")
    print(json.dumps({**result, "evidence": str(output)}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
