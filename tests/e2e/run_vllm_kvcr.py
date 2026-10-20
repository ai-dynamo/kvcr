# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the quick-start's two-rank KVCR peer-transfer check in a serving image."""

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BLOCK_SIZE = 64
METRIC = re.compile(
    r"kvcr_duration_seconds:\(['\"](transfer|remote_deliver)['\"],\s*"
    r"['\"](success|partial|failed)['\"]\)_count=([0-9.]+)"
)


def request_json(url, payload=None, headers=None):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def wait_until(check, timeout, processes, description):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for name, process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited with code {process.returncode}")
        result = check()
        if result:
            return result
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {description}")


def stop_process(process):
    # The vLLM parent starts engine children; stop the whole process group.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(args):
    import torch

    if torch.cuda.device_count() < 2:
        raise RuntimeError("This test requires two visible CUDA GPUs")

    output = args.output_dir
    transfer_config = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": 2_000_000_000,
            "self_describing_kv_events": True,
            "secondary_tiers": [
                {
                    "type": "kvcr",
                    "router_capabilities": ["router_hint"],
                    "control_host": "0.0.0.0",
                    "control_ports": [17771, 17772],
                    "control_advertise_host": "127.0.0.1",
                    "operation_timeout_ms": 5000,
                    "abandon_timeout_ms": 10000,
                    "enable_telemetry": True,
                }
            ],
        },
    }
    events_config = {
        "publisher": "zmq",
        "topic": "kv-events",
        "endpoint": "tcp://*:20081",
        "enable_kv_cache_events": True,
    }
    write_json(output / "kv-transfer-config.json", transfer_config)
    processes = []
    with contextlib.ExitStack() as resources:
        discovery = resources.enter_context(
            tempfile.TemporaryDirectory(prefix="kvcr-e2e-")
        )
        env = {
            **os.environ,
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "zmq",
            "DYN_DISCOVERY_BACKEND": "file",
            "DYN_FILE_KV": discovery,
            "PYTHONHASHSEED": "0",
            "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES": "1",
            "VLLM_LOGGING_LEVEL": "DEBUG",
            "CUDA_VISIBLE_DEVICES": "0,1",
        }
        env.pop("NATS_SERVER", None)

        def start(name, module, *arguments):
            log = resources.enter_context((output / f"{name}.log").open("wb"))
            process = subprocess.Popen(
                [sys.executable, "-m", module, *arguments],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((name, process))
            resources.callback(stop_process, process)
            print(f"Started {name} (PID {process.pid})", flush=True)

        start(
            "frontend",
            "dynamo.frontend",
            *f"--router-mode kv --http-port {args.port}".split(),
        )
        start(
            "worker",
            "dynamo.vllm",
            "--model",
            args.model,
            *f"""
                --tensor-parallel-size 1 --data-parallel-size 2
                --block-size {BLOCK_SIZE} --max-model-len 8192
                --enable-prefix-caching --disable-hybrid-kv-cache-manager
                --enforce-eager --gpu-memory-utilization 0.5
            """.split(),
            "--kv-transfer-config",
            json.dumps(transfer_config),
            "--kv-events-config",
            json.dumps(events_config),
        )
        base_url = f"http://127.0.0.1:{args.port}"

        def model_ready():
            try:
                models = request_json(base_url + "/v1/models")
                return any(item["id"] == args.model for item in models.get("data", []))
            except (urllib.error.URLError, TimeoutError):
                return False

        print("Waiting for model registration and engine startup", flush=True)
        wait_until(model_ready, args.startup_timeout, processes, "model registration")

        worker_id = None

        def complete(name, prompt, rank=None):
            headers = {}
            if rank is not None:
                headers = {
                    "x-dynamo-worker-instance-id": str(worker_id),
                    "x-dynamo-dp-rank": str(rank),
                }
            response = request_json(
                base_url + "/v1/completions",
                {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": 8,
                    "temperature": 0,
                    "nvext": {"extra_fields": ["worker_id"]},
                },
                headers,
            )
            write_json(output / f"{name}.json", response)
            if not response.get("choices"):
                raise RuntimeError(f"Completion has no choices: {response}")
            identity = response["nvext"]["worker_id"]
            actual_worker = identity["decode_worker_id"]
            actual_rank = identity["decode_dp_rank"]
            usage = response["usage"]
            cached = usage["prompt_tokens_details"]["cached_tokens"]
            print(
                f"{name}: worker={actual_worker} rank={actual_rank} "
                f"prompt_tokens={usage['prompt_tokens']} cached_tokens={cached}",
                flush=True,
            )
            if rank is not None and (actual_worker, actual_rank) != (worker_id, rank):
                raise RuntimeError(f"{name} reached the wrong worker/rank: {identity}")
            return actual_worker, cached

        worker_id, _ = complete("probe", f"Worker identity probe {uuid.uuid4().hex}")

        # A fresh prefix makes a target-local hit impossible. Never retry a
        # target request with this prefix: it could then hit recomputed KV.
        tag = uuid.uuid4().hex
        prefix = " ".join(f"{tag} section{i} cache routing data" for i in range(80))
        _, cached = complete("seed", prefix + " seed source", rank=0)
        if cached != 0:
            raise RuntimeError("Fresh seed unexpectedly reported cached tokens")

        print("Seeded rank 0; waiting for host offload and router indexing", flush=True)
        time.sleep(args.offload_wait)
        _, cached = complete("target", prefix + " retrieve on target", rank=1)
        if cached <= 0 or cached % BLOCK_SIZE:
            raise RuntimeError(
                f"Expected positive block-aligned cached tokens, got {cached}"
            )

        def transfer_succeeded():
            text = (output / "worker.log").read_text(errors="replace")
            metrics = {}
            for scope, result, count in METRIC.findall(text):
                key = f"{scope}/{result}"
                metrics[key] = metrics.get(key, 0) + float(count)
            if any(
                metrics.get(f"remote_deliver/{s}", 0) for s in ("partial", "failed")
            ):
                raise RuntimeError(f"Remote delivery failed or was partial: {metrics}")
            if metrics.get("transfer/success", 0) and metrics.get(
                "remote_deliver/success", 0
            ):
                return metrics
            return None

        metrics = wait_until(
            transfer_succeeded, 60, processes, "successful transfer telemetry"
        )
        write_json(
            output / "result.json",
            {"status": "success", "cached_tokens": cached, "metrics": metrics},
        )
        print(
            f"Source transfer: success (count={metrics['transfer/success']:g})\n"
            f"Remote delivery: success (count={metrics['remote_deliver/success']:g})",
            flush=True,
        )
    print("KVCR peer-transfer validation: PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--offload-wait", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=Path("e2e-artifacts"))
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if image := os.environ.get("KVCR_TEST_IMAGE"):
        print(f"Test image: {image}", flush=True)
    print(f"Test model: {args.model}", flush=True)
    try:
        run(args)
    except Exception as error:
        print(
            f"KVCR peer-transfer validation: FAILED ({error})",
            file=sys.stderr,
            flush=True,
        )
        write_json(
            args.output_dir / "result.json", {"status": "failed", "error": str(error)}
        )
        for log in ("frontend.log", "worker.log"):
            path = args.output_dir / log
            if path.exists():
                print(
                    f"\nLast lines of {log}:\n"
                    + "\n".join(path.read_text(errors="replace").splitlines()[-60:]),
                    file=sys.stderr,
                )
        raise


if __name__ == "__main__":
    main()
