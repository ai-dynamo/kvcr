#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Python provides portable deadlines and signal/finally cleanup without GNU timeout.
exec python3 - "$@" <<'PY'
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import uuid


class Failure(Exception):
    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


def command(arguments, *, timeout=60, **kwargs):
    return subprocess.run(arguments, timeout=timeout, check=True, **kwargs)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


parser = argparse.ArgumentParser(
    prog="run_vllm_e2e.sh",
    description="Run the built KVCR image on exactly two explicitly selected GPUs. "
                "Requires a matching committed checkout, Docker, and NVIDIA Container Toolkit.",
    epilog="Example: bash .github/scripts/run_vllm_e2e.sh --gpus 0,1. "
           "For a registry image: --image registry/repo@sha256:... "
           "--revisions source-revisions.json --pull. No ports are published.",
)
parser.add_argument("--gpus", required=True, help="Two distinct indices or full GPU UUIDs, comma separated")
parser.add_argument("--build-output", type=Path, default=Path("artifacts/build"))
parser.add_argument("--image", help="Explicit image; requires --revisions")
parser.add_argument("--revisions", type=Path, help="Expected source-revisions.json for --image")
parser.add_argument("--pull", action="store_true", help="Pull a digest-pinned image before inspecting it")
parser.add_argument("--output-dir", type=Path, help="New or empty evidence directory; default is unique")
parser.add_argument("--timeout", type=int, default=1200, help="Harness deadline in seconds (1–7200)")
parser.add_argument("--cpus", type=float, help="Docker CPU limit within the allocation")
parser.add_argument("--cpuset-cpus", help="Allowed host CPU list, for example 0-15,32-47")
parser.add_argument("--memory", help="Docker host-memory limit, for example 64g")
args = parser.parse_args()
devices = [part.strip() for part in args.gpus.split(",")]
gpu_uuid = r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
if len(devices) != 2 or any(not re.fullmatch(r"[0-9]+|" + gpu_uuid, item) for item in devices):
    parser.error("--gpus requires exactly two numeric indices or full GPU UUIDs; 'all' is not allowed")
devices = [str(int(item)) if item.isdecimal() else item.lower().replace("gpu-", "GPU-", 1)
           for item in devices]
if len(set(devices)) != 2:
    parser.error("--gpus requires two distinct devices")
if devices[0].isdecimal() != devices[1].isdecimal():
    parser.error("--gpus must use either two indices or two full GPU UUIDs")
if bool(args.image) != bool(args.revisions):
    parser.error("--image and --revisions must be supplied together")
if not 1 <= args.timeout <= 7200:
    parser.error("--timeout must be between 1 and 7200 seconds")
if args.cpus is not None and (not math.isfinite(args.cpus) or args.cpus <= 0):
    parser.error("--cpus must be a finite positive number")
if args.cpuset_cpus and not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", args.cpuset_cpus):
    parser.error("--cpuset-cpus must be a comma-separated list of CPU indices or ranges")
if args.memory and not re.fullmatch(r"[1-9][0-9]*(?:[kmg]b?)?", args.memory, re.I):
    parser.error("--memory must be a positive Docker memory size, for example 64g")

run_id = uuid.uuid4().hex
name = "kvcr-e2e-" + run_id
label = "io.ai-dynamo.kvcr.run-id"
output = None
container = None
create_attempted = False
result = {"run_id": run_id, "status": "failed", "gpus": devices,
          "started_at": datetime.now(timezone.utc).isoformat()}
exit_code = 1


def interrupted(number, _frame):
    raise Failure(f"Interrupted by signal {number}", 128 + number)


for number in (signal.SIGINT, signal.SIGTERM):
    signal.signal(number, interrupted)

try:
    repo = Path(command(["git", "rev-parse", "--show-toplevel"],
                        capture_output=True, text=True).stdout.strip()).resolve()
    os.chdir(repo)
    head = command(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    harness_relative = "tests/e2e/run_vllm.py"
    harness = repo / harness_relative
    committed = command(["git", "show", f"HEAD:{harness_relative}"], capture_output=True).stdout
    if harness.read_bytes() != committed:
        raise Failure("Harness differs from the checked-out commit; commit it before running")
    manifest_path = args.revisions or args.build_output / "source-revisions.json"
    manifest = read_json(manifest_path)
    for key in ("kvcr_commit", "dynamo_commit", "vllm_adapter_commit"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get(key, ""))):
            raise Failure(f"Expected a full source SHA: {key}")
    if manifest["kvcr_commit"] != head:
        raise Failure("Checkout HEAD does not match image KVCR source revision")
    if not re.fullmatch(r"\S+@sha256:[0-9a-f]{64}", str(manifest.get("vllm_base_image", ""))):
        raise Failure("Expected a digest-pinned vLLM base image in source revisions")
    expected = {
        "org.opencontainers.image.revision": manifest["kvcr_commit"],
        "io.ai-dynamo.kvcr.dynamo.sha": manifest["dynamo_commit"],
        "io.ai-dynamo.kvcr.vllm.sha": manifest["vllm_adapter_commit"],
        "io.ai-dynamo.kvcr.base-image": manifest["vllm_base_image"],
    }
    identity = None
    if args.image:
        image = args.image
    else:
        identity = read_json(args.build_output / "image-identity.json")
        if identity.get("verified_labels") != expected or type(identity.get("published")) is not bool:
            raise Failure("Build image identity does not match source revisions")
        image = identity.get("image_uri", "")
    if not isinstance(image, str) or not image or image.startswith("-") or re.search(r"\s", image):
        raise Failure("Invalid image reference")
    digest = image.rsplit("@", 1)[-1] if "@" in image else None
    if digest and not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise Failure("Invalid image digest")
    if identity and identity["published"] and identity.get("digest") != digest:
        raise Failure("Published build identity must select its recorded image digest")
    if args.pull and not digest:
        raise Failure("--pull requires an image pinned by digest")
    candidate_output = (args.output_dir or Path("artifacts/e2e") / ("local-" + run_id)).resolve()
    if candidate_output.exists() and (not candidate_output.is_dir() or any(candidate_output.iterdir())):
        raise Failure("Output directory must be new or empty")
    if any(character in str(path) for path in (harness, candidate_output) for character in (",", "\n", "\r")):
        raise Failure("Docker bind-mount paths cannot contain commas or newlines")
    candidate_output.mkdir(parents=True, exist_ok=True)
    output = candidate_output
    save_json(output / "source-revisions.json", manifest)
    if args.pull:
        command(["docker", "pull", "--platform", "linux/amd64", image], timeout=600)
    inspected = json.loads(command(["docker", "image", "inspect", image],
                                  capture_output=True, text=True).stdout)
    if len(inspected) != 1:
        raise Failure("Expected exactly one local image")
    inspected = inspected[0]
    image_id = inspected.get("Id", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise Failure("Docker returned an invalid image ID")
    if inspected.get("Architecture") != "amd64" or inspected.get("Os") != "linux":
        raise Failure("Image must target linux/amd64")
    labels = inspected.get("Config", {}).get("Labels") or {}
    if any(labels.get(key) != value for key, value in expected.items()):
        raise Failure("Actual image provenance labels do not match source revisions")
    if digest and not any(item.endswith("@" + digest) for item in inspected.get("RepoDigests") or []):
        raise Failure("Actual image lacks the requested repository digest")
    if identity and not identity["published"]:
        metadata = read_json(args.build_output / "build-metadata.json")
        if metadata.get("containerimage.config.digest") != image_id:
            raise Failure("Actual local image ID differs from the recorded build configuration digest")
    result.update(image_uri=image, image_id=image_id, kvcr_commit=head)
    save_json(output / "image-identity.json", {"image_uri": image, "image_id": image_id,
                                              "verified_labels": expected})
    limits = []
    if args.cpus is not None:
        limits += ["--cpus", str(args.cpus)]
    if args.cpuset_cpus:
        limits += ["--cpuset-cpus", args.cpuset_cpus]
    if args.memory:
        limits += ["--memory", args.memory, "--memory-swap", args.memory]
    create = [
        "docker", "create", "--name", name, "--label", f"{label}={run_id}",
        *limits,
        "--platform", "linux/amd64", "--pull", "never", "--network", "bridge",
        "--gpus", '"device=' + ",".join(devices) + '"', "--shm-size", "16g",
        "--ulimit", "memlock=-1", "--user", "root", "--workdir", "/tmp",
        "--mount", f"type=bind,src={harness},dst=/kvcr-tests/run_vllm.py,readonly",
        "--mount", f"type=bind,src={output},dst=/evidence",
        "--env", "HF_HOME=/tmp/kvcr-hf-cache", "--entrypoint", "python3", image_id,
        "/kvcr-tests/run_vllm.py", "--output-dir", "/evidence/run",
        "--timeout", str(args.timeout), "--check-output-consistency",
    ]
    create_attempted = True
    container = command(create, capture_output=True, text=True).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{64}", container):
        container = None
        raise Failure("Docker create returned no valid container ID")
    result["container_id"] = container
    baked_path = output / "image-source-revisions.json"
    command(["docker", "cp", container + ":/opt/kvcr-build/source-revisions.json", str(baked_path)])
    baked = read_json(baked_path)
    for key in ("kvcr_commit", "dynamo_commit", "vllm_adapter_commit", "vllm_base_image"):
        if baked.get(key) != manifest[key]:
            raise Failure(f"Baked image provenance mismatch: {key}")
    with (output / "container.log").open("w") as log:
        completed = subprocess.run(["docker", "start", "--attach", container],
                                   stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout + 90)
    exit_code = completed.returncode if completed.returncode >= 0 else 128 - completed.returncode
    if exit_code:
        raise Failure(f"Container test failed with exit code {exit_code}; see container.log", exit_code)
    evidence = read_json(output / "run/result.json")
    if evidence.get("status") != "passed" or evidence.get("peer_transfer_verified") is not True:
        raise Failure("Harness did not record successful peer transfer evidence")
    result["status"] = "passed"
    exit_code = 0
except Failure as error:
    result["error"] = str(error)
    exit_code = error.code
except subprocess.TimeoutExpired:
    result["error"] = "Docker operation exceeded its deadline"
    exit_code = 124
except Exception as error:
    result["error"] = f"{type(error).__name__}: {error}"
    exit_code = 1
finally:
    # Signal traps ignore further interruptions while removing only our labelled container.
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, signal.SIG_IGN)
    if create_attempted:
        try:
            checked = subprocess.run(["docker", "inspect", container or name],
                                     capture_output=True, text=True, timeout=30)
            if checked.returncode:
                raise Failure("Could not confirm owned container state for cleanup")
            owned = json.loads(checked.stdout)[0]
            if owned.get("Config", {}).get("Labels", {}).get(label) != run_id:
                raise Failure("Container ownership label mismatch; refused cleanup")
            command(["docker", "rm", "--force", owned["Id"]], timeout=45,
                    capture_output=True, text=True)
            result["container_removed"] = True
        except Exception as error:
            result.update(status="failed", cleanup_error=str(error))
            exit_code = exit_code or 1
    result.update(exit_code=exit_code, finished_at=datetime.now(timezone.utc).isoformat())
    if output is not None and output.is_dir():
        save_json(output / "runner-result.json", result)
    print(json.dumps({**result, "output_dir": str(output) if output else None}, indent=2))
sys.exit(exit_code)
PY
