#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run KVCR test plan v7 Phase 1 (TC-01..TC-04) in the built integration image
# on exactly two explicitly selected GPUs.
#
#   bash scripts/docker/run_phase1.sh --gpus 0,1 [--image <ref>] [--output-dir DIR]
#        [--mode guard|nonguard] [--cases TC-01,TC-02] [--timeout SEC]
#        [--cpus N] [--cpuset-cpus LIST] [--memory SIZE] [--hf-cache DIR]
#
# Without --image the wrapper uses artifacts/build/image-identity.json from
# build_image.sh. Guard mode needs a host kernel of 6.5 or newer; the harness
# reports BLOCKED otherwise. The container gets 16 GiB /dev/shm because each
# worker owns a 2 GiB Guard pool plus vLLM's host tier.
set -euo pipefail

exec python3 - "$@" <<'PY'
import argparse
import json
import re
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser(prog="run_phase1.sh")
parser.add_argument("--gpus", required=True,
                    help="one or two GPU indices or UUIDs, comma separated (one = both workers share it)")
parser.add_argument("--image", help="image reference; default from artifacts/build")
parser.add_argument("--build-output", type=Path, default=Path("artifacts/build"))
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--mode", choices=("guard", "nonguard"), default="guard")
parser.add_argument("--cases", default="TC-01,TC-02,TC-03,TC-04")
parser.add_argument("--timeout", type=int, default=2400)
parser.add_argument("--cpus")
parser.add_argument("--cpuset-cpus")
parser.add_argument("--memory")
parser.add_argument("--hf-cache", type=Path, help="host directory reused as HF_HOME")
parser.add_argument("--allow-blocked", action="store_true")
parser.add_argument("--harness-arg", action="append", default=[],
                    help="extra run_phase1.py argument, repeatable")
parser.add_argument("--docker-arg", action="append", default=[],
                    help="extra docker run argument (e.g. --network=host, -v host:ctr), repeatable")
parser.add_argument("--network", default="bridge", help="docker network mode")
args = parser.parse_args()

devices = [part.strip() for part in args.gpus.split(",")]
uuid_form = r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
if len(devices) not in (1, 2) or len(set(devices)) != len(devices) or any(
    not re.fullmatch(r"[0-9]+|" + uuid_form, item) for item in devices
):
    parser.error("--gpus requires one or two distinct indices or full GPU UUIDs")
# One device: both workers share it (harness index 0,0; caller lowers utilization).
harness_gpus = "0,1" if len(devices) == 2 else "0,0"
if not 60 <= args.timeout <= 7200:
    parser.error("--timeout must be between 60 and 7200 seconds")

repo = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True,
                           capture_output=True, text=True).stdout.strip())
tests = repo / "tests" / "e2e"
if args.image:
    image = args.image
else:
    identity = json.loads((args.build_output / "image-identity.json").read_text())
    image = identity.get("local_image_id") or identity["image_uri"]
run_id = uuid.uuid4().hex[:12]
output = (args.output_dir or Path("artifacts/phase1") / run_id).resolve()
if output.exists() and any(output.iterdir()):
    parser.error(f"output directory must be new or empty: {output}")
output.mkdir(parents=True, exist_ok=True)
for path in (tests, output):
    if any(c in str(path) for c in ",\n\r"):
        parser.error("bind-mount paths cannot contain commas or newlines")

name = f"kvcr-phase1-{run_id}"
limits = []
if args.cpus:
    limits += ["--cpus", args.cpus]
if args.cpuset_cpus:
    limits += ["--cpuset-cpus", args.cpuset_cpus]
if args.memory:
    limits += ["--memory", args.memory, "--memory-swap", args.memory]
mounts = [
    "--mount", f"type=bind,src={tests},dst=/kvcr-tests,readonly",
    "--mount", f"type=bind,src={output},dst=/evidence",
]
env = ["--env", "HF_HOME=/tmp/phase1-hf-cache"]
if args.hf_cache:
    args.hf_cache.mkdir(parents=True, exist_ok=True)
    mounts += ["--mount", f"type=bind,src={args.hf_cache.resolve()},dst=/tmp/phase1-hf-cache"]
harness = [
    "python3", "/kvcr-tests/run_phase1.py",
    "--output-dir", "/evidence/run",
    "--mode", args.mode,
    "--cases", args.cases,
    "--timeout", str(args.timeout),
    "--gpus", harness_gpus,
    *(["--allow-blocked"] if args.allow_blocked else []),
    *args.harness_arg,
]
command = [
    "docker", "run", "--rm", "--name", name,
    "--label", f"io.ai-dynamo.kvcr.phase1-run={run_id}",
    *limits,
    "--gpus", '"device=' + ",".join(devices) + '"',
    "--shm-size", "16g", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
    "--network", args.network, "--user", "root", "--workdir", "/tmp",
    *mounts, *env, *args.docker_arg,
    "--entrypoint", harness[0], image, *harness[1:],
]
record = {
    "run_id": run_id, "image": image, "gpus": devices, "mode": args.mode,
    "cases": args.cases, "started_at": datetime.now(timezone.utc).isoformat(),
    "command": " ".join(shlex.quote(part) for part in command),
}
(output / "runner-command.json").write_text(json.dumps(record, indent=2) + "\n")
print(record["command"], file=sys.stderr)
exit_code = 1
try:
    with (output / "container.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=args.timeout + 180)
    exit_code = completed.returncode
except subprocess.TimeoutExpired:
    subprocess.run(["docker", "rm", "--force", name], capture_output=True)
    record["error"] = "container exceeded its deadline"
    exit_code = 124
finally:
    record.update(exit_code=exit_code, finished_at=datetime.now(timezone.utc).isoformat())
    result = output / "run" / "result.json"
    if result.exists():
        summary = json.loads(result.read_text())
        record["status"] = summary.get("status")
        record["cases_status"] = {c["case"]: c["status"] for c in summary.get("cases", [])}
    (output / "runner-result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record.get(k) for k in ("status", "cases_status", "exit_code")}))
    print(f"evidence: {output}")
sys.exit(exit_code)
PY
