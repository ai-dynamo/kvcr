# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render or run one configuration from the vLLM CI matrix."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

E2E = Path(__file__).resolve().parent


def load_profiles() -> dict[str, dict]:
    catalog = json.loads((E2E / "ci_profiles.json").read_text())
    return {
        f"vllm-{family}-{topology}-{mode}": {
            **model,
            "topology": topology,
            "mode": mode,
            "gpus": gpus,
        }
        for family, model in catalog["models"].items()
        for topology, gpus in catalog["topologies"].items()
        for mode in catalog["modes"]
    }


def build_command(profile: dict, output: Path, tag: str, gpus: str | None = None):
    launch = profile["launch"]
    cases = "TC-01,TC-02,TC-03" + (",TC-04" if profile["mode"] == "guard" else "")
    command = [
        sys.executable,
        "-u",
        str(E2E / "run_e2e.py"),
        "--output-dir",
        str(output / "run"),
        "--model",
        profile["id"],
        "--engine",
        "vllm",
        "--mode",
        "nonguard" if profile["mode"] == "native" else "guard",
        "--topology",
        profile["topology"],
        "--cases",
        cases,
        "--gpus",
        gpus or ",".join(map(str, profile["gpus"])),
        "--gpu-memory-utilization",
        str(profile["gpu_memory_utilization"][profile["topology"]]),
        "--frontend-env=DYN_ROUTER_WORKER_SELECTION_POLICY=kvcr",
        "--prompt-tag",
        tag,
    ]
    for key in (
        "max_model_len",
        "max_tokens",
        "cpu_bytes_to_use",
        "hit_block_tokens",
        "ssm_groups",
        "page_bytes",
        "tc02_shared_sections",
        "tc02_sections",
        "g2_slots",
        "timeout",
        "request_timeout",
    ):
        command += ["--" + key.replace("_", "-"), str(launch[key])]
    for key in ("batch_invariant", "keep_hybrid_kv_cache_manager", "tc02_seed_prefix"):
        if launch[key]:
            command.append("--" + key.replace("_", "-"))
    if launch["keep_hybrid_kv_cache_manager"]:
        command.append("--stop-on-fail")
        if profile["topology"] == "disagg":
            command.append("--worker-env=VLLM_SSM_CONV_STATE_LAYOUT=DS")
    for value in (
        f"--revision={profile['revision']}",
        f"--code-revision={profile['revision']}",
        *launch["worker_args"],
    ):
        command.append(f"--worker-arg={value}")
    return command


def main() -> int:
    profiles = load_profiles()
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--matrix", action="store_true")
    selection.add_argument("--profile", choices=profiles)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--gpus", help="local GPU override; CI uses the configured mapping"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.matrix:
        print(json.dumps({"include": [{"profile": name} for name in profiles]}))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required with --profile")
    profile = profiles[args.profile]
    tag = "_".join(
        (
            "ci",
            args.profile,
            os.environ.get("GITHUB_RUN_ID", "local"),
            os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
        )
    ).replace("-", "_")
    command = build_command(profile, args.output_dir.resolve(), tag, args.gpus)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # One prefetch before any reference/candidate worker starts downloading.
    # Use Hub's ordinary revision cache and blob locking, without a custom cache.
    with (args.output_dir / "prefetch.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-c",
                "from huggingface_hub import snapshot_download; import sys; "
                "print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2]))",
                profile["id"],
                profile["revision"],
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1200,
            check=True,
        )
    os.environ["VLLM_BATCH_INVARIANT"] = (
        "1" if profile["launch"]["batch_invariant"] else "0"
    )
    os.environ["UCX_TLS"] = "tcp,sm,self,cuda_copy"
    # Replace this process so cancellation reaches the existing harness directly.
    os.execv(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
