# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualified vLLM policy and launch semantics; shared code owns supervision."""

from __future__ import annotations

import copy
import itertools
import json
import re
import sys
from pathlib import Path

from phase1.common import validate_gpu_budget

E2E = Path(__file__).resolve().parents[1]
CATALOG = E2E / "ci_profiles.json"
SMOKE_ID = "vllm-dense-peer-transfer-smoke"
FAMILIES = ("dense", "mamba", "gdn")
TOPOLOGIES = ("agg", "disagg")
MODES = ("guard", "native")
IMAGE_PROFILE = "vllm"

LAUNCH_FIELDS = {
    "max_model_len",
    "max_tokens",
    "cpu_bytes_to_use",
    "batch_invariant",
    "keep_hybrid_kv_cache_manager",
    "hit_block_tokens",
    "ssm_groups",
    "page_bytes",
    "tc02_shared_sections",
    "tc02_sections",
    "tc02_seed_prefix",
    "g2_slots",
    "timeout",
    "request_timeout",
    "worker_args",
}


def validate_catalog(raw: dict) -> dict:
    if set(raw["models"]) != set(FAMILIES):
        raise ValueError("catalog must contain the three vLLM model families")
    for model in raw["models"].values():
        launch = model["launch"]
        if set(launch) != LAUNCH_FIELDS:
            raise ValueError("unknown or missing launch fields")
        booleans = {
            "batch_invariant",
            "keep_hybrid_kv_cache_manager",
            "tc02_seed_prefix",
        }
        for key in booleans:
            if type(launch[key]) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key in LAUNCH_FIELDS - booleans - {"worker_args"}:
            if type(launch[key]) is not int or launch[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if not 1 <= launch["g2_slots"] <= 65536 or launch["timeout"] < 1:
            raise ValueError("invalid native memory budget or timeout")
        if not isinstance(launch["worker_args"], list) or any(
            not isinstance(arg, str)
            or not re.fullmatch(
                r"--(?:dtype=(?:auto|bfloat16|float16|float32)|max-num-seqs=[1-9][0-9]*|trust-remote-code)",
                arg,
            )
            for arg in launch["worker_args"]
        ):
            raise ValueError(
                "worker arguments cannot override configuration identity or oracle"
            )
    seen_ids, seen_tuples = set(), set()
    for profile in raw["profiles"]:
        if set(profile) != {
            "id",
            "integration",
            "model_family",
            "topology",
            "mode",
            "resources",
            "qualification",
        }:
            raise ValueError("unknown or missing profile fields")
        family, topology, mode = (
            profile[key] for key in ("model_family", "topology", "mode")
        )
        identity = (family, topology, mode)
        if profile["integration"] != "vllm" or identity not in set(
            itertools.product(FAMILIES, TOPOLOGIES, MODES)
        ):
            raise ValueError("unsupported integration or configuration")
        expected_id = f"vllm-{family}-{topology}-{mode}"
        if (
            profile["id"] != expected_id
            or expected_id in seen_ids
            or identity in seen_tuples
        ):
            raise ValueError("duplicate or inconsistent profile identity")
        seen_ids.add(expected_id)
        seen_tuples.add(identity)
        resources = profile["resources"]
        if set(resources) != {"gpus", "gpu_memory_utilization"}:
            raise ValueError("unknown or missing resource fields")
        mapping = resources["gpus"]
        if (
            not isinstance(mapping, list)
            or len(mapping) != (2 if topology == "agg" else 3)
            or any(type(gpu) is not int or gpu < 0 for gpu in mapping)
        ):
            raise ValueError("invalid GPU layout")
        utilization = resources["gpu_memory_utilization"]
        if type(utilization) not in (float, int):
            raise ValueError("invalid per-worker GPU memory budget")
        validate_gpu_budget(mapping, utilization)
        if profile["qualification"] not in ("previous-ci-pass", "pending"):
            raise ValueError("invalid historical qualification label")
    if seen_tuples != set(itertools.product(FAMILIES, TOPOLOGIES, MODES)):
        raise ValueError("catalog must cover all twelve required configurations")
    return raw


def required_profile_ids(catalog: dict | None = None) -> set[str]:
    return {
        f"vllm-{family}-{topology}-{mode}"
        for family, topology, mode in itertools.product(FAMILIES, TOPOLOGIES, MODES)
    }


def resolve_profile(item: dict, model: dict) -> dict:
    profile = copy.deepcopy(item)
    profile.update(
        model_key=model["model_key"],
        model_id=model["id"],
        model_revision=model["revision"],
        launch=copy.deepcopy(model["launch"]),
    )
    profile["cases"] = ["TC-01", "TC-02", "TC-03"] + (
        ["TC-04"] if profile["mode"] == "guard" else []
    )
    return profile


def smoke_profiles(catalog: dict) -> list[dict]:
    # TC-03 owns the production KVCR peer assertions. Rank setup is checked
    # against the actual installed adapter in the image build canary.
    return []


def diagnostic_profiles(catalog: dict) -> list[dict]:
    """Retain the former DP2 manual diagnostic; it does not qualify CI."""
    model = catalog["models"]["dense"]
    return [
        {
            "id": SMOKE_ID,
            "integration": "vllm",
            "model_family": "dense",
            "model_key": model["model_key"],
            "model_id": model["id"],
            "model_revision": model["revision"],
            "topology": "peer-transfer",
            "mode": "native",
            "cases": [],
            "resources": {"gpus": [0, 1], "gpu_memory_utilization": 0.5},
            "launch": {"timeout": 1500},
            "qualification": "previous-ci-pass",
        }
    ]


def environment(profile: dict) -> dict[str, str]:
    values = {"DYN_ROUTER_WORKER_SELECTION_POLICY": "kvcr"}
    if profile["model_family"] != "dense":
        values["VLLM_BATCH_INVARIANT"] = "0"
    return values


def validate_model_snapshot(profile: dict, path: Path, inventory: dict) -> dict:
    """Validate tokenizer files present in the three qualified checkpoints.

    This is vLLM policy for the current checkpoint inventory, not a shared
    assumption about tokenizer formats required by future engine adapters.
    """
    if profile["model_id"] not in {
        "Qwen/Qwen3-0.6B",
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "Qwen/Qwen3.5-0.8B",
    }:
        raise ValueError("vLLM tokenizer prerequisites are unqualified for this model")
    files = []
    for name in ("tokenizer.json", "tokenizer_config.json"):
        source = path / name
        try:
            value = json.loads(source.read_text())
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            files.append({"name": name, "bytes": source.stat().st_size})
        except (OSError, ValueError) as error:
            raise ValueError(
                f"missing, unreadable or malformed tokenizer prerequisite: {name}"
            ) from error
    return {
        **inventory,
        "required_files": sorted(
            set(inventory["required_files"]) | {item["name"] for item in files}
        ),
        "tokenizer_files": files,
        "validation": "config-json-required-weights-and-vllm-tokenizer-json",
    }


def build_command(
    profile: dict,
    output: Path,
    snapshot: Path,
    prompt_tag: str,
    gpus: list[int] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(
            E2E / ("run_vllm_kvcr.py" if profile["id"] == SMOKE_ID else "run_phase1.py")
        ),
        "--output-dir",
        str(output / "run"),
        "--model",
        profile["model_id"],
        "--model-path",
        str(snapshot),
    ]
    if profile["id"] == SMOKE_ID:
        return command
    launch = profile["launch"]
    command += [
        "--engine",
        "vllm",
        "--mode",
        "nonguard" if profile["mode"] == "native" else "guard",
        "--topology",
        profile["topology"],
        "--cases",
        ",".join(profile["cases"]),
        "--gpus",
        ",".join(map(str, gpus if gpus is not None else profile["resources"]["gpus"])),
        "--gpu-memory-utilization",
        str(profile["resources"]["gpu_memory_utilization"]),
        "--frontend-env=DYN_ROUTER_WORKER_SELECTION_POLICY=kvcr",
        "--prompt-tag",
        prompt_tag,
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
    if profile["model_family"] != "dense":
        command.append("--stop-on-fail")
        if profile["topology"] == "disagg":
            # NIXL's hybrid conv-state transfer requires dimension-first layout.
            # Apply to reference and KVCR workers through the shared worker env.
            command.append("--worker-env=VLLM_SSM_CONV_STATE_LAYOUT=DS")
    for argument in [
        f"--revision={profile['model_revision']}",
        f"--code-revision={profile['model_revision']}",
        *launch["worker_args"],
    ]:
        command.append(f"--worker-arg={argument}")
    return command


def runtime_resources(profile: dict) -> dict:
    if profile["topology"] == "peer-transfer":
        return {"tcp_ports": [8000, 17771, 17772, 20081]}
    ports = [8000, 8001, 17771, 17772, 17773, 20081, 20082, 20083, 9091, 9092, 9093]
    if profile.get("mode") == "guard":
        ports.append(9094)
    if profile["topology"] == "disagg":
        ports += [5601, 5602, 5603]
    return {"tcp_ports": ports}


def validate_result(
    profile: dict, envelope: dict, directory: Path, errors: list[str]
) -> None:
    """Keep vLLM engine/proof semantics outside the common verdict envelope."""
    from .vllm_evidence import validate_result as validate

    validate(profile, envelope, directory, errors)
