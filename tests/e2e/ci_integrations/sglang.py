# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SGLang policy shared by the HiCache and direct-linker integrations.

KVCR runs inside each SGLang worker: there is no Guard service, no sidecar and
no supported disaggregated layout, so every profile is aggregated native
(``--mode nonguard``) and runs TC-01..TC-03. Each integration module binds the
integration name, catalog and image profile; this module owns launch, resource
and evidence semantics. Nothing here is copied from the vLLM byte geometry:
SGLang telemetry is normalized by ``phase1.telemetry`` and byte checks use the
measured KVCR page size (``page_bytes`` stays 0).
"""

from __future__ import annotations

import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path

from ci_cleanup import validate_phase1_cleanup
from phase1.cases import TELEMETRY_LABELS, required_proof_checks
from phase1.common import validate_gpu_budget

from .vllm_evidence import REQUIRED_PROOFS, _observed_overlap, _options, _read

E2E = Path(__file__).resolve().parents[1]
FAMILIES = ("dense", "mamba", "gdn")
TOPOLOGY = "agg"
MODE = "native"
CASES = ["TC-01", "TC-02", "TC-03"]
MODELS = {
    "Qwen/Qwen3-0.6B",
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    "Qwen/Qwen3.5-0.8B",
}
BOOLEANS = {"batch_invariant", "tc02_seed_prefix"}
LAUNCH_FIELDS = BOOLEANS | {
    "max_model_len",
    "max_tokens",
    "pool_gb",
    "hit_block_tokens",
    "tc02_shared_sections",
    "tc02_sections",
    "timeout",
    "request_timeout",
    "worker_args",
}
# Engine flags a catalog may add. Anything that sets configuration identity or
# the reference oracle (model, page size, KVCR flags) stays harness-owned.
WORKER_ARG = re.compile(
    r"--(?:attention-backend=(?:fa3|flashinfer|triton)|trust-remote-code)"
)
# Harness defaults (phase1.stack.Settings) for one aggregated SGLang stack:
# HTTP, KVCR control/NIXL, KV events, Dynamo system, state agent, SGLang.
PORTS = [
    8000,
    8001,
    17771,
    17772,
    18771,
    18772,
    20081,
    20082,
    9091,
    9092,
    9094,
    24010,
    24011,
]


def profile_id(integration: str, family: str) -> str:
    return f"{integration}-{family}-{TOPOLOGY}-{MODE}"


def validate_catalog(raw: dict, integration: str) -> dict:
    if set(raw["models"]) != set(FAMILIES):
        raise ValueError("catalog must contain the three SGLang model families")
    for model in raw["models"].values():
        launch = model["launch"]
        if set(launch) != LAUNCH_FIELDS:
            raise ValueError("unknown or missing launch fields")
        for key in BOOLEANS:
            if type(launch[key]) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key in LAUNCH_FIELDS - BOOLEANS - {"worker_args"}:
            if type(launch[key]) is not int or launch[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if launch["max_model_len"] > 8192:
            # The harness caps SGLang context and token budget at 8192.
            raise ValueError("SGLang max_model_len must be at most 8192")
        if not isinstance(launch["worker_args"], list) or any(
            not isinstance(arg, str) or not WORKER_ARG.fullmatch(arg)
            for arg in launch["worker_args"]
        ):
            raise ValueError(
                "worker arguments cannot override configuration identity or oracle"
            )
        if len(_options(launch["worker_args"])) != len(launch["worker_args"]):
            raise ValueError("duplicate worker argument")
    seen = set()
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
        family = profile["model_family"]
        if (
            profile["integration"] != integration
            or family not in FAMILIES
            or profile["topology"] != TOPOLOGY
            or profile["mode"] != MODE
        ):
            raise ValueError("unsupported integration or configuration")
        if profile["id"] != profile_id(integration, family) or family in seen:
            raise ValueError("duplicate or inconsistent profile identity")
        seen.add(family)
        resources = profile["resources"]
        if set(resources) != {"gpus", "gpu_memory_utilization"}:
            raise ValueError("unknown or missing resource fields")
        gpus = resources["gpus"]
        if (
            not isinstance(gpus, list)
            or len(gpus) != 2
            or any(type(gpu) is not int or gpu < 0 for gpu in gpus)
        ):
            raise ValueError("invalid GPU layout")
        if type(resources["gpu_memory_utilization"]) not in (float, int):
            raise ValueError("invalid per-worker GPU memory budget")
        validate_gpu_budget(gpus, resources["gpu_memory_utilization"])
        if profile["qualification"] not in ("previous-ci-pass", "pending"):
            raise ValueError("invalid historical qualification label")
    if seen != set(FAMILIES):
        raise ValueError("catalog must cover every SGLang model family")
    return raw


def required_profile_ids(integration: str) -> set[str]:
    return {profile_id(integration, family) for family in FAMILIES}


def resolve_profile(item: dict, model: dict) -> dict:
    profile = copy.deepcopy(item)
    profile.update(
        model_key=model["model_key"],
        model_id=model["id"],
        model_revision=model["revision"],
        launch=copy.deepcopy(model["launch"]),
        cases=list(CASES),
    )
    return profile


def environment(profile: dict) -> dict[str, str]:
    return {"DYN_ROUTER_WORKER_SELECTION_POLICY": "kvcr"}


def validate_model_snapshot(profile: dict, path: Path, inventory: dict) -> dict:
    """SGLang loads the tokenizer from the snapshot; require its JSON files."""
    if profile["model_id"] not in MODELS:
        raise ValueError(
            "SGLang tokenizer prerequisites are unqualified for this model"
        )
    files = []
    for name in ("tokenizer.json", "tokenizer_config.json"):
        source = path / name
        try:
            if not isinstance(json.loads(source.read_text()), dict):
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
        "validation": "config-json-required-weights-and-sglang-tokenizer-json",
    }


def build_command(
    profile: dict,
    output: Path,
    snapshot: Path,
    prompt_tag: str,
    gpus: list[int] | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    launch = profile["launch"]
    command = [
        sys.executable,
        "-u",
        str(E2E / "run_phase1.py"),
        "--output-dir",
        str(output / "run"),
        "--model",
        profile["model_id"],
        "--model-path",
        str(snapshot),
        "--engine",
        "sglang",
        *(extra or []),
        "--mode",
        "nonguard",
        "--topology",
        TOPOLOGY,
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
        "pool_gb",
        "hit_block_tokens",
        "tc02_shared_sections",
        "tc02_sections",
        "timeout",
        "request_timeout",
    ):
        command += ["--" + key.replace("_", "-"), str(launch[key])]
    for key in ("batch_invariant", "tc02_seed_prefix"):
        if launch[key]:
            command.append("--" + key.replace("_", "-"))
    command += [f"--worker-arg={argument}" for argument in launch["worker_args"]]
    return command


def runtime_resources(profile: dict) -> dict:
    return {"tcp_ports": list(PORTS)}


def _proofs(native: dict, errors: list[str]) -> None:
    """Recompute TC-01..TC-03 proofs from the executed checks, as for vLLM."""
    for case in native.get("cases", []):
        if not isinstance(case, dict):
            continue  # The common harness-shape validator reports this.
        name = case.get("case")
        if name not in CASES:
            errors.append(f"No SGLang evidence policy for {name}")
            continue
        checks = [check for check in case.get("checks", []) if isinstance(check, dict)]
        names = Counter(check.get("name") for check in checks)

        def passed(label: str) -> bool:
            return names[label] == 1 and any(
                check.get("name") == label and check.get("passed") is True
                for check in checks
            )

        proofs = {
            proof: all(passed(label) for label in labels)
            for proof, labels in required_proof_checks(name).items()
        }
        proofs["telemetry_health"] = all(
            passed(f"{label}: required KVCR telemetry is complete and healthy")
            for label in TELEMETRY_LABELS[name]
        )
        measurements = case.get("measurements", {})
        if not isinstance(measurements, dict):
            raise ValueError("case measurements are malformed")
        if name == "TC-02":
            capability = measurements.get("engine_partial_prefix_reuse", {})
            counts = (
                capability.get("reference_cached_tokens")
                if isinstance(capability, dict)
                else None
            )
            proofs["partial_reuse"] = (
                proofs["partial_reuse"]
                and capability.get("reuses") is True
                and isinstance(counts, list)
                and bool(counts)
                and type(counts[0]) is int
                and counts[0] > 0
            )
        if name == "TC-03":
            proofs["request_overlap"] = passed(
                "tc03: observed request intervals are valid"
            ) and _observed_overlap(measurements.get("request_overlap", {}))
        recorded = case.get("proofs")
        for proof in sorted(REQUIRED_PROOFS[name]):
            if not proofs.get(proof):
                errors.append(f"{name}: required {proof} proof is missing or failed")
            if not isinstance(recorded, dict) or recorded.get(proof) is not True:
                errors.append(f"{name}: recorded {proof} verdict is missing or failed")


def _worker(
    directory: Path,
    name: str,
    profile: dict,
    envelope: dict,
    kvcr_flags: dict[str, str | bool],
    errors: list[str],
) -> None:
    launch = _read(directory / "run/stack" / f"{name}.launch.json")
    command = launch.get("command")
    if not (
        isinstance(command, list)
        and len(command) > 2
        and command[1:3] == ["-m", "dynamo.sglang"]
    ):
        errors.append(f"{name}: actual worker engine is not SGLang")
        return
    flags = _options(command)
    context = str(min(profile["launch"]["max_model_len"], 8192))
    expected: dict[str, str | bool] = {
        "--model-path": envelope["model_path"],
        "--served-model-name": profile["model_id"],
        "--tp-size": "1",
        "--page-size": "64",
        "--context-length": context,
        "--mem-fraction-static": str(profile["resources"]["gpu_memory_utilization"]),
        "--disable-cuda-graph": True,
    }
    if profile["launch"]["batch_invariant"]:
        expected["--enable-deterministic-inference"] = True
    for flag, value in {**expected, **kvcr_flags}.items():
        if flags.get(flag) != value:
            errors.append(f"{name}: actual launch differs: {flag}")
    if not kvcr_flags and any(
        flag.startswith(("--hicache-storage", "--unified-cache-external-linker"))
        for flag in flags
    ):
        errors.append(f"{name}: KVCR-disabled reference enables KVCR")
    for flag, value in _options(profile["launch"]["worker_args"]).items():
        if flags.get(flag) != value:
            errors.append(f"{name}: configured worker argument differs: {flag}")


def validate_result(
    profile: dict,
    envelope: dict,
    directory: Path,
    errors: list[str],
    *,
    kvcr_flags: dict[str, str | bool],
    native_fields: dict | None = None,
    arguments: dict | None = None,
) -> None:
    native = _read(directory / "run/result.json")
    validate_phase1_cleanup(_read(directory / "run/stack/cleanup.json"))
    launch = profile["launch"]
    expected = {
        "engine": "sglang",
        "model": profile["model_id"],
        "model_path": envelope.get("model_path"),
        "mode": "nonguard",
        "topology": TOPOLOGY,
        "selected_cases": profile["cases"],
        "state_agent": True,
        "batch_invariant": launch["batch_invariant"],
        "max_tokens": launch["max_tokens"],
        **(native_fields or {}),
    }
    for field, value in expected.items():
        if native.get(field) != value:
            errors.append(f"Native harness identity differs: {field}")
    recorded = _read(directory / "run/provenance.json").get("arguments", {})
    expected_arguments = {
        "engine": "sglang",
        "model": profile["model_id"],
        "model_path": envelope.get("model_path"),
        "mode": "nonguard",
        "topology": TOPOLOGY,
        "cases": profile["cases"],
        "gpus": profile["resources"]["gpus"],
        "gpu_memory_utilization": profile["resources"]["gpu_memory_utilization"],
        "allow_blocked": False,
        "no_state_agent": False,
        "worker_arg": launch["worker_args"],
        **{key: value for key, value in launch.items() if key != "worker_args"},
        **(arguments or {}),
    }
    for field, value in expected_arguments.items():
        if recorded.get(field) != value:
            errors.append(f"Native provenance argument differs: {field}")
    _proofs(native, errors)
    for index in (0, 1):
        _worker(directory, f"worker-{index}", profile, envelope, kvcr_flags, errors)
    _worker(directory, "worker-0-reference", profile, envelope, {}, errors)
    for name in ("kvcr-service-0", "kvcr-service-1"):
        if (directory / "run/stack" / f"{name}.launch.json").is_file():
            errors.append(f"SGLang native run launched a Guard service: {name}")
    if not (directory / "run/stack/kv-state-agent.launch.json").is_file():
        errors.append("SGLang run did not launch the KV state agent")
    frontend = _read(directory / "run/stack/frontend.launch.json")
    if (
        frontend.get("environment_overrides", {}).get(
            "DYN_ROUTER_WORKER_SELECTION_POLICY"
        )
        != "kvcr"
    ):
        errors.append("Actual frontend did not select the KVCR worker-selection policy")
