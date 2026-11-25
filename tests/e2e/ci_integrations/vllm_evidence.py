# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM launch identity and semantic proofs for the common production verdict.

Named assertions are recomputed from the native harness evidence. Uploaded
``proofs`` dictionaries are corroborating metadata, never the policy authority.
TC-01 warm GPU reuse is intentionally not called a local KVCR restore proof.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from ci_cleanup import validate_phase1_cleanup
from ci_identity import OVERLAY_COMMIT, OVERLAY_FILES
from phase1.cases import TELEMETRY_LABELS, required_proof_checks
from phase1.telemetry import ANSI, VLLM_EPOCH_MARKER, VLLM_LOGGING_PREFIX, parse_log

REQUIRED_PROOFS = {
    "TC-01": {"reference_exact", "local_commit", "warm_reuse", "telemetry_health"},
    "TC-02": {
        "reference_exact",
        "peer_transfer",
        "partial_reuse",
        "miss",
        "release",
        "telemetry_health",
    },
    "TC-03": {
        "reference_exact",
        "peer_transfer",
        "hint_isolation",
        "request_overlap",
        "telemetry_health",
    },
    "TC-04": {
        "reference_exact",
        "guard_promote",
        "guard_serve",
        "guard_reattach",
        "local_restore",
        "telemetry_health",
    },
}


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path.name}")
    return value


def _options(command: list[str]) -> dict[str, str | bool]:
    """Parse engine flags while rejecting last-option-wins identity overrides."""
    if not isinstance(command, list) or any(
        not isinstance(arg, str) for arg in command
    ):
        raise ValueError("malformed recorded command")
    values = {}
    index = 0
    while index < len(command):
        argument = command[index]
        if not argument.startswith("--"):
            index += 1
            continue
        key, equals, value = argument.partition("=")
        if not equals:
            value = True
            if index + 1 < len(command) and not command[index + 1].startswith("--"):
                index += 1
                value = command[index]
        if key in values:
            raise ValueError(f"duplicate engine flag: {key}")
        values[key] = value
        index += 1
    return values


def _module(command: list[str], module: str) -> bool:
    return (
        isinstance(command, list)
        and len(command) > 2
        and command[1:3] == ["-m", module]
    )


def _observed_overlap(record: dict) -> bool:
    """Use the recorded actual intervals, not a scheduling or proof flag."""
    if not isinstance(record, dict) or record.get("observed") is not True:
        return False
    try:
        requests = record["requests"]
        if len(requests) != 2:
            return False
        intervals = [
            (
                datetime.fromisoformat(item["start"].replace("Z", "+00:00")),
                datetime.fromisoformat(item["end"].replace("Z", "+00:00")),
            )
            for item in requests
        ]
        if any(
            start.tzinfo is None or end.tzinfo is None or end < start
            for start, end in intervals
        ):
            return False
        actual = (
            min(end for _, end in intervals) - max(start for start, _ in intervals)
        ).total_seconds()
        seconds = record["overlap_seconds"]
        return (
            isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and math.isfinite(seconds)
            and actual > 0
            and math.isclose(seconds, actual, abs_tol=0.000001)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _proofs(native: dict, errors: list[str], page_bytes: int = 0) -> None:
    for case in native.get("cases", []):
        if not isinstance(case, dict):
            continue  # The common harness-shape validator reports this.
        name = case.get("case")
        if name not in REQUIRED_PROOFS:
            errors.append(f"No vLLM evidence policy for {name}")
            continue
        checks = case.get("checks", [])
        names = Counter(
            check.get("name") for check in checks if isinstance(check, dict)
        )

        def passed(label: str) -> bool:
            return names[label] == 1 and any(
                check.get("name") == label and check.get("passed") is True
                for check in checks
                if isinstance(check, dict)
            )

        proofs = {
            proof: all(passed(label) for label in labels)
            for proof, labels in required_proof_checks(name, page_bytes).items()
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
            overlap = measurements.get("request_overlap", {})
            proofs["request_overlap"] = passed(
                "tc03: observed request intervals are valid"
            ) and _observed_overlap(overlap)
        recorded = case.get("proofs")
        for proof in sorted(REQUIRED_PROOFS[name]):
            if not proofs.get(proof):
                errors.append(f"{name}: required {proof} proof is missing or failed")
            if not isinstance(recorded, dict) or recorded.get(proof) is not True:
                errors.append(f"{name}: recorded {proof} verdict is missing or failed")


def _offload(config: dict, profile: dict, index: int, errors: list[str]) -> None:
    if (
        config.get("kv_connector") != "OffloadingConnector"
        or config.get("kv_role") != "kv_both"
    ):
        errors.append(f"worker-{index}: KVCR OffloadingConnector was not selected")
        return
    extra = config.get("kv_connector_extra_config", {})
    if (
        extra.get("spec_name") != "TieringOffloadingSpec"
        or extra.get("cpu_bytes_to_use") != profile["launch"]["cpu_bytes_to_use"]
    ):
        errors.append(f"worker-{index}: primary tier configuration differs")
    tiers = extra.get("secondary_tiers", [])
    if (
        not isinstance(tiers, list)
        or len(tiers) != 1
        or not isinstance(tiers[0], dict)
        or tiers[0].get("type") != "kvcr"
    ):
        errors.append(
            f"worker-{index}: expected exactly one actual KVCR secondary tier"
        )
        return
    tier = tiers[0]
    guarded = profile["mode"] == "guard"
    if (
        bool(tier.get("kvcr_service_socket_path")) != guarded
        or bool(extra.get("dynamo_state_agent")) != guarded
    ):
        errors.append(f"worker-{index}: Guard/native service configuration differs")
    if not guarded and tier.get("secondary_g2_slots") != profile["launch"]["g2_slots"]:
        errors.append(f"worker-{index}: native KVCR capacity differs")
    if tier.get("enable_telemetry") is not True or "router_hint" not in tier.get(
        "router_capabilities", []
    ):
        errors.append(
            f"worker-{index}: required KVCR telemetry/router hints are disabled"
        )


def _precise_kvcr_log(directory: Path, name: str, errors: list[str]) -> None:
    """Observe precision on every required KVCR sample, not just launch intent."""
    try:
        text = (directory / "run/stack" / f"{name}.log").read_text(errors="replace")
    except OSError:
        errors.append(f"{name}: required precise KVCR telemetry log is missing")
        return
    samples, parse_errors = parse_log(text)
    if not text.endswith("\n") and "KV Transfer metrics:" in text.rsplit("\n", 1)[-1]:
        parse_errors.append("incomplete final KVCR metric line")
    kvcr_samples = [
        sample
        for sample in samples
        if any(key.startswith("vllm:kvcr_") for key in sample.metrics)
    ]
    lines = text.split("\n")
    if (
        parse_errors
        or not kvcr_samples
        or any(
            sample.second_precision
            or not ANSI.sub("", lines[sample.line - 1])
            .strip()
            .startswith(VLLM_EPOCH_MARKER)
            for sample in kvcr_samples
        )
    ):
        errors.append(
            f"{name}: required precise KVCR telemetry is missing or malformed"
        )


def _worker(
    directory: Path,
    name: str,
    profile: dict,
    envelope: dict,
    index: int,
    reference: bool,
    errors: list[str],
) -> None:
    launch = _read(directory / "run/stack" / f"{name}.launch.json")
    command = launch.get("command")
    if not _module(command, "dynamo.vllm"):
        errors.append(f"{name}: actual worker engine is not vLLM")
        return
    flags = _options(command)
    expected = {
        "--model": envelope["model_path"],
        "--served-model-name": profile["model_id"],
        "--tensor-parallel-size": "1",
        "--data-parallel-size": "1",
        "--block-size": "64",
        "--max-model-len": str(profile["launch"]["max_model_len"]),
        "--gpu-memory-utilization": str(profile["resources"]["gpu_memory_utilization"]),
        "--revision": profile["model_revision"],
        "--code-revision": profile["model_revision"],
        "--enable-prefix-caching": True,
        "--enforce-eager": True,
    }
    for flag, value in expected.items():
        if flags.get(flag) != value:
            errors.append(f"{name}: actual launch differs: {flag}")
    if ("--disable-hybrid-kv-cache-manager" in flags) == profile["launch"][
        "keep_hybrid_kv_cache_manager"
    ]:
        errors.append(f"{name}: hybrid cache-manager policy differs")
    for flag, expected_value in _options(profile["launch"]["worker_args"]).items():
        if flags.get(flag) != expected_value:
            errors.append(f"{name}: configured worker argument differs: {flag}")
    env = launch.get("environment_overrides", {})
    if env.get("VLLM_LOGGING_PREFIX") != VLLM_LOGGING_PREFIX:
        errors.append(f"{name}: precise vLLM logging launch configuration differs")
    if not reference and index != 2:
        _precise_kvcr_log(directory, name, errors)
    if env.get("CUDA_VISIBLE_DEVICES") != str(profile["resources"]["gpus"][index]):
        errors.append(f"{name}: launch GPU placement differs")
    if env.get("VLLM_BATCH_INVARIANT") != (
        "1" if profile["launch"]["batch_invariant"] else "0"
    ):
        errors.append(f"{name}: reference/candidate determinism setting differs")
    needs_ds = profile["topology"] == "disagg" and profile["model_family"] != "dense"
    if needs_ds and env.get("VLLM_SSM_CONV_STATE_LAYOUT") != "DS":
        errors.append(f"{name}: hybrid NIXL dimension-first layout is missing")
    if not needs_ds and "VLLM_SSM_CONV_STATE_LAYOUT" in env:
        errors.append(f"{name}: unexpected hybrid transfer layout override")
    if "--mamba-cache-mode" in flags:
        errors.append(f"{name}: obsolete Mamba alignment override is present")
    disagg = profile["topology"] == "disagg"
    role = "decode" if index == 2 else "prefill"
    if flags.get("--disaggregation-mode") != (role if disagg else None):
        errors.append(f"{name}: actual deployment role differs")
    config = (
        json.loads(flags["--kv-transfer-config"])
        if "--kv-transfer-config" in flags
        else None
    )
    if not disagg:
        if reference:
            if config is not None:
                errors.append(f"{name}: independent reference has a transfer connector")
        elif isinstance(config, dict):
            _offload(config, profile, index, errors)
        else:
            errors.append(f"{name}: candidate KVCR connector is missing")
    elif reference or index == 2:
        expected_nixl = {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_consumer" if index == 2 else "kv_producer",
        }
        if config != expected_nixl:
            errors.append(
                f"{name}: reference/decoder NIXL role differs or includes KVCR"
            )
    elif isinstance(config, dict) and config.get("kv_connector") == "MultiConnector":
        extra = config.get("kv_connector_extra_config", {})
        connectors = extra.get("connectors", [])
        if len(connectors) != 2 or connectors[1] != {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer",
        }:
            errors.append(
                f"{name}: prefill connectors do not contain KVCR plus NIXL producer"
            )
        else:
            _offload(connectors[0], profile, index, errors)
            nested = connectors[0].get("kv_connector_extra_config", {})
            for key in (
                "secondary_tiers",
                "dynamo_state_agent",
                "self_describing_kv_events",
            ):
                if extra.get(key) != nested.get(key):
                    errors.append(f"{name}: top-level router metadata differs: {key}")
    else:
        errors.append(f"{name}: disaggregated prefill MultiConnector is missing")


def _smoke(
    native: dict, directory: Path, profile: dict, envelope: dict, errors: list[str]
) -> None:
    if native.get("status") != "success" or native.get("error"):
        errors.append("Peer-transfer smoke did not pass")
    for field, expected in (
        ("engine", "vllm"),
        ("mode", "native"),
        ("topology", "peer-transfer"),
        ("model", profile["model_id"]),
        ("model_path", envelope.get("model_path")),
    ):
        if native.get(field) != expected:
            errors.append(f"Peer-transfer native identity differs: {field}")
    cached = native.get("cached_tokens")
    if type(cached) is not int or cached <= 0 or cached % 64:
        errors.append("Peer-transfer smoke lacks positive aligned reuse")
    metrics = native.get("metrics", {})
    for name in ("transfer/success", "remote_deliver/success"):
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            errors.append(f"Peer-transfer smoke lacks positive {name}")
    launch = _read(directory / "run/worker.launch.json")
    command = launch.get("command")
    if not _module(command, "dynamo.vllm"):
        errors.append("Peer-transfer actual worker engine differs")
        return
    flags = _options(command)
    for field, expected in (
        ("--model", envelope.get("model_path")),
        ("--served-model-name", profile["model_id"]),
        ("--tensor-parallel-size", "1"),
        ("--data-parallel-size", "2"),
    ):
        if flags.get(field) != expected:
            errors.append(f"Peer-transfer actual launch differs: {field}")
    config = json.loads(flags.get("--kv-transfer-config", "{}"))
    if config.get("kv_connector") != "OffloadingConnector":
        errors.append("Peer-transfer smoke did not use KVCR OffloadingConnector")
    tiers = config.get("kv_connector_extra_config", {}).get("secondary_tiers", [])
    if len(tiers) != 1 or tiers[0].get("type") != "kvcr":
        errors.append("Peer-transfer smoke lacks the actual KVCR secondary tier")


def validate_result(
    profile: dict, envelope: dict, directory: Path, errors: list[str]
) -> None:
    metadata = envelope.get("image_build_metadata", {})
    overlay = metadata.get("dynamo_vllm_fix", {})
    hashes = overlay.get("files", {})
    if (
        overlay.get("commit") != OVERLAY_COMMIT
        or set(hashes) != OVERLAY_FILES
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in hashes.values()
        )
    ):
        errors.append("Required Dynamo compatibility overlay provenance differs")
    native = _read(directory / "run/result.json")
    if profile["topology"] == "peer-transfer":
        _smoke(native, directory, profile, envelope, errors)
        return
    validate_phase1_cleanup(_read(directory / "run/stack/cleanup.json"))
    mode = "guard" if profile["mode"] == "guard" else "nonguard"
    expected = {
        "engine": "vllm",
        "model": profile["model_id"],
        "model_path": envelope.get("model_path"),
        "mode": mode,
        "topology": profile["topology"],
        "selected_cases": profile["cases"],
        "state_agent": profile["mode"] == "guard",
        "batch_invariant": profile["launch"]["batch_invariant"],
        "disable_hybrid_kv_cache_manager": not profile["launch"][
            "keep_hybrid_kv_cache_manager"
        ],
        "g2_slots": profile["launch"]["g2_slots"],
        "max_tokens": profile["launch"]["max_tokens"],
        "enforce_eager": True,
    }
    for field, value in expected.items():
        if native.get(field) != value:
            errors.append(f"Native harness identity differs: {field}")
    args = _read(directory / "run/provenance.json").get("arguments", {})
    launch_policy = profile["launch"]
    arguments = {
        "engine": "vllm",
        "model": profile["model_id"],
        "model_path": envelope.get("model_path"),
        "mode": mode,
        "topology": profile["topology"],
        "cases": profile["cases"],
        "gpus": profile["resources"]["gpus"],
        "gpu_memory_utilization": profile["resources"]["gpu_memory_utilization"],
        "allow_blocked": False,
        "no_state_agent": False,
        "no_enforce_eager": False,
        "keep_hybrid_kv_cache_manager": launch_policy["keep_hybrid_kv_cache_manager"],
    }
    arguments.update(
        {
            key: value
            for key, value in launch_policy.items()
            if key not in ("worker_args", "keep_hybrid_kv_cache_manager")
        }
    )
    for field, value in arguments.items():
        if args.get(field) != value:
            errors.append(f"Native provenance argument differs: {field}")
    expected_worker = [
        f"--revision={profile['model_revision']}",
        f"--code-revision={profile['model_revision']}",
        *launch_policy["worker_args"],
    ]
    if args.get("worker_arg") != expected_worker:
        errors.append("Native provenance worker arguments differ")
    _proofs(native, errors, launch_policy["page_bytes"])
    indexes = range(len(profile["resources"]["gpus"]))
    for index in indexes:
        _worker(directory, f"worker-{index}", profile, envelope, index, False, errors)
    _worker(directory, "worker-0-reference", profile, envelope, 0, True, errors)
    if profile["topology"] == "disagg":
        _worker(directory, "worker-2-reference", profile, envelope, 2, True, errors)
    guarded = profile["mode"] == "guard"
    for name in ("kv-state-agent", "kvcr-service-0", "kvcr-service-1"):
        if (directory / "run/stack" / f"{name}.launch.json").is_file() != guarded:
            errors.append(f"Actual Guard/native service set differs: {name}")
    if guarded:
        _worker(directory, "worker-0-replacement", profile, envelope, 0, False, errors)
    frontend = _read(directory / "run/stack/frontend.launch.json")
    if (
        frontend.get("environment_overrides", {}).get(
            "DYN_ROUTER_WORKER_SELECTION_POLICY"
        )
        != "kvcr"
    ):
        errors.append("Actual frontend did not select the KVCR worker-selection policy")
