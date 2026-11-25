# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed verdict for one selected integration configuration matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

from ci_integrations import get_integration
from ci_profiles import get_profile, load_catalog, manifest_profile


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _summarize_v1(
    manifest: dict[str, Any],
    artifact_root: Path,
    job_results: dict[str, Any],
) -> dict[str, Any]:
    """Compare expected identities, Actions outcomes and actual test evidence."""
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    for name in ("plan", "peer-transfer", "profiles"):
        job = job_results.get(name)
        conclusion = job.get("result") if isinstance(job, dict) else None
        if conclusion != "success":
            errors.append(f"Actions job {name}: {conclusion or 'missing'}")
    if manifest.get("schema_version") != 1:
        errors.append("Unsupported or missing manifest schema")
    if manifest.get("selection") not in ("all", "diagnostic"):
        errors.append("Invalid or missing profile selection identity")
    profiles = manifest.get("profiles")
    smoke = manifest.get("smoke")
    if not isinstance(profiles, list) or not profiles:
        errors.append("Expected configuration selection is empty or malformed")
        profiles = []
    if manifest.get("selection") == "all" and len(profiles) != 12:
        errors.append("Full vLLM selection requires twelve configuration profiles")
    if manifest.get("selection") == "all":
        tuples = {
            tuple(profile.get(key) for key in ("model_family", "topology", "mode"))
            for profile in profiles
            if isinstance(profile, dict)
        }
        if tuples != set(
            product(("dense", "mamba", "gdn"), ("agg", "disagg"), ("guard", "native"))
        ):
            errors.append("Full vLLM selection does not cover every required tuple")
    if (
        not isinstance(smoke, dict)
        or smoke.get("id") != "vllm-dense-peer-transfer-smoke"
    ):
        errors.append("Expected dense peer-transfer smoke is missing or malformed")
        smoke = {}
    expected: dict[str, dict[str, Any]] = {}
    for profile in [*profiles, smoke]:
        if not isinstance(profile, dict) or not isinstance(profile.get("id"), str):
            errors.append("Malformed expected profile")
            continue
        profile_id = profile["id"]
        if profile_id in expected:
            errors.append(f"Duplicate expected profile: {profile_id}")
        expected[profile_id] = profile
        if profile_id != "vllm-dense-peer-transfer-smoke":
            cases = ["TC-01", "TC-02", "TC-03"] + (
                ["TC-04"] if profile.get("mode") == "guard" else []
            )
            if (
                profile.get("integration") != "vllm"
                or profile.get("mode") not in ("guard", "native")
                or profile.get("topology") not in ("agg", "disagg")
                or profile.get("cases") != cases
            ):
                errors.append(f"{profile_id}: invalid configuration/case applicability")
    found: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(artifact_root.rglob("profile-result.json")):
        try:
            result = read_object(path)
            profile_id = result.get("profile_id")
            if not isinstance(profile_id, str):
                raise ValueError("missing profile_id")
            found.setdefault(profile_id, []).append((path.parent, result))
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"Malformed artifact {path}: {error}")
    for unexpected in sorted(found.keys() - expected.keys()):
        errors.append(f"Unexpected profile result: {unexpected}")
    for profile_id, profile in expected.items():
        profile_errors: list[str] = []
        timings: dict[str, float] = {}
        matches = found.get(profile_id, [])
        if len(matches) != 1:
            profile_errors.append(
                f"Expected exactly one artifact, found {len(matches)}"
            )
        else:
            directory, result = matches[0]
            for field in (
                "schema_version",
                "run_id",
                "run_attempt",
                "image",
                "source_sha",
                "workflow_sha",
            ):
                if not manifest.get(field) or result.get(field) != manifest[field]:
                    profile_errors.append(f"Wrong or missing {field}")
            if (
                not profile.get("fingerprint")
                or result.get("profile_fingerprint") != profile["fingerprint"]
            ):
                profile_errors.append("Wrong or missing profile fingerprint")
            for field in (
                "cases",
                "integration",
                "model_family",
                "model_id",
                "model_revision",
                "topology",
                "mode",
            ):
                if field not in profile or result.get(field) != profile[field]:
                    profile_errors.append(f"Wrong or missing profile {field}")
            if result.get("status") != "PASS":
                profile_errors.append(
                    f"Profile verdict: {result.get('status', 'missing')}"
                )
            if result.get("diagnostic") is not False:
                profile_errors.append("Execution used diagnostic overrides")
            if result.get("installed_source_sha") != manifest.get("source_sha"):
                profile_errors.append(
                    "Installed image source differs from expected source"
                )
            if not profile.get("gpus") or result.get("actual_gpus") != profile["gpus"]:
                profile_errors.append(
                    "Executed GPU placement differs from selected profile"
                )
            for stage in ("setup", "harness", "cleanup"):
                detail = result.get(stage)
                if not isinstance(detail, dict) or detail.get("status") != "PASS":
                    profile_errors.append(f"{stage} did not pass")
                    continue
                seconds = detail.get("duration_seconds")
                if (
                    not isinstance(seconds, (int, float))
                    or isinstance(seconds, bool)
                    or not math.isfinite(seconds)
                    or seconds < 0
                ):
                    profile_errors.append(f"Invalid {stage} duration")
                else:
                    timings[stage] = seconds
            try:
                native = read_object(directory / "run" / "result.json")
                if profile_id == smoke.get("id"):
                    _check_smoke(native, profile_errors)
                    cleanup = read_object(directory / "run" / "cleanup.json")
                    if cleanup.get("status") != "PASS":
                        profile_errors.append("Smoke owned cleanup did not pass")
                else:
                    _check_harness(native, profile.get("cases"), profile_errors)
                    cleanup = read_object(directory / "run" / "stack" / "cleanup.json")
                    provenance = read_object(directory / "run" / "provenance.json")
                    harness_sha = result.get("harness_sha256")
                    if (
                        not isinstance(harness_sha, str)
                        or len(harness_sha) != 64
                        or provenance.get("harness_sha256") != harness_sha
                    ):
                        profile_errors.append(
                            "Harness provenance does not match execution"
                        )
                owned = cleanup.get("owned_processes")
                if not isinstance(owned, list) or any(
                    not isinstance(process, dict) or "error" in process
                    for process in owned
                ):
                    profile_errors.append(
                        "Owned-process cleanup evidence is missing or failed"
                    )
            except (OSError, ValueError, TypeError) as error:
                profile_errors.append(
                    f"Missing or malformed native harness result: {error}"
                )
        rows.append(
            {
                "id": profile_id,
                "status": "FAIL" if profile_errors else "PASS",
                "cases": profile.get("cases", []),
                "errors": profile_errors,
                "duration_seconds": timings,
            }
        )
        errors.extend(f"{profile_id}: {error}" for error in profile_errors)
    return {
        "schema_version": 1,
        "status": "FAIL" if errors else "PASS",
        "selection": manifest.get("selection", "invalid"),
        "run_id": manifest.get("run_id"),
        "run_attempt": manifest.get("run_attempt"),
        "image": manifest.get("image"),
        "source_sha": manifest.get("source_sha"),
        "profiles": rows,
        "errors": errors,
    }


def _finite_duration(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _model_receipt(
    directory: Path, result: dict, profile: dict, errors: list[str]
) -> None:
    receipt = read_object(directory / "model-receipt.json")
    expected = {
        "schema_version": 1,
        "status": "PASS",
        "diagnostic_override": False,
        "requested_repository": profile["model_id"],
        "resolved_repository": profile["model_id"],
        "requested_revision": profile["model_revision"],
        "resolved_revision": profile["model_revision"],
        "resolved_path": result.get("model_path"),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        errors.append(
            "Model receipt does not prove the selected repository/revision/path"
        )
    path = Path(str(receipt.get("resolved_path", "")))
    if (
        not path.is_absolute()
        or path.name != profile["model_revision"]
        or path.parent.name != "snapshots"
        or path.parent.parent.name
        != "models--" + profile["model_id"].replace("/", "--")
    ):
        errors.append("Model snapshot path does not identify the selected checkpoint")
    weights = receipt.get("weight_files")
    required = receipt.get("required_files")
    if (
        not isinstance(weights, list)
        or not weights
        or not isinstance(required, list)
        or "config.json" not in required
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not item["name"].endswith((".safetensors", ".bin"))
            or Path(item["name"]).is_absolute()
            or ".." in Path(item["name"]).parts
            or item["name"] not in required
            or type(item.get("bytes")) is not int
            or item["bytes"] <= 0
            for item in weights
        )
    ):
        errors.append("Model receipt lacks a complete readable weight inventory")
    elif receipt.get("weight_bytes") != sum(item["bytes"] for item in weights):
        errors.append("Model receipt weight-byte accounting differs")
    if (
        not isinstance(receipt.get("huggingface_hub_version"), str)
        or not receipt["huggingface_hub_version"]
        or not _finite_duration(receipt.get("duration_seconds"))
    ):
        errors.append("Model preparation version/timing receipt is missing")


def _common_v2_evidence(
    manifest: dict,
    profile: dict,
    result: dict,
    directory: Path,
    errors: list[str],
    resource_policy: dict,
) -> tuple[dict[str, float], list[str]]:
    # Import here: the launcher has no dependency on the verdict module, and the
    # hash covers both files plus the registered adapter/identity policy sources.
    from run_ci_profile import policy_fingerprint

    timings = {}
    for field in (
        "schema_version",
        "integration",
        "run_id",
        "run_attempt",
        "source_sha",
        "workflow_sha",
        "recipe_sha",
        "image",
        "image_profile",
        "resolved_inputs_sha256",
    ):
        if result.get(field) != manifest.get(field):
            errors.append(f"Wrong or missing {field}")
    for field in (
        "cases",
        "integration",
        "model_family",
        "model_id",
        "model_revision",
        "topology",
        "mode",
    ):
        if result.get(field) != profile.get(field):
            errors.append(f"Wrong or missing profile {field}")
    if result.get("profile_fingerprint") != profile.get("fingerprint"):
        errors.append("Wrong or missing profile fingerprint")
    if result.get("policy_sha256") != policy_fingerprint():
        errors.append("Executed policy fingerprint differs from registered source")
    for field, expected in (
        ("status", "PASS"),
        ("diagnostic", False),
        ("installed_source_sha", manifest["source_sha"]),
        ("checked_out_source_sha", manifest["recipe_sha"]),
        ("source_mismatches", []),
        ("requested_gpu_mapping", profile["gpus"]),
        ("intended_gpus", profile["gpus"]),
    ):
        if result.get(field) != expected:
            errors.append(f"Wrong or missing {field}")
    metadata = result.get("image_build_metadata")
    if not isinstance(metadata, dict):
        errors.append("Installed image build provenance is missing")
    else:
        for field, expected in (
            ("schema_version", 2),
            ("kvcr_sha", manifest["source_sha"]),
            ("recipe_sha", manifest["recipe_sha"]),
            ("image_profile", manifest["image_profile"]),
            ("resolved_inputs_sha256", manifest["resolved_inputs_sha256"]),
        ):
            if metadata.get(field) != expected:
                errors.append(f"Installed image build provenance differs: {field}")
        if not re.fullmatch(
            r".+@sha256:[0-9a-f]{64}", str(metadata.get("base_image", ""))
        ):
            errors.append("Installed image lacks its immutable base digest")
    devices = result.get("gpu_observation", {})
    observed = devices.get("allocated_devices") if isinstance(devices, dict) else None
    if (
        devices.get("status") != "OBSERVED"
        or not isinstance(observed, list)
        or not observed
        or any(
            not isinstance(device, dict)
            or type(device.get("index")) is not int
            or not isinstance(device.get("uuid"), str)
            or not device["uuid"]
            for device in observed
        )
    ):
        errors.append("Allocated GPU observation is missing or malformed")
    elif (
        not set(profile["gpus"]).issubset({device["index"] for device in observed})
        or len({device["index"] for device in observed}) != len(observed)
        or len({device["uuid"] for device in observed}) != len(observed)
    ):
        errors.append("Allocated GPU indices/UUIDs do not cover the selected placement")
    resources = result.get("resource_observations", {})
    if resources.get("declared") != resource_policy:
        errors.append("Declared runtime resources differ from the registered adapter")
    for stage in ("before_ports", "after_ports"):
        observation = resources.get(stage, {})
        if (
            observation.get("status") != "AVAILABLE"
            or observation.get("ports") != sorted(set(resource_policy["tcp_ports"]))
            or observation.get("unavailable_ports") != []
            or observation.get("errors") != []
            or observation.get("scope") != "container-network-namespace"
        ):
            errors.append(
                f"Runtime port availability/reclamation was not verified: {stage}"
            )
    for stage in ("setup", "harness", "cleanup"):
        value = result.get(stage)
        if not isinstance(value, dict) or value.get("status") != "PASS":
            errors.append(f"{stage} did not pass")
        elif not _finite_duration(value.get("duration_seconds")):
            errors.append(f"Invalid {stage} duration")
        else:
            timings[stage] = value["duration_seconds"]
    harness = result.get("harness", {})
    ownership = harness.get("owned_process_cleanup", {})
    if (
        harness.get("returncode") != 0
        or any(
            harness.get(flag) is not False
            for flag in ("interrupted", "timed_out", "forced_kill")
        )
        or ownership.get("status") != "PASS"
        or ownership.get("required_fallback") is not False
        or ownership.get("surviving_pids") != []
    ):
        errors.append(
            "Harness supervision did not finish normally "
            "with owned descendants reclaimed"
        )
    _model_receipt(directory, result, profile, errors)
    native = read_object(directory / "run/result.json")
    if profile["cases"]:
        _check_harness(native, profile["cases"], errors)
        provenance = read_object(directory / "run/provenance.json")
        harness_hash = result.get("harness_sha256")
        if (
            not isinstance(harness_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", harness_hash)
            or provenance.get("harness_sha256") != harness_hash
            or provenance.get("image_source_revisions") != metadata
        ):
            errors.append(
                "Native harness provenance differs from execution/image identity"
            )
    cleanup = read_object(
        directory
        / "run"
        / ("stack/cleanup.json" if profile["cases"] else "cleanup.json")
    )
    if cleanup.get("status") != "PASS":
        errors.append("Owned cleanup did not pass")
    owned = cleanup.get("owned_processes")
    if not isinstance(owned, list) or any(
        not isinstance(process, dict) or "error" in process for process in owned
    ):
        errors.append("Owned-process cleanup is missing or failed")
    for field, status_field, expected in (
        ("process_states", "stopped", True),
        ("owned_paths", "status", "PASS"),
    ):
        if any(
            not isinstance(item, dict) or item.get(status_field) != expected
            for item in cleanup.get(field, [])
        ):
            errors.append(f"Owned cleanup reports unreclaimed {field}")
    warnings = cleanup.get("warnings", [])
    if not isinstance(warnings, list):
        errors.append("Cleanup warnings are malformed")
        warnings = []
    return timings, [f"{profile['id']}: {warning}" for warning in warnings]


def summarize(
    manifest: dict[str, Any],
    artifact_root: Path,
    job_results: dict[str, Any],
    *,
    legacy_replay: bool = False,
) -> dict[str, Any]:
    """Qualify v2 against checked-out adapter policy; v1 is explicit audit only."""
    if legacy_replay:
        report = _summarize_v1(manifest, artifact_root, job_results)
        report["legacy_replay"] = True
        report["production_qualified"] = False
        return report
    errors: list[str] = []
    rows: list[dict] = []
    warnings: list[str] = []
    report = {
        "schema_version": 2,
        "status": "FAIL",
        "integration": manifest.get("integration"),
        "selection": manifest.get("selection", "invalid"),
        "run_id": manifest.get("run_id"),
        "run_attempt": manifest.get("run_attempt"),
        "image": manifest.get("image"),
        "source_sha": manifest.get("source_sha"),
        "recipe_sha": manifest.get("recipe_sha"),
        "image_profile": manifest.get("image_profile"),
        "profiles": rows,
        "errors": errors,
        "warnings": warnings,
        "production_qualified": False,
    }
    try:
        if manifest.get("schema_version") != 2:
            raise ValueError(
                "Production verdict requires schema v2; "
                "use --legacy-replay for historical artifacts"
            )
        integration = manifest.get("integration")
        adapter = get_integration(integration)
        catalog = load_catalog(integration=integration)
        if manifest.get("image_profile") != adapter.IMAGE_PROFILE:
            raise ValueError("Image profile differs from registered integration policy")
        for field, pattern in (
            ("source_sha", r"[0-9a-f]{40}"),
            ("recipe_sha", r"[0-9a-f]{40}"),
            ("resolved_inputs_sha256", r"[0-9a-f]{64}"),
            ("image", r".+@sha256:[0-9a-f]{64}"),
        ):
            if not re.fullmatch(pattern, str(manifest.get(field, ""))):
                raise ValueError(f"Missing or invalid manifest {field}")
        if manifest.get("workflow_sha") != manifest["recipe_sha"]:
            raise ValueError("Workflow recipe identity differs")
        if not manifest.get("run_id") or not manifest.get("run_attempt"):
            raise ValueError("Missing run identity")
        profiles = manifest.get("profiles")
        smokes = manifest.get("smokes")
        if (
            not isinstance(profiles, list)
            or not profiles
            or not isinstance(smokes, list)
        ):
            raise ValueError("Missing selected profiles or smoke inventory")
        if any(
            not isinstance(item, dict) or not isinstance(item.get("id"), str)
            for item in [*profiles, *smokes]
        ):
            raise ValueError("Malformed selected profile identity")
        ids = [item["id"] for item in profiles]
        required_ids = adapter.required_profile_ids(catalog)
        if manifest.get("selection") == "all":
            if set(ids) != required_ids or len(ids) != len(required_ids):
                raise ValueError(
                    "Full selection differs from source-registered required profiles"
                )
        elif manifest.get("selection") == "diagnostic":
            if not set(ids).issubset(required_ids) or len(ids) != len(set(ids)):
                raise ValueError(
                    "Diagnostic selection contains unknown or duplicate profiles"
                )
        else:
            raise ValueError("Invalid profile selection identity")
        smoke_ids = [item["id"] for item in smokes]
        required_smokes = {item["id"] for item in adapter.smoke_profiles(catalog)}
        if set(smoke_ids) != required_smokes or len(smoke_ids) != len(required_smokes):
            raise ValueError(
                "Smoke selection differs from source-registered required smokes"
            )
        for name in ("plan", "profiles", "smokes"):
            if name == "smokes" and not required_smokes and name not in job_results:
                continue
            job = job_results.get(name, {})
            allowed = (
                {"success", "skipped"}
                if name == "smokes" and not required_smokes
                else {"success"}
            )
            if not isinstance(job, dict) or job.get("result") not in allowed:
                conclusion = (
                    job.get("result", "missing")
                    if isinstance(job, dict)
                    else "malformed"
                )
                errors.append(f"Actions job {name}: {conclusion}")
        expected = {}
        effective = {}
        for item in [*profiles, *smokes]:
            profile = get_profile(catalog, item["id"])
            authoritative = manifest_profile(profile)
            if any(
                item.get(key) != value
                for key, value in authoritative.items()
                if key != "qualification"
            ):
                errors.append(
                    f"{item['id']}: manifest differs from source-registered profile"
                )
            if item["id"] in expected:
                errors.append(f"Duplicate expected profile: {item['id']}")
            expected[item["id"]] = authoritative
            effective[item["id"]] = profile
        found = {}
        for path in sorted(artifact_root.rglob("profile-result.json")):
            try:
                value = read_object(path)
                if not isinstance(value.get("profile_id"), str):
                    raise ValueError("missing profile_id")
                found.setdefault(value["profile_id"], []).append((path.parent, value))
            except (OSError, ValueError, TypeError) as error:
                errors.append(f"Malformed artifact {path}: {error}")
        for name in sorted(found.keys() - expected.keys()):
            errors.append(f"Unexpected profile result: {name}")
        for name, profile in expected.items():
            row_errors = []
            timings = {}
            matches = found.get(name, [])
            if len(matches) != 1:
                row_errors.append(
                    f"Expected exactly one artifact, found {len(matches)}"
                )
            else:
                directory, envelope = matches[0]
                try:
                    timings, row_warnings = _common_v2_evidence(
                        manifest,
                        profile,
                        envelope,
                        directory,
                        row_errors,
                        adapter.runtime_resources(effective[name]),
                    )
                    warnings.extend(row_warnings)
                    adapter.validate_result(
                        effective[name], envelope, directory, row_errors
                    )
                except (
                    OSError,
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                ) as error:
                    row_errors.append(
                        f"Missing or malformed execution evidence: {error}"
                    )
            rows.append(
                {
                    "id": name,
                    "status": "FAIL" if row_errors else "PASS",
                    "cases": profile["cases"],
                    "errors": row_errors,
                    "duration_seconds": timings,
                }
            )
            errors.extend(f"{name}: {error}" for error in row_errors)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        errors.append(f"Cannot establish registered matrix: {error}")
    report["status"] = "FAIL" if errors else "PASS"
    report["production_qualified"] = not errors and report["selection"] == "all"
    return report


def _check_harness(native: dict[str, Any], cases: Any, errors: list[str]) -> None:
    if native.get("status") != "PASS":
        errors.append(f"Harness verdict: {native.get('status', 'missing')}")
    if (
        native.get("error")
        or native.get("cleanup_error")
        or native.get("missing_cases")
    ):
        errors.append("Harness reports execution, cleanup or missing-case errors")
    actual_cases = native.get("cases")
    if not isinstance(actual_cases, list) or not isinstance(cases, list) or not cases:
        errors.append("Missing or malformed required cases")
    else:
        ids = [
            case.get("case") if isinstance(case, dict) else None
            for case in actual_cases
        ]
        if Counter(ids) != Counter(cases):
            errors.append("Missing, duplicate or unexpected harness cases")
        if any(
            not isinstance(case, dict) or case.get("status") != "PASS"
            for case in actual_cases
        ):
            errors.append("At least one required case did not pass")
        for case in actual_cases:
            checks = case.get("checks") if isinstance(case, dict) else None
            if (
                not isinstance(checks, list)
                or not checks
                or any(
                    not isinstance(check, dict) or check.get("passed") is not True
                    for check in checks
                )
            ):
                errors.append("Case assertions are missing or failed")
                break
    reference = native.get("reference")
    if (
        not isinstance(reference, dict)
        or reference.get("kvcr_disabled_in_launch") is not True
    ):
        errors.append("Independent reference was not verified KVCR-disabled")
        return
    if reference.get("unstable"):
        errors.append("Independent reference was unstable")
    prompts = reference.get("prompts")
    if (
        not isinstance(prompts, dict)
        or not prompts
        or any(
            not isinstance(prompt, dict) or prompt.get("stable") is not True
            for prompt in prompts.values()
        )
    ):
        errors.append("Independent reference prompt stability is missing or failed")


def _check_smoke(native: dict[str, Any], errors: list[str]) -> None:
    if native.get("status") != "success" or native.get("error"):
        errors.append("Peer-transfer smoke did not pass")
    cached = native.get("cached_tokens")
    if (
        not isinstance(cached, int)
        or isinstance(cached, bool)
        or cached <= 0
        or cached % 64
    ):
        errors.append("Peer-transfer smoke lacks positive block-aligned cache reuse")
    metrics = native.get("metrics", {})
    for name in ("transfer/success", "remote_deliver/success"):
        count = metrics.get(name) if isinstance(metrics, dict) else None
        if (
            not isinstance(count, (int, float))
            or isinstance(count, bool)
            or not math.isfinite(count)
            or count <= 0
        ):
            errors.append(f"Peer-transfer smoke lacks positive {name}")


def markdown(report: dict[str, Any]) -> str:
    integration = report.get("integration", "vllm")
    scope = (
        f"Full {integration} matrix"
        if report["selection"] == "all"
        else "Diagnostic subset — not full coverage"
    )
    lines = [
        f"## {scope}: {report['status']}",
        "",
        "| Configuration | Cases | Setup | Execution | Cleanup | Result |",
        "|---|---|---|---|---|---|",
    ]
    for profile in report["profiles"]:
        cases = ", ".join(profile["cases"]) or "DP=2 smoke"
        durations = profile.get("duration_seconds", {})
        timing = " | ".join(
            f"{durations[stage]:.1f}s" if stage in durations else "missing"
            for stage in ("setup", "harness", "cleanup")
        )
        lines.append(f"| {profile['id']} | {cases} | {timing} | {profile['status']} |")
    lines.extend(
        [
            "",
            "Execution includes the independent reference, cases and harness cleanup; "
            "the cleanup column shows its measured subduration and verification. "
            "Runner queue and artifact upload time are reported by Actions.",
        ]
    )
    if report["errors"]:
        lines.extend(
            ["", "Failures:", "", *(f"- {error}" for error in report["errors"])]
        )
    if report.get("legacy_replay"):
        lines.extend(
            ["", "Historical replay only; does not qualify a production v2 run."]
        )
    if report.get("warnings"):
        lines.extend(
            [
                "",
                "Cleanup observations "
                "(reclamation does not establish graceful shutdown):",
                "",
                *(f"- {warning}" for warning in report["warnings"]),
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--job-results", required=True, help="JSON from Actions needs context"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--legacy-replay",
        action="store_true",
        help="Explicit historical v1 audit; never production qualification",
    )
    args = parser.parse_args()
    try:
        manifest = read_object(args.manifest)
        job_results = json.loads(args.job_results)
        if not isinstance(job_results, dict):
            raise ValueError("Actions job results must be an object")
        report = summarize(
            manifest, args.artifacts, job_results, legacy_replay=args.legacy_replay
        )
    except (OSError, ValueError, TypeError) as error:
        report = {
            "schema_version": 1,
            "status": "FAIL",
            "selection": "invalid",
            "profiles": [],
            "errors": [f"Cannot establish expected matrix: {error}"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    text = markdown(report)
    print(text)
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(step_summary).open("a") as stream:
            stream.write(text)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
