# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production verdict contracts use recorded inputs, not an artifact's policy."""

from __future__ import annotations

import copy
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

E2E = Path(__file__).parents[1] / "e2e"
sys.path.insert(0, str(E2E))

import ci_profiles as profiles  # noqa: E402
import run_ci_profile as launcher  # noqa: E402
import run_phase1 as phase1  # noqa: E402
import summarize_ci as summary  # noqa: E402
from ci_identity import OVERLAY_COMMIT, OVERLAY_FILES  # noqa: E402
from ci_integrations import vllm  # noqa: E402
from phase1.cases import (  # noqa: E402
    PROOF_CHECKS,
    TELEMETRY_LABELS,
    required_proof_checks,
)


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


def mutate(path, change):
    document = json.loads(path.read_text())
    change(document)
    write(path, document)


def make_evidence(root, selection="vllm-gdn-disagg-guard"):
    catalog = profiles.load_catalog()
    # Synthetic identities must not inherit the hosting Actions run's SHA.
    # Keep the production environment/recipe identity check enabled.
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("GITHUB_SHA", "d" * 40)
        manifest = profiles.make_manifest(
            catalog,
            selection,
            "registry/kvcr@sha256:" + "c" * 64,
            "a" * 40,
            "123",
            "2",
            recipe_sha="d" * 40,
            image_profile="vllm",
            resolved_inputs_sha256="f" * 64,
        )
    metadata = {
        "schema_version": 2,
        "kvcr_sha": manifest["source_sha"],
        "recipe_sha": manifest["recipe_sha"],
        "image_profile": "vllm",
        "resolved_inputs_sha256": manifest["resolved_inputs_sha256"],
        "base_image": "vllm/image@sha256:" + "9" * 64,
        "packages": {
            "ai-dynamo": "1.8.0.dev20261123",
            "ai-dynamo-runtime": "1.8.0.dev20261123",
            "vllm": "0.34.0",
        },
        "dynamo_vllm_fix": {
            "commit": OVERLAY_COMMIT,
            "files": {name: "8" * 64 for name in OVERLAY_FILES},
        },
    }
    for item in [*manifest["profiles"], *manifest["smokes"]]:
        profile = profiles.get_profile(catalog, item["id"])
        directory = root / item["id"]
        run = directory / "run"
        snapshot = (
            Path("/scratch/huggingface/hub")
            / ("models--" + item["model_id"].replace("/", "--"))
            / "snapshots"
            / item["model_revision"]
        )
        envelope = {
            key: manifest[key]
            for key in (
                "schema_version",
                "integration",
                "run_id",
                "run_attempt",
                "source_sha",
                "workflow_sha",
                "image",
                "recipe_sha",
                "image_profile",
                "resolved_inputs_sha256",
            )
        }
        envelope.update(
            profile_id=item["id"],
            profile_fingerprint=item["fingerprint"],
            policy_sha256=launcher.policy_fingerprint(),
            status="PASS",
            diagnostic=False,
            installed_source_sha=manifest["source_sha"],
            checked_out_source_sha=manifest["recipe_sha"],
            source_mismatches=[],
            image_build_metadata=copy.deepcopy(metadata),
            actual_gpus=item["gpus"],
            intended_gpus=item["gpus"],
            requested_gpu_mapping=item["gpus"],
            gpu_observation={
                "status": "OBSERVED",
                "allocated_devices": [
                    {
                        "index": i,
                        "uuid": f"GPU-{i}",
                        "name": "NVIDIA H100",
                        "memory_total_mib": 80000,
                    }
                    for i in sorted(set(item["gpus"]))
                ],
            },
            harness_sha256="e" * 64,
            model_path=str(snapshot),
            **{
                key: item[key]
                for key in (
                    "cases",
                    "model_family",
                    "model_id",
                    "model_revision",
                    "topology",
                    "mode",
                )
            },
            **{
                stage: {"status": "PASS", "duration_seconds": 1.0}
                for stage in ("setup", "harness", "cleanup")
            },
        )
        ports = vllm.runtime_resources(profile)["tcp_ports"]
        port_receipt = {
            "status": "AVAILABLE",
            "ports": sorted(ports),
            "unavailable_ports": [],
            "errors": [],
            "scope": "container-network-namespace",
        }
        envelope["resource_observations"] = {
            "declared": vllm.runtime_resources(profile),
            "before_ports": copy.deepcopy(port_receipt),
            "after_ports": copy.deepcopy(port_receipt),
        }
        envelope["harness"].update(
            returncode=0,
            interrupted=False,
            timed_out=False,
            forced_kill=False,
            owned_process_cleanup={
                "status": "PASS",
                "required_fallback": False,
                "surviving_pids": [],
            },
        )
        envelope["command"] = vllm.build_command(profile, directory, snapshot, "test")
        receipt = {
            "schema_version": 1,
            "status": "PASS",
            "diagnostic_override": False,
            "requested_repository": item["model_id"],
            "resolved_repository": item["model_id"],
            "requested_revision": item["model_revision"],
            "resolved_revision": item["model_revision"],
            "resolved_path": str(snapshot),
            "huggingface_hub_version": "1.0.0",
            "cache_scope": "job-container",
            "cache_path": "/scratch/huggingface/hub",
            "weight_files": [{"name": "model.safetensors", "bytes": 1024}],
            "required_files": ["config.json", "model.safetensors"],
            "weight_bytes": 1024,
            "validation": "config-json-and-required-weight-readability",
            "weights_rehashed": False,
            "duration_seconds": 1.0,
        }
        write(directory / "profile-result.json", envelope)
        write(directory / "model-receipt.json", receipt)
        cleanup = {
            "schema_version": 2,
            "status": "PASS",
            "reclaimed": True,
            "graceful": False,
            "observation_complete": False,
            "owned_processes": [{"name": "worker-0", "pid": 1000, "exit_code": 0}],
            "process_states": [{"name": "worker-0", "pid": 1000, "stopped": True}],
            "owned_paths": [
                {"path": "/tmp/owned-runtime", "status": "PASS", "errors": []}
            ],
            "warnings": [],
        }
        if not item["cases"]:
            from types import SimpleNamespace

            from run_vllm_kvcr import worker_arguments

            command = [
                sys.executable,
                "-m",
                "dynamo.vllm",
                *worker_arguments(
                    SimpleNamespace(model=item["model_id"], model_path=snapshot)
                ),
            ]
            write(
                run / "worker.launch.json",
                {"command": command, "environment_overrides": {}, "pid": 900},
            )
            write(
                run / "result.json",
                {
                    "status": "success",
                    "engine": "vllm",
                    "mode": "native",
                    "topology": "peer-transfer",
                    "model": item["model_id"],
                    "model_path": str(snapshot),
                    "cached_tokens": 128,
                    "metrics": {"transfer/success": 1, "remote_deliver/success": 1},
                },
            )
            write(run / "cleanup.json", cleanup)
            continue
        # Parse the real adapter command but keep the absolute runtime snapshot
        # path without reading it on this CPU-only test host.
        command = envelope["command"]
        with pytest.MonkeyPatch.context() as patch:
            original = Path.is_dir
            patch.setattr(
                Path, "is_dir", lambda p: True if p == snapshot else original(p)
            )
            args = phase1.arguments(command[3:])
        runner = phase1.Runner(args)
        runner.stack.advertise_host = lambda: "127.0.0.1"
        native = runner.result
        native.update(status="PASS", missing_cases=[])
        native["reference"] = {
            "kvcr_disabled_in_launch": True,
            "unstable": {},
            "prompts": {"seed": {"stable": True}},
        }
        for case_id in profile["cases"]:
            required = required_proof_checks(case_id, profile["launch"]["page_bytes"])
            names = [name for checks in required.values() for name in checks]
            names.extend(
                f"{label}: required KVCR telemetry is complete and healthy"
                for label in TELEMETRY_LABELS[case_id]
            )
            if case_id == "TC-03":
                names.append("tc03: observed request intervals are valid")
            native["cases"].append(
                {
                    "case": case_id,
                    "status": "PASS",
                    "checks": [{"name": name, "passed": True} for name in names],
                    "proofs": {
                        **{key: True for key in required},
                        "telemetry_health": True,
                        **({"request_overlap": True} if case_id == "TC-03" else {}),
                    },
                    "measurements": {
                        "request_overlap": {
                            "observed": True,
                            "overlap_seconds": 0.01,
                            "requests": [
                                {
                                    "stage": "hinted",
                                    "start": "2026-11-25T10:00:00Z",
                                    "end": "2026-11-25T10:00:01Z",
                                },
                                {
                                    "stage": "isolated",
                                    "start": "2026-11-25T10:00:00.99Z",
                                    "end": "2026-11-25T10:00:02Z",
                                },
                            ],
                        },
                        "engine_partial_prefix_reuse": {
                            "reuses": True,
                            "reference_cached_tokens": [128],
                        },
                    },
                }
            )
        write(run / "result.json", native)
        arguments = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        write(
            run / "provenance.json",
            {
                "harness_sha256": envelope["harness_sha256"],
                "arguments": arguments,
                "image_source_revisions": metadata,
            },
        )
        settings = runner.settings
        worker_names = [(f"worker-{i}", i, True) for i in range(len(item["gpus"]))]
        worker_names.append(("worker-0-reference", 0, False))
        if profile["topology"] == "disagg":
            worker_names.append(("worker-2-reference", 2, False))
        if profile["mode"] == "guard":
            worker_names.append(("worker-0-replacement", 0, True))
        for name, i, enabled in worker_names:
            write(
                run / "stack" / f"{name}.launch.json",
                {
                    "command": runner.stack.worker_command(i, kvcr=enabled),
                    "pid": 1000 + i,
                    "environment_overrides": {
                        "CUDA_VISIBLE_DEVICES": str(item["gpus"][i]),
                        "VLLM_BATCH_INVARIANT": "1"
                        if settings.batch_invariant
                        else "0",
                        **settings.worker_env,
                        "VLLM_LOGGING_PREFIX": "CI_LOG_EPOCH=%(created).9f ",
                    },
                },
            )
            if enabled and i != 2:
                (run / "stack" / f"{name}.log").write_text(
                    "CI_LOG_EPOCH=1795600800.250000000 (MainProcess pid=1) "
                    "INFO 11-25 10:00:00 [metrics.py:108] KV Transfer metrics: "
                    "vllm:kvcr_state:('block_records',)=0\n"
                )
        write(
            run / "stack" / "frontend.launch.json",
            {
                "command": [sys.executable, "-m", "dynamo.frontend"],
                "environment_overrides": {"DYN_ROUTER_WORKER_SELECTION_POLICY": "kvcr"},
                "pid": 2000,
            },
        )
        if profile["mode"] == "guard":
            for name in ("kv-state-agent", "kvcr-service-0", "kvcr-service-1"):
                write(
                    run / "stack" / f"{name}.launch.json",
                    {
                        "command": [sys.executable, "-m", name],
                        "environment_overrides": {},
                        "pid": 3000,
                    },
                )
        runner.stack.cleanup()
        write(run / "stack" / "cleanup.json", cleanup)
    return (
        manifest,
        root,
        {name: {"result": "success"} for name in ("plan", "smokes", "profiles")},
    )


@pytest.fixture
def evidence(tmp_path):
    return make_evidence(tmp_path)


def artifact_path(evidence, relative="profile-result.json"):
    return evidence[1] / evidence[0]["profiles"][0]["id"] / relative


def test_production_v2_evidence_passes(evidence):
    result = summary.summarize(*evidence)
    assert result["status"] == "PASS", result["errors"]
    assert result["integration"] == "vllm"
    assert result["schema_version"] == 2


def test_full_vllm_contract_has_twelve_profiles_without_standalone_smoke(tmp_path):
    result = summary.summarize(*make_evidence(tmp_path, "all"))
    assert result["status"] == "PASS", result["errors"]
    assert len(result["profiles"]) == 12


def test_removing_profile_from_both_manifest_and_results_still_fails(tmp_path):
    evidence = make_evidence(tmp_path, "all")
    removed = evidence[0]["profiles"].pop()
    shutil.rmtree(tmp_path / removed["id"])
    assert summary.summarize(*evidence)["status"] == "FAIL"


def test_unexpected_standalone_smoke_artifact_fails(evidence):
    source = artifact_path(evidence)
    document = json.loads(source.read_text())
    document["profile_id"] = "vllm-dense-peer-transfer-smoke"
    write(evidence[1] / "unexpected-smoke" / "profile-result.json", document)
    assert summary.summarize(*evidence)["status"] == "FAIL"


@pytest.mark.parametrize("outcome", ["skipped", None])
def test_empty_source_smoke_policy_allows_skipped_or_absent_job(evidence, outcome):
    assert evidence[0]["smokes"] == []
    if outcome is None:
        evidence[2].pop("smokes", None)
    else:
        evidence[2]["smokes"]["result"] = outcome
    result = summary.summarize(*evidence)
    assert result["status"] == "PASS", result["errors"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("recipe_sha", "1" * 40),
        ("image_profile", "sglang-hicache"),
        ("resolved_inputs_sha256", "1" * 64),
        ("policy_sha256", "0" * 64),
        ("checked_out_source_sha", "1" * 40),
        ("schema_version", 1),
    ],
)
def test_wrong_identity_cannot_pass(evidence, field, value):
    mutate(artifact_path(evidence), lambda doc: doc.update({field: value}))
    assert summary.summarize(*evidence)["status"] == "FAIL"


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "FAIL"),
        ("requested_revision", "0" * 40),
        ("resolved_repository", "wrong/model"),
        ("resolved_path", "/wrong/snapshot"),
        ("diagnostic_override", True),
        ("weight_files", []),
    ],
)
def test_bad_model_receipt_cannot_pass(evidence, field, value):
    mutate(
        artifact_path(evidence, "model-receipt.json"),
        lambda doc: doc.update({field: value}),
    )
    assert summary.summarize(*evidence)["status"] == "FAIL"


def test_missing_model_receipt_fails(evidence):
    artifact_path(evidence, "model-receipt.json").unlink()
    assert summary.summarize(*evidence)["status"] == "FAIL"


def test_wrong_installed_build_identity_fails(evidence):
    mutate(
        artifact_path(evidence),
        lambda doc: doc["image_build_metadata"].update(recipe_sha="0" * 40),
    )
    assert summary.summarize(*evidence)["status"] == "FAIL"


@pytest.mark.parametrize(
    "field,value",
    [
        ("engine", "sglang"),
        ("model", "wrong/model"),
        ("mode", "nonguard"),
        ("topology", "agg"),
        ("model_path", "/wrong/snapshot"),
    ],
)
def test_native_settings_checked_against_registered_policy(evidence, field, value):
    mutate(
        artifact_path(evidence, "run/result.json"),
        lambda doc: doc.update({field: value}),
    )
    assert summary.summarize(*evidence)["status"] == "FAIL"


@pytest.mark.parametrize(
    "field,value",
    [
        ("allow_blocked", True),
        ("gpus", [0, 1, 0]),
        ("keep_hybrid_kv_cache_manager", False),
        ("model_path", "/wrong/snapshot"),
    ],
)
def test_provenance_arguments_cannot_disagree(evidence, field, value):
    mutate(
        artifact_path(evidence, "run/provenance.json"),
        lambda doc: doc["arguments"].update({field: value}),
    )
    assert summary.summarize(*evidence)["status"] == "FAIL"


@pytest.mark.parametrize(
    "proof",
    [
        "partial_reuse",
        "peer_transfer",
        "telemetry_health",
        "local_restore",
        "request_overlap",
    ],
)
def test_self_reported_proof_cannot_hide_missing_assertion(evidence, proof):
    def corrupt(doc):
        case_id = {"local_restore": "TC-04", "request_overlap": "TC-03"}.get(
            proof, "TC-02"
        )
        case = next(c for c in doc["cases"] if c["case"] == case_id)
        name = (
            "tc03: observed request intervals are valid"
            if proof == "request_overlap"
            else (
                f"{TELEMETRY_LABELS[case_id][0]}: "
                "required KVCR telemetry is complete and healthy"
            )
            if proof == "telemetry_health"
            else PROOF_CHECKS[case_id][proof][0]
        )
        case["checks"] = [c for c in case["checks"] if c["name"] != name]
        case["proofs"][proof] = True

    mutate(artifact_path(evidence, "run/result.json"), corrupt)
    assert summary.summarize(*evidence)["status"] == "FAIL"


@pytest.mark.parametrize(
    "fragment",
    ["transfer bytes == pages * page bytes", "KVCR failed/partial metrics"],
)
def test_positive_proof_flags_cannot_hide_missing_transfer_guards(evidence, fragment):
    removed = []

    def corrupt(doc):
        for case in doc["cases"]:
            removed.extend(c for c in case["checks"] if fragment in c["name"])
            case["checks"] = [c for c in case["checks"] if fragment not in c["name"]]

    mutate(artifact_path(evidence, "run/result.json"), corrupt)
    assert removed
    assert summary.summarize(*evidence)["status"] == "FAIL"


def test_partial_reuse_reference_capability_cannot_be_waived(evidence):
    def corrupt(doc):
        case = next(c for c in doc["cases"] if c["case"] == "TC-02")
        case["measurements"]["engine_partial_prefix_reuse"] = {
            "reuses": False,
            "reference_cached_tokens": [0],
        }

    mutate(artifact_path(evidence, "run/result.json"), corrupt)
    assert summary.summarize(*evidence)["status"] == "FAIL"


def test_tc03_requires_observed_overlap(evidence):
    mutate(
        artifact_path(evidence, "run/result.json"),
        lambda doc: next(c for c in doc["cases"] if c["case"] == "TC-03")[
            "measurements"
        ]["request_overlap"].update(observed=False),
    )
    assert summary.summarize(*evidence)["status"] == "FAIL"


def test_overlap_uses_actual_timestamp_intersection_without_invented_valid_flag(
    evidence,
):
    def change(doc):
        case = next(c for c in doc["cases"] if c["case"] == "TC-03")
        case["measurements"]["request_overlap"] = {
            "observed": True,
            "overlap_seconds": 0.01,
            "requests": [
                {
                    "stage": "hinted",
                    "start": "2026-11-25T10:00:00Z",
                    "end": "2026-11-25T10:00:01Z",
                },
                {
                    "stage": "isolated",
                    "start": "2026-11-25T10:00:00.99Z",
                    "end": "2026-11-25T10:00:02Z",
                },
            ],
        }

    mutate(artifact_path(evidence, "run/result.json"), change)
    result = summary.summarize(*evidence)
    assert result["status"] == "PASS", result["errors"]


def test_worker_role_and_launch_gpu_are_verified(evidence):
    path = artifact_path(evidence, "run/stack/worker-1.launch.json")
    mutate(
        path, lambda doc: doc["environment_overrides"].update(CUDA_VISIBLE_DEVICES="1")
    )
    assert summary.summarize(*evidence)["status"] == "FAIL"


def test_reference_must_have_actual_bypass_launch(evidence):
    source = artifact_path(evidence, "run/stack/worker-0.launch.json")
    target = artifact_path(evidence, "run/stack/worker-0-reference.launch.json")
    target.write_text(source.read_text())
    assert summary.summarize(*evidence)["status"] == "FAIL"


def test_required_observed_devices_have_distinct_uuids(evidence):
    mutate(
        artifact_path(evidence),
        lambda doc: doc["gpu_observation"]["allocated_devices"][1].update(uuid="GPU-0"),
    )
    assert summary.summarize(*evidence)["status"] == "FAIL"


@pytest.mark.parametrize("job", ["plan", "profiles"])
@pytest.mark.parametrize("state", ["cancelled", "failure", "skipped"])
def test_actions_failure_is_fatal(evidence, job, state):
    evidence[2][job]["result"] = state
    assert summary.summarize(*evidence)["status"] == "FAIL"


def test_unexpected_termination_is_reported_without_claiming_graceful(evidence):
    mutate(
        artifact_path(evidence, "run/stack/cleanup.json"),
        lambda doc: doc.update(
            warnings=[
                {
                    "name": "worker-1",
                    "classification": "escalated_kill",
                    "exit_code": -9,
                }
            ]
        ),
    )
    result = summary.summarize(*evidence)
    assert result["status"] == "PASS", result["errors"]
    assert result["warnings"]


def test_runtime_path_cleanup_failure_is_fatal(evidence):
    mutate(
        artifact_path(evidence, "run/stack/cleanup.json"),
        lambda doc: doc["owned_paths"].append(
            {"path": "/tmp/owned", "status": "FAIL", "errors": ["not removed"]}
        ),
    )
    assert summary.summarize(*evidence)["status"] == "FAIL"


@pytest.mark.parametrize(
    "cleanup",
    [
        {"status": "PASS", "owned_processes": []},
        {
            "schema_version": 2,
            "status": "PASS",
            "reclaimed": True,
            "owned_processes": [],
            "process_states": [],
            "owned_paths": [],
        },
    ],
)
def test_reduced_cleanup_receipt_cannot_claim_reclamation(evidence, cleanup):
    write(artifact_path(evidence, "run/stack/cleanup.json"), cleanup)
    assert summary.summarize(*evidence)["status"] == "FAIL"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 1),
        ("reclaimed", 1),
        ("owned_processes", []),
        ("process_states", []),
        ("owned_paths", []),
        ("owned_processes", [{"name": "worker-0", "pid": True, "exit_code": 0}]),
        ("owned_processes", [{"name": "worker-0", "pid": 1000}]),
        ("process_states", [{"name": "worker-1", "pid": 1001, "stopped": True}]),
        ("process_states", [{"name": "worker-0", "pid": 1000, "stopped": 1}]),
        (
            "owned_paths",
            [{"path": "/tmp/owned", "status": "PASS", "errors": ["permission denied"]}],
        ),
    ],
)
def test_cleanup_ledger_is_complete_and_typed(evidence, field, value):
    mutate(
        artifact_path(evidence, "run/stack/cleanup.json"),
        lambda doc: doc.update({field: value}),
    )
    assert summary.summarize(*evidence)["status"] == "FAIL"


@pytest.mark.parametrize("stage", ["before_ports", "after_ports"])
def test_port_collision_or_unobserved_release_fails(evidence, stage):
    mutate(
        artifact_path(evidence),
        lambda doc: doc["resource_observations"][stage].update(
            status="COLLISION", unavailable_ports=[8000]
        ),
    )
    assert summary.summarize(*evidence)["status"] == "FAIL"


def test_legacy_schema_requires_explicit_replay(evidence, tmp_path):
    manifest, root, jobs = evidence
    manifest["schema_version"] = 1
    write(tmp_path / "manifest.json", manifest)
    result = subprocess.run(
        [
            sys.executable,
            str(E2E / "summarize_ci.py"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--artifacts",
            str(root),
            "--job-results",
            json.dumps(jobs),
            "--output",
            str(tmp_path / "report.json"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert (
        "legacy"
        in json.loads((tmp_path / "report.json").read_text())["errors"][0].lower()
    )


def test_adapter_exposes_native_validation_hook():
    assert callable(
        getattr(
            importlib.import_module("ci_integrations.vllm"), "validate_result", None
        )
    )


@pytest.mark.parametrize("worker", ["worker-0", "worker-1", "worker-0-replacement"])
@pytest.mark.parametrize("kind", ["missing", "coarse", "invalid", "mixed", "truncated"])
def test_v2_requires_observed_precise_kvcr_metrics(evidence, worker, kind):
    path = artifact_path(evidence, f"run/stack/{worker}.log")
    good = path.read_text()
    coarse = good.split(" ", 1)[1]
    if kind == "missing":
        path.unlink()
    elif kind == "coarse":
        path.write_text(coarse)
    elif kind == "invalid":
        path.write_text(good.replace("1795600800.250000000", "nan"))
    elif kind == "truncated":
        path.write_text(good + good.rstrip("\n"))
    else:
        path.write_text(good + coarse)
    result = summary.summarize(*evidence)
    assert result["status"] == "FAIL", result
    assert any("precision" in error or "precise" in error for error in result["errors"])


@pytest.mark.parametrize(
    "worker",
    [
        "worker-0",
        "worker-1",
        "worker-2",
        "worker-0-reference",
        "worker-2-reference",
        "worker-0-replacement",
    ],
)
def test_v2_requires_precise_logging_launch_contract(evidence, worker):
    mutate(
        artifact_path(evidence, f"run/stack/{worker}.launch.json"),
        lambda doc: doc["environment_overrides"].pop("VLLM_LOGGING_PREFIX"),
    )
    assert summary.summarize(*evidence)["status"] == "FAIL"


def test_v2_reference_and_nixl_decoder_need_no_kvcr_metric_samples(evidence):
    for name in ("worker-0-reference", "worker-2-reference", "worker-2"):
        artifact_path(evidence, f"run/stack/{name}.log").write_text(
            "(MainProcess pid=1) INFO 11-25 10:00:00 [metrics.py:108] "
            "KV Transfer metrics: Num successful transfers=1, Num failed transfers=0\n"
        )
    result = summary.summarize(*evidence)
    assert result["status"] == "PASS", result["errors"]
