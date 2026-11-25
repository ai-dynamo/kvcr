# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial CPU checks of the configuration-matrix verdict."""

import copy
import importlib
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "e2e"))

_SPEC = importlib.util.spec_from_file_location(
    "kvcr_ci_summary", Path(__file__).parents[1] / "e2e" / "summarize_ci.py"
)
assert _SPEC is not None and _SPEC.loader is not None
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)

# These synthetic v1 fixtures exercise the historical audit contract only.
legacy_summarize = partial(summary.summarize, legacy_replay=True)


@pytest.fixture
def evidence(tmp_path):
    profile = {
        "id": "vllm-dense-agg-guard",
        "cases": ["TC-01", "TC-02", "TC-03", "TC-04"],
        "fingerprint": "f" * 64,
        "integration": "vllm",
        "model_family": "dense",
        "model_id": "Qwen/Qwen3-0.6B",
        "model_revision": "b" * 40,
        "topology": "agg",
        "mode": "guard",
        "gpus": [0, 1],
    }
    smoke = {
        **profile,
        "id": "vllm-dense-peer-transfer-smoke",
        "cases": [],
        "topology": "peer-transfer",
        "mode": "native",
    }
    manifest = {
        "schema_version": 1,
        "run_id": "123",
        "run_attempt": "2",
        "source_sha": "a" * 40,
        "workflow_sha": "d" * 40,
        "image": "registry/kvcr@sha256:" + "c" * 64,
        "selection": "diagnostic",
        "profiles": [profile],
        "smoke": smoke,
    }
    for expected in (profile, smoke):
        directory = tmp_path / expected["id"]
        (directory / "run").mkdir(parents=True)
        envelope = {
            key: manifest[key]
            for key in (
                "schema_version",
                "run_id",
                "run_attempt",
                "source_sha",
                "workflow_sha",
                "image",
            )
        }
        envelope.update(
            profile_id=expected["id"],
            profile_fingerprint=expected["fingerprint"],
            status="PASS",
            diagnostic=False,
            installed_source_sha=manifest["source_sha"],
            actual_gpus=expected["gpus"],
            harness_sha256="e" * 64,
            **{
                key: expected[key]
                for key in (
                    "cases",
                    "integration",
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
        (directory / "profile-result.json").write_text(json.dumps(envelope))
        result = (
            {
                "status": "PASS",
                "cases": [
                    {
                        "case": case,
                        "status": "PASS",
                        "checks": [
                            {"name": "token IDs and cache path", "passed": True}
                        ],
                    }
                    for case in expected["cases"]
                ],
                "missing_cases": [],
                "reference": {
                    "kvcr_disabled_in_launch": True,
                    "unstable": {},
                    "prompts": {"seed": {"stable": True}},
                },
            }
            if expected["cases"]
            else {
                "status": "success",
                "cached_tokens": 64,
                "metrics": {"transfer/success": 1, "remote_deliver/success": 1},
            }
        )
        (directory / "run" / "result.json").write_text(json.dumps(result))
        if expected["cases"]:
            (directory / "run" / "stack").mkdir()
            (directory / "run" / "stack" / "cleanup.json").write_text(
                json.dumps({"owned_processes": []})
            )
            (directory / "run" / "provenance.json").write_text(
                json.dumps(
                    {
                        "harness_sha256": envelope["harness_sha256"],
                    }
                )
            )
        else:
            (directory / "run" / "cleanup.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "owned_processes": [],
                    }
                )
            )
    jobs = {
        name: {"result": "success"} for name in ("plan", "peer-transfer", "profiles")
    }
    return manifest, tmp_path, jobs


def mutate_json(path, change):
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


def test_complete_selected_set_passes_and_is_labelled_diagnostic(evidence):
    report = legacy_summarize(*evidence)
    assert report["status"] == "PASS"
    assert report["selection"] == "diagnostic"
    assert len(report["profiles"]) == 2
    assert report["errors"] == []


@pytest.mark.parametrize("job", ["plan", "peer-transfer", "profiles"])
@pytest.mark.parametrize("result", ["failure", "skipped", "cancelled", ""])
def test_pass_artifacts_cannot_override_failed_or_absent_job(evidence, job, result):
    evidence[2][job]["result"] = result
    assert legacy_summarize(*evidence)["status"] == "FAIL"


def test_missing_dependency_is_fatal(evidence):
    del evidence[2]["profiles"]
    assert legacy_summarize(*evidence)["status"] == "FAIL"


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "124"),
        ("run_attempt", "1"),
        ("image", "unrelated:latest"),
        ("source_sha", "d" * 40),
        ("workflow_sha", "0" * 40),
        ("installed_source_sha", "0" * 40),
        ("actual_gpus", [0, 0]),
        ("diagnostic", True),
        ("profile_fingerprint", "0" * 64),
        ("model_revision", "e" * 40),
        ("topology", "disagg"),
        ("mode", "nonguard"),
        ("status", "BLOCKED"),
        ("schema_version", 99),
    ],
)
def test_stale_or_wrong_identity_is_rejected(evidence, field, value):
    path = evidence[1] / evidence[0]["profiles"][0]["id"] / "profile-result.json"
    mutate_json(path, lambda document: document.update({field: value}))
    assert legacy_summarize(*evidence)["status"] == "FAIL"


@pytest.mark.parametrize("stage", ["setup", "harness", "cleanup"])
def test_stage_failure_is_fatal_even_if_outer_status_passes(evidence, stage):
    path = evidence[1] / evidence[0]["profiles"][0]["id"] / "profile-result.json"
    mutate_json(path, lambda document: document[stage].update(status="FAIL"))
    assert legacy_summarize(*evidence)["status"] == "FAIL"


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "blocked", "cleanup", "reference"]
)
def test_harness_semantics_are_checked_independently(evidence, failure):
    path = evidence[1] / evidence[0]["profiles"][0]["id"] / "run" / "result.json"

    def corrupt(document):
        if failure == "missing":
            document["cases"].pop()
        elif failure == "duplicate":
            document["cases"].append(copy.deepcopy(document["cases"][0]))
        elif failure == "blocked":
            document["cases"][0]["status"] = "BLOCKED"
        elif failure == "cleanup":
            document["cleanup_error"] = "worker still alive"
        else:
            document["reference"]["kvcr_disabled_in_launch"] = False

    mutate_json(path, corrupt)
    assert legacy_summarize(*evidence)["status"] == "FAIL"


@pytest.mark.parametrize("failure", ["missing", "duplicate", "malformed", "unexpected"])
def test_missing_duplicate_malformed_or_unexpected_artifacts_fail(evidence, failure):
    root = evidence[1]
    path = root / evidence[0]["profiles"][0]["id"] / "profile-result.json"
    if failure == "missing":
        path.unlink()
    elif failure == "malformed":
        path.write_text("{broken")
    else:
        extra = root / "another-artifact"
        extra.mkdir()
        document = json.loads(path.read_text())
        if failure == "unexpected":
            document["profile_id"] = "vllm-unselected-agg-guard"
        (extra / "profile-result.json").write_text(json.dumps(document))
    assert legacy_summarize(*evidence)["status"] == "FAIL"


def test_empty_accidental_matrix_is_not_success(evidence):
    evidence[0]["profiles"] = []
    assert legacy_summarize(*evidence)["status"] == "FAIL"


def test_full_claim_requires_twelve_configuration_profiles(evidence):
    evidence[0]["selection"] = "all"
    assert legacy_summarize(*evidence)["status"] == "FAIL"


def test_smoke_requires_positive_real_transfer_evidence(evidence):
    path = evidence[1] / evidence[0]["smoke"]["id"] / "run" / "result.json"
    mutate_json(
        path, lambda document: document["metrics"].update({"transfer/success": 0})
    )
    assert legacy_summarize(*evidence)["status"] == "FAIL"


def test_reference_provenance_matches_wrapper_harness_identity(evidence):
    path = evidence[1] / evidence[0]["profiles"][0]["id"] / "run" / "provenance.json"
    mutate_json(path, lambda document: document.update(harness_sha256="0" * 64))
    assert legacy_summarize(*evidence)["status"] == "FAIL"


def test_false_check_cannot_hide_behind_case_pass(evidence):
    path = evidence[1] / evidence[0]["profiles"][0]["id"] / "run" / "result.json"
    mutate_json(
        path, lambda document: document["cases"][0]["checks"][0].update(passed=False)
    )
    assert legacy_summarize(*evidence)["status"] == "FAIL"


def test_smoke_cleanup_failure_cannot_hide_behind_success(evidence):
    path = evidence[1] / evidence[0]["smoke"]["id"] / "run" / "cleanup.json"
    mutate_json(path, lambda document: document.update(status="FAIL"))
    assert legacy_summarize(*evidence)["status"] == "FAIL"


def test_native_cleanup_failure_cannot_hide_behind_case_success(evidence):
    path = (
        evidence[1]
        / evidence[0]["profiles"][0]["id"]
        / "run"
        / "stack"
        / "cleanup.json"
    )
    mutate_json(
        path,
        lambda document: document.update(
            owned_processes=[{"pid": 123, "error": "survived SIGKILL"}]
        ),
    )
    assert legacy_summarize(*evidence)["status"] == "FAIL"


def test_catalog_and_summary_agree_on_the_complete_contract(evidence, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "e2e"))
    catalog = importlib.import_module("ci_profiles")
    old_manifest, root, jobs = evidence
    manifest = catalog.make_manifest(
        catalog.load_catalog(),
        "all",
        old_manifest["image"],
        old_manifest["source_sha"],
        old_manifest["run_id"],
        old_manifest["run_attempt"],
    )
    # Replay legacy envelopes using the current catalog identity inventory.
    manifest["schema_version"] = 1
    manifest["smoke"] = copy.deepcopy(old_manifest["smoke"])
    manifest["workflow_sha"] = old_manifest["workflow_sha"]
    source = root / old_manifest["profiles"][0]["id"]
    template = json.loads((source / "profile-result.json").read_text())
    harness = json.loads((source / "run" / "result.json").read_text())
    for profile in manifest["profiles"]:
        directory = root / profile["id"]
        (directory / "run").mkdir(parents=True, exist_ok=True)
        (directory / "run" / "stack").mkdir(exist_ok=True)
        (directory / "run" / "stack" / "cleanup.json").write_text(
            json.dumps({"owned_processes": []})
        )
        result = copy.deepcopy(template)
        for field in (
            "integration",
            "model_family",
            "model_id",
            "model_revision",
            "topology",
            "mode",
            "cases",
        ):
            result[field] = profile[field]
        result.update(
            profile_id=profile["id"],
            profile_fingerprint=profile["fingerprint"],
            actual_gpus=profile["gpus"],
        )
        (directory / "profile-result.json").write_text(json.dumps(result))
        native = copy.deepcopy(harness)
        native["cases"] = [
            {
                "case": case,
                "status": "PASS",
                "checks": [{"name": "tokens", "passed": True}],
            }
            for case in profile["cases"]
        ]
        (directory / "run" / "result.json").write_text(json.dumps(native))
        (directory / "run" / "provenance.json").write_text(
            json.dumps({"harness_sha256": result["harness_sha256"]})
        )
    smoke = manifest["smoke"]
    smoke_result = root / smoke["id"] / "profile-result.json"

    def align_smoke(document):
        document.update(
            {
                field: smoke[field]
                for field in (
                    "integration",
                    "model_family",
                    "model_id",
                    "model_revision",
                    "topology",
                    "mode",
                    "cases",
                )
            }
        )
        document.update(
            profile_fingerprint=smoke["fingerprint"], actual_gpus=smoke["gpus"]
        )

    mutate_json(smoke_result, align_smoke)
    report = legacy_summarize(manifest, root, jobs)
    assert report["status"] == "PASS", report["errors"]
    assert len(report["profiles"]) == 13
    assert sum(len(row["cases"]) for row in report["profiles"]) == 42


def test_native_expected_profile_cannot_claim_guard_case(evidence):
    evidence[0]["profiles"][0]["mode"] = "native"
    assert legacy_summarize(*evidence)["status"] == "FAIL"
