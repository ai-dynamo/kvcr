# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The per-PR coverage matrix follows its rules and its rendered copy is current."""

import copy
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "kvcr_ci_matrix", Path(__file__).parents[1] / "e2e" / "ci_matrix.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ci_matrix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ci_matrix)


@pytest.fixture(scope="module")
def loaded():
    return ci_matrix.load()


@pytest.fixture
def matrix(loaded):
    return copy.deepcopy(loaded)


def entry(matrix, name):
    return next(e for e in matrix["entries"] if e["id"] == name)


def test_matrix_follows_its_rules(loaded):
    assert ci_matrix.validate(loaded) == []


def test_rendered_matrix_is_current(loaded):
    assert ci_matrix.RENDERED.read_text(encoding="utf-8") == ci_matrix.render(loaded), (
        f"docs/ci-matrix.md is stale; run `{ci_matrix.RENDER_COMMAND}`"
    )


def test_automated_needs_a_job_on_main_and_ci_evidence(matrix):
    pending = entry(matrix, "vllm-dense-agg-guard")
    pending["status"] = "automated"
    assert ci_matrix.validate(matrix) == [
        "vllm-dense-agg-guard: automated needs a job with source main",
        "vllm-dense-agg-guard: automated needs ci evidence",
    ]

    smoke = entry(matrix, "vllm-smoke-peer-transfer")
    pending["status"] = "pending"
    smoke["evidence"] = [
        {"kind": "manual", "url": "https://example.com", "checked": "2026-11-24"}
    ]
    assert ci_matrix.validate(matrix) == [
        "vllm-smoke-peer-transfer: automated needs ci evidence"
    ]


def test_a_job_on_main_must_exist_in_its_workflow(matrix):
    entry(matrix, "vllm-smoke-peer-transfer")["job"]["jobs"] = ["phase1"]
    assert ci_matrix.validate(matrix) == [
        "vllm-smoke-peer-transfer: no job 'phase1' in .github/workflows/vllm-e2e.yml"
    ]


def test_automated_needs_an_enabled_lane(matrix):
    matrix["lanes"]["vllm"]["enabled"] = False
    assert ci_matrix.validate(matrix) == [
        "vllm-smoke-peer-transfer: automated needs an enabled lane"
    ]


def test_pending_needs_a_reason_and_an_owner_or_issue(matrix):
    pending = entry(matrix, "vllm-dense-agg-native")
    del pending["reason"], pending["owner"], pending["issue"]
    assert ci_matrix.validate(matrix) == [
        "vllm-dense-agg-native: pending needs a reason",
        "vllm-dense-agg-native: pending needs an owner or an issue",
    ]


def test_any_is_only_for_not_applicable_and_skipped(matrix):
    entry(matrix, "vllm-gdn-agg-native")["topology"] = "any"
    assert ci_matrix.validate(matrix) == [
        "vllm-gdn-agg-native: topology `any` is not allowed on pending"
    ]


def test_fixed_vocabularies_match_what_the_script_accepts(matrix):
    del matrix["placeholders"]["none"]
    matrix["job_sources"]["fork"] = "Not a source the rules know."
    del matrix["evidence_kinds"]["manual"]
    assert ci_matrix.validate(matrix)[:3] == [
        "placeholders must define exactly any, none",
        "job_sources must define exactly candidate, main",
        "evidence_kinds must define exactly ci, manual",
    ]


def test_unknown_names_and_duplicate_ids_are_reported(matrix):
    first, second = matrix["entries"][1:3]
    first["cases"] = ["TC-99"]
    first["model_profile"] = "sparse"
    second["id"] = first["id"]
    assert ci_matrix.validate(matrix) == [
        f"{first['id']}: unknown model_profile 'sparse'",
        f"{first['id']}: unknown case 'TC-99'",
        f"{first['id']}: duplicate id",
    ]


def test_select_filters_and_any_matches_every_value(loaded):
    ids = [
        e["id"] for e in ci_matrix.select(loaded, lane="sglang-hicache", mode="guard")
    ]
    assert ids == ["sglang-hicache-guard", "cross-node", "per-pr-perf"]
    automated = ci_matrix.select(loaded, status="automated")
    assert [e["id"] for e in automated] == [
        "cpu-lint-unit",
        "vllm-smoke-peer-transfer",
    ]


def test_every_target_configuration_has_exactly_one_entry(loaded):
    # The per-PR target: vLLM runs both modes; the SGLang lanes are native only.
    target = {
        (lane, model, topology, mode)
        for lane, modes in (
            ("vllm", ("guard", "native")),
            ("sglang-hicache", ("native",)),
            ("sglang-linker", ("native",)),
        )
        for model in ("dense", "mamba", "gdn")
        for topology in ("agg", "disagg")
        for mode in modes
    }
    assert len(target) == 24
    rows = [
        (e["lane"], e["model_profile"], e["topology"], e["mode"])
        for e in loaded["entries"]
        if e["cases"][:1] == ["TC-01"]
    ]
    assert len(rows) == len(set(rows))
    assert target <= set(rows)
