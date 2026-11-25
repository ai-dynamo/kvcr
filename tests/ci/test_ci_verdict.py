# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Root verdict must require full source-selected integration coverage."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "e2e"))


def test_diagnostic_success_does_not_satisfy_full_root_gate():
    verdict = importlib.import_module("ci_verdict")
    lanes = {"vllm": {"build_job": "build", "test_job": "e2e"}}
    jobs = {name: {"result": "success"} for name in ("resolve", "build", "e2e")}
    assert verdict.assess(lanes, jobs, "all")["status"] == "PASS"
    assert verdict.assess(lanes, jobs, "vllm-dense-agg-native")["status"] == "FAIL"


def test_missing_active_integration_cannot_hide_behind_other_success():
    verdict = importlib.import_module("ci_verdict")
    lanes = {
        "vllm": {"build_job": "build", "test_job": "e2e"},
        "second": {"build_job": "build-second", "test_job": "e2e-second"},
    }
    jobs = {name: {"result": "success"} for name in ("resolve", "build", "e2e")}
    result = verdict.assess(lanes, jobs, "all")
    assert result["status"] == "FAIL"
    assert "second" in str(result["errors"])
