# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The normal root gate requires full coverage from every active integration."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ci_integrations import get_integration


def load_lanes() -> dict:
    value = json.loads(Path(__file__).with_name("ci_lanes.json").read_text())
    if (
        value.get("schema_version") != 1
        or not isinstance(value.get("integrations"), dict)
        or not value["integrations"]
    ):
        raise ValueError("missing active integration policy")
    for name, lane in value["integrations"].items():
        get_integration(name)
        if set(lane) != {"build_job", "test_job"} or any(
            not isinstance(job, str)
            or not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_-]*", job)
            for job in lane.values()
        ):
            raise ValueError("invalid active integration job contract")
    return value["integrations"]


def assess(lanes: dict, jobs: dict, selection: str) -> dict:
    errors = []
    if selection != "all":
        errors.append("Diagnostic subset does not qualify the normal full GPU E2E gate")
    if not lanes:
        errors.append("No active integration policy")
    if (
        not isinstance(jobs.get("resolve"), dict)
        or jobs["resolve"].get("result") != "success"
    ):
        errors.append("Build input resolution did not pass")
    for name, lane in lanes.items():
        for role, job_id in lane.items():
            job = jobs.get(job_id)
            if not isinstance(job, dict) or job.get("result") != "success":
                errors.append(f"{name}: required {role} {job_id} did not pass")
    return {
        "schema_version": 2,
        "status": "FAIL" if errors else "PASS",
        "selection": selection,
        "integrations": list(lanes),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-results", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads(args.job_results)
    if not isinstance(jobs, dict):
        raise ValueError("job results must be an object")
    result = assess(load_lanes(), jobs, args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
