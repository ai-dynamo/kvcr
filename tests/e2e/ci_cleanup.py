# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate Phase 1's explicit owned-resource reclamation receipt.

This contract proves recorded reclamation, not graceful process-tree shutdown
or zero physical GPU memory. Other harnesses select their own adapter policy.
"""

from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath


def validate_phase1_cleanup(report: dict) -> None:
    if (
        not isinstance(report, dict)
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != 2
        or report.get("status") != "PASS"
        or report.get("reclaimed") is not True
    ):
        raise ValueError("Phase 1 cleanup must record schema2 PASS and reclaimed=true")
    for field in ("owned_processes", "process_states", "owned_paths"):
        if not isinstance(report.get(field), list) or not report[field]:
            raise ValueError(f"Phase 1 cleanup requires a nonempty {field} ledger")
    for field in ("owned_processes", "process_states"):
        for item in report[field]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or not item["name"]
                or type(item.get("pid")) is not int
                or item["pid"] <= 0
                or "error" in item
            ):
                raise ValueError(f"Malformed or failed Phase 1 {field} entry")
    owned = report["owned_processes"]
    states = report["process_states"]
    if any(type(item.get("exit_code")) is not int for item in owned):
        raise ValueError("Phase 1 cleanup must record every owned leader's exit code")
    if any(item.get("stopped") is not True for item in states):
        raise ValueError("Phase 1 cleanup reports an unreclaimed process")
    owned_ids = Counter((item["name"], item["pid"]) for item in owned)
    state_ids = Counter((item["name"], item["pid"]) for item in states)
    if owned_ids != state_ids or any(count != 1 for count in owned_ids.values()):
        raise ValueError("Phase 1 cleanup process outcome/state ledgers differ")
    paths = []
    for item in report["owned_paths"]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not PurePosixPath(item["path"]).is_absolute()
            or item.get("status") != "PASS"
            or item.get("errors", []) != []
        ):
            raise ValueError("Phase 1 cleanup reports a malformed or unreclaimed path")
        paths.append(item["path"])
    if len(paths) != len(set(paths)):
        raise ValueError("Phase 1 cleanup repeats an owned runtime path")
