#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate, render and query the per-PR CI coverage matrix (ci_matrix.json).

python tests/e2e/ci_matrix.py render           # rewrite docs/ci-matrix.md
python tests/e2e/ci_matrix.py render --check   # fail if it is stale
python tests/e2e/ci_matrix.py list --lane vllm --status pending
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MATRIX = Path(__file__).with_suffix(".json")
RENDERED = REPO / "docs" / "ci-matrix.md"
RENDER_COMMAND = "python tests/e2e/ci_matrix.py render"

ANY = "any"
NONE = "none"
# Statuses whose entries may use `any` in place of a concrete value.
WILDCARD_STATUSES = {"not_applicable", "skipped"}
# Entry field -> the top-level table its value must be a key of.
DIMENSIONS = {
    "lane": "lanes",
    "model_profile": "model_profiles",
    "mode": "modes",
    "topology": "topologies",
}
REQUIRED = ("id", *DIMENSIONS, "cases", "status")
# Fixed vocabularies the rules below depend on: top-level table -> its keys.
# The JSON describes each value; validate() keeps the two in step.
FIXED = {
    "placeholders": {ANY, NONE},
    "job_sources": {"main", "candidate"},
    "evidence_kinds": {"ci", "manual"},
}
# JSON has no comments, so the rules validate() enforces are stated here and
# printed on the rendered page.
EDITING_RULES = (
    "`automated` is only for a job defined on main. The entry needs `job` with"
    " `source: main`, job names that exist in the workflow, and at least one"
    " `evidence` item of kind `ci`.",
    "A job that exists only on an unmerged PR is `pending`. It may name that job"
    " with `job.source: candidate` and `job.ref` naming the PR, or leave `job`"
    " out while the PR is still changing.",
    "An `automated` entry belongs to an enabled lane: test selection runs enabled"
    " lanes only, so the matrix cannot claim a job that is never selected.",
    "`pending` needs a `reason` and an `owner` or an `issue`. Owner and issue are"
    " empty for now and will be filled in.",
    "`not_applicable` and `skipped` need a `reason`.",
    "There is no manual-only status. A manual run is `evidence` of kind `manual`"
    " on a pending entry; it never makes the entry automated.",
    "`any` (in lane, model_profile, mode or topology) is allowed only on"
    " `not_applicable` and `skipped` entries. `none` is for entries that use no"
    " model or GPU.",
    "`model_profile` names a key of `model_profiles`, so a model changes in one place.",
    "A combination with no entry is not claimed; this is not a full Cartesian product.",
)


def load(path: Path = MATRIX) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _workflow_jobs(path: Path) -> set[str]:
    """Job ids of a workflow file: the two-space keys under `jobs:` (no YAML parser)."""
    text = path.read_text(encoding="utf-8")
    _, _, body = text.partition("\njobs:\n")
    return set(re.findall(r"^  ([\w-]+):", body, flags=re.MULTILINE))


def validate(matrix: dict) -> list[str]:
    """Return every violation of the editing rules (EDITING_RULES)."""
    errors = []
    for table, names in FIXED.items():
        if set(matrix[table]) != names:
            errors.append(f"{table} must define exactly {', '.join(sorted(names))}")
    seen = set()
    for index, entry in enumerate(matrix["entries"]):
        name = entry.get("id", f"entries[{index}]")
        missing = [key for key in REQUIRED if key not in entry]
        if missing:
            errors.append(f"{name}: missing {', '.join(missing)}")
            continue
        if name in seen:
            errors.append(f"{name}: duplicate id")
        seen.add(name)

        status = entry["status"]
        if status not in matrix["statuses"]:
            errors.append(f"{name}: unknown status {status!r}")
        for field, table in DIMENSIONS.items():
            value = entry[field]
            if value == ANY:
                if status not in WILDCARD_STATUSES:
                    errors.append(f"{name}: {field} `any` is not allowed on {status}")
            elif value == NONE:
                if field == "lane":
                    errors.append(f"{name}: lane cannot be `none`")
            elif value not in matrix[table]:
                errors.append(f"{name}: unknown {field} {value!r}")
        for case in entry["cases"]:
            if case not in matrix["test_cases"]:
                errors.append(f"{name}: unknown case {case!r}")

        job = entry.get("job")
        if job:
            if job.get("source") not in FIXED["job_sources"]:
                errors.append(f"{name}: job.source must be main or candidate")
            if not job.get("workflow") or not job.get("jobs"):
                errors.append(f"{name}: job needs workflow and jobs")
            elif not (REPO / job["workflow"]).is_file():
                errors.append(f"{name}: no such workflow {job['workflow']}")
            elif job.get("source") == "main":
                defined = _workflow_jobs(REPO / job["workflow"])
                for missing_job in (j for j in job["jobs"] if j not in defined):
                    errors.append(
                        f"{name}: no job {missing_job!r} in {job['workflow']}"
                    )
            if job.get("source") == "candidate" and not job.get("ref"):
                errors.append(f"{name}: a candidate job needs job.ref")
        for item in entry.get("evidence", []):
            if item.get("kind") not in FIXED["evidence_kinds"]:
                errors.append(f"{name}: evidence kind must be ci or manual")
            if not item.get("url") or not item.get("checked"):
                errors.append(f"{name}: evidence needs url and checked")

        on_main = bool(job) and job.get("source") == "main"
        if status == "automated":
            if not on_main:
                errors.append(f"{name}: automated needs a job with source main")
            if not any(e.get("kind") == "ci" for e in entry.get("evidence", [])):
                errors.append(f"{name}: automated needs ci evidence")
            lane = matrix["lanes"].get(entry["lane"], {})
            if not lane.get("enabled"):
                errors.append(f"{name}: automated needs an enabled lane")
        else:
            if on_main:
                errors.append(f"{name}: has a job on main but is {status}")
            if not entry.get("reason"):
                errors.append(f"{name}: {status} needs a reason")
        # Both may be empty for now; they are filled in once the owners are agreed.
        if status == "pending" and not ("owner" in entry or "issue" in entry):
            errors.append(f"{name}: pending needs an owner or an issue")
    return errors


def select(matrix: dict, **filters: str | None) -> list[dict]:
    """Entries matching every given filter; an entry's `any` matches all values."""
    entries = matrix["entries"]
    for field, wanted in filters.items():
        if wanted is None:
            continue
        entries = [
            e
            for e in entries
            if e[field] == wanted or (field in DIMENSIONS and e[field] == ANY)
        ]
    return entries


# -- rendering -----------------------------------------------------------------


def _cell(text: object) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _model(matrix: dict, profile: str) -> str:
    if profile in (ANY, NONE):
        return profile
    model = matrix["model_profiles"][profile]["model"]
    return f"{profile} ({model or 'model not chosen'})"


def _job(entry: dict) -> str:
    job = entry.get("job")
    if not job:
        return "none"
    workflow = Path(job["workflow"]).name
    names = ", ".join(f"`{name}`" for name in job["jobs"])
    if job["source"] == "main":
        return f"{names} in [{workflow}](../{job['workflow']})"
    return f"{names} in {workflow}, only on {job['ref']}"


def _details(entry: dict) -> str:
    parts = [entry[key] for key in ("reason", "notes") if entry.get(key)]
    for item in entry.get("evidence", []):
        when = item.get("date") or f"checked {item['checked']}"
        label = "CI run" if item["kind"] == "ci" else "Manual run"
        note = f": {item['note']}" if item.get("note") else ""
        parts.append(f"[{label}, {when}]({item['url']}){note}")
    return "<br>".join(_cell(part) for part in parts)


def _owner(entry: dict) -> str:
    return ", ".join(entry[k] for k in ("owner", "issue") if entry.get(k)) or ""


def render(matrix: dict) -> str:
    entries = matrix["entries"]
    statuses = list(matrix["statuses"])
    lanes = [*matrix["lanes"], ANY]
    out = [
        "<!--",
        "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES."
        " All rights reserved.",
        "SPDX-License-Identifier: Apache-2.0",
        "-->",
        "",
        "# Per-PR CI coverage matrix",
        "",
        "Generated from [tests/e2e/ci_matrix.json](../tests/e2e/ci_matrix.json)."
        f" Do not edit this file: change the JSON and run `{RENDER_COMMAND}`.",
        "",
        "This is functional configuration coverage, not code-coverage percentage. A"
        " combination that is not listed is not claimed.",
        "",
        "## Status meanings",
        "",
        *_table(
            ["Status", "Meaning"], [[f"`{s}`", matrix["statuses"][s]] for s in statuses]
        ),
        "",
        "## Summary",
        "",
    ]
    counts = Counter((e["lane"], e["status"]) for e in entries)
    out += _table(
        ["Lane", "Selectable per PR", *statuses],
        [
            [
                "all lanes" if lane == ANY else lane,
                ""
                if lane == ANY
                else ("yes" if matrix["lanes"][lane]["enabled"] else "no"),
                *[str(counts[lane, s] or "") for s in statuses],
            ]
            for lane in lanes
            if any(e["lane"] == lane for e in entries)
        ],
    )
    for lane in lanes:
        rows = [e for e in entries if e["lane"] == lane]
        if not rows:
            continue
        out += ["", f"## {'All lanes' if lane == ANY else lane}", ""]
        if lane != ANY:
            out += [matrix["lanes"][lane]["summary"], ""]
        out += _table(
            [
                "Entry",
                "Model",
                "Mode",
                "Topology",
                "Cases",
                "Status",
                "Job",
                "Owner",
                "Details",
            ],
            [
                [
                    f"`{e['id']}`",
                    _model(matrix, e["model_profile"]),
                    e["mode"],
                    e["topology"],
                    ", ".join(e["cases"]) or "none",
                    f"`{e['status']}`",
                    _job(e),
                    _owner(e),
                    _details(e),
                ]
                for e in rows
            ],
        )
    out += ["", "## Test cases", ""]
    out += _table(
        ["Case", "What it checks", "Needs GPU"],
        [
            [name, case["summary"], "yes" if case["gpu"] else "no"]
            for name, case in matrix["test_cases"].items()
        ],
    )
    out += ["", "## Modes and topologies", ""]
    out += _table(
        ["Name", "Meaning"],
        [[f"`{k}`", v] for k, v in {**matrix["modes"], **matrix["topologies"]}.items()],
    )
    for title, table in (
        ("Placeholders", "placeholders"),
        ("Job sources", "job_sources"),
        ("Evidence kinds", "evidence_kinds"),
    ):
        out += ["", f"## {title}", ""]
        out += _table(
            ["Name", "Meaning"], [[f"`{k}`", v] for k, v in matrix[table].items()]
        )
    out += ["", "## Editing rules", ""]
    out += [f"- {rule}" for rule in EDITING_RULES]
    return "\n".join(out) + "\n"


# -- command line --------------------------------------------------------------


def _render_command(args: argparse.Namespace, matrix: dict) -> int:
    text = render(matrix)
    if not args.check:
        RENDERED.write_text(text, encoding="utf-8")
        print(f"wrote {RENDERED.relative_to(REPO)}")
        return 0
    if RENDERED.is_file() and RENDERED.read_text(encoding="utf-8") == text:
        return 0
    print(
        f"{RENDERED.relative_to(REPO)} is stale; run `{RENDER_COMMAND}`",
        file=sys.stderr,
    )
    return 1


def _list_command(args: argparse.Namespace, matrix: dict) -> int:
    entries = select(
        matrix,
        lane=args.lane,
        model_profile=args.model_profile,
        mode=args.mode,
        topology=args.topology,
        status=args.status,
    )
    rows = [
        [e["id"], e["lane"], e["model_profile"], e["mode"], e["topology"],
         ",".join(e["cases"]), e["status"]]
        for e in entries
    ]  # fmt: skip
    rows.insert(0, ["ENTRY", "LANE", "MODEL", "MODE", "TOPOLOGY", "CASES", "STATUS"])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    render_parser = commands.add_parser("render", help="write docs/ci-matrix.md")
    render_parser.add_argument(
        "--check", action="store_true", help="only verify it is current"
    )
    list_parser = commands.add_parser("list", help="print entries matching the filters")
    for flag in ("lane", "model-profile", "mode", "topology", "status"):
        list_parser.add_argument(f"--{flag}")
    args = parser.parse_args(argv)

    matrix = load()
    errors = validate(matrix)
    if errors:
        print(
            "\n".join(f"ci_matrix.json: {error}" for error in errors), file=sys.stderr
        )
        return 1
    return (_render_command if args.command == "render" else _list_command)(
        args, matrix
    )


if __name__ == "__main__":
    sys.exit(main())
