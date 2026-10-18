# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared result types, token comparison, and evidence writers."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BLOCK_SIZE = 64
PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class ValidationError(RuntimeError):
    """Observed behaviour or evidence does not satisfy the test plan."""


class BlockedError(RuntimeError):
    """The combination cannot run here; record the missing condition, not FAIL."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def save_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class TokenSample:
    """Generated token IDs and finish reason captured from one response."""

    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    finish_reason: str | None

    @property
    def length(self) -> int:
        return len(self.completion_token_ids)


def extract_sample(body: dict[str, Any]) -> TokenSample:
    """Read token IDs from a Dynamo completions response.

    The plan requires token IDs captured from the engine layer or, through the
    frontend, a response field carrying them. Absence is BLOCKED, not a
    comparison failure.
    """
    nvext = body.get("nvext") or {}
    completion = nvext.get("completion_token_ids")
    prompt = nvext.get("prompt_token_ids")
    if not isinstance(completion, list) or not all(
        type(token) is int for token in completion
    ):
        raise BlockedError(
            "response carries no nvext.completion_token_ids; token capture through "
            "the Dynamo frontend is unavailable on this build"
        )
    if not isinstance(prompt, list) or not all(type(token) is int for token in prompt):
        raise BlockedError("response carries no nvext.prompt_token_ids")
    choices = body.get("choices") or []
    finish = choices[0].get("finish_reason") if choices else None
    return TokenSample(tuple(prompt), tuple(completion), finish)


def compare_tokens(reference: TokenSample, observed: TokenSample) -> list[str]:
    """Return the list of mismatches; empty means identical.

    Exactly two outcomes exist in the plan: identical passes, everything else
    is an error with no distinction by cause. The messages are diagnostics.
    """
    problems = []
    if observed.prompt_token_ids != reference.prompt_token_ids:
        problems.append("effective prompt token IDs differ from the reference")
    if observed.length != reference.length:
        problems.append(
            f"generated length {observed.length} != reference {reference.length}"
        )
    if observed.completion_token_ids != reference.completion_token_ids:
        first = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(reference.completion_token_ids, observed.completion_token_ids)
                )
                if left != right
            ),
            min(reference.length, observed.length),
        )
        problems.append(f"generated token IDs diverge at position {first}")
    if observed.finish_reason != reference.finish_reason:
        problems.append(
            f"finish reason {observed.finish_reason!r} != "
            f"reference {reference.finish_reason!r}"
        )
    return problems


def cached_tokens(body: dict[str, Any]) -> int | None:
    details = (body.get("usage") or {}).get("prompt_tokens_details") or {}
    value = details.get("cached_tokens")
    return value if type(value) is int else None


def worker_identity(
    body: dict[str, Any], role: str = "decode"
) -> tuple[str | None, int | None]:
    identity = (body.get("nvext") or {}).get("worker_id") or {}
    worker = identity.get(f"{role}_worker_id")
    rank = identity.get("decode_dp_rank")
    return (
        str(worker) if type(worker) in (str, int) and str(worker) else None,
        rank if type(rank) is int else None,
    )


def expected_hit_blocks(
    prompt_tokens: int, shared_prefix_tokens: int, block: int = BLOCK_SIZE
) -> int:
    """Upper bound of complete blocks a prefix-cache hit can cover.

    vLLM never reuses the block containing the last prompt token, so a full
    hit covers floor((n - 1) / block) blocks of an n-token prompt, capped by the
    complete blocks inside the shared prefix.
    """
    if prompt_tokens <= 0 or shared_prefix_tokens <= 0:
        return 0
    return min((prompt_tokens - 1) // block, shared_prefix_tokens // block)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    case: str
    title: str
    requirement: str
    status: str = "RUNNING"
    checks: list[Check] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    error: str | None = None
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    measurements: dict[str, Any] = field(default_factory=dict)

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append(Check(name, bool(passed), detail))
        return bool(passed)

    def finish(self) -> None:
        self.finished_at = utc_now()
        if self.status == "RUNNING":
            self.status = PASS if all(check.passed for check in self.checks) else FAIL
            if not self.checks:
                self.status = FAIL
                self.error = self.error or "no checks were recorded"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_junit(output: Path, run: dict[str, Any]) -> None:
    cases = run.get("cases", [])
    suite = ET.Element(
        "testsuite",
        name="kvcr-phase1",
        tests=str(len(cases)),
        failures=str(sum(case["status"] == FAIL for case in cases)),
        skipped=str(sum(case["status"] == BLOCKED for case in cases)),
        errors="0",
        timestamp=run.get("started_at", ""),
    )
    for case in cases:
        element = ET.SubElement(
            suite,
            "testcase",
            classname="kvcr.phase1",
            name=f"{case['case']} {case['title']}",
        )
        lines = [
            f"[{'ok' if check['passed'] else 'FAIL'}] "
            f"{check['name']}: {check['detail']}"
            for check in case["checks"]
        ]
        if case.get("error"):
            lines.append(f"error: {case['error']}")
        text = "\n".join(lines)
        if case["status"] == FAIL:
            ET.SubElement(
                element, "failure", message=case.get("error") or "FAIL"
            ).text = text
        elif case["status"] == BLOCKED:
            ET.SubElement(
                element, "skipped", message=case.get("error") or "BLOCKED"
            ).text = text
        else:
            ET.SubElement(element, "system-out").text = text
    ET.ElementTree(suite).write(
        output / "junit.xml", encoding="unicode", xml_declaration=True
    )


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, int(round(fraction * len(ordered) + 0.5)))
    return ordered[min(rank, len(ordered)) - 1]


def summarize(values: list[float]) -> dict[str, Any]:
    """n / median / p95 / min / max / mean, rounded, for a latency series."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0}
    ordered = sorted(clean)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "n": len(ordered),
        "median": round(median, 6),
        "p95": round(percentile(ordered, 0.95) or 0.0, 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "mean": round(sum(ordered) / len(ordered), 6),
    }
