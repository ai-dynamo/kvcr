# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The matrix peer proof retains the removed DP2 smoke's KVCR assertions."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "e2e"))

from phase1 import cases, common  # noqa: E402

IDENTITY = "tc03: source and target worker IDs are nonempty and distinct"
FRESH = "tc03: prefix has not been sent to any candidate worker"
PREREQUISITES = (
    IDENTITY,
    FRESH,
    "tc03-seed-w0: served by worker 0",
    "tc03-seed-w0: cold request has 0 cached tokens",
    "tc03-hinted-w1: served by worker 1",
    "tc03: target worker 1 is cold for this prefix",
)


def context(tmp_path, identities=None):
    return SimpleNamespace(
        stack=SimpleNamespace(
            worker_ids={0: "source", 1: "target"} if identities is None else identities
        ),
        client=SimpleNamespace(evidence=tmp_path, log=[]),
        prompts=cases.Prompts("unique-run"),
    )


@pytest.mark.parametrize(
    "identities", [{0: "same", 1: "same"}, {0: "source"}, {0: "", 1: "target"}]
)
def test_peer_prerequisites_reject_missing_or_shared_identity(tmp_path, identities):
    ctx = context(tmp_path, identities)
    result = common.CaseResult("TC-03", "peer", "R4")
    assert not cases.tc03_peer_prerequisites(ctx, result)
    assert next(item for item in result.checks if item.name == IDENTITY).passed is False


@pytest.mark.parametrize("prior_worker", ["source", "target", None])
def test_peer_prerequisites_reject_prefix_already_used_by_any_candidate(
    tmp_path, prior_worker
):
    ctx = context(tmp_path)
    request = {
        "headers": {"x-dynamo-worker-instance-id": prior_worker},
        "body": {"prompt": ctx.prompts.tc03 + " previous suffix"},
    }
    (tmp_path / "previous.request.json").write_text(json.dumps(request))
    ctx.client.log.append({"request_file": "previous.request.json"})
    result = common.CaseResult("TC-03", "peer", "R4")
    assert not cases.tc03_peer_prerequisites(ctx, result)
    assert next(item for item in result.checks if item.name == FRESH).passed is False


def test_peer_prerequisites_accept_distinct_workers_and_unused_case_prefix(tmp_path):
    ctx = context(tmp_path)
    (tmp_path / "earlier.request.json").write_text(
        json.dumps({"body": {"prompt": ctx.prompts.tc01}})
    )
    ctx.client.log.append({"request_file": "earlier.request.json"})
    result = common.CaseResult("TC-03", "peer", "R4")
    assert cases.tc03_peer_prerequisites(ctx, result)
    assert all(item.passed for item in result.checks)


def test_peer_prerequisites_reject_missing_prior_request_evidence(tmp_path):
    ctx = context(tmp_path)
    ctx.client.log.append({"request_file": "missing.request.json"})
    result = common.CaseResult("TC-03", "peer", "R4")
    assert not cases.tc03_peer_prerequisites(ctx, result)


@pytest.mark.parametrize("prerequisite", PREREQUISITES)
def test_peer_proof_cannot_omit_identity_freshness_or_routing(prerequisite):
    required = cases.PROOF_CHECKS["TC-03"]["peer_transfer"]
    assert prerequisite in required
    result = common.CaseResult("TC-03", "peer", "R4")
    for name in required:
        if name != prerequisite:
            result.check(name, True)
    # Even all delivery/page/byte checks passing cannot replace a missing
    # identity, freshness, source coldness or actual routing prerequisite.
    cases.record_case_proofs(result)
    assert result.proofs["peer_transfer"] is False
