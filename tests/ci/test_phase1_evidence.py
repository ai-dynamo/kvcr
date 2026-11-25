# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Adversarial checks of the functional harness's cache-path evidence."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "e2e"))

from phase1 import cases, common, telemetry  # noqa: E402
from phase1.client import Response  # noqa: E402


def window(*, local=0, remote=0, successes=0, errors=(), dropped=0):
    return telemetry.Window(
        totals={
            telemetry.metric(telemetry.TRANSFER_BLOCKS, telemetry.LOCAL_DELIVER): local,
            telemetry.metric(
                telemetry.TRANSFER_BLOCKS, telemetry.REMOTE_DELIVER
            ): remote,
            telemetry.duration_count(telemetry.LOCAL_DELIVER): successes,
            telemetry.TIER_READ_BYTES: local * 100,
            "_dropped_observations": dropped,
        },
        samples=1,
        first_after_end=False,
        peer_failures=[],
        parse_errors=list(errors),
    )


def context(**geometry):
    return cases.Context(
        None,
        None,
        None,
        {},
        index_wait=0,
        telemetry_timeout=0,
        recovery_timeout=0,
        **geometry,
    )


@pytest.mark.parametrize("damage", ["parse", "dropped", "missing", "empty"])
def test_window_health_rejects_untrustworthy_zero_claims(damage):
    observed = window()
    if damage == "parse":
        observed.parse_errors.append("unparsable required KVCR sample")
    elif damage == "dropped":
        observed.totals["_dropped_observations"] = 1
    elif damage == "missing":
        observed.totals = {"unrelated": 1}
    else:
        observed.samples = 0
    assert observed.health_errors()


def test_window_health_accepts_observed_zero_kvcr_counters():
    assert window().health_errors() == []


@pytest.mark.parametrize("damage", ["parse", "dropped"])
def test_local_commit_cannot_pass_with_damaged_telemetry(monkeypatch, damage):
    observed = window(
        errors=["bad sample"] if damage == "parse" else (),
        dropped=int(damage == "dropped"),
    )
    observed.totals[
        telemetry.metric(telemetry.TRANSFER_BLOCKS, telemetry.LOCAL_FILL)
    ] = 1
    observed.totals[telemetry.duration_count(telemetry.LOCAL_FILL)] = 1
    ctx = context()
    monkeypatch.setattr(ctx, "wait_window", lambda *args, **kwargs: observed)
    result = common.CaseResult("TC-01", "commit", "R1")
    ctx.local_commit_checks(result, 0, "unused", "commit")
    result.finish()
    assert result.status == "FAIL"


@pytest.mark.parametrize(
    "local,remote,successes,cached,expected",
    [
        (2, 0, 1, 128, True),
        (0, 0, 0, 128, False),
        (2, 1, 1, 128, False),
        (2, 0, 0, 128, False),
        (1, 0, 1, 128, False),
        (2, 0, 1, 0, False),
    ],
)
def test_local_restore_requires_local_delivery_covering_actual_hit(
    local, remote, successes, cached, expected
):
    ctx = context()
    ok, _ = cases.local_restore_verdict(
        ctx, window(local=local, remote=remote, successes=successes), cached
    )
    assert ok is expected


def test_local_restore_rejects_untrustworthy_delivery_evidence():
    observed = window(local=2, successes=1, errors=["bad KVCR interval"])
    assert not cases.local_restore_verdict(context(), observed, 128)[0]


def test_local_restore_preserves_hybrid_page_geometry():
    ctx = context(hit_block_tokens=576, ssm_groups=3)
    assert cases.local_restore_verdict(ctx, window(local=5, successes=1), 1152)[0]
    assert not cases.local_restore_verdict(ctx, window(local=2, successes=1), 1152)[0]


def response(stage, start, end):
    return Response(
        stage, start, end, 1, 200, {}, None, "request.json", "response.json"
    )


@pytest.mark.parametrize(
    "second_start,second_end,expected",
    [
        ("2026-11-25T10:00:00.500Z", "2026-11-25T10:00:01.500Z", True),
        ("2026-11-25T10:00:01Z", "2026-11-25T10:00:02Z", False),
        ("2026-11-25T10:00:02Z", "2026-11-25T10:00:03Z", False),
        ("bad", "2026-11-25T10:00:03Z", False),
        ("2026-11-25T10:00:00.500", "2026-11-25T10:00:01.500", False),
    ],
)
def test_request_overlap_uses_observed_valid_intervals(
    second_start, second_end, expected
):
    first = response("first", "2026-11-25T10:00:00Z", "2026-11-25T10:00:01Z")
    second = response("second", second_start, second_end)
    assert cases.request_overlap(first, second)["observed"] is expected


def test_nixl_decoder_metrics_are_not_required_kvcr_parse_failures():
    text = (
        "2026-11-25T10:00:00Z INFO KV Transfer metrics: Num successful transfers=5, "
        "Avg xfer time (ms)=24.134, Num failed transfers=0\n"
    )
    snapshots, errors = telemetry.parse_log(text)
    assert snapshots == [] and errors == []


def test_malformed_kvcr_metrics_remain_visible_as_errors():
    snapshots, errors = telemetry.parse_log(
        "2026-11-25T10:00:00Z INFO KV Transfer metrics: vllm:kvcr_broken=oops\n"
    )
    assert snapshots == [] and errors


def test_partial_or_failed_reference_checks_cannot_claim_typed_exact_proof():
    result = common.CaseResult("TC-01", "integration", "R1")
    result.check("tc01-cold-w0: tokens equal reference (tc01)", True)
    cases.record_case_proofs(result)
    assert result.proofs["reference_exact"] is False
    result.check("tc01-reuse-w0: tokens equal reference (tc01)", False)
    cases.record_case_proofs(result)
    assert result.proofs["reference_exact"] is False


def test_exact_proof_is_derived_from_all_required_named_checks():
    result = common.CaseResult("TC-01", "integration", "R1")
    result.check("tc01-cold-w0: tokens equal reference (tc01)", True)
    result.check("tc01-reuse-w0: tokens equal reference (tc01)", True)
    cases.record_case_proofs(result)
    assert result.proofs["reference_exact"] is True
    assert result.proofs["local_commit"] is False


def test_serial_client_intervals_cannot_claim_request_overlap_proof():
    result = common.CaseResult("TC-03", "isolation", "R4")
    result.check("tc03: observed request intervals are valid", True)
    result.measurements["request_overlap"] = {"observed": False}
    cases.record_case_proofs(result)
    assert result.proofs["request_overlap"] is False


def test_engine_limit_waiver_cannot_claim_required_partial_reuse():
    result = common.CaseResult("TC-02", "DRAM", "R2")
    result.measurements["engine_partial_prefix_reuse"] = {
        "reference_cached_tokens": [0, 128],
        "reuses": False,
    }
    result.check(
        "tc02-partial-w1: partial-prefix reuse unsupported by the engine "
        "itself (reference cached_tokens=0), hit checks not applicable",
        True,
    )
    cases.record_case_proofs(result)
    assert result.proofs["partial_reuse"] is False


def test_positive_candidate_partial_hit_requires_positive_reference_capability():
    result = common.CaseResult("TC-02", "DRAM", "R2")
    result.check("tc02-partial-w1: cache hit is positive and block-aligned", True)
    result.check(
        "tc02-partial-w1: hit covers fewer blocks than the seeded prompt", True
    )
    result.measurements["engine_partial_prefix_reuse"] = {
        "reference_cached_tokens": [0, 128],
        "reuses": False,
    }
    cases.record_case_proofs(result)
    assert result.proofs["partial_reuse"] is False
    result.measurements["engine_partial_prefix_reuse"] = {
        "reference_cached_tokens": [64, 128],
        "reuses": True,
    }
    cases.record_case_proofs(result)
    assert result.proofs["partial_reuse"] is True


def test_one_healthy_window_cannot_substitute_for_all_required_windows():
    result = common.CaseResult("TC-01", "integration", "R1")
    result.check("tc01 reuse: required KVCR telemetry is complete and healthy", True)
    cases.record_case_proofs(result)
    assert result.proofs["telemetry_health"] is False


def test_hicache_read_byte_evidence_is_labelled_derived():
    payload = {
        "epoch": 1790000010,
        "metrics": {
            json.dumps(["counter", "kvcr_transfer_blocks", ["local_fill"], "value"]): 2,
            json.dumps(
                ["counter", "kvcr_transfer_bytes", ["local_fill"], "value"]
            ): 200,
            json.dumps(
                ["counter", "kvcr_transfer_blocks", ["remote_deliver"], "value"]
            ): 3,
        },
    }
    snapshots, errors = telemetry.parse_log(
        f"KVCR L3 native stats {json.dumps(payload)}\n"
    )
    observed = telemetry.window(
        snapshots, errors, cases.parse_iso("2026-01-01T00:00:00Z")
    )
    assert observed.provenance[telemetry.TIER_READ_BYTES] == "derived"
    assert observed.provenance[telemetry.TIER_WRITE_BYTES] == "measured:local_fill"


def test_malformed_hicache_metric_key_does_not_disappear():
    payload = {"epoch": 1790000010, "metrics": {"bad required KVCR key": 1}}
    _, errors = telemetry.parse_log(f"KVCR L3 native stats {json.dumps(payload)}\n")
    assert errors


@pytest.mark.parametrize("bad_value", ["'damaged'", "True", "1e309"])
def test_multiconnector_invalid_failure_counter_is_a_parse_error(bad_value):
    failed = telemetry.duration_count(telemetry.REMOTE_DELIVER, "failed")
    gauge = telemetry.metric(telemetry.STATE, "pinned_keys")
    text = (
        "2026-11-25T10:00:00Z INFO KV Transfer metrics: "
        f"OffloadingConnector={{{gauge!r}: 0, {failed!r}: {bad_value}}}, "
        "NixlConnector={'Num successful transfers': 1}\n"
    )
    snapshots, errors = telemetry.parse_log(text)
    assert errors, "a damaged failure counter must never become observed zero"
    observed = telemetry.window(
        snapshots, errors, cases.parse_iso("2026-01-01T00:00:00Z")
    )
    assert observed.health_errors()


@pytest.mark.parametrize(
    "case_id,proof,required_name",
    [
        (
            "TC-01",
            "local_commit",
            "tc01 store->commit: no positive KVCR failed/partial metrics",
        ),
        (
            "TC-01",
            "warm_reuse",
            "tc01-reuse-w0: no positive KVCR failed/partial metrics",
        ),
        (
            "TC-02",
            "peer_transfer",
            "tc02 partial deliver: no unexpected KVCR failed/partial metrics",
        ),
        (
            "TC-02",
            "peer_transfer",
            "tc02-present-w1: no positive KVCR failed/partial metrics",
        ),
        (
            "TC-03",
            "peer_transfer",
            "tc03 transfer: no unexpected KVCR failed/partial metrics",
        ),
        (
            "TC-04",
            "guard_serve",
            "tc04-guard-served-w1: "
            "no positive KVCR failed/partial metrics while Guard served",
        ),
        (
            "TC-04",
            "local_restore",
            "tc04-preserved-w0-replacement: no positive KVCR failed/partial metrics",
        ),
    ],
)
def test_missing_no_failure_assertion_cannot_qualify_semantic_proof(
    case_id, proof, required_name
):
    result = common.CaseResult(case_id, "proof", "requirement")
    for name in cases.PROOF_CHECKS[case_id][proof]:
        if name != required_name:
            result.check(name, True)
    cases.record_case_proofs(result)
    assert result.proofs[proof] is False


@pytest.mark.parametrize(
    "case_id,proof,label",
    [
        ("TC-02", "peer_transfer", "tc02 partial deliver"),
        ("TC-03", "peer_transfer", "tc03 transfer"),
        ("TC-04", "guard_serve", "tc04-guard-served-w1"),
    ],
)
def test_known_page_geometry_requires_executed_byte_assertion(case_id, proof, label):
    byte_check = f"{label}: transfer bytes == pages * page bytes"
    assert byte_check in cases.required_proof_checks(case_id, page_bytes=100)[proof]
    assert byte_check not in cases.required_proof_checks(case_id, page_bytes=0)[proof]
    result = common.CaseResult(case_id, "proof", "requirement")
    for name in cases.required_proof_checks(case_id, page_bytes=100)[proof]:
        if name != byte_check:
            result.check(name, True)
    cases.record_case_proofs(result, page_bytes=100)
    assert result.proofs[proof] is False
