# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks of the Phase 1 harness logic; these never claim a GPU pass."""

import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pytest

E2E = Path(__file__).parents[1] / "e2e"
sys.path.insert(0, str(E2E))

from phase1 import cases, common, stack, telemetry  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "kvcr_phase1_runner", E2E / "run_phase1.py"
)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def sample(metrics, stamp="2026-09-25T10:00:05Z"):
    values = ", ".join(f"{key}={value}" for key, value in metrics.items())
    return (
        f"{stamp} INFO loggers.log: Engine 000: statistics\n"
        f"{stamp} INFO metrics.log: KV Transfer metrics: {values}\n"
    )


def at(second):
    return datetime(2026, 9, 25, 10, 0, second, tzinfo=timezone.utc)


# -- telemetry ---------------------------------------------------------------


def test_metric_names_match_vllm_log_form():
    assert telemetry.metric("kvcr_transfer_blocks", "source_write") == (
        "vllm:kvcr_transfer_blocks:('source_write',)"
    )
    assert telemetry.duration_count("remote_deliver") == (
        "vllm:kvcr_duration_seconds:('remote_deliver', 'success')_count"
    )
    assert telemetry.TIER_READ_BYTES == "vllm:kv_offload_tiering_read_bytes:('1:kvcr',)"


def test_window_sums_intervals_and_keeps_latest_gauge():
    blocks = telemetry.metric(telemetry.TRANSFER_BLOCKS, telemetry.REMOTE_DELIVER)
    gauge = telemetry.metric(telemetry.STATE, "pinned_keys")
    text = (
        sample({blocks: 3, gauge: 7}, "2026-09-25T10:00:01Z")
        + sample({blocks: 4, gauge: 2}, "2026-09-25T10:00:06Z")
        + sample({blocks: 9, gauge: 0}, "2026-09-25T10:00:30Z")
        + "2026-09-25T10:00:31Z INFO metrics.log: KV Transfer metrics: vllm:x=1"
    )
    snapshots, errors = telemetry.parse_log(text)
    assert errors == [] and len(snapshots) == 3  # the trailing partial line waits
    window = telemetry.window(snapshots, errors, at(5), at(10))
    assert window.blocks(telemetry.REMOTE_DELIVER) == 4
    assert window.gauge("pinned_keys") == 2
    assert window.samples == 1 and window.first_after_end is True
    everything = telemetry.window(snapshots, errors, at(0))
    assert everything.blocks(telemetry.REMOTE_DELIVER) == 16
    assert everything.gauge("pinned_keys") == 0


def test_window_flags_positive_failed_and_partial_kvcr_metrics():
    failed = telemetry.duration_count(telemetry.REMOTE_DELIVER, "failed")
    partial = telemetry.duration_count(telemetry.REMOTE_DELIVER, "partial")
    text = sample({failed: 0, partial: 1})
    snapshots, errors = telemetry.parse_log(text)
    window = telemetry.window(snapshots, errors, at(0))
    assert [entry["metric"] for entry in window.peer_failures] == [partial]


def test_parse_log_reports_incomplete_lines():
    text = "2026-09-25T10:00:05Z INFO KV Transfer metrics: vllm:a=1, vllm:b=oops\n"
    snapshots, errors = telemetry.parse_log(text)
    assert snapshots == [] and errors and "incomplete" in errors[0]


def test_latest_gauge_reads_the_last_sample(tmp_path):
    gauge = telemetry.metric(telemetry.STATE, "local_g2_allocated_slots")
    log = tmp_path / "worker.log"
    log.write_text(sample({gauge: 40}, "2026-09-25T10:00:01Z") + sample({gauge: 41}))
    assert telemetry.latest_gauge(log, "local_g2_allocated_slots") == 41
    assert telemetry.latest_gauge(log, "missing") is None


# -- token comparison --------------------------------------------------------


def body(prompt, completion, finish="length", extra=None):
    payload = {
        "choices": [{"text": "x", "finish_reason": finish}],
        "usage": {"prompt_tokens_details": {"cached_tokens": 128}},
        "nvext": {"prompt_token_ids": prompt, "completion_token_ids": completion},
    }
    payload["nvext"].update(extra or {})
    return payload


def test_extract_sample_and_identical_tokens_pass():
    reference = common.extract_sample(body([1, 2, 3], [7, 8]))
    observed = common.extract_sample(body([1, 2, 3], [7, 8]))
    assert common.compare_tokens(reference, observed) == []
    assert common.cached_tokens(body([1], [2])) == 128


@pytest.mark.parametrize(
    "observed, expected",
    [
        (body([1, 2, 3], [7, 9]), "diverge at position 1"),
        (body([1, 2, 3], [7]), "length 1 != reference 2"),
        (body([1, 2, 4], [7, 8]), "prompt token IDs differ"),
        (body([1, 2, 3], [7, 8], finish="stop"), "finish reason"),
    ],
)
def test_compare_tokens_reports_every_kind_of_mismatch(observed, expected):
    reference = common.extract_sample(body([1, 2, 3], [7, 8]))
    problems = common.compare_tokens(reference, common.extract_sample(observed))
    assert any(expected in problem for problem in problems)


def test_missing_token_ids_is_blocked_not_failed():
    with pytest.raises(common.BlockedError):
        common.extract_sample({"choices": [{"text": "x"}], "nvext": {}})
    with pytest.raises(common.BlockedError):
        common.extract_sample(
            {
                "choices": [],
                "nvext": {"completion_token_ids": [1], "prompt_token_ids": "x"},
            }
        )


def test_expected_hit_blocks_excludes_last_block_and_caps_by_shared_prefix():
    assert common.expected_hit_blocks(64 * 10, 64 * 10) == 9
    assert common.expected_hit_blocks(64 * 10 + 1, 64 * 10) == 10
    assert common.expected_hit_blocks(64 * 40, 64 * 20 + 5) == 20
    assert common.expected_hit_blocks(0, 100) == 0


# -- prompts -----------------------------------------------------------------


def test_prompts_share_only_the_intended_prefix():
    prompts = cases.Prompts("abc")
    texts = prompts.all()
    assert len(set(texts.values())) == len(texts)
    p1 = prompts.tc02_p1
    assert texts["tc02_full"].startswith(p1) and texts["tc02_partial"].startswith(p1)
    assert texts["tc02_full"][len(p1) :] != texts["tc02_partial"][len(p1) :]
    assert not texts["tc02_absent"].startswith(p1[:20])
    assert all(text.endswith(prompts.tail) for text in texts.values())


# -- case results, BLOCKED handling, JUnit -----------------------------------


def test_case_result_status_follows_checks():
    result = common.CaseResult("TC-09", "x", "R9")
    result.check("a", True)
    result.check("b", False, "why")
    result.finish()
    assert result.status == common.FAIL
    empty = common.CaseResult("TC-09", "x", "R9")
    empty.finish()
    assert empty.status == common.FAIL and "no checks" in empty.error


def test_run_case_maps_blocked_and_exceptions_to_statuses(monkeypatch):
    def blocked(_ctx):
        raise common.BlockedError("needs Linux 6.5")

    def broken(_ctx):
        raise RuntimeError("boom")

    monkeypatch.setitem(cases.CASES, "TC-B", blocked)
    monkeypatch.setitem(cases.CASES, "TC-X", broken)
    assert cases.run_case("TC-B", None).status == common.BLOCKED
    failed = cases.run_case("TC-X", None)
    assert failed.status == common.FAIL and "boom" in failed.error


def test_junit_marks_blocked_as_skipped_and_fail_as_failure(tmp_path):
    run = {
        "started_at": "2026-09-25T10:00:00+00:00",
        "cases": [
            {
                "case": "TC-01",
                "title": "a",
                "status": "PASS",
                "checks": [{"name": "n", "passed": True, "detail": "d"}],
                "error": None,
            },
            {
                "case": "TC-02",
                "title": "b",
                "status": "FAIL",
                "checks": [{"name": "n", "passed": False, "detail": "d"}],
                "error": "bad",
            },
            {
                "case": "TC-04",
                "title": "c",
                "status": "BLOCKED",
                "checks": [],
                "error": "kernel",
            },
        ],
    }
    common.write_junit(tmp_path, run)
    suite = ET.parse(tmp_path / "junit.xml").getroot()
    assert suite.attrib["tests"] == "3"
    assert suite.attrib["failures"] == "1" and suite.attrib["skipped"] == "1"
    kinds = [child.tag for case in suite for child in case]
    assert kinds == ["system-out", "failure", "skipped"]


# -- stack configuration -----------------------------------------------------


def test_guard_and_nonguard_tier_configs(tmp_path):
    settings = stack.Settings(model="m", output=tmp_path, mode=stack.GUARD)
    guard = stack.Stack(settings, tmp_path / "g")
    tier = guard.tier_config(1)
    assert tier["kvcr_service_socket_path"].endswith("memory-1.sock")
    assert tier["compatibility_digest"] == settings.compatibility_digest
    assert tier["control_ports"] == [17772] and "secondary_g2_slots" not in tier
    plain = stack.Stack(
        stack.Settings(model="m", output=tmp_path, mode=stack.NON_GUARD), tmp_path / "n"
    )
    plain_tier = plain.tier_config(0)
    assert plain_tier["secondary_g2_slots"] == 256
    assert "kvcr_service_socket_path" not in plain_tier
    command = guard.worker_command(0, kvcr=False)
    assert "--kv-transfer-config" not in command and "--data-parallel-size" in command
    with_kvcr = guard.worker_command(0, kvcr=True)
    transfer = json.loads(with_kvcr[with_kvcr.index("--kv-transfer-config") + 1])
    assert transfer["kv_connector_extra_config"]["secondary_tiers"][0]["type"] == "kvcr"


def test_kernel_release_parsing(monkeypatch):
    monkeypatch.setattr(stack.platform, "release", lambda: "5.15.0-1055-aws")
    assert stack.kernel_release() == (5, 15) < stack.MIN_GUARD_KERNEL
    monkeypatch.setattr(stack.platform, "release", lambda: "6.14.0-37-generic")
    assert stack.kernel_release() >= stack.MIN_GUARD_KERNEL
    monkeypatch.setattr(stack.platform, "release", lambda: "weird")
    assert stack.kernel_release() == (0, 0)


# -- runner arguments --------------------------------------------------------


def test_arguments_validate_gpus_and_cases(tmp_path):
    args = runner.arguments(["--output-dir", str(tmp_path), "--cases", "tc-01, TC-04"])
    assert args.gpus == [0, 1] and args.cases == ["TC-01", "TC-04"]
    for bad in (
        ["--gpus", "0"],
        ["--gpus", "1,1"],
        ["--cases", "TC-99"],
        ["--http-port", "8001"],
        ["--timeout", "0"],
    ):
        with pytest.raises(SystemExit):
            runner.arguments(["--output-dir", str(tmp_path), *bad])


def test_runner_refuses_non_empty_output(tmp_path):
    (tmp_path / "stale").write_text("x")
    args = runner.arguments(["--output-dir", str(tmp_path)])
    with pytest.raises(common.ValidationError):
        runner.Runner(args)


def test_discovery_instance_ids_reads_url_encoded_record_names(tmp_path):
    settings = stack.Settings(model="m", output=tmp_path, mode=stack.NON_GUARD)
    s = stack.Stack(settings, tmp_path / "s")
    s.base_env()
    root = Path(s.discovery.name)
    (root / "v1" / "instances").mkdir(parents=True)
    (root / "v1" / "models").mkdir(parents=True)
    for name in (
        "dynamo%2Fbackend%2Fgenerate%2Fc69d07089b72b3c9",
        "dynamo%2Fbackend%2Fclear%5Fkv%5Fblocks%2Fc69d07089b72b3c9",
    ):
        (root / "v1" / "instances" / name).write_text("{}")
    (root / "v1" / "models" / "dynamo%2Fbackend%2Fgenerate%2F4ee6").write_text("{}")
    (root / "other").mkdir()
    (root / "other" / "ff").write_text("{}")
    assert s.discovery_instance_ids() == {str(0xC69D07089B72B3C9), str(0x4EE6)}
    assert str(0xC69D07089B72B3C9) == "14311602924402422729"
    s.reset_discovery()
