# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks of the Phase 1 harness logic; these never claim a GPU pass."""

import importlib.util
import io
import json
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

E2E = Path(__file__).parents[1] / "e2e"
sys.path.insert(0, str(E2E))

from phase1 import cases, common, perf, stack, telemetry  # noqa: E402

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


def vllm_sample(metrics, stamp):
    """A metrics line in vLLM's own log format: second-precision stamp."""
    values = ", ".join(f"{key}={value}" for key, value in metrics.items())
    return (
        f"(MainProcess pid=1) INFO {stamp} [metrics.py:108] "
        f"KV Transfer metrics: {values}\n"
    )


def test_window_same_second_sample_counts_only_when_flushed_after_the_request():
    # vLLM's logger stamps to the second. A request started at 10:00:10.415;
    # a sample flushed at 10:00:10.9 (after Dynamo logged the request at
    # 10:00:10.43) carries its delivery and must count; a sample flushed at
    # 10:00:10.1, before any request line, belongs to the previous request.
    blocks = telemetry.metric(telemetry.TRANSFER_BLOCKS, telemetry.REMOTE_DELIVER)
    year = datetime.now(timezone.utc).year
    stamp = "09-25 10:00:10"
    start = datetime(year, 9, 25, 10, 0, 10, 415000, tzinfo=timezone.utc)
    end = start.replace(second=14)
    after = (
        vllm_sample({blocks: 5}, "09-25 10:00:00")
        + f"{year}-09-25T10:00:10.430000Z INFO handle_payload: request\n"
        + vllm_sample({blocks: 22}, stamp)
    )
    snapshots, errors = telemetry.parse_log(after)
    assert errors == []
    assert (
        telemetry.window(snapshots, errors, start, end).blocks(telemetry.REMOTE_DELIVER)
        == 22
    )
    before = (
        vllm_sample({blocks: 5}, "09-25 10:00:00")
        + f"{year}-09-25T10:00:09.900000Z INFO handle_payload: previous request\n"
        + vllm_sample({blocks: 11}, stamp)
        + f"{year}-09-25T10:00:10.430000Z INFO handle_payload: this request\n"
    )
    snapshots, errors = telemetry.parse_log(before)
    assert (
        telemetry.window(snapshots, errors, start, end).blocks(telemetry.REMOTE_DELIVER)
        == 0
    )


def test_parse_log_ignores_naive_or_partial_leading_timestamps():
    # Lines that start with a bare date or a zone-less time must not be taken as
    # request stamps: they are naive datetimes and cannot be compared with the
    # aware request start (this crashed TC-01 in CI).
    blocks = telemetry.metric(telemetry.TRANSFER_BLOCKS, telemetry.REMOTE_DELIVER)
    year = datetime.now(timezone.utc).year
    text = (
        f"{year}-09-25 some tool banner\n"
        f"{year}-09-25T10:00:09 naive stamp without zone\n"
        + vllm_sample({blocks: 3}, "09-25 10:00:10")
    )
    snapshots, errors = telemetry.parse_log(text)
    assert errors == [] and len(snapshots) == 1
    assert snapshots[0].second_precision and snapshots[0].preceded_by is None
    start = datetime(year, 9, 25, 10, 0, 10, 415000, tzinfo=timezone.utc)
    window = telemetry.window(snapshots, errors, start)
    assert window.blocks(telemetry.REMOTE_DELIVER) == 0


def test_parse_log_reads_sglang_native_stats():
    # SGLang's KVCR backend logs one JSON payload per interval, stamped with an
    # epoch; its tuple keys map onto the vLLM metric names the cases use, and
    # the tier read/write bytes are derived from the KVCR transfer bytes.
    payload = {
        "epoch": 1790000010.25,
        "agent": "a",
        "endpoint": "tcp://10.0.0.1:17771",
        "interval_seconds": 5.0,
        "metrics": {
            json.dumps(
                ["counter", "kvcr_transfer_blocks", ["remote_deliver"], "value"]
            ): 22,
            json.dumps(["counter", "kvcr_transfer_blocks", ["local_fill"], "value"]): 2,
            json.dumps(
                ["counter", "kvcr_transfer_bytes", ["local_fill"], "value"]
            ): 4096,
            json.dumps(
                [
                    "histogram",
                    "kvcr_duration_seconds",
                    ["remote_deliver", "success"],
                    "count",
                ]
            ): 1,
            json.dumps(
                [
                    "histogram",
                    "kvcr_duration_seconds",
                    ["remote_deliver", "success"],
                    "bucket_3",
                ]
            ): 1,
            json.dumps(
                ["gauge", "kvcr_state", ["local_g2_allocated_slots"], "value"]
            ): 7,
        },
    }
    text = (
        "[2026-09-25 10:00:10] INFO kvcr_store: KVCR L3 ready: endpoint=x\n"
        "[2026-09-25 10:00:10] INFO kvcr_store: KVCR L3 native stats "
        f"{json.dumps(payload)}\n"
    )
    snapshots, errors = telemetry.parse_log(text)
    assert errors == [] and len(snapshots) == 1
    start = datetime.fromtimestamp(1790000000, tz=timezone.utc)
    window = telemetry.window(snapshots, errors, start)
    assert window.blocks(telemetry.REMOTE_DELIVER) == 22
    assert window.get(telemetry.duration_count(telemetry.REMOTE_DELIVER)) == 1
    # KVCR counts no bytes for remote_deliver: 22 blocks x (4096 / 2) per block.
    assert window.get(telemetry.TIER_READ_BYTES) == 22 * 2048
    assert window.get(telemetry.TIER_WRITE_BYTES) == 4096
    assert window.gauge("local_g2_allocated_slots") == 7
    assert snapshots[0].second_precision is False


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


def test_guard_and_nonguard_tier_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(stack.Stack, "advertise_host", staticmethod(lambda: "10.0.0.5"))
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


@pytest.mark.parametrize("mapping", ["0,0,0", "0,1,1"])
def test_disagg_arguments_reject_decode_worker_gpu_overcommit(tmp_path, mapping):
    with pytest.raises(SystemExit):
        runner.arguments(
            [
                "--output-dir",
                str(tmp_path),
                "--topology",
                "disagg",
                "--gpus",
                mapping,
                "--gpu-memory-utilization",
                "0.35" if mapping == "0,0,0" else "0.8",
            ]
        )


def test_three_shared_workers_can_use_a_bounded_diagnostic_budget(tmp_path):
    args = runner.arguments(
        [
            "--output-dir",
            str(tmp_path),
            "--topology",
            "disagg",
            "--gpus",
            "0,0,0",
            "--gpu-memory-utilization",
            "0.3",
        ]
    )
    assert args.gpus == [0, 0, 0]


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
    (
        root / "v1" / "instances" / "dynamo%2Fbackend%2Fworker%5Fkv%5Fquery%2Fabcd"
    ).write_text("{}")
    (root / "other").mkdir()
    (root / "other" / "ff").write_text("{}")
    assert s.discovery_instance_ids() == {str(0xC69D07089B72B3C9), str(0x4EE6)}
    assert str(0xC69D07089B72B3C9) == "14311602924402422729"
    s.reset_discovery()


def test_state_agent_config_only_in_guard_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(stack.Stack, "advertise_host", staticmethod(lambda: "10.0.0.5"))
    guard = stack.Stack(
        stack.Settings(model="m", output=tmp_path, mode=stack.GUARD), tmp_path / "g"
    )
    extra = guard.transfer_config(1)["kv_connector_extra_config"]
    agent = extra["dynamo_state_agent"]
    assert agent["raw_advertise_host"] == "10.0.0.5"
    assert agent["cache_owner_ids"] == {"0": stack.Stack.cache_owner_id(1)}
    assert stack.Stack.cache_owner_id(1).endswith("/" + "0" * 31 + "1")
    plain = stack.Stack(
        stack.Settings(model="m", output=tmp_path, mode=stack.NON_GUARD), tmp_path / "n"
    )
    assert (
        "dynamo_state_agent"
        not in plain.transfer_config(0)["kv_connector_extra_config"]
    )
    off = stack.Stack(
        stack.Settings(model="m", output=tmp_path, mode=stack.GUARD, state_agent=False),
        tmp_path / "o",
    )
    assert (
        "dynamo_state_agent" not in off.transfer_config(0)["kv_connector_extra_config"]
    )


# -- negative verdicts (review of #103) ----------------------------------------


def _ctx(**geometry):
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


def _window(remote_deliver_blocks=0, successes=0, read_bytes=0, failures=()):
    totals = {
        telemetry.metric(
            telemetry.TRANSFER_BLOCKS, telemetry.REMOTE_DELIVER
        ): remote_deliver_blocks,
        telemetry.duration_count(telemetry.REMOTE_DELIVER): successes,
        telemetry.TIER_READ_BYTES: read_bytes,
    }
    return telemetry.Window(
        totals=totals,
        samples=1,
        first_after_end=True,
        peer_failures=list(failures),
        parse_errors=[],
    )


def test_zero_hit_replacement_recovery_fails():
    # A replacement that answers correctly but recomputed the prefix is not a recovery.
    assert cases.hit_verdict(0, 64) == (False, "cached_tokens=0 (block 64)")
    assert cases.hit_verdict(1472, 64)[0]
    assert not cases.hit_verdict(1500, 576)[0], "hybrid hits must be 576-aligned"
    assert not cases.hit_verdict(1472, 64, max_blocks=10)[0], (
        "above the shared-prefix bound"
    )


def test_local_hit_without_guard_delivery_fails():
    ctx = _ctx(hit_block_tokens=64)
    local_recommit = _window(remote_deliver_blocks=0, successes=0)
    ok, detail = perf.guard_delivery_verdict(ctx, local_recommit, 23, 0)
    assert not ok and "remote_deliver blocks=0" in detail

    guard = _window(remote_deliver_blocks=23, successes=1, read_bytes=23 * 9437184)
    assert perf.guard_delivery_verdict(ctx, guard, 23, 0)[0]

    short = _window(remote_deliver_blocks=20, successes=1)
    assert not perf.guard_delivery_verdict(ctx, short, 23, 0)[0], (
        "pages must cover the hit"
    )

    # One failed remote_deliver is the kvcr#51 miss of an earlier attempt, and
    # is tolerated only when such an attempt happened.
    failed = {
        "metric": telemetry.duration_count(telemetry.REMOTE_DELIVER, "failed"),
        "value": 1,
    }
    after_miss = _window(remote_deliver_blocks=23, successes=1, failures=[failed])
    assert perf.guard_delivery_verdict(ctx, after_miss, 23, 1)[0]
    assert not perf.guard_delivery_verdict(ctx, after_miss, 23, 0)[0]


def test_token_mismatch_during_recovery_fails():
    assert perf.attempt_outcome(True, ["token 3: 11 != 12"], 1472) == "fail"
    assert perf.attempt_outcome(True, ["token 3: 11 != 12"], 0) == "fail"
    assert perf.attempt_outcome(True, [], 1472) == "accept"
    assert perf.attempt_outcome(True, [], 0) == "retry", "kvcr#51 first-request miss"
    assert perf.attempt_outcome(False, ["no tokens"], None) == "retry"


def test_missing_ssm_pages_fail_hybrid_delivery():
    page = 18874368  # Qwen3.5-4B KVCR page: 94371840 bytes / 5 pages
    strict = _ctx(hit_block_tokens=576, ssm_groups=3, page_bytes=page)
    assert strict.delivery_covers(5, 2), "2 attention + 3 SSM pages"
    assert not strict.delivery_covers(2, 2), "attention pages only: SSM state missing"
    assert not strict.delivery_covers(4, 2), "one SSM group missing"
    assert not strict.delivery_covers(6, 2), "an extra page is also wrong"
    assert strict.expected_pages(2) == "2+3*1"
    result = common.CaseResult("TC-05", "x", "perf")
    strict.bytes_match_pages(result, "short", 4 * page, 5)
    strict.bytes_match_pages(result, "exact", 5 * page, 5)
    assert [check.passed for check in result.checks] == [False, True]

    # Without the geometry the check is only a lower bound; that is why the
    # hybrid job must pass --ssm-groups and --page-bytes.
    lenient = _ctx(hit_block_tokens=576)
    assert lenient.delivery_covers(2, 2)
    lenient.bytes_match_pages(result, "skipped", 1, 5)
    assert len(result.checks) == 2


# -- configuration-matrix harness contracts ---------------------------------


@pytest.mark.parametrize("mode", [stack.GUARD, stack.NON_GUARD])
def test_local_disagg_owns_only_mode_applicable_services(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(stack.Stack, "advertise_host", staticmethod(lambda: "10.0.0.5"))
    args = runner.arguments(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--topology",
            "disagg",
            "--gpus",
            "0,0,1",
            "--gpu-memory-utilization",
            "0.35",
            "--mode",
            mode,
            "--cases",
            "TC-01,TC-02,TC-03",
        ]
    )
    run = runner.Runner(args)
    s = run.stack
    started = []
    monkeypatch.setattr(s, "start_service", lambda i: started.append(("service", i)))
    monkeypatch.setattr(s, "start_state_agent", lambda: started.append(("agent",)))
    monkeypatch.setattr(
        s, "start_frontend", lambda port: started.append(("frontend", port))
    )
    monkeypatch.setattr(s, "start_worker", lambda i: started.append(("worker", i)))
    monkeypatch.setattr(s, "wait_model", lambda client: None)
    monkeypatch.setattr(
        s, "discover_worker", lambda client, i: s.worker_ids.update({i: str(i + 10)})
    )
    run.stack_phase()
    assert [event for event in started if event[0] == "worker"] == [
        ("worker", 0),
        ("worker", 2),
        ("worker", 1),
    ]
    assert [event for event in started if event[0] == "service"] == (
        [("service", 0), ("service", 1)] if mode == stack.GUARD else []
    )
    assert (("agent",) in started) == (mode == stack.GUARD)
    assert run.result["state_agent"] == (mode == stack.GUARD)
    assert s.pin(0) == {"prefill": "10", "worker": "12"}
    assert s.pin(1) == {"prefill": "11", "worker": "12"}
    assert run.result["deployment"] == "disaggregated"
    s.cleanup()


def test_native_disagg_preserves_kvcr_and_disabled_reference_connectors(tmp_path):
    args = runner.arguments(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--mode",
            "nonguard",
            "--topology",
            "disagg",
            "--gpus",
            "0,0,1",
            "--gpu-memory-utilization",
            "0.35",
            "--g2-slots",
            "48",
            "--cases",
            "TC-01,TC-02,TC-03",
        ]
    )
    run = runner.Runner(args)
    s = run.stack
    for index in (0, 1):
        transfer = s.kv_transfer_config(index, kvcr=True)
        assert transfer["kv_connector"] == "MultiConnector"
        extra = transfer["kv_connector_extra_config"]
        offload, nixl = extra["connectors"]
        assert offload["kv_connector"] == "OffloadingConnector"
        tier = extra["secondary_tiers"][0]
        assert tier["secondary_g2_slots"] == 48
        assert "kvcr_service_socket_path" not in tier
        assert "dynamo_state_agent" not in extra
        assert extra["self_describing_kv_events"]
        assert nixl == {"kv_connector": "NixlConnector", "kv_role": "kv_producer"}
        assert s.kv_transfer_config(index, kvcr=False) == nixl
    decode = {"kv_connector": "NixlConnector", "kv_role": "kv_consumer"}
    assert s.kv_transfer_config(2, kvcr=True) == decode
    assert s.kv_transfer_config(2, kvcr=False) == decode
    s.cleanup()


def test_snapshot_path_keeps_canonical_served_identity(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    args = runner.arguments(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--model",
            "owner/model",
            "--model-path",
            str(snapshot),
            "--g2-slots",
            "24",
            "--mode",
            "nonguard",
        ]
    )
    run = runner.Runner(args)
    assert run.settings.model_path == snapshot.resolve()
    assert run.settings.g2_slots == 24
    for kvcr in (False, True):
        command = run.stack.worker_command(0, kvcr=kvcr)
        assert command[command.index("--model") + 1] == str(snapshot.resolve())
        assert command[command.index("--served-model-name") + 1] == "owner/model"
    assert run.result["model"] == "owner/model"
    assert run.result["model_path"] == str(snapshot.resolve())
    run.stack.cleanup()


@pytest.mark.parametrize("slots", ["0", "-1", "65537"])
def test_native_pool_slots_reject_invalid_capacity(tmp_path, slots, capsys):
    with pytest.raises(SystemExit):
        runner.arguments(["--output-dir", str(tmp_path), "--g2-slots", slots])
    assert "--g2-slots must be 1..65536" in capsys.readouterr().err


def test_snapshot_path_must_be_a_directory(tmp_path, capsys):
    for path in (tmp_path / "missing", tmp_path / "file"):
        if path.name == "file":
            path.write_text("not a directory")
        with pytest.raises(SystemExit):
            runner.arguments(["--output-dir", str(tmp_path), "--model-path", str(path)])
        assert "--model-path must be an existing directory" in capsys.readouterr().err


def test_disagg_saved_connector_matches_actual_launch(tmp_path, monkeypatch):
    s = stack.Stack(
        stack.Settings(
            model="m",
            output=tmp_path,
            mode=stack.NON_GUARD,
            topology="disagg",
            gpus=(0, 0, 1),
        ),
        tmp_path / "stack",
    )
    launches = {}

    def spawn(name, command, env):
        launches[name] = command
        return SimpleNamespace(name=name)

    monkeypatch.setattr(s, "_spawn", spawn)
    for index, kvcr, tag in ((0, False, "-reference"), (0, True, ""), (2, True, "")):
        s.start_worker(index, kvcr=kvcr, tag=tag)
        name = f"worker-{index}{tag}"
        actual = launches[name]
        expected = json.loads(actual[actual.index("--kv-transfer-config") + 1])
        assert (
            json.loads((s.evidence / f"{name}.kv-transfer-config.json").read_text())
            == expected
        )
    s.cleanup()


@pytest.mark.parametrize("legacy", [False, True])
def test_provenance_reads_image_build_metadata(tmp_path, monkeypatch, legacy):
    preferred = "/opt/kvcr-build-metadata.json"
    old = "/opt/kvcr-build/source-revisions.json"
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"kvcr_commit": "installed-source", "overlay": "fix"})
    )
    exists = Path.exists
    read_text = Path.read_text

    def metadata_exists(path):
        if str(path) in (old, preferred):
            return str(path) == (old if legacy else preferred)
        return exists(path)

    def metadata_read(path, *args, **kwargs):
        if str(path) in (old, preferred):
            return read_text(metadata, *args, **kwargs)
        return read_text(path, *args, **kwargs)

    args = runner.arguments(["--output-dir", str(tmp_path / "run")])
    monkeypatch.setattr(Path, "exists", metadata_exists)
    monkeypatch.setattr(Path, "read_text", metadata_read)
    assert (
        runner.provenance(args)["image_source_revisions"]["kvcr_commit"]
        == "installed-source"
    )


def test_sigterm_reaches_runner_cleanup_and_restores_handler(tmp_path, monkeypatch):
    args = runner.arguments(["--output-dir", str(tmp_path / "run")])
    run = runner.Runner(args)
    cleaned = []
    monkeypatch.setattr(runner, "arguments", lambda: args)
    monkeypatch.setattr(runner, "Runner", lambda args: run)
    monkeypatch.setattr(runner, "provenance", lambda args: {})

    def terminate():
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(run.stack, "preflight", terminate)
    monkeypatch.setattr(run.stack, "cleanup", lambda: cleaned.append(True))
    previous = signal.getsignal(signal.SIGTERM)
    assert runner.main() == 1
    assert cleaned == [True]
    assert "KeyboardInterrupt" in run.result["error"]
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.parametrize("failed_reference", [False, True])
def test_phase_durations_include_failure_and_cleanup(
    tmp_path, monkeypatch, failed_reference
):
    args = runner.arguments(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--cases",
            "TC-01",
        ]
    )
    run = runner.Runner(args)
    monkeypatch.setattr(runner, "provenance", lambda args: {})
    clock = iter(range(100))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(run.stack, "preflight", lambda: {})

    def reference():
        if failed_reference:
            raise common.ValidationError("reference was unstable")

    def cases_phase(client):
        case = common.CaseResult("TC-01", "generation", "R1")
        case.check("tokens", True)
        case.finish()
        run.result["cases"].append(case.to_dict())

    monkeypatch.setattr(run, "reference_phase", reference)
    monkeypatch.setattr(run, "stack_phase", lambda: object())
    monkeypatch.setattr(run, "cases_phase", cases_phase)
    monkeypatch.setattr(run.stack, "cleanup", lambda: {})
    assert run.run() == int(failed_reference)
    expected = {"preflight", "reference", "cleanup"}
    if not failed_reference:
        expected |= {"stack", "cases"}
    assert run.result["phase_durations_seconds"] == {name: 1 for name in expected}


def test_external_and_remote_native_services_remain_rejected(tmp_path, capsys):
    base = ["--output-dir", str(tmp_path), "--mode", "nonguard"]
    with pytest.raises(SystemExit):
        runner.arguments(
            base + ["--external-services", str(tmp_path), "--compatibility-digest", "d"]
        )
    assert "--external-services requires --mode guard" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        runner.arguments(base + ["--remote-worker", "1=http://node:1234"])
    assert "--remote-worker needs --external-services" in capsys.readouterr().err


def test_failed_reference_stop_is_retried_and_never_lost_from_cleanup(tmp_path):
    s = stack.Stack(stack.Settings(model="m", output=tmp_path), tmp_path / "stack")
    child = SimpleNamespace(name="reference", pid=123, stopped=False)
    attempts = []

    def stop():
        attempts.append(True)
        # Reproduce a child whose stop method marks itself stopped on failure.
        child.stopped = True
        if len(attempts) == 1:
            return {"name": child.name, "pid": child.pid, "error": "did not exit"}
        return {"name": child.name, "pid": child.pid, "exit_code": 0}

    child.stop = stop
    s.children.append(child)
    with pytest.raises(common.ValidationError, match="stop"):
        s.stop(child)
    assert not child.stopped
    with pytest.raises(common.ValidationError, match="cleanup"):
        s.cleanup()
    report = json.loads((s.evidence / "cleanup.json").read_text())
    assert len(attempts) == 2
    assert len(report["owned_processes"]) == 2
    assert report["owned_processes"][0]["error"] == "did not exit"
    assert report["owned_processes"][1]["exit_code"] == 0
    assert report["process_states"] == [
        {"name": "reference", "pid": 123, "stopped": True}
    ]


def test_cleanup_continues_after_one_child_stop_raises(tmp_path):
    s = stack.Stack(stack.Settings(model="m", output=tmp_path), tmp_path / "stack")
    good = SimpleNamespace(name="good", pid=1, stopped=False)
    bad = SimpleNamespace(name="bad", pid=2, stopped=False)
    good.stop = lambda: {"name": "good", "pid": 1, "exit_code": 0}

    def failed_stop():
        raise PermissionError("cannot signal child")

    bad.stop = failed_stop
    s.children = [good, bad]
    with pytest.raises(common.ValidationError, match="cleanup"):
        s.cleanup()
    report = json.loads((s.evidence / "cleanup.json").read_text())
    assert (
        report["owned_processes"][0]["error"] == "PermissionError: cannot signal child"
    )
    assert report["owned_processes"][1]["name"] == "good"
    assert good.stopped and not bad.stopped


@pytest.mark.parametrize("managed_class", [stack.Managed, stack.RemoteManaged])
def test_managed_failed_stop_remains_retryable(managed_class):
    child = object.__new__(managed_class)
    child.name = "reference"
    child.stopped = False
    child._log = io.StringIO()
    if managed_class is stack.RemoteManaged:
        child.pid = 123
        child.agent_url = "http://node:1234"
    responses = iter(["timeout", "timeout", 0, 0])

    def wait(timeout):
        response = next(responses)
        if response == "timeout":
            raise subprocess.TimeoutExpired("reference", timeout)
        return response

    child.process = SimpleNamespace(pid=123, returncode=0, wait=wait)
    child.signal_group = lambda number: True
    assert "error" in child.stop()
    assert not child.stopped
    assert child.stop()["exit_code"] == 0
    assert child.stopped


def test_injected_kill_failure_keeps_process_retryable(tmp_path):
    s = stack.Stack(stack.Settings(model="m", output=tmp_path), tmp_path / "stack")

    def wait(timeout):
        raise subprocess.TimeoutExpired("worker", timeout)

    child = SimpleNamespace(
        name="worker-0",
        pid=123,
        stopped=False,
        process=SimpleNamespace(wait=wait),
        signal_group=lambda number: True,
    )
    child.stop = lambda: {"name": child.name, "pid": child.pid, "exit_code": -9}
    s.children.append(child)
    s.workers[0] = child
    with pytest.raises(common.ValidationError, match="survived"):
        s.kill_worker(0)
    assert not child.stopped
    with pytest.raises(common.ValidationError, match="cleanup"):
        s.cleanup()
    report = json.loads((s.evidence / "cleanup.json").read_text())
    assert "survived SIGKILL" in report["owned_processes"][0]["error"]
    assert report["owned_processes"][1]["exit_code"] == -9
