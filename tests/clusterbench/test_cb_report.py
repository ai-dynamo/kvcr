# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import msgspec
import pytest
from clusterbench.config import RunConfig
from clusterbench.coordinator import Coordinator
from clusterbench.metrics import LogHist, Window
from clusterbench.report import compare, format_detail, format_summary, summarize


def _inputs(**phases):
    config = msgspec.to_builtins(RunConfig())
    config["phases"].update(phases)
    tally = Coordinator._new_tally()
    lag = LogHist()
    lag.add(20e-6)
    tally.update(lag=lag.to_json())
    tally.update(dispatched=100, dispatched_blocks=50_000, hinted=10, completed=100)
    coord = {
        "name": "t",
        "arm": "base",
        "kvcr_sha": "abc",
        "kvcr_file": "/k",
        "nodes": ["n0", "n1"],
        "warm": {"ok": 2, "failed": 0, "max_seconds": 0.1},
        "coordinator": tally,
    }
    reports = []
    for worker in range(2):
        window = Window()
        window.call("query", 0.004, calls=4000, items=4000)
        window.call("deliver", 0.002, calls=2, items=1000)
        window.counters.update(
            requests=50, deliver_blocks_remote=1000, deposit_blocks_ok=1000
        )
        window.counters.update(forward_seconds=2.0, idle_seconds=1.0)
        for name in ("ttft", "step_kvcr", "step_scheduler", "deliver_remote"):
            window.hists[name].add(0.01)
        window.hists["cache_resolve_remote"].add(0.02)
        report = window.to_json()
        report.update(
            window_s=10.0,
            worker=worker,
            threads={
                "engine": {"cpu_s": 1.0, "delay_s": 0.0, "threads": 1},
                "progress": {"cpu_s": 0.2, "delay_s": 0.0, "threads": 1},
                "native": {"cpu_s": 10.0, "delay_s": 0.0, "threads": 30},
            },
            engine={"primary_occupancy": 0.99},
            rss_mb=100.0,
            drained=True,
        )
        reports.append(report)
    return config, coord, reports


def test_a_clean_run_is_valid_and_normalised():
    summary = summarize(*_inputs())
    assert summary["valid"] and summary["flags"] == []
    assert summary["workers"] == 2 and summary["window_s"] == 10.0
    query = summary["engine_calls"]["query"]
    assert query["calls"] == 8000 and abs(query["us_per_item"] - 1.0) < 1e-9
    headline = summary["headline"]
    # 12 ms of engine time in KVCR over 4000 moved blocks and 20 worker-seconds.
    assert abs(headline["engine_kvcr_us_per_moved_block"] - 3.0) < 1e-9
    assert abs(headline["engine_kvcr_share"] - 0.012 / 20) < 1e-12
    assert abs(headline["progress_us_per_moved_block"] - 100.0) < 1e-9
    assert headline["requests_per_s"] == 10.0
    # The rest of the engine thread: 10 ms per step of scheduler time less the
    # 6 ms in KVCR, the modelled forward pass, and idle steps.
    assert abs(headline["engine_harness_share"] - 0.004 / 10) < 1e-12
    assert abs(headline["engine_harness_us_per_request"] - 80.0) < 1e-9
    assert abs(headline["engine_kvcr_us_per_request"] - 120.0) < 1e-9
    assert headline["engine_forward_share"] == 0.2
    assert headline["engine_idle_share"] == 0.1
    text = format_summary(summary)
    assert "VALID" in text and "modelled GPU 20.0%, idle 10.0%" in text
    # Summaries written before these fields existed still print.
    for key in ("engine_forward_share", "engine_idle_share"):
        del summary["headline"][key]
    del summary["headline"]["engine_harness_us_per_request"]
    assert "modelled GPU 0.0%" in format_summary(summary)
    assert "mesh warm: 2 ok" in format_detail(summary)


def test_each_problem_raises_its_own_flag():
    config, coord, reports = _inputs(min_primary_occupancy=0.9)
    coord["coordinator"]["primary_occupancy_min"] = 0.4
    coord["warm"]["failed"] = 3
    slow = LogHist()
    slow.add(0.5)
    coord["coordinator"]["lag"] = slow.to_json()
    reports[0]["counters"]["bytes_mismatch"] = 2
    reports[0]["counters"]["resilience_events"] = 1
    coord["coordinator"].update(measure_s=10.0, busy_s=9.5)
    reports[0]["threads"]["engine"]["delay_s"] = 1.0
    reports[1]["drained"] = False
    connect = LogHist()
    connect.add(0.01)
    reports[1]["kvcr_hists"]["kvcr_duration_seconds|peer_setup|connected"] = (
        connect.to_json()
    )
    flags = " | ".join(summarize(config, coord, reports)["flags"])
    for expected in (
        "wrong bytes",
        "dispatch lag",
        "engine thread waited for a CPU",
        "first-contact peer connects",
        "mesh-warm pulls failed",
        "resilience events",
        "the coordinator was busy 95% of the window",
        "in flight at exit",
        "only filled to 40% at window start",
    ):
        assert expected in flags, expected
    assert "INVALID" in format_summary(summarize(config, coord, reports))


def test_a_missing_report_and_cold_connects_without_warm():
    config, coord, reports = _inputs(mesh_warm="none")
    connect = LogHist()
    connect.add(0.01)
    reports[0]["kvcr_hists"]["kvcr_duration_seconds|peer_setup|connected"] = (
        connect.to_json()
    )
    assert summarize(config, coord, reports)["valid"]  # cold connects were asked for
    summary = summarize(config, coord, reports[:1])
    assert summary["flags"] == ["reports from 1 of 2 workers"]


def test_compare_reports_change_against_the_first_arm_and_work_mismatch():
    base = summarize(*_inputs())
    config, coord, reports = _inputs()
    for report in reports:
        report["secs"]["query"] *= 2
    coord["arm"] = "cand"
    coord["coordinator"]["dispatched_blocks"] = 60_000
    cand = summarize(config, coord, reports)
    text = compare({"base": [base, base], "cand": [cand]})
    assert "WARNING: cand was offered 20.0% different work than base" in text
    row = next(
        line for line in text.splitlines() if line.startswith("  query us / item")
    )
    assert row.rstrip().endswith("+100.0%")


def test_only_valid_runs_are_compared():
    good = summarize(*_inputs())
    config, coord, reports = _inputs()
    for report in reports:
        report["secs"]["query"] *= 5
        report["counters"]["bytes_mismatch"] = 1  # which makes the run invalid
    bad = summarize(config, coord, reports)
    assert not bad["valid"]
    # Two invalid outliers of three do not move the arm.
    text = compare({"base": [good], "cand": [bad, good, bad]})
    assert "cand: 3 runs, kvcr abc, 2 INVALID and left out" in text
    row = next(
        line for line in text.splitlines() if line.startswith("  query us / item")
    )
    assert row.rstrip().endswith("+0.0%")
    # An arm with nothing valid cannot be compared at all.
    with pytest.raises(ValueError, match="cand has no valid run"):
        compare({"base": [good], "cand": [bad, bad]})
    with pytest.raises(ValueError, match="cand has no valid run"):
        compare({"base": [good], "cand": []})


def test_hinted_blocks_a_peer_did_not_deliver_are_reported_by_kvcr_status():
    config, coord, reports = _inputs()
    for report in reports:
        report["counters"].update(
            {
                "deliver_blocks_failed": 30,
                "hinted_failed|timeout": 4,
                "hinted_failed|unavailable": 9,
                "hinted_failed|aborted": 2,
                "hinted_failed_reason|not_resident": 9,
                "deliver_blocks_failed_beyond_hint": 10,
                "pin_requests": 40,
                "pin_misses": 9,
            }
        )
    summary = summarize(config, coord, reports)
    failures = summary["failures"]
    assert failures["causes"] == {"timeout": 8, "unavailable": 18, "aborted": 4}
    assert failures["beyond_hint"] == 20  # asked for past what the router named
    assert failures["local"] == 10  # this worker's own pool, no peer involved
    # Of the 2,030 hinted blocks asked of a peer.
    headline = summary["headline"]
    assert abs(headline["hinted_block_failure_rate"] - 30 / 2030) < 1e-12
    assert abs(headline["hinted_timeout_rate"] - 8 / 2030) < 1e-12
    assert abs(headline["hinted_unavailable_rate"] - 18 / 2030) < 1e-12
    text = format_summary(summary)
    assert (
        "hinted blocks a peer did not deliver, by KVCR status: timeout 8 "
        "(0.39% of hinted blocks), unavailable 18 (0.89%), aborted 4; "
        "beyond the hint 20; source pins 80 requested, 18 missed, 0 timed out"
    ) in text
    assert "hinted blocks timed out" in compare({"a": [summary], "b": [summary]})
