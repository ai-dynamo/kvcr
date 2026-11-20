# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import hashlib
import json

import msgspec
import pytest
from clusterbench import launch_local
from clusterbench.__main__ import main
from clusterbench.config import RunConfig, load_config
from clusterbench.coordinator import Coordinator, RunFailed
from clusterbench.metrics import LogHist, Window
from clusterbench.report import compare, format_detail, format_summary, summarize
from clusterbench.trace import compile_corpus
from clusterbench.workload import make_workload


def _inputs(workers=2, profile="vllm", **phases):
    config = msgspec.to_builtins(RunConfig())
    config["engine"]["profile"] = profile
    config["phases"].update(phases)
    tally = Coordinator._new_tally()
    lag = LogHist()
    lag.add(20e-6)
    tally.update(
        lag=lag.to_json(),
        workload={"kind": "synthetic"},
        sessions_started=8,
        inflight_max=4,
    )
    tally["offered"].update(dispatched=100, dispatched_blocks=50_000, hinted=10)
    tally["completed"].update(completed=100)
    coord = {
        "name": "t",
        "arm": "base",
        "kvcr_sha": "abc",
        "kvcr_file": "/k",
        "nodes": [f"n{worker % 2}" for worker in range(workers)],
        "warm": {"ok": 2, "failed": 0, "max_seconds": 0.1},
        "coordinator": tally,
    }
    reports = []
    for worker in range(workers):
        window = Window()
        window.call("query", 0.004, calls=4000, items=4000)
        window.call("deliver", 0.002, calls=2, items=1000)
        window.counters.update(
            requests=50, deliver_blocks_remote=1000, deposit_blocks_ok=1000
        )
        window.counters.update(forward_seconds=2.0, idle_seconds=1.0)
        window.kvcr_counters["kvcr_transfer_blocks|local_fill"] = 1000
        for name in ("ttft", "step_kvcr", "deliver_remote"):
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
            profiled=False,
        )
        reports.append(report)
    return config, coord, reports


def _cells(text, label):
    """A comparison row after its label: per arm the median and its spread,
    then each later arm's change."""
    (row,) = [line for line in text.splitlines() if line.startswith(label)]
    return row[len(label) :].split()


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
    assert abs(headline["engine_kvcr_us_per_request"] - 120.0) < 1e-9
    assert headline["engine_forward_share"] == 0.2
    assert headline["engine_idle_share"] == 0.1
    text = format_summary(summary)
    assert text.splitlines()[1] == "VALID"
    assert "modelled GPU 20.0%, idle 10.0%" in text
    assert "mesh warm: 2 ok" in format_detail(summary)


def test_harness_is_what_is_left_of_the_engine_threads_window():
    # vLLM: of 20 worker-seconds, 4 s of forward pass, 2 s idle, 12 ms in KVCR.
    headline = summarize(*_inputs())["headline"]
    assert abs(headline["engine_harness_share"] - 13.988 / 20) < 1e-12
    assert abs(headline["engine_harness_us_per_request"] - 139_880) < 1e-6
    # SGLang's KVCR calls are on the owner thread: its engine thread spends
    # 1 s blocked on the linker and 3 s waiting for loads instead.
    config, coord, reports = _inputs(profile="sglang")
    for report in reports:
        report["counters"].update(linker_wait_seconds=0.5, load_wait_seconds=1.5)
    headline = summarize(config, coord, reports)["headline"]
    assert headline["engine_harness_share"] == 0.5
    assert headline["engine_harness_us_per_request"] == 100_000


def test_deposits_that_copied_nothing_are_not_moved_blocks():
    config, coord, reports = _inputs()
    for report in reports:
        # KVCR already held 400 of the 1,000 blocks each worker deposited.
        report["kvcr_counters"]["kvcr_transfer_blocks|local_fill"] = 600
    summary = summarize(config, coord, reports)
    headline = summary["headline"]
    # Over 2 x (1,000 delivered + 600 copied): 12 ms of KVCR calls, 0.4 s of
    # the progress threads and 20 s of the native ones.
    assert abs(headline["engine_kvcr_us_per_moved_block"] - 3.75) < 1e-9
    assert abs(headline["progress_us_per_moved_block"] - 125.0) < 1e-9
    assert abs(headline["deposit_already_held_share"] - 0.4) < 1e-12
    assert (
        "native 6,250 us; 40% of deposited blocks were already in KVCR's pool"
        in format_summary(summary)
    )
    # A copy counted in the window for a deposit accepted before it.
    for report in reports:
        report["counters"]["deposit_blocks_ok"] = 500
    headline = summarize(config, coord, reports)["headline"]
    assert headline["deposit_already_held_share"] == 0
    for report in reports:
        del report["counters"]["deposit_blocks_ok"]
    headline = summarize(config, coord, reports)["headline"]
    assert headline["deposit_already_held_share"] == 0


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
    reports[1]["profiled"] = True
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
        "profiled (CB_PROFILE_DIR)",
    ):
        assert expected in flags, expected
    assert "INVALID" in format_summary(summarize(config, coord, reports))


def test_one_worker_waiting_for_a_cpu_is_flagged_and_named_among_quiet_ones():
    config, coord, reports = _inputs(workers=8)
    reports[5]["threads"]["engine"]["delay_s"] = 0.5  # of its 10 s window
    reports[2]["threads"]["progress"]["delay_s"] = 0.3
    reports[6]["threads"]["progress"]["delay_s"] = 0.4
    summary = summarize(config, coord, reports)
    assert summary["flags"] == [
        "engine thread waited for a CPU 5.0% of the window on worker 5 "
        "(1 of 8 workers over 1%)",
        "progress thread waited for a CPU 4.0% of the window on worker 6 "
        "(2 of 8 workers over 1%)",
    ]
    # The thread table has the average over the workers, which is under 1%.
    assert summary["threads"]["engine"]["run_delay_share"] == 0.5 / 80


def test_cold_connects_are_not_flagged_when_no_mesh_warm_was_asked_for():
    config, coord, reports = _inputs(mesh_warm="none")
    connect = LogHist()
    connect.add(0.01)
    reports[0]["kvcr_hists"]["kvcr_duration_seconds|peer_setup|connected"] = (
        connect.to_json()
    )
    assert summarize(config, coord, reports)["valid"]


def test_an_sglang_summary_prints_the_owner_and_scheduler_threads():
    config, coord, reports = _inputs(profile="sglang")
    wait = LogHist()
    wait.add(0.004)
    for report in reports:
        report["counters"].update(
            linker_wait_seconds=0.5,
            load_wait_seconds=1.5,
            owner_idle_seconds=6.0,
            owner_spin_seconds=1.0,
            owner_loops=5000,
            aborted_requests=2,
            load_batches=40,
            load_batches_failed=1,
        )
        report["hists"]["step_load_wait"] = report["hists"]["deposit"] = wait.to_json()
    coord["coordinator"]["completed"]["aborted"] = 4
    lines = format_summary(summarize(config, coord, reports)).splitlines()
    assert lines[0].startswith("t  profile=sglang  arm=base  kvcr=abc  2 workers")
    at = lines.index("") + 6  # after the request, block, latency and failure lines
    assert lines[at : at + 5] == [
        "owner thread: KVCR 0.1% of wall (0.12 ms/request), linker 29.9%, "
        "yielding 10.0%, idle 60.0%; 500.0 loops/s",
        "scheduler thread: blocked on linker calls 5.0% (10.0 ms/request), "
        "waiting for loads 15.0% (30.0 ms/request), harness 50.0% "
        "(100.0 ms/request), modelled GPU 20.0%, idle 10.0%",
        "per busy iteration, blocked on linker calls p50/p99 10.0/10.0 ms; "
        "per loading batch p50/p99 4/4 ms",
        "aborted requests 4 (3.85%); load batches failed 2.50%; backup p50/p99 4/4 ms",
        "per moved block: KVCR 3 us, progress 100.0 us, native 5,000 us; 0% of "
        "deposited blocks were already in KVCR's pool",
    ]


def test_a_trace_summary_prints_what_was_replayed(tmp_path):
    source = tmp_path / "corpus.jsonl"
    request = {"t": 0, "type": "s", "in": 128, "out": 64, "hash_ids": [1, 2]}
    trace = {"id": "t0", "block_size": 64, "requests": [request]}
    source.write_text(json.dumps(trace) + "\n")
    replay = tmp_path / "replay.cbtrace"
    compile_corpus(source, replay)
    workload = make_workload(load_config(None, [f"workload.trace={replay}"]))
    workload.next_turn(workload.new_session().id)
    config, coord, reports = _inputs()
    coord["coordinator"]["workload"] = workload.summary()
    lines = format_summary(summarize(config, coord, reports)).splitlines()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    assert lines[1:4] == [
        "VALID",
        f"trace corpus.jsonl ({digest}, 1 traces): 1 plays started, 0 finished, "
        "1 turns sent (0 by sub-agents), at most 4 requests in flight",
        "",
    ]


def test_compare_reports_change_against_the_first_arm_and_work_mismatch():
    base = summarize(*_inputs())
    config, coord, reports = _inputs()
    for report in reports:
        report["secs"]["query"] *= 2
    coord["arm"] = "cand"
    coord["coordinator"]["offered"]["dispatched_blocks"] = 60_000
    cand = summarize(config, coord, reports)
    text = compare({"base": [base, base], "cand": [cand]})
    assert "WARNING: cand was offered 20.0% different work than base" in text
    assert _cells(text, "  query us / item") == ["1", "(0%)", "2", "(0%)", "+100.0%"]


def test_compare_lists_the_calls_the_runs_made():
    base = summarize(*_inputs())
    config, coord, reports = _inputs()
    # 1,000 polls on one worker took 3 ms and completed ten transfers.
    reports[0]["secs"]["poll"] = 0.003
    reports[0]["calls"]["poll"] = 1000
    reports[0]["items"]["poll"] = 10
    cand = summarize(config, coord, reports)
    text = compare({"base": [base], "cand": [cand]})
    calls = text.partition("engine thread, by KVCR call:\n")[2].splitlines()
    assert [line[:32].rstrip() for line in calls] == [
        "  deliver us / item",
        "  poll us / call",  # most polls complete nothing: their cost is per call
        "  query us / item",
    ]
    # The baseline never polled, so there is nothing to take a percentage of.
    assert _cells(text, "  poll us / call") == ["0", "(0%)", "3", "(0%)", "n/a"]


def test_a_change_from_zero_is_not_a_percentage():
    clean = summarize(*_inputs())
    config, coord, reports = _inputs()
    for report in reports:
        report["counters"]["hinted_failed|timeout"] = 10  # of 1,010 asked for
    failing = summarize(config, coord, reports)
    text = compare({"base": [clean], "cand": [failing]})
    rate = "hinted-block failure rate"
    assert _cells(text, rate) == ["0", "(0%)", "0.0099", "(0%)", "n/a"]
    # Zero in both arms is no change.
    unavailable = "hinted blocks unavailable"
    assert _cells(text, unavailable) == ["0", "(0%)", "0", "(0%)", "+0.0%"]
    # Nor is the spread of runs around a zero median a percentage.
    text = compare({"base": [clean, failing, clean], "cand": [failing]})
    assert _cells(text, rate) == ["0", "(n/a)", "0.0099", "(0%)", "n/a"]


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
    assert _cells(text, "  query us / item") == ["1", "(0%)", "1", "(0%)", "+0.0%"]
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


def _saved(directory, summary):
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps(summary))
    return directory


def test_report_arm_exits_0_for_comparable_arms_and_2_for_an_arm_with_no_valid_run(
    tmp_path, capsys
):
    config, coord, reports = _inputs()
    reports[0]["profiled"] = True
    good = _saved(tmp_path / "good", summarize(*_inputs()))
    bad = _saved(tmp_path / "bad", summarize(config, coord, reports))
    assert main(["report", "--arm", f"base={good}", "--arm", f"cand={bad},{good}"]) == 0
    printed = capsys.readouterr().out
    assert "cand: 2 runs, kvcr abc, 1 INVALID and left out" in printed
    assert _cells(printed, "requests / s") == ["10.0", "(0%)", "10.0", "(0%)", "+0.0%"]
    assert main(["report", "--arm", f"base={good}", "--arm", f"cand={bad}"]) == 2
    printed = capsys.readouterr().out
    assert printed.endswith("cand has no valid run: nothing to compare\n")


def test_ab_carries_on_past_a_cell_that_failed(tmp_path, monkeypatch, capsys):
    cells, failing = [], {"cand-0"}

    def run_local(config, out, *, arm, **kwargs):
        cells.append(out.name)
        if out.name in failing:
            raise RunFailed("only 3 of 4 workers registered")
        config, coord, reports = _inputs()
        coord["arm"] = arm
        return summarize(config, coord, reports)

    monkeypatch.setattr(launch_local, "run_local", run_local)
    argv = ["ab", "--arm", "base=/a", "--arm", "cand=/b", "--reps", "2"]
    assert main([*argv, "--out", str(tmp_path)]) == 0
    assert cells == ["base-0", "cand-0", "cand-1", "base-1"]
    printed = capsys.readouterr().out
    assert "[ab] rep 0 arm cand failed: only 3 of 4 workers registered" in printed
    assert "base: 2 runs, kvcr abc\ncand: 1 runs, kvcr abc\n" in printed
    # An arm whose every cell failed leaves nothing to compare.
    failing.add("cand-1")
    assert main([*argv, "--out", str(tmp_path)]) == 2
    assert capsys.readouterr().out.endswith(
        "cand has no valid run: nothing to compare\n"
    )
