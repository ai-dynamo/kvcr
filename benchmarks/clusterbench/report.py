# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Turn a run's raw reports into a summary, validity flags and A/B tables."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from .metrics import LogHist

# Flag thresholds.
MAX_COORDINATOR_LAG_P99_S = 1e-3
# The coordinator is one thread: when it has no slack it limits the load.
MAX_COORDINATOR_BUSY = 0.8
MAX_RUN_DELAY_SHARE = 0.01
# Closed-loop sessions make offered work vary a little run to run (about 2%
# between identical arms on a quiet cluster).
MAX_OFFERED_WORK_DIFF = 0.05

KVCR_CALLS = (
    "submit_hint",
    "query",
    "deliver",
    "deposit",
    "poll",
    "poll_empty",
    "align_sequence",
    "discard_hint",
    "get_stats",
)


def _merge_hists(reports: list[dict], section: str) -> dict[str, LogHist]:
    merged: dict[str, LogHist] = defaultdict(LogHist)
    for report in reports:
        for name, data in report.get(section, {}).items():
            merged[name].merge(LogHist.from_json(data))
    return merged


def _sum(reports: list[dict], section: str) -> dict[str, float]:
    total: dict[str, float] = defaultdict(float)
    for report in reports:
        for name, value in report.get(section, {}).items():
            total[name] += value
    return dict(total)


def summarize(config: dict, coord: dict, reports: list[dict]) -> dict:
    """Aggregate per-worker reports. All inputs and the result are plain JSON."""
    tally = coord["coordinator"]
    nodes = coord["nodes"]
    workers = len(nodes)
    window = sum(r["window_s"] for r in reports)  # worker-seconds
    mean_window = window / len(reports) if reports else 0.0
    secs, calls, items = (_sum(reports, k) for k in ("secs", "calls", "items"))
    counters = _sum(reports, "counters")
    hists = _merge_hists(reports, "hists")
    kvcr_hists = _merge_hists(reports, "kvcr_hists")
    kvcr_counters = _sum(reports, "kvcr_counters")

    engine_calls = {}
    for name in sorted(secs):
        engine_calls[name] = {
            "seconds": secs[name],
            "calls": calls.get(name, 0),
            "items": items.get(name, 0),
            "us_per_call": secs[name] / calls[name] * 1e6 if calls.get(name) else 0,
            "us_per_item": secs[name] / items[name] * 1e6 if items.get(name) else 0,
            "share": secs[name] / window if window else 0,
        }
    kvcr_seconds = sum(secs.values())
    if "engine_harness_seconds" in counters:
        # Engines whose KVCR calls are on another thread say it outright.
        harness_seconds = counters["engine_harness_seconds"]
    else:
        scheduler_seconds = (
            hists["step_scheduler"].total if "step_scheduler" in hists else 0.0
        )
        harness_seconds = max(0.0, scheduler_seconds - kvcr_seconds)

    threads: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for report in reports:
        for role, entry in report.get("threads", {}).items():
            role = "py" if role.startswith("py:") else role
            for key, value in entry.items():
                threads[role][key] += value
    thread_summary = {
        role: {
            "cpu_s": entry["cpu_s"],
            "cores_per_worker": entry["cpu_s"] / window if window else 0,
            "run_delay_share": entry["delay_s"] / window if window else 0,
            "threads_per_worker": entry["threads"] / len(reports) if reports else 0,
        }
        for role, entry in threads.items()
    }

    requests = counters.get("requests", 0)
    remote = counters.get("deliver_blocks_remote", 0)
    local = counters.get("deliver_blocks_local", 0)
    failed = counters.get("deliver_blocks_failed_hinted", 0)
    deposited = counters.get("deposit_blocks_ok", 0)
    moved = remote + local + deposited
    progress_cpu = threads.get("progress", {}).get("cpu_s", 0.0)
    native_cpu = threads.get("native", {}).get("cpu_s", 0.0)

    headline = {
        "requests_per_s": requests / mean_window if mean_window else 0,
        "remote_blocks_per_s": remote / mean_window if mean_window else 0,
        "engine_kvcr_share": kvcr_seconds / window if window else 0,
        "engine_harness_share": harness_seconds / window if window else 0,
        # The modelled forward pass, and steps with nothing to run.
        "engine_forward_share": counters.get("forward_seconds", 0) / window
        if window
        else 0,
        "engine_idle_share": counters.get("idle_seconds", 0) / window if window else 0,
        "engine_kvcr_us_per_request": kvcr_seconds / requests * 1e6 if requests else 0,
        "engine_harness_us_per_request": harness_seconds / requests * 1e6
        if requests
        else 0,
        "engine_kvcr_us_per_moved_block": kvcr_seconds / moved * 1e6 if moved else 0,
        "progress_us_per_moved_block": progress_cpu / moved * 1e6 if moved else 0,
        "native_us_per_moved_block": native_cpu / moved * 1e6 if moved else 0,
        "step_kvcr_p50_ms": hists["step_kvcr"].quantile(0.5) * 1e3,
        "step_kvcr_p99_ms": hists["step_kvcr"].quantile(0.99) * 1e3,
        "ttft_p50_ms": hists["ttft"].quantile(0.5) * 1e3,
        "ttft_p99_ms": hists["ttft"].quantile(0.99) * 1e3,
        "remote_resolve_p50_ms": hists["cache_resolve_remote"].quantile(0.5) * 1e3,
        "remote_resolve_p99_ms": hists["cache_resolve_remote"].quantile(0.99) * 1e3,
        "deliver_remote_p50_ms": hists["deliver_remote"].quantile(0.5) * 1e3,
        "deliver_remote_p99_ms": hists["deliver_remote"].quantile(0.99) * 1e3,
        # Hinted blocks only: blocks past the hint are requested too (vLLM's
        # opportunistic query) and are expected to fail.
        "hinted_block_failure_rate": failed / (remote + failed)
        if remote + failed
        else 0,
    }

    # Why blocks asked of a peer failed. Older KVCR trees report only success
    # or failure, so the engines sort failures by how long the deliver ran: at
    # least the operation timeout, or less (the source answered that it no
    # longer held the block, or the fetch failed at once).
    from_peer = counters.get("deliver_blocks_failed_peer", 0)
    from_peer_hinted = counters.get("deliver_blocks_failed_peer_hinted", 0)
    timed_out = counters.get("deliver_blocks_timed_out", 0)
    timed_out_hinted = counters.get("deliver_blocks_timed_out_hinted", 0)
    pin_timeouts = kvcr_hists.get("kvcr_duration_seconds|framework_pin_wait|timeout")
    failures = {
        "timed_out": timed_out,
        "gone_from_source": from_peer_hinted - timed_out_hinted,
        "beyond_hint": from_peer - from_peer_hinted - (timed_out - timed_out_hinted),
        "local": counters.get("deliver_blocks_failed", 0) - from_peer,
        # Cross-check from KVCR's own timings of peer operations.
        "kvcr_peer_ops_past_timeout": counters.get("kvcr_peer_ops_past_timeout", 0),
        "failed_deliver_p50_ms": hists["deliver_peer_failed"].quantile(0.5) * 1e3,
        "failed_deliver_p99_ms": hists["deliver_peer_failed"].quantile(0.99) * 1e3,
        # The source side, where sources serve from pinned engine memory.
        "pin_requests": counters.get("pin_requests", 0),
        "pin_misses": counters.get("pin_misses", 0),
        "pin_wait_timeouts": pin_timeouts.n if pin_timeouts is not None else 0,
    }
    hinted_asked = remote + from_peer_hinted
    headline["hinted_timeout_rate"] = (
        timed_out_hinted / hinted_asked if hinted_asked else 0
    )
    headline["hinted_gone_rate"] = (
        failures["gone_from_source"] / hinted_asked if hinted_asked else 0
    )

    profile = config.get("engine", {}).get("profile", "vllm")
    aborted = counters.get("aborted_requests", 0)
    if profile == "sglang":
        # The linker's owner thread makes the KVCR calls; the scheduler thread
        # blocks on linker calls and on loads.
        idle = counters.get("owner_idle_seconds", 0)
        spin = counters.get("owner_spin_seconds", 0)
        batches = counters.get("load_batches", 0)
        headline.update(
            {
                "linker_wait_share": counters.get("linker_wait_seconds", 0) / window
                if window
                else 0,
                "load_wait_share": counters.get("load_wait_seconds", 0) / window
                if window
                else 0,
                "linker_wait_us_per_request": counters.get("linker_wait_seconds", 0)
                / requests
                * 1e6
                if requests
                else 0,
                "load_wait_us_per_request": counters.get("load_wait_seconds", 0)
                / requests
                * 1e6
                if requests
                else 0,
                "owner_linker_share": max(0.0, window - idle - spin - kvcr_seconds)
                / window
                if window
                else 0,
                "owner_spin_share": spin / window if window else 0,
                "owner_idle_share": idle / window if window else 0,
                "owner_loops_per_s": counters.get("owner_loops", 0) / window
                if window
                else 0,
                "abort_rate": aborted / (aborted + requests)
                if aborted + requests
                else 0,
                "load_batch_failure_rate": counters.get("load_batches_failed", 0)
                / batches
                if batches
                else 0,
                "step_load_wait_p50_ms": hists["step_load_wait"].quantile(0.5) * 1e3,
                "step_load_wait_p99_ms": hists["step_load_wait"].quantile(0.99) * 1e3,
                "offload_p50_ms": hists["deposit"].quantile(0.5) * 1e3,
                "offload_p99_ms": hists["deposit"].quantile(0.99) * 1e3,
                "reoffered_restored_share": counters.get("reoffered_restored_pages", 0)
                / counters["offload_pages"]
                if counters.get("offload_pages")
                else 0,
            }
        )

    lag = LogHist.from_json(tally["lag"])
    cold = sum(
        h.n
        for name, h in kvcr_hists.items()
        if name.startswith("kvcr_duration_seconds|peer_setup|connected")
    )
    warm_full = config["phases"]["mesh_warm"] == "full" and workers > 1
    flags = []
    if len(reports) != workers:
        flags.append(f"reports from {len(reports)} of {workers} workers")
    if counters.get("bytes_mismatch", 0):
        flags.append(
            f"{int(counters['bytes_mismatch'])} delivered blocks had wrong bytes"
        )
    if lag.quantile(0.99) > MAX_COORDINATOR_LAG_P99_S:
        flags.append(f"coordinator dispatch lag p99 {lag.quantile(0.99) * 1e3:.2f} ms")
    measure_s = tally.get("measure_s", 0)
    coordinator_busy = tally.get("busy_s", 0) / measure_s if measure_s else 0.0
    if coordinator_busy > MAX_COORDINATOR_BUSY:
        flags.append(f"the coordinator was busy {coordinator_busy:.0%} of the window")
    for role in ("engine", "owner", "progress"):
        share = thread_summary.get(role, {}).get("run_delay_share", 0)
        if share > MAX_RUN_DELAY_SHARE:
            flags.append(f"{role} thread waited for a CPU {share:.1%} of the window")
    if warm_full and cold:
        flags.append(f"{cold} first-contact peer connects inside the window")
    if warm_full and coord["warm"]["failed"]:
        flags.append(f"{coord['warm']['failed']} mesh-warm pulls failed")
    if counters.get("resilience_events", 0):
        flags.append(f"{int(counters['resilience_events'])} KVCR resilience events")
    if not all(r.get("drained", True) for r in reports):
        flags.append("some workers still had operations in flight at exit")
    occupancy = tally.get("primary_occupancy_min", 1.0)
    if occupancy < config["phases"].get("min_primary_occupancy", 0.0):
        flags.append(f"a host tier had only filled to {occupancy:.0%} at window start")
    if not requests:
        flags.append("no requests completed in the window")

    return {
        "name": coord["name"],
        "profile": profile,
        "arm": coord["arm"],
        "kvcr_sha": coord["kvcr_sha"],
        "kvcr_file": coord["kvcr_file"],
        "workers": workers,
        "nodes": len(set(nodes)),
        "window_s": mean_window,
        "valid": not flags,
        "flags": flags,
        "headline": headline,
        "failures": failures,
        "workload": {
            **tally.get("workload", {"kind": "synthetic"}),
            "sessions_started": tally.get("sessions_started", 0),
            "inflight_max": tally.get("inflight_max", 0),
        },
        "offered": {
            k: tally[k]
            for k in (
                "dispatched",
                "dispatched_blocks",
                "dispatched_tokens",
                "hinted",
                "hinted_blocks",
                "hinted_same_node",
                "hinted_cross_node",
                "migrations",
            )
        },
        "completed": {
            k: tally.get(k, 0)
            for k in (
                "completed",
                "aborted",
                "prompt_blocks",
                "gpu_blocks",
                "primary_blocks",
                "local_blocks",
                "remote_requests",
                "remote_blocks_same_node",
                "remote_blocks_cross_node",
                "failed_blocks",
                "timed_out_blocks",
                "prefill_tokens",
            )
        },
        "engine_calls": engine_calls,
        "counters": counters,
        "latency": {name: hist.summary() for name, hist in sorted(hists.items())},
        "threads": thread_summary,
        "kvcr": {name: hist.summary() for name, hist in sorted(kvcr_hists.items())},
        "kvcr_counters": kvcr_counters,
        "warm": coord["warm"],
        "coordinator_lag": lag.summary(),
        "coordinator_busy": coordinator_busy,
        "rss_mb_mean": statistics.fmean(r.get("rss_mb", 0) for r in reports)
        if reports
        else 0,
        "engine_gauges": {
            key: statistics.fmean(r["engine"][key] for r in reports)
            for key in (reports[0]["engine"] if reports else {})
        },
    }


def load(run_dir: str | Path) -> dict:
    return json.loads((Path(run_dir) / "summary.json").read_text())


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def format_summary(summary: dict) -> str:
    h = summary["headline"]
    done = summary["completed"]
    offered = summary["offered"]
    sglang = summary.get("profile") == "sglang"
    if sglang:
        thread_lines = [
            f"owner thread: KVCR {h['engine_kvcr_share']:.1%} of wall "
            f"({_fmt(h['engine_kvcr_us_per_request'] / 1e3)} ms/request), linker "
            f"{h['owner_linker_share']:.1%}, yielding {h['owner_spin_share']:.1%}, "
            f"idle {h['owner_idle_share']:.1%}; "
            f"{_fmt(h['owner_loops_per_s'])} loops/s",
            f"scheduler thread: blocked on linker calls {h['linker_wait_share']:.1%} "
            f"({_fmt(h['linker_wait_us_per_request'] / 1e3)} ms/request), waiting "
            f"for loads {h['load_wait_share']:.1%} "
            f"({_fmt(h['load_wait_us_per_request'] / 1e3)} ms/request), harness "
            f"{h['engine_harness_share']:.1%} "
            f"({_fmt(h['engine_harness_us_per_request'] / 1e3)} ms/request), "
            f"modelled GPU {h['engine_forward_share']:.1%}, "
            f"idle {h['engine_idle_share']:.1%}",
            f"per busy iteration, blocked on linker calls p50/p99 "
            f"{_fmt(h['step_kvcr_p50_ms'])}/{_fmt(h['step_kvcr_p99_ms'])} ms; "
            f"per loading batch p50/p99 {_fmt(h['step_load_wait_p50_ms'])}/"
            f"{_fmt(h['step_load_wait_p99_ms'])} ms",
            f"aborted requests {summary['completed'].get('aborted', 0)} "
            f"({h['abort_rate']:.2%}); load batches failed "
            f"{h['load_batch_failure_rate']:.2%}; backup p50/p99 "
            f"{_fmt(h['offload_p50_ms'])}/{_fmt(h['offload_p99_ms'])} ms, "
            f"{h['reoffered_restored_share']:.1%} of backed-up pages had just "
            f"been restored",
        ]
    else:
        thread_lines = [
            f"engine thread: KVCR {h['engine_kvcr_share']:.1%} of wall "
            f"({_fmt(h['engine_kvcr_us_per_request'] / 1e3)} ms/request), harness "
            f"{h['engine_harness_share']:.1%} "
            f"({_fmt(h.get('engine_harness_us_per_request', 0) / 1e3)} ms/request), "
            f"modelled GPU {h.get('engine_forward_share', 0):.1%}, "
            f"idle {h.get('engine_idle_share', 0):.1%}",
            f"KVCR per engine step p50/p99 "
            f"{_fmt(h['step_kvcr_p50_ms'])}/{_fmt(h['step_kvcr_p99_ms'])} ms",
        ]
    lines = [
        f"{summary['name']}  profile={summary.get('profile', 'vllm')}  "
        f"arm={summary['arm'] or '-'}  "
        f"kvcr={summary['kvcr_sha']}  {summary['workers']} workers on "
        f"{summary['nodes']} nodes  window {summary['window_s']:.1f}s",
        "VALID" if summary["valid"] else "INVALID: " + "; ".join(summary["flags"]),
        "",
        f"requests/s {_fmt(h['requests_per_s'])}   "
        f"remote blocks/s {_fmt(h['remote_blocks_per_s'])}   "
        f"hinted {offered['hinted']} of {offered['dispatched']} requests "
        f"({offered['hinted_cross_node']} cross-node)",
        f"blocks: gpu {done['gpu_blocks']}  primary {done['primary_blocks']}  "
        f"kvcr-local {done['local_blocks']}  "
        f"remote {done['remote_blocks_same_node'] + done['remote_blocks_cross_node']}"
        f" ({done['remote_blocks_cross_node']} cross-node)  "
        f"failed {done['failed_blocks']} "
        f"(hinted-block failure rate {h['hinted_block_failure_rate']:.2%})",
        f"ttft p50/p99 {_fmt(h['ttft_p50_ms'])}/{_fmt(h['ttft_p99_ms'])} ms   "
        f"remote resolve p50/p99 {_fmt(h['remote_resolve_p50_ms'])}/"
        f"{_fmt(h['remote_resolve_p99_ms'])} ms   "
        f"remote deliver p50/p99 {_fmt(h['deliver_remote_p50_ms'])}/"
        f"{_fmt(h['deliver_remote_p99_ms'])} ms",
        "",
        *thread_lines,
        f"per moved block: KVCR {_fmt(h['engine_kvcr_us_per_moved_block'])} us, "
        f"progress {_fmt(h['progress_us_per_moved_block'])} us, "
        f"native {_fmt(h['native_us_per_moved_block'])} us",
        "",
        f"{'KVCR call':<16}{'calls':>10}{'items':>12}{'us/call':>10}"
        f"{'us/item':>10}{'share':>9}",
    ]
    failures = summary.get("failures")
    if failures:
        at = lines.index("", 3)  # after the blocks and latency lines
        cause = (
            f"blocks a peer failed to deliver: timed out "
            f"{failures['timed_out']:.0f} ({h['hinted_timeout_rate']:.2%} of "
            f"hinted blocks), no longer on the source "
            f"{failures['gone_from_source']:.0f} ({h['hinted_gone_rate']:.2%}), "
            f"beyond the hint {failures['beyond_hint']:.0f}; KVCR timed "
            f"{failures['kvcr_peer_ops_past_timeout']:.0f} peer operations past "
            f"its timeout"
        )
        if failures["pin_requests"]:
            cause += (
                f"; source pins {failures['pin_requests']:.0f} requested, "
                f"{failures['pin_misses']:.0f} missed, "
                f"{failures['pin_wait_timeouts']:.0f} timed out"
            )
        lines.insert(at, cause)
    workload = summary.get("workload", {})
    if workload.get("kind") == "trace":
        lines.insert(
            2,
            f"trace {workload['trace']} ({workload['sha256'][:12]}, "
            f"{workload['traces']} traces): {workload['plays_started']} plays "
            f"started, {workload['plays_finished']} finished, "
            f"{workload['turns']} turns sent ({workload['child_turns']} by "
            f"sub-agents), at most {workload['inflight_max']} requests in flight",
        )
    for name, call in summary["engine_calls"].items():
        lines.append(
            f"{name:<16}{call['calls']:>10,.0f}{call['items']:>12,.0f}"
            f"{_fmt(call['us_per_call']):>10}{_fmt(call['us_per_item']):>10}"
            f"{call['share']:>9.2%}"
        )
    lines.append("")
    lines.append(f"{'thread':<10}{'cores/worker':>14}{'run-delay':>12}")
    for role, entry in sorted(summary["threads"].items()):
        lines.append(
            f"{role:<10}{entry['cores_per_worker']:>14.3f}"
            f"{entry['run_delay_share']:>12.2%}"
        )
    return "\n".join(lines)


def format_detail(summary: dict) -> str:
    """Every latency histogram, KVCR's own telemetry and the raw counters."""

    def table(title: str, hists: dict) -> list[str]:
        lines = [f"{title:<50}{'n':>9}{'p50 ms':>10}{'p99 ms':>10}{'max ms':>10}"]
        for name, h in hists.items():
            lines.append(
                f"{name:<50}{h['n']:>9,}{_fmt(h['p50_ms']):>10}"
                f"{_fmt(h['p99_ms']):>10}{_fmt(h['max_ms']):>10}"
            )
        return lines

    lines = table("latency (worker clocks)", summary["latency"])
    lines.append("")
    kvcr = {
        name.removeprefix("kvcr_duration_seconds|").replace("|", " / "): h
        for name, h in summary["kvcr"].items()
    }
    lines += table("KVCR telemetry: scope / result", kvcr)
    lines.append("")
    lines.append(
        "counters: "
        + "  ".join(
            f"{name}={value:,.0f}"
            for name, value in sorted(summary["counters"].items())
        )
    )
    lines.append(
        "KVCR counters: "
        + "  ".join(
            f"{name}={value:,.0f}"
            for name, value in sorted(summary["kvcr_counters"].items())
        )
    )
    lines.append(
        "engine gauges at stop (mean): "
        + "  ".join(
            f"{name}={_fmt(value)}" for name, value in summary["engine_gauges"].items()
        )
    )
    warm, lag = summary["warm"], summary["coordinator_lag"]
    lines.append(
        f"mesh warm: {warm['ok']:.0f} ok, {warm['failed']:.0f} failed, slowest worker "
        f"{warm['max_seconds']:.1f}s;  coordinator lag p99 {_fmt(lag['p99_ms'])} ms, "
        f"busy {summary.get('coordinator_busy', 0):.0%};  "
        f"mean RSS {summary['rss_mb_mean']:.0f} MB"
    )
    return "\n".join(lines)


AB_METRICS = (
    ("engine_kvcr_us_per_moved_block", "engine KVCR us / moved block"),
    ("progress_us_per_moved_block", "progress us / moved block"),
    ("native_us_per_moved_block", "native us / moved block"),
    ("engine_kvcr_us_per_request", "engine KVCR us / request"),
    ("engine_kvcr_share", "engine KVCR share of wall"),
    ("step_kvcr_p99_ms", "KVCR per step p99 ms"),
    ("deliver_remote_p50_ms", "remote deliver p50 ms"),
    ("deliver_remote_p99_ms", "remote deliver p99 ms"),
    ("remote_resolve_p99_ms", "remote resolve p99 ms"),
    ("ttft_p99_ms", "ttft p99 ms"),
    ("hinted_block_failure_rate", "hinted-block failure rate"),
    ("hinted_timeout_rate", "hinted blocks timed out"),
    ("hinted_gone_rate", "hinted blocks gone from source"),
    ("requests_per_s", "requests / s"),
    ("remote_blocks_per_s", "remote blocks / s"),
)


def compare(arms: dict[str, list[dict]]) -> str:
    """A/B (or A/A) table: median per arm, spread, and change versus the first."""
    names = list(arms)
    base = names[0]
    lines = []
    for name in names:
        runs = arms[name]
        invalid = [s for s in runs if not s["valid"]]
        shas = sorted({s["kvcr_sha"] for s in runs})
        lines.append(
            f"{name}: {len(runs)} runs, kvcr {','.join(shas)}"
            + (f", {len(invalid)} INVALID" if invalid else "")
        )
        for s in invalid:
            lines.append(f"    {s['name']}: " + "; ".join(s["flags"]))
    offered = {
        name: statistics.median(s["offered"]["dispatched_blocks"] for s in arms[name])
        for name in names
    }
    for name in names[1:]:
        if offered[base]:
            diff = abs(offered[name] - offered[base]) / offered[base]
            if diff > MAX_OFFERED_WORK_DIFF:
                lines.append(
                    f"WARNING: {name} was offered {diff:.1%} different work "
                    f"than {base}; compare normalised rows only"
                )
    header = f"{'metric':<32}" + "".join(f"{n + ' (spread)':>24}" for n in names)
    header += "".join(f"{n + ' vs ' + base:>18}" for n in names[1:])
    lines += ["", header]

    def row(label: str, values: dict[str, list[float]]) -> str:
        medians = {n: statistics.median(v) for n, v in values.items()}
        out = f"{label:<32}"
        for n in names:
            v = values[n]
            spread = (max(v) - min(v)) / medians[n] if medians[n] else 0
            out += f"{_fmt(medians[n]) + f' ({spread:.0%})':>24}"
        for n in names[1:]:
            change = (medians[n] / medians[base] - 1) if medians[base] else 0
            out += f"{change:>+18.1%}"
        return out

    for key, label in AB_METRICS:
        lines.append(
            row(
                label,
                {n: [s["headline"].get(key, 0.0) for s in arms[n]] for n in names},
            )
        )
    lines.append("engine thread, by KVCR call:")
    for call in KVCR_CALLS:
        # poll's cost is per call (most polls complete nothing); the others
        # scale with the keys they are given.
        unit = "call" if call in ("poll", "poll_empty") else "item"
        values = {
            n: [
                s["engine_calls"].get(call, {}).get(f"us_per_{unit}", 0.0)
                for s in arms[n]
            ]
            for n in names
        }
        lines.append(row(f"  {call} us / {unit}", values))
    return "\n".join(lines)
