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
# Per worker: one starved worker must not hide among quiet ones.
MAX_RUN_DELAY_SHARE = 0.01
# Closed-loop sessions make offered work vary a little run to run.
MAX_OFFERED_WORK_DIFF = 0.05


def _ratio(part: float, whole: float) -> float:
    return part / whole if whole else 0.0


def _p50_p99(hist: LogHist, name: str) -> dict[str, float]:
    return {
        f"{name}_p50_ms": hist.quantile(0.5) * 1e3,
        f"{name}_p99_ms": hist.quantile(0.99) * 1e3,
    }


def _merge_hists(reports: list[dict], section: str) -> dict[str, LogHist]:
    merged: dict[str, LogHist] = defaultdict(LogHist)
    for report in reports:
        for name, data in report[section].items():
            merged[name].merge(LogHist.from_json(data))
    return merged


def _sum(reports: list[dict], section: str) -> dict[str, float]:
    total: dict[str, float] = defaultdict(float)
    for report in reports:
        for name, value in report[section].items():
            total[name] += value
    return dict(total)


def summarize(config: dict, coord: dict, reports: list[dict]) -> dict:
    """Aggregate per-worker reports. All inputs and the result are plain JSON."""
    tally = coord["coordinator"]
    nodes = coord["nodes"]
    workers = len(nodes)
    window = sum(r["window_s"] for r in reports)  # worker-seconds
    mean_window = window / workers
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
            "us_per_call": _ratio(secs[name], calls.get(name, 0)) * 1e6,
            "us_per_item": _ratio(secs[name], items.get(name, 0)) * 1e6,
            "share": _ratio(secs[name], window),
        }
    kvcr_seconds = sum(secs.values())
    forward = counters.get("forward_seconds", 0)
    idle = counters.get("idle_seconds", 0)
    # vLLM's own time: its scheduler calls, less the KVCR calls made inside
    # them. Harness is what is left of the engine thread's window, so the work
    # between engine steps counts too.
    vllm_seconds = max(0.0, counters.get("vllm_seconds", 0) - kvcr_seconds)
    harness_seconds = max(0.0, window - forward - idle - kvcr_seconds - vllm_seconds)

    threads: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for report in reports:
        for role, entry in report["threads"].items():
            role = "py" if role.startswith("py:") else role
            for key, value in entry.items():
                threads[role][key] += value
    thread_summary = {
        role: {
            "cpu_s": entry["cpu_s"],
            "cores_per_worker": _ratio(entry["cpu_s"], window),
            "run_delay_share": _ratio(entry["delay_s"], window),
            "threads_per_worker": _ratio(entry["threads"], workers),
        }
        for role, entry in threads.items()
    }

    requests = counters.get("requests", 0)
    remote = counters.get("deliver_blocks_remote", 0)
    local = counters.get("deliver_blocks_local", 0)
    deposited = counters.get("deposit_blocks_ok", 0)
    # KVCR accepts a deposit of blocks it already holds and copies nothing.
    copied = kvcr_counters.get("kvcr_transfer_blocks|local_fill", 0)
    moved = remote + local + copied
    progress_cpu = threads.get("progress", {}).get("cpu_s", 0.0)
    native_cpu = threads.get("native", {}).get("cpu_s", 0.0)

    # Hinted blocks a peer did not deliver, by KVCR's status ("unclassified"
    # on a tree that reports none). Blocks past the hint are asked for too
    # (vLLM's opportunistic query) and are expected to fail.
    causes = {
        name.partition("|")[2]: value
        for name, value in counters.items()
        if name.startswith("hinted_failed|")
    }
    hinted_failed = sum(causes.values())
    hinted_asked = remote + hinted_failed
    beyond_hint = counters.get("deliver_blocks_failed_beyond_hint", 0)
    pin_timeouts = kvcr_hists.get("kvcr_duration_seconds|framework_pin_wait|timeout")
    failures = {
        "causes": causes,
        "beyond_hint": beyond_hint,
        "local": counters.get("deliver_blocks_failed", 0) - hinted_failed - beyond_hint,
        # The source side, where sources serve from pinned engine memory.
        "pin_requests": counters.get("pin_requests", 0),
        "pin_misses": counters.get("pin_misses", 0),
        "pin_wait_timeouts": pin_timeouts.n if pin_timeouts is not None else 0,
    }

    headline = {
        "requests_per_s": _ratio(requests, mean_window),
        "remote_blocks_per_s": _ratio(remote, mean_window),
        "engine_kvcr_share": _ratio(kvcr_seconds, window),
        "engine_vllm_share": _ratio(vllm_seconds, window),
        "engine_harness_share": _ratio(harness_seconds, window),
        # The modelled forward pass, and steps with nothing to run.
        "engine_forward_share": _ratio(forward, window),
        "engine_idle_share": _ratio(idle, window),
        "engine_kvcr_us_per_request": _ratio(kvcr_seconds, requests) * 1e6,
        "engine_vllm_us_per_request": _ratio(vllm_seconds, requests) * 1e6,
        "engine_harness_us_per_request": _ratio(harness_seconds, requests) * 1e6,
        "engine_kvcr_us_per_moved_block": _ratio(kvcr_seconds, moved) * 1e6,
        "progress_us_per_moved_block": _ratio(progress_cpu, moved) * 1e6,
        "native_us_per_moved_block": _ratio(native_cpu, moved) * 1e6,
        "deposit_already_held_share": _ratio(max(0.0, deposited - copied), deposited),
        **_p50_p99(hists["step_kvcr"], "step_kvcr"),
        **_p50_p99(hists["ttft"], "ttft"),
        **_p50_p99(hists["cache_resolve_remote"], "remote_resolve"),
        **_p50_p99(hists["deliver_remote"], "deliver_remote"),
        "hinted_block_failure_rate": _ratio(hinted_failed, hinted_asked),
        "hinted_timeout_rate": _ratio(causes.get("timeout", 0), hinted_asked),
        "hinted_unavailable_rate": _ratio(causes.get("unavailable", 0), hinted_asked),
    }

    lag = LogHist.from_json(tally["lag"])
    cold = sum(
        h.n
        for name, h in kvcr_hists.items()
        if name.startswith("kvcr_duration_seconds|peer_setup|connected")
    )
    warm_full = config["phases"]["mesh_warm"] == "full" and workers > 1
    flags = []
    if counters.get("bytes_mismatch", 0):
        flags.append(
            f"{int(counters['bytes_mismatch'])} delivered blocks had wrong bytes"
        )
    if lag.quantile(0.99) > MAX_COORDINATOR_LAG_P99_S:
        flags.append(f"coordinator dispatch lag p99 {lag.quantile(0.99) * 1e3:.2f} ms")
    coordinator_busy = _ratio(tally["busy_s"], tally["measure_s"])
    if coordinator_busy > MAX_COORDINATOR_BUSY:
        flags.append(f"the coordinator was busy {coordinator_busy:.0%} of the window")
    for role in ("engine", "progress"):
        shares = {
            r["worker"]: _ratio(r["threads"][role]["delay_s"], r["window_s"])
            for r in reports
            if role in r["threads"]
        }
        over = [w for w, share in shares.items() if share > MAX_RUN_DELAY_SHARE]
        if over:
            worst = max(over, key=shares.__getitem__)
            flags.append(
                f"{role} thread waited for a CPU {shares[worst]:.1%} of the window "
                f"on worker {worst} ({len(over)} of {workers} workers over "
                f"{MAX_RUN_DELAY_SHARE:.0%})"
            )
    if warm_full and cold:
        flags.append(f"{cold} first-contact peer connects inside the window")
    if warm_full and coord["warm"]["failed"]:
        flags.append(f"{coord['warm']['failed']} mesh-warm pulls failed")
    if counters.get("resilience_events", 0):
        flags.append(f"{int(counters['resilience_events'])} KVCR resilience events")
    if not all(r["drained"] for r in reports):
        flags.append("some workers still had operations in flight at exit")
    occupancy = tally["primary_occupancy_min"]
    if occupancy < config["phases"]["min_primary_occupancy"]:
        flags.append(f"a host tier had only filled to {occupancy:.0%} at window start")
    if any(r["profiled"] for r in reports):
        flags.append("profiled (CB_PROFILE_DIR)")
    if not requests:
        flags.append("no requests completed in the window")

    return {
        "name": coord["name"],
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
            **tally["workload"],
            "sessions_started": tally["sessions_started"],
            "inflight_max": tally["inflight_max"],
        },
        "offered": tally["offered"],
        "engine_calls": engine_calls,
        "counters": counters,
        "latency": {name: hist.summary() for name, hist in sorted(hists.items())},
        "threads": thread_summary,
        "kvcr": {name: hist.summary() for name, hist in sorted(kvcr_hists.items())},
        "kvcr_counters": kvcr_counters,
        "warm": coord["warm"],
        "coordinator_lag": lag.summary(),
        "coordinator_busy": coordinator_busy,
        "rss_mb_mean": statistics.fmean(r["rss_mb"] for r in reports),
        "engine_gauges": {
            key: statistics.fmean(r["engine"][key] for r in reports)
            for key in reports[0]["engine"]
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
    count = summary["counters"].get
    offered = summary["offered"]
    failures = summary["failures"]
    workload = summary["workload"]
    thread_lines = [
        f"engine thread: KVCR {h['engine_kvcr_share']:.1%} of wall "
        f"({_fmt(h['engine_kvcr_us_per_request'] / 1e3)} ms/request), vLLM "
        f"{h['engine_vllm_share']:.1%} "
        f"({_fmt(h['engine_vllm_us_per_request'] / 1e3)} ms/request), harness "
        f"{h['engine_harness_share']:.1%} "
        f"({_fmt(h['engine_harness_us_per_request'] / 1e3)} ms/request), "
        f"modelled GPU {h['engine_forward_share']:.1%}, "
        f"idle {h['engine_idle_share']:.1%}",
        f"KVCR per engine step p50/p99 "
        f"{_fmt(h['step_kvcr_p50_ms'])}/{_fmt(h['step_kvcr_p99_ms'])} ms",
    ]
    trace_lines = []
    if workload["kind"] == "trace":
        trace_lines = [
            f"trace {workload['trace']} ({workload['sha256'][:12]}, "
            f"{workload['traces']} traces): {workload['plays_started']} plays "
            f"started, {workload['plays_finished']} finished, "
            f"{workload['turns']} turns sent ({workload['child_turns']} by "
            f"sub-agents), at most {workload['inflight_max']} requests in flight"
        ]
    causes = dict(failures["causes"])
    cause = (
        f"hinted blocks a peer did not deliver, by KVCR status: timeout "
        f"{causes.pop('timeout', 0):.0f} ({h['hinted_timeout_rate']:.2%} of "
        f"hinted blocks), unavailable {causes.pop('unavailable', 0):.0f} "
        f"({h['hinted_unavailable_rate']:.2%})"
    )
    cause += "".join(f", {name} {n:.0f}" for name, n in sorted(causes.items()))
    cause += f"; beyond the hint {failures['beyond_hint']:.0f}"
    if failures["pin_requests"]:
        cause += (
            f"; source pins {failures['pin_requests']:.0f} requested, "
            f"{failures['pin_misses']:.0f} missed, "
            f"{failures['pin_wait_timeouts']:.0f} timed out"
        )
    lines = [
        f"{summary['name']}  arm={summary['arm'] or '-'}  "
        f"kvcr={summary['kvcr_sha']}  {summary['workers']} workers on "
        f"{summary['nodes']} nodes  window {summary['window_s']:.1f}s",
        "VALID" if summary["valid"] else "INVALID: " + "; ".join(summary["flags"]),
        *trace_lines,
        "",
        f"requests/s {_fmt(h['requests_per_s'])}   "
        f"remote blocks/s {_fmt(h['remote_blocks_per_s'])}   "
        f"hinted {offered['hinted']} of {offered['dispatched']} requests "
        f"({offered['hinted_cross_node']} cross-node)",
        f"blocks: gpu {count('gpu_hit_blocks', 0):.0f}  "
        f"primary {count('primary_hit_blocks', 0):.0f}  "
        f"kvcr-local {count('deliver_blocks_local', 0):.0f}  "
        f"remote {count('deliver_blocks_remote', 0):.0f}"
        f" ({count('deliver_blocks_remote_cross_node', 0):.0f} cross-node)  "
        f"failed {count('deliver_blocks_failed', 0):.0f} "
        f"(hinted-block failure rate {h['hinted_block_failure_rate']:.2%})",
        f"ttft p50/p99 {_fmt(h['ttft_p50_ms'])}/{_fmt(h['ttft_p99_ms'])} ms   "
        f"remote resolve p50/p99 {_fmt(h['remote_resolve_p50_ms'])}/"
        f"{_fmt(h['remote_resolve_p99_ms'])} ms   "
        f"remote deliver p50/p99 {_fmt(h['deliver_remote_p50_ms'])}/"
        f"{_fmt(h['deliver_remote_p99_ms'])} ms",
        cause,
        "",
        *thread_lines,
        f"per moved block: KVCR {_fmt(h['engine_kvcr_us_per_moved_block'])} us, "
        f"progress {_fmt(h['progress_us_per_moved_block'])} us, "
        f"native {_fmt(h['native_us_per_moved_block'])} us; "
        f"{h['deposit_already_held_share']:.0%} of deposited blocks were already "
        f"in KVCR's pool",
        "",
        f"{'KVCR call':<16}{'calls':>10}{'items':>12}{'us/call':>10}"
        f"{'us/item':>10}{'share':>9}",
    ]
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
        f"busy {summary['coordinator_busy']:.0%};  "
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
    ("hinted_unavailable_rate", "hinted blocks unavailable"),
    ("requests_per_s", "requests / s"),
    ("remote_blocks_per_s", "remote blocks / s"),
)


def _relative(delta: float, base: float, spec: str) -> str:
    """``delta`` as a share of ``base``, which has no value against a zero
    base unless ``delta`` is zero too."""
    if base:
        return format(delta / base, spec)
    return "n/a" if delta else format(0.0, spec)


def compare(arms: dict[str, list[dict]]) -> str:
    """A/B (or A/A) table: median per arm, spread, and change versus the first.

    Only valid runs are compared; an arm without one raises ``ValueError``.
    """
    names = list(arms)
    base = names[0]
    lines = []
    valid: dict[str, list[dict]] = {}
    for name in names:
        runs = arms[name]
        valid[name] = [s for s in runs if s["valid"]]
        invalid = [s for s in runs if not s["valid"]]
        shas = sorted({s["kvcr_sha"] for s in runs})
        lines.append(
            f"{name}: {len(runs)} runs, kvcr {','.join(shas)}"
            + (f", {len(invalid)} INVALID and left out" if invalid else "")
        )
        for s in invalid:
            lines.append(f"    {s['name']}: " + "; ".join(s["flags"]))
        if not valid[name]:
            lines.append(f"{name} has no valid run: nothing to compare")
            raise ValueError("\n".join(lines))
    arms = valid
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
            spread = _relative(max(v) - min(v), medians[n], ".0%")
            out += f"{_fmt(medians[n]) + f' ({spread})':>24}"
        for n in names[1:]:
            change = _relative(medians[n] - medians[base], medians[base], "+.1%")
            out += f"{change:>18}"
        return out

    for key, label in AB_METRICS:
        lines.append(
            row(label, {n: [s["headline"][key] for s in arms[n]] for n in names})
        )
    lines.append("engine thread, by KVCR call:")
    calls = {call for runs in arms.values() for s in runs for call in s["engine_calls"]}
    for call in sorted(calls):
        # poll's cost is per call (most polls complete nothing); the others
        # scale with the keys they are given.
        unit = "call" if call == "poll" else "item"
        values = {
            n: [
                s["engine_calls"].get(call, {}).get(f"us_per_{unit}", 0.0)
                for s in arms[n]
            ]
            for n in names
        }
        lines.append(row(f"  {call} us / {unit}", values))
    return "\n".join(lines)
