# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Phase 1 performance cases TC-05 and TC-06 from the KVCR test plan.

TC-05  Cold-path overhead and reuse latency: TTFT / TPOT / ITL / tokens/s for
       KVCR disabled (recorded in the reference phase), KVCR cold, local hit and
       same-node remote hit, plus transfer bytes and pool occupancy.
TC-06  Guard recovery timing: pre-failure warm latency, engine kill, time to
       the first correct Guard-served response, replacement re-attach time and
       post-recovery latency on the recovered prefixes.

Every perf request still has to reproduce the reference token IDs; a mismatch is
a FAIL, not a data point. No thresholds are applied in this round.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .cases import PROMOTED, Context
from .client import StreamResponse
from .common import (
    BlockedError,
    CaseResult,
    compare_tokens,
    summarize,
    utc_now,
)
from .stack import GUARD, GUARD_PROMOTED
from .telemetry import (
    LOCAL_DELIVER,
    LOCAL_FILL,
    REMOTE_DELIVER,
    SOURCE_WRITE,
    TIER_READ_BYTES,
    TIER_WRITE_BYTES,
    latest_gauge,
)

STREAM_FIELDS = [
    "worker_id",
    "prompt_token_ids",
    "completion_token_ids",
    "detailed_finish_reason",
    "timing",
]


# -- helpers -----------------------------------------------------------------


def gpu_memory_used_mib() -> dict[str, int] | None:
    """Per-GPU used memory from nvidia-smi, or None when unavailable."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        out = subprocess.run(
            [binary, "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    used: dict[str, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[1].isdigit():
            used[parts[0]] = int(parts[1])
    return used


def pool_gauges(ctx: Context, worker_index: int) -> dict[str, Any]:
    log = ctx.stack.workers[worker_index].log_path
    return {
        "local_g2_allocated_slots": latest_gauge(log, "local_g2_allocated_slots"),
        "pinned_keys": latest_gauge(log, "pinned_keys"),
    }


def arm_summary(responses: list[StreamResponse]) -> dict[str, Any]:
    ok = [r for r in responses if r.ok]
    itl: list[float] = []
    for r in ok:
        itl.extend(r.inter_chunk_seconds)
    return {
        "requests": len(responses),
        "ok": len(ok),
        "ttft_seconds": summarize([r.first_token_seconds for r in ok]),
        "tpot_seconds": summarize([r.tpot_seconds for r in ok if r.tpot_seconds]),
        "itl_seconds": summarize(itl),
        "tokens_per_second": summarize([r.tokens_per_second for r in ok]),
        "total_seconds": summarize([r.total_seconds for r in ok]),
        "tokens": summarize([float(r.tokens) for r in ok]),
        "cached_tokens": [r.cached_tokens for r in responses],
    }


def baseline_summary(baselines: dict[str, Any], kind: str) -> dict[str, Any]:
    picks = [entry.get(kind) or {} for entry in baselines.values()]
    return {
        "requests": len(picks),
        "ttft_seconds": summarize([m.get("ttft_seconds") for m in picks]),
        "tpot_seconds": summarize([m.get("tpot_seconds") for m in picks]),
        "tokens_per_second": summarize([m.get("tokens_per_second") for m in picks]),
        "total_seconds": summarize([m.get("total_seconds") for m in picks]),
    }


def delta(a: dict[str, Any], b: dict[str, Any], key: str) -> float | None:
    """median(a) - median(b) for one metric summary, None if either is missing."""
    left = (a.get(key) or {}).get("median")
    right = (b.get(key) or {}).get("median")
    if left is None or right is None:
        return None
    return round(left - right, 6)


def run_arm(
    ctx: Context,
    label: str,
    prompts: list[tuple[str, str]],
    worker_index: int,
    *,
    concurrency: int | None = None,
    max_tokens: int | None = None,
) -> list[StreamResponse]:
    """Stream every (name, prompt) to one worker at the configured concurrency."""
    pin = ctx.stack.pin(worker_index)
    parallel = ctx.perf_concurrency if concurrency is None else concurrency

    def one(item: tuple[str, str]) -> StreamResponse:
        name, prompt = item
        ctx.stack.check_children()
        return ctx.client.stream_completion(
            f"{label}-{name}-w{worker_index}",
            prompt,
            max_tokens=ctx.perf_max_tokens if max_tokens is None else max_tokens,
            extra_fields=STREAM_FIELDS,
            **pin,
        )

    if parallel <= 1:
        return [one(item) for item in prompts]
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        return list(pool.map(one, prompts))


def validate_stream(
    ctx: Context,
    result: CaseResult,
    response: StreamResponse,
    reference_key: str,
    worker_index: int,
    *,
    expect_hit: bool | None,
) -> bool:
    """HTTP ok, right worker, tokens equal reference, hit/cold as expected."""
    stage = response.stage
    result.evidence += [response.request_file, response.response_file]
    if not result.check(
        f"{stage}: streamed with token IDs",
        response.ok,
        response.error or f"status {response.http_status}, tokens={response.tokens}",
    ):
        return False
    expected_worker = ctx.stack.worker_ids[worker_index]
    result.check(
        f"{stage}: served by worker {worker_index}",
        ctx.stack.served(response) in (None, expected_worker),
        f"{ctx.stack.served_role}_worker_id={ctx.stack.served(response)}",
    )
    reference = ctx.references[reference_key]
    observed = response.sample()
    problems = compare_tokens(reference, observed)
    if not response.prompt_token_ids:
        # Streams carry completion IDs per chunk; prompt IDs may be absent.
        problems = [p for p in problems if "prompt token IDs" not in p]
    result.check(
        f"{stage}: streamed tokens equal reference ({reference_key})",
        not problems,
        "; ".join(problems)
        or f"{observed.length} tokens, finish={observed.finish_reason}",
    )
    if expect_hit is None:
        return not problems
    cached = response.cached_tokens
    if cached is None:
        result.check(
            f"{stage}: stream reports cached_tokens",
            False,
            "no usage.prompt_tokens_details.cached_tokens in the stream",
        )
        return False
    if expect_hit:
        result.check(
            f"{stage}: cache hit is positive and block-aligned",
            cached > 0 and cached % ctx.hit_block_tokens == 0,
            f"cached_tokens={cached} (block {ctx.hit_block_tokens})",
        )
    else:
        result.check(
            f"{stage}: cold request has 0 cached tokens",
            cached == 0,
            f"cached_tokens={cached}",
        )
    return not problems


# -- TC-05 ---------------------------------------------------------------------


def tc05(ctx: Context) -> CaseResult:
    result = CaseResult("TC-05", "Cold-path overhead and reuse latency", "perf")
    names = ctx.prompts.perf_names()
    prompts = [(name, ctx.prompts.perf(i)) for i, name in enumerate(names)]
    if not ctx.baseline_timings:
        raise BlockedError(
            "TC-05 needs KVCR-disabled baseline timings from the reference phase"
        )
    result.measurements["settings"] = {
        "samples": len(prompts),
        "max_tokens": ctx.perf_max_tokens,
        "concurrency": ctx.perf_concurrency,
    }
    result.measurements["gpu_memory_used_mib"] = {"before": gpu_memory_used_mib()}
    result.measurements["pool_before"] = {
        "w0": pool_gauges(ctx, 0),
        "w1": pool_gauges(ctx, 1),
    }

    # Arm 1: KVCR disabled (recorded earlier, first request per prompt = cold).
    baseline_cold = baseline_summary(ctx.baseline_timings, "cold")
    baseline_warm = baseline_summary(ctx.baseline_timings, "warm")
    result.measurements["arm_kvcr_disabled_cold"] = baseline_cold
    result.measurements["arm_kvcr_disabled_prefix_cache_hit"] = baseline_warm
    result.check(
        "tc05 baseline: KVCR-disabled timings exist for every perf prompt",
        all(name in ctx.baseline_timings for name in names),
        f"{len(ctx.baseline_timings)} of {len(names)} prompts",
    )

    # Arm 2: KVCR enabled, cold (store + commit on worker 0).
    cold_start = utc_now()
    cold = run_arm(ctx, "tc05-cold", prompts, 0)
    for response, name in zip(cold, names):
        validate_stream(ctx, result, response, name, 0, expect_hit=False)
    cold_window = ctx.wait_window(0, cold_start, lambda w: w.blocks(LOCAL_FILL) > 0)
    result.check(
        "tc05 cold: KVCR committed blocks for the cold requests",
        cold_window.blocks(LOCAL_FILL) > 0 and cold_window.successes(LOCAL_FILL) >= 1,
        f"local_fill blocks={cold_window.blocks(LOCAL_FILL)}, "
        f"tier write bytes={cold_window.get(TIER_WRITE_BYTES)}",
    )
    result.measurements["arm_kvcr_cold"] = arm_summary(cold)
    result.measurements["arm_kvcr_cold_window"] = cold_window.totals
    time.sleep(ctx.index_wait)

    # Arm 3: local hit on worker 0 (same prompts, same worker).
    local_start = utc_now()
    local = run_arm(ctx, "tc05-local", prompts, 0)
    for response, name in zip(local, names):
        validate_stream(ctx, result, response, name, 0, expect_hit=True)
    result.measurements["arm_kvcr_local_hit"] = arm_summary(local)
    result.measurements["arm_kvcr_local_hit_window"] = ctx.settle(0, local_start).totals

    # Arm 4: remote hit on worker 1 (KVCR peer transfer from worker 0).
    remote_start = utc_now()
    remote = run_arm(ctx, "tc05-remote", prompts, 1)
    hit_blocks = 0
    for response, name in zip(remote, names):
        if validate_stream(ctx, result, response, name, 1, expect_hit=True):
            hit_blocks += (response.cached_tokens or 0) // ctx.hit_block_tokens
    # Source first: with hybrid models one hit block maps to several KVCR
    # pages, so the target must be given time to report as many pages as the
    # source wrote, not only as many as the hit blocks.
    source = ctx.wait_window(
        0,
        remote_start,
        lambda w: w.blocks(SOURCE_WRITE) >= hit_blocks and hit_blocks > 0,
    )
    written = max(hit_blocks, source.blocks(SOURCE_WRITE))
    target = ctx.wait_window(
        1,
        remote_start,
        lambda w: w.blocks(REMOTE_DELIVER) >= written and hit_blocks > 0,
    )
    result.check(
        "tc05 remote: KVCR transfer delivered the cached blocks (remote_deliver "
        "covers hits)",
        ctx.delivery_covers(target.blocks(REMOTE_DELIVER), hit_blocks)
        and source.blocks(SOURCE_WRITE) == target.blocks(REMOTE_DELIVER),
        f"remote_deliver={target.blocks(REMOTE_DELIVER)}, "
        f"source_write={source.blocks(SOURCE_WRITE)}, "
        f"cached blocks={hit_blocks}, block tokens={ctx.hit_block_tokens}",
    )
    result.check(
        "tc05 remote: no positive KVCR failed/partial metrics",
        not (source.peer_failures + target.peer_failures),
        str(source.peer_failures + target.peer_failures),
    )
    result.measurements["arm_kvcr_remote_hit"] = arm_summary(remote)
    result.measurements["arm_kvcr_remote_hit_transfer"] = {
        "source_write_bytes": source.bytes(SOURCE_WRITE),
        "target_tier_read_bytes": target.get(TIER_READ_BYTES),
        "remote_deliver_blocks": target.blocks(REMOTE_DELIVER),
        "bytes_per_request": (
            round(source.bytes(SOURCE_WRITE) / len(remote), 1) if remote else None
        ),
        "source_window": source.totals,
        "target_window": target.totals,
    }

    # Arm 5 (optional): local hit from the KVCR DRAM pool. Fill worker 0's GPU
    # prefix cache with never-seen prompts until the perf prompts are evicted,
    # then request them again on worker 0: vLLM misses, the KVCR connector must
    # deliver the blocks from DRAM (local_deliver > 0).
    if ctx.perf_evict_filler > 0:
        fillers = [
            (f"filler_{i}", ctx.prompts.filler(i)) for i in range(ctx.perf_evict_filler)
        ]
        evict_start = utc_now()
        filler_responses = run_arm(
            ctx,
            "tc05-evict",
            fillers,
            0,
            concurrency=max(4, ctx.perf_concurrency),
            max_tokens=1,
        )
        result.check(
            "tc05 dram: every eviction filler request succeeded",
            all(r.ok for r in filler_responses),
            f"{sum(r.ok for r in filler_responses)} of {len(filler_responses)} ok, "
            f"{sum(r.cached_tokens or 0 for r in filler_responses)} cached tokens",
        )
        filler_window = ctx.wait_window(
            0, evict_start, lambda w: w.blocks(LOCAL_FILL) > 0
        )
        time.sleep(ctx.index_wait)
        dram_start = utc_now()
        dram = run_arm(ctx, "tc05-dram", prompts, 0)
        dram_hits = 0
        for response, name in zip(dram, names):
            if validate_stream(ctx, result, response, name, 0, expect_hit=True):
                dram_hits += (response.cached_tokens or 0) // ctx.hit_block_tokens
        dram_window = ctx.wait_window(
            0,
            dram_start,
            lambda w: w.blocks(LOCAL_DELIVER) >= dram_hits and dram_hits > 0,
        )
        result.check(
            "tc05 dram: GPU cache evicted, hits delivered from the KVCR pool "
            "(local_deliver > 0)",
            dram_window.blocks(LOCAL_DELIVER) > 0 and dram_hits > 0,
            f"local_deliver blocks={dram_window.blocks(LOCAL_DELIVER)}, "
            f"cached blocks={dram_hits}, tier read "
            f"bytes={dram_window.get(TIER_READ_BYTES)}",
        )
        result.measurements["arm_kvcr_dram_hit"] = arm_summary(dram)
        result.measurements["arm_kvcr_dram_hit_window"] = dram_window.totals
        result.measurements["eviction"] = {
            "filler_prompts": len(fillers),
            "filler_prompt_tokens": sum(
                (r.usage or {}).get("prompt_tokens") or 0 for r in filler_responses
            ),
            "filler_cached_tokens": [r.cached_tokens for r in filler_responses],
            "filler_window": filler_window.totals,
            "pool_after": pool_gauges(ctx, 0),
        }

    # Arm 6 (optional): concurrency sweep. Each level sends `level` fresh prompts
    # at once: cold on worker 0 (store), then the same prompts on worker 1
    # (peer transfer). No reference tokens exist for these prompts, so only the
    # hit and the timings are recorded.
    sweep: dict[str, Any] = {}
    for level in ctx.perf_concurrency_levels:
        items = [(f"c{level}_{i}", ctx.prompts.sweep(level, i)) for i in range(level)]
        s_cold_start = utc_now()
        s_cold = run_arm(ctx, f"tc05-sweep{level}-cold", items, 0, concurrency=level)
        ctx.wait_window(0, s_cold_start, lambda w: w.blocks(LOCAL_FILL) > 0)
        time.sleep(ctx.index_wait)
        s_remote_start = utc_now()
        s_remote = run_arm(
            ctx, f"tc05-sweep{level}-remote", items, 1, concurrency=level
        )
        s_hits = sum(
            (r.cached_tokens or 0) // ctx.hit_block_tokens for r in s_remote if r.ok
        )
        s_target = ctx.wait_window(
            1,
            s_remote_start,
            lambda w: w.blocks(REMOTE_DELIVER) >= s_hits and s_hits > 0,
        )
        s_source = ctx.wait_window(
            0, s_remote_start, lambda w: w.blocks(SOURCE_WRITE) >= s_hits and s_hits > 0
        )
        result.check(
            f"tc05 sweep c={level}: all requests ok and every remote request hit",
            all(r.ok for r in s_cold + s_remote)
            and all((r.cached_tokens or 0) > 0 for r in s_remote),
            f"cold ok={sum(r.ok for r in s_cold)}/{level}, remote ok="
            f"{sum(r.ok for r in s_remote)}/{level}, remote cached="
            f"{[r.cached_tokens for r in s_remote]}",
        )
        sweep[str(level)] = {
            "cold_w0": arm_summary(s_cold),
            "remote_hit_w1": arm_summary(s_remote),
            "remote_deliver_blocks": s_target.blocks(REMOTE_DELIVER),
            "source_write_bytes": s_source.bytes(SOURCE_WRITE),
            "cached_blocks": s_hits,
            "peer_failures": s_source.peer_failures + s_target.peer_failures,
        }
    if sweep:
        result.measurements["concurrency_sweep"] = sweep

    # Derived: cold-path overhead and reuse gains (medians, seconds).
    cold_s = result.measurements["arm_kvcr_cold"]
    local_s = result.measurements["arm_kvcr_local_hit"]
    remote_s = result.measurements["arm_kvcr_remote_hit"]
    result.measurements["derived"] = {
        "cold_path_overhead_ttft_seconds": delta(cold_s, baseline_cold, "ttft_seconds"),
        "cold_path_overhead_tpot_seconds": delta(cold_s, baseline_cold, "tpot_seconds"),
        "local_hit_ttft_gain_vs_cold_seconds": delta(cold_s, local_s, "ttft_seconds"),
        "remote_hit_ttft_gain_vs_cold_seconds": delta(cold_s, remote_s, "ttft_seconds"),
        "local_hit_vs_native_prefix_cache_ttft_seconds": delta(
            local_s, baseline_warm, "ttft_seconds"
        ),
        "note": "positive overhead = KVCR slower than the KVCR-disabled engine; "
        "no thresholds are applied in this round",
    }
    if "arm_kvcr_dram_hit" in result.measurements:
        result.measurements["derived"]["dram_hit_ttft_gain_vs_cold_seconds"] = delta(
            cold_s, result.measurements["arm_kvcr_dram_hit"], "ttft_seconds"
        )
    result.measurements["pool_after"] = {
        "w0": pool_gauges(ctx, 0),
        "w1": pool_gauges(ctx, 1),
    }
    result.measurements["gpu_memory_used_mib"]["after"] = gpu_memory_used_mib()
    ctx.save(result, "perf", result.measurements)
    result.finish()
    return result


# -- TC-06 ---------------------------------------------------------------------


def tc06(ctx: Context) -> CaseResult:
    result = CaseResult("TC-06", "Guard recovery timing", "perf")
    if ctx.stack.settings.mode != GUARD:
        raise BlockedError("TC-06 runs only in Guard mode")
    service = ctx.stack.services[0]
    names = ctx.prompts.tc06_names()
    prompts = [(name, ctx.prompts.tc06(i)) for i, name in enumerate(names)]
    result.measurements["settings"] = {
        "samples": len(prompts),
        "max_tokens": ctx.perf_max_tokens,
    }

    # Seed worker 0 so the prefixes are committed, then measure warm latency.
    seed_start = utc_now()
    seeded = run_arm(ctx, "tc06-seed", prompts, 0)
    for response, name in zip(seeded, names):
        validate_stream(ctx, result, response, name, 0, expect_hit=False)
    committed = ctx.wait_window(0, seed_start, lambda w: w.blocks(LOCAL_FILL) > 0)
    committed_blocks = committed.blocks(LOCAL_FILL)
    result.check(
        "tc06 seed: KVCR committed the prefixes before failure",
        committed_blocks > 0,
        f"local_fill blocks={committed_blocks}",
    )
    time.sleep(ctx.index_wait)
    pre = run_arm(ctx, "tc06-pre", prompts, 0)
    for response, name in zip(pre, names):
        validate_stream(ctx, result, response, name, 0, expect_hit=True)
    result.measurements["arm_pre_failure_local_hit"] = arm_summary(pre)

    # t0: kill the engine (its KVCR service survives and promotes the Guard).
    promotions_before = service.log_text().count(GUARD_PROMOTED)
    kill = ctx.stack.kill_worker(0)
    t0 = time.monotonic()
    ctx.save(result, "kill", kill)
    text = ctx.stack.wait_log(
        service, GUARD_PROMOTED, ctx.recovery_timeout, promotions_before + 1
    )
    promoted_seconds = time.monotonic() - t0
    match = PROMOTED.findall(text)[-1] if PROMOTED.findall(text) else None
    recovered = int(match[2]) if match else -1
    result.check(
        "tc06: Guard promoted after engine death",
        match is not None,
        f"after {promoted_seconds:.2f}s: {match}",
    )

    # t1: first correct Guard-served response on worker 1 (hinted to the dead
    # worker's endpoint). Poll until the reference tokens come back.
    attempts: list[dict[str, Any]] = []
    first_correct: StreamResponse | None = None
    t1: float | None = None
    deadline = time.monotonic() + ctx.recovery_timeout
    name0, prompt0 = prompts[0]
    while time.monotonic() < deadline and first_correct is None:
        response = ctx.client.stream_completion(
            f"tc06-guard-attempt{len(attempts) + 1}-w1",
            prompt0,
            **ctx.stack.pin(1),
            max_tokens=ctx.perf_max_tokens,
            extra_fields=STREAM_FIELDS,
        )
        sent_at = time.monotonic() - (response.total_seconds or 0)
        problems = (
            compare_tokens(ctx.references[name0], response.sample())
            if response.ok
            else ["no tokens"]
        )
        problems = [p for p in problems if "prompt token IDs" not in p]
        attempts.append(
            {
                "stage": response.stage,
                "ok": response.ok,
                "error": response.error,
                "cached_tokens": response.cached_tokens,
                "ttft_seconds": response.first_token_seconds,
                "problems": problems,
                "seconds_after_kill": round(time.monotonic() - t0, 3),
            }
        )
        result.evidence += [response.request_file, response.response_file]
        if response.ok and not problems and (response.cached_tokens or 0) > 0:
            first_correct = response
            t1 = sent_at + (response.first_token_seconds or 0) - t0
        else:
            time.sleep(1)
    result.measurements["guard_serve_attempts"] = attempts
    result.check(
        "tc06: Guard served the first correct hit to worker 1",
        first_correct is not None,
        f"{len(attempts)} attempts; t1={t1}",
    )
    guard_served = [first_correct] if first_correct else []
    if first_correct is not None:
        # Remaining prefixes from the Guard as well, for a latency sample.
        guard_served += run_arm(ctx, "tc06-guard", prompts[1:], 1)
        for response, name in zip(guard_served[1:], names[1:]):
            validate_stream(ctx, result, response, name, 1, expect_hit=True)
    result.measurements["arm_guard_served_remote_hit"] = arm_summary(guard_served)

    # t2: replacement re-attaches and serves.
    restart_at = time.monotonic()
    ctx.stack.start_worker(0, tag="-replacement-tc06")
    ctx.stack.wait_model(ctx.client)
    ctx.stack.discover_worker(ctx.client, 0)
    t2 = time.monotonic() - t0
    result.check(
        "tc06: replacement attached, single owner (no extra promotion)",
        service.alive()
        and service.log_text().count(GUARD_PROMOTED) == promotions_before + 1,
        f"guard_promoted events={service.log_text().count(GUARD_PROMOTED)}",
    )
    post = run_arm(ctx, "tc06-post", prompts, 0)
    for response, name in zip(post, names):
        # Recovered prefixes should hit from the reclaimed pool; record either way.
        validate_stream(ctx, result, response, name, 0, expect_hit=None)
    result.measurements["arm_post_recovery_replacement"] = arm_summary(post)
    result.measurements["post_recovery_cached_tokens"] = [r.cached_tokens for r in post]

    pre_s = result.measurements["arm_pre_failure_local_hit"]
    post_s = result.measurements["arm_post_recovery_replacement"]
    result.measurements["timeline_seconds"] = {
        "t0_kill": 0.0,
        "guard_promoted": round(promoted_seconds, 3),
        "t1_first_correct_guard_response": round(t1, 3) if t1 is not None else None,
        "t2_replacement_serving": round(t2, 3),
        "replacement_startup": round(t2 - (restart_at - t0), 3),
        "recovered_blocks": recovered,
        "committed_blocks_before_failure": committed_blocks,
    }
    result.measurements["derived"] = {
        "post_vs_pre_ttft_seconds": delta(post_s, pre_s, "ttft_seconds"),
        "post_vs_pre_tpot_seconds": delta(post_s, pre_s, "tpot_seconds"),
        "note": "positive = slower after recovery; no thresholds in this round",
    }
    ctx.save(result, "perf", result.measurements)
    result.finish()
    return result


PERF_CASES = {"TC-05": tc05, "TC-06": tc06}
