# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Whole cells on one machine with real KVCR, NIXL/UCX and ZMQ."""

import itertools
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
from clusterbench.config import load_config
from clusterbench.launch_local import run_local
from clusterbench.trace import compile_corpus

_PROBE = (
    "import nixl._api as api;"
    "api.nixl_agent('cb-probe', api.nixl_agent_config(backends=['UCX']))"
)

REPO = str(Path(__file__).parents[2])

OVERRIDES = [
    "tiers.block_bytes=4096",
    "tiers.gpu_blocks=512",
    # Roomy tiers: inventory never goes stale, so hinted blocks must arrive.
    "tiers.primary_blocks=4096",
    "engine.step_ms=2",
    "engine.prefill_us_per_token=1",
    "engine.verify=full",
    "workload.sessions_per_worker=3",
    "workload.system_prefix_tokens=2048",
    "workload.initial_context_median=1500",
    "workload.turn_tokens_median=300",
    "workload.output_tokens_median=40",
    "workload.output_tokens_sigma=0.4",
    "workload.think_median_s=0.05",
    "workload.think_sigma=0.4",
    "workload.think_cap_s=0.3",
    "workload.max_context_tokens=20000",
    "workload.reset_context_median=1500",
    "workload.ramp_s=0.3",
    "router.migration_prob=0.4",
    "phases.fill_s=2",
    "phases.measure_s=4",
    "phases.drain_s=5",
]


@pytest.fixture(scope="module")
def real_nixl():
    """Skip unless this machine can actually run a UCX NIXL agent."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", UCX_TLS="self,sm,tcp,cuda_copy")
    try:
        probe = subprocess.run(
            [sys.executable, "-c", _PROBE], env=env, capture_output=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("cannot start a NIXL agent here")
    if probe.returncode:
        pytest.skip("cannot start a NIXL agent here")


@pytest.mark.parametrize(
    "mode,tier_overrides",
    [
        ("g2-pool", ["tiers.g2_blocks=8192"]),
        ("peer-only", ["tiers.g2_blocks=0"]),
    ],
)
def test_workers_pull_blocks_from_each_other(real_nixl, tmp_path, mode, tier_overrides):
    config = load_config(None, [f"name={mode}", *OVERRIDES, *tier_overrides])
    # Named as an arm's checkout, so the workers must have loaded this tree.
    summary = run_local(config, tmp_path, nodes=3, per_node=1, kvcr_src=REPO)
    logs = "".join(p.read_text()[-2000:] for p in (tmp_path / "logs").glob("*.log"))
    assert summary["workers"] == 3 and summary["nodes"] == 3, logs
    counters = summary["counters"]
    # Every ordered pair connected during mesh warm, none inside the window.
    assert summary["warm"] == {
        "ok": 6,
        "failed": 0,
        "max_seconds": summary["warm"]["max_seconds"],
    }
    # Blocks moved between workers, and every delivered block had its bytes.
    assert counters["deliver_blocks_remote"] > 100, summary["flags"]
    assert counters["bytes_checked"] >= counters["deliver_blocks_remote"]
    assert counters.get("bytes_mismatch", 0) == 0
    # One worker per node: every peer is on another node.
    cross_node = counters["deliver_blocks_remote_cross_node"]
    assert cross_node == counters["deliver_blocks_remote"]
    assert summary["headline"]["hinted_block_failure_rate"] < 0.05
    assert summary["engine_calls"]["deliver"]["calls"] > 0
    assert summary["engine_calls"]["align_sequence"]["items"] > 0
    if mode == "peer-only":
        # Sources serve from pinned framework memory; deposits have no pool.
        assert counters["pin_keys"] > 0
        assert counters.get("deposit_blocks_ok", 0) == 0
    else:
        assert counters["deposit_blocks_ok"] > 0
    hard = [
        f
        for f in summary["flags"]
        if "waited for a CPU" not in f and "dispatch lag" not in f
    ]
    assert not hard, hard
    # Artifacts a later report or comparison needs.
    for name in ("config.json", "workers.json", "coordinator.json", "summary.json"):
        assert json.loads((tmp_path / name).read_text())
    assert len(list((tmp_path / "reports").glob("*.json"))) == 3
    # The request log has the blocks each request got from a peer.
    log = (tmp_path / "requests.jsonl").read_text().splitlines()
    header, *rows = map(json.loads, log)
    remote = header["fields"].index("remote_blocks")
    assert sum(row[remote] for row in rows if len(row) > remote) > 100


def _corpus(path, sessions=6, turns=40):
    """A small corpus in the AgentX trace format, with sub-agents."""
    rng = random.Random(7)
    with open(path, "w") as file:
        for n in range(sessions):
            fresh = itertools.count(1)

            def new(count):
                return [next(fresh) for _ in range(count)]

            def request(t, ids, out):
                return {
                    "t": t,
                    "type": "s",
                    "in": 64 * len(ids),
                    "out": 64 * out,
                    "hash_ids": list(ids),
                    "api_time": 0.05,
                }

            system = new(24)
            context, requests, t = list(system), [], 0.0
            for turn in range(turns):
                out = rng.randint(1, 2)
                requests.append(request(t, context, out))
                t += 0.1
                if turn % 6 == 2:  # a sub-agent with a context of its own
                    side, inner = new(8), []
                    for step in range(3):
                        inner.append(request(t + 0.1 * step, side, 1))
                        side = side + new(3)
                    group = {"type": "subagent", "t": t, "status": "completed"}
                    requests.append({**group, "requests": inner})
                    t += 0.4
                if turn % 15 == 14:  # compaction: back to the system prompt
                    context = system + new(6)
                else:
                    context = context + new(out + rng.randint(2, 6))
            trace = {"id": f"s{n}", "block_size": 64, "hash_id_scope": "local"}
            file.write(json.dumps({**trace, "requests": requests}) + "\n")


def test_a_replayed_trace_moves_blocks_between_workers(real_nixl, tmp_path):
    _corpus(tmp_path / "source.jsonl")
    stats = compile_corpus(tmp_path / "source.jsonl", tmp_path / "replay.cbtrace")
    assert stats.groups and stats.tail_blocks
    overrides = [
        "name=trace",
        *OVERRIDES,
        "tiers.g2_blocks=8192",
        f"workload.trace={tmp_path / 'replay.cbtrace'}",
    ]
    out = tmp_path / "run"
    summary = run_local(load_config(None, overrides), out, nodes=3, per_node=1)
    logs = "".join(p.read_text()[-2000:] for p in (out / "logs").glob("*.log"))
    assert summary["workers"] == 3, logs
    workload = summary["workload"]
    # Nine plays of six traces: sub-agents ran beside their parents.
    assert workload["kind"] == "trace" and workload["traces"] == 6
    assert workload["plays_started"] >= 9 and workload["child_turns"] > 0
    assert workload["inflight_max"] > 1
    counters = summary["counters"]
    # Sessions moved, and every block a peer delivered had its bytes.
    assert counters["deliver_blocks_remote"] > 50, summary["flags"]
    assert counters["bytes_checked"] >= counters["deliver_blocks_remote"]
    assert counters.get("bytes_mismatch", 0) == 0
    assert counters["requests"] > 50
    hard = [
        f
        for f in summary["flags"]
        if "waited for a CPU" not in f and "dispatch lag" not in f
    ]
    assert not hard, hard


def test_transfers_kvcr_times_out_are_reported_as_timeouts(real_nixl, tmp_path):
    overrides = [
        "name=timeouts",
        *OVERRIDES,
        "tiers.g2_blocks=8192",
        # No mesh warm: the first transfer to each peer has to connect first,
        # which takes far longer than two milliseconds.
        "phases.mesh_warm=none",
        "engine.operation_timeout_ms=2",
        "engine.abandon_timeout_ms=1000",
    ]
    summary = run_local(load_config(None, overrides), tmp_path, nodes=3, per_node=1)
    causes = summary["failures"]["causes"]
    assert causes.get("timeout", 0) > 0, summary["counters"]
    # This tree's KVCR names a status for every block it does not deliver.
    assert "unclassified" not in causes
    assert summary["counters"]["hinted_failed_reason|deadline"] > 0
    assert summary["headline"]["hinted_timeout_rate"] > 0
    # The request log has each request's share of them.
    log = (tmp_path / "requests.jsonl").read_text().splitlines()
    header, *rows = map(json.loads, log)
    timed_out = header["fields"].index("timed_out_blocks")
    assert sum(row[timed_out] for row in rows if len(row) > timed_out) > 0
