# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Whole cells on one machine with real KVCR, NIXL/UCX and ZMQ."""

import json
import os
import subprocess
import sys

import pytest
from clusterbench.config import load_config
from clusterbench.launch_local import run_local

_PROBE = (
    "import nixl._api as api;"
    "api.nixl_agent('cb-probe', api.nixl_agent_config(backends=['UCX']))"
)

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
        ("pinless", ["tiers.g2_blocks=8192", "tiers.framework_pinning=false"]),
    ],
)
def test_workers_pull_blocks_from_each_other(real_nixl, tmp_path, mode, tier_overrides):
    config = load_config(None, [f"name={mode}", *OVERRIDES, *tier_overrides])
    summary = run_local(config, tmp_path, nodes=3, per_node=1)
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
    assert summary["completed"]["remote_blocks_cross_node"] > 0
    assert summary["headline"]["hinted_block_failure_rate"] < 0.05
    assert summary["engine_calls"]["deliver"]["calls"] > 0
    assert summary["engine_calls"]["align_sequence"]["items"] > 0
    if mode == "peer-only":
        # Sources serve from pinned framework memory; deposits have no pool.
        assert counters["pin_keys"] > 0
        assert counters.get("deposit_blocks_ok", 0) == 0
    else:
        assert counters["deposit_blocks_ok"] > 0
    if mode == "pinless":
        assert counters.get("pin_requests", 0) == 0
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


def test_sglang_workers_restore_pages_from_each_other(real_nixl, tmp_path):
    overrides = [
        "name=sglang-linker",
        *OVERRIDES,
        "engine.profile=sglang",
        "tiers.layers=8",
        "tiers.gpu_blocks=2048",
        "tiers.g2_blocks=8192",
    ]
    summary = run_local(load_config(None, overrides), tmp_path, nodes=3, per_node=1)
    logs = "".join(p.read_text()[-2000:] for p in (tmp_path / "logs").glob("*.log"))
    assert summary["workers"] == 3 and summary["profile"] == "sglang", logs
    counters = summary["counters"]
    assert summary["warm"]["ok"] == 6 and summary["warm"]["failed"] == 0
    # Pages moved between workers layer by layer, and every piece had its bytes.
    assert counters["deliver_blocks_remote"] > 100, summary["flags"]
    assert counters["bytes_checked"] >= counters["deliver_blocks_remote"]
    assert counters.get("bytes_mismatch", 0) == 0
    assert summary["completed"]["remote_blocks_cross_node"] > 0
    assert summary["headline"]["abort_rate"] < 0.05
    calls = summary["engine_calls"]
    # One deliver per layer of every load; the linker never aligns a sequence.
    assert calls["deliver"]["calls"] >= 8 * counters["load_batches"]
    assert calls["deposit"]["items"] > 0 and counters["deposit_blocks_ok"] > 0
    assert calls["query"]["items"] > calls["query"]["calls"]
    assert "align_sequence" not in calls
    assert summary["threads"]["owner"]["threads_per_worker"] == 1
    hard = [
        f
        for f in summary["flags"]
        if "waited for a CPU" not in f and "dispatch lag" not in f
    ]
    assert not hard, hard
    assert len(list((tmp_path / "reports").glob("*.json"))) == 3
