# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from pathlib import Path

import pytest
from clusterbench import protocol
from clusterbench.config import RunConfig, decode_config, encode_config, load_config


@pytest.mark.parametrize(
    "message",
    [
        protocol.Hello(
            "n0",
            1,
            "host",
            42,
            "tcp://h:1",
            "agent",
            "base",
            "/k",
            "abc",
            [0, 1],
            {"nofile": 1024},
        ),
        protocol.Ready("warm", {"ok": 3.0}),
        protocol.Done(7, 1.5, 0.5, 100, 60, 10, 5, 20, 5, 320, timed_out_blocks=2),
        protocol.Inventory([(1, 5, 0, 10, False), (0, 5, 8, 10, True)]),
        protocol.Stats(1.0, {"running": 2.0}),
        protocol.Report(b"{}"),
        protocol.Failure("boom"),
        protocol.Configure(3, b"{}", [("n0", "tcp://h:1")]),
        protocol.Phase("measure"),
        protocol.Request(9, 4, [(1, 0, 10), (70, 0, 3)], (70, 3, 4), 900, 80, 2, 12),
    ],
)
def test_messages_round_trip(message):
    decoded = protocol.decode(protocol.encode(message))
    assert type(decoded) is type(message)
    assert protocol.encode(decoded) == protocol.encode(message)


def test_config_round_trips_and_overrides(tmp_path):
    path = tmp_path / "s.toml"
    path.write_text('name = "x"\n[tiers]\nprimary_blocks = 512\n')
    config = load_config(
        path,
        [
            "phases.measure_s=3",
            "router.migration_prob=0.5",
            "tiers.framework_pinning=false",
        ],
    )
    assert config.name == "x" and config.tiers.primary_blocks == 512
    assert config.phases.measure_s == 3 and config.router.migration_prob == 0.5
    assert config.tiers.framework_pinning is False
    assert decode_config(encode_config(config)) == config
    assert load_config(None) == RunConfig()


SCENARIOS = sorted(
    (Path(__file__).parents[2] / "benchmarks/clusterbench/scenarios").glob("*.toml")
)


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda path: path.stem)
def test_shipped_scenarios_load(path):
    assert load_config(path).name == path.stem


def test_scenarios_are_found():
    assert len(SCENARIOS) >= 5


@pytest.mark.parametrize(
    "overrides",
    [
        ["tiers.block_bytes=1000"],
        ["router.mode=cost"],  # an option that does not exist
        ["tiers.gpu_block=64"],
        ["engine.abandon_timeout_ms=1500"],
        ["phases.mesh_warm=some"],
        ["engine.verify=maybe"],
        ["engine.profile=trtllm"],
        ["engine.profile=sglang", "tiers.layers=7"],
        ["engine.profile=sglang", "tiers.g2_blocks=0"],
        ["tiers.g2_blocks=0", "tiers.framework_pinning=false"],
        ["no_equals_sign"],
    ],
)
def test_invalid_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        load_config(None, overrides)
