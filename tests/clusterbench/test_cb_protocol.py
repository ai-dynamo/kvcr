# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from pathlib import Path

import pytest
from clusterbench.config import RunConfig, decode_config, encode_config, load_config


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
    "overrides,message",
    [
        (["tiers.block_bytes=1000"], "tiers.block_bytes must be a multiple of 4096"),
        # Options that do not exist.
        (["router.mode=cost"], r"unknown field `mode` - at `\$.router`"),
        (["router.same_node_prob=1"], r"unknown field `same_node_prob`"),
        (["tiers.gpu_block=64"], r"unknown field `gpu_block` - at `\$.tiers`"),
        (["engine.abandon_timeout_ms=1500"], "must be >= 2x operation timeout"),
        (["phases.mesh_warm=some"], "phases.mesh_warm must be full or none"),
        (["engine.verify=maybe"], "engine.verify must be none, sample or full"),
        (["engine.profile=sglang"], r"unknown field `profile` - at `\$.engine`"),
        (
            ["tiers.g2_blocks=0", "tiers.framework_pinning=false"],
            "no G2 pool and no pinning cannot be a source",
        ),
        (["no_equals_sign"], "override must be section.field=value"),
    ],
)
def test_invalid_config_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        load_config(None, overrides)
