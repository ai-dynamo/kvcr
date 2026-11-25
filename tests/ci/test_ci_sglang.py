# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SGLang integration CPU contracts: catalog, launch, image and lane wiring."""

import copy
import importlib
import json
import re
import sys
from pathlib import Path

import pytest

E2E = Path(__file__).parents[1] / "e2e"
REPO = E2E.parents[1]
sys.path.insert(0, str(E2E))

INTEGRATIONS = ["sglang-hicache", "sglang-linker"]
# Registered in ci_lanes.json, so required by the GPU E2E verdict.
ACTIVE = {"vllm", "sglang-hicache"}


def _raw(integration):
    from ci_integrations import get_integration

    return json.loads(get_integration(integration).CATALOG.read_text())


@pytest.mark.parametrize("integration", INTEGRATIONS)
def test_catalog_covers_every_model_family_aggregated_native(integration):
    from ci_integrations import get_integration
    from ci_profiles import load_catalog, select_profiles

    adapter = get_integration(integration)
    catalog = load_catalog(integration=integration)
    profiles = select_profiles(catalog)
    assert {p["id"] for p in profiles} == adapter.required_profile_ids(catalog)
    assert {p["model_id"] for p in profiles} == {
        "Qwen/Qwen3-0.6B",
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "Qwen/Qwen3.5-0.8B",
    }
    for profile in profiles:
        assert (profile["topology"], profile["mode"]) == ("agg", "native")
        assert profile["cases"] == ["TC-01", "TC-02", "TC-03"]
    assert adapter.smoke_profiles(catalog) == []
    assert adapter.IMAGE_PROFILE == integration


@pytest.mark.parametrize("integration", INTEGRATIONS)
@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["profiles"][0].update(topology="disagg"),
        lambda raw: raw["profiles"][0].update(mode="guard"),
        lambda raw: raw["profiles"].pop(),
        lambda raw: raw["models"]["dense"]["launch"].update(max_model_len=32000),
        lambda raw: raw["models"]["dense"]["launch"].update(g2_slots=32),
        lambda raw: raw["models"]["dense"]["launch"].update(
            worker_args=["--page-size=1"]
        ),
        lambda raw: raw["models"]["dense"]["launch"].update(
            worker_args=["--attention-backend=fa3", "--attention-backend=triton"]
        ),
        lambda raw: raw["profiles"][0]["resources"].update(gpus=[0, 0, 1]),
    ],
)
def test_catalog_rejects_unsupported_configuration(integration, mutate):
    from ci_profiles import validate_catalog

    raw = copy.deepcopy(_raw(integration))
    mutate(raw)
    with pytest.raises(ValueError):
        validate_catalog(raw)


@pytest.mark.parametrize("integration", INTEGRATIONS)
def test_launch_runs_the_sglang_engine_with_catalog_settings(integration):
    from ci_integrations import get_integration
    from ci_profiles import load_catalog, select_profiles

    adapter = get_integration(integration)
    for profile in select_profiles(load_catalog(integration=integration)):
        command = adapter.build_command(
            profile, Path("/output"), Path("/snapshot"), "tag"
        )
        launch = profile["launch"]
        joined = " ".join(command)
        assert command[2].endswith("run_phase1.py")
        assert "--engine sglang" in joined and "--mode nonguard" in joined
        assert "--topology agg" in joined and "--model-path /snapshot" in joined
        assert f"--max-model-len {launch['max_model_len']}" in joined
        assert ("--batch-invariant" in command) == launch["batch_invariant"]
        assert [a for a in command if a.startswith("--worker-arg=")] == [
            f"--worker-arg={arg}" for arg in launch["worker_args"]
        ]
        assert adapter.environment(profile) == {
            "DYN_ROUTER_WORKER_SELECTION_POLICY": "kvcr"
        }
        assert 8000 in adapter.runtime_resources(profile)["tcp_ports"]


def test_harness_accepts_every_sglang_launch_argument(tmp_path):
    from ci_integrations import get_integration
    from ci_profiles import load_catalog, select_profiles

    run_phase1 = importlib.import_module("run_phase1")
    for integration in INTEGRATIONS:
        adapter = get_integration(integration)
        for profile in select_profiles(load_catalog(integration=integration)):
            command = adapter.build_command(profile, tmp_path, tmp_path, "tag")
            args = run_phase1.arguments(command[3:])
            assert args.engine == "sglang" and args.mode == "nonguard"
            assert args.worker_arg == profile["launch"]["worker_args"]


def test_hybrid_attention_backend_replaces_the_harness_default(tmp_path):
    from phase1.stack import Settings, Stack

    settings = Settings(
        model="m",
        output=tmp_path,
        engine="sglang",
        mode="nonguard",
        batch_invariant=True,
        extra_worker_args=["--attention-backend=triton"],
    )
    command = Stack(settings, tmp_path / "stack").sglang_worker_command(0, kvcr=False)
    backends = [a for a in command if a.startswith("--attention-backend")]
    assert backends == ["--attention-backend=triton"]
    settings.extra_worker_args = []
    command = Stack(settings, tmp_path / "stack2").sglang_worker_command(0, kvcr=False)
    assert command[command.index("--attention-backend") + 1] == "fa3"


def test_sglang_image_profile_binds_its_digest_pinned_recipe():
    images = importlib.import_module("ci_images")
    recipe = (REPO / "Dockerfile.quick-start-sglang").read_text()
    base = re.search(r"^ARG SGLANG_IMAGE=(.+@sha256:[0-9a-f]{64})$", recipe, re.M)
    assert base is not None
    record = images.build_inputs(
        "sglang-hicache", "a" * 40, "b" * 40, base.group(1), "nightly", "1.2.0.dev1"
    )
    assert record["recipe"] == "Dockerfile.quick-start-sglang"
    assert record["base_argument"] == "SGLANG_IMAGE"
    assert record["python_version"] == "3.12"
    # The identity build args are declared after every install layer, so a new
    # commit or recipe does not invalidate the cached SGLang and Dynamo layers.
    identity = recipe.index("ARG RESOLVED_INPUTS_SHA256")
    assert recipe.index("--report /opt/kvcr-install-report.json") < identity
    # CI_IMAGE_PROFILE selects the fork commit, so it is declared early; it is
    # constant per image profile and does not change from commit to commit.
    for name in ("KVCR_COMMIT", "CI_RECIPE_SHA", "DYNAMO_POLICY"):
        assert recipe.index(f"ARG {name}") > recipe.index("COPY src /opt/kvcr-src/src")


def test_one_dynamo_version_serves_every_image(monkeypatch, tmp_path, capsys):
    images = importlib.import_module("ci_images")
    seen = []

    def resolve(policy, profiles, query=None):
        seen.append(profiles)
        return "1.2.0.dev1"

    monkeypatch.setattr(images, "resolve_for_profiles", resolve)
    common = ["--source-sha", "a" * 40, "--recipe-sha", "b" * 40]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ci_images.py",
            "--image-profile",
            "vllm",
            *common,
            "--base-image",
            "v@sha256:" + "c" * 64,
            "--dynamo-profiles",
            "vllm,sglang-hicache",
            "--output",
            str(tmp_path / "v.json"),
        ],
    )
    assert images.main() == 0
    assert seen == [["vllm", "sglang-hicache"]]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ci_images.py",
            "--image-profile",
            "sglang-hicache",
            *common,
            "--base-image",
            "s@sha256:" + "d" * 64,
            "--dynamo-version",
            "1.2.0.dev1",
            "--output",
            str(tmp_path / "s.json"),
        ],
    )
    assert images.main() == 0
    assert seen == [["vllm", "sglang-hicache"]]  # reused, not resolved again
    capsys.readouterr()
    versions = {
        json.loads((tmp_path / name).read_text())["dynamo_version"]
        for name in ("v.json", "s.json")
    }
    assert versions == {"1.2.0.dev1"}


def _jobs(text: str) -> dict[str, dict[str, str]]:
    """Top-level job blocks of a workflow: each job's ``key: value`` lines."""
    body = text.split("\njobs:\n", 1)[1]
    jobs: dict[str, dict[str, str]] = {}
    for block in re.split(r"\n(?=  [a-z][a-z0-9-]*:\n)", "\n" + body):
        header = re.match(r"\n?  ([a-z][a-z0-9-]*):\n", block)
        if header is not None:
            jobs[header.group(1)] = dict(
                re.findall(r"^ {4,6}([a-z_]+): (.+)$", block, re.MULTILINE)
            )
    return jobs


def test_every_active_lane_has_its_root_workflow_jobs():
    verdict = importlib.import_module("ci_verdict")
    lanes = verdict.load_lanes()
    jobs = _jobs((REPO / ".github/workflows/docker-build.yml").read_text())
    assert set(lanes) == ACTIVE
    gate = jobs["gpu-verdict"]["needs"].strip("[]").split(", ")
    for name, lane in lanes.items():
        build, test = jobs[lane["build_job"]], jobs[lane["test_job"]]
        assert build["uses"] == "./.github/workflows/build-ci-image.yml"
        assert build["image_profile"] == name
        assert test["needs"] == lane["build_job"]
        assert {lane["build_job"], lane["test_job"]} <= set(gate)
        assert f"needs.{lane['build_job']}.outputs.image" in test["image"]
        if name != "vllm":
            assert test["integration"] == name


def test_linker_lane_runs_without_gating_the_verdict():
    jobs = _jobs((REPO / ".github/workflows/docker-build.yml").read_text())
    gate = jobs["gpu-verdict"]["needs"].strip("[]").split(", ")
    assert jobs["build-sglang-linker"]["image_profile"] == "sglang-linker"
    assert jobs["e2e-sglang-linker"]["integration"] == "sglang-linker"
    assert not {"build-sglang-linker", "e2e-sglang-linker"} & set(gate)


@pytest.mark.parametrize(
    ("integration", "mode"),
    [("sglang-hicache", "hicache"), ("sglang-linker", "linker")],
)
def test_each_integration_selects_its_sglang_mode(integration, mode, tmp_path):
    from ci_integrations import get_integration
    from ci_profiles import load_catalog, select_profiles

    run_phase1 = importlib.import_module("run_phase1")
    adapter = get_integration(integration)
    profile = select_profiles(load_catalog(integration=integration))[0]
    command = adapter.build_command(profile, tmp_path, tmp_path, "tag")
    assert run_phase1.arguments(command[3:]).sglang_mode == mode


@pytest.mark.parametrize(
    ("config", "layers"),
    [
        ({"num_hidden_layers": 28}, 28),
        (
            {
                "num_hidden_layers": 4,
                "hybrid_override_pattern": "M-M*",
            },
            1,
        ),
        (
            {
                "text_config": {
                    "num_hidden_layers": 4,
                    "layer_types": [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                    ],
                }
            },
            1,
        ),
    ],
)
def test_linker_block_divisor_counts_kv_cache_layers(config, layers):
    from types import SimpleNamespace

    from phase1.stack import kv_cache_layers

    def namespace(value):
        return SimpleNamespace(
            **{
                key: namespace(item) if isinstance(item, dict) else item
                for key, item in value.items()
            }
        )

    assert kv_cache_layers(namespace(config)) == layers


def test_linker_telemetry_keeps_raw_values_and_rejects_malformed_counters():
    from phase1.telemetry import (
        TIER_READ_BYTES,
        TIER_WRITE_BYTES,
        metric,
        parse_log,
    )

    stamp = "2026-11-25T09:46:31.385372Z"
    good = {
        "kvcr_transfer_blocks[local_fill].counter": 9,
        "kvcr_transfer_bytes[local_fill].counter": 66060288,
        "kvcr_transfer_blocks[source_write].counter": 252,
    }
    line = f"{stamp} INFO KVCR telemetry rank=0 interval_s=5.0 stats="
    snapshots, errors = parse_log(line + json.dumps(good) + "\n", 28)
    assert errors == []
    values = snapshots[0].metrics
    assert values[metric("kvcr_transfer_blocks", "source_write")] == 9
    assert values["_per_layer:" + metric("kvcr_transfer_blocks", "source_write")] == 252
    assert values[metric("kvcr_transfer_blocks", "local_fill")] == 9
    assert snapshots[0].provenance == {
        TIER_READ_BYTES: "derived",
        TIER_WRITE_BYTES: "measured:local_fill",
    }
    for bad in ({"kvcr_transfer_blocks[x].counter": "1"}, {"no-brackets": 1}):
        snapshots, errors = parse_log(line + json.dumps(bad) + "\n", 28)
        assert snapshots == [] and len(errors) == 1
