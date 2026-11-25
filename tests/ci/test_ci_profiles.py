# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the CI inventory and its process-owning launcher."""

import copy
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

E2E = Path(__file__).parents[1] / "e2e"
sys.path.insert(0, str(E2E))


def module(name):
    assert (E2E / f"{name}.py").is_file(), f"missing CI implementation: {name}"
    return importlib.import_module(name)


def test_all_selection_requires_the_full_cartesian_product():
    profiles = module("ci_profiles")
    selected = profiles.select_profiles(profiles.load_catalog(), "all")
    assert len(selected) == 12
    assert {(p["model_family"], p["topology"], p["mode"]) for p in selected} == {
        (family, topology, mode)
        for family in ("dense", "mamba", "gdn")
        for topology in ("agg", "disagg")
        for mode in ("guard", "native")
    }
    assert sum(len(p["cases"]) for p in selected) == 42
    assert all("TC-04" in p["cases"] for p in selected if p["mode"] == "guard")
    assert all(
        p["cases"] == ["TC-01", "TC-02", "TC-03"]
        for p in selected
        if p["mode"] == "native"
    )


@pytest.mark.parametrize("mapping,utilization", [([0, 0, 0], 0.35), ([0, 1, 1], 0.8)])
def test_catalog_rejects_overcommit_involving_decode_worker(mapping, utilization):
    profiles = module("ci_profiles")
    raw = copy.deepcopy(profiles.load_catalog())
    profile = next(p for p in raw["profiles"] if p["id"] == "vllm-dense-disagg-native")
    profile["resources"] = {
        "gpus": mapping,
        "gpu_memory_utilization": utilization,
    }
    with pytest.raises(ValueError, match="GPU"):
        profiles.validate_catalog(raw)


def test_diagnostic_rejects_three_workers_overcommitting_one_gpu(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(E2E / "run_ci_profile.py"),
            "--profile",
            "vllm-dense-disagg-native",
            "--output-dir",
            str(tmp_path / "run"),
            "--image",
            "local-diagnostic",
            "--expected-source-sha",
            "a" * 40,
            "--diagnostic",
            "--gpus",
            "0,0,0",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "GPU" in result.stderr


def test_forced_timeout_reaps_detached_owned_worker_only(tmp_path):
    runner = module("run_ci_profile")
    pid_file = tmp_path / "detached.pid"
    detached = (
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "time.sleep(60)"
    )
    parent = (
        "import pathlib,signal,subprocess,sys,time; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"child=subprocess.Popen([sys.executable,'-c',{detached!r}],"
        "start_new_session=True); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        outcome = runner.run_process(
            [sys.executable, "-c", parent],
            os.environ.copy(),
            tmp_path / "timeout.log",
            timeout=0.5,
            grace=0.1,
        )
        assert outcome["forced_kill"] is True
        assert outcome["status"] == "FAIL"
        pid = int(pid_file.read_text())
        stat = Path(f"/proc/{pid}/stat")
        assert not stat.exists() or stat.read_text().split(") ", 1)[1].split()[0] == "Z"
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait()
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_explicit_selection_is_diagnostic_and_does_not_inherit_full_coverage():
    profiles = module("ci_profiles")
    catalog = profiles.load_catalog()
    name = profiles.select_profiles(catalog, "all")[0]["id"]
    manifest = profiles.make_manifest(
        catalog, name, "image@sha256:" + "a" * 64, "b" * 40, "123", "2"
    )
    assert manifest["selection"] == "diagnostic"
    assert [p["id"] for p in manifest["profiles"]] == [name]
    assert manifest["smokes"] == []
    assert manifest["run_attempt"] == "2"


@pytest.mark.parametrize(
    "selection",
    ["", "all,missing", "missing", "vllm-dense-agg-guard,vllm-dense-agg-guard"],
)
def test_bad_selections_are_rejected(selection):
    profiles = module("ci_profiles")
    with pytest.raises(ValueError):
        profiles.select_profiles(profiles.load_catalog(), selection)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "bad_mode",
        "unsafe_arg",
        "float_revision",
        "unsupported_engine",
    ],
)
def test_catalog_rejects_invalid_oracle_or_identity(mutation):
    profiles = module("ci_profiles")
    raw = json.loads((E2E / "ci_profiles.json").read_text())
    if mutation == "duplicate":
        raw["profiles"].append(copy.deepcopy(raw["profiles"][0]))
    elif mutation == "missing":
        raw["profiles"].pop()
    elif mutation == "bad_mode":
        raw["profiles"][0]["mode"] = "disabled"
    elif mutation == "unsafe_arg":
        raw["models"]["dense"]["launch"]["worker_args"] = ["--model=wrong"]
    elif mutation == "float_revision":
        raw["models"]["dense"]["revision"] = "main"
    else:
        raw["profiles"][0]["integration"] = "sglang-linker"
    with pytest.raises(ValueError):
        profiles.validate_catalog(raw)


def test_profile_fingerprint_covers_effective_settings():
    profiles = module("ci_profiles")
    profile = profiles.select_profiles(profiles.load_catalog(), "all")[0]
    changed = copy.deepcopy(profile)
    changed["resources"]["gpu_memory_utilization"] -= 0.01
    assert profiles.fingerprint(profile) != profiles.fingerprint(changed)


@pytest.mark.parametrize("topology", ["agg", "disagg"])
@pytest.mark.parametrize("mode", ["guard", "native"])
def test_nemotron_budget_covers_observed_24gb_runtime(topology, mode):
    profiles = module("ci_profiles")
    profile = profiles.get_profile(
        profiles.load_catalog(), f"vllm-mamba-{topology}-{mode}"
    )
    utilization = profile["resources"]["gpu_memory_utilization"]
    # CI45586225652 measured ~8GiB non-cache use on A10G, with ~22GiB
    # usable by vLLM. Leave at least 1.5GiB for cache/profile variation,
    # while two colocated prefills stay within the existing 90% ceiling.
    assert 22 * utilization >= 8 + 1.5
    assert utilization * 2 <= 0.9


def test_plans_preserve_hybrid_geometry_and_native_case_applicability(tmp_path):
    profiles = module("ci_profiles")
    runner = module("run_ci_profile")
    catalog = profiles.load_catalog()
    for profile in profiles.select_profiles(catalog, "all"):
        command = runner.build_command(
            profile, tmp_path / profile["id"], tmp_path / "snapshot", "tag"
        )
        assert command[command.index("--model") + 1] == profile["model_id"]
        assert command[command.index("--model-path") + 1] == str(tmp_path / "snapshot")
        assert command[command.index("--mode") + 1] == (
            "nonguard" if profile["mode"] == "native" else "guard"
        )
        assert command[command.index("--cases") + 1].split(",") == profile["cases"]
        assert "--allow-blocked" not in command
        assert "--external-services" not in command
        if profile["model_family"] != "dense":
            assert "--keep-hybrid-kv-cache-manager" in command
            assert command[command.index("--ssm-groups") + 1] in ("3", "6")


def test_hybrid_disaggregation_selects_nixl_compatible_conv_state_layout(tmp_path):
    profiles = module("ci_profiles")
    runner = module("run_ci_profile")
    for profile in profiles.select_profiles(profiles.load_catalog(), "all"):
        command = runner.build_command(profile, tmp_path, tmp_path / "snapshot", "tag")
        required = (
            profile["model_family"] != "dense" and profile["topology"] == "disagg"
        )
        assert ("--worker-env=VLLM_SSM_CONV_STATE_LAYOUT=DS" in command) is required


def test_dry_run_does_not_download_or_create_output(tmp_path):
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(E2E / "run_ci_profile.py"),
            "--profile",
            "vllm-mamba-agg-guard",
            "--output-dir",
            str(output),
            "--image",
            "repo@sha256:" + "a" * 64,
            "--expected-source-sha",
            "b" * 40,
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["profile_id"] == "vllm-mamba-agg-guard"
    assert plan["diagnostic"] is False
    assert not output.exists()


def test_snapshot_override_requires_diagnostic_and_matching_revision(tmp_path):
    runner = module("run_ci_profile")
    profile = module("ci_profiles").get_profile(
        module("ci_profiles").load_catalog(), "vllm-dense-agg-guard"
    )
    snapshot = tmp_path / profile["model_revision"]
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="diagnostic"):
        runner.validate_snapshot_override(profile, snapshot, diagnostic=False)
    assert (
        runner.validate_snapshot_override(profile, snapshot, diagnostic=True)
        == snapshot.resolve()
    )


def test_run_process_captures_failure_and_log(tmp_path):
    runner = module("run_ci_profile")
    outcome = runner.run_process(
        [sys.executable, "-c", "print('evidence', flush=True); raise SystemExit(7)"],
        os.environ.copy(),
        tmp_path / "console.log",
        timeout=10,
        grace=1,
    )
    assert outcome["returncode"] == 7
    assert outcome["status"] == "FAIL"
    assert "evidence" in (tmp_path / "console.log").read_text()


def test_timeout_reaps_real_owned_child(tmp_path):
    runner = module("run_ci_profile")
    pidfile = tmp_path / "pid"
    code = (
        "import os,time; from pathlib import Path; "
        f"Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    outcome = runner.run_process(
        [sys.executable, "-c", code],
        os.environ.copy(),
        tmp_path / "console.log",
        timeout=0.3,
        grace=0.2,
    )
    assert outcome["status"] == "FAIL"
    assert outcome["timed_out"] is True
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_sigterm_propagates_to_real_child_and_preserves_evidence(tmp_path):
    module("run_ci_profile")
    pidfile = tmp_path / "child.pid"
    outcome_file = tmp_path / "outcome.json"
    child = (
        "import os,time; from pathlib import Path; "
        f"Path({str(pidfile)!r}).write_text(str(os.getpid())); "
        "print('ready',flush=True); time.sleep(60)"
    )
    driver = "\n".join(
        [
            "import json,os,sys",
            f"sys.path.insert(0, {str(E2E)!r})",
            "from pathlib import Path",
            "from run_ci_profile import run_process",
            f"child={child!r}",
            f"log=Path({str(tmp_path / 'console.log')!r})",
            "outcome=run_process([sys.executable,'-c',child],"
            "os.environ.copy(),log,timeout=60,grace=.2)",
            f"Path({str(outcome_file)!r}).write_text(json.dumps(outcome))",
        ]
    )
    process = subprocess.Popen([sys.executable, "-c", driver])
    try:
        deadline = time.monotonic() + 10
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pidfile.exists()
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=10)
        outcome = json.loads(outcome_file.read_text())
        assert outcome["status"] == "FAIL"
        assert outcome["interrupted"] is True
        with pytest.raises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)
        assert "ready" in (tmp_path / "console.log").read_text()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_setup_failure_writes_failure_envelope_without_starting_harness(
    tmp_path, monkeypatch
):
    runner = module("run_ci_profile")
    profiles = module("ci_profiles")
    profile = profiles.get_profile(profiles.load_catalog(), "vllm-dense-agg-guard")

    def reject(*_args, **_kwargs):
        raise RuntimeError("prefetch unavailable")

    monkeypatch.setattr(runner, "prepare_snapshot", reject)
    monkeypatch.setattr(
        runner,
        "source_identity",
        lambda _expected: {"source_sha": "b" * 40, "workflow_sha": "b" * 40},
    )
    code = runner.run_profile(
        profile, tmp_path / "out", "repo@sha256:" + "a" * 64, "b" * 40, "123", "1"
    )
    report = json.loads((tmp_path / "out/profile-result.json").read_text())
    assert code != 0
    assert report["status"] == "FAIL"
    assert report["setup"]["status"] == "FAIL"
    assert report["harness"]["status"] == "NOT_RUN"
    assert "prefetch unavailable" in report["error"]
    assert not (tmp_path / "out/run").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("image", "repo:latest"),
        ("source_sha", "main"),
        ("run_id", ""),
        ("run_attempt", ""),
    ],
)
def test_manifest_rejects_unreproducible_identity(field, value):
    profiles = module("ci_profiles")
    arguments = dict(
        catalog=profiles.load_catalog(),
        selection="all",
        image="repo@sha256:" + "a" * 64,
        source_sha="b" * 40,
        run_id="123",
        run_attempt="1",
    )
    arguments[field] = value
    with pytest.raises(ValueError):
        profiles.make_manifest(**arguments)


@pytest.mark.parametrize(
    "cleanup",
    [{}, {"owned_processes": "invalid"}, {"owned_processes": [{"error": "survived"}]}],
)
def test_missing_or_failed_cleanup_never_passes(tmp_path, cleanup):
    runner = module("run_ci_profile")
    path = tmp_path / "run/stack"
    path.mkdir(parents=True)
    (path / "cleanup.json").write_text(json.dumps(cleanup))
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        runner.cleanup_evidence(tmp_path, smoke=False)


def test_manual_source_override_records_distinct_recipe_and_installed_source(
    tmp_path, monkeypatch
):
    runner = module("run_ci_profile")
    recipe = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=E2E, text=True
    ).strip()
    source = "b" * 40
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"kvcr_sha": source}))
    monkeypatch.setattr(runner, "BUILD_METADATA", metadata)
    monkeypatch.setenv("GITHUB_SHA", recipe)
    result = runner.source_identity(source)
    assert result["checked_out_source_sha"] == recipe
    assert result["workflow_sha"] == recipe
    assert result["installed_source_sha"] == source
    assert result["source_mismatches"] == []
    metadata.write_text(json.dumps({"kvcr_sha": "c" * 40}))
    with pytest.raises(ValueError, match="installed source"):
        runner.source_identity(source)


@pytest.mark.parametrize("smoke", [False, True])
def test_real_harness_process_is_wrapped_with_identity_and_cleanup(
    tmp_path, monkeypatch, smoke
):
    runner = module("run_ci_profile")
    profiles = module("ci_profiles")
    profile = profiles.get_profile(
        profiles.load_catalog(), profiles.SMOKE_ID if smoke else "vllm-dense-agg-native"
    )
    output = tmp_path / "out"
    cleanup_path = "cleanup.json" if smoke else "stack/cleanup.json"
    payload = {
        "status": "success" if smoke else "PASS",
        "phase_durations_seconds": {"cleanup": 0.25},
    }
    code = "\n".join(
        [
            "import json,os; from pathlib import Path",
            f"root=Path({str(output / 'run')!r}); root.mkdir(parents=True)",
            f"(root/'result.json').write_text(json.dumps({payload!r}))",
            f"cleanup=root/{cleanup_path!r}; cleanup.parent.mkdir(exist_ok=True)",
            "cleanup.write_text(json.dumps({'schema_version':2, 'status':'PASS',"
            "'reclaimed':True, 'owned_processes':[{'name':'test-worker',"
            "'pid':os.getpid(),'exit_code':0}],"
            "'process_states':[{'name':'test-worker',"
            "'pid':os.getpid(),'stopped':True}],"
            "'owned_paths':[{'path':'/tmp/owned-fixture',"
            "'status':'PASS','errors':[]}], 'duration_seconds':.25}))",
            "print('harness evidence',flush=True)",
        ]
    )
    monkeypatch.setattr(
        runner,
        "source_identity",
        lambda _expected: {
            "source_sha": "b" * 40,
            "workflow_sha": "b" * 40,
            "installed_source_sha": "b" * 40,
        },
    )
    monkeypatch.setattr(
        runner, "prepare_snapshot", lambda *_args: tmp_path / "snapshot"
    )
    monkeypatch.setattr(
        runner, "build_command", lambda *_args: [sys.executable, "-c", code]
    )
    assert (
        runner.run_profile(
            profile, output, "repo@sha256:" + "a" * 64, "b" * 40, "123", "1"
        )
        == 0
    )
    envelope = json.loads((output / "profile-result.json").read_text())
    assert envelope["status"] == "PASS"
    assert envelope["profile_id"] == profile["id"]
    assert envelope["model_family"] == profile["model_family"]
    assert envelope["actual_gpus"] == profile["resources"]["gpus"]
    assert envelope["cleanup"]["duration_seconds"] >= 0.25
    assert "harness evidence" in (output / "console.log").read_text()


def test_default_manifest_does_not_schedule_retired_peer_smoke():
    profiles = module("ci_profiles")
    manifest = profiles.make_manifest(
        profiles.load_catalog(), "all", "image@sha256:" + "a" * 64, "b" * 40, "123", "1"
    )
    assert manifest["smokes"] == []
    assert len(manifest["profiles"]) == 12
    assert sum(len(row["cases"]) for row in manifest["profiles"]) == 42


def test_retired_smoke_is_not_a_production_functional_selection():
    profiles = module("ci_profiles")
    with pytest.raises(ValueError):
        profiles.select_profiles(profiles.load_catalog(), profiles.SMOKE_ID)
