# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise local orchestration with real Git and fake Docker, without GPUs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / ".github/scripts/run_vllm_e2e.sh"
IMAGE_ID = "sha256:" + "a" * 64
DIGEST = "sha256:" + "b" * 64
CONTAINER_ID = "c" * 64

DOCKER = r"""
import json, os, pathlib, sys
config = json.loads(pathlib.Path(os.environ["FAKE_CONFIG"]).read_text())
args = sys.argv[1:]
with pathlib.Path(os.environ["FAKE_CALLS"]).open("a") as stream:
    stream.write(json.dumps(args) + "\n")
state_path = pathlib.Path(os.environ["FAKE_STATE"])
if args[:2] == ["image", "inspect"]:
    labels = dict(config["labels"])
    if config.get("bad_labels"):
        labels["org.opencontainers.image.revision"] = "0" * 40
    print(json.dumps([{"Id": config["image_id"], "Architecture": "amd64", "Os": "linux",
        "Config": {"Labels": labels},
        "RepoDigests": ["registry/kvcr@" + config["digest"]]}]))
elif args[0] == "create":
    label = args[args.index("--label") + 1].split("=", 1)
    mount = next(x for x in args if x.startswith("type=bind,src=")
                 and x.endswith(",dst=/evidence"))
    output = mount[len("type=bind,src="):-len(",dst=/evidence")]
    state_path.write_text(json.dumps({"label": label, "output": output}))
    print(config["container_id"])
elif args[0] == "cp":
    manifest = dict(config["manifest"])
    if config.get("bad_manifest"):
        manifest["dynamo_commit"] = "0" * 40
    pathlib.Path(args[-1]).write_text(json.dumps(manifest))
elif args[0] == "start":
    if config.get("start_exit"):
        sys.exit(config["start_exit"])
    state = json.loads(state_path.read_text())
    output = pathlib.Path(state["output"]) / "run"
    output.mkdir()
    (output / "result.json").write_text(json.dumps({
        "status": "passed",
        "peer_transfer_verified": not config.get("missing_proof", False)}))
elif args[0] == "inspect":
    state = json.loads(state_path.read_text())
    label = state["label"]
    print(json.dumps([{"Id": config["container_id"], "Config": {"Labels": {
        label[0]: "not-owned" if config.get("wrong_owner") else label[1]}}}]))
elif args[0] == "rm":
    assert args[-1] == config["container_id"]
elif args[0] == "pull":
    pass
else:
    raise SystemExit("Unexpected fake Docker call: " + repr(args))
"""


@pytest.fixture
def local_run(tmp_path):
    repo = tmp_path / "checkout"
    harness = repo / "tests/e2e/run_vllm.py"
    harness.parent.mkdir(parents=True)
    harness.write_text("# Committed fixture harness; fake Docker never executes it.\n")
    environment = dict(
        os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull
    )
    git = [
        "git",
        "-c",
        "user.name=CI Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=" + os.devnull,
    ]
    for args in (["init", "-q"], ["add", "."], ["commit", "-qm", "fixture"]):
        subprocess.run(
            git + args, cwd=repo, env=environment, check=True, capture_output=True
        )
    sha = subprocess.check_output(
        git + ["rev-parse", "HEAD"], cwd=repo, env=environment, text=True
    ).strip()
    manifest = {
        "kvcr_commit": sha,
        "dynamo_commit": "d" * 40,
        "vllm_adapter_commit": "e" * 40,
        "vllm_base_image": "vllm/base@sha256:" + "f" * 64,
    }
    labels = {
        "org.opencontainers.image.revision": sha,
        "io.ai-dynamo.kvcr.dynamo.sha": manifest["dynamo_commit"],
        "io.ai-dynamo.kvcr.vllm.sha": manifest["vllm_adapter_commit"],
        "io.ai-dynamo.kvcr.base-image": manifest["vllm_base_image"],
    }
    build = repo / "artifacts/build"
    build.mkdir(parents=True)
    (build / "source-revisions.json").write_text(json.dumps(manifest))
    (build / "image-identity.json").write_text(
        json.dumps(
            {
                "published": False,
                "image_uri": "local/kvcr:fixture",
                "verified_labels": labels,
            }
        )
    )
    (build / "build-metadata.json").write_text(
        json.dumps({"containerimage.config.digest": IMAGE_ID})
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text(f"#!{sys.executable}\n" + DOCKER)
    docker.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    config_path = tmp_path / "config.json"
    environment.update(
        PATH=str(binary) + os.pathsep + environment["PATH"],
        FAKE_CONFIG=str(config_path),
        FAKE_CALLS=str(calls),
        FAKE_STATE=str(tmp_path / "state.json"),
    )
    output = repo / "evidence"

    def invoke(*arguments, **overrides):
        config_path.write_text(
            json.dumps(
                {
                    "manifest": manifest,
                    "labels": labels,
                    "image_id": IMAGE_ID,
                    "container_id": CONTAINER_ID,
                    "digest": DIGEST,
                    **overrides,
                }
            )
        )
        result = subprocess.run(
            ["bash", str(SCRIPT), "--output-dir", str(output), *arguments],
            cwd=repo,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
        )
        commands = (
            [json.loads(line) for line in calls.read_text().splitlines()]
            if calls.exists()
            else []
        )
        return result, commands

    return invoke, output, harness, build


def test_local_runner_scopes_gpus_mounts_and_cleans_its_container(local_run):
    invoke, output, _, _ = local_run
    result, commands = invoke(
        "--gpus",
        "0,1",
        "--cpus",
        "8",
        "--cpuset-cpus",
        "0-3,8-11",
        "--memory",
        "64g",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    create = next(call for call in commands if call[0] == "create")
    assert create[create.index("--gpus") + 1] == '"device=0,1"'
    assert create[create.index("--shm-size") + 1] == "16g"
    assert create[create.index("--network") + 1] == "bridge"
    assert create[create.index("--cpus") + 1] == "8.0"
    assert create[create.index("--cpuset-cpus") + 1] == "0-3,8-11"
    assert create[create.index("--memory") + 1] == "64g"
    assert create[create.index("--memory-swap") + 1] == "64g"
    assert any(item.endswith("dst=/kvcr-tests/run_vllm.py,readonly") for item in create)
    assert "--publish" not in create and "--privileged" not in create
    assert IMAGE_ID in create
    assert commands[-1] == ["rm", "--force", CONTAINER_ID]
    evidence = json.loads((output / "runner-result.json").read_text())
    assert evidence["status"] == "passed" and evidence["container_removed"] is True


@pytest.mark.parametrize("devices", ["all", "0", "0,0", "00,0", "0,1,2"])
def test_invalid_gpu_selection_never_calls_docker(local_run, devices):
    invoke, output, _, _ = local_run
    result, commands = invoke("--gpus", devices)
    assert result.returncode != 0
    assert commands == [] and not output.exists()


def test_refused_output_directory_is_untouched(local_run):
    invoke, output, _, _ = local_run
    output.mkdir()
    (output / "sentinel").write_text("keep me")
    result, commands = invoke("--gpus", "0,1")
    assert result.returncode != 0 and commands == []
    assert [path.name for path in output.iterdir()] == ["sentinel"]
    assert (output / "sentinel").read_text() == "keep me"


@pytest.mark.parametrize("fault", ["bad_labels", "bad_manifest"])
def test_provenance_mismatch_prevents_test_execution(local_run, fault):
    invoke, output, _, _ = local_run
    result, commands = invoke("--gpus", "0,1", **{fault: True})
    assert result.returncode != 0
    assert not any(call[0] == "start" for call in commands)
    assert json.loads((output / "runner-result.json").read_text())["status"] == "failed"
    if fault == "bad_manifest":
        assert commands[-1] == ["rm", "--force", CONTAINER_ID]
    else:
        assert not any(call[0] == "create" for call in commands)


def test_container_failure_propagates_and_cleanup_runs(local_run):
    invoke, output, _, _ = local_run
    result, commands = invoke("--gpus", "0,1", start_exit=7)
    assert result.returncode == 7
    assert commands[-1] == ["rm", "--force", CONTAINER_ID]
    assert json.loads((output / "runner-result.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("fault", ["missing_proof", "wrong_owner"])
def test_missing_proof_or_cleanup_ownership_failure_cannot_pass(local_run, fault):
    invoke, output, _, _ = local_run
    result, commands = invoke("--gpus", "0,1", **{fault: True})
    assert result.returncode != 0
    assert json.loads((output / "runner-result.json").read_text())["status"] == "failed"
    if fault == "wrong_owner":
        assert not any(call[0] == "rm" for call in commands)


def test_uncommitted_harness_is_rejected_before_docker(local_run):
    invoke, _, harness, _ = local_run
    harness.write_text("# changed after the image was built\n")
    result, commands = invoke("--gpus", "0,1")
    assert result.returncode != 0 and commands == []


def test_published_image_can_be_pulled_by_digest(local_run):
    invoke, _, _, build = local_run
    image = "registry/kvcr@" + DIGEST
    result, commands = invoke(
        "--gpus",
        "0,1",
        "--image",
        image,
        "--revisions",
        str(build / "source-revisions.json"),
        "--pull",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert commands[0] == ["pull", "--platform", "linux/amd64", image]


def test_local_image_config_digest_must_match_build(local_run):
    invoke, _, _, build = local_run
    (build / "build-metadata.json").write_text(
        json.dumps(
            {
                "containerimage.config.digest": "sha256:" + "0" * 64,
            }
        )
    )
    result, commands = invoke("--gpus", "0,1")
    assert result.returncode != 0
    assert not any(call[0] == "create" for call in commands)
