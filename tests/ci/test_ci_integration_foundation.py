# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine registration and immutable model preparation CPU contracts."""

import copy
import json
import sys
from pathlib import Path

import pytest

E2E = Path(__file__).parents[1] / "e2e"
sys.path.insert(0, str(E2E))


def test_vllm_adapter_preserves_every_existing_launch_argument(tmp_path):
    from ci_integrations import get_integration
    from ci_profiles import SMOKE_ID, get_profile, load_catalog, select_profiles

    adapter = get_integration("vllm")
    catalog = load_catalog()
    golden = json.loads(
        (Path(__file__).parent / "fixtures/vllm_launch_commands.json").read_text()
    )
    profiles = select_profiles(catalog) + [get_profile(catalog, SMOKE_ID)]
    actual = {}
    for profile in profiles:
        command = adapter.build_command(
            profile, Path("/output"), Path("/snapshot"), "tag"
        )
        actual[profile["id"]] = [
            argument.replace(str(E2E.resolve()), "<E2E>").replace(
                sys.executable, "<PYTHON>"
            )
            for argument in command
        ]
    assert actual == golden


def test_unqualified_integrations_cannot_be_selected():
    from ci_integrations import get_integration

    for name in ("sglang_hicache", "sglang_linker", "../vllm", "unknown"):
        with pytest.raises(ValueError, match="registered"):
            get_integration(name)


def test_model_pins_have_one_authoritative_inventory():
    from ci_profiles import CATALOG, MODELS, load_catalog

    raw = json.loads(CATALOG.read_text())
    pins = json.loads(MODELS.read_text())["models"]
    resolved = load_catalog()
    for family, model in raw["models"].items():
        assert set(model) == {"model_key", "launch"}
        checkpoint = pins[model["model_key"]]
        assert resolved["models"][family]["id"] == checkpoint["id"]
        assert resolved["models"][family]["revision"] == checkpoint["revision"]


def test_resolved_catalog_cannot_override_authoritative_checkpoint():
    from ci_profiles import load_catalog, validate_catalog

    raw = copy.deepcopy(load_catalog())
    raw["models"]["dense"]["revision"] = "a" * 40
    with pytest.raises(ValueError, match="checkpoint"):
        validate_catalog(raw)


def snapshot(tmp_path, revision="b" * 40):
    path = tmp_path / revision
    path.mkdir()
    (path / "config.json").write_text("{}")
    return path


def test_model_snapshot_rejects_missing_index_shard(tmp_path):
    from ci_model_access import validate_snapshot

    path = snapshot(tmp_path)
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (path / "model-00001-of-00002.safetensors").write_bytes(b"weights")
    with pytest.raises(ValueError, match="shard"):
        validate_snapshot({"model_revision": path.name}, path)


@pytest.mark.parametrize(
    "weight_name", ["../outside.safetensors", "/tmp/outside.safetensors"]
)
def test_model_index_rejects_unsafe_shard_paths(tmp_path, weight_name):
    from ci_model_access import validate_snapshot

    path = snapshot(tmp_path)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": weight_name}})
    )
    with pytest.raises(ValueError, match="shard"):
        validate_snapshot({"model_revision": path.name}, path)


def test_model_snapshot_accepts_hub_symlinked_complete_shards(tmp_path):
    from ci_model_access import validate_snapshot

    path = snapshot(tmp_path)
    blob = tmp_path / "weight-blob"
    blob.write_bytes(b"complete")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}})
    )
    (path / "model-00001-of-00001.safetensors").symlink_to(blob)
    inventory = validate_snapshot({"model_revision": path.name}, path)
    assert inventory["weight_files"] == [
        {"name": "model-00001-of-00001.safetensors", "bytes": 8}
    ]


def test_model_snapshot_rejects_empty_weights_and_malformed_config(tmp_path):
    from ci_model_access import validate_snapshot

    path = snapshot(tmp_path)
    (path / "model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        validate_snapshot({"model_revision": path.name}, path)
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "config.json").write_text("{broken")
    with pytest.raises(ValueError, match="config"):
        validate_snapshot({"model_revision": path.name}, path)


def test_model_preparation_failure_preserves_receipt(tmp_path):
    from ci_model_access import prepare_model

    def fail(*args, **kwargs):
        return {"status": "FAIL", "timed_out": True, "duration_seconds": 1.0}

    profile = {"model_id": "org/model", "model_revision": "b" * 40}
    with pytest.raises(RuntimeError, match="prefetch"):
        prepare_model(profile, tmp_path, {}, fail)
    receipt = json.loads((tmp_path / "model-receipt.json").read_text())
    assert receipt["status"] == "FAIL"
    assert receipt["requested_revision"] == "b" * 40
    assert receipt["cache_scope"] == "caller-configured"
    assert receipt["download"]["timed_out"] is True


def test_model_override_is_diagnostic_and_validated(tmp_path):
    from ci_model_access import prepare_model

    path = snapshot(tmp_path)
    (path / "model.safetensors").write_bytes(b"weights")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (path / name).write_text("{}")
    profile = {
        "integration": "vllm",
        "model_id": "Qwen/Qwen3-0.6B",
        "model_revision": path.name,
    }
    with pytest.raises(ValueError, match="diagnostic"):
        prepare_model(profile, tmp_path, {}, None, override=path)
    assert (
        prepare_model(profile, tmp_path, {}, None, override=path, diagnostic=True)
        == path.resolve()
    )
    receipt = json.loads((tmp_path / "model-receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["diagnostic_override"] is True
    assert receipt["resolved_revision"] == path.name


def test_model_preparation_uses_one_bounded_pinned_hub_process(tmp_path):
    import os
    import subprocess

    from ci_model_access import PREFETCH_TIMEOUT_SECONDS, prepare_model

    output = tmp_path / "output"
    output.mkdir()
    path = tmp_path / "models--Qwen--Qwen3-0.6B" / "snapshots" / ("b" * 40)
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (path / name).write_text("{}")
    stub = tmp_path / "huggingface_hub.py"
    stub.write_text(
        "__version__='test-hub'\n"
        "def snapshot_download(*, repo_id, revision, max_workers):\n"
        "    assert (repo_id,revision,max_workers)==('Qwen/Qwen3-0.6B','"
        + "b"
        * 40
        + "',4)\n"
        f"    return {str(path)!r}\n"
    )
    calls = []

    def supervisor(command, env, log_path, timeout):
        calls.append((command, timeout))
        with log_path.open("w") as log:
            result = subprocess.run(
                command, env=env, stdout=log, stderr=log, timeout=timeout
            )
        return {"status": "PASS" if result.returncode == 0 else "FAIL"}

    profile = {
        "integration": "vllm",
        "model_id": "Qwen/Qwen3-0.6B",
        "model_revision": "b" * 40,
    }
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "HF_HOME": str(tmp_path / "cache"),
    }
    assert prepare_model(profile, output, env, supervisor) == path.resolve()
    assert len(calls) == 1
    assert calls[0][1] == PREFETCH_TIMEOUT_SECONDS
    receipt = json.loads((output / "model-receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["resolved_revision"] == profile["model_revision"]
    assert receipt["huggingface_hub_version"] == "test-hub"
    assert receipt["weights_rehashed"] is False
    assert json.loads((output / "snapshot.json").read_text())["path"] == str(path)


def test_model_receipt_never_records_endpoint_credentials(tmp_path):
    from ci_model_access import cache_context

    context = cache_context(
        {"HF_ENDPOINT": "https://user:secret@example.com/private?token=secret"}
    )
    assert context["endpoint_host"] == "example.com"
    assert "secret" not in json.dumps(context)


def test_model_cache_context_distinguishes_configured_local_cache(tmp_path):
    from ci_model_access import cache_context

    context = cache_context({"GITHUB_ACTIONS": "true", "HF_HOME": str(tmp_path)})
    assert context["cache_scope"] == "caller-configured"
    assert context["cache_path"] == str(tmp_path / "hub")
    assert context["cache_free_bytes_before"] > 0
