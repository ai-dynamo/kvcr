# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model preparation preserves failures and checks adapter prerequisites."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "e2e"))

from ci_profiles import load_models

CHECKPOINTS = list(load_models().values())


def complete_snapshot(tmp_path, checkpoint):
    path = tmp_path / checkpoint["revision"]
    path.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (path / name).write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")
    profile = {
        "integration": "vllm",
        "model_id": checkpoint["id"],
        "model_revision": checkpoint["revision"],
    }
    return profile, path


def test_cache_context_failure_preserves_model_failure_receipt(tmp_path, monkeypatch):
    import ci_model_access

    def inaccessible_cache(_env):
        raise PermissionError("cache filesystem is inaccessible")

    monkeypatch.setattr(ci_model_access, "cache_context", inaccessible_cache)
    profile = {
        "integration": "vllm",
        "model_id": CHECKPOINTS[0]["id"],
        "model_revision": CHECKPOINTS[0]["revision"],
    }
    with pytest.raises(PermissionError, match="cache filesystem"):
        ci_model_access.prepare_model(profile, tmp_path, {}, None)
    receipt_path = tmp_path / "model-receipt.json"
    assert receipt_path.is_file(), "cache context failure lost its model receipt"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "FAIL"
    assert receipt["requested_repository"] == profile["model_id"]
    assert "PermissionError" in receipt["error"]
    assert receipt["duration_seconds"] >= 0


def test_cache_scope_is_identified_as_configuration_not_mount_observation():
    from ci_model_access import cache_context

    context = cache_context(
        {"GITHUB_ACTIONS": "true", "HF_HOME": "/scratch/huggingface"}
    )
    assert context["cache_scope"] == "job-container"
    assert context["cache_scope_evidence"] == "configured-environment-and-path"


@pytest.mark.parametrize("checkpoint", CHECKPOINTS, ids=lambda value: value["id"])
@pytest.mark.parametrize("name", ["tokenizer.json", "tokenizer_config.json"])
@pytest.mark.parametrize("invalid", ["missing", "malformed", "broken-symlink"])
def test_preparation_rejects_missing_or_unreadable_vllm_tokenizer_prerequisite(
    tmp_path, checkpoint, name, invalid
):
    from ci_model_access import prepare_model

    profile, path = complete_snapshot(tmp_path, checkpoint)
    (path / name).unlink()
    if invalid == "malformed":
        (path / name).write_text("{broken")
    elif invalid == "broken-symlink":
        (path / name).symlink_to(tmp_path / "missing-tokenizer-blob")
    with pytest.raises(ValueError, match="tokenizer"):
        prepare_model(profile, tmp_path, {}, None, override=path, diagnostic=True)
    receipt = json.loads((tmp_path / "model-receipt.json").read_text())
    assert receipt["status"] == "FAIL"
    assert name in receipt["error"]
    assert "resolved_path" not in receipt


@pytest.mark.parametrize("checkpoint", CHECKPOINTS, ids=lambda value: value["id"])
def test_preparation_records_vllm_tokenizer_prerequisites(tmp_path, checkpoint):
    from ci_model_access import prepare_model

    profile, path = complete_snapshot(tmp_path, checkpoint)
    assert (
        prepare_model(profile, tmp_path, {}, None, override=path, diagnostic=True)
        == path
    )
    receipt = json.loads((tmp_path / "model-receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert {"tokenizer.json", "tokenizer_config.json"}.issubset(
        receipt["required_files"]
    )
    assert receipt["tokenizer_files"] == [
        {"name": "tokenizer.json", "bytes": 2},
        {"name": "tokenizer_config.json", "bytes": 2},
    ]
