# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable build identity is shared; image recipes remain integration-owned."""

import hashlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "e2e"))
PATCH_SHA256 = "2f91a13244c0b7770d39577850d2b0e938084cb3a25eb1245a874104cb3889dc"


def test_nightly_resolves_once_to_common_usable_package_versions():
    images = importlib.import_module("ci_images")
    available = {
        "ai-dynamo": ["1.8.0.dev20261125", "1.8.0.dev20261124", "1.7.0"],
        "ai-dynamo-runtime": ["1.8.0.dev20261124", "1.7.0"],
    }
    assert images.resolve_dynamo("nightly", available) == "1.8.0.dev20261124"
    with pytest.raises(ValueError, match="common"):
        images.resolve_dynamo(
            "nightly", {"ai-dynamo": ["1.7.0"], "ai-dynamo-runtime": ["1.7.0"]}
        )


def test_explicit_version_must_exist_for_both_packages():
    images = importlib.import_module("ci_images")
    with pytest.raises(ValueError, match="available"):
        images.resolve_dynamo(
            "1.8.0.dev20261125",
            {
                "ai-dynamo": ["1.8.0.dev20261125"],
                "ai-dynamo-runtime": [],
            },
        )


def test_image_identity_binds_recipe_source_base_and_resolved_inputs():
    images = importlib.import_module("ci_images")
    record = images.build_inputs(
        "vllm",
        "a" * 40,
        "b" * 40,
        "vllm/vllm-openai@sha256:" + "c" * 64,
        "nightly",
        "1.8.0.dev20261124",
    )
    assert record["recipe_sha"] == "b" * 40
    assert record["source_sha"] == "a" * 40
    assert record["dynamo_policy"] == "nightly"
    assert record["dynamo_version"] == "1.8.0.dev20261124"
    altered = {**record, "recipe_sha": "d" * 40}
    assert images.inputs_digest(record) != images.inputs_digest(altered)
    with pytest.raises(ValueError, match="profile"):
        images.build_inputs(
            "unqualified",
            "a" * 40,
            "b" * 40,
            record["base_image"],
            "nightly",
            record["dynamo_version"],
        )
    with pytest.raises(ValueError, match="digest"):
        images.build_inputs(
            "vllm",
            "a" * 40,
            "b" * 40,
            "vllm:latest",
            "nightly",
            record["dynamo_version"],
        )


def test_resolved_inputs_declare_both_compatibility_overlays():
    images = importlib.import_module("ci_images")
    record = images.build_inputs(
        "vllm",
        "a" * 40,
        "b" * 40,
        "vllm@sha256:" + "c" * 64,
        "nightly",
        "1.8.0.dev20261123",
    )
    assert record.get("compatibility_overlays") == {
        "dynamo-vllm-18840": {
            "commit": "66ba12c69706a8775f48684e0a9ac4682d41ca8b",
            "patch_sha256": PATCH_SHA256,
        },
        "vllm-kvcr-public-channel": {"revision": "public-channel-v1"},
    }


def test_exact_dynamo_policy_cannot_resolve_to_another_release():
    images = importlib.import_module("ci_images")
    with pytest.raises(ValueError, match="policy"):
        images.build_inputs(
            "vllm",
            "a" * 40,
            "b" * 40,
            "vllm@sha256:" + "c" * 64,
            "1.7.0",
            "1.8.0.dev20261123",
        )


@pytest.fixture
def image_identity(tmp_path, monkeypatch):
    identity = importlib.import_module("ci_identity")
    overlay_dir = tmp_path / "dynamo"
    overlay_dir.mkdir()
    for name in identity.OVERLAY_FILES:
        (overlay_dir / name).write_bytes(b"Dynamo patched module")
    adapter = tmp_path / "manager.py"
    adapter.write_bytes(b"from kvcr import ZmqPeerControlChannel\n")
    monkeypatch.setattr(
        identity.metadata,
        "distribution",
        lambda _name: SimpleNamespace(locate_file=lambda _path: adapter),
    )
    record = {
        "schema_version": 2,
        "kvcr_sha": "a" * 40,
        "recipe_sha": "b" * 40,
        "image_profile": "vllm",
        "resolved_inputs_sha256": "c" * 64,
        "compatibility_overlays": {
            "dynamo-vllm-18840": {
                "commit": identity.OVERLAY_COMMIT,
                "patch_sha256": PATCH_SHA256,
            },
            "vllm-kvcr-public-channel": {"revision": "public-channel-v1"},
        },
        "dynamo_vllm_fix": {
            "commit": identity.OVERLAY_COMMIT,
            "patch_sha256": PATCH_SHA256,
            "files": {
                name: hashlib.sha256((overlay_dir / name).read_bytes()).hexdigest()
                for name in identity.OVERLAY_FILES
            },
        },
        "vllm_kvcr_public_channel": {
            "revision": "public-channel-v1",
            "files": {"manager.py": hashlib.sha256(adapter.read_bytes()).hexdigest()},
        },
    }
    kwargs = {
        "source_sha": "a" * 40,
        "recipe_sha": "b" * 40,
        "image_profile": "vllm",
        "resolved_inputs_sha256": "c" * 64,
        "overlay_directory": overlay_dir,
    }
    return identity, record, kwargs, adapter


def test_installed_compatibility_overlays_match_build_receipt(image_identity):
    identity, record, kwargs, _adapter = image_identity
    identity.verify_build_identity(record, **kwargs)


@pytest.mark.parametrize(
    "field", ["compatibility_overlays", "vllm_kvcr_public_channel"]
)
def test_missing_compatibility_overlay_proof_fails_before_serving(
    image_identity, field
):
    identity, record, kwargs, _adapter = image_identity
    del record[field]
    with pytest.raises(ValueError, match="overlay"):
        identity.verify_build_identity(record, **kwargs)


def test_changed_installed_vllm_adapter_is_rejected(image_identity):
    identity, record, kwargs, adapter = image_identity
    adapter.write_bytes(b"from kvcr.control_channels import ZmqPeerControlChannel\n")
    with pytest.raises(ValueError, match="adapter"):
        identity.verify_build_identity(record, **kwargs)


def test_wrong_dynamo_patch_input_checksum_is_rejected(image_identity):
    identity, record, kwargs, _adapter = image_identity
    record["dynamo_vllm_fix"]["patch_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="overlay"):
        identity.verify_build_identity(record, **kwargs)


def test_resolved_input_overlay_receipt_is_not_mutable_global_state():
    images = importlib.import_module("ci_images")
    args = (
        "vllm",
        "a" * 40,
        "b" * 40,
        "vllm@sha256:" + "c" * 64,
        "nightly",
        "1.8.0.dev20261123",
    )
    original = images.build_inputs(*args)
    mutated = images.build_inputs(*args)
    mutated["compatibility_overlays"]["dynamo-vllm-18840"]["commit"] = "d" * 40
    assert images.inputs_digest(mutated) != images.inputs_digest(original)
    assert images.build_inputs(*args) == original


def test_common_dynamo_release_must_have_both_wheels_for_every_image(monkeypatch):
    images = importlib.import_module("ci_images")
    monkeypatch.setitem(
        images.IMAGE_PROFILES, "qualified-other", {"python_version": "3.12"}
    )
    available = {
        ("ai-dynamo", "3.14"): ["1.8.0.dev20261125", "1.8.0.dev20261124"],
        ("ai-dynamo-runtime", "3.14"): ["1.8.0.dev20261125", "1.8.0.dev20261124"],
        ("ai-dynamo", "3.12"): ["1.8.0.dev20261124", "1.8.0.dev20261123"],
        ("ai-dynamo-runtime", "3.12"): ["1.8.0.dev20261124"],
    }
    queried = []

    def query(package, python_version):
        queried.append((package, python_version))
        return available[(package, python_version)]

    assert (
        images.resolve_for_profiles(
            "nightly",
            ["vllm", "qualified-other", "vllm"],
            query=query,
        )
        == "1.8.0.dev20261124"
    )
    assert set(queried) == set(available)
    assert len(queried) == 4
    with pytest.raises(ValueError, match="available"):
        images.resolve_for_profiles(
            "1.8.0.dev20261125",
            ["vllm", "qualified-other"],
            query=query,
        )


@pytest.mark.parametrize("profiles", [[], ["not-qualified"]])
def test_common_resolver_requires_qualified_image_profiles(profiles):
    images = importlib.import_module("ci_images")
    with pytest.raises(ValueError, match="profile"):
        images.resolve_for_profiles("nightly", profiles, query=lambda *_: [])
