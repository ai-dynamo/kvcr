# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a tested image against its independent workflow build manifest."""

from __future__ import annotations

import hashlib
import re
from importlib import metadata
from pathlib import Path

OVERLAY_COMMIT = "66ba12c69706a8775f48684e0a9ac4682d41ca8b"
OVERLAY_PATCH_SHA256 = (
    "2f91a13244c0b7770d39577850d2b0e938084cb3a25eb1245a874104cb3889dc"
)
OVERLAY_FILES = {"benchmark_worker.py", "instrumented_scheduler.py", "kv_groups.py"}
ADAPTER_OVERLAY_REVISION = "public-channel-v1"
VLLM_OVERLAYS = {
    "dynamo-vllm-18840": {
        "commit": OVERLAY_COMMIT,
        "patch_sha256": OVERLAY_PATCH_SHA256,
    },
    "vllm-kvcr-public-channel": {"revision": ADAPTER_OVERLAY_REVISION},
}


def verify_build_identity(
    record: dict,
    *,
    source_sha: str,
    recipe_sha: str,
    image_profile: str,
    resolved_inputs_sha256: str,
    overlay_directory: Path | None = None,
) -> None:
    expected = {
        "schema_version": 2,
        "kvcr_sha": source_sha,
        "recipe_sha": recipe_sha,
        "image_profile": image_profile,
        "resolved_inputs_sha256": resolved_inputs_sha256,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"image build identity differs: {key}")
    if not re.fullmatch(r"[a-f0-9]{64}", resolved_inputs_sha256):
        raise ValueError("missing resolved input digest")
    if image_profile == "vllm":
        if record.get("compatibility_overlays") != VLLM_OVERLAYS:
            raise ValueError("required compatibility overlay declarations differ")
        overlay = record.get("dynamo_vllm_fix", {})
        if (
            overlay.get("commit") != OVERLAY_COMMIT
            or overlay.get("patch_sha256") != OVERLAY_PATCH_SHA256
            or set(overlay.get("files", {})) != OVERLAY_FILES
        ):
            raise ValueError("required Dynamo compatibility overlay identity differs")
        if overlay_directory is None:
            overlay_directory = Path(
                metadata.distribution("ai-dynamo").locate_file("dynamo/vllm")
            )
        for name, expected_hash in overlay["files"].items():
            if (
                hashlib.sha256((overlay_directory / name).read_bytes()).hexdigest()
                != expected_hash
            ):
                raise ValueError(f"installed Dynamo overlay file differs: {name}")
        adapter = record.get("vllm_kvcr_public_channel", {})
        if adapter.get("revision") != ADAPTER_OVERLAY_REVISION or set(
            adapter.get("files", {})
        ) != {"manager.py"}:
            raise ValueError(
                "required vLLM adapter compatibility overlay identity differs"
            )
        adapter_path = Path(
            metadata.distribution("vllm").locate_file(
                "vllm/v1/kv_offload/tiering/kvcr/manager.py"
            )
        )
        if (
            hashlib.sha256(adapter_path.read_bytes()).hexdigest()
            != adapter["files"]["manager.py"]
        ):
            raise ValueError("installed vLLM adapter overlay file differs: manager.py")
