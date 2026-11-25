# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SGLang HiCache with the KVCR storage backend (sgl-project/sglang#45647).

GPU -> HiCache host staging -> KVCR DRAM in the worker, with remote hints for
peer transfer. Shared SGLang launch/evidence semantics live in ``sglang``.
"""

from __future__ import annotations

from pathlib import Path

from . import sglang

INTEGRATION = "sglang-hicache"
IMAGE_PROFILE = "sglang-hicache"
CATALOG = sglang.E2E / "ci_profiles_sglang_hicache.json"
# The actual worker launch must use HiCache with the KVCR storage backend.
KVCR_FLAGS: dict[str, str | bool] = {
    "--enable-hierarchical-cache": True,
    "--hicache-storage-backend": "kvcr",
    "--hicache-write-policy": "write_through",
}

environment = sglang.environment
resolve_profile = sglang.resolve_profile
runtime_resources = sglang.runtime_resources
validate_model_snapshot = sglang.validate_model_snapshot


def validate_catalog(raw: dict) -> dict:
    return sglang.validate_catalog(raw, INTEGRATION)


def required_profile_ids(catalog: dict | None = None) -> set[str]:
    return sglang.required_profile_ids(INTEGRATION)


def smoke_profiles(catalog: dict) -> list[dict]:
    return []


def build_command(
    profile: dict,
    output: Path,
    snapshot: Path,
    prompt_tag: str,
    gpus: list[int] | None = None,
) -> list[str]:
    return sglang.build_command(profile, output, snapshot, prompt_tag, gpus)


def validate_result(
    profile: dict, envelope: dict, directory: Path, errors: list[str]
) -> None:
    sglang.validate_result(profile, envelope, directory, errors, kvcr_flags=KVCR_FLAGS)
