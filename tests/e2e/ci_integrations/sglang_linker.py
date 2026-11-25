# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SGLang KVCR direct linker (sgl-project/sglang#45650).

GPU KV pages move to and from KVCR DRAM in the worker layer by layer (TP1
only). It is registered for diagnostic coverage but not in ``ci_lanes.json``:
the linker is not a qualified required lane yet. Shared SGLang launch and
evidence semantics live in ``sglang``.
"""

from __future__ import annotations

from pathlib import Path

from . import sglang

INTEGRATION = "sglang-linker"
IMAGE_PROFILE = "sglang-linker"
CATALOG = sglang.E2E / "ci_profiles_sglang_linker.json"
SGLANG_MODE = "linker"
# The actual worker launch must use the KVCR direct linker, not HiCache.
KVCR_FLAGS: dict[str, str | bool] = {
    "--kv-cache-dtype": "bfloat16",
    "--enable-unified-cache-external-linker": True,
    "--unified-cache-external-linker-backend": "kvcr",
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
    return sglang.build_command(
        profile,
        output,
        snapshot,
        prompt_tag,
        gpus,
        extra=["--sglang-mode", SGLANG_MODE],
    )


def validate_result(
    profile: dict, envelope: dict, directory: Path, errors: list[str]
) -> None:
    sglang.validate_result(
        profile,
        envelope,
        directory,
        errors,
        kvcr_flags=KVCR_FLAGS,
        native_fields={"sglang_mode": SGLANG_MODE},
        arguments={"sglang_mode": SGLANG_MODE},
    )
