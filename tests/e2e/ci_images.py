# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve a build's nightly policy once, before any Docker layer is cached.

Add qualified engine recipes here; callers cannot inject a recipe or shell code.
Installed package inventories and pip reports supplement this input identity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from ci_identity import VLLM_OVERLAYS

try:
    from packaging.version import Version
except ImportError:  # Thin builder clients already have pip.
    from pip._vendor.packaging.version import Version

IMAGE_PROFILES = {
    "vllm": {
        "recipe": "Dockerfile.ci",
        "flavor": "vllm",
        "python_version": "3.14",
        "base_argument": "VLLM_IMAGE",
        "compatibility_overlays": VLLM_OVERLAYS,
        "canary_scripts": ["tests/e2e/check_vllm_rank_initialization.py"],
    },
    # The SGLang fork commit, its pinned stack and the import canary live in
    # the recipe, selected by CI_IMAGE_PROFILE; the base image is the digest
    # pinned there (SGLANG_IMAGE).
    "sglang-hicache": {
        "recipe": "Dockerfile.quick-start-sglang",
        "flavor": "vllm",
        "python_version": "3.12",
        "base_argument": "SGLANG_IMAGE",
    },
    "sglang-linker": {
        "recipe": "Dockerfile.quick-start-sglang",
        "flavor": "vllm",
        "python_version": "3.12",
        "base_argument": "SGLANG_IMAGE",
    },
}
PACKAGES = ("ai-dynamo", "ai-dynamo-runtime")


def resolve_dynamo(policy: str, available: dict[str, list[str]]) -> str:
    common = set.intersection(*(set(available[name]) for name in PACKAGES))
    if policy == "nightly":
        versions = [Version(item) for item in common if Version(item).is_devrelease]
        if not versions:
            raise ValueError("no common compatible Dynamo nightly wheels")
        return str(max(versions))
    if policy not in common:
        raise ValueError("requested Dynamo version is not available for both packages")
    return policy


def inputs_digest(record: dict) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_inputs(
    image_profile: str,
    source_sha: str,
    recipe_sha: str,
    base_image: str,
    policy: str,
    version: str,
) -> dict:
    if image_profile not in IMAGE_PROFILES:
        raise ValueError(f"unqualified image profile: {image_profile}")
    if not all(
        re.fullmatch(r"[a-f0-9]{40}", value) for value in (source_sha, recipe_sha)
    ):
        raise ValueError("source and recipe require full commit SHAs")
    if not re.fullmatch(r".+@sha256:[a-f0-9]{64}", base_image):
        raise ValueError("base image requires an immutable digest")
    resolved_version = Version(version)
    if policy != "nightly" and Version(policy) != resolved_version:
        raise ValueError("resolved Dynamo version differs from explicit policy")
    if policy == "nightly" and not resolved_version.is_devrelease:
        raise ValueError("nightly policy requires a Dynamo dev release")
    return {
        "schema_version": 2,
        "image_profile": image_profile,
        "source_sha": source_sha,
        "recipe_sha": recipe_sha,
        "base_image": base_image,
        "dynamo_policy": policy,
        "dynamo_version": version,
        **copy.deepcopy(IMAGE_PROFILES[image_profile]),
    }


def available_versions(package: str, python_version: str) -> list[str]:
    output = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "pip",
            "index",
            "versions",
            package,
            "--pre",
            "--only-binary=:all:",
            "--python-version",
            python_version,
            "--extra-index-url",
            "https://pypi.nvidia.com/",
        ],
        text=True,
        timeout=180,
    )
    found = re.search(r"^Available versions: (.+)$", output, re.MULTILINE)
    if found is None:
        raise ValueError(f"cannot list compatible versions of {package}")
    return found.group(1).split(", ")


def resolve_for_profiles(
    policy: str, image_profiles: list[str], query=available_versions
) -> str:
    """Choose one release usable by every registered image's Python runtime.

    A future qualified image joins the same run-wide resolution by registering
    its runtime in IMAGE_PROFILES and passing its name in image_profiles. No
    separate per-engine nightly lookup or implicit fallback is permitted.
    """
    if not image_profiles or any(name not in IMAGE_PROFILES for name in image_profiles):
        raise ValueError("Dynamo resolution requires qualified image profiles")
    runtimes = {IMAGE_PROFILES[name]["python_version"] for name in image_profiles}
    available = {}
    for package in PACKAGES:
        compatible = [set(query(package, runtime)) for runtime in sorted(runtimes)]
        available[package] = list(set.intersection(*compatible))
    return resolve_dynamo(policy, available)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-profile", required=True, choices=IMAGE_PROFILES)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--recipe-sha", required=True)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--dynamo-policy", default="nightly")
    parser.add_argument(
        "--dynamo-profiles",
        default="",
        help="comma-separated image profiles whose Python runtimes the "
        "resolved Dynamo release must serve (default: --image-profile)",
    )
    parser.add_argument(
        "--dynamo-version",
        default="",
        help="reuse a version already resolved for this run's image profiles",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    profiles = [
        name.strip() for name in args.dynamo_profiles.split(",") if name.strip()
    ] or [args.image_profile]
    if args.image_profile not in profiles:
        parser.error("--dynamo-profiles must include --image-profile")
    version = args.dynamo_version or resolve_for_profiles(
        args.dynamo_policy, profiles
    )
    record = build_inputs(
        args.image_profile,
        args.source_sha,
        args.recipe_sha,
        args.base_image,
        args.dynamo_policy,
        version,
    )
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({**record, "resolved_inputs_sha256": inputs_digest(record)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
