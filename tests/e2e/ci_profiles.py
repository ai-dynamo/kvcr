# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared configuration, immutable checkpoint inventory and integration plans.

Adapters own supported tuples, launch arguments and applicable cases. The common
manifest records the selected registered policy, never an unqualified target row.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path

from ci_integrations import get_integration

CATALOG = Path(__file__).with_suffix(".json")
MODELS = Path(__file__).with_name("ci_models.json")
# Compatibility import for the existing vLLM entry point; common selection uses
# adapter.smoke_profiles instead of imposing this smoke on future integrations.
SMOKE_ID = "vllm-dense-peer-transfer-smoke"


def fingerprint(profile: dict) -> str:
    """Bind effective settings and adapter policy, excluding qualification history."""
    adapter = get_integration(profile["integration"])
    value = {key: value for key, value in profile.items() if key != "qualification"}
    value["adapter_sha256"] = hashlib.sha256(
        Path(adapter.__file__).read_bytes()
    ).hexdigest()
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_models(path: Path = MODELS) -> dict:
    raw = json.loads(path.read_text())
    if set(raw) != {"schema_version", "models"} or raw["schema_version"] != 1:
        raise ValueError("unsupported checkpoint inventory schema")
    if not isinstance(raw["models"], dict) or not raw["models"]:
        raise ValueError("checkpoint inventory must not be empty")
    for name, model in raw["models"].items():
        if (
            not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name)
            or not isinstance(model, dict)
            or set(model) != {"id", "revision"}
            or not isinstance(model["id"], str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model["id"])
            or not isinstance(model["revision"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", model["revision"])
        ):
            raise ValueError(
                "checkpoints require repository IDs and immutable 40-hex revisions"
            )
    return raw["models"]


def validate_catalog(raw: dict, *, models_path: Path = MODELS) -> dict:
    if (
        not isinstance(raw, dict)
        or set(raw) != {"schema_version", "integration", "models", "profiles"}
        or raw["schema_version"] != 2
    ):
        raise ValueError("unsupported catalog schema")
    adapter = get_integration(raw["integration"])
    if not isinstance(raw["models"], dict) or not isinstance(raw["profiles"], list):
        raise ValueError("catalog models/profiles are malformed")
    pins = load_models(models_path)
    resolved = copy.deepcopy(raw)
    for model in resolved["models"].values():
        if not isinstance(model, dict) or set(model) not in (
            {"model_key", "launch"},
            {"model_key", "launch", "id", "revision"},
        ):
            raise ValueError(
                "models require a checkpoint key and adapter launch settings"
            )
        key = model["model_key"]
        if not isinstance(key, str) or key not in pins:
            raise ValueError(f"unknown checkpoint: {key!r}")
        if any(
            field in model and model[field] != pins[key][field]
            for field in ("id", "revision")
        ):
            raise ValueError("resolved model differs from the authoritative checkpoint")
        if not isinstance(model["launch"], dict):
            raise ValueError("model launch settings must be an object")
        model.update(pins[key])
    for profile in resolved["profiles"]:
        if (
            not isinstance(profile, dict)
            or profile.get("integration") != raw["integration"]
        ):
            raise ValueError("profile integration differs from its registered catalog")
        if profile.get("model_family") not in resolved["models"]:
            raise ValueError("profile refers to an unknown model configuration")
    return adapter.validate_catalog(resolved)


def load_catalog(path: Path | None = None, integration: str = "vllm") -> dict:
    adapter = get_integration(integration)
    source = path if path is not None else adapter.CATALOG
    catalog = validate_catalog(json.loads(source.read_text()))
    if catalog["integration"] != integration:
        raise ValueError("requested integration differs from the catalog")
    return catalog


def get_profile(catalog: dict, profile_id: str) -> dict:
    adapter = get_integration(catalog["integration"])
    for profile in [
        *adapter.smoke_profiles(catalog),
        *getattr(adapter, "diagnostic_profiles", lambda _: [])(catalog),
    ]:
        if profile["id"] == profile_id:
            return profile
    for item in catalog["profiles"]:
        if item["id"] == profile_id:
            return adapter.resolve_profile(
                item, catalog["models"][item["model_family"]]
            )
    raise ValueError(f"unknown profile: {profile_id}")


def select_profiles(catalog: dict, selection: str = "all") -> list[dict]:
    if selection == "all":
        names = [profile["id"] for profile in catalog["profiles"]]
    else:
        names = [name.strip() for name in selection.split(",")]
        smoke_ids = {
            item["id"]
            for item in get_integration(catalog["integration"]).smoke_profiles(catalog)
        }
        if (
            not all(names)
            or len(set(names)) != len(names)
            or "all" in names
            or smoke_ids.intersection(names)
            or not set(names).issubset({p["id"] for p in catalog["profiles"]})
        ):
            raise ValueError("selection must be all or distinct functional profile IDs")
    return [get_profile(catalog, name) for name in names]


def manifest_profile(profile: dict) -> dict:
    keys = (
        "id",
        "integration",
        "model_family",
        "model_key",
        "model_id",
        "model_revision",
        "topology",
        "mode",
        "cases",
        "qualification",
    )
    return {
        **{key: profile[key] for key in keys},
        "fingerprint": fingerprint(profile),
        "gpus": profile["resources"]["gpus"],
    }


def make_manifest(
    catalog: dict,
    selection: str,
    image: str,
    source_sha: str,
    run_id: str,
    run_attempt: str,
    *,
    recipe_sha: str | None = None,
    image_profile: str | None = None,
    resolved_inputs_sha256: str = "",
) -> dict:
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", image):
        raise ValueError("manifest requires an immutable image digest")
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("manifest requires the tested source SHA")
    recipe_sha = recipe_sha or os.environ.get("GITHUB_SHA", source_sha)
    if not re.fullmatch(r"[0-9a-f]{40}", recipe_sha):
        raise ValueError("manifest requires the recipe SHA")
    if resolved_inputs_sha256 and not re.fullmatch(
        r"[0-9a-f]{64}", resolved_inputs_sha256
    ):
        raise ValueError("invalid resolved input fingerprint")
    if not str(run_id).strip() or not str(run_attempt).strip():
        raise ValueError("manifest requires nonempty run identity")
    integration = catalog["integration"]
    image_profile = image_profile or integration
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", image_profile):
        raise ValueError("invalid image profile")
    profiles = select_profiles(catalog, selection)
    smokes = [
        manifest_profile(profile)
        for profile in get_integration(integration).smoke_profiles(catalog)
    ]
    manifest = {
        "schema_version": 2,
        "integration": integration,
        "selection": "all" if selection == "all" else "diagnostic",
        "run_id": str(run_id),
        "run_attempt": str(run_attempt),
        "source_sha": source_sha,
        "recipe_sha": recipe_sha,
        "workflow_sha": os.environ.get("GITHUB_SHA", recipe_sha),
        "image": image,
        "image_profile": image_profile,
        "resolved_inputs_sha256": resolved_inputs_sha256,
        "profiles": [manifest_profile(profile) for profile in profiles],
        "smokes": smokes,
    }
    # Old tooling reads this alias; the production v2 contract uses smokes[].
    if len(smokes) == 1:
        manifest["smoke"] = smokes[0]
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration", default="vllm")
    parser.add_argument("--selection", default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="")
    parser.add_argument("--image-profile", default=None)
    parser.add_argument("--resolved-inputs-sha256", default="")
    parser.add_argument("--recipe-sha", default=None)
    parser.add_argument("--source-sha", default=os.environ.get("GITHUB_SHA", ""))
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    parser.add_argument(
        "--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    )
    args = parser.parse_args()
    try:
        manifest = make_manifest(
            load_catalog(integration=args.integration),
            args.selection,
            args.image,
            args.source_sha,
            args.run_id,
            args.run_attempt,
            recipe_sha=args.recipe_sha,
            image_profile=args.image_profile,
            resolved_inputs_sha256=args.resolved_inputs_sha256,
        )
    except (ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "include": [
                    {"profile": profile["id"]} for profile in manifest["profiles"]
                ]
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
