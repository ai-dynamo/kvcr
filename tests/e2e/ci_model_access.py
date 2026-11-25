# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, revision-pinned model preparation using the standard Hub cache.

The CI configuration uses a job-container cache on ephemeral scratch. This helper
neither assumes a persistent shared cache nor installs credentials. Hub's own
blob locks and atomic publication remain responsible for concurrent downloads.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from ci_integrations import get_integration

PREFETCH_TIMEOUT_SECONDS = 1200


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"unreadable or malformed {label}: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path.name}")
    return value


def _readable_size(path: Path, label: str) -> int:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            readable = stream.read(1)
    except OSError as error:
        raise ValueError(f"missing or unreadable {label}: {path.name}") from error
    if not size or not readable:
        raise ValueError(f"empty {label}: {path.name}")
    return size


def validate_snapshot(profile: dict, path: Path) -> dict:
    """Validate the resolved pin and required weight files, without rehashing GBs.

    Hub snapshots normally symlink weight files into its blob store. Readability
    therefore follows those symlinks; requiring snapshot-contained real paths
    would incorrectly reject the standard Hub layout.
    """
    path = path.resolve()
    if path.name != profile["model_revision"]:
        raise ValueError("snapshot must identify the pinned model revision")
    _read_json(path / "config.json", "config")
    required = {"config.json"}
    weights: set[str] = set()
    indexes = sorted(path.glob("*.safetensors.index.json")) + sorted(
        path.glob("*.bin.index.json")
    )
    for index in indexes:
        weight_map = _read_json(index, "weight index").get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"weight index has no shard mapping: {index.name}")
        required.add(index.name)
        for name in weight_map.values():
            if (
                not isinstance(name, str)
                or not name
                or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
                or "\\" in name
                or not name.endswith((".safetensors", ".bin"))
            ):
                raise ValueError("unsafe or malformed weight shard path")
            weights.add(name)
    if not indexes:
        weights.update(file.name for file in path.glob("*.safetensors"))
        weights.update(file.name for file in path.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError("snapshot contains no model weights")
    inventory = [
        {"name": name, "bytes": _readable_size(path / name, "weight shard")}
        for name in sorted(weights)
    ]
    return {
        "weight_files": inventory,
        "required_files": sorted(required | weights),
        "weight_bytes": sum(item["bytes"] for item in inventory),
        "validation": "config-json-and-required-weight-readability",
        "weights_rehashed": False,
    }


def cache_context(env: dict[str, str]) -> dict:
    hub_home = Path(
        env.get(
            "HF_HOME",
            str(
                Path(env.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
                / "huggingface"
            ),
        )
    )
    cache = (
        Path(
            env.get(
                "HF_HUB_CACHE", env.get("HUGGINGFACE_HUB_CACHE", str(hub_home / "hub"))
            )
        )
        .expanduser()
        .absolute()
    )
    existing = cache
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    return {
        "cache_scope": (
            "job-container"
            if env.get("GITHUB_ACTIONS") == "true"
            and cache.is_relative_to(Path("/scratch"))
            else "caller-configured"
        ),
        "cache_scope_evidence": "configured-environment-and-path",
        "cache_path": str(cache),
        "cache_exists_before": cache.is_dir(),
        "cache_parent_writable": os.access(existing, os.W_OK),
        "cache_free_bytes_before": shutil.disk_usage(existing).free,
        "uid": os.getuid(),
        "gid": os.getgid(),
        "endpoint_host": urlsplit(
            env.get("HF_ENDPOINT", "https://huggingface.co")
        ).hostname,
    }


def prepare_model(
    profile: dict,
    output: Path,
    env: dict[str, str],
    run_process,
    override: Path | None = None,
    diagnostic: bool = False,
) -> Path:
    """Prepare one pin once, then give every worker/reference its local path.

    The calling wrapper supplies its existing owned-process supervisor, making
    download timeout/cancellation obey the same containment policy as serving.
    A receipt is preserved for setup failures as well as successful preparation.
    """
    if override is not None and not diagnostic:
        raise ValueError("a supplied model snapshot requires --diagnostic")
    if not re.fullmatch(r"[0-9a-f]{40}", profile["model_revision"]):
        raise ValueError("model preparation requires an immutable revision")
    started = time.monotonic()
    record = output / "snapshot.json"
    receipt = {
        "schema_version": 1,
        "status": "FAIL",
        "requested_repository": profile["model_id"],
        "requested_revision": profile["model_revision"],
        "diagnostic_override": override is not None,
    }
    save_json(output / "model-receipt.json", receipt)
    try:
        receipt.update(cache_context(env))
        if override is None:
            download_env = {
                **env,
                "HF_HUB_ETAG_TIMEOUT": env.get("HF_HUB_ETAG_TIMEOUT", "30"),
                "HF_HUB_DOWNLOAD_TIMEOUT": env.get("HF_HUB_DOWNLOAD_TIMEOUT", "60"),
            }
            outcome = run_process(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--download",
                    "--repository",
                    profile["model_id"],
                    "--revision",
                    profile["model_revision"],
                    "--record",
                    str(record),
                ],
                download_env,
                output / "prefetch.log",
                timeout=PREFETCH_TIMEOUT_SECONDS,
            )
            receipt["download"] = outcome
            if outcome["status"] != "PASS":
                raise RuntimeError(f"pinned model prefetch failed: {outcome}")
            downloaded = _read_json(record, "snapshot record")
            path = Path(downloaded["path"]).resolve()
            expected_cache = "models--" + profile["model_id"].replace("/", "--")
            if (
                path.parent.name != "snapshots"
                or path.parent.parent.name != expected_cache
            ):
                raise ValueError(
                    "snapshot cache repository differs from the requested model"
                )
            receipt["huggingface_hub_version"] = downloaded["huggingface_hub_version"]
        else:
            path = override.resolve()
            save_json(record, {"path": str(path), "diagnostic_override": True})
        inventory = validate_snapshot(profile, path)
        receipt.update(
            get_integration(profile["integration"]).validate_model_snapshot(
                profile, path, inventory
            )
        )
        receipt.update(
            status="PASS",
            resolved_revision=path.name,
            resolved_path=str(path),
            resolved_repository=profile["model_id"],
        )
        return path
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["duration_seconds"] = time.monotonic() - started
        save_json(output / "model-receipt.json", receipt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("download requires an immutable revision")
    import huggingface_hub

    path = huggingface_hub.snapshot_download(
        repo_id=args.repository,
        revision=args.revision,
        max_workers=4,
    )
    save_json(
        args.record,
        {
            "path": path,
            "huggingface_hub_version": huggingface_hub.__version__,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
