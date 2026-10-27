#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Resolve public inputs once, then build/publish the exact checked-out KVCR SHA.
# KVCR_PUSH=0 (default) loads locally; KVCR_PUSH=1 publishes both image tags.
# No build credentials or BuildKit secrets are forwarded into the Dockerfile.
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

resolve_only=${KVCR_RESOLVE_ONLY:-0}
case ${1:-} in
    --resolve-only) resolve_only=1; shift ;;
    --help|-h)
        printf '%s\n' \
            'Usage: build_image.sh [--resolve-only]' \
            'Required for builds: KVCR_E2E_IMAGE=registry/repository' \
            'Optional: KVCR_PUSH=0|1, KVCR_BUILDER, E2E_BUILD_OUTPUT=artifacts/build' \
            'Optional: KVCR_GITHUB_IP=<numeric IPv4 address> (build-only DNS override)'
        exit 0 ;;
esac
[[ $# -eq 0 ]] || die "Unexpected arguments"
[[ "$resolve_only" == 0 || "$resolve_only" == 1 ]] || die "KVCR_RESOLVE_ONLY must be 0 or 1"
publish=${KVCR_PUSH:-0}
[[ "$publish" == 0 || "$publish" == 1 ]] || die "KVCR_PUSH must be 0 or 1"
command -v git >/dev/null || die "git is required"
command -v python3 >/dev/null || die "python3 is required"

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
[[ -f Dockerfile.quick-start ]] || die "Run from the KVCR checkout"
kvcr_sha=$(git rev-parse HEAD)
[[ "$kvcr_sha" =~ ^[0-9a-f]{40}$ ]] || die "KVCR HEAD is not a full commit SHA"
if [[ -n ${GITHUB_SHA:-} && "$kvcr_sha" != "$GITHUB_SHA" ]]; then
    die "Checked-out KVCR HEAD does not match GITHUB_SHA"
fi

# Resolve-only intentionally permits a dirty checkout for reviewing a proposed
# workflow. Actual builds must describe committed source, including Dockerfile.
source_is_clean() {
    git diff --quiet HEAD -- . || return 1
    # Git ignores untracked files in `diff`, but Docker COPY does not. Restrict
    # this check to copied inputs so generated artifacts/build remains allowed.
    # Do not apply .gitignore here: it does not control Docker build context.
    [[ -z "$(git ls-files --others -- src pyproject.toml README.md LICENSE Dockerfile.quick-start)" ]]
}
worktree_clean=true
source_is_clean || worktree_clean=false
if [[ "$resolve_only" == 0 && "$worktree_clean" != true ]]; then
    die "Commit tracked changes and untracked copied source before building a SHA-identified image"
fi

output_dir=${E2E_BUILD_OUTPUT:-artifacts/build}
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd -P)
manifest="$output_dir/source-revisions.json"
metadata_file="$output_dir/build-metadata.json"
identity_file="$output_dir/image-identity.json"
build_log="$output_dir/build.log"

dynamo_repo=https://github.com/ai-dynamo/dynamo.git
dynamo_ref=refs/heads/main
vllm_repo=https://github.com/vllm-project/vllm.git
vllm_ref=refs/pull/53624/head

resolve_ref() {
    python3 - "$1" "$2" <<'PY'
import os
import re
import subprocess
import sys

repository, ref = sys.argv[1:]
environment = dict(os.environ, GIT_TERMINAL_PROMPT="0")
try:
    result = subprocess.run(
        ["git", "-c", "credential.helper=", "-c", "http.extraHeader=",
         "ls-remote", "--exit-code", repository, ref],
        capture_output=True, text=True, env=environment, timeout=60, check=True,
    )
except (subprocess.SubprocessError, OSError) as error:
    raise SystemExit(f"Could not resolve public source {repository} {ref}: {type(error).__name__}")
rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
if len(rows) != 1 or len(rows[0]) != 2 or rows[0][1] != ref or not re.fullmatch(r"[0-9a-f]{40}", rows[0][0]):
    raise SystemExit(f"Expected exactly one full SHA for {repository} {ref}")
print(rows[0][0])
PY
}

# KVCR_PINNED_SOURCES=1 builds the compatibility set the Dockerfile pins
# (release verification) instead of the moving CI refs above.
if [[ "${KVCR_PINNED_SOURCES:-0}" == 1 ]]; then
    dynamo_ref=pinned:Dockerfile.quick-start
    vllm_ref=pinned:Dockerfile.quick-start
    dynamo_sha=$(sed -nE 's/^ARG DYNAMO_REF=([0-9a-f]{40})\s*$/\1/p' Dockerfile.quick-start | head -1)
    vllm_sha=$(sed -nE 's/^ARG KVCR_VLLM_REF=([0-9a-f]{40})\s*$/\1/p' Dockerfile.quick-start | head -1)
    [[ "$dynamo_sha" =~ ^[0-9a-f]{40}$ && "$vllm_sha" =~ ^[0-9a-f]{40}$ ]] \
        || die "Dockerfile.quick-start must pin DYNAMO_REF and KVCR_VLLM_REF to full SHAs"
else
    dynamo_sha=$(resolve_ref "$dynamo_repo" "$dynamo_ref")
    vllm_sha=$(resolve_ref "$vllm_repo" "$vllm_ref")
fi
base_image=$(python3 - <<'PY'
from pathlib import Path
import re

matches = re.findall(r"^ARG VLLM_IMAGE=(\S+)\s*$", Path("Dockerfile.quick-start").read_text(), re.M)
if len(matches) != 1 or not re.fullmatch(r"[a-zA-Z0-9._/:\-]+@sha256:[0-9a-f]{64}", matches[0]):
    raise SystemExit("Dockerfile.quick-start must have exactly one digest-pinned VLLM_IMAGE default")
print(matches[0])
PY
)

python3 - "$manifest" "$kvcr_sha" "$dynamo_repo" "$dynamo_ref" "$dynamo_sha" \
    "$vllm_repo" "$vllm_ref" "$vllm_sha" "$base_image" "$worktree_clean" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

(destination, kvcr_sha, dynamo_repo, dynamo_ref, dynamo_sha,
 vllm_repo, vllm_ref, vllm_sha, base_image, clean) = sys.argv[1:]
manifest = {
    "schema_version": 1, "resolved_at": datetime.now(timezone.utc).isoformat(),
    "kvcr_commit": kvcr_sha, "kvcr_worktree_clean": clean == "true",
    "dynamo_repository": dynamo_repo, "dynamo_ref": dynamo_ref, "dynamo_commit": dynamo_sha,
    "vllm_adapter_repository": vllm_repo, "vllm_adapter_ref": vllm_ref,
    "vllm_adapter_commit": vllm_sha, "vllm_base_image": base_image,
    "platform": "linux/amd64", "runtime_user": "root",
    "dockerfile_sha256": hashlib.sha256(Path("Dockerfile.quick-start").read_bytes()).hexdigest(),
    "github_repository": os.environ.get("GITHUB_REPOSITORY"),
    "github_event_name": os.environ.get("GITHUB_EVENT_NAME"),
    "github_ref": os.environ.get("GITHUB_REF"),
    "github_sha": os.environ.get("GITHUB_SHA"),
    "github_run_id": os.environ.get("GITHUB_RUN_ID"),
    "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
}
path = Path(destination)
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(manifest, indent=2) + "\n")
temporary.replace(path)
print(json.dumps(manifest, indent=2))
PY

if [[ "$resolve_only" == 1 ]]; then
    printf 'Resolved sources only; manifest: %s\n' "$manifest"
    exit 0
fi

command -v docker >/dev/null || die "docker is required for image builds"
image=${KVCR_E2E_IMAGE:-}
[[ -n "$image" ]] || die "KVCR_E2E_IMAGE must name the destination registry/repository"
python3 - "$image" "${KVCR_GITHUB_IP:-}" <<'PY'
import ipaddress
import re
import sys

image, address = sys.argv[1:]
if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*(?::[0-9]+)?(?:/[a-z0-9][a-z0-9._-]*)+", image):
    raise SystemExit("KVCR_E2E_IMAGE must be a lowercase registry/repository without a tag, digest, or credentials")
if address:
    try:
        ipaddress.IPv4Address(address)
    except ipaddress.AddressValueError:
        raise SystemExit("KVCR_GITHUB_IP must be a numeric IPv4 address")
PY

if [[ -n ${GITHUB_RUN_ID:-} ]]; then
    [[ "$GITHUB_RUN_ID" =~ ^[0-9]+$ ]] || die "GITHUB_RUN_ID must be numeric"
    run_attempt=${GITHUB_RUN_ATTEMPT:-1}
    [[ "$run_attempt" =~ ^[0-9]+$ ]] || die "GITHUB_RUN_ATTEMPT must be numeric"
    unique_tag="${kvcr_sha}-run-${GITHUB_RUN_ID}-${run_attempt}"
else
    unique_tag="${kvcr_sha}-local-$(date -u +%Y%m%dT%H%M%S)-$$"
fi
sha_uri="${image}:${kvcr_sha}"
unique_uri="${image}:${unique_tag}"

build_command=(docker buildx build --platform linux/amd64
    --ulimit nofile=65536:65536
    --file Dockerfile.quick-start --progress plain --metadata-file "$metadata_file"
    --tag "$sha_uri" --tag "$unique_uri"
    --build-arg "VLLM_IMAGE=$base_image"
    --build-arg "DYNAMO_REF=$dynamo_sha"
    --build-arg "KVCR_VLLM_REPO=$vllm_repo"
    --build-arg "KVCR_VLLM_REF=$vllm_sha"
    --build-arg "KVCR_COMMIT=$kvcr_sha"
    --build-arg KVCR_RUNTIME_USER=root)
[[ -z ${KVCR_BUILDER:-} ]] || build_command+=(--builder "$KVCR_BUILDER")
[[ -z ${KVCR_GITHUB_IP:-} ]] || build_command+=(--add-host "github.com:${KVCR_GITHUB_IP}")
if [[ "$publish" == 1 ]]; then
    build_command+=(--push)
else
    build_command+=(--load)
fi
build_command+=(.)
rm -f "$metadata_file" "$identity_file"
printf 'Building KVCR %s as %s (publish=%s)\n' "$kvcr_sha" "$unique_uri" "$publish"
if ! "${build_command[@]}" 2>&1 | tee "$build_log"; then
    die "Image build failed; see $build_log"
fi
[[ "$(git rev-parse HEAD)" == "$kvcr_sha" ]] || die "KVCR HEAD changed during the build"
source_is_clean || die "Tracked or copied source changed during the build"

# Inspect the published digest (or local image), rather than trusting tag names.
# Capture image JSON privately and retain only provenance labels and identity.
python3 - "$manifest" "$metadata_file" "$identity_file" "$image" "$unique_uri" \
    "$sha_uri" "$publish" "${GITHUB_OUTPUT:-}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

(manifest_name, metadata_name, identity_name, repository, unique_uri,
 sha_uri, publish, github_output) = sys.argv[1:]
manifest = json.loads(Path(manifest_name).read_text())
metadata = json.loads(Path(metadata_name).read_text())
if hashlib.sha256(Path("Dockerfile.quick-start").read_bytes()).hexdigest() != manifest["dockerfile_sha256"]:
    raise SystemExit("Dockerfile changed after source resolution")
metadata_digests = {}
for key in ("containerimage.digest", "containerimage.config.digest"):
    if key in metadata:
        value = metadata[key]
        if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
            raise SystemExit(f"Build metadata contains an invalid image digest: {key}")
        metadata_digests[key] = value
digest = metadata_digests.get("containerimage.digest")

def invoke(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=120, check=True)
        return json.loads(result.stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        raise SystemExit(f"Image identity inspection failed: {type(error).__name__}")

if publish == "1":
    if not digest:
        raise SystemExit("Published build metadata omitted containerimage.digest")
    image_uri = f"{repository}@{digest}"
    data = invoke(["docker", "buildx", "imagetools", "inspect", image_uri, "--format", "{{json .Image}}"])
    candidates = [data] if isinstance(data, dict) and data.get("architecture") else list(data.values()) if isinstance(data, dict) else []
else:
    image_uri = unique_uri
    data = invoke(["docker", "image", "inspect", unique_uri])
    candidates = data if isinstance(data, list) else []

candidates = [item for item in candidates if isinstance(item, dict) and
              item.get("architecture", item.get("Architecture")) == "amd64" and
              item.get("os", item.get("Os")) == "linux"]
if len(candidates) != 1:
    raise SystemExit("Expected exactly one inspected linux/amd64 image configuration")
config = candidates[0]
labels = config.get("config", config.get("Config", {})).get("Labels", {})
expected_labels = {
    "org.opencontainers.image.revision": manifest["kvcr_commit"],
    "io.ai-dynamo.kvcr.dynamo.sha": manifest["dynamo_commit"],
    "io.ai-dynamo.kvcr.vllm.sha": manifest["vllm_adapter_commit"],
    "io.ai-dynamo.kvcr.base-image": manifest["vllm_base_image"],
}
for key, value in expected_labels.items():
    if labels.get(key) != value:
        raise SystemExit(f"Image provenance label does not match resolved sources: {key}")
if publish != "1":
    local_id = config.get("Id")
    if not isinstance(local_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", local_id):
        raise SystemExit("Loaded image has an invalid local image ID")
    # Classic Docker identifies images by configuration digest; the containerd
    # image store can use the manifest digest and omit the configuration field.
    if local_id not in metadata_digests.values():
        raise SystemExit("Loaded image ID does not match any recorded build digest")
    digest = digest or local_id

identity = {
    "schema_version": 1, "verified_at": datetime.now(timezone.utc).isoformat(),
    "published": publish == "1", "image_uri": image_uri, "digest": digest,
    "sha_tag": sha_uri, "unique_tag": unique_uri,
    "platform": "linux/amd64", "verified_labels": expected_labels,
    "verification": "Inspected image identity and provenance labels match the resolved source manifest",
}
if publish != "1":
    identity["local_image_id"] = local_id
Path(identity_name).write_text(json.dumps(identity, indent=2) + "\n")
if github_output:
    values = {
        "image_uri": image_uri, "digest": digest, "kvcr_sha": manifest["kvcr_commit"],
        "dynamo_sha": manifest["dynamo_commit"], "vllm_sha": manifest["vllm_adapter_commit"],
    }
    with open(github_output, "a") as output:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise SystemExit("Unexpected newline in workflow output")
            output.write(f"{key}={value}\n")
print(json.dumps(identity, indent=2))
PY
