#!/bin/bash
# Build and push the Phase 1 image for the current HEAD from a dlcluster node.
#   scripts/dlcluster/build_image.sh [--repo ghcr.io/luckyq/kvcr-phase1]
# Ships the ghcr token from the local `gh auth token` (account with write:packages).
set -euo pipefail
HOST=dlcluster; REPO=ghcr.io/luckyq/kvcr-phase1; PINNED=1
while [[ $# -gt 0 ]]; do case $1 in --repo) REPO=$2; shift 2 ;; --ci-refs) PINNED=0; shift ;; *) echo "unknown arg $1" >&2; exit 2 ;; esac; done
repo=$(git rev-parse --show-toplevel); cd "$repo"
git diff --quiet HEAD -- . || { echo "commit your changes first (image build needs clean source)" >&2; exit 1; }
ref=$(git rev-parse HEAD)
bundle="kvcr-phase1-${ref:0:12}.bundle"
git bundle create "/tmp/$bundle" HEAD >/dev/null
gh auth token | ssh "$HOST" 'umask 077; cat > ~/.ghcr-token'
scp -q "/tmp/$bundle" scripts/dlcluster/build.sbatch "$HOST:~/"
ssh "$HOST" "mkdir -p ~/kvcr-runs && sbatch --export=ALL,KVCR_BUNDLE=\$HOME/$bundle,KVCR_REF=$ref,KVCR_IMAGE_REPO=$REPO,KVCR_PINNED_SOURCES=$PINNED ~/build.sbatch"
echo "image will be $REPO:$ref"
