#!/bin/bash
#SBATCH --job-name=kvcr-p1-build
#SBATCH --partition=general
#SBATCH --gres=gpu:H100:1
#SBATCH --constraint="x86_64&UB24.04"
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/doxu/kvcr-runs/build-%j.log
# Build the Phase 1 image for one KVCR ref and push it to ghcr (KVCR_PUSH=1).
# A 1-GPU allocation only buys a fast start on a Docker-capable UB24.04 node;
# the build itself is CPU-only. Needs: KVCR_BUNDLE, KVCR_REF, KVCR_IMAGE_REPO,
# ~/.ghcr-token (user luckyq, write:packages).
set -euo pipefail
W=/tmp/doxu-kvcr-build-$SLURM_JOB_ID
RUNS=/home/doxu/kvcr-runs/$SLURM_JOB_ID
mkdir -p "$W" "$RUNS"
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
trap 'rm -rf "$W"' EXIT
log "node=$(hostname) kernel=$(uname -r) job=$SLURM_JOB_ID ref=$KVCR_REF repo=$KVCR_IMAGE_REPO"
command -v docker >/dev/null || { log "FATAL: no docker on $(hostname)"; exit 3; }
docker info --format 'docker {{.ServerVersion}}'
git clone -q "$KVCR_BUNDLE" "$W/repo"
git -C "$W/repo" checkout -q --detach "$KVCR_REF"
test "$(git -C "$W/repo" rev-parse HEAD)" = "$KVCR_REF"
cd "$W/repo"
docker login ghcr.io -u luckyq --password-stdin < "$HOME/.ghcr-token" >/dev/null
log "logged in to ghcr.io"
export KVCR_E2E_IMAGE=$KVCR_IMAGE_REPO KVCR_PUSH=1 E2E_BUILD_OUTPUT="$RUNS/build" KVCR_PINNED_SOURCES=${KVCR_PINNED_SOURCES:-1}
log "pinned sources: $KVCR_PINNED_SOURCES (1 = Dockerfile DYNAMO_REF / KVCR_VLLM_REF)"
mkdir -p "$E2E_BUILD_OUTPUT"
log "build+push start"
bash scripts/docker/build_image.sh
log "build+push done: $KVCR_IMAGE_REPO:$KVCR_REF"
cat "$E2E_BUILD_OUTPUT"/*.json 2>/dev/null | head -60 || true
