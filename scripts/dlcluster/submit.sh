#!/usr/bin/env bash
# Bundle the current kvcr-phase1-ci HEAD, ship it to dlcluster, submit phase1.sbatch.
#
#   scripts/dlcluster/submit.sh [--matrix "dense hybrid"] [--cases TC-01,...] \
#                               [--timeout 3600] [--nodelist NODE] [--partition P] [--image local/kvcr-phase1:<sha>]
#                               [--extra "--harness-arg=--batch-invariant"]
#
# Requires a clean, committed worktree: the image build refuses dirty source.
# KVCR_CASES is passed colon-separated because sbatch --export splits on commas.
set -euo pipefail
HOST=dlcluster
MATRIX="dense hybrid"; CASES=TC-01,TC-02,TC-03,TC-05,TC-04,TC-06; TIMEOUT=3600; IMAGE=""; EXTRA_ARGS=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --matrix) MATRIX=$2; shift 2 ;;
    --cases) CASES=$2; shift 2 ;;
    --timeout) TIMEOUT=$2; shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    --extra) EXTRA_ARGS=$2; shift 2 ;;
    --nodelist) EXTRA+=(--nodelist "$2"); shift 2 ;;
    --partition) EXTRA+=(--partition "$2"); shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
repo=$(git rev-parse --show-toplevel); cd "$repo"
git diff --quiet HEAD -- . || { echo "commit your changes first (image build needs clean source)" >&2; exit 1; }
ref=$(git rev-parse HEAD)
bundle="kvcr-phase1-${ref:0:12}.bundle"
git bundle create "/tmp/$bundle" HEAD >/dev/null
scp -q "/tmp/$bundle" scripts/dlcluster/phase1.sbatch "$HOST:~/"
ssh "$HOST" "mkdir -p ~/kvcr-runs && sbatch ${EXTRA[*]:-} \
  --export=ALL,KVCR_BUNDLE=\$HOME/$bundle,KVCR_REF=$ref,KVCR_MATRIX='$MATRIX',KVCR_CASES=${CASES//,/:},KVCR_TIMEOUT=$TIMEOUT,KVCR_IMAGE=$IMAGE,KVCR_EXTRA='$EXTRA_ARGS' \
  ~/phase1.sbatch"
