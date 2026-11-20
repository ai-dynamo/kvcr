#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Copy the harness and the KVCR source trees under test to the cluster.
#
#   stage.sh ROOT ARM=GIT_REV [ARM=/path/to/kvcr/checkout ...]
#
# ROOT is a directory on the cluster's shared filesystem. Each arm becomes
# ROOT/src/ARM, exported from this repository at GIT_REV (or copied from a
# checkout, uncommitted changes included) with its revision in .kvcr-sha.
#
#   CB_HOST       ssh host of the cluster login node (required)
#   CB_SITE_FROM  existing site-packages directory on the cluster to copy into
#                 ROOT/site when ROOT/site does not exist yet. It must hold
#                 nixl, msgspec and pyzmq built for the compute nodes.
set -euo pipefail
HOST=${CB_HOST:?set CB_HOST to the ssh host of the cluster login node}
ROOT=${1:?usage: stage.sh ROOT ARM=REV...}
shift
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
# One connection for everything: login nodes often throttle rapid new ones.
# If your ssh config already shares one for this host, use it; a site with
# interactive sign-in cannot open another unattended.
if ssh -O check "$HOST" 2>/dev/null; then
  SSH=(ssh)
else
  ctl=$(mktemp -u "${TMPDIR:-/tmp}/cb-ssh-XXXXXX")
  SSH=(ssh -o ControlMaster=auto -o ControlPath="$ctl" -o ControlPersist=60)
  trap '"${SSH[@]}" -O exit "$HOST" 2>/dev/null' EXIT
fi

"${SSH[@]}" "$HOST" "mkdir -p '$ROOT'/{harness,src,runs,logs,control}"
rsync -e "${SSH[*]}" -a --delete --exclude __pycache__ "$REPO/benchmarks/" "$HOST:$ROOT/harness/benchmarks/"
for arm in "$@"; do
  name=${arm%%=*}
  rev=${arm#*=}
  [ -n "$name" ] || { echo "stage.sh: no arm name in '$arm'" >&2; exit 2; }
  # The staged arm is replaced only once its successor is complete, so a bad
  # revision or a broken copy leaves it as it was.
  new=$ROOT/src/$name.new
  if [ -d "$rev" ]; then
    sha=$(git -C "$rev" rev-parse --short=12 HEAD)
    [ -z "$(git -C "$rev" status --porcelain -- src)" ] || sha=$sha-dirty
    "${SSH[@]}" "$HOST" "rm -rf '$new' && mkdir -p '$new'"
    rsync -e "${SSH[*]}" -a --exclude __pycache__ "$rev/src" "$HOST:$new/"
  else
    sha=$(git -C "$REPO" rev-parse --short=12 "$rev^{commit}")
    git -C "$REPO" archive "$rev" src \
      | "${SSH[@]}" "$HOST" "rm -rf '$new' && mkdir -p '$new' && tar -x -C '$new'"
  fi
  "${SSH[@]}" "$HOST" \
    "echo $sha > '$new/.kvcr-sha' && rm -rf '$ROOT/src/$name' && mv '$new' '$ROOT/src/$name'"
  echo "staged $name = $sha"
done
if [ -n "${CB_SITE_FROM:-}" ]; then
  "${SSH[@]}" "$HOST" "[ -d '$ROOT/site' ] || cp -a '$CB_SITE_FROM' '$ROOT/site'"
fi
if "${SSH[@]}" "$HOST" "[ ! -d '$ROOT/site' ]"; then
  echo "note: $ROOT/site is missing; see the README" >&2
fi
