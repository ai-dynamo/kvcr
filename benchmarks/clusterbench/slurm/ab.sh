#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Interleaved A/B (or A/A, or N arms): REPS cells per arm, rotating which arm
# goes first so drift over the allocation does not favour one arm.
#
#   ab.sh JOBID PREFIX NODES PER_NODE CONFIG REPS "ARM ARM ..." [--set ...]
#
# Prints the comparison at the end; the first arm is the baseline.
set -uo pipefail
: "${CB_ROOT:?}"
JOB=$1 PREFIX=$2 NODES=$3 PER_NODE=$4 CONFIG=$5 REPS=$6
read -r -a ARMS <<< "$7"
shift 7
slurm=${CB_HARNESS:-$CB_ROOT/harness}/benchmarks/clusterbench/slurm
declare -A dirs
for rep in $(seq 0 $((REPS - 1))); do
  for k in $(seq 0 $((${#ARMS[@]} - 1))); do
    arm=${ARMS[$(((rep + k) % ${#ARMS[@]}))]}
    cell=$PREFIX-$arm-$rep
    bash "$slurm/run_cell.sh" "$JOB" "$arm" "$cell" "$NODES" "$PER_NODE" "$CONFIG" "$@" > "$CB_ROOT/logs/$cell.out" 2>&1
    echo "$cell rc=$?"
    [ -f "$CB_ROOT/runs/$cell/summary.json" ] && dirs[$arm]+="$CB_ROOT/runs/$cell,"
  done
done
args=()
for arm in "${ARMS[@]}"; do args+=(--arm "$arm=${dirs[$arm]:-}"); done
PYTHONPATH=${CB_HARNESS:-$CB_ROOT/harness}/benchmarks python3 -m clusterbench report "${args[@]}"
