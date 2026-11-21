#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Run one cell inside a keeper allocation, from the login node.
#
#   run_cell.sh JOBID ARM CELL NODES PER_NODE CONFIG [--set section.field=value ...]
#
# The coordinator runs on the first node (on the cores reserved from the
# workers) and NODES x PER_NODE workers run as one job step. Results land in
# $CB_ROOT/runs/CELL.
#
#   CB_ROOT         staging root (required)
#   CB_HARNESS      harness copy to run (default $CB_ROOT/harness), so a queued
#                   job keeps the harness it was submitted with
#   CB_CTRL_IF      interface whose address KVCR's control channel advertises
#   CB_NET_DEVICES  UCX_NET_DEVICES per socket, e.g. mlx5_0:1,mlx5_1:1
#   CB_UCX_TLS      UCX transports (default rc,self,sm,cuda_copy)
#   CB_COORD_PORT   coordinator port (default 17700)
set -uo pipefail
: "${CB_ROOT:?}"
JOB=$1 ARM=$2 CELL=$3 NODES=$4 PER_NODE=$5 CONFIG=$6
shift 6
RUN=$CB_ROOT/runs/$CELL
harness=${CB_HARNESS:-$CB_ROOT/harness}
[ -e "$RUN" ] && { echo "run_cell.sh: $RUN exists" >&2; exit 2; }
[ -d "$CB_ROOT/src/$ARM/src" ] || { echo "run_cell.sh: arm $ARM is not staged in $CB_ROOT/src" >&2; exit 2; }
mapfile -t nodes < <(head -n "$NODES" "$CB_ROOT/control/nodes-$JOB")
[ "${#nodes[@]}" -eq "$NODES" ] || { echo "run_cell.sh: job $JOB has fewer than $NODES nodes" >&2; exit 2; }
# Only now: a start refused above must not leave a directory that refuses the retry.
mkdir -p "$RUN/logs"
head_node=${nodes[0]}
nodelist=$(IFS=,; echo "${nodes[*]}")
port=${CB_COORD_PORT:-17700}

# The workers leave the first two cores of every socket free.
srun --jobid "$JOB" --overlap -N1 -n1 -w "$head_node" --cpu-bind=none \
  env PYTHONPATH="$CB_ROOT/site:$harness/benchmarks" taskset -c 0-1 \
  python3 -m clusterbench coordinator \
  --config "$CONFIG" --bind "tcp://0.0.0.0:$port" --expect $((NODES * PER_NODE)) \
  --per-node "$PER_NODE" --expect-kvcr "$CB_ROOT/src/$ARM/src" \
  --arm "$ARM" --out "$RUN" "$@" \
  > "$RUN/coordinator.log" 2>&1 &
coordinator=$!
srun --jobid "$JOB" --overlap -N "$NODES" -w "$nodelist" --ntasks-per-node="$PER_NODE" \
  --cpu-bind=none --kill-on-bad-exit=1 --output="$RUN/logs/w-%N-%t.log" \
  env CB_ROOT="$CB_ROOT" CB_HARNESS="$harness" CB_ARM="$ARM" \
  CB_COORD="tcp://$head_node:$port" \
  CB_PER_NODE="$PER_NODE" CB_CTRL_IF="${CB_CTRL_IF:-}" \
  CB_NET_DEVICES="${CB_NET_DEVICES:-}" CB_UCX_TLS="${CB_UCX_TLS:-rc,self,sm,cuda_copy}" \
  bash "$harness/benchmarks/clusterbench/slurm/worker_task.sh" &
workers=$!
wait "$coordinator"
rc=$?
# Any failure, a refused argument included, can leave the workers waiting.
[ "$rc" -eq 0 ] || kill "$workers" 2>/dev/null
wait "$workers"
echo "$CELL: coordinator rc=$rc workers rc=$?"
head -n 12 "$RUN/coordinator.log"
exit "$rc"
