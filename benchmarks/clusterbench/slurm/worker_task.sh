#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# One worker. Started once per task by run_cell.sh's srun
# (--ntasks-per-node=K), so SLURM_LOCALID is this worker's index on its node.
set -euo pipefail
: "${CB_ROOT:?}" "${CB_ARM:?}" "${CB_COORD:?}" "${CB_PER_NODE:?}"
export PYTHONPATH=$CB_ROOT/site:$CB_ROOT/src/$CB_ARM/src:${CB_HARNESS:-$CB_ROOT/harness}/benchmarks
export UCX_TLS=${CB_UCX_TLS:-rc,self,sm,cuda_copy}
# NIXL rejects a UCX_TLS list without cuda* when it sees GPUs; hide them.
export CUDA_VISIBLE_DEVICES=
if [ -n "${CB_CTRL_IF:-}" ]; then
  addr=$(ip -4 -o addr show dev "$CB_CTRL_IF" | awk '{print $4}' | cut -d/ -f1 | head -1)
else
  addr=$(hostname -I | awk '{print $1}')
fi
exec python3 -m clusterbench worker \
  --coordinator "$CB_COORD" --node "$(hostname -s)" --local-id "$SLURM_LOCALID" \
  --bind-host "$addr" --control-port $((17800 + SLURM_LOCALID)) \
  --nixl-port $((17900 + SLURM_LOCALID)) \
  --cpus auto --per-node "$CB_PER_NODE" \
  --net-devices "${CB_NET_DEVICES:-}"
