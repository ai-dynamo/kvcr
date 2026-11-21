#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Hold exclusive nodes; cells run as job steps (run_cell.sh). Submit with the
# site's account, partition and job name on the command line, for example:
#
#   sbatch --nodes=4 --time=01:00:00 --account=ACCOUNT --partition=PARTITION \
#     --job-name=NAME --output=$CB_ROOT/logs/keeper-%j.out \
#     --export=ALL,CB_ROOT=$CB_ROOT keeper.sbatch
#
# Release early with: touch $CB_ROOT/control/release-JOBID
#SBATCH --exclusive
#SBATCH --time=01:00:00
set -u
: "${CB_ROOT:?}"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$CB_ROOT/control/nodes-$SLURM_JOB_ID"
while [ ! -f "$CB_ROOT/control/release-$SLURM_JOB_ID" ]; do sleep 10; done
