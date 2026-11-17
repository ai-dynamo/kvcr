#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# What a node offers: CPU layout, RDMA devices (NUMA node, link, netdev),
# interfaces, limits. Run it on a compute node before the first cell to pick
# CB_CTRL_IF and CB_NET_DEVICES.
echo "== $(hostname -s) $(uname -m) $(python3 --version 2>&1)"
lscpu | grep -E "^CPU\(s\)|Thread|Core|Socket|NUMA node[0-9]|Model name"
for d in /sys/class/infiniband/*; do
  [ -e "$d" ] || continue
  n=$(basename "$d")
  echo "hca $n numa=$(cat "$d/device/numa_node") link=$(cat "$d/ports/1/link_layer")" \
    "state=$(cut -d' ' -f2 "$d/ports/1/state") rate=$(cat "$d/ports/1/rate")" \
    "netdev=$(ls "$d/device/net" 2>/dev/null | tr '\n' ' ')"
done
ip -4 -br addr | grep -v "^lo"
echo "memlock=$(ulimit -l) nofile=$(ulimit -n) mem=$(free -g | awk '/Mem:/{print $2}')G"
echo "ptrace_scope=$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null)"
