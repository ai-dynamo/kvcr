# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe declared resources; never terminate processes based on a port or GPU."""

from __future__ import annotations

import socket
import subprocess


def observe_ports(ports: list[int]) -> dict:
    unavailable, errors = [], []
    for port in sorted(set(ports)):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("0.0.0.0", port))
                probe.listen(1)
        except OSError as error:
            unavailable.append(port)
            errors.append({"port": port, "error": str(error)})
    return {
        "status": "COLLISION" if unavailable else "AVAILABLE",
        "ports": sorted(set(ports)),
        "unavailable_ports": unavailable,
        "errors": errors,
        "scope": "container-network-namespace",
    }


def observe_gpu_processes() -> dict:
    try:
        response = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return {
            "status": "OBSERVED",
            "raw_compute_processes": response.splitlines(),
            "pid_namespace_binding": "UNVERIFIED",
            "proves_zero_gpu_memory": False,
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "status": "UNOBSERVED",
            "error": type(error).__name__,
            "proves_zero_gpu_memory": False,
        }
