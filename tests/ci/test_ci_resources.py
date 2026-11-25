# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource checks observe collisions without killing foreign processes."""

import importlib
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "e2e"))


def test_port_observation_detects_listener_and_recovery():
    resources = importlib.import_module("ci_resources")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    try:
        observation = resources.observe_ports([port])
        assert observation["status"] == "COLLISION"
        assert observation["unavailable_ports"] == [port]
    finally:
        listener.close()
    assert resources.observe_ports([port])["status"] == "AVAILABLE"


def test_vllm_resource_policy_includes_disagg_side_channels():
    adapter = importlib.import_module("ci_integrations.vllm")
    profile = {"topology": "disagg"}
    assert {5601, 5602, 5603}.issubset(adapter.runtime_resources(profile)["tcp_ports"])


def test_reusable_closed_server_port_is_not_a_cleanup_collision():
    resources = importlib.import_module("ci_resources")
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port)) as client:
            connection, _ = listener.accept()
            connection.close()  # Server performs active close and enters TIME_WAIT.
            assert client.recv(1) == b""
    assert resources.observe_ports([port])["status"] == "AVAILABLE"
