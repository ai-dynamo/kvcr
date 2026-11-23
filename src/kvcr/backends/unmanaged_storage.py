# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Open a NIXL file client from user G4 config."""

from typing import Any

from nixl import nixl_agent, nixl_agent_config

from ..config import G4Options
from ..runtime.progress import _KVCRProgress


class _G4:
    def __init__(self, options: G4Options, agent_name: str) -> None:
        if type(agent_name) is not str or not agent_name:
            raise ValueError("G4 agent name must be non-empty")
        backend = options.backend
        agent = nixl_agent(agent_name, nixl_agent_config(backends=[]))
        _open_backend(agent, backend, _file_params(options))
        # ro: selects NIXL path mode, which opens the root. A missing path fails here.
        agent.register_memory(
            [(0, 0, 0, f"ro:{options.root}")],
            "FILE",
            backends=[backend],
        )
        self._config = options
        self._agent = agent

    def initialize_progress(self, progress: _KVCRProgress) -> None:
        agent = progress.nixl_agent
        backend = self._config.backend
        if backend not in agent.get_plugin_list():
            raise RuntimeError(f"NIXL {backend} plugin is not available")
        configured = _file_params(self._config)
        if backend not in getattr(agent, "backends", {}):
            agent.create_backend(backend, configured)
        elif configured and any(
            agent.get_backend_params(backend).get(key) != value
            for key, value in configured.items()
        ):
            raise ValueError(f"incompatible options for NIXL backend {backend}")

    def close_progress(self) -> None:
        self._agent = None


def _file_params(options: G4Options) -> dict[str, str]:
    params = dict(options.backend_options)
    if options.backend == "POSIX" and not any(
        name in params for name in ("use_uring", "use_aio", "use_posix_aio")
    ):
        params["use_aio"] = "true"
    return params


def _open_backend(agent: Any, backend: str, params: dict[str, str]) -> None:
    if backend not in agent.get_plugin_list():
        raise RuntimeError(f"NIXL {backend} plugin is not available")
    existing = getattr(agent, "backends", {})
    if backend in existing:
        current = agent.get_backend_params(backend)
        if any(current.get(key) != value for key, value in params.items()):
            raise ValueError(f"incompatible options for NIXL backend {backend}")
        return
    agent.create_backend(backend, params)
