# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ctypes
from pathlib import Path

import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _new_kvcr,
)
from nixl import nixlBackendError

from kvcr.backends import unmanaged_storage
from kvcr.backends.unmanaged_storage import _G4
from kvcr.config import G4Options, LocalDramOptions
from kvcr.runtime import core as kvcr_core
from kvcr.runtime import progress as kvcr_progress


class _Agent:
    def __init__(self, name, config, *, plugins, existing):
        self.name = name
        self.config = config
        self._plugins = plugins
        self.backends = {} if existing is None else dict(existing)
        self.created = None

    def get_plugin_list(self):
        return self._plugins

    def get_backend_params(self, name):
        return self.backends[name]

    def create_backend(self, name, params):
        self.created = (name, params)
        self.backends[name] = params

    def register_memory(self, reg_list, mem_type=None, backends=()):
        self.registered = (tuple(reg_list), mem_type, list(backends))


def _install_agent(monkeypatch, *, plugins, existing=None):
    holder = {}

    def create(name, config):
        holder["agent"] = _Agent(name, config, plugins=plugins, existing=existing)
        return holder["agent"]

    monkeypatch.setattr(unmanaged_storage, "nixl_agent", create)
    monkeypatch.setattr(unmanaged_storage, "nixl_agent_config", lambda **kwargs: kwargs)
    return holder


@pytest.mark.parametrize(
    ("backend_options", "params"),
    [
        ({}, {"use_aio": "true"}),
        ({"use_uring": "true"}, {"use_uring": "true"}),
    ],
)
def test_g4_opens_the_file_backend(
    tmp_path, monkeypatch, backend_options, params
) -> None:
    holder = _install_agent(monkeypatch, plugins=["POSIX"])
    client = _G4(G4Options(root=tmp_path, backend_options=backend_options), "worker-g4")
    agent = holder["agent"]
    assert agent.name == "worker-g4"
    assert agent.config == {"backends": []}
    assert agent.created == ("POSIX", params)
    assert agent.registered == (((0, 0, 0, f"ro:{tmp_path}"),), "FILE", ["POSIX"])
    client.close_progress()
    assert client._agent is None


def test_g4_initialize_progress_opens_the_progress_backend(
    tmp_path: Path, monkeypatch
) -> None:
    _install_agent(monkeypatch, plugins=["POSIX"])
    client = _G4(G4Options(root=tmp_path), "worker-g4")
    progress_agent = _Agent("progress", {}, plugins=["POSIX"], existing=None)

    class _Progress:
        nixl_agent = progress_agent

    client.initialize_progress(_Progress())
    assert progress_agent.created == ("POSIX", {"use_aio": "true"})


def test_g4_reuses_a_matching_backend(tmp_path: Path, monkeypatch) -> None:
    params = {"use_aio": "true"}
    options = G4Options(root=tmp_path)
    holder = _install_agent(monkeypatch, plugins=["POSIX"], existing={"POSIX": params})
    _G4(options, "worker-g4")
    assert holder["agent"].created is None

    _install_agent(monkeypatch, plugins=["POSIX"], existing={"POSIX": {}})
    with pytest.raises(ValueError, match="incompatible options"):
        _G4(options, "worker-g4")


def test_g4_nixl_client_fails_for_an_unreachable_root(tmp_path: Path) -> None:
    with pytest.raises(nixlBackendError):
        _G4(G4Options(root=tmp_path / "missing"), "worker-g4-missing")


def test_g4_requires_the_nixl_plugin(tmp_path: Path, monkeypatch) -> None:
    _install_agent(monkeypatch, plugins=[])
    with pytest.raises(RuntimeError, match="POSIX plugin is not available"):
        _G4(G4Options(root=tmp_path), "worker-g4")


def test_kvcr_opens_and_closes_the_g4_client(tmp_path: Path, monkeypatch) -> None:
    calls = []

    class _Recording:
        def __init__(self, options, agent_name):
            calls.append(("open", options.root, agent_name))

        def initialize_progress(self, progress):
            return None

        def close_progress(self):
            calls.append("close")

    monkeypatch.setattr(kvcr_core, "_G4", _Recording)
    monkeypatch.setattr(kvcr_progress, "nixl_agent_config", lambda **kwargs: kwargs)
    local = ctypes.create_string_buffer(16)
    kvcr = _new_kvcr(
        FakeNixlAgent(),
        FakePrimaryPinning(),
        FakeBytesControl(),
        local_dram=LocalDramOptions([("", ctypes.addressof(local), len(local))]),
        g4=G4Options(root=tmp_path),
    )
    assert calls == [("open", tmp_path, "target-g4")]
    kvcr.close()
    assert calls[-1] == "close"
