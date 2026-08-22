# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Core KVCR API, lifecycle, and close-contract tests."""

import threading
from contextlib import suppress
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from _kvcr_test_utils import (
    _OPEN_KVCRS,
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    FakeTelemetryStats,
    _new_kvcr,
)

from kvcr import KVCR, KVCRBindings
from kvcr import api as kvcr_api
from kvcr import progress as kvcr_progress
from kvcr.config import (
    FrameworkDramInput,
    KVCRBackendConfigs,
    KVCRConfig,
    KVCRGuardConfig,
    LocalDramInfo,
)

_GUARD_CONFIG = KVCRGuardConfig(
    kvcr_service_socket_path="/tmp/kvcr.sock",
    pool_index=3,
    row_stride=1024,
    compatibility_digest="Opaque-Digest",
)


def test_service_dram_is_resolved_and_released_after_core(monkeypatch) -> None:
    events: list[str] = []
    hold = SimpleNamespace(
        local_dram=LocalDramInfo(1234, 8192, 8),
        release=lambda: events.append("hold.release"),
    )
    claim = Mock(return_value=hold)
    core = Mock()
    core.start.side_effect = lambda: events.append("core.start")
    core.close.side_effect = lambda: events.append("core.close")
    constructor = Mock(return_value=core)

    monkeypatch.setattr(kvcr_api, "KVCRClient", Mock(return_value=Mock(claim=claim)))
    monkeypatch.setattr(kvcr_api, "_KVCRCore", constructor)
    controller = KVCR(
        KVCRConfig(
            nixl_agent_name="target",
            nixl_listen_port=1,
        ),
        KVCRBindings(Mock(), Mock(), Mock()),
        KVCRBackendConfigs(),
        _GUARD_CONFIG,
    )

    claim.assert_called_once_with(3, 1024, "Opaque-Digest")
    assert constructor.call_args.args[2].local_dram is hold.local_dram
    controller.close()
    assert events == ["core.start", "core.close", "hold.release"]


@pytest.mark.parametrize(
    "guard_config",
    [None, _GUARD_CONFIG],
    ids=["local-backends", "service-pool"],
)
def test_startup_timeout_retains_nonquiescent_resources(
    monkeypatch, guard_config
) -> None:
    entered = threading.Event()
    unblock = threading.Event()
    hold = SimpleNamespace(
        local_dram=LocalDramInfo(1234, 8192, 8),
        release=Mock(),
    )
    retained: list[tuple[object, object | None]] = []
    cores: list[Any] = []
    core_type = kvcr_api._KVCRCore

    def create_core(*args, **kwargs):
        core = core_type(*args, **kwargs)
        cores.append(core)
        return core

    def create_agent(*_args, **_kwargs):
        entered.set()
        unblock.wait()
        return FakeNixlAgent()

    monkeypatch.setattr(
        kvcr_api,
        "KVCRClient",
        Mock(return_value=Mock(claim=Mock(return_value=hold))),
    )
    monkeypatch.setattr(kvcr_api, "_KVCRCore", create_core)
    monkeypatch.setattr(kvcr_api, "_NONQUIESCENT_STARTUP_RESOURCES", retained)
    monkeypatch.setattr(kvcr_progress, "_JOIN_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(kvcr_progress, "nixl_agent", create_agent)
    monkeypatch.setattr(kvcr_progress, "nixl_agent_config", lambda **kwargs: kwargs)

    try:
        with pytest.raises(RuntimeError, match="progress thread did not start"):
            KVCR(
                KVCRConfig(nixl_agent_name="target", nixl_listen_port=1),
                KVCRBindings(Mock(), Mock(), Mock()),
                KVCRBackendConfigs(),
                guard_config,
            )

        assert entered.wait(timeout=1)
        assert len(cores) == 1
        core = cores[0]
        assert not core.is_quiescent()
        hold.release.assert_not_called()
        assert retained == [(core, hold if guard_config is not None else None)]
    finally:
        unblock.set()
        if cores:
            cores[0]._progress._thread.join(timeout=1)

    assert not cores[0]._progress._thread.is_alive()
    assert cores[0].is_quiescent()


def test_service_dram_rejects_explicit_local_dram_before_claim(monkeypatch) -> None:
    client = Mock()
    monkeypatch.setattr(kvcr_api, "KVCRClient", client)

    with pytest.raises(ValueError, match="local_dram"):
        KVCR(
            KVCRConfig(nixl_agent_name="target"),
            KVCRBindings(Mock(), Mock(), Mock()),
            KVCRBackendConfigs(local_dram=LocalDramInfo(1234, 8192, 8)),
            _GUARD_CONFIG,
        )

    client.assert_not_called()


def test_get_stats_emits_public_state_metric_name() -> None:
    kvcr = _new_kvcr(
        FakeNixlAgent(),
        FakePrimaryPinning(),
        FakeBytesControl(),
        KVCRConfig(nixl_agent_name="target", enable_telemetry=True),
    )

    stats = kvcr.get_stats()

    assert isinstance(stats, FakeTelemetryStats)
    assert {record[1] for record in stats.records} == {"kvcr_state"}


def test_nixl_lifecycle_stays_on_progress_thread(monkeypatch) -> None:
    main_thread = threading.get_ident()
    lifecycle_threads: list[int] = []
    agents: list[Any] = []

    class LifecycleAgent(FakeNixlAgent):
        def __init__(self, name, config):
            lifecycle_threads.append(threading.get_ident())
            super().__init__()
            self.name = name
            self.config = config
            agents.append(self)

        def register_memory(self, descs, mem_type="DRAM"):
            lifecycle_threads.append(threading.get_ident())
            return super().register_memory(descs, mem_type)

        def deregister_memory(self, handle):
            lifecycle_threads.append(threading.get_ident())
            super().deregister_memory(handle)

    pinning = FakePrimaryPinning()
    monkeypatch.setattr(kvcr_progress, "nixl_agent", LifecycleAgent)
    monkeypatch.setattr(kvcr_progress, "nixl_agent_config", lambda **kwargs: kwargs)

    kvcr = KVCR(
        KVCRConfig(
            nixl_agent_name="target",
            nixl_listen_port=1234,
        ),
        KVCRBindings(
            pinning.request_pin,
            pinning.poll_pin_results,
            pinning.release_pin,
        ),
        KVCRBackendConfigs(
            framework_dram=FrameworkDramInput(128, 256),
            local_dram=LocalDramInfo(384, 128, 2),
        ),
    )
    kvcr.close()

    assert len(agents) == 1
    agent = agents[0]
    assert agent.name == "target"
    assert agent.config["listen_port"] == 1234
    assert agent.config["enable_listen_thread"] is True
    assert len(set(lifecycle_threads)) == 1
    assert lifecycle_threads[0] != main_thread
    assert agent.registrations == [
        ([(128, 256, 0, "")], "DRAM"),
        ([(384, 128, 0, "")], "DRAM"),
    ]
    assert agent.deregistered == [2, 1]


@pytest.fixture
def new_kvcr():
    """Build KVCRs and retain their real progress threads for cleanup."""
    started = []

    def make():
        kvcr = _new_kvcr(FakeNixlAgent(), FakePrimaryPinning(), FakeBytesControl())
        _OPEN_KVCRS.remove(kvcr)
        started.append(kvcr._core._progress)
        return kvcr

    yield make
    for progress in started:
        with suppress(Exception):
            progress.close()


class _StubProgress:
    def __init__(self, error: BaseException | None, quiescent: bool) -> None:
        self._error = error
        self._quiescent = quiescent

    def close(self) -> None:
        if self._error is not None:
            raise self._error

    def is_quiescent(self) -> bool:
        return self._quiescent


def test_close_cleans_backends_once_when_progress_is_quiescent(
    monkeypatch, new_kvcr
) -> None:
    """A quiescent progress loop permits backend teardown exactly once."""
    kvcr = new_kvcr()
    core = kvcr._core
    cleaned: list[str] = []
    monkeypatch.setattr(
        core._remote_fw_dram, "close_main", lambda: cleaned.append("remote")
    )
    monkeypatch.setattr(core, "_progress", _StubProgress(None, quiescent=True))

    kvcr.close()
    assert cleaned == ["remote"]

    kvcr.close()
    assert cleaned == ["remote"], "close must not tear down a second time"


@pytest.mark.parametrize(
    "error",
    [
        None,
        RuntimeError("nixl deregistration failed"),
    ],
    ids=["no_progress_error", "progress_error"],
)
def test_close_preserves_backends_when_progress_is_not_quiescent(
    monkeypatch, new_kvcr, error: BaseException | None
) -> None:
    """Close preserves backends while native state may still reference them."""
    kvcr = new_kvcr()
    core = kvcr._core
    cleaned: list[str] = []
    monkeypatch.setattr(
        core._remote_fw_dram, "close_main", lambda: cleaned.append("remote")
    )
    monkeypatch.setattr(core, "_progress", _StubProgress(error, quiescent=False))

    expected = str(error) if error is not None else "not quiescent"
    for _ in range(2):
        with pytest.raises(RuntimeError, match=expected):
            kvcr.close()
    assert cleaned == []
