# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Core KVCR API, lifecycle, and close-contract tests."""

import ctypes
import logging
import threading
from contextlib import nullcontext, suppress
from functools import partial
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
    _mem_descriptor,
    _new_kvcr,
    _new_local_kvcr,
    _poll_until,
)

from kvcr import KVCR, KVCRBindings
from kvcr import api as kvcr_api
from kvcr import progress as kvcr_progress
from kvcr import recovery_journal as kvcr_recovery
from kvcr.config import (
    FrameworkDramInput,
    G3Options,
    KVCRBackendConfigs,
    KVCRConfig,
    KVCRGuardConfig,
    LocalDramOptions,
    RemoteFWDramOptions,
)
from kvcr.core import _BlockRecord
from kvcr.guard_protocol import KVCRPoolHold
from kvcr.local_disk import _G3Residency
from kvcr.local_dram import _LocalDramResidency, _LocalDramState
from kvcr.memory import KVCRPoolSpec
from kvcr.remote_fw_dram import _FwMemResidency
from kvcr.types import BlockKey


def _fake_hold(**fields: Any) -> SimpleNamespace:
    """A hold double that hands its listener over exactly like the real one."""
    hold = SimpleNamespace(**fields)
    hold.hand_listener_to = partial(KVCRPoolHold.hand_listener_to, hold)
    return hold


def test_local_dram_observer_reports_only_stable_slot_changes() -> None:
    """Observers see only stable transitions: bytes READY in a slot, then gone."""
    block_size = 16
    primary = ctypes.create_string_buffer(block_size * 3)
    primary.raw = b"a" * block_size + b"b" * block_size + b"c" * block_size
    local = ctypes.create_string_buffer(block_size)
    agent = FakeNixlAgent()
    agent.state = "DONE"
    kvcr = _new_local_kvcr(agent, local, 1)
    backend = kvcr._core._local_dram
    assert backend is not None
    keys = tuple(BlockKey(f"k{index}".encode()) for index in range(3))
    observed: list[tuple[BlockKey, int | None]] = []

    def observe(key: BlockKey, record: _BlockRecord) -> None:
        residency = record.local_dram
        if residency is None:
            assert 0 not in backend._free_slots
        else:
            assert residency.state is _LocalDramState.READY
            assert local.raw == bytes((ord("a") + keys.index(key),)) * block_size
        observed.append((key, None if residency is None else residency.slot))

    backend.observe_residency(observe)
    address = ctypes.addressof(primary)

    first = kvcr.deposit({keys[0]: [_mem_descriptor(address, block_size)]})
    _poll_until(kvcr, lambda done: first in dict(done))
    assert observed == [(keys[0], 0)]

    agent.state = "ERR"
    failed = kvcr.deposit(
        {keys[1]: [_mem_descriptor(address + block_size, block_size)]}
    )
    failed_result = dict(_poll_until(kvcr, lambda done: failed in dict(done)))[failed]
    assert not failed_result[keys[1]].success
    assert observed == [
        (keys[0], 0),
        (keys[0], None),
    ]

    agent.state = "DONE"
    third = kvcr.deposit(
        {keys[2]: [_mem_descriptor(address + 2 * block_size, block_size)]}
    )
    _poll_until(kvcr, lambda done: third in dict(done))
    backend.acquire_sources((keys[2],))
    backend.retire_sources((keys[2],))
    assert len(observed) == 3
    backend.release_sources((keys[2],))
    assert observed[2:] == [
        (keys[2], 0),
        (keys[2], None),
    ]


_GUARD_CONFIG = KVCRGuardConfig(
    kvcr_service_socket_path="/tmp/kvcr.sock",
    guard_index=3,
    compatibility_digest="Opaque-Digest",
)
# No Guard has handed anything back, so nothing is at its handback path.
_UNSERVED_POOL = SimpleNamespace(
    mapped_snapshot=lambda: nullcontext(None),
    release_snapshot_region=lambda: None,
    _spec=KVCRPoolSpec(
        pool_id="pool_3",
        path="/tmp/kvcr-pool_3-" + "b" * 32,
        generation="b" * 32,
        device=0,
        inode=0,
        mapping_bytes=8192 + 32,
        journal_bytes=8192,
    ),
)


@pytest.mark.parametrize(
    ("stage", "error", "match", "expected_events"),
    [
        ("control-absent", ValueError, "share its control endpoint", []),
        ("control-cannot-share", ValueError, "share its control endpoint", []),
        (
            "handback-unreadable",
            RuntimeError,
            "region unreadable",
            ["claim", "hold.release"],
        ),
        (
            "install-fails",
            RuntimeError,
            "install failed",
            ["claim", "core.close", "hold.release"],
        ),
    ],
    ids=[
        "control-absent",
        "control-cannot-share",
        "handback-unreadable",
        "install-fails",
    ],
)
def test_a_guarded_startup_that_fails_gives_back_everything_it_took(
    monkeypatch, stage, error, match, expected_events
) -> None:
    """Refused before the claim, or unwound after it: core closed, pool returned."""
    events: list[str] = []
    hold = _fake_hold(
        local_dram=LocalDramOptions(1234, [(8192, "")]),
        _attachment=_UNSERVED_POOL,
        _control_listener_fd=None,
        release=lambda **_kwargs: events.append("hold.release"),
    )

    def claim(*_args, **_kwargs) -> SimpleNamespace:
        events.append("claim")
        return hold

    monkeypatch.setattr(
        kvcr_recovery, "KVCRClient", Mock(return_value=Mock(claim=claim))
    )
    # Supplying a guard_config opts into both the service pool and the Guard, so
    # a control that cannot share its endpoint is refused before anything is
    # taken.
    control: Any = Mock()
    control.control_bind_address.return_value = ("127.0.0.1", 5555)
    if stage == "control-absent":
        control = None
    elif stage == "control-cannot-share":
        control = SimpleNamespace(control_bind_address=None, adopt_listener=None)
    elif stage == "handback-unreadable":
        # The lease is live well before the caller is handed anything.
        monkeypatch.setattr(
            kvcr_recovery,
            "read_handback",
            Mock(side_effect=RuntimeError("region unreadable")),
        )
    elif stage == "install-fails":
        # The core exists before wiring can fail, and it maps the pool's bytes,
        # so it has to close before the pool goes back.

        class _Core:
            def __init__(self, *_args, **_kwargs) -> None:
                self._local_dram = Mock()
                self._g3 = None
                self._block_record_map: dict = {}

            def close(self) -> None:
                events.append("core.close")

            def is_quiescent(self) -> bool:
                return True

        monkeypatch.setattr(kvcr_recovery, "_KVCRCore", _Core)
        monkeypatch.setattr(kvcr_recovery, "RecoveryJournal", Mock())
        monkeypatch.setattr(kvcr_recovery, "_attach_journal", Mock())
        monkeypatch.setattr(
            kvcr_recovery,
            "install_recovery_records",
            Mock(side_effect=RuntimeError("install failed")),
        )

    with pytest.raises(error, match=match):
        KVCR(
            KVCRConfig(
                nixl_agent_name="target",
                pool_layout=[(1024, "")],
                nixl_listen_port=1,
            ),
            KVCRBindings(Mock(), Mock(), Mock(), framework_control=control),
            KVCRBackendConfigs(),
            _GUARD_CONFIG,
        )

    assert events == expected_events


@pytest.mark.parametrize(
    "guard_config",
    [None, _GUARD_CONFIG],
    ids=["local-backends", "service-pool"],
)
def test_startup_timeout_retains_nonquiescent_resources(
    monkeypatch, guard_config
) -> None:
    # A guard_config now requires a control that can hand its endpoint over.
    guarded_control = Mock()
    guarded_control.control_bind_address.return_value = ("127.0.0.1", 5555)
    entered = threading.Event()
    unblock = threading.Event()
    hold = _fake_hold(
        local_dram=LocalDramOptions(1234, [(8192, "")]),
        _attachment=_UNSERVED_POOL,
        _control_listener_fd=None,
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
        kvcr_recovery,
        "KVCRClient",
        Mock(return_value=Mock(claim=Mock(return_value=hold))),
    )
    # Either path may build it: a plain core in api, a claimed one in recovery.
    monkeypatch.setattr(kvcr_api, "_KVCRCore", create_core)
    monkeypatch.setattr(kvcr_recovery, "_KVCRCore", create_core)
    monkeypatch.setattr(kvcr_recovery, "RecoveryJournal", Mock())
    monkeypatch.setattr(kvcr_api, "_NONQUIESCENT_STARTUP_RESOURCES", retained)
    monkeypatch.setattr(kvcr_progress, "_JOIN_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(kvcr_progress, "nixl_agent", create_agent)
    monkeypatch.setattr(kvcr_progress, "nixl_agent_config", lambda **kwargs: kwargs)

    try:
        with pytest.raises(RuntimeError, match="progress thread did not start"):
            KVCR(
                KVCRConfig(
                    nixl_agent_name="target",
                    pool_layout=[(1024, "")],
                    nixl_listen_port=1,
                ),
                KVCRBindings(
                    Mock(),
                    Mock(),
                    Mock(),
                    framework_control=None if guard_config is None else guarded_control,
                ),
                KVCRBackendConfigs(
                    local_dram=hold.local_dram if guard_config is None else None
                ),
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


def test_service_journal_is_attached_before_primary_start(
    tmp_path, monkeypatch
) -> None:
    """Journal, records, and listener are wired in before the core may start."""
    events: list[str] = []
    attachment = _UNSERVED_POOL
    hold = _fake_hold(
        local_dram=LocalDramOptions(1234, [(8192, "")]),
        _attachment=attachment,
        _control_listener_fd=7,
        release=lambda **_kwargs: events.append("hold.release"),
    )
    claim = Mock(return_value=hold)
    local_dram = object()
    g3 = object()
    core = Mock(_local_dram=local_dram, _g3=g3)
    core.start.side_effect = lambda: events.append("core.start")
    core.close.side_effect = lambda: events.append("core.close")
    constructor = Mock(return_value=core)
    journal = object()

    def make_journal(pool) -> object:
        assert pool is attachment
        events.append("journal")
        return journal

    def attach_journal(local, configured_journal, disk) -> None:
        assert (local, configured_journal, disk) == (local_dram, journal, g3)
        events.append("attach")

    monkeypatch.setattr(
        kvcr_recovery, "KVCRClient", Mock(return_value=Mock(claim=claim))
    )
    monkeypatch.setattr(kvcr_recovery, "_KVCRCore", constructor)
    monkeypatch.setattr(kvcr_recovery, "RecoveryJournal", make_journal)
    monkeypatch.setattr(kvcr_recovery, "_attach_journal", attach_journal)
    monkeypatch.setattr(
        kvcr_recovery,
        "install_recovery_records",
        lambda _core, _records: events.append("install"),
    )
    monkeypatch.setattr(
        kvcr_recovery,
        "clear_recovery_snapshot",
        lambda _pool: events.append("clear"),
    )
    primary_control = Mock()
    primary_control.control_bind_address.return_value = ("127.0.0.1", 5555)
    primary_control.adopt_listener.side_effect = lambda fd: events.append(f"adopt:{fd}")
    g3_config = G3Options(
        paths=(tmp_path / "g3",),
        capacity_bytes_per_file=8192,
    )
    backend_configs = KVCRBackendConfigs(
        g3=g3_config,
        remote_fw_dram=RemoteFWDramOptions(backend="REMOTE"),
    )
    controller = KVCR(
        KVCRConfig(nixl_agent_name="target", pool_layout=[(1024, "")]),
        KVCRBindings(Mock(), Mock(), Mock(), framework_control=primary_control),
        backend_configs,
        KVCRGuardConfig(
            kvcr_service_socket_path="/tmp/kvcr.sock",
            guard_index=3,
            compatibility_digest="Opaque-Digest",
        ),
    )

    claim.assert_called_once_with(
        3,
        [(1024, "")],
        "Opaque-Digest",
        ("127.0.0.1", 5555),
        g3_config,
        "REMOTE",
    )
    assert constructor.call_args.args[1].framework_control is primary_control
    assert constructor.call_args.args[2].g3 is g3_config
    # The region is consumed after the core starts, and the listener is taken last
    # during adoption -- a failure before that leaves the hold owning the descriptor.
    assert events == [
        "journal",
        "attach",
        "install",
        "adopt:7",
        "core.start",
        "clear",
    ]
    # Disowned once the channel has it, so nothing closes the fd twice.
    assert hold._control_listener_fd is None
    controller.close()
    assert events[-2:] == ["core.close", "hold.release"]
    assert controller._pool_hold is None


def test_service_dram_rejects_explicit_local_dram_before_claim(monkeypatch) -> None:
    client = Mock()
    monkeypatch.setattr(kvcr_recovery, "KVCRClient", client)

    with pytest.raises(ValueError, match="local_dram"):
        KVCR(
            KVCRConfig(nixl_agent_name="target", pool_layout=[(1024, "")]),
            KVCRBindings(Mock(), Mock(), Mock()),
            KVCRBackendConfigs(local_dram=LocalDramOptions(1234, [(8192, "")])),
            _GUARD_CONFIG,
        )

    client.assert_not_called()


def test_kvcr_rejects_no_dram_backends() -> None:
    with pytest.raises(ValueError, match="at least one DRAM backend"):
        KVCR(
            KVCRConfig(nixl_agent_name="target", pool_layout=[(16, "")]),
            KVCRBindings(Mock(), Mock(), Mock()),
            KVCRBackendConfigs(),
        )


def test_kvcr_rejects_multi_pool_layout() -> None:
    with pytest.raises(ValueError, match="only a single pool"):
        KVCR(
            KVCRConfig(
                nixl_agent_name="target",
                pool_layout=[(8, "full"), (8, "swa")],
            ),
            KVCRBindings(Mock(), Mock(), Mock()),
            KVCRBackendConfigs(),
        )


def test_kvcr_rejects_ambiguous_pool_names() -> None:
    bindings = KVCRBindings(Mock(), Mock(), Mock())
    for pool_layout, message in (
        ([(8, ""), (8, "swa")], "empty"),
        ([(8, "swa"), (8, "swa")], "unique"),
    ):
        config = KVCRConfig(nixl_agent_name="target", pool_layout=pool_layout)
        with pytest.raises(ValueError, match=message):
            KVCR(config, bindings, KVCRBackendConfigs())


def test_fetch_requires_layout_for_a_named_single_pool() -> None:
    local = ctypes.create_string_buffer(16)
    kvcr = _new_kvcr(
        FakeNixlAgent(),
        FakePrimaryPinning(),
        FakeBytesControl(),
        KVCRConfig(nixl_agent_name="target", pool_layout=[(16, "named")]),
        local_dram=LocalDramOptions(ctypes.addressof(local), [(16, "named")]),
    )

    with pytest.raises(ValueError, match="expected layout"):
        kvcr.fetch((BlockKey(b"key"),))
    kvcr.fetch((BlockKey(b"key"),), expected_layout=["named"])


def test_get_stats_emits_public_state_metric_name() -> None:
    kvcr = _new_kvcr(
        FakeNixlAgent(),
        FakePrimaryPinning(),
        FakeBytesControl(),
        KVCRConfig(
            nixl_agent_name="target",
            pool_layout=[(16, "")],
            enable_telemetry=True,
        ),
    )

    stats = kvcr.get_stats()

    assert isinstance(stats, FakeTelemetryStats)
    assert {record[1] for record in stats.records} == {"kvcr_state"}


@pytest.mark.parametrize(
    ("remote_enabled", "expected_backends"),
    [(False, {"LOCAL"}), (True, {"LOCAL", "REMOTE"})],
)
def test_nixl_lifecycle_stays_on_progress_thread(
    monkeypatch, remote_enabled, expected_backends
) -> None:
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
            pool_layout=[(64, "")],
            nixl_listen_port=1234,
        ),
        KVCRBindings(
            pinning.request_pin,
            pinning.poll_pin_results,
            pinning.release_pin,
            framework_control=(
                SimpleNamespace(recv=lambda: ()) if remote_enabled else None
            ),
        ),
        KVCRBackendConfigs(
            framework_dram=FrameworkDramInput(128, 256),
            local_dram=LocalDramOptions(384, [(128, "")], "LOCAL"),
            remote_fw_dram=RemoteFWDramOptions(backend="REMOTE"),
        ),
    )
    kvcr.close()

    assert len(agents) == 1
    agent = agents[0]
    assert agent.name == "target"
    assert agent.config["listen_port"] == 1234
    assert agent.config["enable_listen_thread"] is True
    assert set(agent.config["backends"]) == expected_backends
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


def test_block_record_holds_no_set_while_no_operation_is_in_flight() -> None:
    """Pin the storage, not just the behaviour.

    Every observable use of these helpers behaves identically if the empty set
    is kept instead of dropped, so only an explicit check keeps the record from
    quietly growing an allocation per resident block again.
    """
    record = _BlockRecord()
    assert record.in_flight_ops is None
    assert record.active_op_ids == ()

    record.discard_in_flight_op(("target", 1))
    assert record.in_flight_ops is None

    record.add_in_flight_op(("target", 1))
    record.add_in_flight_op(("source", 2))
    assert record.in_flight_ops == {("target", 1), ("source", 2)}

    # The snapshot is what lets a caller retire ops while iterating.
    snapshot = record.active_op_ids
    record.discard_in_flight_op(("target", 1))
    assert set(snapshot) == {("target", 1), ("source", 2)}
    assert record.in_flight_ops == {("source", 2)}

    record.discard_in_flight_op(("source", 2))
    assert record.in_flight_ops is None
    assert record.active_op_ids == ()


def test_resident_records_carry_no_instance_dictionary() -> None:
    """Every record a resident block can hold, so none of them grows one back."""
    for residency in (
        _BlockRecord(),
        _LocalDramResidency(0, _LocalDramState.READY),
        _G3Residency(0),
        _FwMemResidency(_mem_descriptor(), object()),
    ):
        assert not hasattr(residency, "__dict__"), type(residency).__name__


@pytest.mark.parametrize(
    "release_fails", [False, True], ids=["release-works", "release-also-fails"]
)
def test_close_gives_the_pool_back_when_the_core_errors_but_quiesces(
    monkeypatch, new_kvcr, caplog, release_fails
) -> None:
    """A quiesced core's close error is raised, but the pool release still runs."""
    kvcr = new_kvcr()
    core_error = RuntimeError("core close failed")
    monkeypatch.setattr(kvcr._core, "close", Mock(side_effect=core_error))
    monkeypatch.setattr(kvcr._core, "is_quiescent", lambda: True)
    hold = Mock()
    if release_fails:
        hold.release.side_effect = RuntimeError("release failed")
    kvcr._pool_hold = hold

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as raised:
            kvcr.close()

    assert raised.value is core_error
    hold.release.assert_called_once_with()
    if release_fails:
        # The release failure is a consequence of the core's; it is only logged.
        assert "Failed to release the KVCR pool after close" in caplog.text
    else:
        # Forgotten as well as released, so nothing can give the pool back twice.
        assert kvcr._pool_hold is None
