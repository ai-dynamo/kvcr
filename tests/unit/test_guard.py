# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Private Guard lifecycle tests."""

import errno
import logging
import os
import queue
import select
import socket
import threading
import time
import uuid
from contextlib import nullcontext
from unittest.mock import Mock

import pytest
from _kvcr_test_utils import _recovered_record

from kvcr.config import LocalDramInfo
from kvcr.control_channels import KVCRServiceError, ZmqPeerControlChannel
from kvcr.core import _BlockRecord
from kvcr.guard import (
    _Command,
    _ConfiguredTier,
    _Guard,
    _Lease,
    _Phase,
)
from kvcr.guard_protocol import _G3Config, _TierConfig
from kvcr.local_disk import _G3Residency
from kvcr.local_dram import _LocalDramResidency, _LocalDramState
from kvcr.memory import KVCRPoolSpec
from kvcr.recovery_journal import (
    _RECORD_BLOCK,
    _RECOVERY_ENCODER,
    RecoveryJournalError,
    RecoveryMirrorError,
    _project_recovery_record,
    _RecoveryBlock,
    _RecoveryMirror,
)
from kvcr.types import BlockKey

_TEST_SPEC = KVCRPoolSpec(
    pool_id="pool_0",
    path="/tmp/kvcr-pool_0-" + "a" * 32,
    generation="a" * 32,
    device=0,
    inode=0,
    mapping_bytes=8192 + 32,
    journal_bytes=8192,
)
_TEST_DIGEST = "opaque digest: Preserve-Me EXACTLY"
# G3 terms are refused at decode unless a real claimant could open them, so
# tests that carry G3 use page-aligned strides over a page-sized pool.
_PAGE_STRIDE = os.sysconf("SC_PAGE_SIZE")
_PAGE_SPEC = KVCRPoolSpec(
    pool_id="pool_0",
    path="/tmp/kvcr-pool_0-" + "a" * 32,
    generation="a" * 32,
    device=0,
    inode=0,
    mapping_bytes=8192 + 2 * _PAGE_STRIDE,
    journal_bytes=8192,
)


def _fake_attachment(**overrides) -> Mock:
    """A stand-in with the pool-tail surface a Guard reaches for."""
    attachment = Mock(
        address=1234,
        data_address=1234 + _TEST_SPEC.journal_bytes,
        _spec=_TEST_SPEC,
        **overrides,
    )
    attachment.mapped_snapshot.return_value = nullcontext(None)
    return attachment


def _wait_until(predicate) -> None:
    deadline = time.monotonic() + 2
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not reached")
        time.sleep(0.001)


def test_a_command_reaching_a_dead_actor_is_answered_with_a_typed_error() -> None:
    """Nothing may wait forever on a thread that will never answer."""
    guard = _guard_with_thread(alive=False)

    with pytest.raises(KVCRServiceError, match="closed"):
        guard._submit(_Command("release", (object(),)))


def test_a_close_beginning_mid_poll_still_blocks_the_promotion() -> None:
    """The one window the first gate cannot see: closing set while poll ran."""
    guard = _configurable_guard()
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"x")
        guard._phase = _Phase.PRIMARY
        guard._lease = _Lease(Mock(fileno=lambda: read_fd))
        guard._promote_for = Mock()
        poller = Mock()

        def poll_during_which_close_begins(_timeout):
            guard._closing = True
            return [(read_fd, select.POLLIN)]

        poller.poll.side_effect = poll_during_which_close_begins
        poller.register = Mock()
        import kvcr.guard as guard_module

        with_mocked = Mock(return_value=poller)
        original = guard_module.select.poll
        guard_module.select.poll = with_mocked
        try:
            guard._observe_holder()
        finally:
            guard_module.select.poll = original

        guard._promote_for.assert_not_called()
        assert guard._reserved is None
        assert guard._phase is _Phase.PRIMARY
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_a_claim_on_a_busy_pool_answers_busy_immediately() -> None:
    """The reservation is taken on the caller's thread, before anything queues."""
    guard = _configurable_guard()
    guard._reserved = _Phase.CLAIMING

    with pytest.raises(KVCRServiceError, match="busy"):
        guard._reserve_claim()


def test_a_claim_on_a_failed_pool_reports_the_failure() -> None:
    guard = _configurable_guard()
    failure = RuntimeError("promotion failed")
    guard._failure = failure

    with pytest.raises(RuntimeError) as raised:
        guard._reserve_claim()

    assert raised.value is failure


def test_serving_guard_notifies_service_and_fences_core_after_poll_failure(
    caplog,
) -> None:
    error = RuntimeError("poll failed")
    core = Mock()
    core.poll_completed.side_effect = error
    control = Mock()
    failure_callback = Mock()
    guard = _Guard(
        _TEST_SPEC,
        failure_callback,
        compatibility_digest=_TEST_DIGEST,
    )
    guard._control = control
    guard._configure(_TierConfig(16, None))
    guard._serving = True
    guard._core = core
    caplog.set_level(logging.ERROR, logger="kvcr.guard")

    guard._poll()

    assert guard._failure is error
    assert caplog.messages == ["KVCR Guard background polling failed"]
    assert caplog.records[0].exc_info is not None
    assert caplog.records[0].exc_info[1] is error
    core.close.assert_called_once_with()
    control.close.assert_not_called()
    failure_callback.assert_called_once_with(guard, error)


def test_a_serving_guard_that_cannot_fence_its_endpoint_says_so() -> None:
    """An address still bound to a dead Guard is the service's problem."""
    error = RuntimeError("poll failed")
    core = Mock()
    core.poll_completed.side_effect = error
    core.close.side_effect = OSError("close failed")
    failure_callback = Mock()
    guard = _Guard(
        _TEST_SPEC,
        failure_callback,
        compatibility_digest=_TEST_DIGEST,
    )
    guard._control = Mock()
    guard._configure(_TierConfig(16, None))
    guard._serving = True
    guard._core = core

    guard._poll()

    failure_callback.assert_called_once_with(guard, error)


def test_standby_guard_failure_releases_adopted_listener() -> None:
    """A standby that has failed must stop holding the pool's endpoint."""
    listener = socket.create_server(("127.0.0.1", 0))
    port = int(listener.getsockname()[1])
    control = ZmqPeerControlChannel.from_shared_listener(
        socket.socket(fileno=os.dup(listener.fileno()))
    )
    failure_callback = Mock()
    error = RuntimeError("journal poll failed")
    journal = Mock()
    journal.read_next.side_effect = error
    guard = _Guard(
        _TEST_SPEC,
        failure_callback,
        compatibility_digest=_TEST_DIGEST,
    )
    guard._control = control
    guard._configure(_TierConfig(16, None))
    guard._journal = journal
    guard._mirror = Mock()

    guard._poll()

    failure_callback.assert_called_once_with(guard, error)
    listener.close()
    # Only the Guard's duplicate could still be holding this now.
    with socket.socket() as replacement:
        replacement.bind(("127.0.0.1", port))


class _Journal:
    def __init__(self, pending=()) -> None:
        self.reset_called = False
        self.pending = list(pending)

    def reset(self) -> None:
        self.reset_called = True

    def read_next(self):
        return self.pending.pop(0) if self.pending else None

    def drain(self):
        while (record := self.read_next()) is not None:
            yield record


def test_guard_prepares_promotes_and_closes_in_ownership_order(
    tmp_path, monkeypatch
) -> None:
    first, second, g3_only = (
        BlockKey(b"first"),
        BlockKey(b"second"),
        BlockKey(b"g3-only"),
    )
    first_record = _BlockRecord(
        local_dram=_LocalDramResidency(0, _LocalDramState.READY),
        g3=_G3Residency(7),
    )
    second_record = _BlockRecord(
        local_dram=_LocalDramResidency(1, _LocalDramState.READY)
    )
    recovered = {
        first: first_record,
        second: second_record,
        g3_only: _BlockRecord(g3=_G3Residency(9)),
    }
    frames = list(
        (
            _RECORD_BLOCK,
            bytes(key),
            _RECOVERY_ENCODER.encode(_project_recovery_record(record)),
        )
        for key, record in recovered.items()
    )
    journal = _Journal(frames)
    closed = []
    attachment = _fake_attachment()
    attachment.close.side_effect = lambda: closed.append("attachment")
    local_dram = Mock()
    core = Mock(_local_dram=local_dram, _g3=None, _block_record_map={})
    core.close.side_effect = lambda: closed.append("core")
    constructed = []
    order = []

    # The seeding mechanics live on the core (adopt_recovery_records); this
    # test orders the Guard's calls around it, not what happens inside it.
    core.adopt_recovery_records.side_effect = lambda records: order.append(
        ("adopt", tuple(records))
    )
    core.start.side_effect = lambda: order.append("start")
    monkeypatch.setattr(
        "kvcr.guard.clear_recovery_snapshot",
        lambda _pool: order.append("clear"),
    )

    def new_core(config, bindings, backends):
        constructed.append((config, bindings, backends))
        return core

    attach = Mock(return_value=attachment)
    channel = Mock()
    channel.close.side_effect = lambda: closed.append("control")
    monkeypatch.setattr("kvcr.guard.KVCRPoolAttachment.attach", attach)
    monkeypatch.setattr("kvcr.guard.RecoveryJournal", Mock(return_value=journal))
    monkeypatch.setattr("kvcr.guard._KVCRCore", new_core)
    spec = _PAGE_SPEC
    g3_config = _G3Config(
        paths=(str(tmp_path / "g3.data"),),
        capacity_bytes_per_file=10 * _PAGE_STRIDE,
        backend="FILE",
        backend_options={},
    )
    guard = _Guard(spec, compatibility_digest=_TEST_DIGEST)
    # Driven directly, then the thread starts already busy: the actor blocks
    # on an empty mailbox when idle, so mutating around a sleeping thread
    # would race its wakeup instead of testing the ordering.
    guard._started = True
    guard._prepare()
    guard._adopt(channel, _TierConfig(_PAGE_STRIDE, g3_config))
    try:
        attach.assert_called_once_with(spec)
        assert journal.reset_called
        assert constructed == []
        core.start.assert_not_called()

        guard._promote()
        guard._thread.start()

        config, bindings, backends = constructed[0]
        prefix = "KVCR-Guard-"
        assert config.nixl_agent_name.startswith(prefix)
        uuid.UUID(config.nixl_agent_name.removeprefix(prefix))
        assert config.nixl_listen_port == 0
        assert bindings.framework_control is channel
        assert backends.local_dram == LocalDramInfo(1234 + 8192, 2 * _PAGE_STRIDE, 2)
        # A Guard opens no G3: it serves the G2 half and keeps the rest for the
        # primary that takes the pool back.
        assert backends.g3 is None
        assert set(guard._g3_records) == {first, g3_only}

        assert journal.pending == []
        assert order == [
            ("adopt", (first, second)),
            # Dropped before a slot moves: it describes rows about to be overwritten.
            "clear",
            "start",
        ]
        core.start.assert_called_once_with()
        _wait_until(lambda: core.poll_completed.call_count > 0)
    finally:
        guard.close()

    assert closed == ["core", "control", "attachment"]


def test_adopting_a_new_primary_canonicalises_the_records_it_keeps(
    tmp_path, monkeypatch
) -> None:
    """Kept rather than rebuilt, so what is kept must be what a replay gives."""
    core = Mock(_local_dram=Mock(), _g3=None, _block_record_map={})
    monkeypatch.setattr(
        "kvcr.guard.KVCRPoolAttachment.attach",
        Mock(return_value=_fake_attachment()),
    )
    monkeypatch.setattr("kvcr.guard.RecoveryJournal", Mock(return_value=_Journal(())))
    monkeypatch.setattr("kvcr.guard._KVCRCore", Mock(return_value=core))
    monkeypatch.setattr(
        "kvcr.guard.write_recovery_snapshot", lambda *args, **kwargs: None
    )
    guard = _Guard(_TEST_SPEC, compatibility_digest=_TEST_DIGEST)
    guard.start()
    guard._adopt(Mock(), _TierConfig(16, None))
    try:
        guard._promote()
        unfinished = BlockKey(b"unfinished")
        record = _BlockRecord(
            local_dram=_LocalDramResidency(0, _LocalDramState.FILLING),
            g3=_G3Residency(7),
        )
        record.g3.claim_count = 3
        core._block_record_map = {unfinished: record}

        guard._g3_records = {unfinished: _G3Residency(7)}

        guard._adopt(Mock(), _TierConfig(16, None))

        # The half-written G2 slot is dropped; the G3 half is carried whole.
        records = guard._mirror.take_records()
        assert records == {unfinished: _recovered_record(g3=7)}
    finally:
        guard.close()


@pytest.mark.parametrize("reader", ["poll", "release", "promote"])
def test_an_invalid_journal_costs_the_pool_on_every_path_that_reads_it(
    reader: str,
) -> None:
    """Which reader gets there first is a race; none of them may take the service."""
    guard = _configurable_guard()
    # Every one of these readers runs on a pool a primary has already claimed.
    guard._configured = _ConfiguredTier.derive(_TEST_SPEC, _TierConfig(16, None))
    guard._mirror = _RecoveryMirror()
    guard._attachment = Mock()
    guard._control = None
    journal = Mock()
    error = RecoveryJournalError("recovery journal is invalid")
    journal.read_next.side_effect = error
    journal.drain.side_effect = error
    guard._journal = journal
    written: list[object] = []
    guard._write_handback = lambda records, stride: written.append(records)
    served: list[dict] = []
    guard._serve = served.append
    reported: list[BaseException] = []
    guard._failure_callback = lambda _guard, failure: reported.append(failure)

    if reader == "poll":
        # Dropped rather than served: what is left of it is incomplete.
        assert guard._poll() is False
        assert guard._mirror is None
        assert served == []
    elif reader == "release":
        guard._release()
        # A standby that gave up recovery keeps nothing and serves nothing.
        assert guard._mirror is None
        assert served == []
    else:
        guard._promote()
        # Promotion still happens, with nothing in it: the endpoint has to answer.
        assert served == [{}]

    # A primary outrunning its Guard is not a Guard failure.
    assert reported == []
    assert guard._failure is None
    # Nothing is handed on: a prefix of what the primary published would name
    # slots it has since reused.
    assert written == []


def test_a_guard_with_nothing_left_to_recover_still_takes_the_endpoint() -> None:
    """The pool comes back cold, but its address still has to answer.

    A Guard that declined to serve would leave the dead primary's peers sending
    into an address nobody reads, waiting on a reply that is never sent.
    """
    guard = _configurable_guard()
    guard._journal = Mock()
    guard._mirror = None
    served: list[dict] = []
    guard._serve = served.append

    guard._promote()

    assert served == [{}]
    # A handover after this still needs somewhere to put the core's records.
    assert guard._mirror is not None


def test_a_second_promotion_takes_the_g3_half_from_the_generation_that_left_it(
    monkeypatch,
) -> None:
    """Two failovers: the second must not hand on the first's disk slots."""
    stale, fresh = BlockKey(b"stale"), BlockKey(b"fresh")

    def frames(key: BlockKey, slot: int) -> list[tuple[int, bytes, bytes]]:
        return [
            (
                _RECORD_BLOCK,
                bytes(key),
                _RECOVERY_ENCODER.encode(
                    _project_recovery_record(
                        _BlockRecord(
                            local_dram=_LocalDramResidency(0, _LocalDramState.READY),
                            g3=_G3Residency(slot),
                        )
                    )
                ),
            )
        ]

    def evicted(key: BlockKey) -> tuple[int, bytes, bytes]:
        return (
            _RECORD_BLOCK,
            bytes(key),
            _RECOVERY_ENCODER.encode(_project_recovery_record(_BlockRecord())),
        )

    journals = [_Journal(frames(stale, 1))]
    cores: list[Mock] = []

    def new_core(*_args, **_kwargs) -> Mock:
        # A promotion builds its own core, so the second starts empty.
        core = Mock(_local_dram=Mock(), _g3=None, _block_record_map={})
        cores.append(core)
        return core

    monkeypatch.setattr(
        "kvcr.guard.KVCRPoolAttachment.attach",
        Mock(return_value=_fake_attachment()),
    )
    monkeypatch.setattr("kvcr.guard.RecoveryJournal", Mock(side_effect=journals))
    monkeypatch.setattr("kvcr.guard._KVCRCore", new_core)
    monkeypatch.setattr(
        "kvcr.guard.write_recovery_snapshot", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "kvcr.guard.clear_recovery_snapshot", lambda *args, **kwargs: None
    )
    guard = _Guard(_TEST_SPEC, compatibility_digest=_TEST_DIGEST)
    guard.start()
    guard._adopt(Mock(), _TierConfig(16, None))
    try:
        guard._promote()
        assert set(guard._g3_records) == {stale}

        # A replacement takes the pool back, so the Guard stops holding it.
        cores[-1]._block_record_map = {}
        guard._adopt(Mock(), _TierConfig(16, None))
        assert guard._g3_records == {}

        # That replacement dies too, having dropped the old block and spilled
        # a different one into the slot it freed.
        guard._journal = _Journal([evicted(stale), *frames(fresh, 1)])
        guard._promote()

        assert set(guard._g3_records) == {fresh}
        assert guard._g3_records[fresh].slot == 1
    finally:
        guard.close()


def test_configuring_a_pool_is_what_fixes_the_tiers_it_will_accept(
    tmp_path,
) -> None:
    """A real Guard, so the geometry the claim is measured against is the pool's."""
    g3 = _G3Config(
        paths=(str(tmp_path / "g3.data"),),
        capacity_bytes_per_file=_PAGE_STRIDE,
        backend="FILE",
        backend_options={},
    )
    guard = _Guard(_TEST_SPEC, compatibility_digest=_TEST_DIGEST)

    # Unclaimed, so any shape is still available.
    guard._refuse_incompatible(_TierConfig(_PAGE_STRIDE, g3))
    guard._configure(_TierConfig(32, None))

    guard._refuse_incompatible(_TierConfig(32, None))
    with pytest.raises(RecoveryMirrorError, match="another tier configuration"):
        guard._refuse_incompatible(_TierConfig(_PAGE_STRIDE, g3))


def test_guard_closes_control_when_its_thread_does_not_start(monkeypatch) -> None:
    control = Mock()
    attachment = _fake_attachment()
    monkeypatch.setattr(
        "kvcr.guard.KVCRPoolAttachment.attach", Mock(return_value=attachment)
    )
    monkeypatch.setattr("kvcr.guard.RecoveryJournal", Mock())
    guard = _Guard(
        _TEST_SPEC,
        compatibility_digest=_TEST_DIGEST,
    )
    guard._control = control
    guard._configure(_TierConfig(16, None))
    guard._thread.start = Mock(side_effect=RuntimeError("thread start failed"))

    with pytest.raises(RuntimeError, match="thread start failed"):
        guard.start()
    guard.close()

    control.close.assert_called_once_with()
    # The pool was attached before the thread was asked for, so close gives it back.
    attachment.close.assert_called_once_with()


def test_guard_retains_resources_when_promoted_core_is_not_quiescent() -> None:
    control = Mock()
    attachment = Mock()
    core = Mock()
    close_error = RuntimeError("progress did not stop")
    core.close.side_effect = close_error
    core.is_quiescent.return_value = False
    guard = _Guard(
        _TEST_SPEC,
        compatibility_digest=_TEST_DIGEST,
    )
    guard._control = control
    guard._configure(_TierConfig(16, None))
    guard._core = core
    guard._attachment = attachment

    with pytest.raises(RuntimeError) as raised:
        guard.close()

    assert raised.value is close_error
    control.close.assert_not_called()
    attachment.close.assert_not_called()

    core.close.side_effect = None
    guard.close()
    control.close.assert_called_once_with()
    attachment.close.assert_called_once_with()


def test_guard_contains_core_close_failure_after_quiescence(caplog) -> None:
    control = Mock()
    attachment = Mock()
    core = Mock()
    error = RuntimeError("close failed")
    core.close.side_effect = error
    core.is_quiescent.return_value = True
    guard = _Guard(
        _TEST_SPEC,
        compatibility_digest=_TEST_DIGEST,
    )
    guard._control = control
    guard._configure(_TierConfig(16, None))
    guard._core = core
    guard._attachment = attachment
    caplog.set_level(logging.WARNING, logger="kvcr.guard")

    guard.close()

    assert caplog.messages == ["KVCR Guard core close failed after reaching quiescence"]
    assert caplog.records[0].exc_info is not None
    assert caplog.records[0].exc_info[1] is error
    control.close.assert_called_once_with()
    attachment.close.assert_called_once_with()


def _guard_with_thread(*, alive: bool) -> _Guard:
    """A Guard with only what _submit and close touch."""
    guard = object.__new__(_Guard)
    guard._thread = Mock()
    guard._thread.is_alive.return_value = alive
    guard._commands = queue.Queue()
    guard._phase_lock = threading.Lock()
    guard._closing = False
    guard._started = True
    guard._closed = False
    return guard


def _configurable_guard() -> _Guard:
    """A Guard past preparation, with nothing held and no thread running."""
    guard = object.__new__(_Guard)
    guard._spec = _TEST_SPEC
    guard._compatibility_digest = _TEST_DIGEST
    guard._failure = None
    guard._configured = None
    guard._g3_records = {}
    guard._serving = False
    guard._resumable = False
    guard._core = None
    guard._mirror = None
    guard._phase_lock = threading.Lock()
    guard._phase = _Phase.IDLE
    guard._reserved = None
    guard._closing = False
    guard._lease = None
    guard._refusing = lambda: False
    guard._pool_index = 0
    guard._listener = None
    guard._bind = None
    return guard


def test_the_same_g3_paths_in_another_order_are_another_configuration() -> None:
    """A slot names its file by position, so reordering renames every slot."""
    guard = _configurable_guard()
    guard._configured = _ConfiguredTier.derive(
        _PAGE_SPEC,
        _TierConfig(_PAGE_STRIDE, _G3Config(("/a", "/b"), _PAGE_STRIDE, "MOCK", {})),
    )

    with pytest.raises(RecoveryMirrorError, match="another tier configuration"):
        guard._refuse_incompatible(
            _TierConfig(_PAGE_STRIDE, _G3Config(("/b", "/a"), _PAGE_STRIDE, "MOCK", {}))
        )


def test_a_failure_once_the_pool_has_changed_hands_is_the_services() -> None:
    """A hand-back that fails cannot be reported as a refused claim."""
    reported: list[BaseException] = []
    guard = _configurable_guard()
    guard._control = None
    guard._journal = Mock()
    guard._attachment = None
    guard._failure_callback = lambda _guard, error: reported.append(error)
    guard._configured = _ConfiguredTier.derive(_TEST_SPEC, _TierConfig(16, None))
    guard._serving = True
    guard._mirror = _RecoveryMirror()
    guard._core = Mock(_block_record_map={})
    failure = OSError("no space left on device")
    guard._hand_back = Mock(side_effect=failure)
    control = Mock()

    with pytest.raises(OSError, match="no space left"):
        guard._adopt(control, _TierConfig(16, None))

    assert reported == [failure]
    assert guard._failure is failure
    control.close.assert_called_once_with()


@pytest.mark.parametrize("refused_by", ["geometry", "handback"])
def test_a_refused_claim_leaves_the_pool_choosable_for_the_next_one(
    monkeypatch,
    refused_by: str,
) -> None:
    """The service survives these claims, so the pool they failed on must too."""
    reported: list[BaseException] = []
    guard = _configurable_guard()
    guard._attachment = Mock()
    guard._journal = Mock()
    guard._failure_callback = lambda _guard, error: reported.append(error)
    guard._hand_back = Mock(side_effect=AssertionError("stood down for a bad claim"))
    control = Mock()
    if refused_by == "geometry":
        expected: type[Exception] = ValueError
        tier_config = _TierConfig(_TEST_SPEC.mapping_bytes, None)
        handback = Mock(return_value=_RecoveryMirror())
    else:
        expected = RecoveryJournalError
        tier_config = _TierConfig(16, None)
        handback = Mock(side_effect=RecoveryJournalError("written for other terms"))
    monkeypatch.setattr("kvcr.guard.read_handback", handback)

    with pytest.raises(expected):
        guard._adopt(control, tier_config)

    assert reported == []
    assert guard._failure is None
    assert guard._serving is False
    guard._hand_back.assert_not_called()
    control.close.assert_called_once_with()
    # Nothing was chosen, so a corrected claim can still have this pool.
    assert guard._configured is None
    guard._refuse_incompatible(_TierConfig(32, None))


def test_a_handback_the_filesystem_refuses_leaves_a_cold_pool() -> None:
    """ENOSPC at the pool tail is the ring-full precedent, not a dead service."""
    guard = _configurable_guard()
    guard._mirror = _RecoveryMirror()
    guard._core = Mock(
        _block_record_map={
            BlockKey(b"warm"): _BlockRecord(
                local_dram=_LocalDramResidency(0, _LocalDramState.READY)
            )
        }
    )
    guard._serving = True
    guard._write_handback = Mock(
        side_effect=OSError(errno.ENOSPC, "No space left on device")
    )

    guard._hand_back(16)

    assert guard._serving is False
    assert guard._core is None
    assert guard._mirror is None


def test_a_dropped_handback_still_leaves_the_new_lease_mirrored() -> None:
    """ENOSPC at adopt-time handback costs one generation, not every one after."""
    guard = _configurable_guard()
    guard._control = None
    guard._failure_callback = lambda *_args: None
    guard._configured = _ConfiguredTier.derive(_TEST_SPEC, _TierConfig(16, None))
    guard._mirror = _RecoveryMirror()
    guard._core = Mock(
        _block_record_map={
            BlockKey(b"warm"): _BlockRecord(
                local_dram=_LocalDramResidency(0, _LocalDramState.READY)
            )
        }
    )
    guard._serving = True
    guard._journal = _Journal()
    guard._write_handback = Mock(side_effect=OSError(errno.ENOSPC, "No space left"))

    guard._adopt(Mock(), _TierConfig(16, None))

    # The claimant was told cold; the new lease is still mirrored, so this
    # primary's deposits survive its own death.
    assert guard._mirror is not None
    # And the grant is retractable: the Guard it stood down can resume.
    assert guard._resumable is True
    guard._journal.pending = [
        (_RECORD_BLOCK, b"fresh", _RECOVERY_ENCODER.encode(_RecoveryBlock(g2=1)))
    ]
    guard._poll()
    assert BlockKey(b"fresh") in guard._mirror._records


def test_a_grant_that_never_arrived_resumes_the_guard_it_stood_down() -> None:
    """An aborted grant re-promotes after a hand-back; otherwise it releases."""
    guard = _configurable_guard()
    guard._resumable = True
    guard._mirror = _RecoveryMirror()
    outcomes: list[str] = []
    guard._promote = lambda: outcomes.append("promote")
    guard._release = lambda: outcomes.append("release")

    lease = _Lease(Mock())
    guard._lease = lease
    guard._abort(lease)
    assert outcomes == ["promote"]
    lease.liveness.close.assert_called_once_with()
    assert guard._lease is None
    assert guard._phase is _Phase.STANDBY

    guard._resumable = False
    stale = _Lease(Mock())
    guard._lease = stale
    guard._abort(stale)
    assert outcomes == ["promote", "release"]
    assert guard._phase is _Phase.IDLE


def test_a_handback_with_an_unexpected_storage_error_fails() -> None:
    guard = _configurable_guard()
    guard._mirror = _RecoveryMirror()
    guard._core = Mock(
        _block_record_map={
            BlockKey(b"warm"): _BlockRecord(
                local_dram=_LocalDramResidency(0, _LocalDramState.READY)
            )
        }
    )
    guard._serving = True
    error = OSError(errno.EIO, "I/O error")
    guard._write_handback = Mock(side_effect=error)

    with pytest.raises(OSError) as raised:
        guard._hand_back(16)

    assert raised.value is error


def test_a_release_the_filesystem_refuses_still_releases() -> None:
    guard = _configurable_guard()
    control = Mock()
    guard._control = control
    guard._configured = _ConfiguredTier.derive(_TEST_SPEC, _TierConfig(16, None))
    guard._mirror = _RecoveryMirror()
    guard._journal = _Journal()
    guard._write_handback = Mock(
        side_effect=OSError(errno.ENOSPC, "No space left on device")
    )

    guard._release()

    assert guard._mirror is None
    control.close.assert_called_once()


def test_a_clean_release_hands_its_cache_on_instead_of_keeping_it() -> None:
    """A mirror the next primary is never told about is a mirror that lies."""
    guard = _configurable_guard()
    guard._control = Mock()
    guard._configured = _ConfiguredTier.derive(_TEST_SPEC, _TierConfig(16, None))
    guard._mirror = _RecoveryMirror()
    guard._mirror.apply(
        _RECORD_BLOCK, b"published", _RECOVERY_ENCODER.encode(_RecoveryBlock(g2=0))
    )
    # One frame the Guard had not polled yet when the release arrived.
    tail = (_RECORD_BLOCK, b"tail", _RECOVERY_ENCODER.encode(_RecoveryBlock(g2=1)))
    guard._journal = _Journal(pending=[tail])
    guard._write_handback = Mock()

    guard._release()

    assert guard._mirror is None
    records, row_stride = guard._write_handback.call_args.args
    assert set(records) == {BlockKey(b"published"), BlockKey(b"tail")}
    assert row_stride == 16
