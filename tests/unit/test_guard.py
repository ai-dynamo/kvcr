# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Private Guard lifecycle tests."""

import logging
import os
import queue
import socket
import time
import uuid
from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from kvcr.config import LocalDramInfo
from kvcr.control_channels import ZmqPeerControlChannel
from kvcr.core import _BlockRecord
from kvcr.guard import (
    _Command,
    _ConfiguredTier,
    _Guard,
    _with_g3,
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


def test_a_command_the_lifecycle_thread_cannot_take_is_refused() -> None:
    """Queueing it would wait on a thread that will never answer."""
    guard = _guard_with_thread(alive=False)

    with pytest.raises(RuntimeError, match="lifecycle thread stopped"):
        guard._submit(_Command("close"))
    assert guard._commands.empty()


def test_an_adopt_the_guard_never_ran_closes_the_channel_it_was_given() -> None:
    """Nothing else owns the pool listener's duplicate until _adopt takes it."""
    guard = _guard_with_thread(alive=False)
    control = Mock()

    with pytest.raises(RuntimeError, match="lifecycle thread stopped"):
        guard.adopt(control, _TierConfig(16, None))

    control.close.assert_called_once_with()


def test_a_command_that_does_not_finish_in_time_is_a_timeout() -> None:
    guard = _guard_with_thread(alive=True)
    command = _Command("close")

    with pytest.raises(TimeoutError, match="close timed out"):
        guard._submit(command, 0)

    # Queued regardless: a deadline here cancels nothing.
    assert guard._commands.get_nowait() is command


def test_guard_waits_for_state_changing_commands_without_a_deadline(
    monkeypatch,
) -> None:
    submitted: list[tuple[str, float | None]] = []
    monkeypatch.setattr(
        _Guard,
        "_submit",
        lambda self, command, timeout=None: submitted.append(
            (command.operation, timeout)
        ),
    )
    guard = object.__new__(_Guard)

    guard.promote_after_death()
    guard.adopt(Mock(), _TierConfig(16, None))

    # Giving up does not cancel them; the thread carries on regardless.
    assert submitted == [("promote", None), ("adopt", None)]


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

    def adopt(tier):
        def adopted(records):
            order.append((f"adopt:{tier}", tuple(records)))

        return adopted

    local_dram.adopt_recovery_slots.side_effect = adopt("g2")
    core._policy.on_ingest.side_effect = lambda *_: order.append("ingest")
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
    spec = _TEST_SPEC
    g3_config = _G3Config(
        paths=(str(tmp_path / "g3.data"),),
        capacity_bytes_per_file=40960,
        backend="FILE",
        backend_options={},
    )
    guard = _Guard(spec, compatibility_digest=_TEST_DIGEST)
    guard.start()
    guard.adopt(channel, _TierConfig(16, g3_config))
    try:
        attach.assert_called_once_with(spec)
        assert journal.reset_called
        assert constructed == []
        core.start.assert_not_called()

        guard.promote_after_death()

        config, bindings, backends = constructed[0]
        prefix = "KVCR-Guard-"
        assert config.nixl_agent_name.startswith(prefix)
        uuid.UUID(config.nixl_agent_name.removeprefix(prefix))
        assert config.nixl_listen_port == 0
        assert bindings.framework_control is channel
        assert backends.local_dram == LocalDramInfo(1234 + 8192, 32, 2)
        # A Guard opens no G3: it serves the G2 half and keeps the rest for the
        # primary that takes the pool back.
        assert backends.g3 is None
        assert set(guard._g3_records) == {first, g3_only}

        assert journal.pending == []
        assert order == [
            ("adopt:g2", (first, second)),
            "ingest",
            "ingest",
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
    guard.adopt(Mock(), _TierConfig(16, None))
    try:
        guard.promote_after_death()
        unfinished = BlockKey(b"unfinished")
        record = _BlockRecord(
            local_dram=_LocalDramResidency(0, _LocalDramState.FILLING),
            g3=_G3Residency(7),
        )
        record.g3.claim_count = 3
        core._block_record_map = {unfinished: record}

        guard.adopt(Mock(), _TierConfig(16, None))

        kept = guard._mirror.take_records()[unfinished]
        assert kept.local_dram is None
        assert kept.g3 is not None
        assert kept.g3.slot == 7
        assert kept.g3.claim_count == 0
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
    guard.adopt(Mock(), _TierConfig(16, None))
    try:
        guard.promote_after_death()
        assert set(guard._g3_records) == {stale}

        # A replacement takes the pool back, so the Guard stops holding it.
        cores[-1]._block_record_map = {}
        guard.adopt(Mock(), _TierConfig(16, None))
        assert guard._g3_records == {}

        # That replacement dies too, having dropped the old block and spilled
        # a different one into the slot it freed.
        guard._journal = _Journal([evicted(stale), *frames(fresh, 1)])
        guard.promote_after_death()

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
        capacity_bytes_per_file=4096,
        backend="FILE",
        backend_options={},
    )
    guard = _Guard(_TEST_SPEC, compatibility_digest=_TEST_DIGEST)

    # Unclaimed, so any shape is still available.
    guard._refuse_incompatible(_TierConfig(16, g3))
    guard._configure(_TierConfig(32, None))

    guard._refuse_incompatible(_TierConfig(32, None))
    with pytest.raises(RecoveryMirrorError, match="another tier configuration"):
        guard._refuse_incompatible(_TierConfig(16, g3))


def test_merging_the_g3_half_back_keeps_a_dual_resident_block_whole() -> None:
    """A block hot enough for DRAM and already spilled must keep both."""
    key = BlockKey(b"both")
    records = {
        key: _BlockRecord(local_dram=_LocalDramResidency(0, _LocalDramState.READY))
    }

    _with_g3(records, {key: _G3Residency(7)})

    assert records[key].g3 is not None
    assert records[key].g3.slot == 7
    assert records[key].local_dram is not None


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
    """A Guard with only what _submit touches."""
    guard = object.__new__(_Guard)
    guard._thread = Mock()
    guard._thread.is_alive.return_value = alive
    guard._commands = queue.Queue()
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
    guard._core = None
    guard._mirror = None
    return guard


def test_a_refused_stride_leaves_no_configuration_behind() -> None:
    """A rejected claim must not be able to make the next one skip the check."""
    guard = _configurable_guard()
    impossible = _TierConfig(_TEST_SPEC.mapping_bytes, None)

    for _ in range(2):
        with pytest.raises(ValueError, match="one complete KV row"):
            guard._configure(impossible)
        assert guard._configured is None


def test_the_same_g3_paths_in_another_order_are_another_configuration() -> None:
    """A slot names its file by position, so reordering renames every slot."""
    guard = _configurable_guard()
    guard._configured = _ConfiguredTier.derive(
        _TEST_SPEC, _TierConfig(16, _G3Config(("/a", "/b"), 4096, "MOCK", {}))
    )

    with pytest.raises(RecoveryMirrorError, match="another tier configuration"):
        guard._refuse_incompatible(
            _TierConfig(16, _G3Config(("/b", "/a"), 4096, "MOCK", {}))
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
    guard._core = Mock(_block_record_map={BlockKey(b"warm"): object()})
    guard._serving = True
    guard._write_handback = Mock(side_effect=OSError(28, "No space left on device"))

    guard._hand_back(16)

    assert guard._serving is False
    assert guard._core is None
    assert guard._mirror is None


def test_a_release_the_filesystem_refuses_still_releases() -> None:
    guard = _configurable_guard()
    control = Mock()
    guard._control = control
    guard._configured = _ConfiguredTier.derive(_TEST_SPEC, _TierConfig(16, None))
    guard._mirror = _RecoveryMirror()
    guard._journal = _Journal()
    guard._write_handback = Mock(side_effect=OSError(28, "No space left on device"))

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
