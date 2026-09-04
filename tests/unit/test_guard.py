# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Private Guard lifecycle tests."""

import concurrent.futures
import errno
import logging
import os
import queue
import select
import socket
import uuid
from contextlib import nullcontext
from unittest.mock import Mock

import msgspec
import pytest
from _kvcr_test_utils import _recovered_record, _wait_until

from kvcr.config import LocalDramOptions
from kvcr.control_channels import KVCRServiceError, ZmqPeerControlChannel
from kvcr.core import _BlockRecord
from kvcr.guard import (
    _Command,
    _Guard,
    _Phase,
    _PoolLease,
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
_PAGE_SPEC = msgspec.structs.replace(_TEST_SPEC, mapping_bytes=8192 + 2 * _PAGE_STRIDE)


def _fake_attachment() -> Mock:
    """A stand-in with the pool-tail surface a Guard reaches for."""
    attachment = Mock(
        address=1234,
        data_address=1234 + _TEST_SPEC.journal_bytes,
        _spec=_TEST_SPEC,
    )
    attachment.mapped_snapshot.return_value = nullcontext(None)
    return attachment


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


def _frame(key: BlockKey, record: _BlockRecord) -> tuple[int, bytes, bytes]:
    """One journal frame, exactly as a primary would publish it."""
    payload = _RECOVERY_ENCODER.encode(_project_recovery_record(record))
    return (_RECORD_BLOCK, bytes(key), payload)


def _give_serving_core(guard: _Guard) -> Mock:
    """A serving core still holding one READY G2 block."""
    record = _BlockRecord(local_dram=_LocalDramResidency(0, _LocalDramState.READY))
    core = Mock(_block_record_map={BlockKey(b"warm"): record})
    guard._core = core
    guard._serving = True
    return core


def _configurable_guard() -> _Guard:
    """A Guard past preparation, with nothing held and no thread running."""
    guard = _Guard(_TEST_SPEC, compatibility_digest=_TEST_DIGEST)
    guard._phase = _Phase.IDLE
    return guard


def test_a_wait_timeout_racing_the_answer_returns_the_answer() -> None:
    """A slow command's success must never be reported back as a timeout."""
    guard = object.__new__(_Guard)
    guard._commands = queue.Queue()

    # The wait times out; the actor answers before the re-check.
    command = _Command("claim")
    command.future = Mock(
        result=Mock(side_effect=[concurrent.futures.TimeoutError, "the-grant"]),
        done=Mock(return_value=True),
        exception=Mock(return_value=None),
    )
    assert guard._submit(command) == "the-grant"

    # The handler's own TimeoutError is the answer and must propagate.
    error = TimeoutError("hand-back timed out")
    command = _Command("release")
    command.future = Mock(
        result=Mock(side_effect=error),
        done=Mock(return_value=True),
        exception=Mock(return_value=error),
    )
    with pytest.raises(TimeoutError, match="hand-back timed out"):
        guard._submit(command)


def test_a_command_reaching_a_dead_actor_is_answered_with_a_typed_error() -> None:
    """Nothing may wait forever on a thread that will never answer."""
    guard = object.__new__(_Guard)
    guard._thread = Mock()
    guard._thread.is_alive.return_value = False
    guard._commands = queue.Queue()

    with pytest.raises(KVCRServiceError, match="closed"):
        guard._submit(_Command("release", (object(),)))


def test_a_close_beginning_mid_poll_still_blocks_the_promotion(monkeypatch) -> None:
    """The one window the first gate cannot see: closing set while poll ran."""
    guard = _configurable_guard()
    guard._phase = _Phase.PRIMARY
    guard._pool_lease.current = Mock(fileno=lambda: 7)
    guard._promote_for = Mock()
    poller = Mock()

    def poll_during_which_close_begins(_timeout):
        guard._closing = True
        return [(7, select.POLLIN)]

    poller.poll.side_effect = poll_during_which_close_begins
    monkeypatch.setattr("kvcr.guard.select.poll", Mock(return_value=poller))

    guard._observe_holder()

    guard._promote_for.assert_not_called()
    assert guard._reserved is None
    assert guard._phase is _Phase.PRIMARY


def test_a_serving_guard_reports_a_poll_failure_and_fences_its_core(caplog) -> None:
    """A mirror failure retires its journal, fences the core, and is reported."""
    error = RecoveryMirrorError("recovery record is malformed")
    core = Mock()
    core.poll_completed.side_effect = error
    core.close.side_effect = OSError("close failed")
    control = Mock()
    journal = Mock()
    failure_callback = Mock()
    guard = _Guard(_TEST_SPEC, failure_callback, compatibility_digest=_TEST_DIGEST)
    guard._control = control
    guard._configure(_TierConfig([(16, "")], None))
    guard._recovery._journal = journal
    guard._serving = True
    guard._core = core
    caplog.set_level(logging.ERROR, logger="kvcr.guard")

    guard._poll()

    assert guard._failure is error
    assert caplog.messages == [
        "KVCR Guard background polling failed",
        "Failed to fence a failed KVCR Guard endpoint",
    ]
    assert caplog.records[0].exc_info is not None
    assert caplog.records[0].exc_info[1] is error
    core.close.assert_called_once_with()
    control.close.assert_not_called()
    journal.invalidate.assert_called_once_with()
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
    guard = _Guard(_TEST_SPEC, failure_callback, compatibility_digest=_TEST_DIGEST)
    guard._control = control
    guard._configure(_TierConfig([(16, "")], None))
    guard._recovery._journal = journal
    guard._recovery.mirror = Mock()

    guard._poll()

    failure_callback.assert_called_once_with(guard, error)
    listener.close()
    # Only the Guard's duplicate could still be holding this now.
    with socket.socket() as replacement:
        replacement.bind(("127.0.0.1", port))


def test_guard_lives_out_adopt_promote_and_readopt_in_ownership_order(
    tmp_path, monkeypatch
) -> None:
    """The Guard's whole life: each stage hands the next exactly what it left."""
    first, second, g3_only, fresh = (
        BlockKey(b"first"),
        BlockKey(b"second"),
        BlockKey(b"g3-only"),
        BlockKey(b"fresh"),
    )
    journal = _Journal(
        [
            _frame(first, _recovered_record(g2=0, g3=7)),
            _frame(second, _recovered_record(g2=1)),
            _frame(g3_only, _recovered_record(g3=9)),
        ]
    )
    closed: list[str] = []
    order: list[object] = []
    constructed: list[tuple] = []
    cores: list[Mock] = []
    channels: list[Mock] = []
    attachment = _fake_attachment()
    attachment.close.side_effect = lambda: closed.append("attachment")

    # The seeding mechanics live on the core (adopt_recovery_records); this
    # test orders the Guard's calls around it, not what happens inside it.
    def new_core(config, bindings, backends) -> Mock:
        constructed.append((config, bindings, backends))
        core = Mock(_local_dram=Mock(), _g3=None, _block_record_map={})
        core.adopt_recovery_records.side_effect = lambda records: order.append(
            ("adopt", tuple(records))
        )
        core.start.side_effect = lambda: order.append("start")
        label = f"core{len(cores) + 1}"
        core.close.side_effect = lambda: closed.append(label)
        cores.append(core)
        return core

    def new_channel() -> Mock:
        channel = Mock()
        label = f"control{len(channels) + 1}"
        channel.close.side_effect = lambda: closed.append(label)
        channels.append(channel)
        return channel

    attach = Mock(return_value=attachment)
    monkeypatch.setattr("kvcr.guard.KVCRPoolAttachment.attach", attach)
    monkeypatch.setattr("kvcr.guard.RecoveryJournal", Mock(return_value=journal))
    monkeypatch.setattr("kvcr.guard._KVCRCore", new_core)
    attachment.release_snapshot_region.side_effect = lambda: order.append("clear")
    g3_config = _G3Config(
        paths=(str(tmp_path / "g3.data"),),
        capacity_bytes_per_file=10 * _PAGE_STRIDE,
        backend="FILE",
        backend_options={},
    )
    tier = _TierConfig([(_PAGE_STRIDE, "")], g3_config, "REMOTE")
    guard = _Guard(_PAGE_SPEC, compatibility_digest=_TEST_DIGEST)
    # Driven directly, then the thread starts already busy: the actor blocks
    # on an empty mailbox when idle, so mutating around a sleeping thread
    # would race its wakeup instead of testing the ordering.
    guard._started = True
    guard._recovery.prepare()
    # Unclaimed, so any tier shape is still available; the first claim fixes it.
    guard._refuse_incompatible(_TierConfig([(16, "")], None))
    guard._adopt(new_channel(), tier)
    try:
        attach.assert_called_once_with(_PAGE_SPEC)
        assert journal.reset_called
        # Adoption only grants; a core exists once a promotion needs one.
        assert constructed == []
        with pytest.raises(RecoveryMirrorError, match="another tier configuration"):
            guard._refuse_incompatible(_TierConfig([(16, "")], None))

        promoted_records = guard._recovery.mirror._records
        guard._promote()

        config, bindings, backends = constructed[0]
        prefix = "KVCR-Guard-"
        assert config.nixl_agent_name.startswith(prefix)
        uuid.UUID(config.nixl_agent_name.removeprefix(prefix))
        assert config.nixl_listen_port == 0
        assert bindings.framework_control is channels[0]
        assert backends.local_dram == LocalDramOptions(
            1234 + 8192,
            [(2 * _PAGE_STRIDE, "")],
            "REMOTE",
        )
        assert backends.remote_fw_dram.backend == "REMOTE"
        # A Guard opens no G3: it serves the G2 half and keeps the rest for the
        # primary that takes the pool back.
        assert backends.g3 is None
        assert journal.pending == []
        assert order == [
            ("adopt", (first, second)),
            # Dropped before a slot moves: it describes rows about to be overwritten.
            "clear",
            "start",
        ]
        assert cores[0].adopt_recovery_records.call_args.args[0] is promoted_records
        assert set(guard._recovery._g3_records) == {first, g3_only}

        # A replacement claims the pool. What is kept must be what a replay
        # gives: the half-written G2 slot is dropped, and the G3 halves are
        # carried whole -- g3_only no longer names any live record, so a
        # rebuild from the core's map could not produce it.
        retained_g3 = guard._recovery._g3_records[first]
        records = {
            first: _BlockRecord(
                local_dram=_LocalDramResidency(0, _LocalDramState.FILLING),
                g3=_G3Residency(7),
            ),
            second: _recovered_record(g2=1),
        }
        cores[0]._block_record_map = records
        write_handback = Mock()
        guard._recovery._write_handback = write_handback
        guard._adopt(new_channel(), tier)
        assert write_handback.call_args.args[0] is records
        assert guard._recovery.mirror._records is records
        assert guard._recovery.mirror._records[first].g3 is retained_g3
        assert guard._recovery.mirror._records == {
            first: _recovered_record(g3=7),
            second: _recovered_record(g2=1),
            g3_only: _recovered_record(g3=9),
        }
        assert guard._recovery._g3_records == {}

        # The replacement dies too, having dropped the first generation's
        # blocks and spilled a different one into the slot that freed.
        guard._recovery._journal = _Journal(
            [
                _frame(first, _BlockRecord()),
                _frame(g3_only, _BlockRecord()),
                _frame(fresh, _recovered_record(g2=0, g3=7)),
            ]
        )
        guard._promote()

        assert set(guard._recovery._g3_records) == {fresh}
        assert guard._recovery._g3_records[fresh].slot == 7
        assert order[3:] == [("adopt", (second, fresh)), "clear", "start"]

        guard._thread.start()
        _wait_until(lambda: cores[-1].poll_completed.call_count > 0, timeout=2)
    finally:
        guard.close()

    # Each generation's core went before its channel; the pool went last.
    assert closed == ["core1", "control1", "core2", "control2", "attachment"]


@pytest.mark.parametrize("reader", ["poll", "release", "promote", "cold-promote"])
def test_a_pool_that_lost_its_recovery_stays_claimable_on_every_path(
    reader: str,
) -> None:
    """Which reader finds the invalid journal is a race; none may take the service."""
    guard = _configurable_guard()
    # Every one of these readers runs on a pool a primary has already claimed.
    guard._configured = _TierConfig([(16, "")], None)
    guard._recovery.mirror = _RecoveryMirror()
    guard._recovery.attachment = Mock()
    guard._control = None
    journal = Mock()
    error = RecoveryJournalError("recovery journal is invalid")
    journal.read_next.side_effect = error
    journal.drain.side_effect = error
    guard._recovery._journal = journal
    written: list[object] = []
    guard._recovery._write_handback = lambda records, stride: written.append(records)
    served: list[dict] = []
    guard._serve = served.append
    reported: list[BaseException] = []
    guard._failure_callback = lambda _guard, failure: reported.append(failure)

    if reader == "poll":
        # Dropped rather than served: what is left of it is incomplete.
        assert guard._poll() is False
        assert guard._recovery.mirror is None
        assert served == []
    elif reader == "release":
        guard._release()
        # A standby that gave up recovery keeps nothing and serves nothing.
        assert guard._recovery.mirror is None
        assert served == []
    elif reader == "promote":
        guard._promote()
        # Promotion still happens, with nothing in it: the endpoint has to answer.
        # A Guard that declined to serve would leave the dead primary's peers
        # sending into an address nobody reads, waiting on a reply never sent.
        assert served == [{}]
    else:
        # No mirror at all -- recovery was never there, or already given up.
        guard._recovery.mirror = None
        guard._promote()
        assert served == [{}]
        # A handover after this still needs somewhere to put the core's records.
        assert guard._recovery.mirror is not None

    # A primary outrunning its Guard is not a Guard failure.
    assert reported == []
    assert guard._failure is None
    # Nothing is handed on: a prefix of what the primary published would name
    # slots it has since reused.
    assert written == []


def test_the_same_g3_paths_in_another_order_are_another_configuration() -> None:
    """A slot names its file by position, so reordering renames every slot."""
    guard = _configurable_guard()
    guard._configured = _TierConfig(
        [(_PAGE_STRIDE, "")],
        _G3Config(("/a", "/b"), _PAGE_STRIDE, "MOCK", {}),
    )
    with pytest.raises(RecoveryMirrorError, match="another tier configuration"):
        guard._refuse_incompatible(
            _TierConfig(
                [(_PAGE_STRIDE, "")],
                _G3Config(("/b", "/a"), _PAGE_STRIDE, "MOCK", {}),
            )
        )


def test_guard_closes_control_when_its_thread_does_not_start(monkeypatch) -> None:
    """A start that never got a thread must not keep the pool or its endpoint."""
    control = Mock()
    attachment = _fake_attachment()
    monkeypatch.setattr(
        "kvcr.guard.KVCRPoolAttachment.attach", Mock(return_value=attachment)
    )
    monkeypatch.setattr("kvcr.guard.RecoveryJournal", Mock())
    guard = _Guard(_TEST_SPEC, compatibility_digest=_TEST_DIGEST)
    guard._control = control
    guard._configure(_TierConfig([(16, "")], None))
    guard._thread.start = Mock(side_effect=RuntimeError("thread start failed"))

    with pytest.raises(RuntimeError, match="thread start failed"):
        guard.start()
    guard.close()

    control.close.assert_called_once_with()
    # The pool was attached before the thread was asked for, so close gives it back.
    attachment.close.assert_called_once_with()


def test_pool_lease_closes_listener_after_holder_failure_and_retries_holder() -> None:
    """One failed resource must not skip the other or lose the failed holder."""
    holder_error = RuntimeError("holder close failed")
    holder = Mock(close=Mock(side_effect=[holder_error, None]))
    listener = Mock()
    lease = _PoolLease(0)
    lease.current = holder
    lease.listener = listener
    lease.bind_address = ("127.0.0.1", 1234)

    with pytest.raises(RuntimeError) as raised:
        lease.close()

    assert raised.value is holder_error
    assert lease.current is holder
    assert lease.listener is None
    assert lease.bind_address is None
    listener.close.assert_called_once_with()

    lease.close()
    assert lease.current is None
    assert holder.close.call_count == 2
    listener.close.assert_called_once_with()


def test_recovery_close_error_stays_first_while_lease_cleanup_continues() -> None:
    """Recovery stays on failed unmap, then drops before later cleanup."""
    attachment_error = RuntimeError("attachment close failed")
    attachment = Mock(close=Mock(side_effect=[attachment_error, None]))
    holder = Mock(close=Mock(side_effect=RuntimeError("holder close failed")))
    owner = Mock()
    guard = _Guard(_TEST_SPEC, compatibility_digest=_TEST_DIGEST, owner=owner)
    guard._recovery.attachment = attachment
    mirror = guard._recovery.mirror = _RecoveryMirror()
    g3_records = guard._recovery._g3_records = {BlockKey(b"g3"): _G3Residency(0)}
    guard._pool_lease.current = holder

    with pytest.raises(RuntimeError) as first:
        guard.close()

    assert first.value is attachment_error
    assert guard._recovery.attachment is attachment
    assert guard._recovery.mirror is mirror
    assert guard._recovery._g3_records is g3_records
    holder.close.assert_called_once_with()
    owner.close.assert_not_called()

    with pytest.raises(RuntimeError, match="holder close failed"):
        guard.close()

    assert guard._recovery.mirror is None
    assert guard._recovery._g3_records == {}


def test_a_close_refused_by_a_moving_core_retains_the_pool_until_quiescent(
    caplog,
) -> None:
    """Close keeps everything while the core moves; once quiescent it contains."""
    control = Mock()
    attachment = Mock()
    core = Mock()
    error = RuntimeError("close failed")
    core.close.side_effect = error
    core.is_quiescent.return_value = False
    guard = _Guard(_TEST_SPEC, compatibility_digest=_TEST_DIGEST)
    guard._control = control
    guard._configure(_TierConfig([(16, "")], None))
    guard._core = core
    guard._recovery.attachment = attachment
    caplog.set_level(logging.WARNING, logger="kvcr.guard")

    # Still moving bytes: nothing may be unmapped under it.
    with pytest.raises(RuntimeError) as raised:
        guard.close()
    assert raised.value is error
    control.close.assert_not_called()
    attachment.close.assert_not_called()

    # Quiescent: the same failure is contained and everything goes back.
    core.is_quiescent.return_value = True
    guard.close()
    assert caplog.messages == ["KVCR Guard core close failed after reaching quiescence"]
    assert caplog.records[0].exc_info is not None
    assert caplog.records[0].exc_info[1] is error
    control.close.assert_called_once_with()
    attachment.close.assert_called_once_with()


@pytest.mark.parametrize("refused_by", ["geometry", "handback", "hand-over"])
def test_only_a_claim_refused_before_the_pool_moves_costs_nothing(
    monkeypatch,
    refused_by: str,
) -> None:
    """Refusals before the pool moves leave it choosable; failures after are fatal."""
    reported: list[BaseException] = []
    guard = _configurable_guard()
    guard._recovery.attachment = Mock()
    guard._recovery._journal = Mock()
    guard._failure_callback = lambda _guard, error: reported.append(error)
    control = Mock()

    if refused_by == "hand-over":
        # A hand-back that fails cannot be reported as a refused claim.
        guard._configured = _TierConfig([(16, "")], None)
        guard._serving = True
        guard._recovery.mirror = _RecoveryMirror()
        guard._core = Mock(_block_record_map={})
        failure = OSError("no space left on device")
        guard._hand_back = Mock(side_effect=failure)

        with pytest.raises(OSError, match="no space left"):
            guard._adopt(control, _TierConfig([(16, "")], None))

        assert reported == [failure]
        assert guard._failure is failure
    else:
        guard._hand_back = Mock(
            side_effect=AssertionError("stood down for a bad claim")
        )
        if refused_by == "geometry":
            expected: type[Exception] = ValueError
            tier_config = _TierConfig([(_TEST_SPEC.mapping_bytes, "")], None)
            handback = Mock(return_value=_RecoveryMirror())
        else:
            expected = RecoveryJournalError
            tier_config = _TierConfig([(16, "")], None)
            handback = Mock(side_effect=RecoveryJournalError("written for other terms"))
        monkeypatch.setattr("kvcr.guard.read_handback", handback)

        with pytest.raises(expected):
            guard._adopt(control, tier_config)

        # The service survives these claims, so the pool they failed on must too.
        assert reported == []
        assert guard._failure is None
        assert guard._serving is False
        guard._hand_back.assert_not_called()
        # Nothing was chosen, so a corrected claim can still have this pool.
        assert guard._configured is None
        guard._refuse_incompatible(_TierConfig([(32, "")], None))
    control.close.assert_called_once_with()


def test_a_handback_with_an_unexpected_storage_error_fails() -> None:
    """Only capacity errors (ENOSPC/EDQUOT) are survivable at the handback writer."""
    guard = _configurable_guard()
    guard._recovery.mirror = _RecoveryMirror()
    _give_serving_core(guard)
    error = OSError(errno.EIO, "I/O error")
    guard._recovery._write_handback = Mock(side_effect=error)

    with pytest.raises(OSError) as raised:
        guard._hand_back(16)

    assert raised.value is error


def test_a_handback_without_a_mirror_does_not_close_the_core() -> None:
    """Validate both halves before stopping a core that recovery cannot accept."""
    guard = _configurable_guard()
    core = _give_serving_core(guard)

    with pytest.raises(RecoveryMirrorError, match="no state to hand back"):
        guard._hand_back(16)

    core.close.assert_not_called()


def test_a_handback_the_filesystem_refuses_leaves_a_cold_pool() -> None:
    """ENOSPC at the pool tail drops the mirror with the handback it refused."""
    guard = _configurable_guard()
    guard._recovery.mirror = _RecoveryMirror()
    _give_serving_core(guard)
    guard._recovery._write_handback = Mock(
        side_effect=OSError(errno.ENOSPC, "No space left on device")
    )

    guard._hand_back(16)

    assert guard._serving is False
    assert guard._core is None
    assert guard._recovery.mirror is None


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EDQUOT])
def test_a_dropped_handback_still_leaves_the_new_lease_mirrored(code: int) -> None:
    """A capacity-refused handback goes cold, one generation only, never fatal."""
    guard = _configurable_guard()
    guard._control = None
    guard._failure_callback = lambda *_args: None
    guard._configured = _TierConfig([(16, "")], None)
    guard._recovery.mirror = _RecoveryMirror()
    _give_serving_core(guard)
    guard._recovery._journal = _Journal()
    guard._recovery._write_handback = Mock(side_effect=OSError(code, "No space left"))

    guard._adopt(Mock(), _TierConfig([(16, "")], None))

    # The pool went cold, not fatal: the Guard stood down and dropped the core.
    assert guard._serving is False
    assert guard._core is None
    # The claimant was told cold; the new lease is still mirrored, so this
    # primary's deposits survive its own death.
    assert guard._recovery.mirror is not None
    # And the grant is retractable: the Guard it stood down can resume.
    assert guard._resumable is True
    guard._recovery._journal.pending = [
        (_RECORD_BLOCK, b"fresh", _RECOVERY_ENCODER.encode(_RecoveryBlock(g2=1)))
    ]
    guard._poll()
    assert BlockKey(b"fresh") in guard._recovery.mirror._records


def test_a_grant_that_never_arrived_resumes_the_guard_it_stood_down() -> None:
    """An aborted grant re-promotes after a hand-back; otherwise it releases."""
    guard = _configurable_guard()
    guard._resumable = True
    guard._recovery.mirror = _RecoveryMirror()
    outcomes: list[str] = []
    guard._promote = lambda: outcomes.append("promote")
    guard._release = lambda: outcomes.append("release")

    lease = Mock()
    guard._pool_lease.current = lease
    guard._abort(lease)
    assert outcomes == ["promote"]
    lease.close.assert_called_once_with()
    assert guard._pool_lease.current is None
    assert guard._phase is _Phase.STANDBY

    guard._resumable = False
    stale = Mock()
    guard._pool_lease.current = stale
    guard._abort(stale)
    assert outcomes == ["promote", "release"]
    assert guard._phase is _Phase.IDLE


@pytest.mark.parametrize("mode", ["accepts", "refuses", "serving"])
def test_a_release_drops_its_mirror_after_handing_back_what_it_can(
    mode: str,
) -> None:
    """A mirror the next primary is never told about is a mirror that lies."""
    guard = _configurable_guard()
    control = Mock()
    guard._control = control
    guard._configured = _TierConfig([(16, "")], None)
    guard._recovery.mirror = _RecoveryMirror()
    if mode == "serving":
        _give_serving_core(guard)
    else:
        guard._recovery.mirror.apply(
            *_frame(BlockKey(b"published"), _recovered_record(g2=0))
        )
        tail = _frame(BlockKey(b"tail"), _recovered_record(g2=1))
        guard._recovery._journal = _Journal(pending=[tail])
    guard._recovery._write_handback = Mock(
        side_effect=OSError(errno.ENOSPC, "No space left")
        if mode == "refuses"
        else None
    )

    guard._release()

    assert guard._recovery.mirror is None
    control.close.assert_called_once_with()
    if mode == "accepts":
        records, pool_layout = guard._recovery._write_handback.call_args.args
        assert set(records) == {BlockKey(b"published"), BlockKey(b"tail")}
        assert pool_layout == [(16, "")]
    elif mode == "serving":
        assert guard._serving is False
        assert guard._core is None
