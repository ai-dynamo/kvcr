# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Private journal-backed Guard for one service-owned pool."""

import concurrent.futures
import enum
import errno
import logging
import os
import queue
import select
import socket
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import Any

from .api import KVCRBindings
from .config import KVCRBackendConfigs, KVCRConfig, LocalDramInfo
from .control_channels import KVCRServiceError, ZmqPeerControlChannel
from .core import _BlockRecord, _KVCRCore
from .guard_protocol import PidfdLiveness, _TierConfig
from .local_disk import _G3Residency
from .memory import (
    KVCRPoolAttachment,
    KVCRPoolSpec,
    _compute_pool_geometry,
    _KVCRPoolOwner,
)
from .recovery_journal import (
    RecoveryJournal,
    RecoveryJournalError,
    RecoveryMirrorError,
    _recovery_frames,
    _RecoveryMirror,
    canonical_pool_terms,
    read_handback,
    write_recovery_snapshot,
)
from .types import BlockKey

logger = logging.getLogger(__name__)

# TODO: Wake the mirror on publication rather than polling. A standby still
# drains at roughly a fifth of the rate a primary can publish.
_POLL_SECONDS, _POLL_BATCH = 0.001, 64
_RECOVERY_CAPACITY_ERRORS = (errno.ENOSPC, errno.EDQUOT)

# Lease identity is the pidfd object itself: a release acts only on THIS
# object, so nothing stale (reused pid, retried release) can touch a newer lease.
_Lease = PidfdLiveness


class _PoolLease:
    """One pool's holder identity and persistent control listener."""

    def __init__(self, guard_index: int) -> None:
        self._guard_index = guard_index
        self.current: _Lease | None = None
        self.listener: socket.socket | None = None
        # As the claimant asked: getsockname() is numeric and rejects aliases.
        self.bind_address: tuple[str, int] | None = None

    def clear_if_current(self, lease: _Lease) -> None:
        if self.current is lease:
            self.current = None

    def poll_pidfd(self, lease: _Lease) -> int | None:
        """Return raw pidfd events for interpretation after lifecycle validation."""
        poller = select.poll()
        poller.register(lease.fileno(), select.POLLIN)
        events = poller.poll(0)
        return events[0][1] if events else None

    def bind(self, address: tuple[str, int]) -> tuple[socket.socket, bool]:
        """Bind once and return the listener plus whether this call created it."""
        if self.listener is not None:
            if self.bind_address != address:
                raise KVCRServiceError(
                    f"KVCR Guard {self._guard_index} answers on "
                    f"{self.bind_address[0]}:{self.bind_address[1]} and cannot be "
                    "moved to "
                    f"{address[0]}:{address[1]}"
                )
            return self.listener, False
        try:
            listener = socket.create_server(address)
        except OSError as error:
            raise KVCRServiceError(
                f"KVCR Guard {self._guard_index} control listener "
                f"{address[0]}:{address[1]} is unavailable: {error}"
            ) from error
        self.listener = listener
        self.bind_address = address
        return listener, True

    def unbind(self) -> BaseException | None:
        """Forget a first-claim listener and report any failure to close it."""
        listener, self.listener = self.listener, None
        self.bind_address = None
        if listener is None:
            return None
        try:
            listener.close()
        except BaseException as error:  # noqa: BLE001 - the Guard decides severity
            return error
        return None

    def close(self) -> None:
        """Close holder then listener, retaining failures for a later retry."""
        failure: BaseException | None = None
        lease = self.current
        if lease is not None:
            try:
                lease.close()
            except BaseException as error:  # noqa: BLE001 - first failure wins
                failure = error
            else:
                self.clear_if_current(lease)
        # Unlike unbind(), a failure keeps the listener so close can retry it.
        if self.listener is not None:
            try:
                self.listener.close()
            except BaseException as error:  # noqa: BLE001 - holder failure still wins
                failure = failure or error
            else:
                self.listener = None
                self.bind_address = None
        if failure is not None:
            raise failure


def _without_g3(
    records: dict[BlockKey, _BlockRecord],
) -> dict[BlockKey, _BlockRecord]:
    """Strip the half a Guard cannot serve, in place.

    In place because the table can hold a tier's worth of blocks and a second
    copy of it is exactly what take_records avoids.
    """
    for key in [key for key, record in records.items() if record.local_dram is None]:
        del records[key]
    for record in records.values():
        record.g3 = None
    return records


def _with_g3(
    records: dict[BlockKey, _BlockRecord],
    g3_records: Mapping[BlockKey, _G3Residency],
) -> dict[BlockKey, _BlockRecord]:
    """Put the kept half back, in place, for the primary that will open G3.

    A block the Guard evicted keeps its G3 residency and loses its G2 one, which
    is what a returning primary has to be told.
    """
    for key, g3 in g3_records.items():
        record = records.get(key)
        if record is None:
            records[key] = _BlockRecord(g3=g3)
        else:
            record.g3 = g3
    return records


class _RecoveryState:
    """One pool's attachment, journal, and mutable recovery ownership."""

    def __init__(self, spec: KVCRPoolSpec, compatibility_digest: str) -> None:
        self._spec = spec
        self._compatibility_digest = compatibility_digest
        self.attachment: KVCRPoolAttachment | None = None
        self._journal: RecoveryJournal | None = None
        self.mirror: _RecoveryMirror | None = None
        # Recovered half a Guard cannot serve; kept for the next primary, which can.
        self._g3_records: dict[BlockKey, _G3Residency] = {}

    def prepare(self) -> None:
        """Attach everything that depends only on the pool."""
        self.attachment = KVCRPoolAttachment.attach(self._spec)
        self._journal = RecoveryJournal(self.attachment)

    def recover(self, row_stride: int) -> _RecoveryMirror:
        """Return held recovery or read the prior handback under this stride."""
        if self.mirror is not None:
            return self.mirror
        return read_handback(self.attachment, self._compatibility_digest, row_stride)

    def start_primary(self) -> None:
        """Arm recovery for the accepted primary and reset its journal."""
        if self.mirror is None:
            self.mirror = _RecoveryMirror()
        self._journal.reset()

    def poll(self) -> bool:
        """Mirror one bounded journal batch; True if more may remain."""
        mirror = self.mirror
        if mirror is None:
            return False
        try:
            for _ in range(_POLL_BATCH):
                frame = self._journal.read_next()
                if frame is None:
                    return False
                mirror.apply(*frame)
            return True
        except RecoveryJournalError as error:
            self._drop_recovery(error)
        return False

    def invalidate_journal(self) -> None:
        if self._journal is not None:
            with suppress(Exception):
                self._journal.invalidate()

    def take_for_promotion(self) -> dict[BlockKey, _BlockRecord]:
        """Drain and transfer recovered records, leaving a fresh mirror."""
        records: dict[BlockKey, _BlockRecord] = {}
        mirror = self.mirror
        if mirror is None:
            logger.warning("KVCR pool has no recovered state to promote")
        else:
            try:
                # The primary is gone, so there is no more journal traffic coming.
                for frame in self._journal.drain():
                    mirror.apply(*frame)
                records = mirror.take_records()
            except RecoveryJournalError as error:
                self._drop_recovery(error)
        # A handover still needs somewhere to put the core's eventual records.
        self.mirror = _RecoveryMirror()
        return records

    def prepare_to_serve(
        self, records: dict[BlockKey, _BlockRecord]
    ) -> dict[BlockKey, _BlockRecord]:
        """Keep the unserved G3 half and transfer the G2 half in place."""
        self._g3_records = {
            key: record.g3 for key, record in records.items() if record.g3 is not None
        }
        return _without_g3(records)

    def local_dram_info(self, effective_bytes: int, rows: int) -> LocalDramInfo:
        return LocalDramInfo(self.attachment.data_address, effective_bytes, rows)

    def release_snapshot_region(self) -> None:
        self.attachment.release_snapshot_region()

    # TODO: Verify the G3 files a handover describes. Nothing holds them while
    # this Guard serves -- the exclusive lock lives with the tier, and a Guard
    # opens no G3 -- so a second KVCR on the same paths goes unnoticed and the
    # replacement serves whatever is in the slots. Refusing two pools that name
    # the same paths is the cheap first step; it does not cover a second
    # service, or a KVCR using G3 with no pool at all.
    def hand_back(self, records: dict[BlockKey, _BlockRecord], row_stride: int) -> None:
        """Write and mirror a closed core's map under the stride it served."""
        mirror = self.mirror
        records = _with_g3(records, self._g3_records)
        try:
            self._write_handback(records, row_stride)
        except OSError as error:
            if error.errno not in _RECOVERY_CAPACITY_ERRORS:
                raise
            # A truncated snapshot and a retained mirror must never diverge.
            self._drop_recovery(error)
        else:
            mirror.adopt(records)
        self._g3_records = {}

    def release(self, row_stride: int) -> None:
        """Write the current primary's journal tail, then drop its mirror."""
        mirror = self.mirror
        try:
            while (frame := self._journal.read_next()) is not None:
                mirror.apply(*frame)
            self._write_handback(mirror.take_records(), row_stride)
        except RecoveryJournalError as error:
            self._drop_recovery(error)
        except OSError as error:
            if error.errno not in _RECOVERY_CAPACITY_ERRORS:
                raise
            self._drop_recovery(error)
        self.mirror = None

    def _drop_recovery(self, error: RecoveryJournalError | OSError) -> None:
        logger.warning(
            "KVCR pool recovery disabled; claimable but cold if this primary dies: %s",
            error,
        )
        self.mirror = None

    def _write_handback(
        self,
        records: Mapping[BlockKey, _BlockRecord],
        row_stride: int,
    ) -> None:
        write_recovery_snapshot(
            self.attachment,
            canonical_pool_terms(self._compatibility_digest, row_stride, self._spec),
            _recovery_frames(records),
        )

    def close(self) -> None:
        """Close the attachment, retaining it if close must be retried."""
        if self.attachment is not None:
            self.attachment.close()
            self.attachment = None
            self._journal = None
            self.mirror = None
            self._g3_records = {}


class _Command:
    """One request on the pool's mailbox, and the future its answer arrives on."""

    def __init__(self, operation: str, args: tuple[Any, ...] = ()) -> None:
        self.operation, self.args = operation, args
        self.future: concurrent.futures.Future[Any] = concurrent.futures.Future()


class _Phase(enum.Enum):
    """First six: stable resting states. Last three: transient reservations taken
    before a command is queued, so a conflicting command is refused immediately.
    """

    UNCONFIGURED = enum.auto()
    IDLE = enum.auto()
    STANDBY = enum.auto()
    PRIMARY = enum.auto()
    FAILED = enum.auto()
    CLOSED = enum.auto()
    CLAIMING = enum.auto()
    RELEASING = enum.auto()
    PROMOTING = enum.auto()


class _Guard:
    """One pool's lifecycle actor: owns the pool file, control endpoint, holder
    pidfd, and recovery state, alive as long as the service owns the pool.
    Outlives every primary: a claim is reported to it, not what creates it.
    """

    def __init__(
        self,
        spec: KVCRPoolSpec,
        failure_callback: Callable[..., None] | None = None,
        *,
        compatibility_digest: str,
        guard_index: int = 0,
        owner: _KVCRPoolOwner | None = None,
        refusing: Callable[[], bool] = lambda: False,
    ) -> None:
        self._spec = spec
        self._guard_index = guard_index
        # Owned here, not by the registry: one thread owns one pool, so a
        # claim needs no lock -- the mailbox is the reservation.
        self._owner = owner
        self._refusing = refusing
        self._pool_lease = _PoolLease(guard_index)
        # Owned by the current primary.
        self._control: ZmqPeerControlChannel | None = None
        self._configured: _TierConfig | None = None
        self._recovery = _RecoveryState(spec, compatibility_digest)
        self._core = None
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._ops = {
            "claim": self._claim,
            "release": self._stand_down,
            "abort": self._abort,
            "close": self._close,
        }
        self._thread = threading.Thread(
            target=self._run, name=f"kvcr-guard-{spec.pool_id}", daemon=True
        )
        self._started = self._closed = self._serving = self._resumable = False
        # Guards phase, reservation, lease, and failure; never held across
        # blocking work. Callers reserve transitions; only the actor runs them.
        self._phase_lock = threading.Lock()
        self._phase = _Phase.UNCONFIGURED
        self._reserved: _Phase | None = None
        self._closing = False
        self._failure: BaseException | None = None
        self._failure_callback = failure_callback or (lambda guard, error: None)

    def _fail(self, error: BaseException) -> None:
        with self._phase_lock:
            self._failure = error
            self._phase = _Phase.FAILED
        self._escalate(error)

    def start(self) -> None:
        """Attach the pool and begin the lifecycle thread, before any claim."""
        if self._started:
            raise RuntimeError("Guard preparation was already attempted")
        self._started = True
        # Attaching is not thread-affine; done here so a failure surfaces
        # directly, with nothing to tear down.
        self._recovery.prepare()
        self._thread.start()

    def claim(
        self, liveness: PidfdLiveness, tier_config: _TierConfig, bind: tuple[str, int]
    ) -> "tuple[KVCRPoolSpec, int, _Lease]":
        """Give this pool to a primary, with the endpoint its Guard answers on.
        Reserved on the requesting thread: a mid-transition pool answers busy
        immediately instead of queueing the claimant.
        """
        self._reserve_claim()
        return self._submit(_Command("claim", (liveness, tier_config, bind)))

    def release(self, lease: "_Lease") -> None:
        """End a lease. The pool keeps its Guard, and the Guard its records."""
        self._end_lease(lease, "release")

    def abort_grant(self, lease: "_Lease") -> None:
        """Roll back a lease its claimant declared it never served.
        Sent only after the claimant stopped local access, so a stood-down
        serving Guard may resume; otherwise an ordinary release.
        """
        self._end_lease(lease, "abort")

    def _end_lease(self, lease: "_Lease", operation: str) -> None:
        """Queue a release or abort; a stale lease is a no-op. Absorbed during
        shutdown too: the close path owns every resource this touches.
        """
        with self._phase_lock:
            if self._closing or self._pool_lease.current is not lease:
                return
            if self._reserved is not None:
                if self._reserved is _Phase.PROMOTING:
                    # The death of this same lease got here first; it wins.
                    return
                raise KVCRServiceError(f"KVCR Guard {self._guard_index} is busy")
            self._reserved = _Phase.RELEASING
        self._submit(_Command(operation, (lease,)))

    def close(self) -> None:
        self.begin_close()
        if self.finish_close(time.monotonic() + 30.0):
            raise TimeoutError("KVCR Guard lifecycle thread did not stop")

    def begin_close(self) -> None:
        """Stop taking work and queue the teardown without waiting: a wedged
        pool must not keep its neighbours from being told to close.
        """
        with self._phase_lock:
            already = self._closing
            self._closing = True
        if not self._started or not self._thread.is_alive():
            # Inline, and retried on a later call if it raised the first time.
            if not self._closed:
                self._close_resources()
                self._closed = True
            return
        if not already:
            self._commands.put(_Command("close"))

    def finish_close(self, deadline: float) -> bool:
        """Wait out a begun close; True if the actor is wedged past the
        deadline. A close that finished but failed raises its reason here.
        """
        if not self._started or self._thread.ident is None:
            # The thread never ran; begin_close already closed inline.
            return False
        # Always join: _closed is set before the actor's final drain, so the
        # flag alone could report done while the thread still holds the mailbox.
        self._thread.join(max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive():
            return True
        if not self._closed and self._failure is not None:
            raise self._failure
        return False

    def _reserve_claim(self) -> None:
        """Take the transition slot for a claim, or refuse right now."""
        with self._phase_lock:
            if self._closing:
                raise KVCRServiceError("KVCR pool registry is closed")
            if self._failure is not None:
                raise self._failure
            if self._reserved is not None:
                raise KVCRServiceError(f"KVCR Guard {self._guard_index} is busy")
            if self._phase is _Phase.PRIMARY:
                lease = self._pool_lease.current
                if self._pool_lease.poll_pidfd(lease) is None:
                    raise KVCRServiceError(
                        f"KVCR Guard {self._guard_index} is held by another worker"
                    )
                # Dead but not yet promoted: the actor is the sole authority.
                raise KVCRServiceError(f"KVCR Guard {self._guard_index} is busy")
            if self._phase in (_Phase.FAILED, _Phase.CLOSED):
                raise KVCRServiceError("KVCR pool registry is closed")
            self._reserved = _Phase.CLAIMING

    def _submit(self, command: _Command) -> Any:
        """Run one command on the actor thread and wait for its outcome.
        The unbounded wait is guarded: the reservation refuses later callers,
        and an actor exiting mid-wait answers with a typed error.
        """
        self._commands.put(command)
        while True:
            try:
                # futures.TimeoutError != the builtin before 3.11; 3.10 is the floor.
                return command.future.result(timeout=0.1)
            except concurrent.futures.TimeoutError as error:
                if command.future.done():
                    if command.future.exception() is error:
                        # The handler raised this TimeoutError; it is the answer.
                        raise
                    # Answered in the gap after the timeout; re-read, or a
                    # committed lease would be reported as a timeout.
                    continue
                if not self._thread.is_alive():
                    if command.future.done():
                        # Completed in the gap between the wait and this check.
                        continue
                    # The exit drain answers everything queued; this covers a racer.
                    raise KVCRServiceError("KVCR pool registry is closed") from None

    def _run(self) -> None:
        try:
            self._serve_commands()
        except BaseException as error:  # noqa: BLE001 - the loop itself failed
            self._record_background_failure(error)
        finally:
            # However this thread ended, nothing queued may wait forever.
            with self._phase_lock:
                self._closing = True
            with suppress(queue.Empty):
                while True:
                    self._commands.get_nowait().future.set_exception(
                        KVCRServiceError("KVCR pool registry is closed")
                    )

    def _serve_commands(self) -> None:
        draining = False
        while True:
            try:
                if draining:
                    # More was waiting last time round, so do not go back to
                    # sleep on it -- but still take a command if one arrived.
                    command = self._commands.get_nowait()
                else:
                    # Nothing publishes into a pool no primary holds, so an
                    # unclaimed Guard waits rather than waking on an empty journal.
                    busy = self._serving or self._control is not None
                    timeout = _POLL_SECONDS if busy and not self._failure else None
                    command = self._commands.get(timeout=timeout)
            except queue.Empty:
                self._observe_holder()
                draining = self._poll()
                continue
            failed: BaseException | None = None
            try:
                result = self._ops[command.operation](*command.args)
            except BaseException as error:  # noqa: BLE001 - returned to caller
                failed = error
                with self._phase_lock:
                    # The one rollback point: success commits a stable phase itself.
                    self._reserved = None
                command.future.set_exception(error)
            else:
                with self._phase_lock:
                    self._reserved = None
                command.future.set_result(result)
            if command.operation == "close":
                if failed is not None:
                    # A failed close is final; finish_close raises the reason.
                    with self._phase_lock:
                        self._failure = failed
                        self._phase = _Phase.FAILED
                return

    def _observe_holder(self) -> None:
        """Notice the current primary dying. Polled between commands on the
        actor thread, so a death and every command are totally ordered.
        """
        with self._phase_lock:
            if (
                self._phase is not _Phase.PRIMARY
                or self._reserved is not None
                or self._closing
            ):
                return
            lease = self._pool_lease.current
        flags = self._pool_lease.poll_pidfd(lease)
        if flags is None:
            return
        with self._phase_lock:
            if (
                self._pool_lease.current is not lease
                or self._reserved is not None
                or self._closing
            ):
                # Interpret only while this lease is current and no transition began.
                return
            self._reserved = _Phase.PROMOTING
        try:
            if not flags & select.POLLIN:
                # The process may still be alive: promoting could seat a
                # second server over a live mapping.
                raise OSError(f"pidfd poll returned without POLLIN: {flags:#x}")
            self._promote_for(lease)
        except BaseException as error:  # noqa: BLE001 - service-fatal
            self._fail(error)
        finally:
            with self._phase_lock:
                self._reserved = None

    def _claim(
        self, liveness: PidfdLiveness, tier_config: _TierConfig, bind: tuple[str, int]
    ) -> "tuple[KVCRPoolSpec, int, _Lease]":
        """Give the pool to a primary, and take up the endpoint it named.
        All fallible work runs before the lease exists, and commit and refusal
        share one lock: a lease is never half-granted.
        """
        listener, bound_here = self._pool_lease.bind(bind)
        granted_fd = -1
        try:
            granted_fd = os.dup(listener.fileno())
            # from_shared_listener detaches its argument; give it a duplicate.
            duplicate = socket.socket(fileno=os.dup(listener.fileno()))
            try:
                control = ZmqPeerControlChannel.from_shared_listener(duplicate)
            except BaseException:
                duplicate.close()
                raise
            self._adopt(control, tier_config)
            with self._phase_lock:
                if not self._closing and not self._refusing():
                    self._pool_lease.current = liveness
                    self._phase = _Phase.PRIMARY
                    return self._spec, granted_fd, liveness
            # Refused at the commit: a closing service must not grant a pool.
            # Everything adopted goes back as a release would have put it.
            self._release()
            with self._phase_lock:
                self._phase = _Phase.IDLE
            raise KVCRServiceError("KVCR pool registry is closed")
        except BaseException as error:
            if granted_fd >= 0:
                with suppress(OSError):
                    os.close(granted_fd)
            if bound_here:
                # Keeping an address this claim chose would refuse a retry as a move.
                unbind_error = self._pool_lease.unbind()
                if unbind_error is not None:
                    self._escalate(unbind_error)
            if isinstance(error, (ValueError, RecoveryMirrorError)):
                raise KVCRServiceError(str(error)) from error
            raise

    def _stand_down(self, lease: "_Lease") -> None:
        """End this lease, keeping what it left for the next primary.
        Staleness was decided at submission; here the lease is current.
        """
        try:
            self._release()
        except BaseException as error:
            # Escalated before the pool is exposed: a partial handback may remain.
            self._fail(error)
            raise
        finally:
            lease.close()
            with self._phase_lock:
                self._pool_lease.clear_if_current(lease)
        with self._phase_lock:
            self._phase = _Phase.IDLE

    def _abort(self, lease: "_Lease") -> None:
        """Undo a grant its claimant declared unserved: resume, or release.
        Resume is safe: the claimant stopped local access for good, the mirror
        holds the handback, the journal is reset -- promotion serves it back.
        """
        if self._resumable and not self._serving and self._recovery.mirror is not None:
            try:
                self._promote_for(lease)
            except BaseException as error:
                self._fail(error)
                raise
            return
        self._stand_down(lease)

    def _promote_for(self, lease: "_Lease") -> None:
        """Take the pool over from the primary that just died."""
        try:
            self._promote()
        finally:
            lease.close()
            with self._phase_lock:
                self._pool_lease.clear_if_current(lease)
        with self._phase_lock:
            self._phase = _Phase.STANDBY

    def _escalate(self, error: BaseException) -> None:
        logger.critical("KVCR Guard %d failed", self._guard_index)
        try:
            self._failure_callback(self, error)
        except BaseException:  # noqa: BLE001 - retain the original failure
            logger.exception("Failed to notify KVCR-Service of Guard failure")

    def _close(self) -> None:
        self._close_resources()
        self._closed = True
        with self._phase_lock:
            self._phase = _Phase.CLOSED

    def _adopt(self, control: ZmqPeerControlChannel, tier_config: _TierConfig) -> None:
        """Take up a new primary, handing over whatever the last one left.
        A serving Guard hands back first, so it stops answering before the
        replacement starts; held records are kept, not re-read from the region.
        """
        try:
            # Everything that can refuse this claim runs before anything
            # moves: refusing later leaves the pool with no reader.
            if self._failure is not None:
                raise self._failure
            self._refuse_incompatible(tier_config)
            served_under = self._configured.row_stride if self._configured else 0
            # The prior handback is this lease's baseline. Read now, under
            # the claim's stride: refusing at promotion stops the service,
            # and a claimant dying in between takes everything with it.
            recovered = self._recovery.recover(tier_config.row_stride)
            # Last, once nothing left can refuse this claim: a pool whose handback
            # would not replay has not chosen anything, and a corrected claim can
            # still have it.
            self._configure(tier_config)
            self._recovery.mirror = recovered
        except BaseException:
            control.close()
            raise

        try:
            if self._serving:
                self._hand_back(served_under)
                self._resumable = True
            # A refused handback is cold for the new lease, not unmirrored.
            self._recovery.start_primary()
            # The old channel is the last reference to the prior primary's listener.
            if self._control is not None:
                self._control.close()
        except BaseException as error:
            # The pool has changed hands and nothing here can put it back, so
            # this stopped being something a claimant could be told about.
            control.close()
            self._record_background_failure(error)
            raise
        self._control = control

    def _release(self) -> None:
        """Give up the current primary. The mirror is written out and dropped:
        kept, it would map keys to slots a later claimant allocated over.
        """
        self._resumable = False
        if self._failure is not None:
            raise self._failure
        if self._serving:
            self._hand_back(self._configured.row_stride)
            # Re-adopt lets start_primary() retain or replace the mirror.
            self._recovery.mirror = None
        elif self._recovery.mirror is not None:
            self._recovery.release(self._configured.row_stride)
        if self._control is not None:
            self._control.close()
            self._control = None

    def _refuse_incompatible(self, tier_config: _TierConfig) -> None:
        """Refuse tiers other than the ones this pool was claimed with.
        The first claim fixes configuration for the service's lifetime: the
        bytes stay, and a changed stride or G3 path order misnames every slot.
        """
        if self._configured is not None and self._configured != tier_config:
            raise RecoveryMirrorError(
                "KVCR pool was claimed with another tier configuration"
            )

    def _configure(self, tier_config: _TierConfig) -> None:
        """Take up this primary's tiers: the geometry check runs before the
        assignment, so a bad configuration leaves the old one intact.
        """
        _compute_pool_geometry(self._spec.data_bytes, tier_config.row_stride)
        self._configured = tier_config

    def _poll(self) -> bool:
        """Mirror what is waiting; True if more remains. The batch bounds a
        command's wait, not how much drains.
        """
        if self._failure is not None:
            return False
        try:
            if self._serving:
                self._core.poll_completed()
            else:
                return self._recovery.poll()
        except RecoveryMirrorError as error:
            self._recovery.invalidate_journal()
            self._record_background_failure(error)
        except BaseException as error:  # noqa: BLE001 - promotion/close observes it
            self._record_background_failure(error)
        return False

    def _record_background_failure(self, error: BaseException) -> None:
        with self._phase_lock:
            self._failure = error
            self._phase = _Phase.FAILED
        logger.exception("KVCR Guard background polling failed")
        # Shut what this answered on so the address stops accepting what nothing
        # will read. Best effort; the process exit closes what this could not.
        try:
            if self._serving and self._core is not None:
                self._core.close()
            elif self._control is not None:
                self._control.close()
        except BaseException:  # noqa: BLE001 - retain the original failure
            logger.exception("Failed to fence a failed KVCR Guard endpoint")
        try:
            self._failure_callback(self, error)
        except BaseException:  # noqa: BLE001 - retain the original failure
            logger.exception("Failed to notify KVCR-Service of Guard failure")

    def _promote(self) -> None:
        """Take the pool over from the dead primary, warm if anything survived."""
        self._resumable = False
        if self._failure is not None:
            raise self._failure
        self._serve(self._recovery.take_for_promotion())

    def _serve(self, records: dict[BlockKey, _BlockRecord]) -> None:
        """Answer on this pool's endpoint, with whatever came back from it.

        Serving nothing is still serving: answering refuses peers that staying
        bound would leave hanging. G2 only, no G3: that half is kept whole for
        the replacement. A new NIXL agent name keeps peers off the dead one's.
        """
        records = self._recovery.prepare_to_serve(records)

        def reject_pin(keys: object) -> int:
            raise RuntimeError("Guard has no framework-owned memory")

        effective_bytes, rows = _compute_pool_geometry(
            self._spec.data_bytes, self._configured.row_stride
        )
        dram = self._recovery.local_dram_info(effective_bytes, rows)
        core = _KVCRCore(
            KVCRConfig(
                nixl_agent_name=f"KVCR-Guard-{uuid.uuid4()}",
                inventory_report_interval_ms=0,
                nixl_listen_port=0,
            ),
            KVCRBindings(
                reject_pin,
                lambda: (),
                lambda _handle: False,
                framework_control=self._control,
            ),
            KVCRBackendConfigs(local_dram=dram, g3=None),
        )
        self._core = core
        core.adopt_recovery_records(records)
        # A previous handover describes slots this Guard is about to move, and it is
        # already in the mirror. Leaving it would map keys to overwritten bytes.
        self._recovery.release_snapshot_region()
        core.start()
        self._serving = True

    def _hand_back(self, row_stride: int) -> None:
        """Stop serving, leaving this pool's state where the next primary looks.
        The core closes first: the Guard stops answering, and region and
        records both come from the map close leaves behind.
        """
        core = self._core
        if core is None or self._recovery.mirror is None:
            raise RecoveryMirrorError("a serving Guard has no state to hand back")
        core.close()
        self._recovery.hand_back(core._block_record_map, row_stride)
        self._core = None
        self._serving = False

    def _close_resources(self) -> None:
        try:
            if self._core is not None:
                self._core.close()
        except BaseException:
            if not self._core.is_quiescent():
                # Still moving bytes, so nothing below runs: unmapping the
                # pool under a thread still writing into it faults the process.
                raise
            logger.warning(
                "KVCR Guard core close failed after reaching quiescence",
                exc_info=True,
            )
        # Every close runs regardless; the first failure is the raised one.
        failure: BaseException | None = None
        for give_back in (
            self._close_control,
            self._recovery.close,
            self._pool_lease.close,
            self._close_owner,
        ):
            try:
                give_back()
            except BaseException as error:  # noqa: BLE001 - raised below
                failure = failure or error
        if failure is not None:
            raise failure

    def _close_control(self) -> None:
        if self._control is not None:
            self._control.close()
            self._control = None

    def _close_owner(self) -> None:
        if self._owner is None:
            return
        if self._recovery.attachment is not None:
            # The mapping would not close; unlinking now would hide
            # still-committed RAM from the next start's purge.
            return
        self._owner.close()
        self._owner = None
