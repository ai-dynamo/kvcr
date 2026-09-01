# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Private journal-backed Guard for one service-owned pool.

Asserts in this module are Optional-narrowing only: each names a field the
phase machine established before that line can run. Under ``python -O`` a
violated one would surface as an AttributeError on None -- a worse message,
never a different outcome.
"""

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
from dataclasses import dataclass, field
from typing import Any

from .api import KVCRBindings
from .config import (
    KVCRBackendConfigs,
    KVCRConfig,
    LocalDramInfo,
)
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
    clear_recovery_snapshot,
    install_recovery_records,
    read_handback,
    write_recovery_snapshot,
)
from .types import BlockKey

logger = logging.getLogger(__name__)

# TODO: Wake the mirror on publication rather than polling. A standby still
# drains at roughly a fifth of the rate a primary can publish.
_POLL_SECONDS = 0.001
_POLL_BATCH = 64
_LIFECYCLE_TIMEOUT_SECONDS = 30.0
_RECOVERY_CAPACITY_ERRORS = (errno.ENOSPC, errno.EDQUOT)


@dataclass
class _Command:
    """One request on the pool's mailbox, and the future its answer arrives on."""

    operation: str
    args: tuple[Any, ...] = ()
    future: "concurrent.futures.Future[Any]" = field(
        default_factory=concurrent.futures.Future
    )


class _Phase(enum.Enum):
    """Where a pool is in its life.

    The first six are stable: states a pool can rest in, each naming exactly
    which resources exist. The last three are transient reservations a command
    holds while it runs, taken before the command is queued, so a conflicting
    command on the same pool is refused now rather than parked behind work
    that may outlive its caller's patience.
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


class _Lease:
    """One primary's exclusive hold on its pool.

    Identity is the authority: a release acts only if it names THIS object, so
    nothing stale -- a reused pid, a retried release, an old connection -- can
    ever touch a newer lease.
    """

    __slots__ = ("liveness",)

    def __init__(self, liveness: PidfdLiveness) -> None:
        self.liveness = liveness


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


@dataclass(frozen=True)
class _ConfiguredTier:
    """The tiers a claim chose, and the geometry this pool has under them.

    One record because the four are only meaningful together: a stride without
    the rows it yields describes nothing. Deriving them together is also what
    makes committing a configuration a single assignment, so a failure part way
    through choosing one cannot leave a pool half-chosen.
    """

    tier_config: _TierConfig
    effective_bytes: int
    rows: int

    @property
    def row_stride(self) -> int:
        return self.tier_config.row_stride

    @classmethod
    def derive(cls, spec: KVCRPoolSpec, tier_config: _TierConfig) -> "_ConfiguredTier":
        effective_bytes, rows = _compute_pool_geometry(
            spec.data_bytes, tier_config.row_stride
        )
        return cls(tier_config, effective_bytes, rows)


class _Guard:
    """One pool's standby, for as long as the service owns the pool.

    Built with the pool and outliving every primary that holds it, so a claim is
    something this is told about rather than something that creates it.
    """

    def __init__(
        self,
        spec: KVCRPoolSpec,
        failure_callback: Callable[..., None] | None = None,
        *,
        compatibility_digest: str,
        pool_index: int = 0,
        owner: _KVCRPoolOwner | None = None,
        refusing: Callable[[], bool] = lambda: False,
    ) -> None:
        self._spec = spec
        self._compatibility_digest = compatibility_digest
        self._pool_index = pool_index
        # The pool itself, and the lease over it. Owned here rather than by the
        # registry: one thread owns everything about one pool, so a claim needs
        # no lock and no reservation against the next one -- the mailbox is the
        # reservation.
        self._owner = owner
        self._refusing = refusing
        self._lease: _Lease | None = None
        self._listener: socket.socket | None = None
        # The address as the claimant asked for it: getsockname() answers
        # numerically and would reject every alias of the same address.
        self._bind: tuple[str, int] | None = None
        # All of this belongs to whichever primary currently holds the pool.
        self._control: ZmqPeerControlChannel | None = None
        self._configured: _ConfiguredTier | None = None
        # The G3 half of what was recovered. A Guard does not serve it, but the
        # primary that takes the pool back does, so it is kept rather than dropped.
        self._g3_records: dict[BlockKey, _G3Residency] = {}
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name=f"kvcr-guard-{spec.pool_id}",
            daemon=True,
        )
        self._started = False
        self._closed = False
        self._serving = False
        self._resumable = False
        # One small lock over the phase, its reservation, the lease and the
        # failure -- never held across blocking work. It exists so a caller can
        # reserve a transition before queueing it; the actor thread is still
        # the only thing that runs one.
        self._phase_lock = threading.Lock()
        self._phase = _Phase.UNCONFIGURED
        self._reserved: _Phase | None = None
        self._closing = False
        self._failure: BaseException | None = None
        self._failure_callback = failure_callback or (lambda guard, error: None)
        self._attachment: KVCRPoolAttachment | None = None
        self._journal: RecoveryJournal | None = None
        self._mirror: _RecoveryMirror | None = None
        self._core: _KVCRCore | None = None
        self._handlers: dict[str, Callable[..., Any]] = {
            "claim": self._claim,
            "release": self._stand_down,
            "abort": self._abort,
            "close": self._close,
        }

    def start(self) -> None:
        """Attach the pool and begin the lifecycle thread, before any claim."""
        if self._started:
            raise RuntimeError("Guard preparation was already attempted")
        self._started = True
        # On this thread: attaching a pool is not thread-affine, no command can
        # exist yet, and a caller that sees the failure directly never has to
        # decide what to tear down behind a thread still opening it.
        self._prepare()
        self._thread.start()

    def claim(
        self,
        liveness: PidfdLiveness,
        tier_config: _TierConfig,
        control_bind: tuple[str, int],
    ) -> "tuple[KVCRPoolSpec, int, _Lease]":
        """Give this pool to a primary, with the endpoint its Guard answers on.

        The reservation is taken here, on the requesting thread, so a pool
        mid-transition answers busy immediately instead of queueing this
        claimant behind work that may outlive its patience.
        """
        self._reserve_claim()
        return self._submit(_Command("claim", (liveness, tier_config, control_bind)))

    def release(self, lease: "_Lease") -> None:
        """End a lease. The pool keeps its Guard, and the Guard its records.

        A stale lease -- one a promotion or an earlier release already ended --
        is a no-op. During shutdown a release is absorbed the same way: the
        close path owns every resource a release would have touched.
        """
        with self._phase_lock:
            if self._closing or lease is not self._lease:
                return
            if self._reserved is not None:
                if self._reserved is _Phase.PROMOTING:
                    # The death of this same lease got here first; it wins.
                    return
                raise KVCRServiceError(f"KVCR pool {self._pool_index} is busy")
            self._reserved = _Phase.RELEASING
        self._submit(_Command("release", (lease,)))

    def abort_grant(self, lease: "_Lease") -> None:
        """Roll back a lease its claimant declared it never served.

        The claimant sends this only after stopping local access, so if this
        claim stood a serving Guard down, the Guard may resume; otherwise it
        is an ordinary release. Staleness is decided exactly as for a release.
        """
        with self._phase_lock:
            if self._closing or lease is not self._lease:
                return
            if self._reserved is not None:
                if self._reserved is _Phase.PROMOTING:
                    # The death of this same lease got here first; it wins.
                    return
                raise KVCRServiceError(f"KVCR pool {self._pool_index} is busy")
            self._reserved = _Phase.RELEASING
        self._submit(_Command("abort", (lease,)))

    def close(self) -> None:
        self.begin_close()
        if self.finish_close(time.monotonic() + _LIFECYCLE_TIMEOUT_SECONDS):
            raise TimeoutError("KVCR Guard lifecycle thread did not stop")

    def begin_close(self) -> None:
        """Stop taking work and queue the teardown, without waiting for it.

        Split from the wait so shutdown reaches every pool before it waits on
        any one of them: a wedged pool must not keep its neighbours from being
        told to close.
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
        """Wait out a begun close. True if the actor is wedged past the deadline.

        A close that finished but failed raises its reason here, so shutdown
        keeps the pool -- and its resources -- visible instead of forgetting it.
        """
        if not self._started or self._thread.ident is None:
            # The thread never ran; begin_close already closed inline.
            return False
        # Always joined once it ran: _closed is set just before the actor's
        # final drain, so returning on the flag alone could report a close
        # finished while the thread still held the mailbox.
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
                raise KVCRServiceError(f"KVCR pool {self._pool_index} is busy")
            if self._phase is _Phase.PRIMARY:
                lease = self._lease
                assert lease is not None
                poller = select.poll()
                poller.register(lease.liveness.fileno(), select.POLLIN)
                if not poller.poll(0):
                    raise KVCRServiceError(
                        f"KVCR pool {self._pool_index} is held by another worker"
                    )
                # Dead but not yet promoted: the actor is the sole authority.
                raise KVCRServiceError(f"KVCR pool {self._pool_index} is busy")
            if self._phase in (_Phase.FAILED, _Phase.CLOSED):
                raise KVCRServiceError("KVCR pool registry is closed")
            self._reserved = _Phase.CLAIMING

    def _submit(self, command: _Command) -> Any:
        """Run one command on the actor thread and wait for its outcome.

        The wait is unbounded but not unguarded: a reservation was taken before
        anything was queued, so every later caller is refused with busy instead
        of piling up here -- and an actor that exits mid-wait answers this
        command with a typed error instead of leaving its caller waiting.
        """
        self._commands.put(command)
        while True:
            try:
                return command.future.result(timeout=0.1)
            except TimeoutError:
                if command.future.done():
                    # The handler itself raised a TimeoutError; that is the
                    # answer, not a wait that has not finished.
                    raise
                if not self._thread.is_alive():
                    if command.future.done():
                        # Completed in the gap between the wait and this check.
                        continue
                    # The drain on the actor's way out answers everything
                    # queued; this covers a command that raced the drain.
                    raise KVCRServiceError("KVCR pool registry is closed") from None

    def _run(self) -> None:
        try:
            self._serve_commands()
        except BaseException as error:  # noqa: BLE001 - the loop itself failed
            self._record_background_failure(error)
        finally:
            # Whatever ended this thread -- CLOSED, FAILED, or a defect in the
            # loop -- nothing queued may be left waiting on an answer that is
            # never coming.
            with self._phase_lock:
                self._closing = True
            while True:
                try:
                    abandoned = self._commands.get_nowait()
                except queue.Empty:
                    break
                abandoned.future.set_exception(
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
                handler = self._handlers.get(command.operation)
                if handler is None:
                    raise AssertionError(f"unknown Guard command: {command.operation}")
                result = handler(*command.args)
            except BaseException as error:  # noqa: BLE001 - returned to caller
                failed = error
                with self._phase_lock:
                    # The one rollback point: success commits a stable phase
                    # itself; failure gives the reservation back here.
                    self._reserved = None
                command.future.set_exception(error)
            else:
                with self._phase_lock:
                    self._reserved = None
                command.future.set_result(result)
            if command.operation == "close":
                if failed is not None:
                    # A close that failed cannot be retried into success; the
                    # actor ends either way and finish_close raises the reason.
                    with self._phase_lock:
                        self._failure = failed
                        self._phase = _Phase.FAILED
                return

    def _observe_holder(self) -> None:
        """Notice the current primary dying. The actor is the only watcher.

        The pidfd is polled here, between commands and on the same thread that
        runs them, so a death and every command are totally ordered: there is
        no watcher thread to race, misreport a shutdown, or outlive the pool.
        """
        with self._phase_lock:
            if (
                self._phase is not _Phase.PRIMARY
                or self._reserved is not None
                or self._closing
            ):
                return
            lease = self._lease
        assert lease is not None
        poller = select.poll()
        poller.register(lease.liveness.fileno(), select.POLLIN)
        events = poller.poll(0)
        if not events:
            return
        flags = events[0][1]
        with self._phase_lock:
            if self._lease is not lease or self._reserved is not None or self._closing:
                # A close that began while this poll was in flight owns the
                # teardown; promoting now would race it for the same resources.
                return
            self._reserved = _Phase.PROMOTING
        try:
            if not flags & select.POLLIN:
                # The descriptor broke while its process may still be alive:
                # promotion here could seat a second server over a live mapping.
                raise OSError(f"pidfd poll returned without POLLIN: {flags:#x}")
            self._promote_for(lease)
        except BaseException as error:  # noqa: BLE001 - service-fatal
            with self._phase_lock:
                self._failure = error
                self._phase = _Phase.FAILED
            self._escalate(error)
        finally:
            with self._phase_lock:
                self._reserved = None

    def _claim(
        self,
        liveness: PidfdLiveness,
        tier_config: _TierConfig,
        control_bind: tuple[str, int],
    ) -> "tuple[KVCRPoolSpec, int, _Lease]":
        """Give the pool to a primary, and take up the endpoint it named.

        Everything that can fail runs before the lease exists, and the lease is
        committed under the same lock a refusal is read under: there is no
        instant at which this pool is both granted and refused, and no failure
        that leaves it half-granted.
        """
        bound_here = self._listener is None
        listener = self._bind_listener(control_bind)
        granted_fd = -1
        try:
            granted_fd = os.dup(listener.fileno())
            # from_shared_listener detaches what it is given, so the Guard gets a
            # duplicate and this pool keeps the original.
            duplicate = socket.socket(fileno=os.dup(listener.fileno()))
            try:
                control = ZmqPeerControlChannel.from_shared_listener(duplicate)
            except BaseException:
                duplicate.close()
                raise
            self._adopt(control, tier_config)
            lease = _Lease(liveness)
            with self._phase_lock:
                if not self._closing and not self._refusing():
                    self._lease = lease
                    self._phase = _Phase.PRIMARY
                    return self._spec, granted_fd, lease
            # Refused at the commit itself: a service on its way out must not
            # grant a pool. Everything adopted goes back where a release would
            # have put it, so the pool is left safe rather than serving.
            self._release()
            with self._phase_lock:
                self._phase = _Phase.IDLE
            raise KVCRServiceError("KVCR pool registry is closed")
        except BaseException as error:
            if granted_fd >= 0:
                with suppress(OSError):
                    os.close(granted_fd)
            if bound_here:
                # This claim chose the address and then failed, so keeping it
                # would refuse the retry as an endpoint move.
                self._unbind_listener()
            if isinstance(error, (ValueError, RecoveryMirrorError)):
                raise KVCRServiceError(str(error)) from error
            raise

    def _stand_down(self, lease: "_Lease") -> None:
        """End this lease, keeping what it left for the next primary.

        Staleness was already decided under the phase lock at submission; by
        the time this runs, the lease is the current one.
        """
        try:
            self._release()
        except BaseException as error:
            # Escalated before the pool is exposed: a Guard that failed here may
            # have left a partial handback, and the next claimant could take it.
            with self._phase_lock:
                self._failure = error
                self._phase = _Phase.FAILED
            self._escalate(error)
            raise
        finally:
            lease.liveness.close()
            with self._phase_lock:
                if self._lease is lease:
                    self._lease = None
        with self._phase_lock:
            self._phase = _Phase.IDLE

    def _abort(self, lease: "_Lease") -> None:
        """Undo a grant that never arrived: resume serving, or release.

        Resuming is safe exactly because the claimant could not decode the
        grant -- it can never map the pool this Guard re-serves. The mirror
        still holds what the handback wrote, and the journal was reset with
        nothing published since, so promotion serves the same state back.
        """
        if self._resumable and not self._serving and self._mirror is not None:
            try:
                self._promote_for(lease)
            except BaseException as error:
                with self._phase_lock:
                    self._failure = error
                    self._phase = _Phase.FAILED
                self._escalate(error)
                raise
            return
        self._stand_down(lease)

    def _promote_for(self, lease: "_Lease") -> None:
        """Take the pool over from the primary that just died."""
        try:
            self._promote()
        finally:
            lease.liveness.close()
            with self._phase_lock:
                if self._lease is lease:
                    self._lease = None
        with self._phase_lock:
            self._phase = _Phase.STANDBY

    def _bind_listener(self, control_bind: tuple[str, int]) -> socket.socket:
        """The address this pool answers on, bound once and never moved.

        A Guard inherits the endpoint its primary used, so a claim naming a
        different one is refused rather than migrated.
        """
        if self._listener is not None:
            if self._bind != control_bind:
                assert self._bind is not None
                raise KVCRServiceError(
                    f"KVCR pool {self._pool_index} answers on "
                    f"{self._bind[0]}:{self._bind[1]} and cannot be moved to "
                    f"{control_bind[0]}:{control_bind[1]}"
                )
            return self._listener
        try:
            listener = socket.create_server(control_bind)
        except OSError as error:
            raise KVCRServiceError(
                f"KVCR pool {self._pool_index} control listener "
                f"{control_bind[0]}:{control_bind[1]} is unavailable: {error}"
            ) from error
        self._listener = listener
        self._bind = control_bind
        return listener

    def _unbind_listener(self) -> None:
        listener, self._listener = self._listener, None
        self._bind = None
        if listener is None:
            return
        try:
            listener.close()
        except BaseException as error:  # noqa: BLE001 - escalated, not swallowed
            # An address that will not close is one the service can neither
            # reach nor hand out, so nothing may claim this pool again.
            self._escalate(error)

    def _escalate(self, error: BaseException) -> None:
        logger.critical("KVCR pool %d Guard failed", self._pool_index)
        try:
            self._failure_callback(self, error)
        except BaseException:  # noqa: BLE001 - retain the original failure
            logger.exception("Failed to notify KVCR-Service of Guard failure")

    def _close(self) -> None:
        self._close_resources()
        self._closed = True
        with self._phase_lock:
            self._phase = _Phase.CLOSED

    def _prepare(self) -> None:
        """Everything that depends only on the pool, and so needs no claim."""
        attachment = KVCRPoolAttachment.attach(self._spec)
        self._attachment = attachment
        self._journal = RecoveryJournal(attachment)

    def _adopt(self, control: ZmqPeerControlChannel, tier_config: _TierConfig) -> None:
        """Take up a new primary, handing over whatever the last one left.

        A Guard that is serving hands the pool back first, which is what stops
        it answering before the replacement starts. A Guard that already holds
        records keeps them rather than reading back the region it just wrote.
        """
        try:
            # Everything that can refuse this claim runs before anything
            # moves: refusing later leaves the pool with no reader.
            if self._failure is not None:
                raise self._failure
            self._refuse_incompatible(tier_config)
            served_under = self._configured.row_stride if self._configured else 0
            recovered = self._mirror
            if recovered is None:
                # Nothing kept, so whatever a previous service left in the pool's
                # tail is the baseline this lease's deltas apply to.
                recovered = self._recovered_baseline(tier_config.row_stride)
            # Last, once nothing left can refuse this claim: a pool whose handback
            # would not replay has not chosen anything, and a corrected claim can
            # still have it.
            self._configure(tier_config)
            self._mirror = recovered
        except BaseException:
            control.close()
            raise

        try:
            if self._serving:
                self._hand_back(served_under)
                self._resumable = True
                if self._mirror is None:
                    # The handback did not fit, so the claimant was told cold:
                    # this lease's baseline is empty, not absent. A lease with
                    # no mirror would never be read again.
                    self._mirror = _RecoveryMirror()
            assert self._journal is not None
            self._journal.reset()
        except BaseException as error:
            # The pool has changed hands and nothing here can put it back, so
            # this stopped being something a claimant could be told about.
            control.close()
            self._record_background_failure(error)
            raise

        try:
            if self._control is not None:
                self._control.close()
        except BaseException as error:
            # The new channel owns a duplicate of the pool's listener and this is
            # the only reference to it left.
            control.close()
            self._record_background_failure(error)
            raise
        self._control = control

    def _release(self) -> None:
        """Give up the current primary, leaving its state where the next looks.

        The next primary reads only the handback region, so the mirror is written out
        and dropped. Keeping it would map keys to slots a claimant, told the pool was
        empty, has since allocated over.
        """
        self._resumable = False
        if self._failure is not None:
            raise self._failure
        assert self._configured is not None
        if self._serving:
            self._hand_back(self._configured.row_stride)
        elif self._mirror is not None:
            try:
                # A primary that is asking to release has stopped publishing, so
                # what is still in the ring is the tail of what it did publish.
                assert self._journal is not None
                while (frame := self._journal.read_next()) is not None:
                    self._mirror.apply(*frame)
                # Taken, not copied: the mirror is dropped on the next line, and
                # the table can hold a tier's worth of blocks.
                self._write_handback(
                    self._mirror.take_records(), self._configured.row_stride
                )
            except RecoveryJournalError as error:
                # Reached before _poll noticed the same thing -- or the tail
                # was already bad. The pool comes back cold rather than partly
                # described.
                self._drop_recovery(error)
            except OSError as error:
                if error.errno not in _RECOVERY_CAPACITY_ERRORS:
                    raise
                self._drop_recovery(error)
        self._mirror = None
        if self._control is not None:
            self._control.close()
            self._control = None

    def _refuse_incompatible(self, tier_config: _TierConfig) -> None:
        """Refuse a primary whose tiers are not the ones this pool was claimed with.

        A pool's configuration is fixed for the service's lifetime, and holding no
        records is not licence to change it: the records go, the bytes stay. A
        different stride, or the same G3 paths in a different order, leaves every
        slot number naming bytes it did not name before. Only the first claim
        chooses.
        """
        configured = self._configured
        if configured is None or configured.tier_config == tier_config:
            return
        raise RecoveryMirrorError(
            "KVCR pool was claimed with another tier configuration"
        )

    def _configure(self, tier_config: _TierConfig) -> None:
        """Take up the tiers this primary asked for, if this pool can have them.

        One assignment of a record derived whole, so a configuration this pool
        cannot have leaves the Guard in the one it already had.
        """
        self._configured = _ConfiguredTier.derive(self._spec, tier_config)

    def _recovered_baseline(self, row_stride: int) -> _RecoveryMirror:
        """The region a previous handover left, checked against these tiers.

        Checked while the claim is still being granted rather than at promotion,
        where refusing it stops the service: a claimant that dies in between would
        otherwise take everything with it. The stride is the incoming claim's,
        because this runs before the claim has been allowed to choose it.
        """
        assert self._attachment is not None
        return read_handback(self._attachment, self._compatibility_digest, row_stride)

    def _poll(self) -> bool:
        """Mirror what is waiting, and say whether there was more of it.

        A full batch means the primary is outrunning one wait, so the caller comes
        straight back. The batch bounds how long a command waits, not how much drains.
        """
        if self._failure is not None:
            return False
        try:
            if self._serving:
                assert self._core is not None
                self._core.poll_completed()
            elif self._mirror is not None:
                # A mirror means a primary holds this pool. Without one the
                # Guard is waiting to be claimed, and nothing is publishing.
                assert self._journal is not None
                for _ in range(_POLL_BATCH):
                    frame = self._journal.read_next()
                    if frame is None:
                        return False
                    self._mirror.apply(*frame)
                return True
        except RecoveryJournalError as error:
            self._drop_recovery(error)
        except RecoveryMirrorError as error:
            assert self._journal is not None
            # Suppressed: this thread dying here would leave the pool unclaimable
            # with nobody told, which is the opposite of the intended failure.
            with suppress(Exception):
                self._journal.invalidate()
            self._record_background_failure(error)
        except BaseException as error:  # noqa: BLE001 - promotion/close observes it
            self._record_background_failure(error)
        return False

    def _drop_recovery(self, error: RecoveryJournalError | OSError) -> None:
        """Lose this pool's recovery without losing the service.

        The primary outran this mirror, or the ring went bad under it. What is left
        is incomplete, so it is dropped rather than served -- but a pool that can no
        longer be recovered is not a Guard that has failed, and the primary treats
        the same condition as survivable. Every reader of the journal ends up here,
        because which one reaches it first is a race.
        """
        logger.warning(
            "KVCR pool recovery disabled; the pool will be claimable but "
            "cold if this primary dies: %s",
            error,
        )
        self._mirror = None

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
        assert self._journal is not None
        records: dict[BlockKey, _BlockRecord] = {}
        if self._mirror is None:
            logger.warning("KVCR pool has no recovered state to promote")
        else:
            try:
                # The primary's process is gone, so what is in the ring is all of
                # what it published: there is no more coming to be short of.
                for frame in self._journal.drain():
                    self._mirror.apply(*frame)
                records = self._mirror.take_records()
            except RecoveryJournalError as error:
                self._drop_recovery(error)
        # A fresh mirror either way: a handover still has somewhere to put the
        # records the core ends up holding.
        self._mirror = _RecoveryMirror()
        self._serve(records)

    def _serve(self, records: dict[BlockKey, _BlockRecord]) -> None:
        """Answer on this pool's endpoint, with whatever came back from it.

        Serving nothing is still serving. The dead primary's peers go on sending
        to this address, and one that answers refuses them, where one that only
        stays bound leaves them waiting on a reply nobody is going to send.
        """
        # A Guard serves G2 and opens no G3: staging a block up to serve it would
        # cost the tier a slot for a lease that ends at the next claim. The G3 half
        # is kept whole and handed to the replacement, which does open G3.
        self._g3_records = {
            key: record.g3 for key, record in records.items() if record.g3 is not None
        }
        core = self._serving_core()
        self._core = core
        install_recovery_records(core, _without_g3(records))
        # A previous handover describes slots this Guard is about to move, and it is
        # already in the mirror. Leaving it would map keys to overwritten bytes.
        assert self._attachment is not None
        clear_recovery_snapshot(self._attachment)
        core.start()
        self._serving = True

    def _serving_core(self) -> _KVCRCore:
        """A core over this pool's mapping, under an agent name of its own.

        A new NIXL agent because a peer must not reach the dead primary's, and no
        framework memory or G3: a Guard serves what the pool already holds.
        """

        def reject_pin(keys: object) -> int:
            del keys
            raise RuntimeError("Guard has no framework-owned memory")

        assert self._attachment is not None
        configured = self._configured
        assert configured is not None
        return _KVCRCore(
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
            KVCRBackendConfigs(
                local_dram=LocalDramInfo(
                    self._attachment.data_address,
                    configured.effective_bytes,
                    configured.rows,
                ),
                g3=None,
            ),
        )

    def _hand_back(self, row_stride: int) -> None:
        """Stop serving, leaving this pool's state where the next primary looks.

        Closing the core first stops this Guard answering before anything replaces it,
        and makes the two halves agree: the region and the records both come from the
        map close leaves behind.
        """
        core = self._core
        mirror = self._mirror
        if core is None or mirror is None:
            raise RecoveryMirrorError("a serving Guard has no state to hand back")
        core.close()
        records = _with_g3(core._block_record_map, self._g3_records)
        try:
            self._write_handback(records, row_stride)
        except OSError as error:
            if error.errno not in _RECOVERY_CAPACITY_ERRORS:
                raise
            # The ring-full precedent: a pool whose state will not fit at the
            # tail (ENOSPC or EDQUOT) is cold, not fatal. The region's error path
            # truncated what it started, so the next claimant is told nothing
            # rather than half of something -- and the mirror is dropped with
            # it, because a claimant told the pool is empty must not race a
            # mirror that still names slots.
            self._drop_recovery(error)
        else:
            mirror.adopt(records)
        self._g3_records = {}
        self._core = None
        self._serving = False

    # TODO: Verify the G3 files a handover describes. Nothing holds them while
    # this Guard serves -- the exclusive lock lives with the tier, and a Guard
    # opens no G3 -- so a second KVCR on the same paths goes unnoticed and the
    # replacement serves whatever is in the slots. Refusing two pools that name
    # the same paths is the cheap first step; it does not cover a second
    # service, or a KVCR using G3 with no pool at all.
    def _write_handback(
        self,
        records: Mapping[BlockKey, _BlockRecord],
        row_stride: int,
    ) -> None:
        """Leave these records where the next primary to claim will look.

        The stride is the one the records were written under, not whatever is
        incoming: the terms have to name the pool the records describe.
        """
        assert self._attachment is not None
        write_recovery_snapshot(
            self._attachment,
            canonical_pool_terms(self._compatibility_digest, row_stride, self._spec),
            _recovery_frames(records),
        )

    def _close_resources(self) -> None:
        try:
            if self._core is not None:
                self._core.close()
        except BaseException:
            assert self._core is not None
            if not self._core.is_quiescent():
                # Still moving bytes, so nothing below runs: unmapping the
                # pool under a thread still writing into it faults the process.
                raise
            logger.warning(
                "KVCR Guard core close failed after reaching quiescence",
                exc_info=True,
            )
        # Everything else the pool has, whatever any one of them does about it:
        # the first refusal is the one that explains the rest.
        failure: BaseException | None = None
        for give_back in (
            self._close_control,
            self._close_attachment,
            self._close_lease,
            self._close_listener,
            self._close_owner,
        ):
            try:
                give_back()
            except BaseException as error:  # noqa: BLE001 - raised below
                failure = failure or error
        if failure is not None:
            raise failure

    # Each helper clears its field only once the close succeeded: a resource
    # whose close failed stays referenced, so a kept pool can still name -- and
    # retry -- what it leaked. Every close here is idempotent.

    def _close_control(self) -> None:
        if self._control is not None:
            self._control.close()
            self._control = None

    def _close_attachment(self) -> None:
        if self._attachment is not None:
            self._attachment.close()
            self._attachment = None

    def _close_lease(self) -> None:
        if self._lease is not None:
            self._lease.liveness.close()
            self._lease = None

    def _close_listener(self) -> None:
        # Unlike _unbind_listener, a failure here raises: at teardown there is
        # no claim error to protect, and the kept pool must still name the
        # endpoint it could not free.
        if self._listener is not None:
            self._listener.close()
            self._listener = None
            self._bind = None

    def _close_owner(self) -> None:
        if self._owner is None:
            return
        if self._attachment is not None:
            # The mapping would not close, so the file must keep naming it:
            # unlinking now would hide still-committed RAM from the next
            # start's purge. The attachment failure above reports the pool.
            return
        self._owner.close()
        self._owner = None
