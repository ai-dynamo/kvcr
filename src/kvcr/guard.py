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

_OPS = dict(claim="_claim", release="_stand_down", abort="_abort", close="_close")


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


class _Command:
    """One request on the pool's mailbox, and the future its answer arrives on."""

    def __init__(self, operation: str, args: tuple[Any, ...] = ()) -> None:
        self.operation, self.args = operation, args
        self.future: concurrent.futures.Future[Any] = concurrent.futures.Future()


# First six: stable resting states. Last three: transient reservations taken
# before a command is queued, so a conflicting command is refused immediately.
_Phase = enum.Enum(
    "_Phase",
    "UNCONFIGURED IDLE STANDBY PRIMARY FAILED CLOSED CLAIMING RELEASING PROMOTING",
)


class _Guard:
    """One pool's standby, alive as long as the service owns the pool.
    Outlives every primary: a claim is reported to it, not what creates it.
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
        # Owned here, not by the registry: one thread owns one pool, so a
        # claim needs no lock -- the mailbox is the reservation.
        self._owner = owner
        self._refusing = refusing
        self._lease = self._listener = None
        # As the claimant asked: getsockname() is numeric and rejects aliases.
        self._bind: tuple[str, int] | None = None
        # Owned by the current primary.
        self._control: ZmqPeerControlChannel | None = None
        self._configured: _TierConfig | None = None
        self._attachment = self._journal = self._mirror = self._core = None
        # Recovered half a Guard cannot serve; kept for the next primary, which can.
        self._g3_records: dict[BlockKey, _G3Residency] = {}
        self._commands: queue.Queue[_Command] = queue.Queue()
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
        self._prepare()
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
            if self._closing or lease is not self._lease:
                return
            if self._reserved is not None:
                if self._reserved is _Phase.PROMOTING:
                    # The death of this same lease got here first; it wins.
                    return
                raise KVCRServiceError(f"KVCR pool {self._pool_index} is busy")
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
                raise KVCRServiceError(f"KVCR pool {self._pool_index} is busy")
            if self._phase is _Phase.PRIMARY:
                poller = select.poll()
                poller.register(self._lease.fileno(), select.POLLIN)
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
                result = getattr(self, _OPS[command.operation])(*command.args)
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
            lease = self._lease
        poller = select.poll()
        poller.register(lease.fileno(), select.POLLIN)
        if not (events := poller.poll(0)):
            return
        flags = events[0][1]
        with self._phase_lock:
            if self._lease is not lease or self._reserved is not None or self._closing:
                # An in-flight close owns the teardown; promoting would race it.
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
        bound_here = self._listener is None
        listener = self._bind_listener(bind)
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
                    self._lease = liveness
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
                self._unbind_listener()
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
                if self._lease is lease:
                    self._lease = None
        with self._phase_lock:
            self._phase = _Phase.IDLE

    def _abort(self, lease: "_Lease") -> None:
        """Undo a grant its claimant declared unserved: resume, or release.
        Resume is safe: the claimant stopped local access for good, the mirror
        holds the handback, the journal is reset -- promotion serves it back.
        """
        if self._resumable and not self._serving and self._mirror is not None:
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
                if self._lease is lease:
                    self._lease = None
        with self._phase_lock:
            self._phase = _Phase.STANDBY

    def _bind_listener(self, control_bind: tuple[str, int]) -> socket.socket:
        """Bind this pool's address, once and never moved: a claim naming a
        different endpoint is refused rather than migrated.
        """
        if self._listener is not None:
            if self._bind != control_bind:
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
            # An address that will not close: nothing may claim this pool again.
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
        self._attachment = KVCRPoolAttachment.attach(self._spec)
        self._journal = RecoveryJournal(self._attachment)

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
            recovered = self._mirror
            if recovered is None:
                # The prior handback is this lease's baseline. Read now, under
                # the claim's stride: refusing at promotion stops the service,
                # and a claimant dying in between takes everything with it.
                recovered = read_handback(
                    self._attachment, self._compatibility_digest, tier_config.row_stride
                )
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
            self._journal.reset()
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
        elif self._mirror is not None:
            try:
                # A primary that is asking to release has stopped publishing, so
                # what is still in the ring is the tail of what it did publish.
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
            elif self._mirror is not None:
                # A mirror means a primary holds this pool. Without one the
                # Guard is waiting to be claimed, and nothing is publishing.
                for _ in range(_POLL_BATCH):
                    frame = self._journal.read_next()
                    if frame is None:
                        return False
                    self._mirror.apply(*frame)
                return True
        except RecoveryJournalError as error:
            self._drop_recovery(error)
        except RecoveryMirrorError as error:
            # Suppressed: this thread dying here would leave the pool unclaimable
            # with nobody told, which is the opposite of the intended failure.
            with suppress(Exception):
                self._journal.invalidate()
            self._record_background_failure(error)
        except BaseException as error:  # noqa: BLE001 - promotion/close observes it
            self._record_background_failure(error)
        return False

    def _drop_recovery(self, error: RecoveryJournalError | OSError) -> None:
        """Lose this pool's recovery without losing the service: dropped, not
        served, and not a Guard failure. Every reader of the journal ends up here,
        because which one reaches it first is a race.
        """
        logger.warning(
            "KVCR pool recovery disabled; claimable but cold if this primary dies: %s",
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

        Serving nothing is still serving: answering refuses peers that staying
        bound would leave hanging. G2 only, no G3: that half is kept whole for
        the replacement. A new NIXL agent name keeps peers off the dead one's.
        """
        self._g3_records = {
            key: record.g3 for key, record in records.items() if record.g3 is not None
        }

        def reject_pin(keys: object) -> int:
            raise RuntimeError("Guard has no framework-owned memory")

        effective_bytes, rows = _compute_pool_geometry(
            self._spec.data_bytes, self._configured.row_stride
        )
        dram = LocalDramInfo(self._attachment.data_address, effective_bytes, rows)
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
        core.adopt_recovery_records(_without_g3(records))
        # A previous handover describes slots this Guard is about to move, and it is
        # already in the mirror. Leaving it would map keys to overwritten bytes.
        self._attachment.release_snapshot_region()
        core.start()
        self._serving = True

    def _hand_back(self, row_stride: int) -> None:
        """Stop serving, leaving this pool's state where the next primary looks.
        The core closes first: the Guard stops answering, and region and
        records both come from the map close leaves behind.
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
            # tail (ENOSPC or EDQUOT) is cold, not fatal. The write was
            # truncated, and the mirror is dropped too: a claimant told the
            # pool is empty must not race one.
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
        """Leave this state where the next primary to claim will look.
        The stride is the one the records were written under, not the incoming.
        """
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
            lambda: self._close_field("_control"),
            lambda: self._close_field("_attachment"),
            lambda: self._close_field("_lease"),
            self._close_listener,
            self._close_owner,
        ):
            try:
                give_back()
            except BaseException as error:  # noqa: BLE001 - raised below
                failure = failure or error
        if failure is not None:
            raise failure

    # A field is cleared only once its close succeeds, so a failed close stays
    # referenced and retryable. Every close here is idempotent.

    def _close_field(self, name: str) -> None:
        held = getattr(self, name)
        if held is not None:
            held.close()
            setattr(self, name, None)

    def _close_listener(self) -> None:
        # Unlike _unbind_listener, a failure raises: the kept pool must name the leak.
        if self._listener is not None:
            self._listener.close()
            self._listener = None
            self._bind = None

    def _close_owner(self) -> None:
        if self._owner is None:
            return
        if self._attachment is not None:
            # The mapping would not close; unlinking now would hide
            # still-committed RAM from the next start's purge.
            return
        self._owner.close()
        self._owner = None
