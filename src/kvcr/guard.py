# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Private journal-backed Guard for one service-owned pool.

Asserts in this module are Optional-narrowing only: each names a field the
phase machine established before that line can run. Under ``python -O`` a
violated one would surface as an AttributeError on None -- a worse message,
never a different outcome.
"""

import errno
import logging
import queue
import threading
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field

from .api import KVCRBindings
from .config import (
    KVCRBackendConfigs,
    KVCRConfig,
    LocalDramInfo,
)
from .control_channels import ZmqPeerControlChannel
from .core import _BlockRecord, _KVCRCore
from .guard_protocol import _TierConfig
from .local_disk import _G3Residency
from .memory import KVCRPoolAttachment, KVCRPoolSpec, _compute_pool_geometry
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
    operation: str
    control: ZmqPeerControlChannel | None = None
    tier_config: "_TierConfig | None" = None
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


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
    ) -> None:
        self._spec = spec
        self._compatibility_digest = compatibility_digest
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
        self._failure: BaseException | None = None
        self._failure_callback = failure_callback or (lambda guard, error: None)
        self._attachment: KVCRPoolAttachment | None = None
        self._journal: RecoveryJournal | None = None
        self._mirror: _RecoveryMirror | None = None
        self._core: _KVCRCore | None = None

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

    def promote_after_death(self) -> None:
        self._submit(_Command("promote"))

    def adopt(self, control: ZmqPeerControlChannel, tier_config: _TierConfig) -> None:
        """Take up a new primary, handing over whatever the last one left."""
        command = _Command("adopt", control=control, tier_config=tier_config)
        try:
            self._submit(command)
        except BaseException:
            # _adopt closes this once the command reaches the thread. If it
            # never got there, nothing else owns the pool listener's duplicate.
            if not command.done.is_set():
                control.close()
            raise

    def release(self) -> None:
        """Let the current primary go, keeping what it left for the next one."""
        self._submit(_Command("release"))

    def abort_grant(self) -> None:
        """Roll back a grant the claimant provably never received.

        A failed send of the length-prefixed grant means the claimant cannot
        decode it, so it can never map the pool: if this claim stood a serving
        Guard down, the Guard resumes; otherwise this is an ordinary release.
        """
        self._submit(_Command("abort"))

    def close(self) -> None:
        if self._closed:
            return
        if not self._started or not self._thread.is_alive():
            self._close_resources()
            self._closed = True
            return
        self._submit(_Command("close"), _LIFECYCLE_TIMEOUT_SECONDS)
        self._thread.join(_LIFECYCLE_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise TimeoutError("KVCR Guard lifecycle thread did not stop")

    def _submit(self, command: _Command, timeout: float | None = None) -> None:
        """Run one command on the lifecycle thread and wait for its outcome.

        State-changing commands wait untimed: a deadline cancels nothing, so a
        promotion reported as timed out could still land against a registry that has
        already rolled the claim back. A wedged Guard is contained by the registry's
        own transition deadline instead.

        One command at a time needs no lock here: the registry marks a pool in
        transition for exactly as long as a command can be in flight, and it is
        this Guard's only caller.
        """
        if not self._thread.is_alive():
            raise RuntimeError("Guard lifecycle thread stopped unexpectedly")
        self._commands.put(command)
        if not command.done.wait(timeout):
            raise TimeoutError(f"KVCR Guard {command.operation} timed out")
        if command.error is not None:
            raise command.error

    def _run(self) -> None:
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
                draining = self._poll()
                continue
            try:
                if command.operation == "promote":
                    self._promote()
                elif command.operation == "adopt":
                    assert command.control is not None
                    assert command.tier_config is not None
                    self._adopt(command.control, command.tier_config)
                elif command.operation == "release":
                    self._release()
                elif command.operation == "abort":
                    self._abort()
                elif command.operation == "close":
                    self._close_resources()
                    self._closed = True
                else:
                    raise AssertionError(f"unknown Guard command: {command.operation}")
            except BaseException as error:  # noqa: BLE001 - returned to caller
                command.error = error
                if command.operation == "promote":
                    self._failure = error
            finally:
                command.done.set()
            if command.operation == "close" and command.error is None:
                return

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
        self._failure = error
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

    def _abort(self) -> None:
        """Undo a grant that never arrived: resume serving, or release.

        Resuming is safe exactly because the claimant could not decode the
        grant -- it can never map the pool this Guard re-serves.
        """
        if self._resumable and not self._serving and self._mirror is not None:
            self._promote()
            return
        self._release()

    def _promote(self) -> None:
        self._resumable = False
        """Take the pool over from the dead primary, warm if anything survived."""
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
        # BaseException: even an interrupt must not skip giving the pool back
        # once the core is quiescent.
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
        try:
            if self._control is not None:
                self._control.close()
        finally:
            if self._attachment is not None:
                self._attachment.close()
