# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Threaded progress and NIXL transfer lifecycle for KVCR backends."""

import logging
import queue
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from nixl import nixl_agent, nixl_agent_config

from .types import BlockKey, MemDescriptor

logger = logging.getLogger(__name__)
_IDLE_WAIT_SECONDS = 0.001
_CLOSE_TIMEOUT_SECONDS = 5.0
_JOIN_TIMEOUT_SECONDS = 10.0
_PROGRESS_LOG_INTERVAL_SECONDS = 1.0
_RELEASE_LOG_INTERVAL_SECONDS = 1.0
_STOP = object()
_OpId = tuple[str, Any]


@dataclass(slots=True)
class _TransferState:
    """Hold one progress-owned NIXL transfer handle."""

    handle: Any
    capture_telemetry: bool = False
    outcome: bool | None = None
    telemetry: Any | None = None
    next_progress_log_at: float = 0.0
    next_release_log_at: float = 0.0


@dataclass
class _Op:
    """Identity and block dependencies shared by all KVCR operations."""

    op_id: _OpId
    keys: set[BlockKey]


class _ProgressOp(_Op, ABC):
    """An operation that can be owned and advanced by progress."""

    @abstractmethod
    def progress(
        self, progress: "_KVCRProgress", event: object | None
    ) -> tuple[bool, bool]:
        """Return whether the operation completed and whether work occurred."""

    def close(self, progress: "_KVCRProgress") -> bool:
        return True


_Initialize = Callable[["_KVCRProgress"], None]
_Poll = Callable[["_KVCRProgress", list[object]], tuple[dict[object, object], bool]]
_Flush = Callable[[], list[object]]
_Close = Callable[[], None]


class _KVCRProgress:
    """Run backend progress on one exclusively owning thread."""

    def __init__(
        self,
        initialize: _Initialize,
        poll: _Poll,
        flush: _Flush,
        close: _Close,
        *,
        batch_size: int = 64,
        nixl_agent_name: str | None = None,
        nixl_listen_port: int | None = None,
        memory_regions: tuple[tuple[int, int], ...] = (),
    ) -> None:
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        self._initialize = initialize
        self._poll = poll
        self._flush = flush
        self._close = close
        self._batch_size = batch_size or sys.maxsize
        self._nixl_agent_name = nixl_agent_name
        self._nixl_agent: Any | None = None
        self._active_transfers: dict[int, _TransferState] = {}
        self._next_transfer_id = 0
        self._nixl_listen_port = nixl_listen_port
        self._memory_regions = memory_regions
        self._memory_registrations: list[Any] = []
        self._nixl_agent_metadata: bytes | None = None
        self._submissions: queue.SimpleQueue[object] = queue.SimpleQueue()
        self._completed: queue.SimpleQueue[object] = queue.SimpleQueue()
        self._completed_backlog: deque[object] = deque()
        self._in_flight_ops: dict[_OpId, _ProgressOp] = {}
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="kvcr-progress",
        )
        self._failure: BaseException | None = None
        self._stop_requested = False

    @property
    def nixl_agent(self) -> Any:
        agent = self._nixl_agent
        if agent is None:
            raise RuntimeError("KVCR NIXL agent is not initialized")
        return agent

    @property
    def nixl_agent_metadata(self) -> bytes | None:
        return self._nixl_agent_metadata

    @property
    def nixl_agent_name(self) -> str:
        name = self._nixl_agent_name
        if not name:
            raise RuntimeError("KVCR NIXL agent name is not configured")
        return name

    def poll_transfer(
        self,
        transfer_id: int,
        *,
        cancellation_requested: bool = False,
    ) -> tuple[bool, Any | None] | None:
        """Return a transfer only after NIXL reports a terminal state.

        ``cancellation_requested`` is advisory: the Python NIXL API does not
        expose an abort fence, so PROC/PEND and polling faults must retain the
        handle and every buffer it can still mutate.
        """
        state = self._active_transfers.get(transfer_id)
        if state is None:
            raise KeyError(f"unknown transfer {transfer_id}")
        agent = self.nixl_agent
        outcome = state.outcome
        if outcome is None:
            try:
                xfer_state = agent.check_xfer_state(state.handle)
            except Exception:
                now = time.monotonic()
                if now >= state.next_progress_log_at:
                    logger.warning(
                        "NIXL transfer progress failed",
                        exc_info=True,
                    )
                    state.next_progress_log_at = now + _PROGRESS_LOG_INTERVAL_SECONDS
                return None
            if xfer_state in ("PROC", "PEND"):
                return None
            if xfer_state == "DONE":
                outcome = True
            elif xfer_state == "ERR":
                outcome = False
            else:
                logger.warning(
                    "NIXL transfer returned unexpected progress state %r",
                    xfer_state,
                )
                return None
            state.outcome = outcome
        if outcome and state.capture_telemetry:
            get_telemetry = getattr(agent, "get_xfer_telemetry", None)
            if get_telemetry is not None:
                try:
                    state.telemetry = get_telemetry(state.handle)
                except Exception:
                    logger.debug("NIXL telemetry failed", exc_info=True)
            state.capture_telemetry = False
        if not self._release_transfer(transfer_id, state):
            return None
        return bool(outcome), state.telemetry

    def submit_transfer(
        self,
        operation: str,
        local_descriptors: Sequence[MemDescriptor],
        remote_descriptors: Sequence[MemDescriptor],
        *,
        remote_side_agent: str | bytes,
        backend: str | None = None,
        notif_msg: bytes = b"",
        capture_telemetry: bool = False,
    ) -> tuple[int, bool]:
        """Submit aligned local and remote descriptors to NIXL."""
        if not local_descriptors or not remote_descriptors:
            raise ValueError("NIXL transfer descriptors must be non-empty")
        if not isinstance(remote_side_agent, (str, bytes)) or not remote_side_agent:
            raise ValueError("NIXL remote-side agent must be non-empty")
        agent = self.nixl_agent
        handle = agent.initialize_xfer(
            operation,
            self._make_transfer_descriptors(local_descriptors),
            self._make_transfer_descriptors(remote_descriptors),
            remote_side_agent,
            notif_msg=notif_msg,
            backends=[backend] if backend else self._dram_capable_backends(),
        )
        if handle is None:
            raise RuntimeError("initialize_xfer returned None")
        self._next_transfer_id += 1
        transfer_id = self._next_transfer_id
        state = _TransferState(handle, capture_telemetry)
        self._active_transfers[transfer_id] = state
        submitted = True
        try:
            post_state = agent.transfer(handle)
            if post_state == "DONE":
                state.outcome = True
            elif post_state == "ERR":
                state.outcome = False
                submitted = False
            elif post_state not in ("PROC", "PEND"):
                submitted = False
                logger.warning("NIXL transfer returned unexpected state %r", post_state)
        except Exception:
            submitted = False
            logger.warning("NIXL transfer submission was ambiguous", exc_info=True)
        return transfer_id, submitted

    def cancel_transfer(self, transfer_id: int) -> bool:
        """Drain a cancelled transfer, retaining it until NIXL is terminal."""
        state = self._active_transfers.get(transfer_id)
        if state is None:
            return True
        return self.poll_transfer(transfer_id, cancellation_requested=True) is not None

    def _make_transfer_descriptors(self, descriptors: Sequence[MemDescriptor]) -> Any:
        mem_type = descriptors[0].mem_type
        if any(descriptor.mem_type != mem_type for descriptor in descriptors):
            raise ValueError("one NIXL descriptor list cannot mix memory types")
        return self.nixl_agent.get_xfer_descs(
            [
                (descriptor.addr, descriptor.size, descriptor.device_Id)
                for descriptor in descriptors
            ],
            mem_type=mem_type,
        )

    def _release_transfer(self, transfer_id: int, state: _TransferState) -> bool:
        release_xfer = getattr(self.nixl_agent, "release_xfer_handle", None)
        if release_xfer is None:
            return False
        try:
            released = release_xfer(state.handle) is not False
        except Exception:
            now = time.monotonic()
            if now >= state.next_release_log_at:
                logger.warning(
                    "NIXL transfer release failed for %r",
                    state.handle,
                    exc_info=True,
                )
                state.next_release_log_at = now + _RELEASE_LOG_INTERVAL_SECONDS
            return False
        if not released:
            return False
        self._active_transfers.pop(transfer_id, None)
        return True

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=_JOIN_TIMEOUT_SECONDS):
            raise RuntimeError("KVCR progress thread did not start")
        self.raise_if_failed()

    def submit(self, item: object) -> None:
        self.raise_if_failed()
        self._submissions.put(item)

    def take_completed(self) -> list[object]:
        self.raise_if_failed()
        completed: list[object] = []
        while len(completed) < self._batch_size:
            try:
                completed.append(self._completed.get_nowait())
            except queue.Empty:
                break
        return completed

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    def is_quiescent(self) -> bool:
        """Report whether nothing native can still touch backend resources.

        A stopped thread is not enough: teardown leaves the loop as soon as a
        transfer or registration will not release, so all three must be empty.
        """
        if self._thread.is_alive():
            return False
        return not (
            self._active_transfers or self._in_flight_ops or self._memory_registrations
        )

    def close(self) -> None:
        if self._thread.is_alive():
            self._submissions.put(_STOP)
            # An interrupt can cut join() short, so the recheck below is
            # what callers rely on, not that join() returned.
            self._thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                raise RuntimeError("KVCR progress thread did not stop")
        self.raise_if_failed()

    def _run(self) -> None:
        try:
            self._initialize_nixl()
            # Let KVCR backends initialize NIXL resources before common
            # memory registration.
            self._initialize(self)
            self._register_memory_regions()
            self._capture_agent_metadata()
            self._ready.set()
            while not self._stop_requested:
                if not self._run_one_iteration():
                    time.sleep(_IDLE_WAIT_SECONDS)
        except BaseException as error:
            self._failure = error
        finally:
            try:
                self._close_progress_ops()
                if self._in_flight_ops or self._active_transfers:
                    raise RuntimeError(
                        "KVCR native operations are not quiescent at progress close"
                    )
                # Backend-specific registrations and common NIXL registrations
                # may still be referenced by those operations.  Close neither
                # tier unless the operation drain above proved quiescence.
                self._close()
                self._close_nixl()
            except BaseException as error:
                if self._failure is None:
                    self._failure = error
            self._ready.set()

    def _close_progress_ops(self) -> None:
        deadline = time.monotonic() + _CLOSE_TIMEOUT_SECONDS
        while self._in_flight_ops:
            for op_id, op in list(self._in_flight_ops.items()):
                if op.close(self):
                    self._in_flight_ops.pop(op_id, None)
            if not self._in_flight_ops:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("KVCR progress operations did not close")
            time.sleep(_IDLE_WAIT_SECONDS)

    def _run_one_iteration(self) -> bool:
        published = self._publish_completed(self._batch_size)
        submissions: list[object] = []
        while len(submissions) < self._batch_size:
            try:
                item = self._submissions.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                self._stop_requested = True
                break
            submissions.append(item)

        backend_items: list[object] = []
        for item in submissions:
            if isinstance(item, _ProgressOp):
                self._in_flight_ops[item.op_id] = item
            else:
                backend_items.append(item)

        events, backend_work = self._poll(self, backend_items)
        completed_ops: list[object] = []
        for op_id, op in list(self._in_flight_ops.items()):
            done, op_work = op.progress(self, events.pop(op_id, None))
            backend_work |= op_work
            if done and self._in_flight_ops.get(op_id) is op:
                self._in_flight_ops.pop(op_id)
                completed_ops.append(op)
        completed = self._flush()
        completed.extend(completed_ops)
        self._completed_backlog.extend(completed)
        published += self._publish_completed(self._batch_size - published)
        return bool(submissions) or backend_work or bool(completed) or published > 0

    def _publish_completed(self, limit: int) -> int:
        count = 0
        while self._completed_backlog and count < limit:
            self._completed.put(self._completed_backlog.popleft())
            count += 1
        return count

    def _initialize_nixl(self) -> None:
        if self._nixl_agent is None and self._nixl_agent_name is not None:
            if self._nixl_listen_port is None:
                raise RuntimeError("KVCR NIXL listen port is not configured")
            self._nixl_agent = nixl_agent(
                self._nixl_agent_name,
                nixl_agent_config(
                    num_threads=4,
                    capture_telemetry=True,
                    enable_listen_thread=True,
                    listen_port=self._nixl_listen_port,
                ),
            )

    def _dram_capable_backends(self) -> list[str]:
        """The instantiated backends that can carry a DRAM transfer.

        With a file backend created for G3, an unpinned DRAM copy is NIXL's
        choice across every created backend -- and file backends advertise
        DRAM_SEG for their memory side while requiring FILE_SEG on the other,
        so a memory-to-memory copy must not ride them. Read per transfer,
        because G3 creates its backend after the agent exists. An empty list
        keeps NIXL's own selection.
        """
        backend_mems = getattr(self._nixl_agent, "backend_mems", None)
        if not isinstance(backend_mems, dict) or not backend_mems:
            return []
        capable = [
            name
            for name, kinds in backend_mems.items()
            if any("DRAM" in str(kind).upper() for kind in kinds)
            and not any("FILE" in str(kind).upper() for kind in kinds)
        ]
        # Only a strict subset is worth pinning; otherwise leave the choice
        # exactly as NIXL would have made it.
        return capable if capable and len(capable) < len(backend_mems) else []

    def _register_memory_regions(self) -> None:
        if self._nixl_agent is None:
            return
        for address, size in self._memory_regions:
            self._memory_registrations.append(
                self._nixl_agent.register_memory(
                    [(address, size, 0, "")],
                    mem_type="DRAM",
                )
            )

    def _capture_agent_metadata(self) -> None:
        get_agent_metadata = getattr(self._nixl_agent, "get_agent_metadata", None)
        if get_agent_metadata is not None:
            self._nixl_agent_metadata = get_agent_metadata()

    def _close_nixl(self) -> None:
        if self._in_flight_ops:
            raise RuntimeError("cannot close NIXL with in-flight operations")
        if self._active_transfers:
            raise RuntimeError("cannot close NIXL with active transfers")
        failure: BaseException | None = None
        pending_registrations: list[Any] = []
        if self._nixl_agent is not None:
            for registration in reversed(self._memory_registrations):
                try:
                    released = (
                        self._nixl_agent.deregister_memory(registration) is not False
                    )
                except BaseException as error:
                    released = False
                    if failure is None:
                        failure = error
                if not released:
                    pending_registrations.append(registration)
        self._memory_registrations = list(reversed(pending_registrations))
        if not self._memory_registrations:
            self._nixl_agent_metadata = None
            self._nixl_agent = None
        if failure is not None:
            raise failure
        if self._memory_registrations:
            raise RuntimeError("NIXL memory registrations did not close")
