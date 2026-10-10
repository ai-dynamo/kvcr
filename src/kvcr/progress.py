# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Threaded progress and NIXL transfer lifecycle for KVCR backends."""

import logging
import math
import os
import queue
import select
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Annotated, Any

import msgspec
import numpy as np
from nixl import nixl_agent, nixl_agent_config, nixlRemoteDisconnectError

from .types import BlockKey, RegionDescriptor

logger = logging.getLogger(__name__)
_ACTIVE_WAIT_SECONDS = 0.0001
_IDLE_WAIT_SECONDS = 0.020
_OP_CLEANUP_WAIT_SECONDS = 0.001
_OP_CLEANUP_TIMEOUT_SECONDS = 5.0
_JOIN_TIMEOUT_SECONDS = 10.0
_STARTUP_TIMEOUT_SECONDS = 30.0
_RELEASE_LOG_INTERVAL_SECONDS = 1.0
_STOP = object()
_OpId = tuple[str, Any]
_RegionMaps = tuple[dict[str, RegionDescriptor], dict[str, RegionDescriptor]]


@dataclass(frozen=True)
class _MemDescriptor:
    """Raw memory addresses or file offsets for FILE transfers."""

    mem_type: str
    addr: int
    size: int
    device_Id: int


@dataclass(frozen=True, slots=True)
class _TransferRef:
    """An element in an agent's framework or KVCR-owned registered memory."""

    end_point_name: Annotated[str, msgspec.Meta(min_length=1)]
    element_index: Annotated[int, msgspec.Meta(ge=0)]
    label: str = ""
    framework: bool = True


def _resolve_region(
    ref: _TransferRef, regions: _RegionMaps, agent_name: str
) -> RegionDescriptor:
    if not isinstance(ref, _TransferRef):
        raise ValueError("memory transfers require _TransferRef entries")
    if ref.end_point_name != agent_name:
        raise ValueError("memory reference has the wrong owning agent")
    if type(ref.framework) is not bool:
        raise ValueError("memory reference framework selector must be a bool")
    if not isinstance(ref.label, str):
        raise ValueError("memory reference label must be a string")
    framework, kvcr = regions
    if ref.framework:
        region = framework.get(ref.label)
        if region is None and ":" in ref.label:
            region = framework.get(ref.label.partition(":")[0] + ":*")
    else:
        region = kvcr.get(ref.label.partition(":")[0])
    if region is None:
        owner = "framework" if ref.framework else "KVCR"
        raise ValueError(f"no {owner} registration for {ref.label!r}")
    if type(ref.element_index) is not int or not 0 <= ref.element_index < region.count:
        raise ValueError("memory reference element_index is outside its region")
    return region


@dataclass(slots=True)
class _TransferState:
    """Hold one progress-owned NIXL transfer handle."""

    handle: Any
    remote_side_agent: str
    capture_telemetry: bool = False
    outcome: bool | None = None
    telemetry: Any | None = None
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
        dram_backends: list[str] = [],
        batch_size: int = 64,
        nixl_agent_name: str | None = None,
        nixl_listen_port: int | None = None,
        memory_regions: _RegionMaps | None = None,
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
        self._disconnected_remote_agents: set[str] = set()
        self._next_transfer_id = 0
        self._nixl_listen_port = nixl_listen_port
        self._dram_backends = dram_backends
        self._memory_regions = (
            memory_regions if memory_regions is not None else ({}, {})
        )
        self._memory_registrations: list[Any] = []
        self._prepared: dict[
            str, dict[str, tuple[Any, tuple[dict[str, int], dict[str, int]]]]
        ] = {}
        self._catalogs: dict[str, tuple[str, _RegionMaps]] = {}
        self._nixl_agent_metadata: bytes | None = None
        self._submissions: queue.SimpleQueue[object] = queue.SimpleQueue()
        self._wake_lock = threading.Lock()
        self._wake_read: int | None = None
        self._wake_write: int | None = None
        self._idle_waiter: Callable[[float, int], None] | None = None
        self._idle_wait_seconds = _IDLE_WAIT_SECONDS
        self._completed: queue.SimpleQueue[object] = queue.SimpleQueue()
        self._completed_backlog: deque[object] = deque()
        self._in_flight_ops: dict[_OpId, _ProgressOp] = {}
        self._ready = threading.Event()
        self._prepare_ready = threading.Event()
        self._activate = threading.Event()
        self._close_requested = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="kvcr-progress",
        )
        self._failure: BaseException | None = None
        self._stop_requested = False
        self._startup_stage = "thread startup"

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
        require_completion: bool = False,
    ) -> tuple[bool, Any | None] | None:
        """Advance a transfer and return its result after releasing its handle."""
        state = self._active_transfers.get(transfer_id)
        if state is None:
            raise KeyError(f"unknown transfer {transfer_id}")
        agent = self.nixl_agent
        outcome = state.outcome
        if outcome is None or (require_completion and not outcome):
            try:
                xfer_state = agent.check_xfer_state(state.handle)
            except Exception as error:
                if isinstance(error, nixlRemoteDisconnectError):
                    self._disconnected_remote_agents.add(state.remote_side_agent)
                    # NIXL keeps this as the handle's final status; releasing it
                    # cancels any request still queued on the failed endpoint.
                    require_completion = False
                if state.outcome is not False:
                    logger.warning("NIXL transfer progress failed", exc_info=True)
                xfer_state = "ERR"
            # Releasing a pending NIXL/UCX handle can leave DMA running and lose
            # its completion signal. Remote writes retain it until actual DONE.
            if require_completion and xfer_state != "DONE":
                if xfer_state not in ("PROC", "PEND"):
                    state.outcome = False
                return None
            pending = xfer_state in ("PROC", "PEND")
            if pending and not cancellation_requested:
                return None
            outcome = xfer_state == "DONE" and state.outcome is not False
            if not pending:
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
        local_descriptors: Sequence[_TransferRef] | Sequence[_MemDescriptor],
        remote_descriptors: Sequence[_TransferRef] | Sequence[_MemDescriptor],
        *,
        remote_side_agent: str,
        backend: str | None = None,
        notif_msg: bytes = b"",
        capture_telemetry: bool = False,
    ) -> tuple[int, bool]:
        """Submit aligned local and remote descriptors to NIXL."""
        if not local_descriptors or not remote_descriptors:
            raise ValueError("NIXL transfer descriptors must be non-empty")
        if not isinstance(remote_side_agent, str) or not remote_side_agent:
            raise ValueError("NIXL remote-side agent must be non-empty")
        agent = self.nixl_agent
        if any(
            isinstance(descriptors[0], _MemDescriptor)
            and descriptors[0].mem_type == "FILE"
            for descriptors in (local_descriptors, remote_descriptors)
        ):
            handle = agent.initialize_xfer(
                operation,
                self._make_transfer_descriptors(local_descriptors),
                self._make_transfer_descriptors(remote_descriptors),
                remote_side_agent,
                notif_msg=notif_msg,
                backends=[backend] if backend else [],
            )
        else:
            local, local_indices = self._prepared_indices("", local_descriptors)
            remote, remote_indices = self._prepared_indices(
                remote_side_agent, remote_descriptors
            )
            handle = agent.make_prepped_xfer(
                operation,
                local,
                local_indices,
                remote,
                remote_indices,
                notif_msg=notif_msg,
                backends=[backend] if backend else [],
            )
        if handle is None:
            raise RuntimeError("NIXL transfer creation returned None")
        self._next_transfer_id += 1
        transfer_id = self._next_transfer_id
        state = _TransferState(handle, remote_side_agent, capture_telemetry)
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
        except Exception as error:
            if isinstance(error, nixlRemoteDisconnectError):
                self._disconnected_remote_agents.add(state.remote_side_agent)
            submitted = False
            logger.warning("NIXL transfer submission was ambiguous", exc_info=True)
        return transfer_id, submitted

    def cancel_transfer(self, transfer_id: int) -> bool:
        """Cancel if necessary and forget only after NIXL releases the handle."""
        state = self._active_transfers.get(transfer_id)
        if state is None:
            return True
        state.outcome = False
        return self._release_transfer(transfer_id, state)

    def _make_transfer_descriptors(self, descriptors: Sequence[_MemDescriptor]) -> Any:
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

    def prepare_memory(
        self,
        agent_name: str,
        regions: _RegionMaps,
        *,
        owner_name: str | None = None,
    ) -> None:
        """Initialize physical grids after registration or new metadata install."""
        if agent_name in self._prepared:
            raise RuntimeError("NIXL prepared memory is already initialized")
        owner = (
            owner_name
            if owner_name is not None
            else (agent_name or self.nixl_agent_name)
        )
        self._catalogs[agent_name] = (owner, regions)
        by_type: dict[str, list[tuple[bool, RegionDescriptor]]] = {}
        framework, kvcr = regions
        for is_framework, registered in ((True, framework), (False, kvcr)):
            for region in registered.values():
                by_type.setdefault(region.mem_type, []).append((is_framework, region))
        prepared = self._prepared.setdefault(agent_name, {})
        try:
            for mem_type, entries in by_type.items():
                framework_offsets: dict[str, int] = {}
                kvcr_offsets: dict[str, int] = {}
                offset = 0
                for is_framework, region in entries:
                    offsets = framework_offsets if is_framework else kvcr_offsets
                    offsets[region.label] = offset
                    offset += region.count
                rows = np.array(
                    [
                        (r.addr, r.size, r.device_Id, r.stride or r.size, r.count)
                        for _, r in entries
                    ],
                    dtype=np.uint64,
                )
                handle = self.nixl_agent.prep_xfer_dlist(
                    agent_name, rows, mem_type=mem_type, backends=self._dram_backends
                )
                if handle is None:
                    raise RuntimeError("prep_xfer_dlist returned None")
                prepared[mem_type] = (handle, (framework_offsets, kvcr_offsets))
        except BaseException:
            self.release_prepared(agent_name)
            raise
        if not prepared:
            self._prepared.pop(agent_name, None)
            self._catalogs.pop(agent_name, None)

    def release_prepared(self, agent_name: str) -> None:
        """Release a catalog only after every transfer using it has released."""
        if any(
            agent_name == "" or state.remote_side_agent == agent_name
            for state in self._active_transfers.values()
        ):
            raise RuntimeError("cannot release prepared memory with active transfers")
        prepared = self._prepared.get(agent_name, {})
        for mem_type, (handle, _) in list(prepared.items()):
            if self.nixl_agent.release_dlist_handle(handle) is False:
                raise RuntimeError("NIXL prepared memory did not close")
            del prepared[mem_type]
        self._prepared.pop(agent_name, None)
        self._catalogs.pop(agent_name, None)

    def _prepared_indices(
        self, agent_name: str, descriptors: Sequence[_TransferRef]
    ) -> tuple[Any, list[int]]:
        catalog = self._catalogs.get(agent_name)
        if catalog is None:
            raise ValueError(f"no prepared memory for agent {agent_name!r}")
        owner, regions = catalog
        first = _resolve_region(descriptors[0], regions, owner)
        handle, (framework_offsets, kvcr_offsets) = self._prepared[agent_name][
            first.mem_type
        ]
        indices = []
        for ref in descriptors:
            region = _resolve_region(ref, regions, owner)
            if region.mem_type != first.mem_type:
                raise ValueError("one NIXL descriptor list cannot mix memory types")
            offsets = framework_offsets if ref.framework else kvcr_offsets
            indices.append(offsets[region.label] + ref.element_index)
        return handle, indices

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
        self._activate.set()
        if not self._thread.is_alive():
            self._thread.start()
        if not self._ready.wait(timeout=_STARTUP_TIMEOUT_SECONDS):
            raise RuntimeError(
                "KVCR progress initialization timed out after "
                f"{_STARTUP_TIMEOUT_SECONDS:g}s (stage: {self._startup_stage})"
            )
        self.raise_if_failed()

    def prepare(self) -> None:
        """Create the agent and register memory without starting backends."""
        self._thread.start()
        if not self._prepare_ready.wait(timeout=_STARTUP_TIMEOUT_SECONDS):
            raise RuntimeError(
                "KVCR progress preparation timed out after "
                f"{_STARTUP_TIMEOUT_SECONDS:g}s (stage: {self._startup_stage})"
            )
        self.raise_if_failed()

    def submit(self, item: object) -> None:
        self.raise_if_failed()
        self._submissions.put(item)
        self._wake()

    def _wake(self) -> None:
        # Serialize with close: a stale fd could have been reused elsewhere.
        with self._wake_lock:
            if self._wake_write is not None:
                try:
                    os.write(self._wake_write, b"\0")
                except BlockingIOError:
                    pass  # A full pipe already guarantees a wakeup.

    def _wait_for_work(self) -> None:
        wake_fd = self._wake_read
        assert wake_fd is not None
        timeout = self._idle_wait_seconds
        if self._idle_waiter is not None:
            self._idle_waiter(timeout, wake_fd)
        else:
            poller = select.poll()
            poller.register(wake_fd, select.POLLIN)
            poller.poll(math.ceil(timeout * 1000))
        # Queue-before-signal plus a persistent readable byte covers submissions
        # arriving between the empty-queue check and the wait. Read only once;
        # a busy producer must not keep us draining signals instead of work.
        try:
            os.read(wake_fd, 4096)
        except BlockingIOError:
            pass

    def _close_wakeup(self) -> None:
        with self._wake_lock:
            descriptors = (self._wake_read, self._wake_write)
            self._wake_read = self._wake_write = None
            for descriptor in descriptors:
                if descriptor is not None:
                    os.close(descriptor)

    def take_completed(self) -> list[object]:
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
        transfer, prepared catalog or registration will not release.
        """
        if self._thread.is_alive():
            return False
        return not (
            self._active_transfers
            or self._in_flight_ops
            or self._prepared
            or self._memory_registrations
        )

    def close(self) -> None:
        if self._thread.is_alive():
            self._close_requested = True
            self._activate.set()
            self._submissions.put(_STOP)
            self._wake()
            # An interrupt can cut join() short, so the recheck below is
            # what callers rely on, not that join() returned.
            self._thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                raise RuntimeError("KVCR progress thread did not stop")
        self.raise_if_failed()

    def _log_startup_stage(self, stage: str) -> None:
        self._startup_stage = stage
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "KVCR_EVENT progress_startup_stage stage=%s agent=%s tid=%d "
                "monotonic_ns=%d thread_cpu_ns=%d",
                stage,
                self._nixl_agent_name,
                threading.get_native_id(),
                time.monotonic_ns(),
                time.thread_time_ns(),
            )

    def _run(self) -> None:
        try:
            # Allocate only on the owning thread; constructing an unused core
            # must not leak descriptors. Startup failures close these too.
            with self._wake_lock:
                self._wake_read, self._wake_write = os.pipe2(
                    os.O_NONBLOCK | os.O_CLOEXEC
                )
            self._log_startup_stage("NIXL agent initialization")
            self._initialize_nixl()
            if self._close_requested:
                return
            if not self._activate.is_set():
                self._log_startup_stage("memory registration")
                self._register_memory_regions()
                self._prepare_ready.set()
                self._activate.wait()
                if self._close_requested:
                    return
                self._log_startup_stage("backend initialization")
                self._initialize(self)
            else:
                # Preserve the normal ordering for backends that create NIXL
                # resources before common memory registration.
                self._log_startup_stage("backend initialization")
                self._initialize(self)
                self._log_startup_stage("memory registration")
                self._register_memory_regions()
                self._prepare_ready.set()
            self._log_startup_stage("agent metadata capture")
            self._capture_agent_metadata()
            self._log_startup_stage("ready")
            self._ready.set()
            while not self._stop_requested:
                if not self._run_one_iteration() and not self._stop_requested:
                    if self._in_flight_ops:
                        time.sleep(_ACTIVE_WAIT_SECONDS)
                    else:
                        self._wait_for_work()
        except BaseException as error:
            self._failure = error
        finally:
            self._prepare_ready.set()
            self._log_startup_stage("cleanup")
            try:
                try:
                    self._close_progress_ops()
                finally:
                    try:
                        self._completed_backlog.extend(self._flush())
                        self._publish_completed(sys.maxsize)
                    finally:
                        try:
                            self._close()
                        finally:
                            self._close_nixl()
            except BaseException as error:
                if self._failure is None:
                    self._failure = error
            self._close_wakeup()
            self._ready.set()

    def _close_progress_ops(self) -> None:
        self._stop_requested = True
        deadline = time.monotonic() + _OP_CLEANUP_TIMEOUT_SECONDS
        while self._in_flight_ops:
            events, _ = self._poll(self, [])
            for op_id, op in list(self._in_flight_ops.items()):
                closed = op.close(self)
                # Pending cleanup still needs timers and peer replies.
                if not closed:
                    closed, _ = op.progress(self, events.get(op_id))
                if closed:
                    self._in_flight_ops.pop(op_id, None)
            if not self._in_flight_ops:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("KVCR progress operations did not close")
            time.sleep(_OP_CLEANUP_WAIT_SECONDS)

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
                    backends=self._dram_backends,
                ),
            )

    def _register_memory_regions(self) -> None:
        if self._nixl_agent is None or not any(self._memory_regions):
            return
        for region in chain.from_iterable(
            regions.values() for regions in self._memory_regions
        ):
            # Assume the entire extent, including stride gaps, is valid to register.
            extent = region.size + (region.count - 1) * (region.stride or region.size)
            self._memory_registrations.append(
                self._nixl_agent.register_memory(
                    [(region.addr, extent, region.device_Id, "")],
                    mem_type=region.mem_type,
                )
            )
        self.prepare_memory("", self._memory_regions)
        self.prepare_memory(self.nixl_agent_name, self._memory_regions)

    def _capture_agent_metadata(self) -> None:
        get_agent_metadata = getattr(self._nixl_agent, "get_agent_metadata", None)
        if get_agent_metadata is not None:
            self._nixl_agent_metadata = get_agent_metadata()

    def _close_nixl(self) -> None:
        if self._active_transfers:
            raise RuntimeError("cannot close NIXL with active transfers")
        if self._in_flight_ops:
            raise RuntimeError("cannot close NIXL with unresolved operations")
        for agent_name in list(self._prepared):
            self.release_prepared(agent_name)
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
