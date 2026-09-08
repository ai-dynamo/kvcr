# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Remote DRAM path used by KVCR.

This module owns the G2-specific control protocol, framework pinning, and
NIXL source writes. Sources may be local KVCR DRAM or framework memory;
destinations may be local KVCR DRAM or framework memory. Progress owns
accepted operations; KVCR owns block state.

Target: hint/query -> fetch/deliver -> start_write -> write_done.
Source: start_write -> local claim/framework pin -> write -> write_done.
"""

import math
import time
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

import msgspec

from .config import KeyAdapter, RemoteFWDramOptions
from .core import (
    DURATION_METRIC,
    TRANSFER_BLOCKS_METRIC,
    TRANSFER_BYTES_METRIC,
    logger,
)
from .progress import _KVCRProgress, _Op, _OpId, _ProgressOp
from .types import (
    BlockKey,
    MemDescriptor,
    OpEntryResult,
    OpEntryStatus,
    OpHandle,
    PinHandle,
    PinRequestId,
    PinResult,
)

if TYPE_CHECKING:
    from .core import _KVCRCore


_MEM_DESCRIPTORS_TYPE = tuple[MemDescriptor, ...]


@dataclass(slots=True)
class _FwMemResidency:
    descriptor: MemDescriptor
    pin_handle: PinHandle


@dataclass(frozen=True)
class _RequestHint:
    source: str | None
    block_hashes: frozenset[int]
    submitted_at: float | None
    failed: bool = False


class _TargetPullState(Enum):
    START_WRITE = auto()
    WAITING_WRITE_DONE = auto()
    WAITING_TERMINAL = auto()
    FINISHED = auto()


class _SourceWriteState(Enum):
    READY_TO_WRITE = auto()
    NOTIFY_FAILURE = auto()
    WRITING = auto()
    CANCEL_PENDING = auto()
    FINISHED = auto()


@dataclass
class _RemoteOp(_ProgressOp):
    """Timing shared by remote framework-DRAM progress operations."""

    started_at: float | None
    deadline: float


@dataclass
class _TargetPullOp(_RemoteOp):
    """Target-side remote write into framework or KVCR-owned memory."""

    state: _TargetPullState
    local_fill: bool
    remote_ctrl_ep: str
    _backend: "_RemoteFWDram" = field(repr=False, compare=False)
    ordered_keys: tuple[BlockKey, ...] = ()
    dst_descriptors: tuple[MemDescriptor, ...] = ()
    request_id: str | None = None
    success: bool = False
    completed_keys: set[BlockKey] = field(default_factory=set)

    def progress(
        self, progress: _KVCRProgress, event: object | None
    ) -> tuple[bool, bool]:
        backend = self._backend
        now = backend._kvcr._clock()
        scope = "remote_fetch" if self.local_fill else "remote_deliver"
        if self.state is _TargetPullState.START_WRITE:
            if now >= self.deadline:
                self.success = False
                self.state = _TargetPullState.FINISHED
                backend._record_progress_duration(scope, self.started_at, "failed")
                return True, True
            sent = backend._send_control(
                progress,
                self.remote_ctrl_ep,
                {
                    "type": "start_write",
                    "op_handle": self.op_id[1],
                    "remaining_timeout_ms": (self.deadline - now) * 1000,
                    "keys": list(self.ordered_keys),
                    "dst_descriptors": self.dst_descriptors,
                },
            )
            if not sent:
                self.success = False
                self.state = _TargetPullState.FINISHED
                backend._record_progress_duration(scope, self.started_at, "failed")
                return True, True
            self.state = _TargetPullState.WAITING_WRITE_DONE
            return False, True

        if self.state in (
            _TargetPullState.WAITING_WRITE_DONE,
            _TargetPullState.WAITING_TERMINAL,
        ) and isinstance(event, Mapping):
            success = bool(event.get("success")) and (
                not self.local_fill
                or (
                    self.state is _TargetPullState.WAITING_WRITE_DONE
                    and now < self.deadline
                )
            )
            try:
                if success and "completed_count" in event:
                    completed_count = _notification_completed_count(
                        event, len(self.ordered_keys)
                    )
                    if set(self.ordered_keys) != self.keys:
                        raise TypeError("missing ordered keys")
                    completed_keys = set(self.ordered_keys[:completed_count])
                else:
                    completed_keys = set(self.keys) if success else set()
            except TypeError:
                success = False
                completed_keys = set()
            all_completed = success and completed_keys == self.keys
            self.success = success
            self.completed_keys = completed_keys
            self.state = _TargetPullState.FINISHED
            if completed_keys:
                backend._record_progress_counter(
                    TRANSFER_BLOCKS_METRIC,
                    len(completed_keys),
                    (scope,),
                )
            result = (
                "success"
                if all_completed
                else "partial"
                if completed_keys
                else "failed"
            )
            backend._record_progress_duration(scope, self.started_at, result)
            return True, True

        if now >= self.deadline:
            if self.state is _TargetPullState.WAITING_TERMINAL:
                return False, False
            # TODO: Safely retry operations that span primary-to-Guard promotion. A
            # deadline says the write has not been reported, not that the source cannot
            # still make it, so the destination is held rather than handed back. Until
            # then, callers must submit a new request.
            self.state = _TargetPullState.WAITING_TERMINAL
            if self.local_fill:
                backend._progress_outbound.append(
                    replace(
                        self,
                        keys=set(self.keys),
                        completed_keys=set(self.completed_keys),
                    )
                )
            backend._invalidate_control_peer(self.remote_ctrl_ep)
            return False, True
        return False, False


@dataclass
class _SourcePinOp(_Op):
    """Main-thread source acquisition for a pending write."""

    started_at: float | None
    deadline: float
    remote_agent: bytes
    op_handle: int
    ordered_keys: tuple[BlockKey, ...]
    dst_descriptors: tuple[MemDescriptor, ...]
    route: tuple[str, int] = ("", 0)
    framework_pins: set[PinHandle] = field(default_factory=set)
    pending_pin_ids: set[PinRequestId] = field(default_factory=set)
    framework_acquire_attempted: bool = False


@dataclass
class _PendingFrameworkSources:
    """Source descriptors waiting for framework pin results."""

    pending_pins: tuple[PinRequestId, ...]
    framework_pins: set[PinHandle]


@dataclass
class _SourceWriteOp(_RemoteOp):
    """Progress-owned NIXL write from prepared source descriptors."""

    state: _SourceWriteState
    remote_agent: bytes
    op_handle: int
    ordered_keys: tuple[BlockKey, ...]
    dst_descriptors: tuple[MemDescriptor, ...]
    _backend: "_RemoteFWDram" = field(repr=False, compare=False)
    framework_pins: set[PinHandle] = field(default_factory=set)
    src_descriptors: tuple[MemDescriptor, ...] = ()
    transfer_id: int | None = None
    success: bool = False
    completed_count: int = 0
    route: tuple[str, int] = ("", 0)

    def progress(
        self, progress: _KVCRProgress, _event: object | None
    ) -> tuple[bool, bool]:
        backend = self._backend
        observed_work = False
        if self.transfer_id is None:
            if (
                self.state is _SourceWriteState.NOTIFY_FAILURE
                or backend._kvcr._clock() >= self.deadline
            ):
                backend._send_write_done(
                    progress, self.remote_agent, self.op_handle, False
                )
                self.success = False
                self.state = _SourceWriteState.FINISHED
                backend._record_progress_duration(
                    "source_write", self.started_at, "failed"
                )
                return True, True
            if self.state is not _SourceWriteState.READY_TO_WRITE:
                raise RuntimeError(f"KVCR source operation {self.op_id!r} is not ready")
            route_name, route_generation = self.route
            if route_name and (
                backend._route_generation.get(route_name, 0) != route_generation
            ):
                # Re-checked here, on the thread that submits: the route can be
                # replaced between queueing and this write, and NIXL hands the
                # same handle back for a reused name. The target hears a
                # refusal instead of receiving the dead generation's bytes.
                self.state = _SourceWriteState.NOTIFY_FAILURE
                return False, True
            submit_started_at = backend._kvcr._timer()
            try:
                transfer_id, submitted = progress.submit_transfer(
                    "WRITE",
                    self.src_descriptors,
                    self.dst_descriptors[: self.completed_count],
                    remote_side_agent=self.remote_agent,
                    backend=backend._options.backend,
                    notif_msg=_write_done_notif(
                        self.op_handle,
                        True,
                        completed_count=self.completed_count,
                    ),
                    capture_telemetry=backend._telemetry_enabled,
                )
                self.transfer_id = transfer_id
                self.state = (
                    _SourceWriteState.WRITING
                    if submitted
                    else _SourceWriteState.CANCEL_PENDING
                )
                result = "success" if submitted else "failed"
                backend._record_progress_duration(
                    "transfer_submit", submit_started_at, result
                )
                if not submitted:
                    logger.warning(
                        "KVCR start_write submission failed for op=%d",
                        self.op_handle,
                    )
                observed_work = True
            except Exception:
                logger.warning(
                    "KVCR start_write failed for op=%d",
                    self.op_handle,
                    exc_info=True,
                )
                backend._record_progress_duration(
                    "transfer_submit", submit_started_at, "failed"
                )
                self.success = False
                backend._send_write_done(
                    progress, self.remote_agent, self.op_handle, False
                )
                backend._record_progress_duration(
                    "source_write", self.started_at, "failed"
                )
                self.state = _SourceWriteState.FINISHED
                return True, True

        transfer_id = self.transfer_id
        if transfer_id is None:
            raise RuntimeError(f"KVCR source operation {self.op_id!r} lost transfer")
        if (
            self.state is not _SourceWriteState.CANCEL_PENDING
            and backend._kvcr._clock() >= self.deadline
        ):
            self.state = _SourceWriteState.CANCEL_PENDING
            observed_work = True
        transfer_result = progress.poll_transfer(
            transfer_id,
            cancellation_requested=self.state is _SourceWriteState.CANCEL_PENDING,
        )
        if transfer_result is None:
            return False, observed_work
        self.transfer_id = None
        success, telemetry = transfer_result
        if success:
            backend._record_transfer_telemetry(telemetry)
            backend._record_progress_counter(
                TRANSFER_BLOCKS_METRIC,
                len(self.keys),
                ("source_write",),
            )
        else:
            backend._send_write_done(progress, self.remote_agent, self.op_handle, False)
        result = "success" if success else "failed"
        backend._record_progress_duration("source_write", self.started_at, result)
        self.success = success
        self.state = _SourceWriteState.FINISHED
        return True, True

    def close(self, progress: _KVCRProgress) -> bool:
        if self.transfer_id is not None:
            if not progress.cancel_transfer(self.transfer_id):
                return False
            self.transfer_id = None
        return True


@dataclass(frozen=True)
class _TargetMetadataRequest:
    """Request eager peer metadata setup from progress."""

    endpoint: str


@dataclass
class _ProgressUpdate:
    """Bounded progress-only telemetry and gauges returned to main."""

    metrics: list[tuple[str, str, int | float, tuple[str, ...]]]
    connected_remote_count: int


@dataclass
class _PendingPinWait:
    """One framework pin request and the G2 operations waiting on it."""

    request: PinRequestId
    keys: tuple[BlockKey, ...]
    started_at: float | None
    op_ids: set[_OpId] = field(default_factory=set)


class _RemoteFWDram:
    """G2 remote framework-memory implementation behind KVCR."""

    def __init__(
        self,
        kvcr: "_KVCRCore",
        options: RemoteFWDramOptions,
        key_adapter: KeyAdapter | None,
    ) -> None:
        if options.metadata_retry_interval_ms <= 0:
            raise ValueError("metadata_retry_interval_ms must be positive")
        if not options.backend:
            raise ValueError("remote framework DRAM NIXL backend must be non-empty")
        self._kvcr = kvcr
        self._options = options
        self._key_adapter = key_adapter

        # Main-thread state: request hints, framework pins, and progress state.
        self._closed = False
        self._request_hints: dict[str, _RequestHint] = {}
        self._connected_remote_count = 0
        self._source_pin_ops: dict[_OpId, _SourcePinOp] = {}
        self._pending_pin_ops: dict[PinRequestId, _PendingPinWait] = {}
        self._pending_pin_keys: dict[BlockKey, set[PinRequestId]] = {}
        # Pins retained while their source operations execute.
        self._fw_pins_by_op: dict[_OpId, set[PinHandle]] = {}

        # Progress-thread state: G2 control and outbound events.
        self._progress_outbound: list[object] = []
        self._progress_metrics: list[tuple[str, str, int | float, tuple[str, ...]]] = []
        self._telemetry_enabled = kvcr.config.enable_telemetry
        self._remote_agents_by_target: dict[str, tuple[bytes, bytes]] = {}
        # Bumped whenever a name's route is replaced: NIXL hands the same
        # handle back for a reused name, so queued operations from the dead
        # generation must be fenced by number, not by handle.
        self._route_generation: dict[str, int] = {}
        self._published_remote_count = 0
        self._metadata_acked_sources: set[str] = set()
        self._metadata_retry_after: dict[str, float] = {}
        self._refused_writes: dict[_OpId, dict[str, bool]] = {}
        self._next_source_op_id = 1
        self._control = kvcr.framework_control

    # -------------------------------------------------------------------------
    # Backend interface used by KVCR.
    # -------------------------------------------------------------------------

    def submit_hint(
        self,
        src: str | None,
        block_hashes: frozenset[int],
        request_id: str | None,
    ) -> None:
        kvcr = self._kvcr
        if request_id is not None:
            previous = self._request_hints.get(request_id)
            if previous is not None and previous.source != src:
                self._request_hints[request_id] = replace(previous, failed=True)
                return
            self._request_hints[request_id] = _RequestHint(
                source=src,
                block_hashes=block_hashes,
                submitted_at=(
                    previous.submitted_at
                    if previous is not None and previous.submitted_at is not None
                    else kvcr._timer()
                ),
            )
        if src is not None and self._options.eager_ctrl_connect:
            kvcr._progress.submit(_TargetMetadataRequest(src))

    def query(self, key: BlockKey, request_id: str) -> bool:
        """Return whether ``key`` matches the request's remote hint."""
        request_hint = self._request_hints.get(request_id)
        adapter = self._key_adapter
        if (
            request_hint is None
            or request_hint.source is None
            or request_hint.failed
            or adapter is None
        ):
            return False
        if (
            not self._options.opportunistic_query
            and adapter.decode(key) not in request_hint.block_hashes
        ):
            return False
        return True

    def _start_target_pull(
        self,
        blocks: Mapping[BlockKey, MemDescriptor],
        request_id: str | None,
        deadline: float,
        op_handle: OpHandle,
        *,
        local_fill: bool,
    ) -> bool:
        kvcr = self._kvcr
        started_at = kvcr._timer()
        keys = tuple(blocks)
        scope = "remote_fetch" if local_fill else "remote_deliver"
        current_hint = (
            self._request_hints.get(request_id) if request_id is not None else None
        )
        if current_hint is not None and current_hint.failed:
            kvcr._record_duration(scope, started_at, "failed")
            return False

        if current_hint is None or current_hint.source is None:
            kvcr._record_duration(scope, started_at, "failed")
            return False
        kvcr._record_duration("hint_wait", current_hint.submitted_at, "complete")
        if request_id is not None:
            self._request_hints[request_id] = replace(current_hint, submitted_at=None)

        op = _TargetPullOp(
            state=_TargetPullState.START_WRITE,
            local_fill=local_fill,
            keys=set(keys),
            started_at=started_at,
            deadline=deadline,
            op_id=("target", op_handle),
            remote_ctrl_ep=current_hint.source,
            _backend=self,
            ordered_keys=keys,
            dst_descriptors=tuple(blocks[key] for key in keys),
            request_id=request_id,
        )
        kvcr._add_block_dependencies(op, new_operation=True)
        kvcr._progress.submit(op)
        return True

    def deliver(
        self,
        op_handle: OpHandle,
        blocks: Mapping[BlockKey, MemDescriptor],
        request_id: str | None,
        *,
        deadline: float,
    ) -> None:
        kvcr = self._kvcr
        if not blocks:
            kvcr._complete(op_handle, {})
            return
        if self._start_target_pull(
            blocks,
            request_id,
            deadline,
            op_handle,
            local_fill=False,
        ):
            return
        kvcr._complete(
            op_handle,
            {key: OpEntryResult(OpEntryStatus.FAILED) for key in blocks},
        )

    def fetch(
        self,
        blocks: Mapping[BlockKey, MemDescriptor],
        request_id: str | None,
        deadline: float,
        *,
        op_handle: OpHandle,
    ) -> bool:
        return self._start_target_pull(
            blocks,
            request_id,
            deadline,
            op_handle,
            local_fill=True,
        )

    def poll_main(self, items: Collection[object]) -> None:
        for item in items:
            if isinstance(item, _SourcePinOp):
                self._start_source_pin(item)
            elif isinstance(item, _SourceWriteOp):
                if item.state is _SourceWriteState.FINISHED:
                    self._fw_pins_by_op.pop(item.op_id, None)
                    self._kvcr._remove_block_dependencies(item)
                    if item.success:
                        self._kvcr._record_access(
                            self._kvcr._local_dram_sources_by_op.get(item.op_id, ())
                        )
                    self._kvcr._release_local_dram_sources(item.op_id)
                    self._release_framework_pins(item.framework_pins)
                else:
                    raise RuntimeError(
                        f"KVCR source operation {item.op_id!r} returned to main "
                        f"in state {item.state.name}"
                    )
            elif isinstance(item, _TargetPullOp):
                if item.state is _TargetPullState.WAITING_TERMINAL:
                    if not item.local_fill:
                        raise RuntimeError(
                            "non-local target pull is waiting for terminal state"
                        )
                    self._kvcr._discard_local_dram_fill(item.ordered_keys)
                elif item.state is _TargetPullState.FINISHED:
                    self._finish_target_pull(item)
                else:
                    raise RuntimeError(
                        f"KVCR target operation {item.op_id!r} returned to main "
                        f"in state {item.state.name}"
                    )
            elif isinstance(item, _ProgressUpdate):
                self._apply_progress_update(item)
            else:
                raise TypeError(f"unsupported KVCR main item: {type(item)!r}")
        if not self._closed:
            self._process_pending_pin_results()
            now = self._kvcr._clock()
            for op_id, op in list(self._source_pin_ops.items()):
                if now >= op.deadline:
                    logger.warning("KVCR operation %r expired", op_id)
                    self._expire_source_pin(op_id, op)

    def discard_hint(self, request_id: str) -> None:
        self._request_hints.pop(request_id, None)

    def _fail_request_hint(self, request_id: str | None) -> None:
        if request_id is None:
            return
        request_hint = self._request_hints.get(request_id)
        if request_hint is not None:
            self._request_hints[request_id] = replace(request_hint, failed=True)

    def close_main(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._request_hints.clear()
        for op_id, op in list(self._source_pin_ops.items()):
            self._remove_source_pin(op_id, op)

        for request_id in list(self._pending_pin_ops):
            wait = self._remove_pending_pin_state(request_id)
            if wait is None:
                continue
            self._record_pending_pin_wait(wait, "cancelled")
            self._cancel_pending_pin(wait.request)
        for _, result in self._poll_framework_pin_results():
            self._discard_pin_result(result)

        self._fw_pins_by_op.clear()
        self._release_framework_pins(tuple(self._kvcr._framework_pin_keys))
        # During normal operation a failed release is logged and retried later.
        # At shutdown there is no later, so an unreleased pin has to be
        # reported: the framework is still holding memory on KVCR's behalf.
        if self._kvcr._framework_pin_keys:
            raise RuntimeError(
                "KVCR could not release framework pins: "
                f"{sorted(self._kvcr._framework_pin_keys)}"
            )

    # -------------------------------------------------------------------------
    # Target side: request and receive remote KV.
    # -------------------------------------------------------------------------

    def _finish_target_pull(self, op: _TargetPullOp) -> None:
        kvcr = self._kvcr
        kvcr._remove_block_dependencies(op)
        completed_keys = op.completed_keys if op.success else set()
        if completed_keys != op.keys:
            self._fail_request_hint(op.request_id)
        if op.local_fill:
            completed_count = len(completed_keys)
            if completed_count:
                kvcr._complete_local_dram_fill(
                    op.ordered_keys[:completed_count], success=True
                )
            if completed_count < len(op.ordered_keys):
                kvcr._complete_local_dram_fill(
                    op.ordered_keys[completed_count:], success=False
                )
            return
        kvcr._complete(
            cast(OpHandle, op.op_id[1]),
            {
                key: OpEntryResult(
                    OpEntryStatus.SUCCESS
                    if bool(op.success) and key in completed_keys
                    else OpEntryStatus.FAILED
                )
                for key in op.keys
            },
        )

    # -------------------------------------------------------------------------
    # Progress-thread lifecycle and control transport.
    # -------------------------------------------------------------------------

    def initialize_progress(self, _progress: _KVCRProgress) -> None:
        initialize_control = getattr(self._control, "initialize", None)
        if initialize_control is not None:
            initialize_control()

    def poll_progress(
        self, progress: _KVCRProgress, submissions: list[object]
    ) -> tuple[dict[object, object], bool]:
        observed_work = bool(submissions)
        for item in submissions:
            if isinstance(item, _TargetMetadataRequest):
                if (
                    item.endpoint in self._metadata_acked_sources
                    or self._kvcr._clock()
                    < self._metadata_retry_after.get(item.endpoint, 0)
                ):
                    continue
                self._send_control(progress, item.endpoint, {"type": "target_metadata"})
            else:
                raise TypeError(f"unsupported KVCR progress item: {type(item)!r}")

        observed_work |= self._process_control_messages(progress)
        # A real notification outranks a refusal for the same operation.
        events = {**self._refused_writes, **self._poll_notifications(progress)}
        self._refused_writes.clear()
        observed_work |= bool(events)
        return events, observed_work

    def flush_progress(self) -> list[object]:
        remote_count = len(self._remote_agents_by_target)
        if self._progress_metrics or remote_count != self._published_remote_count:
            self._progress_outbound.insert(
                0,
                _ProgressUpdate(
                    self._progress_metrics,
                    remote_count,
                ),
            )
            self._progress_metrics = []
            self._published_remote_count = remote_count
        outbound = self._progress_outbound
        self._progress_outbound = []
        return outbound

    def close_progress(self) -> None:
        close_control = getattr(self._control, "close", None)
        if close_control is not None:
            close_control()

    def _process_control_messages(self, progress: _KVCRProgress) -> bool:
        if self._control is None:
            return False
        try:
            messages = self._control.recv()
        except Exception:
            logger.warning("KVCR control receive failed", exc_info=True)
            return False
        handled = False
        for message in messages:
            try:
                payload = msgspec.msgpack.decode(message)
            except (TypeError, msgspec.DecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            message_type = payload.get("type")
            if not isinstance(message_type, str):
                continue
            handled_at = self._kvcr._timer()
            if message_type == "target_metadata":
                self._handle_target_metadata(progress, payload)
            elif message_type == "target_metadata_ack":
                self._handle_target_metadata_ack(payload)
            elif message_type == "write_refused":
                self._handle_write_refused(progress, payload)
            elif message_type == "start_write":
                self._handle_start_write(progress, payload)
            else:
                continue
            handled = True
            self._record_progress_duration(
                f"control_{message_type}", handled_at, "handled"
            )
        return handled

    def _send_control(
        self,
        progress: _KVCRProgress,
        endpoint: str,
        payload: dict[str, Any],
    ) -> bool:
        kvcr = self._kvcr
        started_at = kvcr._timer()
        if self._control is None or progress.nixl_agent_metadata is None:
            self._record_progress_duration("control_enqueue", started_at, "failed")
            return False
        payload["target_agent"] = kvcr.nixl_agent_name
        sender_endpoint = getattr(self._control, "endpoint", None)
        if isinstance(sender_endpoint, str):
            payload.setdefault("sender_control_endpoint", sender_endpoint)
        message_type = payload.get("type")
        if message_type != "target_metadata_ack":
            payload["source_control_endpoint"] = endpoint
        includes_metadata = (
            message_type not in ("target_metadata_ack", "write_refused")
            and endpoint not in self._metadata_acked_sources
            and (
                message_type in ("target_metadata", "start_write")
                or kvcr._clock() >= self._metadata_retry_after.get(endpoint, 0)
            )
        )
        if includes_metadata:
            payload["target_agent_metadata"] = progress.nixl_agent_metadata
        try:
            sent = self._control.send(endpoint, msgspec.msgpack.encode(payload))
        except Exception:
            logger.warning("KVCR control send failed to %s", endpoint, exc_info=True)
            sent = False
        self._record_progress_duration(
            "control_enqueue", started_at, "success" if sent else "failed"
        )
        if not sent:
            self._invalidate_control_peer(endpoint)
            return False
        if includes_metadata:
            self._metadata_retry_after[endpoint] = (
                kvcr._clock() + self._options.metadata_retry_interval_ms / 1000
            )
        return True

    def _invalidate_control_peer(self, endpoint: str) -> None:
        self._metadata_acked_sources.discard(endpoint)
        self._metadata_retry_after.pop(endpoint, None)

    def _handle_target_metadata(
        self, progress: _KVCRProgress, payload: dict[str, Any]
    ) -> None:
        try:
            target_agent, _ = self._remote_agent(progress, payload)
            self._ack_target_metadata(progress, payload, target_agent)
        except Exception:
            return

    def _handle_write_refused(
        self, progress: _KVCRProgress, payload: dict[str, Any]
    ) -> None:
        endpoint = payload.get("sender_control_endpoint")
        op_handle = payload.get("op_handle")
        if not isinstance(endpoint, str) or type(op_handle) is not int:
            return
        op = progress._in_flight_ops.get(("target", op_handle))
        if getattr(op, "remote_ctrl_ep", None) != endpoint:
            return
        # Whatever answered is not the process our metadata was loaded into, so
        # drop the snapshot and let the next request republish it.
        self._invalidate_control_peer(endpoint)
        # A refusal is only sent before a write is submitted, so it is as terminal as a
        # failed write_done and safe to act on even from WAITING_TERMINAL. Deliberately
        # unauthenticated: a "resend it" signal on a channel already trusted to let a
        # start_write make a source write.
        self._refused_writes[("target", op_handle)] = {"success": False}

    def _handle_target_metadata_ack(self, payload: dict[str, Any]) -> None:
        source_endpoint = payload.get("sender_control_endpoint")
        if not isinstance(source_endpoint, str) or not source_endpoint:
            return
        self._metadata_acked_sources.add(source_endpoint)
        self._metadata_retry_after.pop(source_endpoint, None)

    def _ack_target_metadata(
        self,
        progress: _KVCRProgress,
        payload: Mapping[str, Any],
        target_agent: str,
    ) -> None:
        if not isinstance(payload.get("target_agent_metadata"), bytes):
            return
        target_control_endpoint = payload.get("sender_control_endpoint")
        if not isinstance(target_control_endpoint, str) or not target_control_endpoint:
            return
        source_control_endpoint = payload.get("source_control_endpoint")
        if not isinstance(source_control_endpoint, str) or not source_control_endpoint:
            return
        response: dict[str, Any] = {
            "type": "target_metadata_ack",
            "sender_control_endpoint": source_control_endpoint,
        }
        if not self._send_control(progress, target_control_endpoint, response):
            # Kept, route and cache both: an operation this start_write queued
            # still transfers over this route, and the peer re-sends its
            # metadata until the ack lands -- matching bytes reuse the entry,
            # changed bytes replace it through the refresh path.
            return

    # -------------------------------------------------------------------------
    # Source side: progress parses, main pins, then progress writes to the target.
    # -------------------------------------------------------------------------

    def _handle_start_write(
        self, progress: _KVCRProgress, payload: dict[str, Any]
    ) -> None:
        started_at = self._kvcr._timer()
        received_at = self._kvcr._clock()
        try:
            op_handle = int(payload["op_handle"])
        except (KeyError, TypeError, ValueError):
            logger.warning("KVCR malformed start_write: missing op_handle")
            return
        try:
            remaining_timeout_ms = payload["remaining_timeout_ms"]
            if (
                isinstance(remaining_timeout_ms, bool)
                or not isinstance(remaining_timeout_ms, (int, float))
                or not math.isfinite(remaining_timeout_ms)
                or remaining_timeout_ms <= 0
            ):
                raise TypeError("invalid remaining_timeout_ms")
            keys = _message_keys(payload)
            dst_descriptors = tuple(
                self._kvcr._normalize_descriptors([descriptor])
                for descriptor in msgspec.convert(
                    payload["dst_descriptors"], type=_MEM_DESCRIPTORS_TYPE
                )
            )
        except (KeyError, TypeError, ValueError, msgspec.ValidationError):
            logger.warning("KVCR malformed start_write op=%d", op_handle)
            self._notify_start_write_failure(progress, payload, op_handle)
            return
        if not keys or len(keys) != len(dst_descriptors):
            logger.warning("KVCR malformed start_write op=%d", op_handle)
            self._notify_start_write_failure(progress, payload, op_handle)
            return

        remaining_timeout_ms = min(
            float(remaining_timeout_ms),
            self._kvcr.config.operation_timeout_ms,
        )
        deadline = received_at + remaining_timeout_ms / 1000
        try:
            fallback_target = dst_descriptors[0].end_point_name
            target_agent, remote_agent = self._remote_agent(
                progress, payload, fallback_target=fallback_target
            )
            self._ack_target_metadata(progress, payload, target_agent)
        except Exception:
            logger.warning(
                "KVCR start_write setup failed for op=%d", op_handle, exc_info=True
            )
            self._notify_start_write_failure(progress, payload, op_handle)
            return

        op_id = ("source", self._next_source_op_id)
        self._next_source_op_id += 1
        self._progress_outbound.append(
            _SourcePinOp(
                op_id=op_id,
                keys=set(keys),
                started_at=started_at,
                deadline=deadline,
                remote_agent=remote_agent,
                op_handle=op_handle,
                ordered_keys=keys,
                dst_descriptors=dst_descriptors,
                route=(target_agent, self._route_generation.get(target_agent, 0)),
            )
        )

    def _submit_prepared_source_write(
        self,
        op_id: _OpId,
        source_pin: _SourcePinOp,
        *,
        force_failure: bool = False,
    ) -> None:
        kvcr = self._kvcr
        route_name, route_generation = source_pin.route
        if route_name and (
            self._route_generation.get(route_name, 0) != route_generation
        ):
            # The name was re-routed while this operation queued. NIXL hands
            # the same handle back for a reused name, so submitting would
            # write the dead generation's destinations through the new route.
            force_failure = True
        local_sources = kvcr._claim_local_dram_sources(
            source_pin.op_id, source_pin.ordered_keys
        )
        framework_sources = {
            key: record.fw_mem.descriptor
            for key in source_pin.ordered_keys
            if key not in local_sources
            and (record := kvcr._block_record_map.get(key)) is not None
            and record.fw_mem is not None
        }
        sources = {} if force_failure else {**framework_sources, **local_sources}
        completed_keys: tuple[BlockKey, ...] = ()
        for index, key in enumerate(source_pin.ordered_keys):
            if key not in sources:
                break
            completed_keys = source_pin.ordered_keys[: index + 1]

        kvcr._release_local_dram_sources(
            source_pin.op_id, local_sources.keys() - set(completed_keys)
        )
        relevant_pins = {
            record.fw_mem.pin_handle
            for key in completed_keys
            if key not in local_sources
            and (record := kvcr._block_record_map.get(key)) is not None
            and record.fw_mem is not None
        }
        # Every pin the write reads through, not just the ones this operation acquired:
        # fw_mem is per block, so holding another operation's pin is what stops its
        # release from unpinning it mid-read.
        framework_pins = set(relevant_pins)
        unused_pins = source_pin.framework_pins - framework_pins
        source_pin.framework_pins.clear()
        self._source_pin_ops.pop(op_id, None)
        kvcr._remove_block_dependencies(source_pin)

        source_write = _SourceWriteOp(
            state=(
                _SourceWriteState.READY_TO_WRITE
                if completed_keys
                else _SourceWriteState.NOTIFY_FAILURE
            ),
            keys=set(completed_keys or source_pin.ordered_keys),
            started_at=source_pin.started_at,
            deadline=source_pin.deadline,
            op_id=source_pin.op_id,
            remote_agent=source_pin.remote_agent,
            op_handle=source_pin.op_handle,
            ordered_keys=source_pin.ordered_keys,
            dst_descriptors=source_pin.dst_descriptors,
            route=source_pin.route,
            _backend=self,
            framework_pins=framework_pins,
            src_descriptors=tuple(sources[key] for key in completed_keys),
            completed_count=len(completed_keys),
        )
        kvcr._add_block_dependencies(source_write, new_operation=True)
        self._fw_pins_by_op[source_write.op_id] = set(source_write.framework_pins)
        kvcr._progress.submit(source_write)
        self._release_framework_pins(unused_pins)

    def _notify_start_write_failure(
        self,
        progress: _KVCRProgress,
        payload: Mapping[str, Any],
        op_handle: OpHandle,
    ) -> None:
        try:
            _, remote_agent = self._remote_agent(progress, payload)
        except Exception:
            # A promoted Guard adopts the dead primary's control endpoint but
            # not its peer table, so there is no NIXL route home to report on.
            # Refusing over the control channel is the only way this operation
            # ever reaches a terminal.
            logger.warning("KVCR refusing unresolvable start_write op=%d", op_handle)
            reply_to = payload.get("sender_control_endpoint")
            reflected_self = payload.get("source_control_endpoint")
            if isinstance(reply_to, str) and isinstance(reflected_self, str):
                self._send_control(
                    progress,
                    reply_to,
                    {
                        "type": "write_refused",
                        "sender_control_endpoint": reflected_self,
                        "op_handle": op_handle,
                    },
                )
            return
        self._send_write_done(progress, remote_agent, op_handle, False)

    # Shared framework-pin coordination.

    def _poll_framework_pin_results(self) -> Iterator[tuple[PinRequestId, PinResult]]:
        try:
            yield from self._kvcr._poll_pin_results_callback()
        except Exception:
            logger.warning("KVCR framework pin result polling failed", exc_info=True)

    def _process_pending_pin_results(self) -> None:
        kvcr = self._kvcr
        for request, result in self._poll_framework_pin_results():
            wait = self._remove_pending_pin_state(request)
            op_ids = wait.op_ids if wait is not None else set()
            ops = [
                (op_id, op)
                for op_id in op_ids
                if (op := self._source_pin_ops.get(op_id)) is not None
                and request in op.pending_pin_ids
            ]
            if not ops:
                self._discard_pin_result(result)
                continue

            now = kvcr._clock()
            expired_ops = [(op_id, op) for op_id, op in ops if now >= op.deadline]
            active_ops = [(op_id, op) for op_id, op in ops if now < op.deadline]
            if not active_ops:
                self._record_pending_pin_wait(wait, "timeout")
                self._discard_pin_result(result)
                for op_id, op in expired_ops:
                    op.pending_pin_ids.discard(request)
                    self._expire_source_pin(op_id, op)
                continue

            pin_handle: PinHandle | None = None
            if result is not None and wait is not None:
                pin_handle = self._install_framework_pin(wait.keys, result)
            self._record_pending_pin_wait(
                wait, "success" if pin_handle is not None else "failed"
            )
            if pin_handle is not None:
                for _, op in active_ops:
                    op.framework_pins.add(pin_handle)
            for op_id, op in expired_ops:
                op.pending_pin_ids.discard(request)
                self._expire_source_pin(op_id, op)
            for op_id, op in active_ops:
                self._resolve_pending_source_pin(
                    op_id,
                    op,
                    request,
                )

            if pin_handle is not None:
                self._release_framework_pins({pin_handle})

    def _resolve_pending_source_pin(
        self,
        op_id: _OpId,
        op: _SourcePinOp,
        request_id: PinRequestId,
    ) -> None:
        kvcr = self._kvcr
        if request_id not in op.pending_pin_ids:
            return
        op.pending_pin_ids.discard(request_id)
        if op.pending_pin_ids:
            return

        relevant_pins: set[PinHandle] = set()
        for key in op.keys:
            record = kvcr._block_record_map.get(key)
            if record is not None and record.fw_mem is not None:
                relevant_pins.add(record.fw_mem.pin_handle)
        unused_pins = op.framework_pins - relevant_pins
        op.framework_pins = relevant_pins
        self._release_framework_pins(unused_pins)
        self._resume_source_pin(op_id, op)

    def _expire_source_pin(self, op_id: _OpId, op: _SourcePinOp) -> None:
        self._cancel_pending_pin_for_op(op_id, op, result="timeout")
        self._submit_prepared_source_write(op_id, op, force_failure=True)

    def _start_source_pin(self, op: _SourcePinOp) -> None:
        kvcr = self._kvcr
        kvcr._add_block_dependencies(op, new_operation=True)
        self._source_pin_ops[op.op_id] = op
        self._resume_source_pin(op.op_id, op)

    def _resume_source_pin(self, op_id: _OpId, op: _SourcePinOp) -> None:
        if self._source_pin_ops.get(op_id) is not op:
            return
        kvcr = self._kvcr
        if kvcr._clock() >= op.deadline:
            self._submit_prepared_source_write(op.op_id, op, force_failure=True)
            return

        local_sources = kvcr._claim_local_dram_sources(op.op_id, op.ordered_keys)
        unresolved_keys = tuple(
            key for key in op.ordered_keys if key not in local_sources
        )
        if unresolved_keys and not op.framework_acquire_attempted:
            op.framework_acquire_attempted = True
            framework_sources = self._acquire_framework_sources(unresolved_keys)
            if isinstance(framework_sources, _PendingFrameworkSources):
                op.framework_pins.update(framework_sources.framework_pins)
                for request in framework_sources.pending_pins:
                    self._register_pending_pin(request, op.op_id)
                return
            if framework_sources is not None:
                _, framework_pins = framework_sources
                op.framework_pins.update(framework_pins)

        self._submit_prepared_source_write(op_id, op)

    def _remove_source_pin(self, op_id: _OpId, op: _SourcePinOp) -> None:
        self._cancel_pending_pin_for_op(op_id, op)
        self._source_pin_ops.pop(op_id, None)
        self._kvcr._remove_block_dependencies(op)
        self._kvcr._release_local_dram_sources(op.op_id)
        framework_pins = set(op.framework_pins)
        op.framework_pins.clear()
        self._release_framework_pins(framework_pins)

    def _register_pending_pin(self, request: PinRequestId, op_id: _OpId) -> None:
        wait = self._pending_pin_ops.get(request)
        if wait is None:
            logger.warning(
                "KVCR operation %r referenced unknown pending pin %d",
                op_id,
                request,
            )
            return
        wait.op_ids.add(op_id)
        op = self._source_pin_ops.get(op_id)
        if op is not None:
            op.pending_pin_ids.add(request)

    def _remove_pending_pin_state(
        self, request_id: PinRequestId
    ) -> _PendingPinWait | None:
        wait = self._pending_pin_ops.pop(request_id, None)
        if wait is None:
            return None
        for key in wait.keys:
            request_ids = self._pending_pin_keys.get(key)
            if request_ids is None:
                continue
            request_ids.discard(request_id)
            if not request_ids:
                self._pending_pin_keys.pop(key, None)
        return wait

    def _find_pending_pins(
        self, keys: Collection[BlockKey]
    ) -> tuple[list[PinRequestId], set[BlockKey]]:
        pending_pins: list[PinRequestId] = []
        pending_ids: set[PinRequestId] = set()
        covered_keys: set[BlockKey] = set()
        for key in keys:
            request_ids = self._pending_pin_keys.get(key)
            if not request_ids:
                continue
            for request_id in sorted(request_ids):
                wait = self._pending_pin_ops.get(request_id)
                if wait is not None:
                    covered_keys.add(key)
                    if request_id not in pending_ids:
                        pending_ids.add(request_id)
                        pending_pins.append(wait.request)
                    break
        return pending_pins, covered_keys

    def _cancel_pending_pin_for_op(
        self,
        op_id: _OpId,
        op: _SourcePinOp,
        *,
        result: str = "cancelled",
    ) -> None:
        for request_id in tuple(op.pending_pin_ids):
            wait = self._pending_pin_ops.get(request_id)
            if wait is None:
                continue
            wait.op_ids.discard(op_id)
            if not wait.op_ids:
                wait = self._remove_pending_pin_state(request_id)
                if wait is not None:
                    self._record_pending_pin_wait(wait, result)
                    self._cancel_pending_pin(wait.request)
        op.pending_pin_ids.clear()

    def _cancel_pending_pin(self, request: PinRequestId) -> None:
        try:
            if self._kvcr._cancel_pin_request_callback is not None:
                self._kvcr._cancel_pin_request_callback(request)
        except Exception:
            logger.warning(
                "KVCR pending framework pin cancellation failed", exc_info=True
            )

    def _record_pending_pin_wait(
        self, wait: _PendingPinWait | None, result: str
    ) -> None:
        if wait is not None:
            self._kvcr._record_duration("framework_pin_wait", wait.started_at, result)

    def _discard_pin_result(self, result: PinResult) -> None:
        if result is None:
            return
        try:
            pin_handle = result[0]
        except (IndexError, TypeError):
            return
        if isinstance(pin_handle, str):
            self._try_release_pin(pin_handle)

    # Framework pin ownership.

    def _pin_framework_keys(self, keys: Collection[BlockKey]) -> PinRequestId | None:
        kvcr = self._kvcr
        if not keys:
            return None
        keys = tuple(keys)
        started_at = kvcr._timer()
        result = "failed"
        try:
            request = kvcr._request_pin_callback(keys)
            if request in self._pending_pin_ops:
                logger.warning("KVCR reused pin request id %d", request)
                return None
            wait = _PendingPinWait(request, keys, kvcr._timer())
            self._pending_pin_ops[request] = wait
            for key in keys:
                self._pending_pin_keys.setdefault(key, set()).add(request)
            result = "pending"
            return request
        except Exception:
            return None
        finally:
            kvcr._record_duration("source_acquire", started_at, result)

    def _install_framework_pin(
        self,
        keys: Collection[BlockKey],
        pin_result: tuple[PinHandle, Mapping[BlockKey, list[MemDescriptor] | None]],
    ) -> PinHandle | None:
        pin_handle: PinHandle | None = None
        try:
            pin_handle, descriptors = pin_result
            if not isinstance(pin_handle, str) or not isinstance(descriptors, Mapping):
                raise TypeError("invalid framework pin result")
            requested_keys = set(keys)
            if set(descriptors) != requested_keys:
                raise KeyError("request_pin returned incomplete descriptors")
            normalized = {
                key: (
                    None
                    if descriptor_list is None
                    else self._kvcr._normalize_descriptors(descriptor_list)
                )
                for key, descriptor_list in descriptors.items()
            }
            if not any(descriptor is not None for descriptor in normalized.values()):
                raise ValueError("request_pin returned no descriptors")
            pin_keys = self._kvcr._framework_pin_keys.setdefault(pin_handle, set())
            for key in keys:
                descriptor = normalized[key]
                if descriptor is None:
                    continue
                record = self._kvcr._block_record(key)
                if record.fw_mem is not None:
                    continue
                record.fw_mem = _FwMemResidency(descriptor, pin_handle)
                pin_keys.add(key)
            return pin_handle
        except Exception:
            if pin_handle is not None:
                self._try_release_pin(pin_handle)
            return None

    def _acquire_framework_sources(
        self,
        keys: tuple[BlockKey, ...],
    ) -> (
        tuple[dict[BlockKey, MemDescriptor], set[PinHandle]]
        | _PendingFrameworkSources
        | None
    ):
        kvcr = self._kvcr
        started_at = kvcr._timer()
        keys_to_pin: list[BlockKey] = []
        for key in keys:
            record = kvcr._block_record_map.get(key)
            residency = record.fw_mem if record is not None else None
            if residency is None:
                keys_to_pin.append(key)

        if not keys_to_pin:
            kvcr._record_duration("source_acquire", started_at, "reused")
        else:
            held_framework_pins = {
                residency.pin_handle
                for key in keys
                if (record := kvcr._block_record_map.get(key)) is not None
                and (residency := record.fw_mem) is not None
            }
            pending_pins, covered_keys = self._find_pending_pins(keys_to_pin)
            uncovered_keys = [key for key in keys_to_pin if key not in covered_keys]
            pin_request = self._pin_framework_keys(uncovered_keys)
            if pin_request is not None:
                pending_pins.append(pin_request)
            if pending_pins:
                return _PendingFrameworkSources(
                    pending_pins=tuple(pending_pins),
                    framework_pins=held_framework_pins,
                )

        descriptors: dict[BlockKey, MemDescriptor] = {}
        framework_pins: set[PinHandle] = set()
        for key in keys:
            record = kvcr._block_record_map.get(key)
            residency = record.fw_mem if record is not None else None
            if residency is None:
                continue
            descriptors[key] = residency.descriptor
            framework_pins.add(residency.pin_handle)
        if not descriptors:
            return None
        return descriptors, framework_pins

    def _release_framework_pins(self, framework_pins: Collection[PinHandle]) -> None:
        kvcr = self._kvcr
        for pin_handle in framework_pins:
            if any(
                pin_handle in op.framework_pins for op in self._source_pin_ops.values()
            ) or any(pin_handle in pins for pins in self._fw_pins_by_op.values()):
                continue
            pin_keys = kvcr._framework_pin_keys.get(pin_handle)
            if pin_keys is None:
                continue
            if not self._try_release_pin(pin_handle):
                continue
            kvcr._framework_pin_keys.pop(pin_handle, None)
            for key in pin_keys:
                record = kvcr._block_record_map.get(key)
                if (
                    record is not None
                    and record.fw_mem is not None
                    and record.fw_mem.pin_handle == pin_handle
                ):
                    record.fw_mem = None
                    kvcr._prune_block_record(key)

    def _try_release_pin(self, pin_handle: PinHandle) -> bool:
        try:
            released = self._kvcr._release_pin_callback(pin_handle)
        except Exception:
            logger.warning(
                "KVCR release_pin failed for pin=%r", pin_handle, exc_info=True
            )
            return False
        if released is False:
            logger.warning("KVCR release_pin failed for pin=%r", pin_handle)
            return False
        return True

    # NIXL peer and descriptor setup.

    def _remote_agent(
        self,
        progress: _KVCRProgress,
        payload: Mapping[str, Any],
        fallback_target: str | None = None,
    ) -> tuple[str, bytes]:
        kvcr = self._kvcr
        agent = progress.nixl_agent
        target_agent = payload.get("target_agent", fallback_target)
        if not isinstance(target_agent, str) or not target_agent:
            raise TypeError("missing target agent")
        target_metadata = payload.get("target_agent_metadata")
        cached = self._remote_agents_by_target.get(target_agent)
        if cached is not None:
            cached_metadata, remote_agent = cached
            if not isinstance(target_metadata, bytes) or (
                target_metadata == cached_metadata
            ):
                reused_at = kvcr._timer()
                self._record_progress_duration("peer_setup", reused_at, "reused")
                return target_agent, remote_agent
            # Same name, new metadata: the process behind the name was replaced,
            # and the cached route still points at the dead one. A route that
            # cannot be unloaded propagates: the retained entry retries next
            # time instead of silently keeping the dead destination.
            remove = getattr(agent, "remove_remote_agent", None)
            if remove is not None:
                remove(remote_agent)
            self._remote_agents_by_target.pop(target_agent, None)
            self._route_generation[target_agent] = (
                self._route_generation.get(target_agent, 0) + 1
            )
        started_at = kvcr._timer()
        try:
            if not isinstance(target_metadata, bytes):
                raise TypeError("missing target agent metadata")
            remote_agent = agent.add_remote_agent(target_metadata)
            if not isinstance(remote_agent, bytes) or not remote_agent:
                raise RuntimeError("add_remote_agent returned no agent name")
        except Exception:
            self._record_progress_duration("peer_setup", started_at, "failed")
            raise
        self._record_progress_duration("peer_setup", started_at, "connected")
        self._remote_agents_by_target[target_agent] = (target_metadata, remote_agent)
        return target_agent, remote_agent

    # Progress notifications, telemetry, and resource cleanup.

    def _poll_notifications(self, progress: _KVCRProgress) -> dict[object, object]:
        agent = progress.nixl_agent
        get_new_notifs = getattr(agent, "get_new_notifs", None)
        if get_new_notifs is None:
            return {}
        events: dict[object, object] = {}
        try:
            for notifs in get_new_notifs().values():
                for raw in notifs:
                    payload = _decode_notif(raw)
                    if payload is None or payload.get("type") != "write_done":
                        continue
                    try:
                        op_handle = int(payload["op_handle"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    events[("target", op_handle)] = payload
        except Exception:
            logger.warning("KVCR notification receive failed", exc_info=True)
            return {}
        return events

    def _record_transfer_telemetry(self, telemetry: Any | None) -> None:
        if not self._telemetry_enabled or telemetry is None:
            return
        try:
            self._progress_metrics.append(
                (
                    "histogram",
                    DURATION_METRIC,
                    telemetry.postDuration / 1e6,
                    ("transfer_post", "success"),
                )
            )
            self._progress_metrics.append(
                (
                    "histogram",
                    DURATION_METRIC,
                    telemetry.xferDuration / 1e6,
                    ("transfer", "success"),
                )
            )
            self._record_progress_counter(
                TRANSFER_BYTES_METRIC,
                telemetry.totalBytes,
                ("source_write",),
            )
        except Exception:
            logger.debug("KVCR failed to collect NIXL telemetry", exc_info=True)

    def _record_progress_duration(
        self, scope: str, started_at: float | None, result: str
    ) -> None:
        if not self._telemetry_enabled or started_at is None:
            return
        self._progress_metrics.append(
            (
                "histogram",
                DURATION_METRIC,
                time.monotonic() - started_at,
                (scope, result),
            )
        )

    def _record_progress_counter(
        self,
        name: str,
        value: int | float,
        labels: tuple[str, ...],
    ) -> None:
        if self._telemetry_enabled:
            self._progress_metrics.append(("counter", name, value, labels))

    def _apply_progress_update(self, update: _ProgressUpdate) -> None:
        self._connected_remote_count = update.connected_remote_count
        stats = self._kvcr._stats
        if stats is None:
            return
        for kind, name, value, labels in update.metrics:
            if kind == "histogram":
                stats.observe_histogram(name, value, labels)
            else:
                stats.increase_counter(name, value, labels)

    def _send_write_done(
        self,
        progress: _KVCRProgress,
        remote_agent: bytes,
        op_handle: OpHandle,
        success: bool,
    ) -> None:
        agent = progress.nixl_agent
        send_notif = getattr(agent, "send_notif", None)
        if send_notif is None:
            logger.warning(
                "KVCR write_done notification failed for op=%d: API unavailable",
                op_handle,
            )
            return
        try:
            result = send_notif(remote_agent, _write_done_notif(op_handle, success))
        except Exception:
            logger.warning(
                "KVCR write_done notification failed for op=%d",
                op_handle,
                exc_info=True,
            )
            return
        if result is False:
            logger.warning("KVCR write_done notification failed for op=%d", op_handle)


# Control wire-format helpers.

_NOTIF_PREFIX = b"KVCR:"


def _message_keys(payload: Mapping[str, Any]) -> tuple[BlockKey, ...]:
    raw_keys = payload["keys"]
    if (
        not isinstance(raw_keys, list)
        or not raw_keys
        or not all(isinstance(key, bytes) for key in raw_keys)
    ):
        raise TypeError("invalid keys")
    return tuple(BlockKey(key) for key in raw_keys)


def _write_done_notif(
    op_handle: OpHandle,
    success: bool,
    completed_count: int | None = None,
) -> bytes:
    payload: dict[str, Any] = {
        "type": "write_done",
        "op_handle": op_handle,
        "success": success,
    }
    if completed_count is not None:
        payload["completed_count"] = completed_count
    return _NOTIF_PREFIX + msgspec.msgpack.encode(payload)


def _notification_completed_count(
    payload: Mapping[str, Any], requested_count: int
) -> int:
    completed_count = payload.get("completed_count")
    if (
        isinstance(completed_count, bool)
        or not isinstance(completed_count, int)
        or not 0 <= completed_count <= requested_count
    ):
        raise TypeError("invalid completed count")
    return completed_count


def _decode_notif(notif: bytes) -> dict[str, Any] | None:
    if not isinstance(notif, bytes) or not notif.startswith(_NOTIF_PREFIX):
        return None
    try:
        payload = msgspec.msgpack.decode(notif[len(_NOTIF_PREFIX) :])
    except msgspec.DecodeError:
        return None
    return payload if isinstance(payload, dict) else None
