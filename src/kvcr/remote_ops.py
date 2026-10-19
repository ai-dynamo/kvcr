# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Remote transfer operations and wire-format helpers."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from itertools import chain
from typing import TYPE_CHECKING, Any

import msgspec

from . import _nvtx
from .core import TRANSFER_BLOCKS_METRIC, logger
from .progress import _KVCRProgress, _ProgressOp, _RegionMaps, _TransferRef
from .types import BlockKey, OpHandle, PinHandle

if TYPE_CHECKING:
    from .remote_fw_dram import _RemoteFWDram


class _PeerMessage(msgspec.Struct, kw_only=True, omit_defaults=True):
    target_agent: str | msgspec.UnsetType = msgspec.UNSET
    target_agent_metadata: bytes | None = None
    target_regions: _RegionMaps | msgspec.UnsetType = msgspec.UNSET
    sender_incarnation: str | None = None
    sender_control_endpoint: str | None = None
    source_control_endpoint: str | None = None


class _StartWrite(_PeerMessage, tag="start_write"):
    op_handle: OpHandle
    remaining_timeout_ms: float
    keys: tuple[BlockKey, ...]
    dst_descriptors: tuple[tuple[_TransferRef, ...], ...]
    key_layout: tuple[str, ...] = ()
    source_incarnation: str | None = None


class _TargetMetadata(_PeerMessage, tag="target_metadata"):
    pass


class _TargetMetadataAck(_PeerMessage, tag="target_metadata_ack"):
    op_handle: OpHandle | None = None


class _WriteRefused(_PeerMessage, tag="write_refused"):
    op_handle: OpHandle
    reason: str | None = None


class _WriteProbe(_PeerMessage, tag="write_probe"):
    op_handle: OpHandle
    source_incarnation: str | None = None


class _WriteProbeAck(_PeerMessage, tag="write_probe_ack"):
    op_handle: OpHandle
    terminal: bool
    dead_incarnation: str | None = None


_ControlMessage = (
    _StartWrite
    | _TargetMetadata
    | _TargetMetadataAck
    | _WriteRefused
    | _WriteProbe
    | _WriteProbeAck
)
_CONTROL_DECODER = msgspec.msgpack.Decoder(_ControlMessage)


class _TargetPullState(Enum):
    START_WRITE = auto()
    WAITING_WRITE_DONE = auto()
    WAITING_TERMINAL = auto()
    QUARANTINED = auto()
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
    # Keys sent to the source; keys also includes remembered misses.
    ordered_keys: tuple[BlockKey, ...] = ()
    dst_descriptors: tuple[tuple[_TransferRef, ...], ...] = ()
    request_id: str | None = None
    key_layout: tuple[str, ...] = ()
    success: bool = False
    completed_keys: set[BlockKey] = field(default_factory=set)
    probe_sent: bool = False
    source_incarnation: str | None = None
    uncertain: bool = False
    retried: bool = False

    def progress(
        self, progress: _KVCRProgress, event: object | None
    ) -> tuple[bool, bool]:
        backend = self._backend
        now = backend._kvcr._clock()
        cancelled = isinstance(event, Mapping) and event.get("cancelled", False)
        scope = "remote_fetch" if self.local_fill else "remote_deliver"
        if (
            self.state is _TargetPullState.WAITING_WRITE_DONE
            and not self.retried
            and isinstance(event, Mapping)
            and isinstance(event.get("retry_metadata"), str)
        ):
            # Pre-submit refusal: nothing was written, so resend in this poll.
            logger.debug(
                "KVCR_EVENT metadata_refresh_retry op=%d endpoint=%s incarnation=%s",
                self.op_id[1],
                self.remote_ctrl_ep,
                event["retry_metadata"],
            )
            self.source_incarnation = event["retry_metadata"]
            self.retried = True
            self.state = _TargetPullState.START_WRITE
        if self.state is _TargetPullState.START_WRITE:
            if not self.retried:
                self.source_incarnation = backend._dangling_ops.sources.get(
                    self.remote_ctrl_ep
                )
            else:
                # A stale ACK from the dead peer may have re-acked the endpoint.
                backend._metadata_acked_sources.discard(self.remote_ctrl_ep)
            if now >= self.deadline:
                self.success = False
                self.state = _TargetPullState.FINISHED
                backend._record_progress_duration(scope, self.started_at, "failed")
                return True, True
            sent = backend._send_control(
                progress,
                self.remote_ctrl_ep,
                _StartWrite(
                    op_handle=self.op_id[1],
                    remaining_timeout_ms=(self.deadline - now) * 1000,
                    source_incarnation=self.source_incarnation,
                    keys=self.ordered_keys,
                    dst_descriptors=self.dst_descriptors,
                    key_layout=self.key_layout if self.local_fill else (),
                ),
            )
            if not sent:
                self.success = False
                self.state = _TargetPullState.FINISHED
                backend._record_progress_duration(scope, self.started_at, "failed")
                return True, True
            self.state = _TargetPullState.WAITING_WRITE_DONE
            return False, True

        if (
            self.state
            in (
                _TargetPullState.WAITING_WRITE_DONE,
                _TargetPullState.WAITING_TERMINAL,
                _TargetPullState.QUARANTINED,
            )
            and isinstance(event, Mapping)
            and event.get("terminal", True) is True
        ):
            if self.uncertain:
                backend._dangling_ops.report_target(self, "quiesced")
                self.state = _TargetPullState.FINISHED
                return True, True
            # After cancellation, native success only permits cleanup.
            success = (
                event.get("success") is True
                and not cancelled
                and self.state is _TargetPullState.WAITING_WRITE_DONE
                and (not self.local_fill or now < self.deadline)
            )
            try:
                if success:
                    completed_indices = _notification_completed_indices(
                        event, len(self.ordered_keys)
                    )
                    completed_keys = {
                        self.ordered_keys[index] for index in completed_indices
                    }
                else:
                    completed_keys = set()
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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "KVCR_EVENT target_transfer_completed scope=%s request_id=%s "
                    "op=%d source=%s blocks=%d bytes=%d result=%s",
                    scope,
                    self.request_id,
                    self.op_id[1],
                    self.remote_ctrl_ep,
                    len(completed_keys),
                    backend._kvcr._descriptor_bytes(
                        descriptor
                        for key, descriptors in zip(
                            self.ordered_keys, self.dst_descriptors
                        )
                        if key in completed_keys
                        for descriptor in descriptors
                    ),
                    result,
                )
            return True, True

        if self.state is _TargetPullState.QUARANTINED:
            # Retain this tombstone indefinitely until quiescence is proven;
            # elapsed time alone cannot make its destination safe to reuse.
            return False, backend._dangling_ops.poll_target(progress, self)

        if now >= self.deadline or (
            cancelled and self.state is _TargetPullState.WAITING_WRITE_DONE
        ):
            if self.state is _TargetPullState.WAITING_TERMINAL:
                self.state = _TargetPullState.QUARANTINED
                backend._dangling_ops.poll_target(progress, self)
                backend._record_progress_duration(scope, self.started_at, "failed")
                return False, True
            self.state = _TargetPullState.WAITING_TERMINAL
            config = backend._kvcr.config
            self.deadline += (
                config.abandon_timeout_ms - config.operation_timeout_ms
            ) / 1000
            backend._invalidate_control_peer(self.remote_ctrl_ep)
            self.probe_sent = backend._dangling_ops.probe(progress, self)
            if self.local_fill:
                backend._progress_outbound.append(
                    replace(
                        self,
                        completed_keys=set(self.completed_keys),
                    )
                )
            return False, True
        if self.state is _TargetPullState.WAITING_TERMINAL and not self.probe_sent:
            # Retry failed enqueues without extending the grace period.
            self.probe_sent = backend._dangling_ops.probe(progress, self)
            return False, self.probe_sent
        return False, False

    def close(self, progress: _KVCRProgress) -> bool:
        # Shutdown must not release destinations still awaiting a remote write.
        return self.state in (_TargetPullState.START_WRITE, _TargetPullState.FINISHED)


@dataclass(slots=True)
class _NvtxState:
    requested_blocks: int = -1
    target_incarnation: str | None = None
    trace: _nvtx._LifecycleTrace | None = field(default=None, repr=False, compare=False)
    trace_initialized: bool = False
    failure_reason: _nvtx.Reason = _nvtx.Reason.UNKNOWN


@dataclass
class _SourceWriteOp(_RemoteOp):
    """Progress-owned NIXL write from prepared source descriptors."""

    state: _SourceWriteState
    remote_agent: str
    op_handle: int
    source_keys: tuple[BlockKey, ...]
    dst_descriptors: tuple[tuple[_TransferRef, ...], ...]
    _backend: "_RemoteFWDram" = field(repr=False, compare=False)
    framework_pins: set[PinHandle] = field(default_factory=set)
    src_descriptors: tuple[tuple[_TransferRef, ...], ...] = ()
    transfer_id: int | None = None
    success: bool = False
    completed_indices: tuple[int, ...] = ()
    route: tuple[str, int] = ("", 0)
    nvtx_state: _NvtxState | None = None

    def progress(
        self, progress: _KVCRProgress, _event: object | None
    ) -> tuple[bool, bool]:
        backend = self._backend
        nvtx_state = self.nvtx_state
        if nvtx_state is not None and not nvtx_state.trace_initialized:
            nvtx_state.trace_initialized = True
            if backend._nvtx is not None:
                nvtx_state.trace = backend._nvtx.source(self, backend._kvcr)
        trace = nvtx_state.trace if nvtx_state is not None else None
        observed_work = False
        write_id = (self.route[0], self.op_handle)
        status = backend._dangling_ops.source_writes[write_id]
        if self.transfer_id is None:
            source_responsive = backend._dangling_ops.check_source_progress()
            if not source_responsive or status.cancel_requested:
                self.state = _SourceWriteState.NOTIFY_FAILURE
            if (
                self.state is _SourceWriteState.NOTIFY_FAILURE
                or backend._kvcr._clock() >= self.deadline
            ):
                if trace is not None:
                    if status.cancel_requested:
                        failure_reason = _nvtx.Reason.CANCELLED
                    elif not source_responsive:
                        failure_reason = _nvtx.Reason.SOURCE_STALLED
                    elif self.state is _SourceWriteState.NOTIFY_FAILURE:
                        failure_reason = nvtx_state.failure_reason
                    else:
                        failure_reason = _nvtx.Reason.DEADLINE
                    if failure_reason is _nvtx.Reason.UNKNOWN:
                        failure_reason = trace.failure_reason
                backend._send_write_done(
                    progress, self.remote_agent, self.op_handle, False
                )
                self.success = False
                self.state = _SourceWriteState.FINISHED
                backend._record_progress_duration(
                    "source_write", self.started_at, "failed"
                )
                backend._dangling_ops.finish_source(self)
                if trace is not None:
                    trace.complete_source(False, 0, reason=failure_reason)
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
                if trace is not None:
                    trace.reject("source.write.refused", _nvtx.Reason.ROUTE_CHANGED)
                return False, True
            if not self.src_descriptors:
                backend._send_write_done(
                    progress, self.remote_agent, self.op_handle, True
                )
                self.success = True
                self.state = _SourceWriteState.FINISHED
                backend._record_progress_duration(
                    "source_write", self.started_at, "failed"
                )
                backend._dangling_ops.finish_source(self)
                if trace is not None:
                    trace.complete_source(False, 0, reason=_nvtx.Reason.UNKNOWN)
                return True, True
            submit_started_at = backend._kvcr._timer()
            status.submitted = True
            try:
                transfer_id, submitted = progress.submit_transfer(
                    "WRITE",
                    tuple(chain.from_iterable(self.src_descriptors)),
                    tuple(chain.from_iterable(self.dst_descriptors)),
                    remote_side_agent=self.remote_agent,
                    backend=backend._options.backend,
                    notif_msg=_write_done_notif(
                        self.op_handle,
                        True,
                        completed_indices=self.completed_indices,
                    ),
                    capture_telemetry=backend._telemetry_enabled,
                    trace=trace,
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
                backend._dangling_ops.finish_source(self)
                if trace is not None:
                    trace.complete_source(False, 0, reason=_nvtx.Reason.CREATE_ERROR)
                return True, True

        transfer_id = self.transfer_id
        if transfer_id is None:
            raise RuntimeError(f"KVCR source operation {self.op_id!r} lost transfer")
        if self.state is not _SourceWriteState.CANCEL_PENDING and (
            status.cancel_requested or backend._kvcr._clock() >= self.deadline
        ):
            self.state = _SourceWriteState.CANCEL_PENDING
            if trace is not None:
                trace.mark(
                    "source.write.cancel_requested",
                    once=True,
                    status=_nvtx.Status.CANCELLED
                    if status.cancel_requested
                    else _nvtx.Status.TIMEOUT,
                    reason=_nvtx.Reason.CANCELLED
                    if status.cancel_requested
                    else _nvtx.Reason.DEADLINE,
                )
            observed_work = True
        cancelling = self.state is _SourceWriteState.CANCEL_PENDING
        transfer_result = backend._dangling_ops.poll_source(
            progress, self, cancelling=cancelling
        )
        if transfer_result is None:
            if progress._active_transfers[transfer_id].outcome is False:
                self.state = _SourceWriteState.CANCEL_PENDING
            return False, observed_work
        self.transfer_id = None
        success, telemetry = transfer_result
        self.success = success and not cancelling
        if self.success:
            backend._record_transfer_telemetry(telemetry)
            backend._record_progress_counter(
                TRANSFER_BLOCKS_METRIC,
                len(self.keys),
                ("source_write",),
            )
        else:
            backend._send_write_done(progress, self.remote_agent, self.op_handle, False)
        result = "success" if self.success else "failed"
        backend._record_progress_duration("source_write", self.started_at, result)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "KVCR_EVENT source_transfer_completed op=%d source_op=%d target=%s "
                "blocks=%d bytes=%d result=%s",
                self.op_handle,
                self.op_id[1],
                self.route[0],
                len(self.source_keys) if self.success else 0,
                backend._kvcr._descriptor_bytes(
                    chain.from_iterable(self.src_descriptors)
                )
                if self.success
                else 0,
                result,
            )
        self.state = _SourceWriteState.FINISHED
        backend._dangling_ops.finish_source(self)
        if trace is not None:
            trace.complete_source(self.success, len(self.source_keys))
        return True, True

    def close(self, progress: _KVCRProgress) -> bool:
        trace = self.nvtx_state.trace if self.nvtx_state is not None else None
        if self.transfer_id is not None:
            if (
                progress.poll_transfer(self.transfer_id, require_completion=True)
                is None
            ):
                if trace is not None:
                    trace.mark(
                        "source.write.shutdown_unresolved",
                        once=True,
                        status=_nvtx.Status.UNRESOLVED,
                        reason=_nvtx.Reason.SHUTDOWN,
                    )
                return False
            self.transfer_id = None
        self._backend._send_write_done(
            progress, self.remote_agent, self.op_handle, False
        )
        self._backend._dangling_ops.finish_source(self)
        if trace is not None:
            trace.complete_source(
                False,
                0,
                reason=trace.failure_reason
                if trace.failure_reason is not _nvtx.Reason.UNKNOWN
                else _nvtx.Reason.SHUTDOWN,
            )
        return True


_NOTIF_PREFIX = b"KVCR:"


def _write_done_notif(
    op_handle: OpHandle,
    success: bool,
    completed_indices: tuple[int, ...] = (),
    *,
    terminal: bool = True,
) -> bytes:
    payload: dict[str, Any] = {
        "type": "write_done",
        "op_handle": op_handle,
        "success": success,
    }
    if success:
        payload["completed_indices"] = completed_indices
    if not terminal:
        payload["terminal"] = False
    return _NOTIF_PREFIX + msgspec.msgpack.encode(payload)


def _notification_completed_indices(
    payload: Mapping[str, Any], requested_count: int
) -> list[int]:
    indices = payload.get("completed_indices")
    if (
        not isinstance(indices, list)
        or len(indices) > requested_count
        or any(
            type(index) is not int or not 0 <= index < requested_count
            for index in indices
        )
        or len(set(indices)) != len(indices)
    ):
        raise TypeError("invalid completed indices")
    return indices


def _decode_notif(notif: bytes) -> dict[str, Any] | None:
    if not isinstance(notif, bytes) or not notif.startswith(_NOTIF_PREFIX):
        return None
    try:
        payload = msgspec.msgpack.decode(notif[len(_NOTIF_PREFIX) :])
    except msgspec.DecodeError:
        return None
    return payload if isinstance(payload, dict) else None
