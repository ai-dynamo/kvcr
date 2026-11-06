# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Remote transfer operations and wire-format helpers."""

import logging
from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from itertools import chain
from typing import TYPE_CHECKING, Any

import msgspec

from .. import _nvtx
from ..runtime.core import TRANSFER_BLOCKS_METRIC, UNSERVED_BLOCKS_METRIC, logger
from ..runtime.progress import (
    _FAILED_RESULT,
    _SUCCESS_RESULT,
    _KVCRProgress,
    _ProgressOp,
    _RegionMaps,
    _TransferRef,
)
from ..types import BlockKey, OpEntryResult, OpEntryStatus, OpHandle

if TYPE_CHECKING:
    from .local_dram_pools import _LocalDramResidency
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


# Callers get a status that says what to do; the finer reasons stay on the
# wire, in logs and in metric labels. Each worker reports its own health: a
# block the source could not serve is UNAVAILABLE here, a source-side fault is
# a transfer that did not happen (ABORTED), and FAILED is kept for faults on
# this worker, including any reason not listed.
_STATUS_BY_REASON = {
    _nvtx.Reason.NOT_RESIDENT: OpEntryStatus.UNAVAILABLE,
    _nvtx.Reason.REMEMBERED_MISS: OpEntryStatus.UNAVAILABLE,
    _nvtx.Reason.HINT_UNAVAILABLE: OpEntryStatus.UNAVAILABLE,
    _nvtx.Reason.HINT_INVALIDATED: OpEntryStatus.UNAVAILABLE,
    _nvtx.Reason.PIN_FAILED: OpEntryStatus.UNAVAILABLE,
    _nvtx.Reason.LAYOUT_MISMATCH: OpEntryStatus.UNAVAILABLE,
    _nvtx.Reason.DEADLINE: OpEntryStatus.TIMEOUT,
    _nvtx.Reason.SOURCE_STALLED: OpEntryStatus.TIMEOUT,
    _nvtx.Reason.CANCELLED: OpEntryStatus.ABORTED,
    _nvtx.Reason.STOPPING: OpEntryStatus.ABORTED,
    _nvtx.Reason.SHUTDOWN: OpEntryStatus.ABORTED,
    _nvtx.Reason.ROUTE_CHANGED: OpEntryStatus.ABORTED,
    _nvtx.Reason.REFUSED: OpEntryStatus.ABORTED,
    _nvtx.Reason.SOURCE_RESTARTED: OpEntryStatus.ABORTED,
    _nvtx.Reason.METADATA_REQUIRED: OpEntryStatus.ABORTED,
    _nvtx.Reason.CREATE_ERROR: OpEntryStatus.ABORTED,
    _nvtx.Reason.TRANSFER_FAILED: OpEntryStatus.ABORTED,
    _nvtx.Reason.SETUP_FAILED: OpEntryStatus.ABORTED,
    _nvtx.Reason.MALFORMED: OpEntryStatus.ABORTED,
    _nvtx.Reason.REMOTE_FAILURE: OpEntryStatus.ABORTED,
    _nvtx.Reason.UNKNOWN: OpEntryStatus.ABORTED,
}
_RESULTS = {
    reason.wire: OpEntryResult(status) for reason, status in _STATUS_BY_REASON.items()
}
_UNAVAILABLE_RESULT = OpEntryResult(OpEntryStatus.UNAVAILABLE)
_MISS_REASONS = frozenset(
    (_nvtx.Reason.NOT_RESIDENT.wire, _nvtx.Reason.REMEMBERED_MISS.wire)
)
# Duration result labels: "failed" is kept for this worker's own faults.
_LABEL_BY_STATUS = {
    OpEntryStatus.UNAVAILABLE: "unavailable",
    OpEntryStatus.TIMEOUT: "timeout",
    OpEntryStatus.ABORTED: "aborted",
}
# The source labels these as its own faults; the target does not (see above).
_SOURCE_FAULTS = frozenset(
    reason.wire
    for reason in (
        _nvtx.Reason.PIN_FAILED,
        _nvtx.Reason.LAYOUT_MISMATCH,
        _nvtx.Reason.CREATE_ERROR,
        _nvtx.Reason.TRANSFER_FAILED,
        _nvtx.Reason.SETUP_FAILED,
        _nvtx.Reason.MALFORMED,
    )
)


def _failed_result(reason: str | None) -> OpEntryResult:
    return _RESULTS.get(reason or "", _FAILED_RESULT)


def _outcome(reason: str | None) -> str:
    if reason in _MISS_REASONS:
        return "miss"
    return _LABEL_BY_STATUS.get(_failed_result(reason).status, "failed")


def _source_outcome(reason: str | None) -> str:
    return "failed" if reason in _SOURCE_FAULTS else _outcome(reason)


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
    # Why the whole operation failed, or why each unserved key was not written.
    failure_reason: str | None = None
    unserved: dict[BlockKey, str] = field(default_factory=dict)
    trace: _nvtx._LifecycleTrace | None = field(default=None, repr=False, compare=False)

    def progress(
        self, progress: _KVCRProgress, event: object | None
    ) -> tuple[bool, bool]:
        completed, did_work = self._progress(progress, event)
        if completed and self.trace is not None:
            self.trace.end("target.remote")
        return completed, did_work

    def _progress(
        self, progress: _KVCRProgress, event: object | None
    ) -> tuple[bool, bool]:
        backend = self._backend
        trace = self.trace
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
                if trace is not None:
                    trace.reject(
                        "target.start_write.rejected",
                        _nvtx.Reason.DEADLINE,
                        status=_nvtx.Status.TIMEOUT,
                    )
                self.success = False
                self.failure_reason = _nvtx.Reason.DEADLINE.wire
                self.state = _TargetPullState.FINISHED
                backend._record_progress_duration(scope, self.started_at, "timeout")
                self._record_unserved(scope)
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
                if trace is not None:
                    trace.reject(
                        "target.start_write.rejected", _nvtx.Reason.CONTROL_ERROR
                    )
                self.success = False
                self.failure_reason = _nvtx.Reason.CONTROL_ERROR.wire
                self.state = _TargetPullState.FINISHED
                backend._record_progress_duration(scope, self.started_at, "failed")
                self._record_unserved(scope)
                return True, True
            self.state = _TargetPullState.WAITING_WRITE_DONE
            if trace is not None:
                trace.mark("target.start_write.enqueued", once=True)
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
                if trace is not None:
                    trace.mark(
                        "target.quiesced", once=True, status=_nvtx.Status.SUCCESS
                    )
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
            if trace is not None:
                reason = _nvtx.Reason.NONE
                if not success:
                    reason = trace.failure_reason
                    if reason is _nvtx.Reason.UNKNOWN:
                        reason = _nvtx.Reason.REMOTE_FAILURE
                        if (
                            self.local_fill
                            and not cancelled
                            and self.state is _TargetPullState.WAITING_WRITE_DONE
                            and event.get("success") is True
                            and now >= self.deadline
                        ):
                            reason = _nvtx.Reason.DEADLINE
            invalid = False
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
                invalid = True
                completed_keys = set()
                if trace is not None:
                    reason = _nvtx.Reason.INVALID_NOTIFICATION
            all_completed = success and completed_keys == self.keys
            self.success = success
            self.completed_keys = completed_keys
            if success:
                self.unserved = _notification_unserved(
                    event, self.ordered_keys, self.keys - completed_keys
                )
            elif self.failure_reason is None:
                self.failure_reason = (
                    _nvtx.Reason.INVALID_NOTIFICATION.wire
                    if invalid
                    else _nvtx.Reason.DEADLINE.wire
                    if event.get("success") is True and not cancelled
                    else _known_reason(event.get("reason"))
                    or _nvtx.Reason.REMOTE_FAILURE.wire
                )
            if trace is not None:
                if not success and reason is _nvtx.Reason.REMOTE_FAILURE:
                    reason = _nvtx_reason(self.failure_reason)
                    if reason is _nvtx.Reason.UNKNOWN:
                        reason = _nvtx.Reason.REMOTE_FAILURE
                elif success and not all_completed:
                    reason = _dominant_reason(self.unserved.values())
                trace.target_result(self, reason)
            self.state = _TargetPullState.FINISHED
            if completed_keys:
                backend._record_progress_counter(
                    TRANSFER_BLOCKS_METRIC,
                    len(completed_keys),
                    (scope,),
                )
            # Zero blocks from a source that answered: a miss, or unavailable
            # when it held them but could not serve them.
            result = (
                "success"
                if all_completed
                else "partial"
                if completed_keys
                else "miss"
                if success and _dominant_wire(self.unserved.values()) in _MISS_REASONS
                else "unavailable"
                if success
                else _outcome(self.failure_reason)
            )
            backend._record_progress_duration(scope, self.started_at, result)
            unserved_counts = self._record_unserved(scope)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "KVCR_EVENT target_transfer_completed scope=%s request_id=%s "
                    "op=%d source=%s requested=%d blocks=%d bytes=%d result=%s "
                    "unserved=%s",
                    scope,
                    self.request_id,
                    self.op_id[1],
                    self.remote_ctrl_ep,
                    len(self.keys),
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
                    _format_counts(unserved_counts),
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
                if trace is not None:
                    trace.mark(
                        "target.quarantined",
                        once=True,
                        status=_nvtx.Status.UNRESOLVED,
                        reason=_nvtx.Reason.DEADLINE,
                        retain_cause=True,
                    )
                backend._dangling_ops.poll_target(progress, self)
                backend._record_progress_duration(scope, self.started_at, "timeout")
                return False, True
            self.failure_reason = (
                (_known_reason(event.get("reason")) or _nvtx.Reason.REMOTE_FAILURE.wire)
                if cancelled
                else _nvtx.Reason.DEADLINE.wire
            )
            self.state = _TargetPullState.WAITING_TERMINAL
            if trace is not None:
                trace.cancel_requested("target.cancel_requested", cancelled)
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
        safe = self.state in (_TargetPullState.START_WRITE, _TargetPullState.FINISHED)
        if not safe and self.trace is not None:
            self.trace.shutdown_unresolved("target.shutdown_unresolved")
        elif safe and self.trace is not None:
            self.trace.end("target.remote")
        return safe

    def _record_unserved(self, scope: str) -> Counter[str]:
        """Count every requested key that was not written, by reason."""
        counts = Counter(self.unserved.values())
        if not self.success:
            failure = self.failure_reason or _nvtx.Reason.UNKNOWN.wire
            counts[failure] += len(self.keys) - len(self.unserved)
        for reason, count in counts.items():
            self._backend._record_progress_counter(
                UNSERVED_BLOCKS_METRIC, count, (scope, reason)
            )
        return counts

    def key_results(self) -> dict[BlockKey, OpEntryResult]:
        """Per-key results: delivered, unavailable, or the operation's failure."""
        completed = self.completed_keys if self.success else ()
        failure = _failed_result(self.failure_reason)
        return {
            key: _SUCCESS_RESULT
            if key in completed
            else _UNAVAILABLE_RESULT
            if key in self.unserved
            else failure
            for key in self.keys
        }


@dataclass(slots=True)
class _NvtxState:
    requested_blocks: int = -1
    target_incarnation: str | None = None
    trace: _nvtx._LifecycleTrace | None = field(default=None, repr=False, compare=False)
    trace_initialized: bool = False


@dataclass
class _SourceWriteOp(_RemoteOp):
    """Progress-owned NIXL write from prepared source descriptors."""

    state: _SourceWriteState
    remote_agent: str
    op_handle: int
    source_keys: tuple[BlockKey, ...]
    dst_descriptors: tuple[tuple[_TransferRef, ...], ...]
    _backend: "_RemoteFWDram" = field(repr=False, compare=False)
    src_descriptors: tuple[tuple[_TransferRef, ...], ...] = ()
    transfer_id: int | None = None
    success: bool = False
    completed_indices: tuple[int, ...] = ()
    route: tuple[str, int] = ("", 0)
    nvtx_state: _NvtxState | None = None
    # None uses caller cleanup; a tuple (even empty) is owned by progress.
    local_sources: tuple["_LocalDramResidency", ...] | None = None
    # Reported to the target: why the operation failed, or why requested
    # indices were left out of completed_indices.
    failure_reason: _nvtx.Reason = _nvtx.Reason.UNKNOWN
    unserved: dict[str, list[int]] = field(default_factory=dict)

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
                if status.cancel_requested:
                    failure_reason = _nvtx.Reason.CANCELLED
                elif not source_responsive:
                    failure_reason = _nvtx.Reason.SOURCE_STALLED
                elif self.state is _SourceWriteState.NOTIFY_FAILURE:
                    failure_reason = self.failure_reason
                else:
                    failure_reason = _nvtx.Reason.DEADLINE
                if failure_reason is _nvtx.Reason.UNKNOWN and trace is not None:
                    failure_reason = trace.failure_reason
                backend._send_write_done(
                    progress,
                    self.remote_agent,
                    self.op_handle,
                    False,
                    reason=_wire_reason(failure_reason),
                )
                self.success = False
                self.state = _SourceWriteState.FINISHED
                backend._record_progress_duration(
                    "source_write",
                    self.started_at,
                    _source_outcome(_wire_reason(failure_reason)),
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
                self.failure_reason = _nvtx.Reason.ROUTE_CHANGED
                if trace is not None:
                    trace.reject("source.write.refused", _nvtx.Reason.ROUTE_CHANGED)
                return False, True
            if not self.src_descriptors:
                backend._send_write_done(
                    progress,
                    self.remote_agent,
                    self.op_handle,
                    True,
                    unserved=self.unserved,
                )
                self.success = True
                self.state = _SourceWriteState.FINISHED
                # Nothing to write: a miss, or this source's own fault.
                result = _source_outcome(
                    _dominant_wire(self._unserved_reasons())
                    or _nvtx.Reason.NOT_RESIDENT.wire
                )
                backend._record_progress_duration(
                    "source_write", self.started_at, result
                )
                self._log_completed(result)
                backend._dangling_ops.finish_source(self)
                if trace is not None:
                    trace.complete_source(
                        True, 0, reason=_dominant_reason(self._unserved_reasons())
                    )
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
                        unserved=self.unserved,
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
                    progress,
                    self.remote_agent,
                    self.op_handle,
                    False,
                    reason=_wire_reason(_nvtx.Reason.CREATE_ERROR),
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
                trace.cancel_requested(
                    "source.write.cancel_requested", status.cancel_requested
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
        failure_reason: str | None = None
        if self.success:
            backend._record_transfer_telemetry(telemetry)
            backend._record_progress_counter(
                TRANSFER_BLOCKS_METRIC,
                len(self.keys),
                ("source_write",),
            )
        else:
            failure_reason = self.abort_reason(status.cancel_requested)
            backend._send_write_done(
                progress,
                self.remote_agent,
                self.op_handle,
                False,
                reason=failure_reason,
            )
        result = (
            ("partial" if self.unserved else "success")
            if self.success
            else _source_outcome(failure_reason)
        )
        backend._record_progress_duration("source_write", self.started_at, result)
        self._log_completed(result)
        self.state = _SourceWriteState.FINISHED
        backend._dangling_ops.finish_source(self)
        if trace is not None:
            trace.complete_source(
                self.success,
                len(self.source_keys),
                reason=_dominant_reason(self._unserved_reasons())
                if self.success and self.unserved
                else None,
            )
        return True, True

    def _unserved_reasons(self) -> Iterator[str]:
        for reason, indices in self.unserved.items():
            yield from (reason,) * len(indices)

    def _log_completed(self, result: str) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "KVCR_EVENT source_transfer_completed op=%d source_op=%d target=%s "
            "blocks=%d bytes=%d result=%s unserved=%s",
            self.op_handle,
            self.op_id[1],
            self.route[0],
            len(self.source_keys) if self.success else 0,
            self._backend._kvcr._descriptor_bytes(
                chain.from_iterable(self.src_descriptors)
            )
            if self.success
            else 0,
            result,
            _format_counts(
                {reason: len(indices) for reason, indices in self.unserved.items()}
            ),
        )

    def abort_reason(self, cancel_requested: bool) -> str:
        # A failed transfer also passes through CANCEL_PENDING.
        if cancel_requested:
            return _nvtx.Reason.CANCELLED.wire
        if self._backend._kvcr._clock() >= self.deadline:
            return _nvtx.Reason.DEADLINE.wire
        return _nvtx.Reason.TRANSFER_FAILED.wire

    def close(self, progress: _KVCRProgress) -> bool:
        trace = self.nvtx_state.trace if self.nvtx_state is not None else None
        if self.transfer_id is not None:
            if (
                progress.poll_transfer(self.transfer_id, require_completion=True)
                is None
            ):
                if trace is not None:
                    trace.shutdown_unresolved("source.write.shutdown_unresolved")
                return False
            self.transfer_id = None
        self._backend._send_write_done(
            progress,
            self.remote_agent,
            self.op_handle,
            False,
            reason=_wire_reason(_nvtx.Reason.SHUTDOWN),
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
    reason: str | None = None,
    unserved: Mapping[str, list[int]] | None = None,
) -> bytes:
    # reason and unserved are diagnostics; older targets ignore unknown fields.
    payload: dict[str, Any] = {
        "type": "write_done",
        "op_handle": op_handle,
        "success": success,
    }
    if success:
        payload["completed_indices"] = completed_indices
        if unserved:
            payload["unserved"] = unserved
    elif reason is not None:
        payload["reason"] = reason
    if not terminal:
        payload["terminal"] = False
    return _NOTIF_PREFIX + msgspec.msgpack.encode(payload)


def _format_counts(counts: Mapping[str, int]) -> str:
    return ",".join(f"{key}:{count}" for key, count in sorted(counts.items())) or "-"


def _nvtx_reason(reason: str | None) -> _nvtx.Reason:
    return _nvtx.Reason.__members__.get((reason or "").upper(), _nvtx.Reason.UNKNOWN)


def _dominant_wire(reasons: Iterable[str]) -> str | None:
    most_common = Counter(reasons).most_common(1)
    return most_common[0][0] if most_common else None


def _dominant_reason(reasons: Iterable[str]) -> _nvtx.Reason:
    return _nvtx_reason(_dominant_wire(reasons))


def _wire_reason(reason: _nvtx.Reason) -> str | None:
    return None if reason is _nvtx.Reason.UNKNOWN else reason.wire


# Peer-supplied reasons reach logs and metric labels, so only known names pass.
_KNOWN_REASONS = frozenset(
    reason.wire
    for reason in _nvtx.Reason
    if reason not in (_nvtx.Reason.UNKNOWN, _nvtx.Reason.NONE)
)


def _known_reason(reason: object) -> str | None:
    return reason if type(reason) is str and reason in _KNOWN_REASONS else None


def _notification_unserved(
    payload: Mapping[str, Any],
    ordered_keys: tuple[BlockKey, ...],
    missing: Collection[BlockKey],
) -> dict[BlockKey, str]:
    """Return why each missing key was not written; malformed input is ignored."""
    sent = set(ordered_keys)
    unserved = {
        key: _nvtx.Reason.UNKNOWN.wire
        if key in sent
        else _nvtx.Reason.REMEMBERED_MISS.wire
        for key in missing
    }
    reported = payload.get("unserved")
    if not isinstance(reported, Mapping):
        return unserved
    for reason, indices in reported.items():
        if _known_reason(reason) is None or not isinstance(indices, list):
            continue
        for index in indices:
            if type(index) is int and 0 <= index < len(ordered_keys):
                key = ordered_keys[index]
                if key in unserved:
                    unserved[key] = reason
    return unserved


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
