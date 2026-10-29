# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Remote DRAM path used by KVCR.

This module owns the G2-specific control protocol and NIXL source writes.
Caller-side source acquisition and framework pins are owned by remote_sources.
Sources and destinations may be local KVCR DRAM or framework memory.
Progress owns accepted operations; KVCR owns block state.

Target: hint/query -> fetch/deliver -> start_write -> write_done.
Source: start_write -> local claim/framework pin -> write -> write_done.
"""

import logging
import math
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import TYPE_CHECKING, Any, cast

import msgspec

from .. import _nvtx
from ..config import KeyAdapter, RemoteFWDramOptions
from ..runtime.core import (
    DURATION_METRIC,
    TRANSFER_BYTES_METRIC,
    _validate_memory_regions,
    logger,
)
from ..runtime.progress import (
    _FAILED_RESULT,
    _SUCCESS_RESULT,
    _KVCRProgress,
    _OpId,
    _RegionMaps,
    _TransferRef,
)
from ..types import (
    BlockKey,
    CacheTier,
    OpHandle,
)
from .dangling_ops import _DanglingOps, _SourceWriteStatus
from .local_dram_pools import _LocalDramState
from .remote_ops import (
    _CONTROL_DECODER,
    _ControlMessage,
    _decode_notif,
    _NvtxState,
    _PeerMessage,
    _SourceWriteOp,
    _SourceWriteState,
    _StartWrite,
    _TargetMetadata,
    _TargetMetadataAck,
    _TargetPullOp,
    _TargetPullState,
    _write_done_notif,
    _WriteProbe,
    _WriteProbeAck,
    _WriteRefused,
)
from .remote_sources import _RemoteSources, _SourcePinOp

if TYPE_CHECKING:
    from ..runtime.core import _KVCRCore


@dataclass(frozen=True)
class _RequestHint:
    source: str | None
    block_hashes: frozenset[int]
    submitted_at: float | None
    failed: bool = False
    missing_keys: frozenset[BlockKey] = frozenset()
    trace: _nvtx._LifecycleTrace | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class _TargetMetadataRequest:
    """Request eager peer metadata setup from progress."""

    endpoint: str


@dataclass
class _ProgressUpdate:
    """Bounded progress-only telemetry and gauges returned to main."""

    metrics: list[tuple[str, str, int | float, tuple[str, ...]]]
    connected_remote_count: int


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
        self._nvtx = _nvtx.create_tracer()

        # Main-thread state: request hints, source ownership, and progress state.
        self._closed = False
        self._request_hints: dict[str, _RequestHint] = {}
        self._connected_remote_count = 0
        self._sources = _RemoteSources(self, kvcr)

        # Progress-thread state: G2 control and outbound events.
        self._pending_local_sources: list[_SourcePinOp] = []
        self._progress_outbound: list[object] = []
        self._progress_metrics: list[tuple[str, str, int | float, tuple[str, ...]]] = []
        self._telemetry_enabled = kvcr.config.enable_telemetry
        self._remote_agents_by_target: dict[str, tuple[bytes, str, _RegionMaps]] = {}
        # Bumped whenever a name's route is replaced: NIXL hands the same
        # handle back for a reused name, so queued operations from the dead
        # generation must be fenced by number, not by handle.
        self._route_generation: dict[str, int] = {}
        self._published_remote_count = 0
        self._metadata_acked_sources: set[str] = set()
        self._metadata_retry_after: dict[str, float] = {}
        self._refused_writes: dict[_OpId, dict[str, Any]] = {}
        self._dangling_ops = _DanglingOps(self)
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
                if previous.trace is not None:
                    previous.trace.mark(
                        "hint.rejected",
                        status=_nvtx.Status.REJECTED,
                        reason=_nvtx.Reason.HINT_CONFLICT,
                    )
                return
            trace = None
            if self._nvtx is not None:
                trace = self._nvtx.lifecycle(
                    request_id=request_id, requested_blocks=len(block_hashes)
                )
                if previous is not None and previous.trace is not None:
                    previous.trace.mark("hint.replaced", once=True)
                if trace is not None:
                    trace.mark("hint.submitted")
            self._request_hints[request_id] = _RequestHint(
                source=src,
                block_hashes=block_hashes,
                submitted_at=(
                    previous.submitted_at
                    if previous is not None and previous.submitted_at is not None
                    else kvcr._timer()
                ),
                trace=trace,
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
            or key in request_hint.missing_keys
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
        blocks: Mapping[BlockKey, list[_TransferRef]],
        request_id: str | None,
        deadline: float,
        op_handle: OpHandle,
        *,
        local_fill: bool,
    ) -> bool:
        kvcr = self._kvcr
        started_at = kvcr._timer()
        scope = "remote_fetch" if local_fill else "remote_deliver"
        current_hint = (
            self._request_hints.get(request_id) if request_id is not None else None
        )
        if current_hint is not None and (
            current_hint.failed or current_hint.missing_keys.issuperset(blocks)
        ):
            kvcr._record_duration(scope, started_at, "failed")
            return False

        if current_hint is None or current_hint.source is None:
            kvcr._record_duration(scope, started_at, "failed")
            return False
        keys = tuple(key for key in blocks if key not in current_hint.missing_keys)
        kvcr._record_duration("hint_wait", current_hint.submitted_at, "complete")
        if request_id is not None:
            self._request_hints[request_id] = replace(current_hint, submitted_at=None)

        op = _TargetPullOp(
            state=_TargetPullState.START_WRITE,
            local_fill=local_fill,
            keys=frozenset(blocks),
            started_at=started_at,
            deadline=deadline,
            op_id=("target", op_handle),
            remote_ctrl_ep=current_hint.source,
            _backend=self,
            ordered_keys=keys,
            dst_descriptors=tuple(tuple(blocks[key]) for key in keys),
            request_id=request_id,
            key_layout=(
                kvcr._block_record_map[keys[0]].local_dram.key_layout
                if local_fill
                else ()
            ),
        )
        if self._nvtx is not None:
            op.trace = self._nvtx.operation(
                kvcr,
                op_handle,
                blocks,
                request_id,
                local_fill=local_fill,
                hint_trace_id=current_hint.trace.trace_id
                if current_hint.trace is not None
                else 0,
                range_name="target.remote",
            )
            if current_hint.trace is not None:
                current_hint.trace.mark("hint.used", op_handle=op_handle)
            if op.trace is not None:
                op.trace.mark("target.queued")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "KVCR_EVENT target_transfer_started scope=%s request_id=%s op=%d "
                "source=%s blocks=%d bytes=%d",
                scope,
                request_id,
                op_handle,
                current_hint.source,
                len(keys),
                kvcr._descriptor_bytes(chain.from_iterable(op.dst_descriptors)),
            )
        try:
            kvcr._add_block_dependencies(op, new_operation=True)
            kvcr._progress.submit(op)
        except Exception:
            if op.trace is not None:
                op.trace.reject("target.queued.rejected", _nvtx.Reason.SUBMIT_REJECTED)
                op.trace.end("target.remote")
            raise
        if not local_fill and request_id is not None:
            kvcr._remote_delivery_keys.setdefault(request_id, set()).update(keys)
        return True

    def deliver(
        self,
        op_handle: OpHandle,
        blocks: Mapping[BlockKey, list[_TransferRef]],
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
        trace = kvcr._nvtx_operations.get(op_handle)
        if trace is not None:
            trace.reject("target.hint_unavailable", _nvtx.Reason.HINT_UNAVAILABLE)
        kvcr._complete(
            op_handle,
            dict.fromkeys(blocks, _FAILED_RESULT),
        )

    def fetch(
        self,
        blocks: Mapping[BlockKey, list[_TransferRef]],
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
        self._sources.retry_releases()
        for item in items:
            if isinstance(item, _SourcePinOp):
                self._sources.start_source_pin(item)
            elif isinstance(item, _SourceWriteOp):
                self._sources.finish_source_write(item)
            elif isinstance(item, _TargetPullOp):
                if item.state is _TargetPullState.WAITING_TERMINAL:
                    if not item.local_fill:
                        raise RuntimeError(
                            "non-local target pull is waiting for terminal state"
                        )
                    self._kvcr._discard_local_dram_fill(item.keys)
                elif item.state is _TargetPullState.QUARANTINED:
                    self._finish_target_pull(item)
                elif item.state is _TargetPullState.FINISHED and item.uncertain:
                    self._kvcr._remove_block_dependencies(item)
                    if item.local_fill:
                        self._kvcr._complete_local_dram_fill(item.keys, success=False)
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
            self._sources.poll()

    def discard_hint(self, request_id: str) -> None:
        hint = self._request_hints.pop(request_id, None)
        if hint is not None and hint.trace is not None:
            hint.trace.mark("hint.discarded", once=True)

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
        for hint in self._request_hints.values():
            if hint.trace is not None:
                hint.trace.mark(
                    "hint.discarded", once=True, reason=_nvtx.Reason.SHUTDOWN
                )
        self._request_hints.clear()
        self._sources.close()

    # -------------------------------------------------------------------------
    # Target side: request and receive remote KV.
    # -------------------------------------------------------------------------

    def _finish_target_pull(self, op: _TargetPullOp) -> None:
        trace = op.trace
        pushed = trace.push("target.main.consume") if trace is not None else False
        try:
            kvcr = self._kvcr
            caller_trace = kvcr._nvtx_operations.get(op.op_id[1])
            if caller_trace is not None and op.trace is not None:
                caller_trace.failure_reason = op.trace.failure_reason
            if op.state is not _TargetPullState.QUARANTINED:
                kvcr._remove_block_dependencies(op)
            completed_keys = op.completed_keys if op.success else set()
            if kvcr._key_history_enabled:
                kvcr._log_key_history(
                    "remote_fetch_completed"
                    if op.local_fill
                    else "remote_deliver_completed",
                    (key for key in op.ordered_keys if key in completed_keys),
                    CacheTier.REMOTE_G2,
                    request_id=op.request_id,
                    op_handle=cast(OpHandle, op.op_id[1]),
                )
            if not op.success:
                self._fail_request_hint(op.request_id)
            elif (
                op.request_id is not None
                and (hint := self._request_hints.get(op.request_id)) is not None
            ):
                self._request_hints[op.request_id] = replace(
                    hint,
                    missing_keys=(hint.missing_keys | op.keys) - completed_keys,
                )
            if op.local_fill:
                if op.state is _TargetPullState.QUARANTINED:
                    # DISCARDING already failed callers, but still owns the slots.
                    return
                if completed_keys:
                    kvcr._complete_local_dram_fill(
                        tuple(key for key in op.ordered_keys if key in completed_keys),
                        success=True,
                    )
                if completed_keys != op.keys:
                    kvcr._complete_local_dram_fill(
                        op.keys - completed_keys,
                        success=False,
                    )
                return
            kvcr._complete(
                cast(OpHandle, op.op_id[1]),
                {
                    key: _SUCCESS_RESULT if key in completed_keys else _FAILED_RESULT
                    for key in op.keys
                },
            )
        finally:
            if trace is not None:
                trace.tracer.pop(pushed)

    # -------------------------------------------------------------------------
    # Progress-thread lifecycle and control transport.
    # -------------------------------------------------------------------------

    def initialize_progress(self, _progress: _KVCRProgress) -> None:
        if self._nvtx is not None:
            self._nvtx.name_progress_thread()
        initialize_control = getattr(self._control, "initialize", None)
        if initialize_control is not None:
            initialize_control()

    def poll_progress(
        self, progress: _KVCRProgress, submissions: list[object]
    ) -> tuple[dict[object, object], bool]:
        self._dangling_ops.begin_poll()
        observed_work = bool(submissions)
        for item in submissions:
            if isinstance(item, _TargetMetadataRequest):
                if (
                    item.endpoint in self._metadata_acked_sources
                    or self._kvcr._clock()
                    < self._metadata_retry_after.get(item.endpoint, 0)
                ):
                    continue
                self._send_control(progress, item.endpoint, _TargetMetadata())
            else:
                raise TypeError(f"unsupported KVCR progress item: {type(item)!r}")

        # Warning suppression can be added if persistent backend faults
        # cause excessive polling logs.
        observed_work |= self._process_control_messages(progress)
        if self._pending_local_sources and not progress._stop_requested:
            observed_work |= self._retry_local_sources(progress)
        # Only terminal notifications outrank a refusal for the same operation.
        events = self._poll_notifications(progress)
        for op_id, refusal in self._refused_writes.items():
            native = events.get(op_id, {})
            if not native.get("terminal") and not (
                refusal.get("retry_metadata") and native.get("cancelled")
            ):
                events[op_id] = refusal
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
        # Accepted writes may still be waiting for main-thread pin acquisition.
        for (target, handle), status in self._dangling_ops.source_writes.items():
            cached = self._remote_agents_by_target.get(target)
            if not status.submitted and cached is not None:
                self._send_write_done(self._kvcr._progress, cached[1], handle, False)
        self._pending_local_sources.clear()
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
                payload = _CONTROL_DECODER.decode(message)
            except msgspec.ValidationError as error:
                handled |= self._handle_malformed_start_write(progress, message, error)
                continue
            except (TypeError, msgspec.DecodeError):
                continue
            handled_at = self._kvcr._timer()
            if isinstance(payload, _StartWrite):
                self._handle_start_write(progress, payload)
            elif isinstance(payload, _TargetMetadata):
                self._handle_target_metadata(progress, payload)
            elif isinstance(payload, _TargetMetadataAck):
                self._handle_target_metadata_ack(progress, payload)
            elif isinstance(payload, _WriteRefused):
                self._handle_write_refused(progress, payload)
            elif isinstance(payload, _WriteProbe):
                self._dangling_ops.handle_probe(progress, payload)
            elif isinstance(payload, _WriteProbeAck):
                self._dangling_ops.handle_probe_ack(progress, payload)
            handled = True
            if handled_at is not None:
                self._record_progress_duration(
                    f"control_{payload.__struct_config__.tag}", handled_at, "handled"
                )
        return handled

    def _send_control(
        self,
        progress: _KVCRProgress,
        endpoint: str,
        payload: _ControlMessage,
    ) -> bool:
        kvcr = self._kvcr
        started_at = kvcr._timer()
        if self._control is None or progress.nixl_agent_metadata is None:
            self._record_progress_duration("control_enqueue", started_at, "failed")
            return False
        payload.target_agent = kvcr.nixl_agent_name
        payload.sender_incarnation = self._dangling_ops.incarnation
        sender_endpoint = getattr(self._control, "endpoint", None)
        if isinstance(sender_endpoint, str) and payload.sender_control_endpoint is None:
            payload.sender_control_endpoint = sender_endpoint
        if not isinstance(payload, _TargetMetadataAck):
            payload.source_control_endpoint = endpoint
        includes_metadata = (
            not isinstance(payload, (_TargetMetadataAck, _WriteRefused))
            and endpoint not in self._metadata_acked_sources
            and (
                isinstance(payload, (_TargetMetadata, _StartWrite))
                or kvcr._clock() >= self._metadata_retry_after.get(endpoint, 0)
            )
        )
        if includes_metadata:
            payload.target_agent_metadata = progress.nixl_agent_metadata
            payload.target_regions = progress._memory_regions
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
        self, progress: _KVCRProgress, payload: _TargetMetadata
    ) -> None:
        try:
            self._remote_agent(progress, payload)
            self._ack_target_metadata(progress, payload)
        except _RemoteAgentCleanupError:
            raise
        except Exception:
            return

    def _handle_write_refused(
        self, progress: _KVCRProgress, payload: _WriteRefused
    ) -> None:
        endpoint = payload.sender_control_endpoint
        op_handle = payload.op_handle
        if endpoint is None:
            return
        op = progress._in_flight_ops.get(("target", op_handle))
        if not isinstance(op, _TargetPullOp) or op.remote_ctrl_ep != endpoint:
            return
        # Whatever answered is not the process our metadata was loaded into, so
        # drop the snapshot and let the next request republish it.
        self._invalidate_control_peer(endpoint)
        # Refresh the generation for future writes, not already in-flight ones.
        # A delayed refusal must not evict a generation learned after this op.
        refused_incarnation = op.source_incarnation
        if self._dangling_ops.sources.get(endpoint) == refused_incarnation:
            self._dangling_ops.sources.pop(endpoint, None)
        incarnation = payload.sender_incarnation
        if (
            payload.reason in ("metadata_required", "source_restarted")
            and incarnation
            and not op.retried
            and op.state is _TargetPullState.WAITING_WRITE_DONE
            and not progress._stop_requested
            and self._kvcr._clock() < op.deadline
        ):
            self._refused_writes[("target", op_handle)] = {
                "terminal": False,
                "retry_metadata": incarnation,
            }
            return
        # A refusal is only sent before a write is submitted, so it is as terminal as a
        # failed write_done and safe to act on even from WAITING_TERMINAL. Deliberately
        # unauthenticated: a "resend it" signal on a channel already trusted to let a
        # start_write make a source write.
        self._refused_writes[("target", op_handle)] = {"success": False}

    def _handle_target_metadata_ack(
        self, progress: _KVCRProgress, payload: _TargetMetadataAck
    ) -> None:
        source_endpoint = payload.sender_control_endpoint
        if not source_endpoint:
            return
        incarnation = payload.sender_incarnation
        if incarnation:
            self._dangling_ops.sources[source_endpoint] = incarnation
            handle = payload.op_handle
            if handle is not None:
                op = progress._in_flight_ops.get(("target", handle))
                if (
                    isinstance(op, _TargetPullOp)
                    and op.remote_ctrl_ep == source_endpoint
                    and op.source_incarnation is None
                ):
                    op.source_incarnation = incarnation
        self._metadata_acked_sources.add(source_endpoint)
        self._metadata_retry_after.pop(source_endpoint, None)

    def _ack_target_metadata(
        self,
        progress: _KVCRProgress,
        payload: _TargetMetadata | _StartWrite,
    ) -> None:
        if payload.target_agent_metadata is None:
            return
        target_control_endpoint = payload.sender_control_endpoint
        if not target_control_endpoint:
            return
        source_control_endpoint = payload.source_control_endpoint
        if not source_control_endpoint:
            return
        response = _TargetMetadataAck(
            sender_control_endpoint=source_control_endpoint,
            op_handle=payload.op_handle if isinstance(payload, _StartWrite) else None,
        )
        if not self._send_control(progress, target_control_endpoint, response):
            # Kept, route and cache both: an operation this start_write queued
            # still transfers over this route, and the peer re-sends its
            # metadata until the ack lands -- matching bytes reuse the entry,
            # changed bytes replace it through the refresh path.
            return

    # -------------------------------------------------------------------------
    # Source side: progress claims ready local DRAM or queues source acquisition
    # to main; progress writes to the target.
    # -------------------------------------------------------------------------

    def _handle_start_write(
        self, progress: _KVCRProgress, payload: _StartWrite
    ) -> None:
        started_at = self._kvcr._timer()
        received_at = self._kvcr._clock()
        op_handle = payload.op_handle
        remaining_timeout_ms = payload.remaining_timeout_ms
        keys = payload.keys
        dst_descriptors = payload.dst_descriptors
        if (
            not math.isfinite(remaining_timeout_ms)
            or remaining_timeout_ms <= 0
            or not keys
            or len(key_set := frozenset(keys)) != len(keys)
            or len(keys) != len(dst_descriptors)
        ):
            logger.warning("KVCR malformed start_write op=%d", op_handle)
            self._notify_start_write_failure(progress, payload, op_handle)
            return

        remaining_timeout_ms = min(
            remaining_timeout_ms,
            self._kvcr.config.operation_timeout_ms,
        )
        deadline = received_at + remaining_timeout_ms / 1000
        try:
            fallback_target = (
                dst_descriptors[0][0].end_point_name if dst_descriptors[0] else None
            )
            target_agent, remote_agent = self._remote_agent(
                progress, payload, fallback_target=fallback_target
            )
            for descriptors in dst_descriptors:
                self._kvcr._validate_descriptors(
                    descriptors,
                    regions=self._remote_agents_by_target[target_agent][2],
                    agent_name=target_agent,
                )
            self._ack_target_metadata(progress, payload)
        except _RemoteAgentCleanupError:
            raise
        except Exception:
            logger.warning(
                "KVCR start_write setup failed for op=%d", op_handle, exc_info=True
            )
            self._notify_start_write_failure(progress, payload, op_handle)
            return

        write_id = (target_agent, OpHandle(op_handle))
        if status := self._dangling_ops.source_writes.get(write_id):
            if status.cancel_requested and not status.submitted:
                self._send_write_done(progress, remote_agent, op_handle, False)
            return
        expected = payload.source_incarnation
        if expected is not None and expected != self._dangling_ops.incarnation:
            # Another process was quoted: refuse pre-submit so the target retries
            # once quoting us, instead of failing the request natively.
            self._refuse_start_write(progress, payload, op_handle, "source_restarted")
            return
        if progress._stop_requested or not self._dangling_ops.check_source_progress():
            if logger.isEnabledFor(logging.DEBUG):
                reason = "stopping" if progress._stop_requested else "source_stalled"
                logger.debug(
                    "KVCR_EVENT source_transfer_refused op=%d target=%s "
                    "blocks=%d reason=%s expected_incarnation=%s incarnation=%s",
                    op_handle,
                    target_agent,
                    len(keys),
                    reason,
                    expected,
                    self._dangling_ops.incarnation,
                )
            self._send_write_done(progress, remote_agent, op_handle, False)
            return
        self._dangling_ops.source_writes[write_id] = _SourceWriteStatus()
        op_id = ("source", self._next_source_op_id)
        self._next_source_op_id += 1

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "KVCR_EVENT source_transfer_requested op=%d source_op=%d target=%s "
                "blocks=%d bytes=%d",
                op_handle,
                op_id[1],
                target_agent,
                len(keys),
                self._kvcr._descriptor_bytes(chain.from_iterable(dst_descriptors)),
            )

        source_pin = _SourcePinOp(
            op_id=op_id,
            keys=key_set,
            started_at=started_at,
            deadline=deadline,
            remote_agent=remote_agent,
            op_handle=op_handle,
            ordered_keys=keys,
            dst_descriptors=dst_descriptors,
            key_layout=payload.key_layout,
            route=(target_agent, self._route_generation.get(target_agent, 0)),
            nvtx_state=_NvtxState(
                requested_blocks=len(keys),
                target_incarnation=payload.sender_incarnation,
            )
            if self._nvtx is not None
            else None,
        )
        prepared = self._try_local_source_write(progress, source_pin)
        if prepared is None:
            self._pending_local_sources.append(source_pin)
        elif not prepared:
            self._progress_outbound.append(source_pin)

    def _retry_local_sources(self, progress: _KVCRProgress) -> bool:
        pending, self._pending_local_sources = self._pending_local_sources, []
        observed_work = False
        for op in pending:
            status = self._dangling_ops.source_writes[(op.route[0], op.op_handle)]
            if (
                self._kvcr._clock() >= op.deadline
                or status.cancel_requested
                or not self._dangling_ops.check_source_progress()
                or self._route_generation.get(op.route[0], 0) != op.route[1]
            ):
                # No claims were acquired, so there is no caller-side cleanup.
                failure = _SourceWriteOp(
                    state=_SourceWriteState.NOTIFY_FAILURE,
                    op_id=op.op_id,
                    keys=op.keys,
                    started_at=op.started_at,
                    deadline=op.deadline,
                    remote_agent=op.remote_agent,
                    op_handle=op.op_handle,
                    source_keys=(),
                    dst_descriptors=(),
                    route=op.route,
                    nvtx_state=op.nvtx_state,
                    _backend=self,
                )
                if op.nvtx_state is not None:
                    op.nvtx_state.failure_reason = (
                        _nvtx.Reason.DEADLINE
                        if self._kvcr._clock() >= op.deadline
                        else _nvtx.Reason.ROUTE_CHANGED
                    )
                failure.progress(progress, None)
                observed_work = True
                continue
            prepared = self._try_local_source_write(progress, op)
            if prepared is None:
                self._pending_local_sources.append(op)
            else:
                observed_work = True
                if not prepared:
                    self._progress_outbound.append(op)
        return observed_work

    def _try_local_source_write(
        self, progress: _KVCRProgress, source_pin: _SourcePinOp
    ) -> bool | None:
        """Return None to retry contention, False when caller work is needed."""
        kvcr = self._kvcr
        if kvcr._local_dram is None:
            return False
        if not kvcr._state_lock.acquire(blocking=False):
            return None
        selected = []
        try:
            for key, destination in zip(
                source_pin.ordered_keys, source_pin.dst_descriptors
            ):
                record = kvcr._block_record_map.get(key)
                residency = record.local_dram if record is not None else None
                if residency is None or residency.state is not _LocalDramState.READY:
                    return False
                if (
                    source_pin.key_layout
                    and source_pin.key_layout != residency.key_layout
                ):
                    return False
                indices = residency.ready_indices([ref.label for ref in destination])
                if indices is None:
                    return False
                selected.append(indices)
            sources = kvcr._claim_local_dram_sources(
                source_pin.op_id, source_pin.ordered_keys, notify_capacity=False
            )
            source_refs = tuple(
                kvcr._local_dram._pools.source_refs(sources[key], indices)
                for key, indices in zip(source_pin.ordered_keys, selected)
            )
            kvcr._add_block_dependencies(source_pin, new_operation=True)
        finally:
            kvcr._state_lock.release()
        # Claims keep the sources stable while the write is assembled unlocked.
        source_write = _SourceWriteOp(
            state=_SourceWriteState.READY_TO_WRITE,
            op_id=source_pin.op_id,
            keys=source_pin.keys,
            started_at=source_pin.started_at,
            deadline=source_pin.deadline,
            remote_agent=source_pin.remote_agent,
            op_handle=source_pin.op_handle,
            source_keys=source_pin.ordered_keys,
            src_descriptors=source_refs,
            dst_descriptors=source_pin.dst_descriptors,
            completed_indices=tuple(range(len(source_pin.ordered_keys))),
            route=source_pin.route,
            nvtx_state=source_pin.nvtx_state,
            _backend=self,
        )
        # Already on progress: retain cleanup ownership before starting the write.
        progress._in_flight_ops[source_write.op_id] = source_write
        done, _ = source_write.progress(progress, None)
        if done:
            progress._in_flight_ops.pop(source_write.op_id)
            self._progress_outbound.append(source_write)
        return True

    def _handle_malformed_start_write(
        self,
        progress: _KVCRProgress,
        message: bytes,
        error: msgspec.ValidationError,
    ) -> bool:
        # Recover routing fields even when the request body fails typed decoding.
        try:
            payload = msgspec.msgpack.decode(message)
        except (TypeError, msgspec.DecodeError):
            return False
        if not isinstance(payload, dict) or payload.get("type") != "start_write":
            return False
        op_handle = payload.get("op_handle")
        if type(op_handle) is not int:
            logger.warning("KVCR malformed start_write: invalid op_handle: %s", error)
            return True
        logger.warning("KVCR malformed start_write op=%d: %s", op_handle, error)
        try:
            peer = msgspec.convert(payload, type=_PeerMessage)
        except msgspec.ValidationError:
            # Keep supplied metadata so malformed requests cannot invite a retry.
            metadata = payload.get("target_agent_metadata")
            peer = _PeerMessage(
                target_agent_metadata=metadata if isinstance(metadata, bytes) else None
            )
            # Reuse cached routes only when metadata is absent.
            if peer.target_agent_metadata is None:
                target_agent = payload.get("target_agent")
                if isinstance(target_agent, str):
                    peer.target_agent = target_agent
            for name in ("sender_control_endpoint", "source_control_endpoint"):
                value = payload.get(name)
                if isinstance(value, str):
                    setattr(peer, name, value)
        self._notify_start_write_failure(progress, peer, op_handle)
        return True

    def _notify_start_write_failure(
        self,
        progress: _KVCRProgress,
        payload: _PeerMessage,
        op_handle: OpHandle,
    ) -> None:
        # TODO: reuse this request's peer resolution; if it failed, refuse over
        # control instead of retrying setup.
        try:
            _, remote_agent = self._remote_agent(progress, payload)
        except _RemoteAgentCleanupError:
            raise
        except Exception:
            # A promoted Guard adopts the dead primary's control endpoint but
            # not its peer table, so there is no NIXL route home to report on.
            # Refusing over the control channel is the only way this operation
            # ever reaches a terminal.
            logger.warning("KVCR refusing unresolvable start_write op=%d", op_handle)
            # No retry invitation from a stopping source: resolution fails here
            # even with good cached metadata, and we will not be around to serve.
            retryable = (
                payload.target_agent_metadata is None and not progress._stop_requested
            )
            self._refuse_start_write(
                progress, payload, op_handle, "metadata_required" if retryable else None
            )
            return
        self._send_write_done(progress, remote_agent, op_handle, False)

    def _refuse_start_write(
        self,
        progress: _KVCRProgress,
        payload: _PeerMessage,
        op_handle: OpHandle,
        reason: str | None,
    ) -> None:
        reply_to = payload.sender_control_endpoint
        reflected_self = payload.source_control_endpoint
        if reply_to is None or reflected_self is None:
            return
        self._send_control(
            progress,
            reply_to,
            _WriteRefused(
                sender_control_endpoint=reflected_self,
                op_handle=op_handle,
                reason=reason,
            ),
        )

    # NIXL peer and descriptor setup.

    def _remote_agent(
        self,
        progress: _KVCRProgress,
        payload: _PeerMessage,
        fallback_target: str | None = None,
    ) -> tuple[str, str]:
        if progress._stop_requested:
            raise RuntimeError("KVCR progress is stopping")
        kvcr = self._kvcr
        agent = progress.nixl_agent
        target_agent = payload.target_agent
        if target_agent is msgspec.UNSET:
            target_agent = fallback_target
        if not target_agent:
            raise TypeError("missing target agent")
        target_metadata = payload.target_agent_metadata
        regions = None
        if target_metadata is not None:
            regions = payload.target_regions
            if regions is msgspec.UNSET:
                regions = ({}, {})
        if regions is not None:
            if any(
                name != region.label
                for catalog in regions
                for name, region in catalog.items()
            ):
                raise ValueError(
                    "registration dictionary names must match region label"
                )
            _validate_memory_regions([*regions[0].values(), *regions[1].values()])
        cached = self._remote_agents_by_target.get(target_agent)
        if cached is not None and cached[1] in progress._disconnected_remote_agents:
            # A disconnect may drop NIXL's native metadata; reload only if it did.
            if not agent.check_remote_metadata(cached[1]):
                progress.release_prepared(cached[1])
                self._remote_agents_by_target.pop(target_agent)
                self._route_generation[target_agent] = (
                    self._route_generation.get(target_agent, 0) + 1
                )
            progress._disconnected_remote_agents.discard(cached[1])
            cached = self._remote_agents_by_target.get(target_agent)
        if cached is not None:
            cached_metadata, remote_agent, cached_regions = cached
            if target_metadata is None or (
                target_metadata == cached_metadata and regions == cached_regions
            ):
                reused_at = kvcr._timer()
                self._record_progress_duration("peer_setup", reused_at, "reused")
                return target_agent, remote_agent
            # TODO: distinguish peer restarts from additive metadata updates;
            # load additions without disconnecting the existing peer. Changed
            # metadata alone does not establish a restart.
            # Refresh can stay refused indefinitely: operation timeouts do not
            # release the old peer's native transfers.
            _unload_remote_agent(progress, remote_agent)
            self._remote_agents_by_target.pop(target_agent, None)
            self._route_generation[target_agent] = (
                self._route_generation.get(target_agent, 0) + 1
            )
        started_at = kvcr._timer()
        try:
            if target_metadata is None:
                raise TypeError("missing target agent metadata")
            remote_agent = agent.add_remote_agent(target_metadata)
            if isinstance(remote_agent, bytes):
                remote_agent = remote_agent.decode()
            if not isinstance(remote_agent, str) or not remote_agent:
                raise RuntimeError("add_remote_agent returned no agent name")
            owned = (
                remote_agent == agent.name
                or remote_agent in progress._prepared
                or any(
                    cached[1] == remote_agent
                    for cached in self._remote_agents_by_target.values()
                )
            )
            if owned:
                raise RuntimeError("NIXL remote agent is already initialized")
            try:
                progress.prepare_memory(remote_agent, regions, owner_name=target_agent)
            except BaseException:
                _unload_remote_agent(progress, remote_agent)
                raise
            self._remote_agents_by_target[target_agent] = (
                target_metadata,
                remote_agent,
                regions,
            )
        except Exception:
            self._record_progress_duration("peer_setup", started_at, "failed")
            raise
        self._record_progress_duration("peer_setup", started_at, "connected")
        return target_agent, remote_agent

    # Progress notifications, telemetry, and resource cleanup.

    def _poll_notifications(
        self, progress: _KVCRProgress
    ) -> dict[_OpId, dict[str, Any]]:
        agent = progress.nixl_agent
        get_new_notifs = getattr(agent, "get_new_notifs", None)
        if get_new_notifs is None:
            return {}
        events: dict[_OpId, dict[str, Any]] = {}
        try:
            for notifs in get_new_notifs().values():
                for raw in notifs:
                    payload = _decode_notif(raw)
                    if payload is None or payload.get("type") != "write_done":
                        continue
                    # op_handle must be an integer.
                    op_handle = payload.get("op_handle")
                    if type(op_handle) is not int:
                        continue
                    op_id = ("target", op_handle)
                    previous = events.get(op_id, {})
                    terminal = payload.get("terminal", True) is True
                    cancelled = previous.get("cancelled", False) or not payload.get(
                        "success", False
                    )
                    terminal |= previous.get("terminal", False)
                    payload.update(terminal=terminal, cancelled=cancelled)
                    events[op_id] = payload
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
        remote_agent: str,
        op_handle: OpHandle,
        success: bool,
        *,
        terminal: bool = True,
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
            result = send_notif(
                remote_agent, _write_done_notif(op_handle, success, terminal=terminal)
            )
        except Exception:
            logger.warning(
                "KVCR write_done notification failed for op=%d",
                op_handle,
                exc_info=True,
            )
            return
        if result is False:
            logger.warning("KVCR write_done notification failed for op=%d", op_handle)


class _RemoteAgentCleanupError(RuntimeError):
    """Native peer cleanup failed; propagate to the progress-thread handler."""


def _unload_remote_agent(progress: _KVCRProgress, remote_agent: str) -> None:
    if any(
        state.remote_side_agent == remote_agent
        for state in progress._active_transfers.values()
    ):
        raise RuntimeError(f"NIXL remote agent {remote_agent!r} has active transfers")
    try:
        progress.release_prepared(remote_agent)
        progress.nixl_agent.remove_remote_agent(remote_agent)
    except Exception as error:
        raise _RemoteAgentCleanupError(
            f"NIXL remote agent {remote_agent!r} cleanup failed"
        ) from error
