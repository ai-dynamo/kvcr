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

import logging
import math
import time
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import TYPE_CHECKING, Any, cast

import msgspec

from . import _nvtx
from .config import KeyAdapter, RemoteFWDramOptions
from .core import (
    DURATION_METRIC,
    TRANSFER_BYTES_METRIC,
    _validate_memory_regions,
    logger,
)
from .dangling_ops import _DanglingOps, _SourceWriteStatus
from .local_dram import _layout_indices, _LocalDramState
from .progress import (
    _FAILED_RESULT,
    _SUCCESS_RESULT,
    _KVCRProgress,
    _Op,
    _OpId,
    _RegionMaps,
    _TransferRef,
)
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
from .types import (
    BlockKey,
    CacheTier,
    MemoryRef,
    OpHandle,
    PinHandle,
    PinRequestId,
    PinResult,
)

if TYPE_CHECKING:
    from .core import _KVCRCore


@dataclass(slots=True)
class _FwMemResidency:
    descriptors: list[_TransferRef]
    pin_handle: PinHandle


@dataclass(frozen=True)
class _RequestHint:
    source: str | None
    block_hashes: frozenset[int]
    submitted_at: float | None
    failed: bool = False
    missing_keys: frozenset[BlockKey] = frozenset()


@dataclass
class _SourcePinOp(_Op):
    """Main-thread source acquisition for a pending write."""

    started_at: float | None
    deadline: float
    remote_agent: str
    op_handle: int
    # Preserve request positions even when keys drops rejected dependencies.
    ordered_keys: tuple[BlockKey, ...]
    dst_descriptors: tuple[tuple[_TransferRef, ...], ...]
    route: tuple[str, int] = ("", 0)
    key_layout: tuple[str, ...] = ()
    framework_pins: set[PinHandle] = field(default_factory=set)
    pending_pin_ids: set[PinRequestId] = field(default_factory=set)
    framework_acquire_attempted: bool = False
    nvtx_state: _NvtxState | None = None


@dataclass
class _PendingFrameworkSources:
    """Source descriptors waiting for framework pin results."""

    pending_pins: tuple[PinRequestId, ...]
    framework_pins: set[PinHandle]


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
    trace: _nvtx._PinTrace | None = None


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

        # Main-thread state: request hints, framework pins, and progress state.
        self._closed = False
        self._request_hints: dict[str, _RequestHint] = {}
        self._connected_remote_count = 0
        self._source_pin_ops: dict[_OpId, _SourcePinOp] = {}
        self._pending_pin_ops: dict[PinRequestId, _PendingPinWait] = {}
        self._pending_pin_keys: dict[BlockKey, set[PinRequestId]] = {}
        # Pins retained while their source operations execute.
        self._fw_pins_by_op: dict[_OpId, set[PinHandle]] = {}
        self._releasing_framework_pins: set[PinHandle] = set()

        # Progress-thread state: G2 control and outbound events.
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
        kvcr._add_block_dependencies(op, new_operation=True)
        kvcr._progress.submit(op)
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
        self._release_framework_pins(tuple(self._releasing_framework_pins))
        for item in items:
            if isinstance(item, _SourcePinOp):
                self._start_source_pin(item)
            elif isinstance(item, _SourceWriteOp):
                if item.state in (
                    _SourceWriteState.FINISHED,
                    _SourceWriteState.CANCEL_PENDING,
                ):
                    # Framework ownership can be handed back after uncertainty;
                    # KVCR-owned contents stay claimed until native quiescence.
                    pins = self._fw_pins_by_op.pop(item.op_id, None)
                    if item.state is _SourceWriteState.FINISHED:
                        self._kvcr._remove_block_dependencies(item)
                        if item.success:
                            self._kvcr._record_access(
                                self._kvcr._local_dram_sources_by_op.get(item.op_id, ())
                            )
                        self._kvcr._release_local_dram_sources(item.op_id)
                    if pins is not None:
                        self._release_framework_pins(pins)
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
            self._process_pending_pin_results()
            now = self._kvcr._clock()
            for op_id, op in list(self._source_pin_ops.items()):
                if now >= op.deadline:
                    logger.warning("KVCR operation %r expired", op_id)
                    self._expire_source_pin(op_id, op)
                elif not op.pending_pin_ids:
                    self._resume_source_pin(op_id, op)

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
                return  # DISCARDING already failed callers, but still owns the slots.
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
        if not self._try_local_source_write(progress, source_pin):
            self._progress_outbound.append(source_pin)

    def _try_local_source_write(
        self, progress: _KVCRProgress, source_pin: _SourcePinOp
    ) -> bool:
        kvcr = self._kvcr
        if kvcr._local_dram is None:
            return False
        layouts = [
            [descriptor.label for descriptor in destination]
            for destination in source_pin.dst_descriptors
        ]
        if not kvcr._state_lock.acquire(blocking=False):
            # Use the caller queue on contention; add a progress-side
            # second attempt if contention makes this fallback too frequent.
            return False
        selected = []
        try:
            for key, layout in zip(source_pin.ordered_keys, layouts):
                record = kvcr._block_record_map.get(key)
                residency = record.local_dram if record is not None else None
                if residency is None or residency.state is not _LocalDramState.READY:
                    return False
                if (
                    source_pin.key_layout
                    and source_pin.key_layout != residency.key_layout
                ):
                    return False
                indices = residency.ready_indices(layout)
                if indices is None:
                    return False
                selected.append(indices)
            sources = kvcr._claim_local_dram_sources(
                source_pin.op_id, source_pin.ordered_keys, notify_capacity=False
            )
            source_refs = tuple(
                kvcr._local_dram.source_refs(sources[key], indices)
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

    def _submit_prepared_source_write(
        self,
        op_id: _OpId,
        source_pin: _SourcePinOp,
        *,
        force_failure: bool = False,
        failure_reason: _nvtx.Reason = _nvtx.Reason.UNKNOWN,
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
            if failure_reason is _nvtx.Reason.UNKNOWN:
                failure_reason = _nvtx.Reason.ROUTE_CHANGED
        local_sources = kvcr._claim_local_dram_sources(
            source_pin.op_id, source_pin.keys
        )
        framework_sources = {
            key: record.fw_mem.descriptors
            for key in source_pin.keys
            if key not in local_sources
            and (record := kvcr._block_record_map.get(key)) is not None
            and record.fw_mem is not None
        }
        completed_indices = []
        selected_sources = []
        waiting = False
        for index, key in enumerate(() if force_failure else source_pin.ordered_keys):
            if key not in source_pin.keys:
                continue
            residency = local_sources.get(key)
            source = framework_sources.get(key)
            destination = source_pin.dst_descriptors[index]
            if residency is None and source is None:
                continue
            labels = [descriptor.label for descriptor in destination]
            layout = (
                residency.key_layout
                if residency is not None
                else [descriptor.label for descriptor in source]
            )
            if source_pin.key_layout and source_pin.key_layout != tuple(layout):
                indices = None
            elif residency is not None:
                indices = (
                    residency.ready_indices(labels)
                    if residency.state is _LocalDramState.READY
                    else None
                )
                if (
                    indices is None
                    and residency.state is not _LocalDramState.DISCARDING
                    and residency.can_wait_for_pieces(labels)
                ):
                    waiting = True
                    continue
            else:
                indices = _layout_indices(layout, labels)
            if indices is None:
                logger.warning(
                    "KVCR start_write layout mismatch op=%d key=%r",
                    source_pin.op_handle,
                    key,
                )
                source_pin.keys -= {key}
                record = kvcr._block_record_map[key]
                record.discard_in_flight_op(op_id)
                kvcr._prune_block_record(key)
                continue
            completed_indices.append(index)
            selected_sources.append(
                kvcr._local_dram.source_refs(residency, indices)
                if residency is not None
                else tuple(source[span] for span in indices)
            )
        completed_keys = tuple(
            source_pin.ordered_keys[index] for index in completed_indices
        )

        retained_keys = source_pin.keys if waiting else set(completed_keys)
        kvcr._release_local_dram_sources(op_id, local_sources.keys() - retained_keys)
        relevant_pins = {
            record.fw_mem.pin_handle
            for key in retained_keys
            if key not in local_sources
            and (record := kvcr._block_record_map.get(key)) is not None
            and record.fw_mem is not None
        }
        # Every pin the write reads through, not just the ones this operation acquired:
        # fw_mem is per block, so holding another operation's pin is what stops its
        # release from unpinning it mid-read.
        framework_pins = set(relevant_pins)
        unused_pins = source_pin.framework_pins - framework_pins
        if waiting:
            source_pin.framework_pins = framework_pins
            self._release_framework_pins(unused_pins)
            return
        source_pin.framework_pins.clear()
        self._source_pin_ops.pop(op_id, None)
        kvcr._remove_block_dependencies(source_pin)

        if source_pin.nvtx_state is not None:
            source_pin.nvtx_state.failure_reason = failure_reason
        source_write = _SourceWriteOp(
            state=(
                _SourceWriteState.READY_TO_WRITE
                if not force_failure
                else _SourceWriteState.NOTIFY_FAILURE
            ),
            keys=frozenset(completed_keys or source_pin.keys),
            started_at=source_pin.started_at,
            deadline=source_pin.deadline,
            op_id=source_pin.op_id,
            remote_agent=source_pin.remote_agent,
            op_handle=source_pin.op_handle,
            dst_descriptors=tuple(
                source_pin.dst_descriptors[index] for index in completed_indices
            ),
            route=source_pin.route,
            nvtx_state=source_pin.nvtx_state,
            _backend=self,
            framework_pins=framework_pins,
            source_keys=completed_keys,
            src_descriptors=tuple(selected_sources),
            completed_indices=tuple(completed_indices),
        )
        kvcr._add_block_dependencies(source_write, new_operation=True)
        self._fw_pins_by_op[source_write.op_id] = set(source_write.framework_pins)
        kvcr._progress.submit(source_write)
        self._release_framework_pins(unused_pins)

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
                if wait is not None and wait.trace is not None:
                    wait.trace.finish("cancelled", reason=_nvtx.Reason.NO_WAITERS)
                self._discard_pin_result(result, wait.keys if wait is not None else ())
                continue

            now = kvcr._clock()
            expired_ops = [(op_id, op) for op_id, op in ops if now >= op.deadline]
            active_ops = [(op_id, op) for op_id, op in ops if now < op.deadline]
            if not active_ops:
                self._record_pending_pin_wait(wait, "timeout")
                self._discard_pin_result(result, wait.keys)
                for op_id, op in expired_ops:
                    op.pending_pin_ids.discard(request)
                    self._expire_source_pin(op_id, op)
                continue

            pin_handle: PinHandle | None = None
            if result is not None and wait is not None:
                pin_handle = self._install_framework_pin(wait.keys, result)
            self._record_pending_pin_wait(
                wait,
                "success" if pin_handle is not None else "failed",
                reason=(
                    _nvtx.Reason.NONE
                    if pin_handle is not None
                    else _nvtx.Reason.UNKNOWN
                    if result is None
                    else _nvtx.Reason.INVALID_RESULT
                ),
                completed_blocks=(
                    _nvtx.pin_result_blocks(result)
                    if pin_handle is not None and wait.trace is not None
                    else -1
                ),
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
        logger.debug(
            "KVCR_EVENT framework_pin_timeout op=%s pending_pins=%d",
            op_id,
            len(op.pending_pin_ids),
        )
        self._cancel_pending_pin_for_op(op_id, op, result="timeout")
        self._submit_prepared_source_write(
            op_id, op, force_failure=True, failure_reason=_nvtx.Reason.DEADLINE
        )

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
            self._submit_prepared_source_write(
                op_id, op, force_failure=True, failure_reason=_nvtx.Reason.DEADLINE
            )
            return

        local_sources = kvcr._claim_local_dram_sources(
            op.op_id, op.keys, allow_filling=True
        )
        unresolved_keys = tuple(
            key
            for key in op.ordered_keys
            if key in op.keys and key not in local_sources
        )
        if unresolved_keys and not op.framework_acquire_attempted:
            if any(
                not kvcr._framework_pin_keys[pin].isdisjoint(unresolved_keys)
                for pin in self._releasing_framework_pins
            ):
                return
            op.framework_acquire_attempted = True
            framework_sources = self._acquire_framework_sources(unresolved_keys, op=op)
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
        new_waiter = wait.trace is not None and op_id not in wait.op_ids
        wait.op_ids.add(op_id)
        op = self._source_pin_ops.get(op_id)
        if op is not None:
            op.pending_pin_ids.add(request)
            if new_waiter and wait.trace is not None:
                wait.trace.waiter(op_id[1], op.op_handle)

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
            if wait.trace is not None:
                wait.trace.detached(op_id[1], op.op_handle)
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
        self,
        wait: _PendingPinWait | None,
        result: str,
        *,
        reason: _nvtx.Reason | None = None,
        completed_blocks: int = -1,
    ) -> None:
        if wait is not None:
            if wait.trace is not None:
                wait.trace.finish(
                    result,
                    reason=_nvtx.Reason.SHUTDOWN if self._closed else reason,
                    completed_blocks=completed_blocks,
                )
            self._kvcr._record_duration("framework_pin_wait", wait.started_at, result)
            if logger.isEnabledFor(logging.DEBUG):
                # A cancelling op has left wait.op_ids but still holds the request.
                source_ops = set(wait.op_ids)
                source_ops.update(
                    op_id
                    for op_id, op in self._source_pin_ops.items()
                    if wait.request in op.pending_pin_ids
                )
                logger.debug(
                    "KVCR_EVENT framework_pin_result pin_request=%d source_ops=%s "
                    "requested_keys=%d result=%s",
                    wait.request,
                    sorted(source_ops),
                    len(wait.keys),
                    result,
                )

    def _discard_pin_result(
        self, result: PinResult, keys: Collection[BlockKey] = ()
    ) -> None:
        if result is None:
            return
        try:
            pin_handle = result[0]
        except (IndexError, TypeError):
            return
        if isinstance(pin_handle, str):
            pin_keys = self._kvcr._framework_pin_keys.setdefault(pin_handle, set())
            pin_keys.update(keys)
            if len(result) > 1 and isinstance(result[1], Mapping):
                pin_keys.update(result[1])
            self._release_framework_pins((pin_handle,))

    # Framework pin ownership.

    def _pin_framework_keys(
        self,
        keys: Collection[BlockKey],
        *,
        op: _SourcePinOp | None = None,
    ) -> PinRequestId | None:
        kvcr = self._kvcr
        if not keys:
            return None
        keys = tuple(keys)
        started_at = kvcr._timer()
        result = "failed"
        trace = None
        if self._nvtx is not None:
            try:
                trace = self._nvtx.begin(
                    len(keys),
                    source_op_id=op.op_id[1] if op is not None else 0,
                    op_handle=op.op_handle if op is not None else 0,
                )
            except Exception:
                pass
        try:
            pushed = trace.push() if trace is not None else False
            try:
                request = kvcr._request_pin_callback(keys)
            finally:
                if trace is not None:
                    trace.tracer.pop(pushed)
            if trace is not None:
                trace.request_id = request
            if request in self._pending_pin_ops:
                if trace is not None:
                    trace.finish("failed", reason=_nvtx.Reason.DUPLICATE_REQUEST)
                logger.warning("KVCR reused pin request id %d", request)
                return None
            wait = _PendingPinWait(request, keys, kvcr._timer(), trace=trace)
            self._pending_pin_ops[request] = wait
            for key in keys:
                self._pending_pin_keys.setdefault(key, set()).add(request)
            result = "pending"
            if trace is not None:
                trace.registered(request)
            return request
        except Exception:
            if trace is not None:
                trace.finish("failed", reason=_nvtx.Reason.CALLBACK_ERROR)
            return None
        finally:
            kvcr._record_duration("source_acquire", started_at, result)

    def _install_framework_pin(
        self,
        keys: Collection[BlockKey],
        pin_result: tuple[PinHandle, Mapping[BlockKey, list[MemoryRef] | None]],
    ) -> PinHandle | None:
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
            block_record = self._kvcr._block_record
            for key, descriptor in normalized.items():
                if descriptor is None:
                    continue
                record = block_record(key)
                if record.fw_mem is not None:
                    continue
                record.fw_mem = _FwMemResidency(descriptor, pin_handle)
                pin_keys.add(key)
            return pin_handle
        except Exception as error:
            if logger.isEnabledFor(logging.DEBUG):
                # Only expose known validation reasons, not callback data or keys.
                reason = error.args[0] if error.args else None
                if type(reason) is not str or reason not in (
                    "invalid framework pin result",
                    "request_pin returned incomplete descriptors",
                    "request_pin returned no descriptors",
                ):
                    reason = "descriptor normalization or installation failed"
                logger.debug("KVCR_EVENT framework_pin_rejected reason=%s", reason)
            self._discard_pin_result(pin_result, keys)
            return None

    def _acquire_framework_sources(
        self,
        keys: tuple[BlockKey, ...],
        *,
        op: _SourcePinOp | None = None,
    ) -> (
        tuple[dict[BlockKey, list[_TransferRef]], set[PinHandle]]
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
            pin_request = self._pin_framework_keys(uncovered_keys, op=op)
            if pin_request is not None:
                pending_pins.append(pin_request)
            if pending_pins:
                return _PendingFrameworkSources(
                    pending_pins=tuple(pending_pins),
                    framework_pins=held_framework_pins,
                )

        descriptors: dict[BlockKey, list[_TransferRef]] = {}
        framework_pins: set[PinHandle] = set()
        for key in keys:
            record = kvcr._block_record_map.get(key)
            residency = record.fw_mem if record is not None else None
            if residency is None:
                continue
            descriptors[key] = residency.descriptors
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
            # Once release starts, its descriptors are no longer safe to reuse.
            # Keep the keys until release is accepted, so overlapping pins wait.
            first_attempt = pin_handle not in self._releasing_framework_pins
            self._releasing_framework_pins.add(pin_handle)
            for key in pin_keys:
                record = kvcr._block_record_map.get(key)
                if (
                    record is not None
                    and record.fw_mem is not None
                    and record.fw_mem.pin_handle == pin_handle
                ):
                    record.fw_mem = None
                    kvcr._prune_block_record(key)
            if self._try_release_pin(pin_handle, warn=first_attempt):
                kvcr._framework_pin_keys.pop(pin_handle, None)
                self._releasing_framework_pins.discard(pin_handle)

    def _try_release_pin(self, pin_handle: PinHandle, *, warn: bool) -> bool:
        # True means the framework accepts responsibility for completing release.
        # False or exceptions cause retries, which must be safe after partial release.
        try:
            released = self._kvcr._release_pin_callback(pin_handle)
        except Exception:
            if warn:
                logger.warning(
                    "KVCR release_pin failed for pin=%r", pin_handle, exc_info=True
                )
            return False
        if released is not True:
            if warn:
                logger.warning("KVCR release_pin failed for pin=%r", pin_handle)
            return False
        return True

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
