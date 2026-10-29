# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit per-operation observation; no locals snapshots or per-poll JSON."""

import functools
import threading
import time
import uuid
from collections.abc import Mapping
from itertools import chain

from . import compact_telemetry as telemetry

_local = threading.local()


def current():
    return getattr(_local, "context", None)


def reason(value):
    context = current()
    if context and value not in context.reasons:
        context.reasons.append(value)


class Context:
    def __init__(self, capture, backend, side, keys, request_id=None, correlation=None):
        self.capture, self.backend, self.side = capture, backend, side
        # Ingress validation belongs to the handler; observation must not iterate
        # malformed/null scalars first and bypass its failure notification.
        self.keys = (
            tuple(keys)
            if (
                isinstance(keys, (list, tuple))
                or (side == "target" and isinstance(keys, Mapping))
            )
            else ()
        )
        self.source_requested_keys = self.keys if side == "source" else ()
        self.source_ready_keys = ()
        self.operation_key = uuid.uuid4().hex
        self.request_id, self.correlation = request_id, correlation
        self.agent = backend._kvcr.nixl_agent_name
        self.incarnation = backend._dangling_ops.incarnation
        self.timings = {
            "source_receive_ns"
            if side == "source"
            else "target_request_ns": time.monotonic_ns()
        }
        self.cpu_ns = self.depth = 0
        self._handling_lock = threading.Lock()
        self._handling = {}
        self.reasons = []
        self.owned = self.done = self.emitted = False
        self.op = None
        self.native = None
        self.native_submitted = None
        self.native_state = None
        self.capture.emit(
            dict(
                event="operation_start",
                operation_key=self.operation_key,
                agent=self.agent,
                identity=self.agent,
                side=side,
                request_id=request_id,
            )
        )

    def enter(self):
        thread = threading.get_ident()
        with self._handling_lock:
            interval = self._handling.get(thread)
            if interval is None:
                self._handling[thread] = [1, time.thread_time_ns()]
            else:
                interval[0] += 1
            self.depth += 1

    def leave(self):
        thread = threading.get_ident()
        with self._handling_lock:
            interval = self._handling[thread]
            interval[0] -= 1
            self.depth -= 1
            if interval[0] == 0:
                self.cpu_ns += time.thread_time_ns() - interval[1]
                del self._handling[thread]
            emit = self.done and not self._handling
        if emit:
            self.emit_terminal()

    def finish(self, op=None, closed=False):
        with self._handling_lock:
            if self.done:
                return
            self.op, self.closed, self.done = op, closed, True
            now = time.monotonic_ns()
            self.timings.setdefault("cleanup_done_ns", now)
            self.timings[
                "source_end_ns" if self.side == "source" else "target_end_ns"
            ] = now

    def emit_terminal(self):
        with self._handling_lock:
            if self.emitted or not self.done or self._handling:
                return
            self.emitted = True
        try:
            op = self.op
            source = self.side == "source"
            success = bool(op and op.success) and not self.closed
            requested = len(self.keys)
            indices = (
                tuple(getattr(op, "completed_indices", ()))
                if source
                else tuple(
                    i
                    for i, key in enumerate(getattr(op, "ordered_keys", ()))
                    if key in getattr(op, "completed_keys", ())
                )
            )
            blocks = len(getattr(op, "source_keys", ())) if source else len(indices)
            size = None
            if success:
                if source and self.native is not None:
                    size = getattr(self.native, "totalBytes", None)
                if size is None:
                    descriptors = (
                        getattr(op, "src_descriptors", ())
                        if source
                        else tuple(
                            refs
                            for key, refs in zip(op.ordered_keys, op.dst_descriptors)
                            if key in op.completed_keys
                        )
                    )
                    size = self.backend._kvcr._descriptor_bytes(
                        chain.from_iterable(descriptors)
                    )
            started = getattr(op, "started_at", None)
            end_ns = self.timings["source_end_ns" if source else "target_end_ns"]
            elapsed = (
                end_ns / 1e9 - started
                if started is not None
                else (end_ns - next(iter(self.timings.values()))) / 1e9
            )
            elapsed = max(0.0, elapsed)
            self.timings["telemetry_enqueue_ns"] = time.monotonic_ns()
            self.capture.emit(
                dict(
                    event="source_nixl_terminal" if source else "target_terminal",
                    identity=self.agent,
                    agent=self.agent,
                    operation_key=self.operation_key,
                    side=self.side,
                    op_id=getattr(op, "op_id", None),
                    source_op=getattr(op, "op_id", (None, None))[1] if source else None,
                    target_op=getattr(op, "op_handle", None)
                    if source
                    else getattr(op, "op_id", (None, None))[1],
                    request_id=self.request_id,
                    request_correlation=self.correlation,
                    incarnation=self.incarnation,
                    status=("partial" if blocks < requested else "success")
                    if success
                    else "failed",
                    result=("success" if success else "failed") if source else None,
                    reason=self.reasons[-1]
                    if self.reasons
                    else "ready_subset"
                    if success and blocks < requested
                    else "complete"
                    if success
                    else "shutdown"
                    if self.closed
                    else "not_started",
                    reasons=list(self.reasons),
                    source_agent=self.agent if source else None,
                    source_control_endpoint=None
                    if source
                    else getattr(op, "remote_ctrl_ep", None),
                    target_agent=getattr(op, "remote_agent", None)
                    if source
                    else self.agent,
                    bytes=int(size) if size is not None else None,
                    blocks=blocks if success else None,
                    bytes_source="native_telemetry"
                    if source and self.native is not None
                    else "validated_descriptors"
                    if success
                    else None,
                    requested_blocks=requested,
                    guard_requested_count=requested if source else None,
                    completed_indices=list(indices),
                    no_pinning_mode=bool(self.backend._options.no_pinning_mode),
                    initial_caller_handoff="absent_by_no_pinning_progress_path"
                    if source and self.backend._options.no_pinning_mode
                    else None,
                    has_framework_pins=bool(getattr(op, "framework_pins", ())),
                    ownership="kvcr" if source else "target",
                    source_tier="kvcr" if source else None,
                    native_seconds=getattr(self.native, "xferDuration", 0) / 1e6
                    if self.native is not None
                    else None,
                    post_seconds=getattr(self.native, "postDuration", 0) / 1e6
                    if self.native is not None
                    else None,
                    native=dict(
                        submitted=self.native_submitted, xfer_state=self.native_state
                    ),
                    source_operation_seconds=elapsed if source else None,
                    lifecycle_seconds=elapsed,
                    end_to_end_seconds=elapsed,
                    start_monotonic_seconds=started,
                    end_monotonic_seconds=end_ns / 1e9,
                    source_handling_cpu_ns=self.cpu_ns if source else None,
                    handling_cpu_ns=self.cpu_ns,
                    timing_ns=dict(self.timings),
                    ucx_tls=__import__("os").environ.get("UCX_TLS"),
                    ucx_net_devices=__import__("os").environ.get(
                        "UCX_NET_DEVICES", "all"
                    ),
                    end_wall_ns=time.time_ns(),
                )
            )
        except Exception as error:
            self.capture.invalidate(f"terminal serialization: {type(error).__name__}")


def request(side):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(backend, *args, **kwargs):
            output = telemetry.writer()
            if output is None or (
                side == "source" and not backend._options.no_pinning_mode
            ):
                return function(backend, *args, **kwargs)
            payload = args[1] if side == "source" else None
            keys = (
                (
                    payload.get("keys", ())
                    if isinstance(payload, Mapping)
                    else payload.keys
                )
                if payload is not None
                else args[0]
            )
            request_id = (
                (
                    payload.get("request_id")
                    if isinstance(payload, Mapping)
                    else payload.request_id
                )
                if payload is not None
                else args[1]
            )
            context = Context(
                output,
                backend,
                side,
                keys,
                request_id,
                (
                    payload.get("telemetry_request_key")
                    if isinstance(payload, Mapping)
                    else getattr(payload, "telemetry_request_key", None)
                )
                if payload is not None
                else None,
            )
            old = current()
            _local.context = context
            context.enter()
            try:
                if side == "target":
                    hint = backend._request_hints.get(request_id)
                    output.journal(
                        dict(
                            event="target_get",
                            agent=context.agent,
                            incarnation=context.incarnation,
                            operation_key=context.operation_key,
                            request_id=request_id,
                            keys=context.keys,
                            hint_source=getattr(hint, "source", None),
                            allowed_hint_hashes=(
                                list(hint.block_hashes)
                                if hint is not None and len(hint.block_hashes) <= 4096
                                else None
                            ),
                            allowed_hint_hash_count=len(hint.block_hashes)
                            if hint is not None
                            else 0,
                            allowed_hint_hashes_complete=hint is not None
                            and len(hint.block_hashes) <= 4096,
                            allowed_hint_hashes_omitted_reason=(
                                "bounded_prefix_closure_proof"
                                if hint is not None and len(hint.block_hashes) > 4096
                                else None
                            ),
                            hint_group_coverage="not represented by hash-only hint",
                        )
                    )
                return function(backend, *args, **kwargs)
            except BaseException as error:
                reason(f"exception:{type(error).__name__}")
                raise
            finally:
                if not context.owned:
                    context.finish()
                context.leave()
                _local.context = old

        return wrapped

    return decorate


def bind(op):
    context = current()
    if context:
        context.owned = True
        op._capture = context


def operation(function):
    @functools.wraps(function)
    def wrapped(op, *args, **kwargs):
        context = getattr(op, "_capture", None)
        if context is None:
            return function(op, *args, **kwargs)
        old = current()
        _local.context = context
        context.enter()
        try:
            result = function(op, *args, **kwargs)
            done = result if function.__name__ == "close" else result[0]
            if done:
                if function.__name__ == "close":
                    reason("shutdown")
                context.finish(op, closed=function.__name__ == "close")
            return result
        except BaseException as error:
            reason(f"exception:{type(error).__name__}")
            raise
        finally:
            context.leave()
            _local.context = old

    return wrapped


def native_submit(function):
    @functools.wraps(function)
    def wrapped(progress, *args, **kwargs):
        context = current()
        if context is None:
            return function(progress, *args, **kwargs)
        context.timings.setdefault("submit_begin_ns", time.monotonic_ns())
        try:
            result = function(progress, *args, **kwargs)
            context.native_submitted = result[1]
            return result
        except BaseException as error:
            reason(f"submit_exception:{type(error).__name__}")
            raise
        finally:
            context.timings["submit_end_ns"] = time.monotonic_ns()

    return wrapped


def progress_turn(function):
    @functools.wraps(function)
    def wrapped(progress):
        output = getattr(progress, "_capture_writer", None)
        if output is None:
            return function(progress)
        start, cpu = time.monotonic_ns(), time.thread_time_ns()
        previous = getattr(progress, "_capture_turn_start", None)
        previous_end = getattr(progress, "_capture_turn_end", None)
        progress._capture_turn_start = start
        inflight = bool(progress._active_transfers or progress._in_flight_ops)
        work = False
        try:
            work = function(progress)
            return work
        finally:
            end, end_cpu = time.monotonic_ns(), time.thread_time_ns()
            progress._capture_turn_end = end
            inflight |= bool(progress._active_transfers or progress._in_flight_ops)
            output.progress_turn(
                progress._nixl_agent_name,
                bool(work or inflight),
                end - start,
                end_cpu - cpu,
                start - previous if previous is not None else None,
                end_to_start_gap_ns=start - previous_end
                if previous_end is not None
                else None,
                observed_work=bool(work),
                inflight=inflight,
            )

    return wrapped
