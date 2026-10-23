# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optional, best-effort NVTX annotations for framework pin lifetimes.

Context is recorded once; transitions carry only local identity and changes.
Reusable attributes are owned by the emitting thread, with separate storage for
reentrant emission. NVTX consumes payload values synchronously at each call.
Neither profiler presence nor telemetry configuration controls payload work.
"""

import logging
import os
from functools import lru_cache
from hashlib import blake2b
from importlib import import_module
from threading import Lock

from .payload import Reason, Status, _PayloadEmitter
from .pin import _PinTrace

_LOGGER = logging.getLogger(__name__)
_WARNED: set[str] = set()
_NON_FAILURE_REASONS = (
    Reason.NONE,
    Reason.UNKNOWN,
    Reason.RELEASE_ERROR,
    Reason.SHUTDOWN,
)
_MEMORY_KINDS = {"DRAM": 1, "VRAM": 2, "FILE": 3}


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        _LOGGER.warning(message)


def _load_backend():
    nvtx = import_module("nvtx")
    numpy = import_module("numpy")
    return nvtx.get_domain("KVCR"), numpy


def pin_result_blocks(result):
    """Keep diagnostic traversal of a framework-supplied mapping best effort."""
    try:
        return sum(refs is not None for refs in result[1].values())
    except Exception:
        return -1


def create_tracer():
    """Return None before optional imports when off; absence is harmless."""
    level = os.getenv("KVCR_NVTX_LEVEL", "off")
    if level == "off":
        return None
    if level not in ("low", "medium"):
        _warn_once("level", "Invalid KVCR_NVTX_LEVEL; expected off, low or medium.")
        return None
    try:
        domain, numpy = _load_backend()
        return _PinTracer(domain, numpy, level)
    except Exception:
        _warn_once(
            "backend",
            "KVCR NVTX unavailable; install the profiling extra to enable it.",
        )
        return None


class _PinTracer(_PayloadEmitter):
    def __init__(self, domain, numpy, level):
        super().__init__(domain, numpy, level)
        self._worker_identity = lru_cache(maxsize=128)(identity)

    def begin(self, block_count, *, source_op_id=0, op_handle=0):
        pin_id = next(self._ids)
        try:
            wait_range_id = self._emit(
                "source.pin.wait",
                pin_id,
                {},
                category=self.category,
                method=self.domain.start_range,
            )
        except Exception:
            wait_range_id = None
        return _PinTrace(
            self,
            pin_id,
            block_count,
            source_op_id,
            op_handle,
            _wait_range_id=wait_range_id,
        )

    def pop(self, pushed):
        if pushed:
            try:
                self.domain.pop_range()
            except Exception:
                pass

    def name_progress_thread(self):
        # Python 3.12 does not propagate Thread.name to Linux. Nsight displays
        # the OS name; nvtx 0.2.16 has no Python thread-naming API.
        try:
            import ctypes

            ctypes.CDLL(None).prctl(15, ctypes.c_char_p(b"kvcr-progress"), 0, 0, 0)
        except Exception:
            pass

    def lifecycle(
        self, *, target_agent=None, target_incarnation=None, request_id=None, **fields
    ):
        try:
            hi, lo = self._worker_identity(target_agent)
            inc_hi, inc_lo = self._worker_identity(target_incarnation)
            req_hi, req_lo = identity(request_id)
            return _LifecycleTrace(
                self,
                next(self._ids),
                {
                    "target_agent_hi": hi,
                    "target_agent_lo": lo,
                    "target_incarnation_hi": inc_hi,
                    "target_incarnation_lo": inc_lo,
                    "request_hi": req_hi,
                    "request_lo": req_lo,
                    "request_known": request_id is not None,
                    **fields,
                },
                request_id,
            )
        except Exception:
            return None

    def source(self, op, kvcr):
        """Prepare context once; diagnostics cannot interrupt a source write."""
        try:
            detail = {}
            if self.level == "medium":
                refs = tuple(ref for block in op.src_descriptors for ref in block)
                targets = tuple(ref for block in op.dst_descriptors for ref in block)
                detail = dict(
                    selected_bytes=kvcr._descriptor_bytes(refs),
                    source_tier=tier(refs),
                    destination_tier=tier(targets),
                    source_memory=memory_kind(refs, kvcr._memory_regions),
                )
            trace = self.lifecycle(
                target_agent=op.route[0] or None,
                target_incarnation=op.nvtx_state.target_incarnation,
                op_handle=op.op_handle,
                source_op_id=op.op_id[1],
                route_generation=op.route[1],
                requested_blocks=op.nvtx_state.requested_blocks,
                selected_blocks=len(op.source_keys),
                **detail,
            )
            if trace is not None:
                trace.start("source.write")
            return trace
        except Exception:
            return None


def identity(value):
    """Stable 128-bit label identity; no Python hash randomization or truncation."""
    if value is None:
        return 0, 0
    digest = blake2b(value.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest[:8], "big"), int.from_bytes(digest[8:], "big")


def tier(refs):
    """Storage ownership, deliberately independent of DRAM versus VRAM."""
    kinds = {1 if ref.framework else 2 for ref in refs}
    return next(iter(kinds)) if len(kinds) == 1 else 3 if kinds else 0


def memory_kind(refs, regions):
    kinds = set()
    for ref in refs:
        table = regions[0 if ref.framework else 1]
        region = table.get(ref.label)
        if region is None:
            region = table.get(
                ref.label.partition(":")[0] + (":*" if ref.framework else "")
            )
        kinds.add(_MEMORY_KINDS.get(getattr(region, "mem_type", None), 0))
    return next(iter(kinds)) if len(kinds) == 1 else 0


class _LifecycleTrace:
    """One-time context emission plus operation lifecycle bookkeeping."""

    def __init__(self, tracer, trace_id, fields, request_id):
        self.tracer, self.trace_id = tracer, trace_id
        self.requested_blocks = fields.get("requested_blocks", -1)
        self.selected_bytes = fields.get("selected_bytes", -1)
        self._observed = set()
        self.failure_reason = Reason.UNKNOWN
        self._ranges = {}
        self._range_lock = Lock()
        try:
            if tracer.level == "medium" and request_id is not None:
                encoded = request_id.encode("utf-8")
                display = tracer.numpy.frombuffer(
                    encoded[:256].ljust(256, b"\0"), dtype="u1"
                )
                fields.update(
                    request_display_known=1,
                    length=min(len(encoded), 256),
                    truncated=len(encoded) > 256,
                    value_utf8=display,
                )
            tracer._emit(
                "request.context",
                self.trace_id,
                fields,
                method=tracer.domain.mark,
                context=True,
            )
        except Exception:
            pass

    def start(self, name, **fields):
        """The owner finishes start before handing work to another thread.

        Source/native start and end stay on the progress thread. Concurrent
        endings are supported; overlapping start/end of one range is not.
        """
        if name in self._ranges:
            return
        try:
            self._ranges[name] = self.tracer._emit(
                name,
                self.trace_id,
                fields,
                method=self.tracer.domain.start_range,
            )
        except Exception:
            pass

    def end(self, name):
        """Claim the ending once; zero is a valid native range ID."""
        with self._range_lock:
            range_id = self._ranges.pop(name, None)
        if range_id is not None:
            try:
                self.tracer.domain.end_range(range_id)
            except Exception:
                pass

    def reject(self, name, reason, *, status=Status.REJECTED):
        self.mark(name, once=True, status=status, reason=reason)

    def complete_source(self, success, blocks, *, reason=None):
        requested = self.requested_blocks
        status = Status.SUCCESS if success else Status.FAILED
        if success and blocks < requested:
            status = Status.FAILED if blocks == 0 else Status.PARTIAL
        if reason is None:
            reason = (
                Reason.NONE if success and blocks >= requested else self.failure_reason
            )
        try:
            self.mark(
                "source.write.completed",
                once=True,
                status=status,
                reason=reason,
                completed_blocks=blocks if success else 0,
                completed_bytes=self.selected_bytes if success else 0,
            )
        finally:
            self.end("source.write")

    def mark(self, name, *, once=False, detail=False, **fields):
        reason = fields.get("reason", Reason.NONE)
        if reason not in _NON_FAILURE_REASONS:
            self.failure_reason = reason
        if detail and self.tracer.level != "medium":
            return
        if once:
            if name in self._observed:
                return
            self._observed.add(name)
        try:
            self.tracer._emit(
                name, self.trace_id, fields, method=self.tracer.domain.mark
            )
        except Exception:
            pass

    def push(self, name, **fields):
        try:
            self.tracer._emit(
                name, self.trace_id, fields, method=self.tracer.domain.push_range
            )
            return True
        except Exception:
            return False
