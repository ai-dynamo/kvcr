# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded handling of dangling operations and late-completion diagnostics."""

import heapq
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from .core import logger
from .types import OpHandle

if TYPE_CHECKING:
    from .progress import _KVCRProgress
    from .remote_fw_dram import _RemoteFWDram, _SourceWriteOp, _TargetPullOp


class _DanglingOpError(RuntimeError):
    def __init__(self, message: str, op: "_SourceWriteOp | _TargetPullOp") -> None:
        self.op_handle = (
            op.op_id[1]
            if op.op_id[0] == "target"
            else cast("_SourceWriteOp", op).op_handle
        )
        self.dst_descriptors = op.dst_descriptors
        super().__init__(
            f"{message}: op={self.op_handle}, destinations={self.dst_descriptors!r}"
        )


@dataclass
class _SourceWriteStatus:
    submitted: bool = False
    cancel_requested: bool = False
    cancel_deadline: float | None = None


@dataclass
class _Tombstone:
    operation: "_TargetPullOp"
    expires_at: float | None = None


class _DanglingOps:
    """Progress-thread state for probes, stalled sources, and tombstones."""

    def __init__(self, backend: "_RemoteFWDram") -> None:
        self._backend = backend
        self.incarnation = uuid.uuid4().hex
        self.dead_incarnations: set[str] = set()
        self.sources: dict[str, str] = {}
        self.source_writes: dict[tuple[str, OpHandle], _SourceWriteStatus] = {}
        # TODO: Add a retention limit if unresolved tombstones accumulate.
        self.tombstones: dict[OpHandle, _Tombstone] = {}
        self._expirations: list[tuple[float, OpHandle]] = []
        self._last_progress_at: float | None = None
        self._source_stalled = False

    def begin_poll(self) -> None:
        polled_at = time.monotonic()
        self.check_source_progress()
        self._last_progress_at = polled_at

    def check_source_progress(self) -> bool:
        if self._source_stalled or self._last_progress_at is None:
            return not self._source_stalled
        gap_ms = (time.monotonic() - self._last_progress_at) * 1000
        timeout_ms = self._backend._kvcr.config.operation_timeout_ms
        if gap_ms >= timeout_ms:
            # Stay disabled so queued pre-stall requests cannot get fresh deadlines.
            self._source_stalled = True
            logger.error(
                "KVCR source progress stalled for %.1f ms (timeout: %d ms); "
                "new source writes disabled until restart",
                gap_ms,
                timeout_ms,
            )
        return not self._source_stalled

    def poll_source(
        self, progress: "_KVCRProgress", op: "_SourceWriteOp", *, cancelling: bool
    ) -> tuple[bool, Any | None] | None:
        status = self.source_writes[(op.route[0], op.op_handle)]
        first_attempt = cancelling and status.cancel_deadline is None
        if first_attempt:
            status.cancel_deadline = (
                min(op.deadline, self._backend._kvcr._clock())
                + self._backend._kvcr.config.operation_timeout_ms / 1000
            )
        result = progress.poll_transfer(
            cast(int, op.transfer_id), cancellation_requested=cancelling
        )
        if result is None and cancelling:
            if first_attempt:
                self._backend._send_write_done(
                    progress, op.remote_agent, op.op_handle, False, terminal=False
                )
            if self._backend._kvcr._clock() >= cast(float, status.cancel_deadline):
                # Propagate through normal polling; cleanup must retain native work.
                raise _DanglingOpError("KVCR source cancellation timed out", op)
        return result

    def probe(self, progress: "_KVCRProgress", op: "_TargetPullOp") -> bool:
        return self._backend._send_control(
            progress,
            op.remote_ctrl_ep,
            {
                "type": "write_probe",
                "op_handle": op.op_id[1],
                "source_incarnation": op.source_incarnation,
            },
        )

    def abandon(self, progress: "_KVCRProgress", op: "_TargetPullOp") -> None:
        # Best effort, not a transport fence. The caller may reuse these addresses.
        self.tombstones[op.op_id[1]] = _Tombstone(op)
        self._backend._invalidate_control_peer(op.remote_ctrl_ep)
        self.probe(progress, op)  # Cleanup only: never extends the operation deadline.

    def notification(
        self, progress: "_KVCRProgress", op_handle: OpHandle, payload: dict[str, Any]
    ) -> None:
        tombstone = self.tombstones.get(op_handle)
        if tombstone is None:
            return
        if (
            payload.get("success") is True
            and ("target", op_handle) not in progress._in_flight_ops
        ):
            raise _DanglingOpError(
                f"KVCR late remote write from {payload.get('source_agent')!r}",
                tombstone.operation,
            )
        self.tombstones.pop(op_handle)

    def expire(self) -> None:
        now = self._backend._kvcr._clock()
        while self._expirations and self._expirations[0][0] <= now:
            _, handle = heapq.heappop(self._expirations)
            self.tombstones.pop(handle, None)

    def handle_probe(self, progress: "_KVCRProgress", payload: dict[str, Any]) -> None:
        target_agent = payload.get("target_agent")
        handle = payload.get("op_handle")
        reply_to = payload.get("sender_control_endpoint")
        endpoint = payload.get("source_control_endpoint")
        expected = payload.get("source_incarnation")
        if (
            not isinstance(target_agent, str)
            or not target_agent
            or type(handle) is not int
            or not isinstance(reply_to, str)
            or not reply_to
            or not isinstance(endpoint, str)
            or not endpoint
            or (expected is not None and not isinstance(expected, str))
        ):
            return
        status = self.source_writes.get((target_agent, handle))
        if status is not None and expected in (None, self.incarnation):
            status.cancel_requested = True
        response = {
            "type": "write_probe_ack",
            "sender_control_endpoint": endpoint,
            "op_handle": handle,
            "terminal": expected in (None, self.incarnation)
            and (status is None or not status.submitted),
        }
        if expected and expected in self.dead_incarnations:
            response["dead_incarnation"] = expected
        self._backend._send_control(progress, reply_to, response)

    def handle_probe_ack(
        self, progress: "_KVCRProgress", payload: dict[str, Any]
    ) -> None:
        endpoint = payload.get("sender_control_endpoint")
        handle = payload.get("op_handle")
        terminal = payload.get("terminal")
        incarnation = payload.get("sender_incarnation")
        if (
            not isinstance(endpoint, str)
            or type(handle) is not int
            or type(terminal) is not bool
            or not isinstance(incarnation, str)
            or not incarnation
        ):
            return
        active = cast(
            "_TargetPullOp | None", progress._in_flight_ops.get(("target", handle))
        )
        tombstone = self.tombstones.get(handle)
        op = tombstone.operation if tombstone is not None else active
        if op is None or op.remote_ctrl_ep != endpoint:
            return
        expected = op.source_incarnation
        if expected and payload.get("dead_incarnation") == expected:
            if tombstone is None:
                tombstone = self.tombstones[handle] = _Tombstone(op)
            if tombstone.expires_at is None:
                # One metadata-only grace period after confirmed process death.
                # Expiry ends late-write diagnostics; it does not establish a fence.
                tombstone.expires_at = (
                    self._backend._kvcr._clock()
                    + self._backend._kvcr.config.operation_timeout_ms / 1000
                )
                heapq.heappush(self._expirations, (tombstone.expires_at, handle))
            self.sources[endpoint] = incarnation
            if active is not None:
                self._backend._refused_writes[("target", handle)] = {"success": False}
        elif expected == incarnation and terminal:
            self.tombstones.pop(handle, None)
            if active is not None:
                self._backend._refused_writes[("target", handle)] = {"success": False}
