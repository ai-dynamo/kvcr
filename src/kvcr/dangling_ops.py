# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Cancellation, quarantine, and lifecycle events for unresolved remote writes."""

import time
import uuid
from dataclasses import dataclass, replace
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, cast

from .core import logger
from .types import OpHandle, TransferError

if TYPE_CHECKING:
    from .progress import _KVCRProgress
    from .remote_fw_dram import _RemoteFWDram, _SourceWriteOp, _TargetPullOp


def _log_resilience_event(error: Exception) -> None:
    log = (
        logger.info
        if isinstance(error, TransferError) and error.state == "quiesced"
        else logger.error
    )
    log("%s", error)


@dataclass
class _SourceWriteStatus:
    submitted: bool = False
    cancel_requested: bool = False
    cancel_deadline: float | None = None
    abandoned: bool = False


class _DanglingOps:
    """Progress-thread resilience policy for operations that retain their memory."""

    def __init__(self, backend: "_RemoteFWDram") -> None:
        self._backend = backend
        self.incarnation = uuid.uuid4().hex
        self.dead_incarnations: set[str] = set()
        self.sources: dict[str, str] = {}
        self.source_writes: dict[tuple[str, OpHandle], _SourceWriteStatus] = {}
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
            self._backend._progress_outbound.append(
                RuntimeError(
                    f"KVCR source progress stalled for {gap_ms:.1f} ms "
                    f"(timeout: {timeout_ms} ms); "
                    "new source writes disabled until restart"
                )
            )
        return not self._source_stalled

    def poll_source(
        self, progress: "_KVCRProgress", op: "_SourceWriteOp", *, cancelling: bool
    ) -> tuple[bool, Any | None] | None:
        status = self.source_writes[(op.route[0], op.op_handle)]
        if cancelling and status.cancel_deadline is None:
            config = self._backend._kvcr.config
            status.cancel_deadline = (
                op.deadline
                + (config.abandon_timeout_ms - config.operation_timeout_ms) / 1000
            )
            self._backend._send_write_done(
                progress, op.remote_agent, op.op_handle, False, terminal=False
            )
        result = progress.poll_transfer(
            cast(int, op.transfer_id), require_completion=True
        )
        if (
            result is None
            and cancelling
            and not status.abandoned
            and self._backend._kvcr._clock() >= cast(float, status.cancel_deadline)
        ):
            status.abandoned = True
            self.report_source(op, "uncertain")
            self._backend._progress_outbound.append(replace(op))
        return result

    def finish_source(self, op: "_SourceWriteOp") -> None:
        status = self.source_writes.pop((op.route[0], op.op_handle), None)
        if status is not None and status.abandoned:
            self.report_source(op, "quiesced")

    def report_source(
        self, op: "_SourceWriteOp", state: Literal["uncertain", "quiesced"]
    ) -> None:
        self._backend._progress_outbound.append(
            TransferError(
                "KVCR source write memory",
                OpHandle(op.op_id[1]),
                state=state,
                source_blocks={
                    key: list(descriptors)
                    for key, descriptors in zip(op.ordered_keys, op.src_descriptors)
                },
            )
        )

    def poll_target(
        self, progress: "_KVCRProgress", op: "_TargetPullOp", now: float
    ) -> bool:
        first_poll = not op.uncertain
        if first_poll:
            op.uncertain = True
            self.report_target(op, "uncertain")
            self._backend._progress_outbound.append(replace(op))
            self._backend._invalidate_control_peer(op.remote_ctrl_ep)
        if first_poll or now >= op.deadline:
            op.deadline = now + self._backend._kvcr.config.operation_timeout_ms / 1000
            return self.probe(progress, op)
        return False

    def report_target(
        self, op: "_TargetPullOp", state: Literal["uncertain", "quiesced"]
    ) -> None:
        self._backend._progress_outbound.append(
            TransferError(
                "KVCR remote write memory",
                op.op_id[1],
                state=state,
                destination_regions=list(chain.from_iterable(op.dst_descriptors)),
            )
        )

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
        if expected in (None, self.incarnation):
            # A probe can overtake start_write after a control reconnect. Keep
            # this small cancellation fence until restart, even if no op exists.
            status = self.source_writes.setdefault(
                (target_agent, handle), _SourceWriteStatus()
            )
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
        op = cast(
            "_TargetPullOp | None", progress._in_flight_ops.get(("target", handle))
        )
        if op is None or op.remote_ctrl_ep != endpoint:
            return
        expected = op.source_incarnation
        if expected and payload.get("dead_incarnation") == expected:
            # Guard verifies death of this process, not just an expired heartbeat.
            self.sources[endpoint] = incarnation
            self._backend._refused_writes[("target", handle)] = {"success": False}
        elif expected == incarnation and terminal:
            self._backend._refused_writes[("target", handle)] = {"success": False}
