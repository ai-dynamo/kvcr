# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Caller-owned source acquisition and framework pin lifetime for remote writes."""

import logging
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .. import _nvtx
from ..runtime.core import logger
from ..runtime.progress import _Op, _OpId, _TransferRef
from ..types import BlockKey, MemoryRef, PinHandle, PinRequestId, PinResult
from .local_dram_pools import _layout_indices, _LocalDramState
from .remote_ops import _NvtxState, _SourceWriteOp, _SourceWriteState

if TYPE_CHECKING:
    from ..runtime.core import _KVCRCore
    from .remote_fw_dram import _RemoteFWDram


@dataclass(slots=True)
class _FwMemResidency:
    descriptors: list[_TransferRef]
    pin_handle: PinHandle


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


@dataclass
class _PendingPinWait:
    """One framework pin request and the G2 operations waiting on it."""

    request: PinRequestId
    keys: tuple[BlockKey, ...]
    started_at: float | None
    op_ids: set[_OpId] = field(default_factory=set)
    trace: _nvtx._PinTrace | None = None


class _RemoteSources:
    """Own caller-side source operations and pins until their writes return."""

    def __init__(self, backend: "_RemoteFWDram", kvcr: "_KVCRCore") -> None:
        self._backend = backend
        self._kvcr = kvcr
        self._source_pin_ops: dict[_OpId, _SourcePinOp] = {}
        self._pending_pin_ops: dict[PinRequestId, _PendingPinWait] = {}
        self._pending_pin_keys: dict[BlockKey, set[PinRequestId]] = {}
        # Pins retained while their source operations execute.
        self._fw_pins_by_op: dict[_OpId, set[PinHandle]] = {}
        self._releasing_framework_pins: set[PinHandle] = set()

    def retry_releases(self) -> None:
        self._release_framework_pins(tuple(self._releasing_framework_pins))

    def finish_source_write(self, item: _SourceWriteOp) -> None:
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

    def poll(self) -> None:
        self._process_pending_pin_results()
        now = self._kvcr._clock()
        for op_id, op in list(self._source_pin_ops.items()):
            if now >= op.deadline:
                logger.warning("KVCR operation %r expired", op_id)
                self._expire_source_pin(op_id, op)
            elif not op.pending_pin_ids:
                self._resume_source_pin(op_id, op)

    def close(self) -> None:
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
            self._backend._route_generation.get(route_name, 0) != route_generation
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
                else tuple(descriptor.label for descriptor in source)
            )
            if source_pin.key_layout and source_pin.key_layout != layout:
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
                kvcr._local_dram._pools.source_refs(residency, indices)
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
            _backend=self._backend,
            framework_pins=framework_pins,
            source_keys=completed_keys,
            src_descriptors=tuple(selected_sources),
            completed_indices=tuple(completed_indices),
        )
        kvcr._add_block_dependencies(source_write, new_operation=True)
        self._fw_pins_by_op[source_write.op_id] = set(source_write.framework_pins)
        kvcr._progress.submit(source_write)
        self._release_framework_pins(unused_pins)

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

    def start_source_pin(self, op: _SourcePinOp) -> None:
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
            if new_waiter:
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
                    reason=_nvtx.Reason.SHUTDOWN if self._backend._closed else reason,
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
        if self._backend._nvtx is not None:
            try:
                trace = self._backend._nvtx.begin(
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
