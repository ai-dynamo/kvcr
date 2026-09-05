# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KVCR lifecycle and state implementation."""

import functools
import logging
import time
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING

from .config import (
    KVCRBackendConfigs,
    KVCRConfig,
    TelemetryStats,
    _validate_pool_layout,
)
from .hint_parser import _parse_kv_hint
from .local_disk import _G3, _G3Residency
from .local_dram import _LocalDram, _LocalDramResidency, _LocalDramState
from .policy import G3LRUPolicy, LRUPolicy
from .policy_runtime import _PolicyInvoker
from .progress import _KVCRProgress, _Op, _OpId
from .types import (
    BlockKey,
    BlockMeta,
    CacheTier,
    InventoryEvent,
    MemDescriptor,
    OpEntryResult,
    OpEntryStatus,
    OpHandle,
    OpResult,
    PinHandle,
    PlacementDecision,
    QueryStatus,
    RecoveryMirrorError,
    ReleaseHandle,
    ReleaseResult,
)

if TYPE_CHECKING:
    from .api import KVCRBindings
    from .remote_fw_dram import _FwMemResidency

logger = logging.getLogger(__name__)

DURATION_METRIC = "kvcr_duration_seconds"
TRANSFER_BYTES_METRIC = "kvcr_transfer_bytes"
TRANSFER_BLOCKS_METRIC = "kvcr_transfer_blocks"
STATE_METRIC = "kvcr_state"

_Timer = Callable[[], float | None]
_Clock = Callable[[], float]
_RecordDuration = Callable[[str, float | None, str], None]
_RecordTransfer = Callable[[str, float | None, bool, int, int], None]


def _noop_timer() -> None:
    return None


def _noop_record_duration(scope: str, started_at: float | None, result: str) -> None:
    return None


def _noop_record_transfer(
    scope: str,
    started_at: float | None,
    success: bool,
    block_count: int,
    byte_count: int,
) -> None:
    return None


@dataclass(slots=True)
class _BlockRecord:
    # Locally pinned framework-owned G2 memory. This is never remote
    # KVCR residency and exists only while KVCR controls the pin.
    fw_mem: "_FwMemResidency | None" = None
    local_dram: _LocalDramResidency | None = None
    g3: _G3Residency | None = None
    in_flight_ops: set[_OpId] | None = None
    access_count: int = 0
    last_access: float | None = None

    def add_in_flight_op(self, op_id: _OpId) -> None:
        if self.in_flight_ops is None:
            self.in_flight_ops = {op_id}
        else:
            self.in_flight_ops.add(op_id)

    def discard_in_flight_op(self, op_id: _OpId) -> None:
        ops = self.in_flight_ops
        if ops is not None:
            ops.discard(op_id)
            if not ops:
                self.in_flight_ops = None

    @property
    def active_op_ids(self) -> tuple[_OpId, ...]:
        """Snapshot of in-flight op ids, safe to iterate while mutating."""
        ops = self.in_flight_ops
        return () if ops is None else tuple(ops)


class _KVCRCore:
    """KVCR lifecycle/state coordinator for memory-path backends."""

    def __init__(
        self,
        config: KVCRConfig,
        bindings: "KVCRBindings",
        backend_configs: KVCRBackendConfigs,
    ) -> None:
        self.config = config
        self.pool_layout = list(config.pool_layout)
        _validate_pool_layout(self.pool_layout)
        # TODO: Support multiple pools after remote fetch and G3 discover layouts.
        if len(self.pool_layout) != 1:
            raise ValueError("only a single pool is currently supported")
        self.block_size_bytes = self.pool_layout[0][0]
        if self.config.operation_timeout_ms <= 0:
            raise ValueError("operation_timeout_ms must be positive")
        if self.config.inventory_report_interval_ms < 0:
            raise ValueError("inventory_report_interval_ms must be non-negative")
        if not 0 <= self.config.capacity_low_watermark_percent <= 100:
            raise ValueError("capacity_low_watermark_percent must be between 0 and 100")

        self.nixl_agent_name = self.config.nixl_agent_name
        self._request_pin_callback = bindings.request_pin
        self._poll_pin_results_callback = bindings.poll_pin_results
        self._release_pin_callback = bindings.release_pin
        self._cancel_pin_request_callback = bindings.cancel_pin_request
        self.framework_control = bindings.framework_control
        self._inventory_sink_callback = bindings.inventory_sink
        self._capacity_needed_callback = bindings.capacity_needed_callback
        self._stats_factory = bindings.stats_factory
        local_dram_config = backend_configs.local_dram
        g3_config = backend_configs.g3
        policy = bindings.policy
        if policy is None:
            policy = G3LRUPolicy() if g3_config is not None else LRUPolicy()
        configured_tiers = set()
        if local_dram_config is not None:
            configured_tiers.add(CacheTier.LOCAL_G2)
        if g3_config is not None:
            configured_tiers.add(CacheTier.G3)
        missing_tiers = policy.required_tiers - configured_tiers
        if missing_tiers:
            names = ", ".join(sorted(tier.value for tier in missing_tiers))
            raise ValueError(f"{type(policy).__name__} requires configured {names}")
        self._policy = _PolicyInvoker(
            policy,
            (((CacheTier.LOCAL_G2, CacheTier.G3),) if g3_config is not None else ()),
        )

        # Common KVCR state tables.
        if g3_config is not None and local_dram_config is None:
            raise ValueError("G3 requires configured local DRAM")
        self._block_record_map: dict[BlockKey, _BlockRecord] = {}
        self._pending_inventory_events: list[InventoryEvent] = []
        self._inventory_flush_deadline: float | None = None
        self._capacity_pressure_active = False
        self._closed = False
        self._outstanding_operations = 0
        self._framework_pin_keys: dict[PinHandle, set[BlockKey]] = {}
        self._local_dram_sources_by_op: dict[_OpId, dict[BlockKey, MemDescriptor]] = {}

        self._completion_queue: list[OpResult] = []
        self._joined_completions: dict[
            OpHandle, tuple[set[BlockKey], dict[BlockKey, OpEntryResult]]
        ] = {}
        # Both seeded from the clock rather than a constant, so a refusal sent before a
        # restart cannot complete a different operation that reused the number.
        handle_seed = time.monotonic_ns()
        self._next_op_handle: OpHandle = OpHandle(handle_seed)
        self._next_fill_handle: OpHandle = OpHandle(-handle_seed)

        # Operational clock and optional telemetry clock.
        self._clock: _Clock = time.monotonic
        self._inventory_report_interval = (
            self.config.inventory_report_interval_ms / 1000
        )
        stats_factory = self._stats_factory if self.config.enable_telemetry else None
        telemetry_enabled = stats_factory is not None
        self._stats = stats_factory() if stats_factory is not None else None
        self._timer: _Timer = time.monotonic if telemetry_enabled else _noop_timer
        self._record_duration: _RecordDuration = (
            self._record_duration_enabled
            if telemetry_enabled
            else _noop_record_duration
        )
        self._record_transfer: _RecordTransfer = (
            self._record_transfer_enabled
            if telemetry_enabled
            else _noop_record_transfer
        )

        # Import lazily to keep the concrete backend private to KVCR setup.
        from .remote_fw_dram import _RemoteFWDram

        self._local_dram = (
            _LocalDram(self, local_dram_config)
            if local_dram_config is not None
            else None
        )
        self._capacity_low_watermark_slots = ceil(
            (self._local_dram._total_slots if self._local_dram is not None else 0)
            * self.config.capacity_low_watermark_percent
            / 100
        )
        self._remote_fw_dram = _RemoteFWDram(
            self,
            backend_configs.remote_fw_dram,
            bindings.key_adapter,
        )
        self._g3 = (
            _G3(
                self,
                g3_config,
                self.block_size_bytes,
            )
            if g3_config is not None and local_dram_config is not None
            else None
        )
        framework_dram = backend_configs.framework_dram
        memory_regions: list[tuple[int, int]] = []
        if framework_dram is not None:
            memory_regions.append((framework_dram.address, framework_dram.length))
        if self._local_dram is not None:
            memory_regions.append(self._local_dram.memory_region)
        dram_backends: set[str] = set()
        if self._local_dram is not None:
            dram_backends.add(local_dram_config.backend)
        if self.framework_control is not None:
            dram_backends.add(backend_configs.remote_fw_dram.backend)
        if not dram_backends:
            raise ValueError("at least one DRAM backend must be configured")

        def initialize_progress(progress: _KVCRProgress) -> None:
            self._remote_fw_dram.initialize_progress(progress)
            if self._g3 is not None:
                self._g3.initialize_progress(progress)

        def close_progress() -> None:
            try:
                if self._g3 is not None:
                    self._g3.close_progress()
            finally:
                self._remote_fw_dram.close_progress()

        self._progress = _KVCRProgress(
            initialize_progress,
            self._remote_fw_dram.poll_progress,
            self._remote_fw_dram.flush_progress,
            close_progress,
            nixl_agent_name=self.nixl_agent_name,
            nixl_listen_port=self.config.nixl_listen_port,
            dram_backends=list(dram_backends),
            memory_regions=tuple(memory_regions),
        )

    def start(self) -> None:
        self._progress.start()

    def is_quiescent(self) -> bool:
        """Report whether native progress can still access backend resources."""
        return self._progress.is_quiescent()

    def adopt_recovery_records(self, records: dict[BlockKey, _BlockRecord]) -> None:
        """Seed a core that has not started with recovered residencies.

        Admission runs once per block, and ranking runs after all of them.
        """
        if not records:
            return
        local_dram = self._local_dram
        if local_dram is None:
            raise RecoveryMirrorError("recovered records need a local DRAM tier")
        g3 = self._g3
        if g3 is None and any(record.g3 is not None for record in records.values()):
            raise RecoveryMirrorError("recovered G3 residency has no configured G3")
        if self._block_record_map:
            raise RecoveryMirrorError("recovered records need a core that holds none")

        # Transfer, not copy: rebuilding costs a second population of the set.
        self._block_record_map = records

        local_dram.adopt_recovery_slots(records)
        if g3 is not None:
            g3.adopt_recovery_slots(records)

        # One admission per block: _on_ingest fires only for a single-tier block,
        # so routing through it would skip everything recovered into both.
        for key, record in records.items():
            if record.local_dram is not None or g3 is None:
                source, slot_size = CacheTier.LOCAL_G2, local_dram._slot_size
            else:
                source, slot_size = CacheTier.G3, g3._slot_size
            self._policy.on_ingest(self._block_meta(key, record, slot_size), source)

        # After every admission: a block resident in both tiers must not be
        # ranked in one of them while the policy still has half of its record.
        local_dram.rank_recovered(records)
        if g3 is not None:
            g3.rank_recovered(records)

    # Public API.

    def submit_hint(
        self,
        hints: Mapping[str, object],
        request_id: str | None = None,
    ) -> None:
        src, block_hashes, mode = _parse_kv_hint(hints)
        # TODO: Let policy consume mode="move" and no_retain hints.
        if mode != "copy":
            raise ValueError("only copy mode is currently supported")
        self._remote_fw_dram.submit_hint(src, block_hashes, request_id)

    def discard_hint(self, request_id: str) -> None:
        self._remote_fw_dram.discard_hint(request_id)

    # TODO(kvcr-g3): Prefer REMOTE_G2 over G3 in query/fetch/deliver once one
    # operation can fall back between both sources.
    def query(
        self,
        keys: Collection[BlockKey],
        request_id: str | None = None,
    ) -> list[tuple[QueryStatus, CacheTier | None]]:
        """Return the best available local or remote path for each key.

        Exact hint membership is used unless opportunistic querying is enabled.
        """
        statuses = []
        for key in keys:
            record = self._block_record_map.get(key)
            residency = record.local_dram if record is not None else None
            if residency is not None and residency.state is _LocalDramState.READY:
                statuses.append((QueryStatus.HIT, CacheTier.LOCAL_G2))
            elif residency is not None and residency.state is _LocalDramState.FILLING:
                statuses.append((QueryStatus.FETCHING, CacheTier.LOCAL_G2))
            elif record is not None and record.g3 is not None:
                statuses.append((QueryStatus.FETCHABLE, CacheTier.G3))
            # Remote hints are request-scoped and advisory, not block residency.
            elif request_id is not None and self._remote_fw_dram.query(key, request_id):
                statuses.append((QueryStatus.FETCHABLE, CacheTier.REMOTE_G2))
            else:
                statuses.append((QueryStatus.MISS, None))
        return statuses

    # TODO: Add optional completion callbacks to movement APIs.
    def deliver(
        self,
        blocks: Mapping[BlockKey, list[MemDescriptor]],
        request_id: str | None = None,
    ) -> OpHandle:
        op_handle = self._next_op_handle
        self._next_op_handle += 1
        deadline = self._operation_deadline()
        local_dram = self._local_dram
        normalized = {
            key: self._normalize_descriptors(descriptors)
            for key, descriptors in blocks.items()
        }
        local_blocks: dict[BlockKey, MemDescriptor] = {}
        g3_blocks: dict[BlockKey, MemDescriptor] = {}
        remote_blocks: dict[BlockKey, MemDescriptor] = {}
        for key, destination in normalized.items():
            if self._is_local_resident(key):
                local_blocks[key] = destination
            elif self._g3 is not None and self._g3.is_ready(key):
                g3_blocks[key] = destination
            else:
                remote_blocks[key] = destination

        if sum(map(bool, (local_blocks, g3_blocks, remote_blocks))) > 1:
            self._joined_completions[op_handle] = (set(blocks), {})
        if local_dram is not None and local_blocks:
            local_dram.deliver(op_handle, local_blocks, deadline=deadline)
        if g3_blocks and (
            self._g3 is None
            or not self._g3.start_deliver(op_handle, g3_blocks, deadline)
        ):
            self._complete(
                op_handle,
                {key: OpEntryResult(OpEntryStatus.FAILED) for key in g3_blocks},
            )
        if remote_blocks:
            self._remote_fw_dram.deliver(
                op_handle, remote_blocks, request_id, deadline=deadline
            )
        elif not blocks:
            self._complete(op_handle, {})
        return op_handle

    def deposit(
        self,
        blocks: Mapping[BlockKey, list[MemDescriptor]],
        no_evict: bool = False,
        hints: object | None = None,
    ) -> OpHandle:
        op_handle = self._next_op_handle
        self._next_op_handle += 1
        if self._local_dram is None:
            self._complete(
                op_handle,
                {key: OpEntryResult(OpEntryStatus.FAILED) for key in blocks},
            )
        else:
            self._local_dram.deposit(
                op_handle,
                {
                    key: self._normalize_descriptors(descriptors)
                    for key, descriptors in blocks.items()
                },
                no_evict=no_evict,
                hints=hints,
            )
        return op_handle

    def fetch(
        self,
        keys: Collection[BlockKey],
        request_id: str | None = None,
        expected_layout: list[str] | None = None,
        hints: object | None = None,
    ) -> OpHandle:
        expected_layout = [""] if expected_layout is None else expected_layout
        if expected_layout != [self.pool_layout[0][1]]:
            raise ValueError("expected layout must match the configured pool_layout")
        op_handle = self._next_op_handle
        self._next_op_handle += 1
        local_dram = self._local_dram
        ordered_keys = tuple(dict.fromkeys(keys))
        if local_dram is None:
            self._complete(
                op_handle,
                {key: OpEntryResult(OpEntryStatus.FAILED) for key in ordered_keys},
            )
            return op_handle

        sources = {}
        for key in ordered_keys:
            if self._is_local_resident(key):
                continue
            if self._g3 is not None and self._g3.is_ready(key):
                sources[key] = CacheTier.G3
            elif request_id is not None and self._remote_fw_dram.query(key, request_id):
                sources[key] = CacheTier.REMOTE_G2
        deadline = self._operation_deadline()
        destinations = local_dram.fetch(
            op_handle,
            ordered_keys,
            sources,
            request_id,
            deadline,
            hints=hints,
        )
        for source in (CacheTier.G3, CacheTier.REMOTE_G2):
            self._start_local_fill(
                source,
                {
                    key: destination
                    for key, destination in destinations.items()
                    if sources[key] is source
                },
                request_id,
                deadline,
            )
        return op_handle

    def release(self, handles: Collection[ReleaseHandle]) -> list[ReleaseResult]:
        if self._local_dram is None:
            return [(handle, False) for handle in handles]
        return self._local_dram.release(handles)

    # TODO: Expose individual entry completions as they become available.
    def poll_completed(self) -> Iterable[OpResult]:
        progress_items = self._progress.take_completed()
        if self._g3 is not None:
            progress_items = self._g3.poll_main(progress_items)
        if self._local_dram is not None:
            progress_items = self._local_dram.poll_main(progress_items)
        self._remote_fw_dram.poll_main(progress_items)
        self._flush_inventory()
        completed = self._completion_queue
        self._completion_queue = []
        return completed

    def abort(
        self,
        op_handle: OpHandle,
        keys: Collection[BlockKey] | None = None,
    ) -> bool:
        # TODO: Implement best-effort cancellation for fetch and deliver entries.
        return False

    def get_stats(self) -> TelemetryStats | None:
        self._progress.raise_if_failed()
        stats = self._stats
        if stats is None:
            return None
        resources = {
            "block_records": len(self._block_record_map),
            "in_flight_ops": self._outstanding_operations,
            "framework_pins": len(self._framework_pin_keys),
            "pinned_keys": sum(map(len, self._framework_pin_keys.values())),
            "connected_remotes": self._remote_fw_dram._connected_remote_count,
            "completed": len(self._completion_queue),
        }
        if self._local_dram is not None:
            resources.update(self._local_dram.telemetry_state())
        if self._g3 is not None:
            resources.update(self._g3.telemetry_state())
        for resource, value in resources.items():
            stats.set_gauge(STATE_METRIC, value, (resource,))
        self._stats = self._stats_factory() if self._stats_factory else None
        return stats

    def close(self) -> None:
        if self._closed:
            return
        # Assumption: the framework drains submitted jobs before close.
        self._flush_inventory(force=True)
        progress_error: BaseException | None = None
        try:
            self._progress.close()
        except BaseException as error:  # noqa: BLE001 - re-raised below
            progress_error = error

        # Cleanup mutates backend state native operations still reference, and
        # a stopped thread does not prove they are done with it. The caller
        # unmaps the pool as soon as close() returns, so anything short of a
        # quiescent loop must leave the resources alone and raise.
        if not self.is_quiescent():
            logger.error(
                "KVCR progress loop is not quiescent after close; leaving "
                "backend resources in place to avoid racing native transfers, "
                "operations, or registrations"
            )
            raise progress_error or RuntimeError(
                "KVCR progress loop is not quiescent after close"
            )

        self._closed = True
        try:
            self._close_main_resources()
        except BaseException as error:
            # The progress failure came first and explains this one.
            raise error from progress_error
        if progress_error is not None:
            raise progress_error

    def _close_main_resources(self) -> None:
        # Every stage runs even if an earlier one failed: skipping the rest
        # would leak files and pins for a failure that says nothing about them.
        first_error: BaseException | None = None

        def run(stage: Callable[[], None]) -> None:
            nonlocal first_error
            try:
                stage()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                else:
                    logger.warning(
                        "KVCR cleanup stage failed after an earlier failure",
                        exc_info=True,
                    )

        if self._g3 is not None:
            run(self._g3.close_main)
        if self._local_dram is not None:
            run(self._local_dram.close)
        run(self._remote_fw_dram.close_main)
        for op_id in tuple(self._local_dram_sources_by_op):
            run(functools.partial(self._release_local_dram_sources, op_id))
        if first_error is not None:
            raise first_error

    def _publish_inventory(
        self,
        keys: Collection[BlockKey],
        tier: CacheTier,
        *,
        removed: bool,
    ) -> bool:
        if not keys:
            return True
        event = InventoryEvent(tuple(keys), tier, removed)
        if self._inventory_report_interval == 0:
            return self._send_inventory(event)
        if self._inventory_sink_callback is None:
            return False
        self._pending_inventory_events.append(event)
        if self._inventory_flush_deadline is None:
            self._inventory_flush_deadline = (
                self._clock() + self._inventory_report_interval
            )
        return True

    def _send_inventory(self, event: InventoryEvent) -> bool:
        callback = self._inventory_sink_callback
        if callback is None:
            return False
        try:
            callback(event)
        except Exception:
            logger.warning("KVCR inventory sink failed", exc_info=True)
            return False
        return True

    def _flush_inventory(self, *, force: bool = False) -> None:
        deadline = self._inventory_flush_deadline
        if deadline is None or (not force and self._clock() < deadline):
            return
        events = self._pending_inventory_events
        self._pending_inventory_events = []
        self._inventory_flush_deadline = None
        for event in events:
            self._send_inventory(event)

    def _update_capacity_pressure(self, reclaimable_slots: int) -> None:
        callback = self._capacity_needed_callback
        if callback is None or self._capacity_low_watermark_slots == 0:
            return
        if reclaimable_slots >= self._capacity_low_watermark_slots:
            self._capacity_pressure_active = False
            return
        if self._capacity_pressure_active:
            return
        self._capacity_pressure_active = True
        try:
            callback(self._capacity_low_watermark_slots)
        except Exception:
            logger.warning("KVCR capacity callback failed", exc_info=True)

    # Generic operation/block dependency bookkeeping.

    def _add_block_dependencies(self, op: _Op, *, new_operation: bool) -> None:
        if new_operation:
            self._outstanding_operations += 1
        for key in op.keys:
            self._block_record(key).add_in_flight_op(op.op_id)

    def _remove_block_dependencies(self, op: _Op) -> None:
        self._outstanding_operations -= 1
        for key in op.keys:
            record = self._block_record_map.get(key)
            if record is not None:
                record.discard_in_flight_op(op.op_id)
                self._prune_block_record(key)

    # Cross-backend DRAM coordination.

    def _start_local_fill(
        self,
        source: CacheTier,
        blocks: Mapping[BlockKey, MemDescriptor],
        request_id: str | None,
        deadline: float,
    ) -> None:
        if not blocks:
            return
        fill_handle = self._next_fill_handle
        self._next_fill_handle -= 1
        if source is CacheTier.G3:
            started = self._g3 is not None and self._g3.start_fill(
                fill_handle, dict(blocks), deadline
            )
        elif source is CacheTier.REMOTE_G2:
            started = self._remote_fw_dram.fetch(
                blocks, request_id, deadline, op_handle=fill_handle
            )
        else:
            raise ValueError(f"unsupported local fill source {source.value}")
        if not started:
            self._complete_local_dram_fill(blocks, success=False)

    def _claim_local_dram_sources(
        self, op_id: _OpId, keys: Collection[BlockKey]
    ) -> Mapping[BlockKey, MemDescriptor]:
        sources = self._local_dram_sources_by_op.get(op_id, {})
        if self._local_dram is not None:
            claimed = self._local_dram.acquire_sources(
                tuple(key for key in keys if key not in sources)
            )
            if claimed:
                sources.update(claimed)
                self._local_dram_sources_by_op[op_id] = sources
        return sources

    def _normalize_descriptors(self, descriptors: list[MemDescriptor]) -> MemDescriptor:
        if not isinstance(descriptors, list):
            raise TypeError("block descriptors must be a list")
        if len(descriptors) != 1 or not isinstance(descriptors[0], MemDescriptor):
            raise ValueError("each block requires exactly one descriptor")
        descriptor = descriptors[0]
        if descriptor.info != self.pool_layout[0][1]:
            raise ValueError(f"unknown descriptor pool {descriptor.info!r}")
        if descriptor.size != self.block_size_bytes:
            raise ValueError("block descriptor has the wrong byte count")
        return descriptor

    def _release_local_dram_sources(
        self,
        op_id: _OpId,
        keys: Collection[BlockKey] | None = None,
    ) -> None:
        sources = self._local_dram_sources_by_op.get(op_id)
        if not sources:
            return
        released = set(sources) if keys is None else set(keys) & sources.keys()
        if self._local_dram is None:
            raise RuntimeError("controller source exists without local DRAM")
        self._local_dram.release_sources(released)
        for key in released:
            sources.pop(key)
        if not sources:
            self._local_dram_sources_by_op.pop(op_id)

    def _complete_local_dram_fill(
        self, keys: Collection[BlockKey], *, success: bool
    ) -> None:
        if self._local_dram is None:
            raise RuntimeError("remote fetch completed without local DRAM")
        self._local_dram.complete_fill(keys, success=success)

    def _discard_local_dram_fill(self, keys: Collection[BlockKey]) -> None:
        if self._local_dram is None:
            raise RuntimeError("remote fetch update without local DRAM")
        self._local_dram.discard_fill(keys)

    def _block_record(self, key: BlockKey) -> _BlockRecord:
        return self._block_record_map.setdefault(key, _BlockRecord())

    def _is_local_resident(self, key: BlockKey) -> bool:
        """Report local DRAM residency a new operation can still be served from.

        A discarded fill still owns its slot, but its contents are gone, so the
        block has to be sourced again from a lower tier.
        """
        record = self._block_record_map.get(key)
        return (
            record is not None
            and record.local_dram is not None
            and record.local_dram.state is not _LocalDramState.DISCARDING
        )

    def _decide_eviction(
        self, meta: BlockMeta, source: CacheTier, deadline: float
    ) -> tuple[PlacementDecision, bool]:
        """Return the resolved decision and whether capacity is pending."""
        decision = self._policy.decide_eviction(meta, source)
        if self._g3 is not None:
            return self._g3.resolve_eviction(meta, source, decision, deadline)
        return decision, False

    def _on_ingest(self, meta: BlockMeta, source: CacheTier) -> None:
        managed = meta.resident_tiers & {CacheTier.LOCAL_G2, CacheTier.G3}
        if len(managed) == 1:
            self._policy.on_ingest(meta, source)

    def _on_remove(self, meta: BlockMeta) -> None:
        if not meta.resident_tiers & {CacheTier.LOCAL_G2, CacheTier.G3}:
            self._policy.on_remove(meta)

    def _block_meta(
        self,
        key: BlockKey,
        record: _BlockRecord,
        size_bytes: int,
    ) -> BlockMeta:
        resident_tiers = set()
        if record.fw_mem is not None:
            resident_tiers.add(CacheTier.FW_G2)
        if record.local_dram is not None:
            resident_tiers.add(CacheTier.LOCAL_G2)
        if record.g3 is not None:
            resident_tiers.add(CacheTier.G3)
        return BlockMeta(
            block_key=key,
            size_bytes=size_bytes,
            access_count=record.access_count,
            last_access=record.last_access,
            resident_tiers=frozenset(resident_tiers),
        )

    def _record_access(self, keys: Collection[BlockKey]) -> None:
        now = self._clock()
        for key in keys:
            record = self._block_record_map[key]
            record.access_count += 1
            record.last_access = now

    def _prune_block_record(self, key: BlockKey) -> None:
        record = self._block_record_map.get(key)
        if (
            record is not None
            and record.fw_mem is None
            and record.local_dram is None
            and record.g3 is None
            and not record.in_flight_ops
        ):
            self._block_record_map.pop(key)

    def _complete(
        self,
        op_handle: OpHandle,
        entries: Mapping[BlockKey, OpEntryResult],
    ) -> None:
        joined = self._joined_completions.get(op_handle)
        if joined is not None:
            expected, completed = joined
            completed.update(entries)
            if not expected.issubset(completed):
                return
            self._joined_completions.pop(op_handle)
            entries = completed
        self._completion_queue.append((op_handle, entries))

    # Timing and telemetry helpers.

    def _operation_deadline(self) -> float:
        return self._clock() + self.config.operation_timeout_ms / 1000

    def _record_duration_enabled(
        self, scope: str, started_at: float | None, result: str
    ) -> None:
        if self._stats is not None and started_at is not None:
            self._stats.observe_histogram(
                DURATION_METRIC,
                time.monotonic() - started_at,
                (scope, result),
            )

    def _record_transfer_enabled(
        self,
        scope: str,
        started_at: float | None,
        success: bool,
        block_count: int,
        byte_count: int,
    ) -> None:
        self._record_duration_enabled(
            scope, started_at, "success" if success else "failed"
        )
        stats = self._stats
        if success and stats is not None:
            stats.increase_counter(TRANSFER_BLOCKS_METRIC, block_count, (scope,))
            stats.increase_counter(TRANSFER_BYTES_METRIC, byte_count, (scope,))
