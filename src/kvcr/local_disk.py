# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""File-backed G3 residency and NIXL transfers."""

import fcntl
import logging
import os
from collections import deque
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .config import G3Options
from .policy_runtime import _EvictionQueue
from .progress import _KVCRProgress, _OpId, _ProgressOp
from .types import (
    BlockKey,
    BlockMeta,
    CacheTier,
    MemDescriptor,
    OpEntryResult,
    OpEntryStatus,
    OpHandle,
    PlacementAction,
    PlacementDecision,
    PlacementFailure,
)

if TYPE_CHECKING:
    from .core import _KVCRCore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _G3Residency:
    slot: int
    claim_count: int = 0


@dataclass(frozen=True)
class _Reservation:
    key: BlockKey
    slot: int


@dataclass(frozen=True)
class _Claim:
    key: BlockKey
    residency: _G3Residency


@dataclass
class _G3TransferOp(_ProgressOp):
    kind: Literal["store", "fill", "deliver"]
    ordered_keys: tuple[BlockKey, ...]
    memory_descriptors: tuple[MemDescriptor, ...]
    file_descriptors: tuple[MemDescriptor, ...]
    backend: str
    deadline: float
    clock: Any = field(repr=False, compare=False)
    started_at: float | None = field(repr=False, compare=False)
    reservations: tuple[_Reservation, ...] = ()
    claims: tuple[_Claim, ...] = ()
    transfer_id: int | None = None
    cancellation_requested: bool = False
    success: bool = False

    def progress(
        self, progress: _KVCRProgress, event: object | None
    ) -> tuple[bool, bool]:
        if event is not None:
            raise RuntimeError(f"unexpected G3 event: {event!r}")
        observed_work = False
        if self.transfer_id is None:
            if self.clock() >= self.deadline:
                return True, True
            try:
                transfer_id, submitted = progress.submit_transfer(
                    "WRITE" if self.kind == "store" else "READ",
                    self.memory_descriptors,
                    self.file_descriptors,
                    remote_side_agent=progress.nixl_agent_name,
                    backend=self.backend,
                )
            except Exception:
                logger.warning("KVCR G3 transfer submission failed", exc_info=True)
                return True, True
            self.transfer_id = transfer_id
            self.cancellation_requested = not submitted
            observed_work = True

        transfer_id = self.transfer_id
        if transfer_id is None:
            raise RuntimeError(f"G3 operation {self.op_id!r} lost its transfer")
        if not self.cancellation_requested and self.clock() >= self.deadline:
            self.cancellation_requested = True
            observed_work = True
        result = progress.poll_transfer(
            transfer_id,
            cancellation_requested=self.cancellation_requested,
        )
        if result is None:
            return False, observed_work
        self.transfer_id = None
        self.success, _ = result
        return True, True

    def close(self, progress: _KVCRProgress) -> bool:
        if self.transfer_id is not None:
            if not progress.cancel_transfer(self.transfer_id):
                return False
            self.transfer_id = None
        self.success = False
        return True


class _G3:
    """Own bounded files and the metadata needed to use them as G3 cache."""

    def __init__(self, kvcr: "_KVCRCore", config: G3Options, slot_size: int) -> None:
        page_size = os.sysconf("SC_PAGE_SIZE")
        paths = tuple(Path(path).expanduser().resolve() for path in config.paths)
        if not paths:
            raise ValueError("G3 requires at least one file path")
        if len(paths) != len(set(paths)):
            raise ValueError("G3 file paths must be unique")
        if slot_size <= 0 or slot_size % page_size:
            raise ValueError("G3 slot size must be positive and page aligned")
        if (
            config.capacity_bytes_per_file <= 0
            or config.capacity_bytes_per_file % slot_size
        ):
            raise ValueError("G3 file capacity must contain complete slots")
        if not config.backend:
            raise ValueError("G3 NIXL backend must be non-empty")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in config.backend_options.items()
        ):
            raise TypeError("G3 backend options must map strings to strings")

        slots_per_file = config.capacity_bytes_per_file // slot_size
        self._kvcr = kvcr
        self._config = config
        self._paths = paths
        self._slot_size = slot_size
        self._bytes_per_file = config.capacity_bytes_per_file
        self._owner_fds: list[int] = []
        self._direct_fds: list[int] = []
        self._free_slots = deque(range(slots_per_file * len(paths)))
        self._evictable = _EvictionQueue()
        self._unscored: set[BlockKey] = set()
        self._pending_keys: set[BlockKey] = set()
        self._active: dict[_OpId, _G3TransferOp] = {}
        self._file_registration: Any | None = None
        self._nixl_agent: Any | None = None
        try:
            self._open_files()
        except Exception:
            self._close_files()
            raise

    def initialize_progress(self, progress: _KVCRProgress) -> None:
        agent = progress.nixl_agent
        self._nixl_agent = agent
        backend = self._config.backend
        if backend not in agent.get_plugin_list():
            raise RuntimeError(f"NIXL {backend} plugin is not available")
        configured = dict(self._config.backend_options)
        if backend not in getattr(agent, "backends", {}):
            agent.create_backend(backend, configured)
        elif configured and any(
            agent.get_backend_params(backend).get(key) != value
            for key, value in configured.items()
        ):
            raise ValueError(f"incompatible options for NIXL backend {backend}")
        self._file_registration = agent.register_memory(
            [(0, self._bytes_per_file, fd, "") for fd in self._direct_fds],
            "FILE",
            backends=[backend],
        )
        if self._file_registration is None:
            raise RuntimeError("failed to register G3 files")

    def close_progress(self) -> None:
        try:
            registration = self._file_registration
            if registration is not None:
                agent = self._nixl_agent
                if agent is None:
                    raise RuntimeError("G3 NIXL agent is not initialized")
                if (
                    agent.deregister_memory(
                        registration, backends=[self._config.backend]
                    )
                    is False
                ):
                    raise RuntimeError("G3 registration did not close")
                self._file_registration = None
        finally:
            self._nixl_agent = None

    def close_main(self) -> None:
        local = self._kvcr._local_dram
        for op in tuple(self._active.values()):
            if op.kind == "store":
                self._rollback(op.reservations)
                if local is not None:
                    for key in op.ordered_keys:
                        local.abandon_capacity_eviction(key)
                self._kvcr._release_local_dram_sources(op.op_id)
            else:
                self._release_claims(op.claims)
            self._kvcr._remove_block_dependencies(op)
        self._active.clear()
        self._close_files()

    def is_ready(self, key: BlockKey) -> bool:
        record = self._kvcr._block_record_map.get(key)
        return record is not None and record.g3 is not None

    def telemetry_state(self) -> dict[str, int]:
        total_slots = len(self._paths) * self._bytes_per_file // self._slot_size
        return {
            "g3_total_slots": total_slots,
            "g3_free_slots": len(self._free_slots),
            "g3_allocated_slots": total_slots - len(self._free_slots),
            "g3_evictable_slots": len(self._evictable),
            "g3_pending_stores": len(self._pending_keys),
        }

    def resolve_eviction(
        self,
        meta: BlockMeta,
        source: CacheTier,
        decision: PlacementDecision,
        deadline: float,
    ) -> tuple[PlacementDecision, bool]:
        if source is not CacheTier.LOCAL_G2 or decision != (
            PlacementAction.MOVE_TO,
            CacheTier.G3,
        ):
            return decision, False
        key = meta.block_key
        record = self._kvcr._block_record_map.get(key)
        if record is None or record.local_dram is None:
            return (PlacementAction.KEEP, None), False
        if record.g3 is not None:
            return (PlacementAction.DROP, None), False
        if key in self._pending_keys:
            return (PlacementAction.KEEP, None), False

        op_id = ("g3_store", key)
        sources = self._kvcr._claim_local_dram_sources(op_id, (key,))
        if key not in sources:
            return (PlacementAction.KEEP, None), False
        try:
            if not self._start_store(op_id, sources, deadline):
                self._recover_store_failure(key, "G3 destination unavailable")
                return (PlacementAction.KEEP, None), False
        except Exception:
            self._recover_store_failure(key, "store failed to start")
            return (PlacementAction.KEEP, None), False
        return (PlacementAction.KEEP, None), True

    def start_fill(
        self,
        op_handle: OpHandle,
        blocks: dict[BlockKey, MemDescriptor],
        deadline: float,
    ) -> bool:
        return self._start_read("fill", op_handle, blocks, deadline)

    def start_deliver(
        self,
        op_handle: OpHandle,
        blocks: dict[BlockKey, MemDescriptor],
        deadline: float,
    ) -> bool:
        return self._start_read("deliver", op_handle, blocks, deadline)

    def poll_main(self, items: Collection[object]) -> list[object]:
        unhandled: list[object] = []
        for item in items:
            if not isinstance(item, _G3TransferOp):
                unhandled.append(item)
                continue
            if self._active.pop(item.op_id, None) is not item:
                raise RuntimeError(f"unknown completed G3 operation {item.op_id!r}")
            try:
                self._finish(item)
            finally:
                self._kvcr._remove_block_dependencies(item)
        return unhandled

    def _start_store(
        self,
        op_id: _OpId,
        blocks: dict[BlockKey, MemDescriptor],
        deadline: float,
    ) -> bool:
        reservations = self._reserve(tuple(blocks), set(blocks))
        if len(reservations) != len(blocks):
            self._rollback(reservations)
            return False
        op = _G3TransferOp(
            op_id=op_id,
            keys=set(blocks),
            kind="store",
            ordered_keys=tuple(item.key for item in reservations),
            memory_descriptors=tuple(blocks[item.key] for item in reservations),
            file_descriptors=tuple(
                self._descriptor(item.slot) for item in reservations
            ),
            backend=self._config.backend,
            deadline=deadline,
            clock=self._kvcr._clock,
            started_at=self._kvcr._timer(),
            reservations=reservations,
        )
        self._submit(op)
        return True

    def _start_read(
        self,
        kind: Literal["fill", "deliver"],
        op_handle: OpHandle,
        blocks: dict[BlockKey, MemDescriptor],
        deadline: float,
    ) -> bool:
        claims, rejected = self._claim(tuple(blocks))
        # A G3 read moves one whole slot, so every destination must be
        # slot-sized. NIXL requires one memory type per destination descriptor
        # list; mixed destination types would require separate transfers, which
        # this path does not yet split. Fail the whole G3 batch before submit.
        destinations = tuple(blocks.values())
        mem_type = destinations[0].mem_type if destinations else ""
        rejected += tuple(
            key
            for key, destination in blocks.items()
            if destination.size != self._slot_size or destination.mem_type != mem_type
        )
        if rejected or not claims:
            self._release_claims(claims)
            return False
        op = _G3TransferOp(
            op_id=(f"g3_{kind}", op_handle),
            keys=set(blocks),
            kind=kind,
            ordered_keys=tuple(item.key for item in claims),
            memory_descriptors=tuple(blocks[item.key] for item in claims),
            file_descriptors=tuple(
                self._descriptor(item.residency.slot) for item in claims
            ),
            backend=self._config.backend,
            deadline=deadline,
            clock=self._kvcr._clock,
            started_at=self._kvcr._timer(),
            claims=claims,
        )
        self._submit(op)
        return True

    def _submit(self, op: _G3TransferOp) -> None:
        if op.op_id in self._active:
            raise RuntimeError(f"duplicate G3 operation {op.op_id!r}")
        self._active[op.op_id] = op
        self._kvcr._add_block_dependencies(op, new_operation=True)
        try:
            self._kvcr._progress.submit(op)
        except Exception:
            self._active.pop(op.op_id)
            self._kvcr._remove_block_dependencies(op)
            if op.kind == "store":
                self._rollback(op.reservations)
            else:
                self._release_claims(op.claims)
            raise

    def _finish(self, op: _G3TransferOp) -> None:
        if op.kind == "store":
            self._record_transfer(op)
            if op.success:
                for reservation in op.reservations:
                    self._commit(reservation)
                self._retire_local_sources(op.op_id)
            else:
                self._rollback(op.reservations)
                (key,) = op.ordered_keys
                self._recover_store_failure(key, "transfer failed")
            return

        if op.kind == "fill":
            local = self._kvcr._local_dram
            if local is None:
                raise RuntimeError("G3 fill completed without local DRAM")
            success = op.success and all(
                (record := self._kvcr._block_record_map.get(key)) is not None
                and record.local_dram is not None
                and record.local_dram.state.name == "FILLING"
                for key in op.ordered_keys
            )
            self._record_transfer(op, success)
            try:
                local.complete_fill(op.ordered_keys, success=success)
            finally:
                self._release_claims(op.claims)
            return

        self._record_transfer(op)
        try:
            if op.success:
                self._kvcr._record_access(op.ordered_keys)
        finally:
            self._release_claims(op.claims)
        self._kvcr._complete(
            op.op_id[1],
            {
                key: OpEntryResult(
                    OpEntryStatus.SUCCESS if op.success else OpEntryStatus.FAILED
                )
                for key in op.ordered_keys
            },
        )

    def _record_transfer(self, op: _G3TransferOp, success: bool | None = None) -> None:
        self._kvcr._record_transfer(
            f"g3_{op.kind}",
            op.started_at,
            op.success if success is None else success,
            len(op.ordered_keys),
            len(op.ordered_keys) * self._slot_size,
        )

    def _recover_store_failure(
        self,
        key: BlockKey,
        reason: str,
    ) -> None:
        record = self._kvcr._block_record_map[key]
        self._kvcr._policy.decide_recovery(
            self._kvcr._block_meta(key, record, self._slot_size),
            PlacementFailure(
                attempted=(PlacementAction.MOVE_TO, CacheTier.G3),
                source=CacheTier.LOCAL_G2,
                reason=reason,
                failure_count=1,
            ),
        )
        self._retire_local_sources(("g3_store", key))

    def _retire_local_sources(self, op_id: _OpId) -> None:
        sources = self._kvcr._local_dram_sources_by_op.get(op_id)
        if not sources:
            return
        local = self._kvcr._local_dram
        if local is None:
            raise RuntimeError("G3 move completed without local DRAM")
        local.retire_sources(sources)
        self._kvcr._release_local_dram_sources(op_id)

    def _reserve(
        self, keys: tuple[BlockKey, ...], protected: set[BlockKey]
    ) -> tuple[_Reservation, ...]:
        reservations: list[_Reservation] = []
        for key in keys:
            record = self._kvcr._block_record_map.get(key)
            if record is None or record.g3 is not None or key in self._pending_keys:
                continue
            slot = self._allocate_slot(protected)
            if slot is None:
                continue
            self._pending_keys.add(key)
            reservations.append(_Reservation(key, slot))
        return tuple(reservations)

    def _allocate_slot(self, protected: set[BlockKey]) -> int | None:
        if self._free_slots:
            return self._free_slots.popleft()
        self._retry_unscored()
        skipped = set(protected)
        while (key := self._evictable.select(skipped)) is not None:
            record = self._kvcr._block_record_map.get(key)
            residency = record.g3 if record is not None else None
            if record is None or residency is None or residency.claim_count:
                raise RuntimeError(f"invalid G3 eviction candidate {key!r}")
            if any(op_id[0] == "fetch" for op_id in record.in_flight_ops or ()):
                skipped.add(key)
                continue
            decision = self._kvcr._policy.decide_eviction(
                self._kvcr._block_meta(key, record, self._slot_size),
                CacheTier.G3,
            )
            if decision[0] is PlacementAction.KEEP:
                skipped.add(key)
                continue
            self._remove_evictable(key)
            record.g3 = None
            self._kvcr._on_remove(self._kvcr._block_meta(key, record, self._slot_size))
            self._kvcr._publish_inventory((key,), CacheTier.G3, removed=True)
            self._kvcr._prune_block_record(key)
            return residency.slot
        return None

    def _commit(self, reservation: _Reservation) -> None:
        self._pending_keys.remove(reservation.key)
        record = self._kvcr._block_record(reservation.key)
        if record.g3 is not None:
            raise RuntimeError("G3 destination became resident before commit")
        record.g3 = _G3Residency(reservation.slot)
        self._make_evictable(reservation.key)
        self._kvcr._publish_inventory((reservation.key,), CacheTier.G3, removed=False)

    def _rollback(self, reservations: Collection[_Reservation]) -> None:
        for reservation in reservations:
            if reservation.key in self._pending_keys:
                self._pending_keys.remove(reservation.key)
                self._free_slots.append(reservation.slot)

    def _claim(
        self, keys: tuple[BlockKey, ...]
    ) -> tuple[tuple[_Claim, ...], tuple[BlockKey, ...]]:
        claims: list[_Claim] = []
        rejected: list[BlockKey] = []
        for key in keys:
            record = self._kvcr._block_record_map.get(key)
            residency = record.g3 if record is not None else None
            if residency is None:
                rejected.append(key)
                continue
            self._remove_evictable(key)
            residency.claim_count += 1
            claims.append(_Claim(key, residency))
        return tuple(claims), tuple(rejected)

    def _release_claims(self, claims: Collection[_Claim]) -> None:
        for claim in claims:
            record = self._kvcr._block_record_map.get(claim.key)
            if record is None or record.g3 is not claim.residency:
                raise RuntimeError(f"G3 claim lost residency for {claim.key!r}")
            if claim.residency.claim_count <= 0:
                raise RuntimeError(f"invalid G3 claim count for {claim.key!r}")
            claim.residency.claim_count -= 1
            if claim.residency.claim_count == 0:
                self._make_evictable(claim.key)

    def _make_evictable(self, key: BlockKey) -> None:
        record = self._kvcr._block_record_map.get(key)
        residency = record.g3 if record is not None else None
        if record is None or residency is None or residency.claim_count:
            return
        score = self._kvcr._policy.eviction_score(
            self._kvcr._block_meta(key, record, self._slot_size), CacheTier.G3
        )
        if score is None:
            self._unscored.add(key)
            return
        self._unscored.discard(key)
        self._evictable.insert(key, score)

    def _remove_evictable(self, key: BlockKey) -> None:
        self._unscored.discard(key)
        self._evictable.remove(key)

    def _retry_unscored(self) -> None:
        for key in tuple(self._unscored):
            self._make_evictable(key)

    def _descriptor(self, slot: int) -> MemDescriptor:
        file_index = slot % len(self._direct_fds)
        file_slot = slot // len(self._direct_fds)
        return MemDescriptor(
            self._kvcr.nixl_agent_name,
            "FILE",
            file_slot * self._slot_size,
            self._slot_size,
            self._direct_fds[file_index],
            "",
        )

    def _open_files(self) -> None:
        direct_flag = getattr(os, "O_DIRECT", 0)
        if not direct_flag:
            raise RuntimeError("G3 file storage requires O_DIRECT")
        for path in self._paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            owner = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
            try:
                if any(os.path.sameopenfile(owner, fd) for fd in self._owner_fds):
                    raise ValueError("G3 file paths must not alias the same file")
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.fchmod(owner, 0o600)
                size = os.fstat(owner).st_size
                if size == 0:
                    os.posix_fallocate(owner, 0, self._bytes_per_file)
                elif size != self._bytes_per_file:
                    raise ValueError(f"G3 file {path} has incompatible size")
                direct = os.open(path, os.O_RDWR | os.O_CLOEXEC | direct_flag)
                if not os.path.sameopenfile(owner, direct):
                    os.close(direct)
                    raise RuntimeError(f"G3 file path changed while opening: {path}")
            except Exception:
                os.close(owner)
                raise
            self._owner_fds.append(owner)
            self._direct_fds.append(direct)

    def _close_files(self) -> None:
        errors: list[OSError] = []
        for fd in (*self._direct_fds, *self._owner_fds):
            try:
                os.close(fd)
            except OSError as error:
                errors.append(error)
        self._direct_fds.clear()
        self._owner_fds.clear()
        if errors:
            raise RuntimeError(f"failed to close {len(errors)} G3 file resources")
