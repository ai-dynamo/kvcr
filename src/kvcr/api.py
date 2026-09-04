# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Public northbound API for the KV Cache Runner."""

import contextlib
import logging
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import recovery_journal as _recovery
from .config import (
    FrameworkControl,
    InventorySink,
    KeyAdapter,
    KVCRBackendConfigs,
    KVCRConfig,
    KVCRGuardConfig,
    TelemetryStats,
)
from .core import (  # noqa: F401 - public re-exports
    DURATION_METRIC,
    STATE_METRIC,
    TRANSFER_BLOCKS_METRIC,
    TRANSFER_BYTES_METRIC,
    _KVCRCore,
)
from .guard_protocol import KVCRPoolHold
from .types import (
    BlockKey,
    CacheTier,
    MemDescriptor,
    OpHandle,
    OpResult,
    PinHandle,
    PinRequestId,
    PinResult,
    QueryStatus,
    ReleaseHandle,
    ReleaseResult,
)

if TYPE_CHECKING:
    from .policy import KVCachePolicy


logger = logging.getLogger(__name__)


# Startup failure is terminal for the process. Retain native resources until exit
# because they may still own backends or access a service-owned mapping.
_NONQUIESCENT_STARTUP_RESOURCES: list[tuple[_KVCRCore, KVCRPoolHold | None]] = []


@dataclass(frozen=True)
class KVCRBindings:
    """Framework services and integration callbacks used by KVCR."""

    # Framework-owned source pinning.
    request_pin: Callable[[Collection[BlockKey]], PinRequestId]
    poll_pin_results: Callable[[], Iterable[tuple[PinRequestId, PinResult]]]
    release_pin: Callable[[PinHandle], bool]
    cancel_pin_request: Callable[[PinRequestId], None] | None = None

    # Control, key translation, and inventory reporting.
    framework_control: FrameworkControl | None = None
    key_adapter: KeyAdapter | None = None
    inventory_sink: InventorySink | None = None

    # Capacity pressure, telemetry, and placement policy.
    capacity_needed_callback: Callable[[int], None] | None = None
    stats_factory: Callable[[], TelemetryStats] | None = None
    policy: "KVCachePolicy | None" = None


class KVCR:
    """Framework-facing KV Cache Runner."""

    # TODO: Accept hot-start metadata and a side-process heartbeat/handoff binding.
    def __init__(
        self,
        config: KVCRConfig,
        bindings: KVCRBindings,
        backend_configs: KVCRBackendConfigs,
        guard_config: KVCRGuardConfig | None = None,
    ) -> None:
        claimed: _recovery.ClaimedPool | None = None
        pool_hold: KVCRPoolHold | None = None
        core: _KVCRCore | None = None
        try:
            if guard_config is None:
                core = _KVCRCore(config, bindings, backend_configs)
            else:
                claimed = _recovery.claim_guarded_pool(
                    config, guard_config, bindings, backend_configs
                )
                pool_hold = claimed.hold
                core = _recovery.claimed_core(
                    config, bindings, backend_configs, claimed
                )
                _recovery.adopt_claimed_pool(core, claimed)
            core.start()
            _recovery.commit_claimed_pool(claimed)
        except BaseException:
            close_failed = False
            if core is not None:
                try:
                    core.close()
                except BaseException:
                    close_failed = True
            if core is not None and (close_failed or not core.is_quiescent()):
                # A close that raised may have kept the adopted listener even
                # though NIXL went quiescent; a Guard resumed beside it would
                # split the endpoint. Retained until this process dies, when
                # the pidfd frees the pool.
                _NONQUIESCENT_STARTUP_RESOURCES.append((core, pool_hold))
            elif pool_hold is not None:
                with contextlib.suppress(BaseException):
                    pool_hold.release(activated=False)
            raise
        self._core = core
        self._pool_hold = pool_hold

    @property
    def config(self) -> KVCRConfig:
        return self._core.config

    def submit_hint(
        self,
        hints: Mapping[str, object],
        request_id: str | None = None,
    ) -> None:
        """Submit a hint conforming to the hint protocol."""
        self._core.submit_hint(hints, request_id)

    def discard_hint(self, request_id: str) -> None:
        """Discard request-scoped router hints."""
        self._core.discard_hint(request_id)

    def query(
        self,
        keys: Collection[BlockKey],
        request_id: str | None = None,
    ) -> list[tuple[QueryStatus, CacheTier | None]]:
        """Return the best currently known status and tier for each key."""
        return self._core.query(keys, request_id)

    def deliver(
        self,
        blocks: Mapping[BlockKey, list[MemDescriptor]],
        request_id: str | None = None,
    ) -> OpHandle:
        """Asynchronously deliver blocks to caller-provided destinations."""
        return self._core.deliver(blocks, request_id)

    def deposit(
        self,
        blocks: Mapping[BlockKey, list[MemDescriptor]],
        no_evict: bool = False,
        hints: object | None = None,
    ) -> OpHandle:
        """Asynchronously copy blocks into KVCR-managed storage."""
        return self._core.deposit(blocks, no_evict, hints)

    def fetch(
        self,
        keys: Collection[BlockKey],
        request_id: str | None = None,
        hints: object | None = None,
        expected_layouts: Mapping[BlockKey, list[tuple[int, str]]] | None = None,
    ) -> OpHandle:
        """Asynchronously fetch blocks into KVCR-managed storage."""
        return self._core.fetch(keys, request_id, hints, expected_layouts)

    def release(
        self,
        handles: Collection[ReleaseHandle],
    ) -> list[ReleaseResult]:
        """Release block residency claims."""
        return self._core.release(handles)

    def poll_completed(self) -> Iterable[OpResult]:
        """Drain completed operation results."""
        return self._core.poll_completed()

    def abort(
        self,
        op_handle: OpHandle,
        keys: Collection[BlockKey] | None = None,
    ) -> bool:
        """Best-effort abort an operation or selected entries."""
        return self._core.abort(op_handle, keys)

    def get_stats(self) -> TelemetryStats | None:
        """Return telemetry state when telemetry is enabled."""
        return self._core.get_stats()

    def close(self) -> None:
        """Stop progress and release controller-held resources.

        A core that failed to close but reached quiescence still gives the pool
        back: nothing is moving through it.
        """
        try:
            self._core.close()
        except BaseException:
            if not self._core.is_quiescent():
                raise
            try:
                self._release_pool_hold()
            except BaseException:
                # The core failure is the cause; this is its consequence.
                logger.exception("Failed to release the KVCR pool after close")
            raise
        self._release_pool_hold()

    def _release_pool_hold(self) -> None:
        pool_hold = self._pool_hold
        if pool_hold is not None:
            pool_hold.release()
            self._pool_hold = None
