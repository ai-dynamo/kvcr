# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Capture move advice and invalidate copies already held at the destination."""

import logging
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..types import BlockKey

if TYPE_CHECKING:
    from ..runtime.core import _KVCRCore
    from .remote_fw_dram import _RequestHint

logger = logging.getLogger(__name__)


@dataclass
class _PendingMove:
    """Admission snapshot: keys still to transfer, and complete local hits
    awaiting one invalidation message to the source.
    """

    hint: "_RequestHint"
    layout: tuple[str, ...]
    keys: set[BlockKey]
    cached_keys: list[BlockKey] = field(default_factory=list)


class _MoveHandler:
    """Keep move advice stable and send invalidation for complete local hits."""

    def __init__(self, core: "_KVCRCore") -> None:
        self._core = core

    def prepare(
        self,
        keys: Collection[BlockKey],
        request_id: str | None,
        layout: tuple[str, ...],
    ) -> _PendingMove | None:
        """Capture usable move advice; other keys keep ordinary fetch behavior."""
        remote = self._core._remote_fw_dram
        hint = remote._request_hints.get(request_id) if request_id is not None else None
        adapter = remote._key_adapter
        if (
            hint is None
            or not hint.move
            or hint.source is None
            or hint.failed
            or adapter is None
            or hint.source == getattr(self._core.framework_control, "endpoint", None)
        ):
            return None
        matched = {
            key
            for key in keys
            if key not in hint.missing_keys
            and adapter.encode_to_router(key) in hint.block_hashes
        }
        return _PendingMove(hint, layout, matched) if matched else None

    def invalidate_cached(self, move: _PendingMove) -> None:
        """Send one invalidation for the complete local hits found so far."""
        from .remote_ops import _Invalidate

        if not move.cached_keys:
            return
        keys = tuple(move.cached_keys)
        move.cached_keys.clear()
        try:
            self._core._progress.submit(
                _Invalidate(
                    keys=keys,
                    key_layout=move.layout,
                    source_control_endpoint=move.hint.source,
                )
            )
        except Exception:
            # Removal is best effort; the destination result stands.
            logger.warning(
                "KVCR cached-key invalidation could not reach %s",
                move.hint.source,
                exc_info=True,
            )
