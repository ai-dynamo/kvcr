# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``kv.fetch`` hint: parser and engine-destination handler.

The handler stores the per-request remote hint that the engine later turns
into ``fetch``/``deliver`` calls. It is inline only: no state lock, no block
records, and no transfer beyond the store's own eager control connect.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

import msgspec

from .parser import HintType, _Hint
from .results import HintResult, HintStatus

if TYPE_CHECKING:
    from ..core import _KVCRCore
    from .dispatcher import HintDispatcher

_KV_FETCH_ACTION_TYPE = "kv.fetch"
_KV_FETCH_ACTION_VERSION = "1.0"


class _KVFetchHint(msgspec.Struct, frozen=True):
    """A parsed ``kv.fetch`` hint."""

    ACTION_TYPE: ClassVar[str] = _KV_FETCH_ACTION_TYPE

    action_id: str | None
    source_endpoint: str
    block_hashes: frozenset[int]


def _parse_kv_fetch(
    action_id: str | None, payload: Mapping[str, object]
) -> _KVFetchHint:
    # TODO: Add mode and no_retain when KVCR starts consuming them.
    if "no_retain" in payload:
        raise ValueError("no_retain is not currently supported")

    mode = payload.get("mode", "copy")
    if mode != "copy":
        raise ValueError("only copy mode is currently supported")

    source = payload.get("source_control_endpoint")
    if source is None:
        raise ValueError("source-less hints are not currently supported")

    block_hashes = payload.get("block_hashes")
    if not isinstance(source, str) or not source or not isinstance(block_hashes, list):
        raise ValueError("invalid router hint")

    hashes: set[int] = set()
    for block_hash in block_hashes:
        if (
            isinstance(block_hash, bool)
            or not isinstance(block_hash, int)
            or not 0 <= block_hash < 1 << 64
        ):
            raise ValueError("invalid router hint")
        hashes.add(block_hash)
    if not hashes:
        raise ValueError("invalid router hint")

    # Add other fields to the parsed result when KVCR starts consuming them.
    return _KVFetchHint(action_id, source, frozenset(hashes))


class _FetchHintHandler:
    """Engine-destination ``kv.fetch``: stores the per-request remote hint."""

    action_type = _KVFetchHint.ACTION_TYPE

    def __init__(self, core: _KVCRCore, dispatcher: HintDispatcher) -> None:
        self._core = core  # backend access only
        self._dispatcher = dispatcher  # results

    def submit(
        self,
        hint: _Hint,
        request_id: str | None,
        message_id: str | None,
    ) -> None:
        hint: _KVFetchHint = hint  # type: ignore[assignment]
        # One store call per hint; the store's own rules apply (a later hint
        # replaces the request's hashes, a different source fails the request).
        self._core._remote_fw_dram.submit_hint(
            hint.source_endpoint, hint.block_hashes, request_id
        )
        self._dispatcher.report(
            HintResult(
                action_id=hint.action_id,
                action_type=self.action_type,
                status=HintStatus.ACCEPTED,
                request_id=request_id,
                message_id=message_id,
            )
        )

    def close(self) -> None:
        pass


KV_FETCH = HintType(
    _KV_FETCH_ACTION_TYPE,
    _KV_FETCH_ACTION_VERSION,
    _parse_kv_fetch,
    _FetchHintHandler,
)
