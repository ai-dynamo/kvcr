# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Strict ``kv.fetch_gpu`` parsing and GPU-only engine-destination admission."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

import msgspec

from .fetch import _parse_kv_fetch
from .parser import HintType, _Hint, reject_hint
from .results import HintResult, HintStatus

if TYPE_CHECKING:
    from ..runtime.core import _KVCRCore
    from .dispatcher import HintDispatcher


class _KVGPUFetchHint(msgspec.Struct, frozen=True):
    """A parsed GPU action; CPU hints never authorize a CUDA transfer."""

    ACTION_TYPE: ClassVar[str] = "kv.fetch_gpu"

    action_id: str | None
    source_endpoint: str
    block_hashes: frozenset[int]
    compatibility_domain: str


def _parse_kv_fetch_gpu(
    action_id: str | None, payload: Mapping[str, object]
) -> _KVGPUFetchHint:
    if any(
        payload.get(name) != value
        for name, value in (
            ("source_memory_kind", "gpu"),
            ("target_memory_kind", "gpu"),
            ("transport", "nixl_gpu_direct"),
        )
    ):
        raise ValueError("GPU router hint requires direct GPU memory on both peers")
    domain = payload.get("compatibility_domain")
    if (
        not isinstance(domain, str)
        or not domain
        or domain.strip() != domain
        or "\x00" in domain
    ):
        raise ValueError("GPU router hint requires an explicit compatibility domain")
    parsed = _parse_kv_fetch(action_id, payload)
    return _KVGPUFetchHint(
        action_id, parsed.source_endpoint, parsed.block_hashes, domain
    )


class _GPUFetchHintHandler:
    action_type = _KVGPUFetchHint.ACTION_TYPE

    def __init__(self, core: _KVCRCore, dispatcher: HintDispatcher) -> None:
        self._core = core
        self._dispatcher = dispatcher

    def submit(
        self, hint: _Hint, request_id: str | None, message_id: str | None
    ) -> None:
        hint: _KVGPUFetchHint = hint  # type: ignore[assignment]
        options = self._core._gpu_options
        if (
            options is None
            or hint.compatibility_domain != options.compatibility_domain
        ):
            self._dispatcher.report(
                reject_hint(
                    hint.action_id,
                    self.action_type,
                    "GPU endpoint requires its exact compatibility domain",
                    message_id,
                    request_id,
                )
            )
            return
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


KV_FETCH_GPU = HintType(
    "kv.fetch_gpu",
    "1.0",
    _parse_kv_fetch_gpu,
    _GPUFetchHintHandler,
    strict_versions=True,
)
