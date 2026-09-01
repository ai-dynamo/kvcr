# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parsing for compact router_hint source-location metadata."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

ExternalBlockHash: TypeAlias = bytes | int

logger = logging.getLogger(__name__)

_ROUTER_HINT_KEY = "router_hint"
ROUTER_HINT_CAPABILITIES = frozenset({_ROUTER_HINT_KEY})

logger.info("KVCR_ROUTER_HINT_PARSER_LOADED")


@dataclass(frozen=True)
class KvSourceLocationsHint:
    """Router-supplied source candidate for request block reuse."""

    source_control_endpoint: str
    block_hashes: frozenset[ExternalBlockHash]


def extract_kv_hint(
    kv_transfer_params: Mapping[str, Any] | None,
) -> KvSourceLocationsHint | None:
    """Extract a typed router hint from KV transfer params."""
    if not kv_transfer_params:
        return None

    payload = kv_transfer_params.get(_ROUTER_HINT_KEY)
    if not isinstance(payload, Mapping):
        return None

    location = payload.get("source_control_endpoint")
    if not isinstance(location, str) or not location:
        return None

    block_hashes = payload.get("block_hashes")
    if not isinstance(block_hashes, list) or not block_hashes:
        return None

    planned_hashes: set[ExternalBlockHash] = set()
    for block_hash in block_hashes:
        if isinstance(block_hash, bool) or not isinstance(block_hash, (bytes, int)):
            return None
        planned_hashes.add(block_hash)

    return KvSourceLocationsHint(location, frozenset(planned_hashes))
