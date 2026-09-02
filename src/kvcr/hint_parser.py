# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hint protocol parsing."""

from __future__ import annotations

from collections.abc import Mapping

ROUTER_HINT_CAPABILITIES = frozenset({"router_hint"})


def _parse_kv_hint(
    payload: Mapping[str, object],
) -> tuple[str | None, frozenset[int], str]:
    """Validate a JSON-shaped router hint and extract KVCR-owned fields."""
    if not isinstance(payload, Mapping):
        raise ValueError("invalid router hint")

    source = payload.get("source_control_endpoint")
    block_hashes = payload.get("block_hashes")
    mode = payload.get("mode", "copy")
    if (
        (source is not None and (not isinstance(source, str) or not source))
        or not isinstance(block_hashes, list)
        or not isinstance(mode, str)
        or mode not in ("copy", "move")
        or not isinstance(payload.get("no_retain", False), bool)
    ):
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
    return source, frozenset(hashes), mode
