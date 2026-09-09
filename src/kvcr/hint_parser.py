# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parse KV hint envelopes for KVCR fetch metadata.

Hints are advisory request metadata. This module validates versioned KV hint
envelopes, extracts the first ``kv.fetch`` action, and parses the
fields KVCR consumes. Integration constants are exported by ``kvcr.__init__``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

# Public integration constants exported by kvcr.__init__.
ROUTER_HINT_KEY = "kv_hint"
ROUTER_HINT_CAPABILITIES = frozenset({"router_hint"})

# Envelope/action versions currently understood by the parser.
_KV_HINT_PROTOCOL_VERSION = "0.1"
_KV_FETCH_ACTION_TYPE = "kv.fetch"
_KV_FETCH_ACTION_VERSION = "1.0"

# Hint parsing runs on the request path; suppress repeated warnings for the same
# malformed integration issue.
logger = logging.getLogger(__name__)
_LOGGED_HINT_ISSUES: set[str] = set()


def _warn_version_mismatch(
    *,
    supported: object,
    received: object,
    issue: str,
    message: str,
) -> None:
    """Warn once when a supported schema version does not match received value."""
    if received == supported or issue in _LOGGED_HINT_ISSUES:
        return
    _LOGGED_HINT_ISSUES.add(issue)
    logger.warning(message, supported, received)


@dataclass(frozen=True, slots=True)
class _KVFetchHint:
    """Fields KVCR currently supports from the first ``kv.fetch`` action."""

    source_endpoint: str
    block_hashes: frozenset[int]

    @classmethod
    def from_hint(cls, hint: Mapping[str, object]) -> _KVFetchHint:
        """Parse and validate a KV hint envelope."""
        if not isinstance(hint, Mapping):
            raise ValueError("invalid router hint")

        _warn_version_mismatch(
            supported=_KV_HINT_PROTOCOL_VERSION,
            received=hint.get("protocol_version"),
            issue="protocol-version-mismatch",
            message="KV hint protocol_version mismatch; supported=%r received=%r",
        )

        actions = hint.get("actions")
        if not isinstance(actions, list):
            raise ValueError("invalid router hint")

        fetch_payload = None
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            if action.get("action_type") != _KV_FETCH_ACTION_TYPE:
                continue
            _warn_version_mismatch(
                supported=_KV_FETCH_ACTION_VERSION,
                received=action.get("action_version"),
                issue="fetch-version-mismatch",
                message=(
                    "kv.fetch action_version mismatch; supported=%r received=%r; "
                    "processing with supported schema"
                ),
            )
            fetch_payload = action.get("payload")
            break

        if not isinstance(fetch_payload, Mapping):
            raise ValueError("invalid router hint")

        # TODO: Add mode and no_retain when KVCR starts consuming them.
        if "no_retain" in fetch_payload:
            raise ValueError("no_retain is not currently supported")

        mode = fetch_payload.get("mode", "copy")
        if mode != "copy":
            raise ValueError("only copy mode is currently supported")

        source = fetch_payload.get("source_control_endpoint")
        if source is None:
            raise ValueError("source-less hints are not currently supported")

        block_hashes = fetch_payload.get("block_hashes")
        if (
            not isinstance(source, str)
            or not source
            or not isinstance(block_hashes, list)
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
        return cls(source, frozenset(hashes))
