# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parse KV hint envelopes for KVCR fetch metadata.

Hints are advisory request metadata. This module validates versioned KV hint
envelopes, extracts the first usable ``kv.fetch`` action, and parses the
fields KVCR consumes. Integration constants are exported by ``kvcr.__init__``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

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


def _warn_once(issue: str, message: str, *args: object) -> None:
    """Log a parser warning at most once per issue kind.

    Args:
        issue: Stable identifier used to deduplicate warnings.
        message: Warning message format string.
        *args: Values interpolated into ``message``.
    """
    if issue in _LOGGED_HINT_ISSUES:
        return
    _LOGGED_HINT_ISSUES.add(issue)
    logger.warning(message, *args)


def _warn_version_mismatch(
    *,
    supported: object,
    received: object,
    issue: str,
    message: str,
) -> None:
    """Warn once when a supported schema version does not match received value.

    Args:
        supported: Version string currently supported by the parser.
        received: Version value from the hint envelope or action.
        issue: Stable identifier used to deduplicate warnings.
        message: Warning message format string for ``supported`` and
            ``received``.
    """
    if received == supported:
        return
    _warn_once(issue, message, supported, received)


def _parse_kv_hint(
    hint: Mapping[str, object],
) -> tuple[str | None, frozenset[int], str]:
    """Parse a KV hint envelope into KVCR fetch metadata.

    Hints are advisory and parsed on the request path. Version mismatches warn
    once per issue kind and are interpreted with the schema KVCR currently
    supports. KVCR consumes the first valid ``kv.fetch`` action in the envelope.

    Args:
        hint: Versioned KV hint envelope.

    Returns:
        A tuple (source, block_hashes, mode), where source is the optional
        source control endpoint, block_hashes holds deduplicated block hashes,
        and mode is the fetch mode.

    Raises:
        ValueError: If no usable ``kv.fetch`` action exists or the fetch
            payload is malformed.
    """
    if not isinstance(hint, Mapping):
        raise ValueError("invalid router hint")

    _warn_version_mismatch(
        supported=_KV_HINT_PROTOCOL_VERSION,
        received=hint.get("protocol_version"),
        issue="protocol-version-mismatch",
        message="KV hint protocol_version mismatch; supported=%r received=%r",
    )

    fetch_payload = _extract_kv_fetch_payload(hint)
    return _parse_fetch_payload(fetch_payload)


def _parse_fetch_action(action: object) -> Mapping[str, object] | None:
    """Parse one envelope action as a ``kv.fetch`` payload.

    Args:
        action: Candidate action from a KV hint envelope.

    Returns:
        The action payload for a usable ``kv.fetch`` action, or None if the
        action should be skipped.

    Raises:
        ValueError: If the action is ``kv.fetch`` but its payload is malformed.
    """
    if not isinstance(action, Mapping):
        return None
    if action.get("action_type") != _KV_FETCH_ACTION_TYPE:
        return None

    action_version = action.get("action_version")
    _warn_version_mismatch(
        supported=_KV_FETCH_ACTION_VERSION,
        received=action_version,
        issue=f"fetch-version-mismatch:{action_version}",
        message=(
            "kv.fetch action_version mismatch; supported=%r received=%r; "
            "processing with supported schema"
        ),
    )

    fetch_payload = action.get("payload")
    if not isinstance(fetch_payload, Mapping):
        raise ValueError("invalid router hint")
    return fetch_payload


def _extract_kv_fetch_payload(
    hint: Mapping[str, object],
) -> Mapping[str, object]:
    """Extract the first ``kv.fetch`` action payload from a hint envelope.

    Args:
        hint: Versioned KV hint envelope.

    Returns:
        The payload mapping from the first usable ``kv.fetch`` action.

    Raises:
        ValueError: If ``actions`` is missing or malformed, no ``kv.fetch``
            action exists, or the first matching action has a malformed
            payload.
    """
    actions = hint.get("actions")
    if not isinstance(actions, list):
        raise ValueError("invalid router hint")

    for action in actions:
        payload = _parse_fetch_action(action)
        if payload is not None:
            return payload

    raise ValueError("invalid router hint")


def _parse_fetch_payload(
    payload: Mapping[str, object],
) -> tuple[str | None, frozenset[int], str]:
    """Parse and validate fields from a ``kv.fetch`` action payload.

    Args:
        payload: ``kv.fetch`` action payload.

    Returns:
        A tuple (source, block_hashes, mode), where source is the optional
        source control endpoint, block_hashes holds deduplicated block hashes,
        and mode is the fetch mode.

    Raises:
        ValueError: If the payload does not match the supported router hint
            shape.
    """
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
