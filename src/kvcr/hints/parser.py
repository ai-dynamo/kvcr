# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parse KV hint envelopes into typed hints.

The caller supplies a ``HintType`` registry for envelope and payload validation.
Unusable hints become ``REJECTED`` results instead of raising. Integration
constants are exported by ``kvcr.__init__``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

import msgspec

from .hint_types import HintType, _Hint
from .results import HintResult, HintStatus

# Public integration constants exported by kvcr.__init__.
ROUTER_HINT_KEY = "kv_hint"
ROUTER_HINT_CAPABILITIES = frozenset({"router_hint"})

# Envelope version currently understood by the parser.
_KV_HINT_PROTOCOL_VERSION = "0.1"

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


def reject_hint(
    action_id: str | None,
    action_type: str,
    reason: str,
    message_id: str | None,
    request_id: str | None = None,
    *,
    supported: bool = True,
) -> HintResult:
    """Build a ``REJECTED`` result and warn once per (action type, reason).

    Shared with hint handlers, which use it for their own rejections.
    """
    # The dedupe set is process-global, so key it on a bounded value: a type
    # with no ``HintType`` is router-controlled input.
    issue = f"rejected:{action_type if supported else 'unknown'}:{reason}"
    if issue not in _LOGGED_HINT_ISSUES:
        _LOGGED_HINT_ISSUES.add(issue)
        logger.warning(
            "KV hint rejected; action_type=%s reason=%s", action_type, reason
        )
    return HintResult(
        action_id=action_id,
        action_type=action_type,
        status=HintStatus.REJECTED,
        reason=reason,
        request_id=request_id,
        message_id=message_id,
    )


def hint_message_id(hint: msgspec.Struct) -> str | None:
    """The envelope's ``message_id`` when it is a string, else ``None``."""
    message_id = getattr(hint, "message_id", None)
    return message_id if isinstance(message_id, str) else None


def _parse_payload(
    action_id: str | None,
    payload: Mapping[str, object],
    *,
    peer: str,
    hint_factory: Callable[[str | None, str, frozenset[int]], _Hint],
) -> _Hint:
    """The ``{peer}_control_endpoint`` and ``block_hashes`` of a hint payload."""
    no_retain = payload.get("no_retain", False)
    if not isinstance(no_retain, bool):
        raise ValueError("no_retain must be a boolean")
    endpoint = payload.get(f"{peer}_control_endpoint")
    if endpoint is None:
        raise ValueError(f"{peer}-less hints are not currently supported")
    block_hashes = payload.get("block_hashes")
    if (
        not isinstance(endpoint, str)
        or not endpoint
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
    hint = hint_factory(action_id, endpoint, frozenset(hashes))
    if "no_retain" in payload:
        try:
            hint = msgspec.structs.replace(hint, no_retain=no_retain)
        except TypeError as exc:
            raise ValueError("no_retain is not supported by this hint") from exc
    return hint


def parse_hints(
    hint: msgspec.Struct,
    types: Mapping[str, HintType],
) -> tuple[list[_Hint], list[HintResult]]:
    """Parse every hint of a KV hint envelope with ``types`` (by action type).

    Returns the valid hints in envelope order and a ``REJECTED`` result for
    each hint that could not be used. Raises ``ValueError`` only when the
    envelope itself is malformed.
    """
    if not isinstance(hint, msgspec.Struct):
        raise ValueError("invalid router hint")

    _warn_version_mismatch(
        supported=_KV_HINT_PROTOCOL_VERSION,
        received=getattr(hint, "protocol_version", None),
        issue="protocol-version-mismatch",
        message="KV hint protocol_version mismatch; supported=%r received=%r",
    )

    raw_actions = getattr(hint, "actions", None)
    if not isinstance(raw_actions, list):
        raise ValueError("invalid router hint")

    message_id = hint_message_id(hint)

    hints: list[_Hint] = []
    rejected: list[HintResult] = []
    for raw in raw_actions:
        if not isinstance(raw, msgspec.Struct):
            rejected.append(reject_hint(None, "unknown", "invalid_hint", message_id))
            continue

        action_id = getattr(raw, "action_id", None)
        if not isinstance(action_id, str):
            rejected.append(reject_hint(None, "unknown", "invalid_hint", message_id))
            continue
        action_type = getattr(raw, "action_type", None)
        if not isinstance(action_type, str):
            rejected.append(
                reject_hint(action_id, "unknown", "invalid_hint", message_id)
            )
            continue
        hint_type = types.get(action_type)
        if hint_type is None:
            rejected.append(
                reject_hint(
                    action_id,
                    action_type,
                    "unsupported_action_type",
                    message_id,
                    supported=False,
                )
            )
            continue

        _warn_version_mismatch(
            supported=hint_type.version,
            received=getattr(raw, "action_version", None),
            issue=f"{action_type}-version-mismatch",
            message=(
                f"{action_type.replace('%', '%%')} action_version mismatch; "
                "supported=%r received=%r; processing with supported schema"
            ),
        )

        payload = getattr(raw, "payload", None)
        if not isinstance(payload, Mapping):
            rejected.append(
                reject_hint(action_id, action_type, "invalid_payload", message_id)
            )
            continue
        try:
            hints.append(hint_type.parse(action_id, payload))
        except ValueError as exc:
            rejected.append(reject_hint(action_id, action_type, str(exc), message_id))
        except Exception:
            # A parser must not take down the envelope's other hints.
            rejected.append(
                reject_hint(action_id, action_type, "invalid_hint", message_id)
            )
    return hints, rejected
