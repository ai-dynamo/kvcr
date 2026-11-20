# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hint data types and the contracts shared by parsing and dispatch."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import msgspec

if TYPE_CHECKING:
    from ..runtime.core import _KVCRCore
    from .dispatcher import HintDispatcher
    from .hint_handler import _HintHandler


class _HintAction(msgspec.Struct, frozen=True):
    """One wire action; payload validation belongs to the shared parser."""

    action_id: str | None
    action_type: str
    action_version: str
    payload: dict[str, object]


# Control-socket envelopes above this size are dropped without being decoded.
MAX_HINT_MESSAGE_BYTES = 16 << 20


class KvHintsEnvelope(msgspec.Struct, frozen=True):
    """The request-compatible envelope carried by hint messages."""

    protocol_version: str
    message_id: str
    actions: list[_HintAction]


class _Hint(msgspec.Struct, frozen=True):
    """Base for parsed hints; ``ACTION_TYPE`` selects the registered handler."""

    ACTION_TYPE: ClassVar[str]
    action_id: str | None


class _KVFetchHint(_Hint):
    """A parsed ``kv.fetch`` hint."""

    ACTION_TYPE: ClassVar[str] = "kv.fetch"

    source_endpoint: str
    block_hashes: frozenset[int]
    no_retain: bool | None = None


class _KVEvictOnFetchHint(_KVFetchHint):
    ACTION_TYPE: ClassVar[str] = "kv.evict_on_fetch"


class _KVPrefetchHint(_Hint):
    """A parsed ``kv.prefetch`` hint, received by the source."""

    ACTION_TYPE: ClassVar[str] = "kv.prefetch"

    destination_endpoint: str
    block_hashes: frozenset[int]


class _KVMoveHint(_KVPrefetchHint):
    ACTION_TYPE: ClassVar[str] = "kv.move"


class _KVNoRetainHint(_Hint):
    """Set or clear persistent retention advice on the receiving KVCR."""

    ACTION_TYPE: ClassVar[str] = "kv.no_retain"

    block_hashes: frozenset[int]
    no_retain: bool


class _KVInvalidateHint(_Hint):
    ACTION_TYPE: ClassVar[str] = "kv.invalidate"

    block_hashes: frozenset[int]


_HintParser = Callable[[str | None, Mapping[str, object]], _Hint]
_HintHandlerFactory = Callable[["_KVCRCore", "HintDispatcher"], "_HintHandler"]


@dataclass(frozen=True)
class HintType:
    """Registry entry pairing an action's payload parser with its handler.

    ``version`` is the ``action_version`` the parser understands.
    ``parse(action_id, payload)`` returns the typed hint or raises
    ``ValueError(reason)``, which becomes the hint's rejection reason.
    ``handler_factory(core, dispatcher)`` builds the handler: ``core`` for
    backend access, ``dispatcher`` to ``report`` results.
    """

    action_type: str
    version: str
    parse: _HintParser
    handler_factory: _HintHandlerFactory
