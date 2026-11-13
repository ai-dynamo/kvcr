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


class _Hint(msgspec.Struct, frozen=True):
    """Base for parsed hints; ``ACTION_TYPE`` selects the registered handler."""

    ACTION_TYPE: ClassVar[str]
    action_id: str | None


class _KVFetchHint(_Hint):
    """A parsed ``kv.fetch`` hint."""

    ACTION_TYPE: ClassVar[str] = "kv.fetch"

    source_endpoint: str
    block_hashes: frozenset[int]


class _KVEvictOnFetchHint(_KVFetchHint):
    ACTION_TYPE: ClassVar[str] = "kv.evict_on_fetch"


class _KVPrefetchHint(_Hint):
    """A parsed ``kv.prefetch`` hint, received by the source."""

    ACTION_TYPE: ClassVar[str] = "kv.prefetch"

    destination_endpoint: str
    block_hashes: frozenset[int]


class _KVMoveHint(_Hint):
    """A parsed ``kv.move`` hint, received by the source."""

    ACTION_TYPE: ClassVar[str] = "kv.move"

    destination_endpoint: str
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
