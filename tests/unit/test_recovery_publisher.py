# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from collections.abc import Callable

from _kvcr_test_utils import _recovered_record

from kvcr.core import _BlockRecord
from kvcr.recovery_journal import (
    _attach_journal,
    _decode_recovery_record,
)
from kvcr.types import BlockKey


class _Source:
    def __init__(self) -> None:
        self._observer: Callable[[BlockKey, _BlockRecord], None] | None = None

    def observe_residency(
        self, observer: Callable[[BlockKey, _BlockRecord], None]
    ) -> None:
        self._observer = observer

    def emit(self, key: BlockKey, record: _BlockRecord) -> None:
        assert self._observer is not None
        self._observer(key, record)


class _Journal:
    def __init__(self) -> None:
        self.frames: list[tuple[bytes, bytes]] = []
        self.invalidated = False
        self.refuse = False

    def publish(self, record_type: int, key: bytes, payload: bytes) -> bool:
        self.frames.append((key, payload))
        return not self.refuse

    def invalidate(self) -> bool:
        self.invalidated = True
        return False


def test_publisher_emits_stable_state_until_the_journal_refuses(caplog) -> None:
    """Stable G2/G3 mutations publish whole; one refusal warns once and stops all."""
    local_dram = _Source()
    g3 = _Source()
    journal = _Journal()
    key = BlockKey(b"block")
    _attach_journal(local_dram, journal, g3)
    caplog.set_level(logging.WARNING)

    local_dram.emit(key, _recovered_record(g2=2))
    g3.emit(key, _recovered_record(g2=2, g3=7))
    local_dram.emit(key, _recovered_record(g3=7))
    g3.emit(key, _BlockRecord())

    assert [frame_key for frame_key, _ in journal.frames] == [bytes(key)] * 4
    recovered = [_decode_recovery_record(payload) for _, payload in journal.frames]
    assert recovered == [
        _recovered_record(g2=2),
        _recovered_record(g2=2, g3=7),
        _recovered_record(g3=7),
        _BlockRecord(),
    ]
    assert caplog.messages == []

    journal.refuse = True
    local_dram.emit(key, _recovered_record(g2=0))
    local_dram.emit(key, _recovered_record(g2=1))

    # One refused attempt, then publication stays off with no second warning.
    assert len(journal.frames) == 5
    assert caplog.messages == [
        "KVCR recovery publication disabled after the journal rejected a frame"
    ]


def test_publication_failure_invalidates_recovery_without_escaping() -> None:
    """A publish that raises invalidates recovery; the mutation itself survives."""

    class _FailingJournal(_Journal):
        def publish(self, record_type: int, key: bytes, payload: bytes) -> bool:
            del key, payload
            raise RuntimeError("publish failed")

    local_dram = _Source()
    journal = _FailingJournal()
    _attach_journal(local_dram, journal)

    local_dram.emit(BlockKey(b"still-serving"), _recovered_record(g2=0))

    assert journal.invalidated
