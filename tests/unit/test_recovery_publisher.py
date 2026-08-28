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

    def publish(self, record_type: int, key: bytes, payload: bytes) -> bool:
        self.frames.append((key, payload))
        return True

    def invalidate(self) -> bool:
        self.invalidated = True
        return False


def test_publisher_emits_complete_g2_and_g3_state() -> None:
    local_dram = _Source()
    g3 = _Source()
    journal = _Journal()
    key = BlockKey(b"block")
    _attach_journal(local_dram, journal, g3)

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


def test_publication_failure_invalidates_recovery_without_escaping() -> None:
    class _FailingJournal(_Journal):
        def publish(self, record_type: int, key: bytes, payload: bytes) -> bool:
            del key, payload
            raise RuntimeError("publish failed")

    local_dram = _Source()
    journal = _FailingJournal()
    _attach_journal(local_dram, journal)

    local_dram.emit(BlockKey(b"still-serving"), _recovered_record(g2=0))

    assert journal.invalidated


def test_refused_publication_warns_once(caplog) -> None:
    class _RefusingJournal(_Journal):
        def publish(self, record_type: int, key: bytes, payload: bytes) -> bool:
            self.frames.append((key, payload))
            return False

    local_dram = _Source()
    journal = _RefusingJournal()
    _attach_journal(local_dram, journal)
    caplog.set_level(logging.WARNING)

    key = BlockKey(b"still-serving")
    local_dram.emit(key, _recovered_record(g2=0))
    local_dram.emit(key, _recovered_record(g2=1))

    assert len(journal.frames) == 1
    assert caplog.messages == [
        "KVCR recovery publication disabled after the journal rejected a frame"
    ]
