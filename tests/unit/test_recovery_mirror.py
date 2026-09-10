# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import msgspec
import pytest
from _kvcr_test_utils import _recovered_record

from kvcr.core import _BlockRecord
from kvcr.local_disk import _G3Residency
from kvcr.local_dram import _LocalDramResidency, _LocalDramState
from kvcr.recovery_journal import (
    _RECORD_BLOCK,
    _RECOVERY_ENCODER,
    RecoveryMirrorError,
    _decode_recovery_record,
    _project_recovery_record,
    _recovery_frames,
    _RecoveryMirror,
)
from kvcr.types import BlockKey

_ONE_POOL = ("",)
_TWO_POOLS = ("full", "swa")


def _payload(record: _BlockRecord) -> bytes:
    return _RECOVERY_ENCODER.encode(_project_recovery_record(record))


# Every live-only field set, to prove projection strips all of it.
_FULLY_LOADED_RECORD = _BlockRecord(
    fw_mem=object(),
    local_dram=_LocalDramResidency(
        [("", 3)],
        _LocalDramState.READY,
        claim_count=2,
        retire_on_release=True,
    ),
    g3=_G3Residency(5, claim_count=4),
    in_flight_ops={("deposit", 7)},
    access_count=8,
    last_access=9.0,
)


@pytest.mark.parametrize(
    ("record", "pool_names", "wire", "recovered"),
    [
        (
            _FULLY_LOADED_RECORD,
            _ONE_POOL,
            [[["", 3]], 5],
            _recovered_record(g2=[("", 3)], g3=5),
        ),
        (
            _recovered_record(g2=[("full", 7), ("full", 2), ("swa", 9)], g3=5),
            _TWO_POOLS,
            [[["full", 7], ["full", 2], ["swa", 9]], 5],
            _recovered_record(g2=[("full", 7), ("full", 2), ("swa", 9)], g3=5),
        ),
        (_BlockRecord(), _ONE_POOL, [None, None], _BlockRecord()),
        (
            _recovered_record(g2=[("", 3)]),
            _ONE_POOL,
            [[["", 3]], None],
            _recovered_record(g2=[("", 3)]),
        ),
        (_recovered_record(g3=5), _ONE_POOL, [None, 5], _recovered_record(g3=5)),
        # A G2 slot still FILLING or DISCARDING never settled, so it must not wire.
        (
            _BlockRecord(
                local_dram=_LocalDramResidency([("", 0)], _LocalDramState.FILLING)
            ),
            _ONE_POOL,
            [None, None],
            _BlockRecord(),
        ),
        (
            _BlockRecord(
                local_dram=_LocalDramResidency([("", 0)], _LocalDramState.DISCARDING)
            ),
            _ONE_POOL,
            [None, None],
            _BlockRecord(),
        ),
    ],
)
def test_recovery_wire_round_trip_keeps_only_settled_slots(
    record: _BlockRecord,
    pool_names: tuple[str, ...],
    wire: list[object],
    recovered: _BlockRecord,
) -> None:
    """Only settled G2/G3 slots reach the wire; decode rebuilds fresh live state."""
    encoded = _payload(record)

    # The outer record stays positional; G2 locations carry their pool names.
    assert msgspec.msgpack.decode(encoded) == wire
    assert _decode_recovery_record(encoded, pool_names) == recovered


def test_recovery_encoding_accepts_a_field_appended_later() -> None:
    """Appending is the one change this format allows, and it has to work."""

    class _RecoveryBlockV2(msgspec.Struct, frozen=True, array_like=True):
        g2: list[tuple[str, int]] | None = None
        g3: int | None = None
        appended: int = 0

    today = _payload(_recovered_record(g2=[("", 3)], g3=5))

    upgraded = msgspec.msgpack.Decoder(_RecoveryBlockV2).decode(today)
    assert upgraded.g2 == [("", 3)]
    assert upgraded.g3 == 5
    assert upgraded.appended == 0


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        msgspec.msgpack.encode({"g4": {"slot": 0}}),
        msgspec.msgpack.encode({"g2": {"slot": 0, "state": "ready"}}),
        msgspec.msgpack.encode({"g3": {"slot": -1}}),
        msgspec.msgpack.encode({"g2": {"slot": "0"}}),
        msgspec.msgpack.encode([0, None]),
        msgspec.msgpack.encode([[[""]], None]),
        msgspec.msgpack.encode([[["other", 0]], None]),
        msgspec.msgpack.encode([[["", -1]], None]),
        msgspec.msgpack.encode([[], None]),
    ],
)
def test_mirror_rejects_malformed_or_unknown_wire_state(payload: bytes) -> None:
    """A frame that does not decode to valid wire state is refused, not applied."""
    mirror = _RecoveryMirror(_ONE_POOL)

    with pytest.raises(RecoveryMirrorError, match="malformed"):
        mirror.apply(_RECORD_BLOCK, b"block", payload)


def test_mirror_replaces_blocks_whole_and_hands_them_over_uncopied() -> None:
    """Frames replace blocks whole in _records (mirrored table); take transfers it."""
    mirror = _RecoveryMirror(_ONE_POOL)
    key = BlockKey(b"spilled")

    mirror.apply(_RECORD_BLOCK, key, _payload(_recovered_record(g2=[("", 1)])))
    mirror.apply(_RECORD_BLOCK, key, _payload(_recovered_record(g2=[("", 1)], g3=7)))

    assert mirror._records == {key: _recovered_record(g2=[("", 1)], g3=7)}

    mirror.apply(_RECORD_BLOCK, key, _payload(_recovered_record(g3=7)))

    assert mirror._records == {key: _recovered_record(g3=7)}

    # An all-absent record is the tombstone.
    mirror.apply(_RECORD_BLOCK, key, _payload(_BlockRecord()))

    assert mirror._records == {}

    mirror.apply(_RECORD_BLOCK, b"resident", _payload(_recovered_record(g2=[("", 1)])))
    held = mirror._records

    taken = mirror.take_records()

    # Sole ownership: copying would leave two live populations of the set.
    assert taken is held
    assert taken == {BlockKey(b"resident"): _recovered_record(g2=[("", 1)])}
    assert mirror._records == {}


def test_mirror_adopts_exactly_what_a_handback_region_would_carry() -> None:
    """Adoption keeps the served table in place, pruned to what frames would carry."""
    ready, spilled = BlockKey(b"ready"), BlockKey(b"spilled")
    filling, forgotten = BlockKey(b"filling"), BlockKey(b"forgotten")
    filling_spill = BlockKey(b"filling-spill")
    discarding_spill = BlockKey(b"discarding-spill")
    served = {
        ready: _BlockRecord(
            local_dram=_LocalDramResidency(
                [("full", 0), ("swa", 10)],
                _LocalDramState.READY,
                claim_count=1,
                retire_on_release=True,
            ),
            in_flight_ops={("target", 7)},
            access_count=12,
            last_access=99.5,
        ),
        spilled: _BlockRecord(g3=_G3Residency(3, claim_count=2)),
        filling: _BlockRecord(
            local_dram=_LocalDramResidency(
                [("full", 1), ("swa", 11)], _LocalDramState.FILLING
            )
        ),
        forgotten: _BlockRecord(),
        # A good G3 residency must not carry a half-written G2 slot with it.
        filling_spill: _BlockRecord(
            local_dram=_LocalDramResidency(
                [("full", 7), ("swa", 17)], _LocalDramState.FILLING
            ),
            g3=_G3Residency(4),
        ),
        discarding_spill: _BlockRecord(
            local_dram=_LocalDramResidency(
                [("full", 8), ("swa", 18)], _LocalDramState.DISCARDING
            ),
            g3=_G3Residency(5),
        ),
    }
    # A kept mirror must match exactly what the handback frames carry.
    framed = {
        BlockKey(key): _decode_recovery_record(payload, _TWO_POOLS)
        for _, key, payload in _recovery_frames(served)
    }
    mirror = _RecoveryMirror(_TWO_POOLS)
    mirror.adopt(served)

    assert mirror._records is served
    assert mirror._records == framed
    assert mirror._records == {
        ready: _recovered_record(g2=[("full", 0), ("swa", 10)]),
        spilled: _recovered_record(g3=3),
        filling_spill: _recovered_record(g3=4),
        discarding_spill: _recovered_record(g3=5),
    }
