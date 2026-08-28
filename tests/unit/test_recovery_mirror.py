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
    _RECOVERY_DECODER,
    _RECOVERY_ENCODER,
    RecoveryMirrorError,
    _decode_recovery_record,
    _project_recovery_record,
    _recovery_frames,
    _RecoveryBlock,
    _RecoveryMirror,
)
from kvcr.types import BlockKey


def _payload(record: _BlockRecord) -> bytes:
    return _RECOVERY_ENCODER.encode(_project_recovery_record(record))


def test_recovery_projection_has_only_g2_and_g3_wire_state() -> None:
    record = _BlockRecord(
        fw_mem=object(),
        local_dram=_LocalDramResidency(
            3,
            _LocalDramState.READY,
            claim_count=2,
            retire_on_release=True,
        ),
        g3=_G3Residency(5, claim_count=4),
        in_flight_ops={("deposit", 7)},
        access_count=8,
        last_access=9.0,
    )

    recovered = _project_recovery_record(record)

    assert recovered == _RecoveryBlock(3, 5)
    # Positional, so no field names ride along in every record.
    encoded = _RECOVERY_ENCODER.encode(recovered)
    assert msgspec.msgpack.decode(encoded) == [3, 5]
    assert len(encoded) == 3


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (_BlockRecord(), [None, None]),
        (_recovered_record(g2=3), [3, None]),
        (_recovered_record(g3=5), [None, 5]),
    ],
)
def test_recovery_encoding_holds_a_place_for_an_absent_tier(
    record: _BlockRecord,
    expected: list[object],
) -> None:
    """An absent tier still occupies its slot, because position is the name."""
    assert msgspec.msgpack.decode(_payload(record)) == expected


def test_recovery_encoding_accepts_a_field_appended_later() -> None:
    """Appending is the one change this format allows, and it has to work."""

    class _RecoveryBlockV2(msgspec.Struct, frozen=True, array_like=True):
        g2: int | None = None
        g3: int | None = None
        appended: int = 0

    today = _payload(_recovered_record(g2=3, g3=5))

    assert msgspec.msgpack.Decoder(_RecoveryBlockV2).decode(today) == _RecoveryBlockV2(
        3, 5, 0
    )


def test_recovery_decode_builds_fresh_live_state() -> None:
    decoded = _decode_recovery_record(_payload(_recovered_record(g2=3, g3=5)))

    assert decoded.local_dram == _LocalDramResidency(3, _LocalDramState.READY)
    assert decoded.g3 == _G3Residency(5)
    assert decoded.fw_mem is None
    assert decoded.in_flight_ops is None
    assert decoded.access_count == 0
    assert decoded.last_access is None


@pytest.mark.parametrize(
    "state",
    [_LocalDramState.FILLING, _LocalDramState.DISCARDING],
)
def test_recovery_projection_omits_transient_local_dram(
    state: _LocalDramState,
) -> None:
    record = _BlockRecord(local_dram=_LocalDramResidency(0, state))

    assert _project_recovery_record(record).g2 is None


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        msgspec.msgpack.encode({"g4": {"slot": 0}}),
        msgspec.msgpack.encode({"g2": {"slot": 0, "state": "ready"}}),
        msgspec.msgpack.encode({"g3": {"slot": -1}}),
        msgspec.msgpack.encode({"g2": {"slot": "0"}}),
    ],
)
def test_mirror_rejects_malformed_or_unknown_wire_state(payload: bytes) -> None:
    mirror = _RecoveryMirror()

    with pytest.raises(RecoveryMirrorError, match="malformed"):
        mirror.apply(_RECORD_BLOCK, b"block", payload)


def test_mirror_replaces_complete_state() -> None:
    mirror = _RecoveryMirror()
    key = BlockKey(b"spilled")

    mirror.apply(_RECORD_BLOCK, key, _payload(_recovered_record(g2=1)))
    mirror.apply(_RECORD_BLOCK, key, _payload(_recovered_record(g2=1, g3=7)))
    mirror.apply(_RECORD_BLOCK, key, _payload(_recovered_record(g3=7)))

    assert mirror._records == {key: _recovered_record(g3=7)}

    mirror.apply(_RECORD_BLOCK, key, _payload(_BlockRecord()))

    assert mirror._records == {}


def test_mirror_adopts_exactly_what_a_handback_region_would_carry() -> None:
    ready, spilled = BlockKey(b"ready"), BlockKey(b"spilled")
    filling, forgotten = BlockKey(b"filling"), BlockKey(b"forgotten")
    served = {
        ready: _recovered_record(g2=0),
        spilled: _recovered_record(g3=3),
        filling: _BlockRecord(
            local_dram=_LocalDramResidency(1, _LocalDramState.FILLING)
        ),
        forgotten: _BlockRecord(),
    }
    # A kept mirror must match exactly what the handback frames carry.
    framed = {BlockKey(key) for _, key, _ in _recovery_frames(served)}
    assert framed == {ready, spilled}

    mirror = _RecoveryMirror()
    mirror.adopt(served)

    assert mirror._records is served
    assert set(served) == framed
    assert mirror._records == {
        ready: _recovered_record(g2=0),
        spilled: _recovered_record(g3=3),
    }


@pytest.mark.parametrize("state", [_LocalDramState.FILLING, _LocalDramState.DISCARDING])
def test_mirror_adoption_drops_a_g2_slot_that_never_settled(state) -> None:
    """A good G3 residency must not carry a half-written G2 slot with it."""
    spilled = BlockKey(b"spilled")
    served = {
        spilled: _BlockRecord(
            local_dram=_LocalDramResidency(7, state), g3=_G3Residency(3)
        )
    }
    framed = _RECOVERY_DECODER.decode(
        next(payload for _, _, payload in _recovery_frames(dict(served)))
    )

    mirror = _RecoveryMirror()
    mirror.adopt(served)

    # What is kept and what the region would carry have to agree exactly.
    assert framed.g2 is None
    assert mirror._records == {spilled: _recovered_record(g3=3)}


def test_mirror_hands_records_over_without_copying_them() -> None:
    mirror = _RecoveryMirror()
    mirror.apply(_RECORD_BLOCK, b"resident", _payload(_recovered_record(g2=1)))
    held = mirror._records

    taken = mirror.take_records()

    # Sole ownership: copying would leave two live populations of the set.
    assert taken is held
    assert taken == {BlockKey(b"resident"): _recovered_record(g2=1)}
    assert mirror._records == {}


def test_adoption_keeps_exactly_what_the_wire_would_have_carried() -> None:
    """The invariant, stated as an identity rather than a list of fields."""
    key = BlockKey(b"live")
    live = _BlockRecord(
        local_dram=_LocalDramResidency(
            4, _LocalDramState.READY, claim_count=1, retire_on_release=True
        ),
        g3=_G3Residency(9, claim_count=2),
    )
    live.add_in_flight_op(("target", 7))
    live.access_count = 12
    live.last_access = 99.5
    through_the_wire = _decode_recovery_record(
        _RECOVERY_ENCODER.encode(_project_recovery_record(live))
    )

    mirror = _RecoveryMirror()
    mirror.adopt({key: live})

    assert mirror._records == {key: through_the_wire}
