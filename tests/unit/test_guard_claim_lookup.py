# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Source claims reuse matching layout work and reject retiring exports."""

import threading

import pytest

from kvcr import local_dram
from kvcr.local_dram import (
    _GuardExport,
    _LocalDram,
    _LocalDramResidency,
    _LocalDramState,
)
from kvcr.progress import _TransferRef


@pytest.fixture
def exported():
    local = object.__new__(_LocalDram)
    local._export_lock = threading.Lock()
    local._export_held = 0
    local._export_sequence = 0
    local._capture_writer = None
    local._exports = {}
    for index, key in enumerate((b"a", b"b")):
        residency = _LocalDramResidency(
            [("left", index), ("right", index)], _LocalDramState.READY
        )
        refs = tuple(
            _TransferRef("source", index, label=label, framework=False)
            for label in residency.layout
        )
        local._exports[key] = _GuardExport(
            residency, tuple(residency.layout), residency.key_layout, refs
        )
    return local


def test_repeated_layout_mapping_is_computed_once_per_claim(exported, monkeypatch):
    calls = []
    original = local_dram._layout_indices

    def mapping(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(local_dram, "_layout_indices", mapping)
    claims, busy = exported.claim_export_subset(
        (b"a", b"b"), (("right", "left"), ("right", "left")), ("left", "right")
    )
    assert not busy and len(claims) == 2
    assert len(calls) == 1
    assert [ref.label for ref in claims[0][1].refs] == ["right", "left"]
    exported.release_export_claims(tuple(claim for _, claim in claims))
    assert exported._export_held == 0


def test_retiring_residency_is_not_exported_even_if_index_entry_remains(exported):
    exported._exports[b"a"].residency.retire_on_release = True
    claims, busy = exported.claim_export_subset(
        (b"a", b"b"), (("left",), ("left",)), ("left", "right")
    )
    assert not busy and [position for position, _ in claims] == [1]
    assert exported._exports[b"a"].residency.export_readers == 0
    exported.release_export_claims(tuple(claim for _, claim in claims))
