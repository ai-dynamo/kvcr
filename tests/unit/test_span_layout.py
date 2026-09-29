# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Layer projections must be named, ordered, size-safe and delivery-only."""

import pytest
from _kvcr_test_utils import _mem_descriptor

from kvcr.span_layout import delivery_indices, select_delivery_spans


@pytest.mark.parametrize(
    "source,destination,expected",
    [
        (("a", "b", "c"), ("b",), (1,)),
        (("a", "b", "c"), ("a", "c"), (0, 2)),
        (("a", "b", "c"), ("c", "a"), None),
        (("a", "b"), ("b", "a"), None),
        (("a", "b"), ("missing",), None),
        (("a", "a", "b"), ("b",), None),
        (("a", "b", "c"), ("a", "a"), None),
        (("", "b"), ("b",), None),
        (("a", "b"), (), None),
        (("", ""), ("", ""), (0, 1)),
        (("a", "a"), ("a", "a"), (0, 1)),
    ],
)
def test_delivery_indices(source, destination, expected):
    assert delivery_indices(source, destination) == expected


def test_delivery_selection_checks_sizes_and_does_not_weaken_fetch():
    source = [_mem_descriptor(size=16, info="a"), _mem_descriptor(size=8, info="b")]
    destination = [_mem_descriptor(size=8, info="b")]
    assert select_delivery_spans(source, destination, allow_subset=True) == (source[1],)
    assert select_delivery_spans(source, destination, allow_subset=False) is None
    assert (
        select_delivery_spans(
            source, [_mem_descriptor(size=16, info="b")], allow_subset=True
        )
        is None
    )
    assert (
        select_delivery_spans(
            source, [source[0], _mem_descriptor(size=16, info="b")], allow_subset=True
        )
        is None
    )
