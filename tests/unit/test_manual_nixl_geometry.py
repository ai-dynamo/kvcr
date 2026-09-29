# SPDX-License-Identifier: Apache-2.0
"""Pure geometry validation for the standalone GPU probe; no GPU imports."""

import runpy
from pathlib import Path

import pytest

_probe = runpy.run_path(
    str(Path(__file__).parents[1] / "manual" / "nixl_ucx_self_copy.py")
)


def test_mixed_span_pattern():
    sizes = _probe["parse_span_pattern"]("8704:6,17408:2,74880:6,149760:2")
    assert len(sizes) == 16
    assert sum(sizes) == 835840
    spans, extent = _probe["span_layout"](sizes, 4096)
    assert extent == 897280
    assert all(a + length + 4096 == b for (a, length), (b, _) in zip(spans, spans[1:]))


def test_uniform_and_single_span():
    assert _probe["span_layout"]([32, 32], 0) == ([(0, 32), (32, 32)], 64)
    assert _probe["span_layout"]([128], 4096) == ([(0, 128)], 128)


@pytest.mark.parametrize(
    "pattern", ["", "4", "4:0", "0:4", "-4:2", "4:x", "4:2:1", "4:2,"]
)
def test_reject_bad_pattern(pattern):
    with pytest.raises(ValueError):
        _probe["parse_span_pattern"](pattern)


@pytest.mark.parametrize("sizes,gap", [([], 0), ([0], 0), ([-1], 0), ([1], -1)])
def test_reject_bad_layout(sizes, gap):
    with pytest.raises(ValueError):
        _probe["span_layout"](sizes, gap)
