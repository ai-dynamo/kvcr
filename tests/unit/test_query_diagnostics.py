# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ctypes
import logging
from dataclasses import replace

from _kvcr_test_utils import FakeNixlAgent, _new_local_kvcr

from kvcr.core import _KVCRCore
from kvcr.types import BlockKey


def test_query_zero_results_are_bounded_observations_not_failures(monkeypatch, caplog):
    monkeypatch.setenv("KVCR_DIAGNOSTICS", "1")
    caplog.set_level(logging.DEBUG, logger="kvcr.core")
    local = ctypes.create_string_buffer(16)
    kvcr = _new_local_kvcr(FakeNixlAgent(), local, 1)
    core = kvcr._core
    core._remote_fw_dram._options = replace(
        core._remote_fw_dram._options, opportunistic_query=True
    )
    keys = [BlockKey(b"missing")]
    expected = _KVCRCore.query(core, keys, "req")
    for _ in range(10):
        assert core.query(keys, "req") == expected
    messages = [m for m in caplog.messages if "query_decision " in m]
    assert len(messages) == 4
    assert all("remote_fetchable=0 misses=1 opportunistic=True" in m for m in messages)
    assert all("hint_present=False" in m for m in messages)
    for i in range(130):
        core.query(keys, str(i))
    assert len(core._query_diagnostics) == 128
    caplog.clear()
    core.query(keys, "req")
    assert "query_ordinal=1" in caplog.messages[0]
