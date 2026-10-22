# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
from _nvtx_test_utils import RecordingDomain

from kvcr import _nvtx


@pytest.fixture
def recording(monkeypatch):
    np = pytest.importorskip("numpy")
    domain = RecordingDomain()
    monkeypatch.setenv("KVCR_NVTX_LEVEL", "low")
    monkeypatch.setattr(_nvtx, "_load_backend", lambda: (domain, np))
    return domain
