# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging

import pytest
from _kvcr_test_utils import _OPEN_KVCRS
from _nvtx_test_utils import RecordingDomain

from kvcr import _nvtx


@pytest.fixture(autouse=True)
def _close_open_kvcrs():
    yield
    while _OPEN_KVCRS:
        _OPEN_KVCRS.pop().close()


@pytest.fixture()
def kvcr_caplog(caplog):
    kvcr_logger = logging.getLogger("kvcr.core")
    kvcr_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=kvcr_logger.name):
            yield caplog
    finally:
        kvcr_logger.removeHandler(caplog.handler)


@pytest.fixture
def recording(monkeypatch):
    np = pytest.importorskip("numpy")
    domain = RecordingDomain()
    monkeypatch.setenv("KVCR_NVTX_LEVEL", "low")
    monkeypatch.setattr(_nvtx, "_load_backend", lambda: (domain, np))
    return domain
