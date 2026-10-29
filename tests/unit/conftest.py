# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging

import pytest
from _kvcr_test_utils import _OPEN_KVCRS

from kvcr.service import memmap


def pytest_addoption(parser):
    parser.addoption(
        "--hugetlbfs-dir",
        help="directory directly under a hugetlbfs mount, for the real-hugetlbfs test",
    )


@pytest.fixture(autouse=True)
def _close_open_kvcrs():
    yield
    while _OPEN_KVCRS:
        _OPEN_KVCRS.pop().close()


@pytest.fixture(autouse=True)
def _no_hugetlb(monkeypatch):
    monkeypatch.setattr(memmap, "_hugetlbfs_mounts", lambda: [])


@pytest.fixture()
def kvcr_caplog(caplog):
    kvcr_logger = logging.getLogger("kvcr.runtime.core")
    kvcr_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=kvcr_logger.name):
            yield caplog
    finally:
        kvcr_logger.removeHandler(caplog.handler)
