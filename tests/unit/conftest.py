# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging

import pytest
from _kvcr_test_utils import _OPEN_KVCRS


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
