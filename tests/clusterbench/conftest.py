# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""In CI a skip here is a failure: a job that lost its vLLM or its NIXL must
not pass by running nothing."""

import os

import pytest

IN_CI = bool(os.environ.get("CI"))


def _fail_in_ci(report):
    if IN_CI and report.skipped and not hasattr(report, "wasxfail"):
        report.outcome = "failed"
        report.longrepr = f"skipped in CI: {report.longrepr[2]}"
    return report


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    return _fail_in_ci((yield))


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    return _fail_in_ci((yield))
