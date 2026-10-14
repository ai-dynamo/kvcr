# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Reject a partial capture that cannot contain any successful blocks."""

import subprocess
import sys
from pathlib import Path


def test_partial_capture_requires_two_blocks():
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[2] / "examples/nvtx_remote_capture.py"),
            "--scenario",
            "partial",
            "--blocks",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 2
    assert "partial requires at least two blocks" in result.stderr
