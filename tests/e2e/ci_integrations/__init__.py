# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualified engine integrations for the common per-PR CI runner.

Register an adapter only with a validated catalog and evidence policy. A target
architecture row alone does not make an integration runnable or passing.
"""

from importlib import import_module
from types import ModuleType

_INTEGRATIONS = {
    "vllm": "ci_integrations.vllm",
    "sglang-hicache": "ci_integrations.sglang_hicache",
}


def get_integration(name: str) -> ModuleType:
    if name not in _INTEGRATIONS:
        raise ValueError(f"integration is not registered: {name!r}")
    return import_module(_INTEGRATIONS[name])
