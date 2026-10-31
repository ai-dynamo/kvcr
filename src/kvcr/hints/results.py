# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-hint result types, internal to ``kvcr.hints``."""

from enum import Enum

import msgspec


class HintStatus(Enum):
    # Inline hints are terminal at submit time: ACCEPTED or REJECTED only.
    ACCEPTED = "ACCEPTED"  # inline work done
    REJECTED = "REJECTED"  # invalid, unsupported, handler_error


class HintResult(msgspec.Struct, frozen=True):
    action_id: str | None
    action_type: str
    status: HintStatus
    reason: str | None = None
    request_id: str | None = None
    message_id: str | None = None
