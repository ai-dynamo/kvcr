# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Admission and asynchronous cache-operation result types."""

from enum import Enum

import msgspec


class HintStatus(Enum):
    # Admission is distinct from an asynchronous operation's completion.
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"  # invalid, unsupported, handler_error
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class HintResult(msgspec.Struct, frozen=True):
    action_id: str | None
    action_type: str
    status: HintStatus
    reason: str | None = None
    request_id: str | None = None
    message_id: str | None = None
    # Independent actions: the hashes that failed, with a reason each.
    failures: dict[int, str] | None = None
