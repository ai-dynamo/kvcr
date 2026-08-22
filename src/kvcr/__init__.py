# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Framework-neutral KV Cache Runner."""

from .api import (
    DURATION_METRIC,
    KVCR,
    STATE_METRIC,
    TRANSFER_BLOCKS_METRIC,
    TRANSFER_BYTES_METRIC,
    KVCRBindings,
)
from .control_channels import (
    KVCRGuardProtocolError,
    KVCRMsgFramingError,
    KVCRServiceError,
    KVCRSocketError,
)
from .guard_protocol import (
    KVCRClient,
    KVCRPoolHold,
)

__all__ = [
    "DURATION_METRIC",
    "KVCR",
    "STATE_METRIC",
    "TRANSFER_BLOCKS_METRIC",
    "TRANSFER_BYTES_METRIC",
    "KVCRBindings",
    "KVCRClient",
    "KVCRMsgFramingError",
    "KVCRGuardProtocolError",
    "KVCRPoolHold",
    "KVCRServiceError",
    "KVCRSocketError",
]
