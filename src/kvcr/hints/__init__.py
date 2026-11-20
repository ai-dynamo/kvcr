# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KV hint parsing, dispatch, and execution.

``hint_types`` defines wire messages and parsed hint types.
``parser`` validates envelopes and constructs typed hints.
``dispatcher`` registers supported actions, routes hints, and reports results.
``hint_handler`` stores request-scoped advice and runs independent-scoped operations.
``results`` defines hint statuses and failure reports.
"""

from .parser import ROUTER_HINT_CAPABILITIES, ROUTER_HINT_KEY

__all__ = ["ROUTER_HINT_CAPABILITIES", "ROUTER_HINT_KEY"]
