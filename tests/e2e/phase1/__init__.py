# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KVCR test plan v7, Phase 1: Guard-mode critical path TC-01 -> TC-04.

The package runs inside the integration image on one node with two GPUs. It
launches a KVCR-disabled reference engine, records the reference generated
token IDs, then launches a Dynamo KV router and two independent KVCR-enabled
workers (one GPU each) that claim Guard-owned pools from per-worker KVCR
services. Every test case compares generated token IDs with the reference and
requires path evidence from worker and service logs.
"""
