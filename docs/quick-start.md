<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVCR Quick Starts

Choose the inference engine you want to evaluate:

- [vLLM + Dynamo](quick-start-vllm.md): the existing public-preview integration.
- [SGLang HiCache + Dynamo](quick-start-sglang.md): pinned public-source build,
  automatic two-worker launch and fail-closed peer-restore verification.

Each guide states its own validation status and hardware/version boundary.
The engines use different adapters; do not mix their images or configuration.
For API development and Guard/service workflows, see the
[developer guide](dev-guide.md).
