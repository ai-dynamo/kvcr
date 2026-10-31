<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo hint contract fixture

`dynamo_decode_to_prefill_hint.json` uses the wire shape emitted by Dynamo's
[`KvHintAction::fetch`](https://github.com/ai-dynamo/dynamo/blob/main/lib/kv-router/src/kv_hints.rs).
It represents one advisory, request-driven copy from a decode source to a
prefill target. No new KVCR action or transport is needed for these roles.

The hashes are opaque unsigned 64-bit test values, including zero and values
above the signed 64-bit range. They represent a complete root-aligned source
prefix. A target that already has some of this prefix still receives the entire
array; KVCR determines which keys need remote retrieval. The fixture contains
no workload, model, memory address, or source allocation metadata.

`test_dynamo_hint_contract.py` exercises this envelope through parsing and the
public KVCR APIs with fake NIXL agents. Existing remote source and target tests
cover partial completion, failure, cancellation, and transfer quiescence.
