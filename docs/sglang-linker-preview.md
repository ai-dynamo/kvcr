<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang direct-Linker integration preview

This boundary note is **not a runnable customer quick start**. Use the separate
[HiCache quick start](quick-start-sglang.md) for that path. Do not enable both
modes on one engine or assume their storage formats are interchangeable.

HiCache restores KVCR DRAM through SGLang host staging into HBM. The direct
Linker restores into device pools without a HiCache host tier. Linker options
are `--unified-cache-external-linker-backend kvcr`, `dram_size_gb` (GiB per rank),
`advertise_host` and `nixl_port`; HiCache uses different selectors/configuration.

## Review stack and ownership

- [SGLang #45646](https://github.com/sgl-project/sglang/pull/45646): shared contracts.
- [SGLang #45648](https://github.com/sgl-project/sglang/pull/45648): Linker lifetime/inventory contracts.
- [SGLang #45650](https://github.com/sgl-project/sglang/pull/45650): direct-device Linker.
- [Dynamo #16117](https://github.com/ai-dynamo/dynamo/pull/16117): router-hint forwarding.
- [Dynamo #16827](https://github.com/ai-dynamo/dynamo/pull/16827): ownership-aware ingress/lifecycle.

Linker emits `ownership="kvcr"` inventory. Ordinary SGLang KV event ingress
rejects that ownership domain; hint forwarding alone is insufficient. #16827's
opt-in TP1 aggregated path requires an explicit valid
`--kvcr-state-agent-owner` identity, not a boolean. Do not use it with HiCache.
Neither integration exposes unpinned SGLang pages as peer sources.

Recorded source-checkout acceptance used Linker
`20cdafc07c42f4c78767e41ad34058f627a8da13`, Dynamo
`a310f33e9b55e90a8da0ef742e0ca4cb3491bec2`, Qwen3-32B-FP8 revision
`aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`, and two same-host B200 TP1 workers.
This is not public-image, cross-host, TP>1, or Guard replacement validation.

## Outstanding behavior

The current Linker draft still backs externally restored prefixes up into
destination DRAM. Bounded deposits prevent an oversized-copy timeout, but do
not restore the original no-retain behavior. Suppressing redundant backup
needs successful-restore provenance through post-forward insertion: it must
not advertise nonexistent destination-owned DRAM or skip newly computed KV.
No-retain and multi-source/replication policy remain explicit follow-up work.

A public Linker recipe still needs its own pinned image, owner/discovery
configuration and complete two-worker cold/restore/output acceptance. Adding
a Linker flag to the HiCache image does not establish those contracts. Cache
hits alone do not replace terminal native transfer and cold-output evidence.
