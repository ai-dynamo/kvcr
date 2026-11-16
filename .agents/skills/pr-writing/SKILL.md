---
name: pr-writing
description: Use when preparing, opening, or updating a pull request (PR) for ai-dynamo/kvcr, including drafting or revising its title and description.
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVCR PR writing

Write the smallest standalone explanation that tells a reviewer what changes, why it matters, and any consequential limits.

## Rules

1. **Write from the current diff.** Read the changes against the intended base and enough surrounding code to understand their effect. If only verified change notes are available, use those without inventing details. Rewrite stale titles and descriptions when scope changes. Describe the final implemented behavior, its rationale, and concrete limitations. Omit investigation progress, review status, and proposals discussed but not implemented.
2. **Give the title a concrete action and subject.** Prefer titles such as "Perf: avoid quadratic eviction scans across cache pools" over vague "improvements," "cleanup," or "updates." Prefer a category prefix when it clearly fits, such as `Perf:`, `docs:`, `fix:`, `feat:`, or `refactor:`. Use a plain action title when no category fits naturally. There is no required Conventional Commits format.
3. **Lead with the problem or capability and resulting behavior.** For fixes, explain the failure, its cause, and why the change resolves it. Include a minimal event sequence when ordering across peers or threads is essential. For features, explain what callers can now do. Write for a reviewer who has not seen the development conversation.
4. **Use implementation details to explain the benefit.** Connect cached metadata to avoided scans, for example. Omit file inventories, helper lists, and incidental restructuring.
5. **Keep consequential semantics and limits.** State changed defaults, compatibility requirements, migration steps, resource-lifetime guarantees, and known limitations when they affect assessment or use. Avoid generic lists of unchanged behavior.
6. **Use short prose by default.** Usually one to three short paragraphs; use bullets for distinct consequences. No mandatory headings or word limits. A self-explanatory maintenance change may need little or no body. Omit tool-generated promotional footers. Omit commit counts, commit order, review-response history, and abandoned approaches unless they explain a surviving tradeoff.
7. **Omit routine testing boilerplate.** Skip command dumps, passing counts, lint results, and "tests added" filler. Mention coverage when it explains a meaningful regression or tests are the change. Include validation gaps or unusual hardware evidence only when they affect a claim or risk in this PR. Distinguish tests added from tests actually run.
8. **Make performance claims precise.** Explain the work avoided. For measured gains, give the workload/configuration and before/after results in a short paragraph or small table. Include run count and variability when they affect the conclusion. Keep only the results needed to assess the claim. Fewer allocations or validations do not establish an end-to-end speedup. Do not invent results or add unrelated benchmark disclaimers.

## Examples

### Bug fix (adapted from [#108](https://github.com/ai-dynamo/kvcr/pull/108))

Title: fix: fall back locally when remote hints become unusable

A request that selected a remote source could report `MISS` and fail delivery after its hint became unusable, even when a complete local copy was available.

Allow `query()` and `deliver()` to fall back to that complete local copy. Keep remote selection while the hint is usable; partial or excluded-piece copies do not qualify for fallback.

### Compatibility ([#95](https://github.com/ai-dynamo/kvcr/pull/95))

Title: Only support msgspec KV hint envelopes

Require msgspec envelopes and actions in `submit_hint()`, removing dictionary compatibility handling. Callers should pass their hint objects directly instead of converting them with `msgspec.to_builtins()`.

Breaking change: callers submitting dictionary envelopes must update.
