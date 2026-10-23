<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Remote delivery NVTX events

See [profiling](profiling.md) for installation, detail levels and capture commands.

## Source transfers

`nixl.write.submit` is a same-thread synchronous scope including native handle creation. `nixl.write.posted` reports the native posting result, including rejection and ambiguity. `nixl.done_observed` records the first observed native DONE, including synchronous DONE returned by posting. It is distinct from `nixl.write.released` (successful handle release) and `source.write.completed` (logical source operation result). A prior error or cancellation can make the logical result fail even after DONE. The status on the release event reflects the transfer outcome, not a release failure. Release failures emit `nixl.write.release_retry` at medium detail.

`source.write` is a start/end range for the progress-owned source operation. `nixl.write` starts after successful handle creation, before posting, and ends after successful release. Observed DONE, timeout and a failed release attempt do not end the native range. Unresolved shutdown leaves it open with an unresolved mark; successful source cleanup closes the source range with a terminal result mark.

Source/native ranges start and end on the progress thread. Before another thread can complete a range, its owner must finish starting it and then publish the trace/work. The helper synchronizes competing ends, not overlapping start/end.

`nixl.write.error`, `source.write.cancel_requested`, and `source.write.shutdown_unresolved` preserve the error, timeout/cancellation, and unresolved ownership boundaries. Elapsed time does not prove DMA quiescence. `source.write.refused` identifies a stale route before posting. No native numeric error is inferred from exception text; `native_error_known=0` means unavailable. Source completion also preserves deadline and stale-route causes detected while pinning is pending; a known deadline is retained if the route has also changed.

Schema 3 stores immutable lifecycle fields once in `request.context`. Compact events carry `(instance_hi, instance_lo, trace_id)`, status/reason and only the changing fields supplied by that transition. Reconstruct context before joining events. Pin context uses `source.pin.context`; waiter marks override its source operation/target handle because a physical pin can have several waiters. Across workers use `(target_agent_hi, target_agent_lo, target_incarnation_hi, target_incarnation_lo, op_handle)`; incarnation is taken from existing control metadata, without extending the wire protocol. Zero identity fields mean unavailable. Handles remain signed, including local fills. Transfer IDs are local to the source instance. Unknown counts/bytes are `-1`.

At medium detail, `source_tier`/`destination_tier` describe memory ownership: 0 unknown, 1 framework, 2 KVCR-owned, 3 mixed. `source_memory`/`destination_memory` describe a verified registration: 0 unknown, 1 DRAM, 2 VRAM, 3 FILE. Ownership alone does not identify the physical memory tier; remote registration facts are left unknown when absent. Low skips descriptor scans and optional byte calculations; uncomputed byte counts remain `-1` and classification fields remain unknown.

Request identities use a stable 128-bit BLAKE2b digest of the complete UTF-8 string. The digest remains available at low detail. At medium, `request.context` also carries `request_display_known=1` and a display prefix as an unsigned byte array, explicit byte length (up to 256), and a truncation flag. Decode exactly `length` bytes as UTF-8 when the display is known; a truncated prefix may end inside a character. Join by the digest, never by the display prefix. Empty strings and unavailable IDs are distinct via `request_known`. Request strings never become registered event names. Session, parent-session and framework utilization remain explicitly unknown because the current bindings do not provide them. Only worker agent/incarnation identities use a bounded cache; request identities are hashed directly to avoid request churn evicting those reusable entries.

Nonempty source and pin requests with zero completed blocks report FAILED; positive incomplete results report PARTIAL. This is a diagnostic convention and does not change control notifications or transfer ownership.
