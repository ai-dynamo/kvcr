<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Remote delivery NVTX events

Install the `profiling` extra. `KVCR_NVTX_LEVEL=off|low|medium` selects
library-side detail; the default is low when the optional binding is available.
Annotations do not depend on whether Nsight is attached. See [profiling](profiling.md)
for capture commands and the pin schema.

## Source transfers

`nixl.write.submit` is a same-thread synchronous scope. `nixl.write.posted`
reports the native posting result, including rejection and ambiguity.
`nixl.done_observed` records the first observed native DONE, including synchronous
DONE returned by posting. It is distinct from `nixl.write.released` (successful
handle release) and `source.write.completed` (logical source operation result).
A prior error or cancellation can make the logical result fail even after DONE.
The status on the release event reflects the transfer outcome, not a release
failure. Release failures emit `nixl.write.release_retry` at medium detail.

`source.write` is a start/end range for the progress-owned source operation.
`nixl.write` separately covers native handle creation through successful release.
Observed DONE, timeout and a failed release attempt do not end the native range.
Unresolved shutdown leaves it open with an unresolved mark; successful source
cleanup closes the source range with a terminal result mark.

`nixl.write.error`, `source.write.cancel_requested`, and
`source.write.shutdown_unresolved` preserve the error, timeout/cancellation, and
unresolved ownership boundaries. Elapsed time does not prove DMA quiescence.
`source.write.refused` identifies a stale route before posting. No native numeric
error is inferred from exception text; `native_error_known=0` means unavailable.

Schema 3 stores immutable lifecycle fields once in `request.context`. Compact
events carry `(instance_hi, instance_lo, trace_id)`, status/reason and only the
changing fields supplied by that transition. Reconstruct context before joining
events. Pin context uses `source.pin.context`; waiter marks override its source
operation/target handle because a physical pin can have several waiters.
Older schema 1/2 captures carry these fields directly on each event.
Across workers use `(target_agent_hi, target_agent_lo,
target_incarnation_hi, target_incarnation_lo, op_handle)`; incarnation is taken
from existing control metadata, without extending the wire protocol. Zero
identity fields mean unavailable. Handles remain signed, including local fills.
Transfer IDs are local to the source instance. Unknown counts/bytes are `-1`.

At medium detail, `source_tier`/`destination_tier` describe memory ownership: 0 unknown, 1 framework,
2 KVCR-owned, 3 mixed. `source_memory`/`destination_memory` describe a verified
registration: 0 unknown, 1 DRAM, 2 VRAM, 3 FILE. Ownership alone does not identify
the physical memory tier; remote registration facts are left unknown when absent.
Low skips descriptor scans and optional byte calculations; uncomputed byte
counts remain `-1` and classification fields remain unknown.

Request identities use a stable 128-bit BLAKE2b digest of the complete UTF-8
string. The digest remains available at low detail. At medium, `request.context`
also carries `request_display_known=1` and a display prefix as an unsigned byte array,
explicit byte length (up to 256), and a truncation flag. Decode exactly `length`
bytes as UTF-8 when the display is known; a truncated prefix may end inside a character. Join by the digest,
never by the display prefix. Empty strings and unavailable IDs are distinct via
`request_known`. Request strings never become registered event names. Session,
parent-session and framework utilization remain explicitly unknown because the
current bindings do not provide them.

## Target and caller

`hint.submitted`, `hint.used`, `hint.replaced`, `hint.rejected`, and
`hint.discarded` describe the request-scoped hint. Each submission has a separate
trace identity, even if the framework reuses a request string; target events
identify the consumed hint via `hint_trace_id` in the same tracer instance.

`op.deliver` covers the synchronous public call. `op.dispatched` describes its
local-G2, G3 and remote block counts. `target.queued` and
`target.start_write.enqueued` distinguish local progress queueing from successful
control-channel enqueueing; enqueueing is not acknowledgement by the source.
`target.queued.rejected` closes an unaccepted target range when dependency
setup or progress submission raises, before remote work can begin.
`target.write_done.received` is the validated logical result observed by target
progress. `target.main.consume` is a synchronous main-thread consumption scope.
`op.completion_queued` waits for **all** branches of a mixed-tier operation;
`op.completion_returned` records the completion batch returned by `poll_completed`.
These boundaries expose delay between progress completion and caller polling.

`op.deliver.lifecycle` is an independent start/end range from dispatch through
caller-visible completion return. Rejected dispatch and successful shutdown with
an unreturned result also close it, with their distinct status marks.
`target.remote` spans target queueing through a validated terminal result or
confirmed quiescence. It normally starts on the main thread and ends on progress.
It remains open during timeout/quarantine even if a failed caller result has
already been returned. These process-local range IDs never replace payload
`trace_id` or the existing cross-worker agent/incarnation/operation join.

`target.cancel_requested`, `target.quarantined`, `target.quiesced`, and
`target.shutdown_unresolved` describe uncertainty and eventual proof of safe
cleanup. A late success after cancellation cannot create a second successful
caller completion. `op.shutdown_unreturned` records operations whose results
were not returned before successful shutdown; it is not a DMA-state assertion.

Source and target lifecycle events use the `remote_deliver` category. Incidental
remote local-fill events retain negative handles and `local_fill=1`; this is not
full public-fetch fan-out instrumentation. Source request IDs remain unknown;
the existing target identity/handle links to the target's request mapping.

Additional status codes: 7 rejected, 8 ambiguous, 9 unresolved. Additional reason
codes: 9 submit rejected, 10 submit ambiguous, 11 creation error, 12 progress
error, 13 release error, 14 control error, 15 invalid notification, 16 remote
failure, 17 route changed, 18 source stalled, 19 hint unavailable, 20 hint conflict.
Unknown remote/framework causes remain unknown rather than guessed.

Nonempty source and pin requests with zero completed blocks report FAILED;
positive incomplete results report PARTIAL. This is a diagnostic convention and
does not change control notifications or transfer ownership. See the
[identity producer map](nvtx-context.md) for inference versus framework pin IDs
and the unavailable session/utilization input seams.
