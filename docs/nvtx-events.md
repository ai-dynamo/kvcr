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

`nixl.write.error`, `source.write.cancel_requested`, and
`source.write.shutdown_unresolved` preserve the error, timeout/cancellation, and
unresolved ownership boundaries. Elapsed time does not prove DMA quiescence.
`source.write.refused` identifies a stale route before posting. No native numeric
error is inferred from exception text; `native_error_known=0` means unavailable.

Schema 2 events use `(instance_hi, instance_lo, trace_id)` for a local lifecycle.
Join schema 1 pin/waiter records using `(instance_hi, instance_lo, source_op_id)`.
Across workers use `(target_agent_hi, target_agent_lo,
target_incarnation_hi, target_incarnation_lo, op_handle)`; incarnation is taken
from existing control metadata, without extending the wire protocol. Zero
identity fields mean unavailable. Handles remain signed, including local fills.
Transfer IDs are local to the source instance. Unknown counts/bytes are `-1`.

`source_tier`/`destination_tier` describe memory ownership: 0 unknown, 1 framework,
2 KVCR-owned, 3 mixed. `source_memory`/`destination_memory` describe a verified
registration: 0 unknown, 1 DRAM, 2 VRAM, 3 FILE. Ownership alone does not identify
the physical memory tier; remote registration facts are left unknown when absent.

Request identities use a stable 128-bit BLAKE2b digest of the complete UTF-8
string. `request.context` carries a display prefix as an unsigned byte array,
explicit byte length (up to 256), and a truncation flag. Decode exactly `length`
bytes as UTF-8; a truncated prefix may end inside a character. Join by the digest,
never by the display prefix. Empty strings and unavailable IDs are distinct via
`request_known`. Request strings never become registered event names. Session,
parent-session and framework utilization remain explicitly unknown because the
current bindings do not provide them.
