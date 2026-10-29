# Compact transfer and Guard residency capture

Compact capture is opt-in and independent of NVTX. It aggregates query/progress
counters and records transfer lifecycles and exact-key Guard residency events.
It does not change the source selection or memory-lifetime policy.

## Enable capture

Set these variables before constructing KVCR in each worker/Guard process:

```bash
export KVCR_TELEMETRY_DIR=/path/to/capture
export KVCR_SOURCE_ARCHIVE_SHA256=<sha256-of-the-packaged-source-archive>
# Optional provenance manifest and live health snapshots:
export KVCR_SOURCE_MANIFEST=/path/to/source-manifest.json
export KVCR_TELEMETRY_HEALTH_DIR=/path/to/capture-health
```

The manifest may supply `source_archive_sha256`; if both it and the environment
variable supply a value, they must agree. Provenance is required for a valid
capture. The validator checks consistency, not archive authenticity; the caller
must verify the archive independently. `RUN_DIR` alone does not enable capture.

Use `RemoteFWDramOptions(no_pinning_mode=True)` for progress-side Guard READY
subset source operations. Source-operation capture applies to this mode; target
operations, residency events and aggregate counters are also observed. The native
default remains `no_pinning_mode=False`. Successful Guard events avoid per-event
INFO logging; enabled capture retains their operation outcomes and counters.

## Files, counters and clock boundaries

Each process writes `source-<pid>-<session>.jsonl` (schema 2) and its referenced
`keys-<pid>-<session>.bin` partner. Sessions distinguish PID reuse. JSON
`operation_start` pairs with one `source_nixl_terminal` or `target_terminal` using
`operation_key`. Terminals contain actual successful bytes/blocks, requested block
counts, partial/failed outcomes, failure reasons, request correlation, native
transfer statistics and host timing landmarks. Failed transfers do not impute
successful payload bytes.

The monotonic `timing_ns` landmarks distinguish:

1. Source receipt and READY-subset claim start/end.
2. Native submit start/end and the first observed DONE.
3. Successful native handle release, the existing core source-write metric
   boundary, and cleanup completion.
4. Terminal enqueue time after cleanup.

Missing stages remain absent. First observed DONE is a host observation, not the
hardware completion timestamp. `source_operation_seconds` starts at the existing
operation body's `started_at` and ends after cleanup. It excludes receipt before
that body starts; receipt-to-end can be calculated from the landmarks. The
`core_source_write_ns` landmark preserves the earlier core metric boundary before
claim cleanup. Native durations come from SDK microseconds converted to seconds.
These intervals are inclusive and must not be summed as independent costs.

Handling thread CPU deduplicates same-thread nesting and sums each handling
thread's own intervals. Wall time minus thread CPU includes scheduling and poll
gaps; it is not a measurement of GIL wait alone.

`query_outcomes` contains aggregate status/tier counts and a denominator, without
per-query JSON. `progress_schedule` separates observed work from in-flight
occupancy, records active/idle wall and thread CPU totals, and supplies seven-bin
wall/CPU/start-gap/end-to-start-gap histograms. Upper bounds are 10 us, 100 us,
1 ms, 10 ms, 100 ms, 1 s and infinity. Both aggregates include epoch-wall
`interval_start_ns`/`interval_end_ns`; flush time alone cannot assign a phase.

## Exact-key residency lineage

The binary journal starts with `KVCRKEY1`, followed by frames with big-endian
uint32 payload length and CRC32, then a MessagePack map. It preserves exact binary
keys. Events include `admitted`, `ready`, `withdraw`, `source_get` and `target_get`.
Generations distinguish residency incarnations even when a key or slot is reused.

READY and withdrawal sequence numbers/timestamps are sampled under the export
lock; serialization happens after unlocking. Source GETs include requested keys,
READY indices, generations, per-key miss classifications and an export-sequence
snapshot. Correlate these with preceding withdrawals to distinguish capacity
eviction, policy retirement, fill discard and shutdown. Use export sequence rather
than cross-thread file order for transition ordering. Withdrawal removes export
visibility and does not necessarily mean the memory was already freed. Reader
counts sampled after unlock are observational.

## Finalization and validation

Quiesce all KVCR producers/cores before closing the process-wide writer. Process
exit handlers drain it automatically; an explicit collector can also close the
writer after all cores have stopped:

```python
from kvcr.compact_telemetry import validate_capture, writer

capture = writer()
if capture is not None:
    capture.close()
result = validate_capture("/path/to/capture", expected_identities=["worker-0"])
assert result["valid"]
```

`iter_key_journal` streams exact-key frames; `validate_key_journal` validates one
binary file. Full validation additionally checks JSON lifecycles, source proof,
paired-file identity/counts and expected identities. Missing/corrupt footers,
CRC/sequence errors, unmatched operations, overflow, dropped or unflushed records
and instrumentation failures invalidate evidence. A live health snapshot is not
a substitute for final validation.

Each producer thread has a 4 MiB staging buffer, capped at 32 threads per process.
Only the background flusher writes files. Overflow invalidates capture instead
of blocking transfer progress or truncating keys. Temporary serialization and one
copied flush batch add memory beyond staging buffers. Shutdown can overflow when
withdrawing many keys at once; check the final validity result.
Staging admission does not guarantee persistence: metadata added during final
framing can exceed the 4 MiB journal-frame limit. Such records are explicitly
dropped and invalidate the capture; written frame sequences remain contiguous.

The journal is not a complete physical allocation ledger: failed local fills that
never publish READY have no dedicated local-copy terminal. Lifecycle validation
uses memory proportional to total operations. CPU/fake-NIXL tests establish field
and lifetime contracts; they do not establish RDMA throughput or capture overhead.
