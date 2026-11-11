# Remote transfer outcomes: guidelines, states and mapping

This document defines how KVCR reports the outcome of a remote G2 transfer to
the four parties that see it.

1. [Who sees what](#1-who-sees-what)
2. [Guidelines and why](#2-guidelines-and-why)
3. [Final states](#3-final-states)
4. [Mapping reasons to states](#4-mapping-reasons-to-states)
5. [Reporting per channel](#5-reporting-per-channel)

---

## 1. Who sees what

A remote transfer involves two independent workers. The **target** asks for
blocks; the **source** writes the ones it can. Four parties observe the result,
through four channels, and each can act on it in a different way.

| Party | Channel | What they can do with the outcome |
|---|---|---|
| Target worker | `write_done.reason` and `write_done.unserved` from the source | Remember a miss, drop the route, or re-exchange metadata |
| Caller (vLLM/SGLang connector) | `OpEntryResult.status`, with `OpEntryResult.reason` for detail | Recompute the block, or raise |
| Operator | `kvcr_duration_seconds{result}`, `kvcr_unserved_blocks{reason}`, WARNING/ERROR logs | Fix placement, fix configuration, or investigate a fault |
| Developer | DEBUG events, NVTX ranges | Trace one operation end to end |

The amount of detail each party needs is very different. A developer wants the
exact cause. A caller wants a single bit, plus "is it worth retrying". The
guidelines below exist to give each party *their* level of detail and nothing
more.

---

## 2. Guidelines and why

### G1. The status describes the operation, not the reporting worker

> `OpEntryStatus` says what happened to the block: delivered, not available,
> timed out, interrupted, or broken by an error. It does not say whose fault
> that was.

**Example.** Target T asks source S for 8 blocks. S holds them, but its
framework `request_pin` callback raises. S writes nothing and reports the
blocks as `pin_failed`. Something unexpected broke, so both sides call this a
failure: `source_write{result="failed"}` on S, `FAILED` with reason
`pin_failed` to T's caller, `remote_deliver{result="failed"}` on T. The
*reason* is what tells the T operator to look at S rather than at T. Hiding the
error behind a softer status on T would make a broken peer indistinguishable
from a cache miss.

### G2. The status answers the caller's question

> The public status answers one question: *did the block arrive, and if not, is
> there anything to do other than recompute it?*

| Status | What the caller does |
|---|---|
| `SUCCESS` | Use it |
| `UNAVAILABLE` | Recompute, silently; an expected miss or no usable route |
| `TIMEOUT` | Recompute; the deadline passed |
| `ABORTED` | Recompute; the operation was cancelled or interrupted |
| `FAILED` | Recompute; an unexpected error, on either worker |

**Example.** Whether a block was missing on the source, was already known to be
missing from an earlier pull for the same request, or had no route at all, the
caller does the same thing: recompute. Three public statuses for one caller
action is noise in the API. One status, `UNAVAILABLE`, says all the caller
needs; the reason field keeps the three apart for anyone who wants to know.

### G3. Reason is detail, not class

> One reason vocabulary (`kvcr.types.Reason`, lower-cased) is used on the wire,
> in `OpEntryResult.reason`, in counter labels, in DEBUG logs and in NVTX. It
> never widens the public status set.

**Example.** `kvcr_unserved_blocks{scope="remote_deliver", reason="pin_failed"}`
tells an operator precisely why blocks were not delivered. The same block's
result is `FAILED` with `reason="pin_failed"`. Detail and class are separate
axes, and a connector that only reads `.success` sees nothing new.

### G4. One reason, one status, wherever it is reported

> A known reason maps to the same status whether the source reports it for one
> block in `write_done.unserved` or for the whole write in `write_done.reason`.
> Only a block whose cause is unspecified falls back to `UNAVAILABLE`.

**Example.** A source serves 5 of 8 blocks and reports the other 3 as
`pin_failed`. Those 3 are `FAILED` with reason `pin_failed`, exactly as they
would be if the whole write had failed with that reason. The 5 delivered blocks
are `SUCCESS`; the batch carries both. A block the source left out without
saying why is `UNAVAILABLE` with no reason.

### G5. Non-faults are never shown as failures

> Words like `failed`, `error`, `rejected` and `invalid` are reserved for
> unexpected errors. Expected outcomes are named by what happened: `miss`,
> `unavailable`, `timeout`, `aborted`.

**Example.** Previously a source that holds none of the requested blocks answers
`success=True` with an empty block list, yet records
`source_write{result="failed"}`, and the target records
`remote_deliver{result="failed"}` and returns `FAILED` for every key. A perfectly
normal cache miss shows up as a failure on two workers and in the caller's
results. Under G5 this is `miss` on both sides and `UNAVAILABLE` to the caller.

---

## 3. Final states

### 3.1 Public result (`OpEntryResult`)

| Status | Meaning for the caller |
|---|---|
| `SUCCESS` | Delivered |
| `DROPPED` | Dropped by the placement policy |
| `UNAVAILABLE` | Expected miss or no usable route; recompute |
| `TIMEOUT` | The deadline passed |
| `ABORTED` | Cancelled or interrupted (shutdown, route change, peer restart) |
| `FAILED` | An unexpected error, on this worker or the peer |

`reason` is the lower-cased `Reason` name behind the status, or `None` when the
cause is unspecified. `.success` is unchanged. Connectors need only `.success`,
or `status is UNAVAILABLE` → recompute silently, anything else → log with
`reason`.

Local DRAM and disk backends currently return only `SUCCESS`, `DROPPED` and
`FAILED` without a reason; the same guidelines apply to them and are a
follow-up.

### 3.2 Duration result labels (`kvcr_duration_seconds{result}`)

`success`, `partial`, `miss`, `unavailable`, `timeout`, `aborted`, `failed`.

The label is the status of the operation's result, with `miss` split out of
`unavailable` because operators read the miss rate as a placement signal.

### 3.3 Wire reasons

The 32 `kvcr.types.Reason` names, lower-cased, are the single reason vocabulary
on the wire (`write_done.reason`, `write_done.unserved`), in
`OpEntryResult.reason`, in counter labels, in DEBUG logs and in NVTX. A target
accepts only these names from a peer; anything else is `unknown` for a block
and `remote_failure` for a write.

### 3.4 NVTX statuses

Unchanged codes 1–9, plus `10 MISS`: the peer answered but held none of the
blocks. `MISS` is derived from the reasons, never from "zero blocks completed".

---

## 4. Mapping reasons to states

### 4.1 The table

| Reason | Detected by | Status | Label | Target action |
|---|---|---|---|---|
| `not_resident` | source | `UNAVAILABLE` | `miss` | Remember as missing for this request |
| `remembered_miss` | target | `UNAVAILABLE` | `miss` | Skip without asking the source |
| `hint_unavailable`, `hint_invalidated` | target | `UNAVAILABLE` | `unavailable` | Nothing to pull |
| unspecified (block left out, no reason) | source | `UNAVAILABLE`, no reason | `unavailable` | Remember as missing |
| `deadline`, `source_stalled` | either | `TIMEOUT` | `timeout` | Drop the route for this request |
| `cancelled`, `stopping`, `shutdown`, `route_changed` | either | `ABORTED` | `aborted` | Drop the route |
| `refused`, `source_restarted`, `metadata_required` | source, in the control reply | `ABORTED` | `aborted` | Re-exchange metadata |
| `pin_failed`, `layout_mismatch` | source | `FAILED` | `failed` | Remember as missing; keep the route |
| `create_error`, `transfer_failed`, `setup_failed`, `malformed` | source | `FAILED` | `failed` | Drop the route |
| `remote_failure` (peer failed without a usable reason) | target | `FAILED` | `failed` | Drop the route |
| `control_error`, `invalid_notification` | target | `FAILED` | `failed` + WARNING | Drop the route |

The source labels its own `source_write` with the same rule applied to its own
reason: `miss` for a write with nothing resident, `partial` when it served a
subset, `failed` for its own pin, layout and NIXL errors.

### 4.2 Why each group lands where it does

**Misses (`not_resident`, `remembered_miss`).** A cache miss is the most common
outcome of a best-effort fetch. The caller recomputes; the operator reads the
`miss` rate as a placement/hint-quality signal. The distinction between a fresh
miss and a remembered one only matters when tracing one request, so it lives in
the reason.

**No route (`hint_unavailable`, `hint_invalidated`).** The target never contacted
a source. Same caller action as a miss, but the operator's fix is different:
hint plumbing rather than placement. That is why the label stays `unavailable`
rather than `miss`, while the status is the same.

**Unspecified.** An older or terse source left a block out without saying why.
Nothing is known to have broken, so the block is `UNAVAILABLE` with no reason.

**Timeouts (`deadline`, `source_stalled`).** Not a bug, but operationally
significant: operators alert on `timeout` separately from `failed`. The source's
pin-wait timeout and the target's quarantine keep their WARNING/ERROR logs
because memory ownership is in doubt in the latter case.

**Interruptions (`cancelled`, `stopping`, `shutdown`, `route_changed`).** KVCR
or the caller gave up on purpose. Nothing to fix; DEBUG only.

**Peer refusals (`refused`, `source_restarted`, `metadata_required`).** A source
that restarted does not know the target's memory yet and says "not now". The
target recovers automatically by re-exchanging metadata. A routine restart is an
interruption, not an error, so it is `ABORTED`. A refusal with an unrecognised
reason is still a refusal.

**Source errors (`pin_failed`, `layout_mismatch`, `create_error`,
`transfer_failed`, `setup_failed`, `malformed`).** Unexpected errors on the
source: its framework callback failed, the two workers disagree on block layout,
or NIXL could not create or run the transfer. They are `FAILED` on both sides
(G1); the reason tells the target operator which peer to look at. The source
logs a WARNING with the detail.

**Target errors (`control_error`, `invalid_notification`).** The target could
not send a control message, or received a notification it cannot parse. These
are the target's own faults and the only remote outcomes it logs at WARNING.

**Unclassifiable (`remote_failure`).** The peer said the write failed but gave
no usable reason. An unexplained error is still an error: `FAILED`, with the
reason showing it was unclassified.

### 4.3 Zero-block and partial results

Each block keeps its own result. The operation's duration label comes from the
most common reason the source gave when nothing was delivered:

| Situation | Source label | Target label | Caller results |
|---|---|---|---|
| 8 requested, 8 `not_resident` | `miss` | `miss` | 8 × `UNAVAILABLE`/`not_resident` |
| 8 requested, 8 `pin_failed` | `failed` | `failed` | 8 × `FAILED`/`pin_failed` |
| 8 requested, 5 written, 3 `not_resident` | `partial` | `partial` | 5 × `SUCCESS`, 3 × `UNAVAILABLE`/`not_resident` |
| 8 requested, 5 written, 2 `not_resident`, 1 `pin_failed` | `partial` | `partial` | 5 × `SUCCESS`, 2 × `UNAVAILABLE`, 1 × `FAILED`/`pin_failed` |
| 8 requested, transfer creation failed | `failed` | `failed` | 8 × `FAILED`/`create_error` |
| 8 requested, target deadline passed | `timeout` (if it noticed) | `timeout` | 8 × `TIMEOUT`/`deadline` |

---

## 5. Reporting per channel

| Channel | Source | Target |
|---|---|---|
| `OpEntryResult` | — | One immutable `(status, reason)` result per known reason, shared across blocks |
| `kvcr_duration_seconds{result}` | Labels from §3.2, from the source's own reason | Same labels, from the delivery outcome (§4.1) |
| `kvcr_unserved_blocks{scope, reason}` | Not recorded | `scope` ∈ {`remote_deliver`, `remote_fetch`}: every requested block not delivered, counted once, by reason |
| WARNING logs | Own faults only: pin callback errors, layout mismatch (one line per operation), NIXL create/transfer errors, malformed requests | Own faults only: control send/receive failures, invalid notifications; quarantine at ERROR |
| DEBUG `*_transfer_completed` | `requested=`, `blocks=`, `result=`, `unserved=reason:count,…` | Same, plus `source=` |
| NVTX | Own-side status and reasons | Own-side status and reasons |
| Wire | `write_done.reason`, `write_done.unserved: {reason: [indices]}` | Accepts known reason names only |

**Why the counter lives on the target only.** The target is the side that
requested the blocks, sees every one of them exactly once, and receives the
source's per-block reasons in `write_done`. A source-side copy of the counter
counted the same blocks a second time under a different scope, could not see
blocks the source was never asked about (early rejections, remembered misses),
and added nothing an operator acts on that `source_write{result}` and the
source WARNINGs do not already give.
