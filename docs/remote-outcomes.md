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
| Caller (vLLM/SGLang connector) | `OpEntryResult.status` | Recompute the block, or raise |
| Operator | `kvcr_duration_seconds{result}`, `kvcr_unserved_blocks{reason}`, WARNING/ERROR logs | Fix placement, fix configuration, or investigate a fault |
| Developer | DEBUG events, NVTX ranges | Trace one operation end to end |

The amount of detail each party needs is very different. A developer wants the
exact cause. A caller wants a single bit, plus "is it worth retrying". The
guidelines below exist to give each party *their* level of detail and nothing
more.

---

## 2. Guidelines and why

### G1. Each worker reports its own health

> A worker's `failed` means that worker, its framework or its transport broke.
> A peer's fault never appears as `failed` on this side.

**Example.** Target T asks source S for 8 blocks. S holds them, but its
framework `request_pin` callback raises. S writes nothing and reports the blocks
as `pin_failed`.

* On S this is a real fault: `source_write{result="failed"}` and a WARNING.
  The S operator should look at their framework.
* On T nothing broke. T did everything right and simply did not receive the
  blocks. If T also recorded `remote_deliver{result="failed"}`, the T operator
  would chase a bug in T that does not exist, and T's failure rate would track
  the health of every peer it talks to.

Remote fetches are best effort between independent workers. A target cannot fix
a source, so it must not be blamed for one.

### G2. The target reports the caller's experience

> The public status answers one question: *did the block arrive, and if not, is
> there anything to do other than recompute it?*

The caller has exactly four possible experiences:

| Experience | What the caller does |
|---|---|
| Delivered | Use it |
| Not deliverable (miss, no route, peer couldn't serve it) | Recompute, silently |
| Timed out | Recompute; the operator may want to know |
| Aborted (cancelled, shutdown, peer restarted) | Recompute; nothing to fix |
| Failed on this worker | Recompute; something here needs attention |

**Example.** Whether a block was missing on the source, was already known to be
missing from an earlier pull for the same request, or had no route at all, the
caller does the same thing: recompute. Three public statuses for one caller
action is noise in the API, and every connector that matches exhaustively has
to learn KVCR's routing internals to handle it. One status, `UNAVAILABLE`, says
all the caller needs.

### G3. Reason is detail, not class

> Fine-grained reasons travel on the wire, in counter labels, DEBUG logs and
> NVTX. They never widen the public status set and never change the class an
> outcome falls in.

**Example.** `kvcr_unserved_blocks{scope="remote_deliver", reason="pin_failed"}`
tells an operator precisely why blocks were not delivered. The same block's
public status is `UNAVAILABLE`, its duration label is `unavailable`, and the
WARNING about the pin fault appears on the source, where it belongs. Detail and
class are separate axes.

### G4. Where a reason arrives decides its class at the target

This is the operational rule that implements G1 and G2 on the target.

| The source reports it as | Meaning | Target class |
|---|---|---|
| A per-block entry in `write_done.unserved` (the source completed its job; these blocks were not deliverable) | Best-effort shortfall, whatever the cause on the source | `UNAVAILABLE` |
| An operation-level `write_done.reason` with `success=False` (the transfer did not happen) | Deadline, cancellation, or a fault on the source | `TIMEOUT` / `ABORTED` by reason |
| Nothing: the target itself could not send control or received garbage | A fault the target can see in itself | `FAILED` |

**Example.** A source's NIXL transfer creation fails. On the source this is
`failed` with a WARNING (G1). The target receives `success=False,
reason=create_error`. The transfer did not happen through no fault of the
target, so the target reports `ABORTED` with reason `create_error`, not
`FAILED`. The operator still sees `create_error` in the target's counter label
and can go to the source's logs.

### G5. Non-faults are never shown as failures

> Words like `failed`, `error`, `rejected` and `invalid` are reserved for faults.
> Normal outcomes are named by what happened: `miss`, `unavailable`, `timeout`,
> `aborted`.

**Example.** Previously a source that holds none of the requested blocks answers
`success=True` with an empty block list, yet records
`source_write{result="failed"}`, and the target records
`remote_deliver{result="failed"}` and returns `FAILED` for every key. A perfectly
normal cache miss shows up as a failure on two workers and in the caller's
results. Under G5 this is `miss` on both sides and `UNAVAILABLE` to the caller.

---

## 3. Final states

### 3.1 Public status (`OpEntryStatus`)

| Status | Meaning for the caller |
|---|---|
| `SUCCESS` | Delivered |
| `DROPPED` | Dropped by the placement policy |
| `UNAVAILABLE` | Not deliverable; nothing on this worker broke; recompute |
| `TIMEOUT` | The deadline passed |
| `ABORTED` | Given up deliberately, or the peer could not run the transfer |
| `FAILED` | A fault on this worker |

`.success` is unchanged. Connectors need only: `.success`, or
`status is UNAVAILABLE` → recompute silently, anything else → log.

Local DRAM and disk backends currently return only `SUCCESS`, `DROPPED` and
`FAILED`; the same guidelines apply to them and are a follow-up.

### 3.2 Duration result labels (`kvcr_duration_seconds{result}`)

`success`, `partial`, `miss`, `unavailable`, `timeout`, `aborted`, `failed`.

Each worker labels its own operations by its own view (G1). `failed` on any
worker means "look at this worker".

### 3.3 Wire reasons

The 32 `_nvtx.Reason` names, lower-cased, are the single reason vocabulary on
the wire (`write_done.reason`, `write_done.unserved`), in counter labels, in
DEBUG logs and in NVTX. A target accepts only these names from a peer; anything
else is `unknown`.

### 3.4 NVTX statuses

Unchanged codes 1–9, plus `10 MISS`: the peer answered but held none of the
blocks. Each side's NVTX status follows that side's own classification, and
`MISS` is derived from the reasons, never from "zero blocks completed".

---

## 4. Mapping reasons to states

### 4.1 The table

| Reason | Detected by | Source side (own view) | Target status | Target label | Target action |
|---|---|---|---|---|---|
| `not_resident` | source | `miss` | `UNAVAILABLE` | `miss` | Remember as missing for this request |
| `remembered_miss` | target | — | `UNAVAILABLE` | `miss` | Skip without asking the source |
| `hint_unavailable`, `hint_invalidated` | target | — | `UNAVAILABLE` | `unavailable` | Nothing to pull |
| `pin_failed` | source | `failed` + WARNING | `UNAVAILABLE` | `unavailable` | Remember as missing; keep the route |
| `layout_mismatch` | source | `failed` + WARNING | `UNAVAILABLE` | `unavailable` | Remember as missing; keep the route |
| `deadline`, `source_stalled` | either | `timeout` | `TIMEOUT` | `timeout` | Drop the route for this request |
| `cancelled`, `stopping`, `shutdown`, `route_changed` | either | `aborted` | `ABORTED` | `aborted` | Drop the route |
| `refused`, `source_restarted`, `metadata_required` | source, in the control reply | no operation is created | `ABORTED` | `aborted` | Re-exchange metadata |
| `create_error`, `transfer_failed`, `setup_failed`, `malformed` | source | `failed` + WARNING | `ABORTED` | `aborted` | Drop the route |
| `control_error`, `invalid_notification` | target | — | `FAILED` | `failed` + WARNING | Drop the route |
| `remote_failure`, `unknown` | target (peer sent no usable reason) | — | `ABORTED` | `aborted` | Drop the route |

### 4.2 Why each group lands where it does

**Misses (`not_resident`, `remembered_miss`).** A cache miss is the most common
outcome of a best-effort fetch. The caller recomputes; the operator reads the
`miss` rate as a placement/hint-quality signal. The distinction between a fresh
miss and a remembered one only matters when tracing one request, so it lives in
the reason label and NVTX.

**No route (`hint_unavailable`, `hint_invalidated`).** The target never contacted
a source. Same caller action as a miss, but the operator's fix is different:
hint plumbing rather than placement. That is why the label stays `unavailable`
rather than `miss`, while the status is the same.

**Peer could not serve (`pin_failed`, `layout_mismatch`).** Real faults *on the
source*: its framework callback failed, or the two workers disagree on block
layout. The source says so loudly (`failed`, WARNING). The target reports them
exactly like any other undeliverable block (G1, G4): `UNAVAILABLE`, label
`unavailable`, reason preserved. A target operator whose peers are misconfigured
sees `unavailable` rising with reason `layout_mismatch` and follows it to the
source.

**Timeouts (`deadline`, `source_stalled`).** Not a bug, but operationally
significant: operators alert on `timeout` separately from `failed`. The source's
pin-wait timeout and the target's quarantine keep their WARNING/ERROR logs
because memory ownership is in doubt in the latter case.

**Deliberate aborts (`cancelled`, `stopping`, `shutdown`, `route_changed`).**
KVCR or the caller gave up on purpose. Nothing to fix; DEBUG only.

**Peer refusals (`refused`, `source_restarted`, `metadata_required`).** A source
that restarted does not know the target's memory yet and says "not now". The
target recovers automatically by re-exchanging metadata. Reporting this as `FAILED` would make a routine restart look like a fault, so
it is `ABORTED`.

**Source-side transport faults (`create_error`, `transfer_failed`,
`setup_failed`, `malformed`).** Faults on the source (G1): `failed` + WARNING
there. At the target the transfer did not happen through no fault of its own
(G4): `ABORTED`, reason preserved.

**Target-side faults (`control_error`, `invalid_notification`).** The target
could not send a control message, or received a notification it cannot parse.
These are the only remote outcomes a target calls `failed`.

**Unclassifiable (`remote_failure`, `unknown`).** An older or misbehaving peer
sent no usable reason. The transfer did not happen; the target cannot blame
itself, so `ABORTED`, with the reason label showing it was unclassified.

### 4.3 Zero-block and partial results

A result with fewer blocks than requested takes its label from the dominant
unserved reason through the table above, on each side independently:

| Situation | Source label | Target label | Caller statuses |
|---|---|---|---|
| 8 requested, 8 `not_resident` | `miss` | `miss` | 8 × `UNAVAILABLE` |
| 8 requested, 8 `pin_failed` | `failed` | `unavailable` | 8 × `UNAVAILABLE` |
| 8 requested, 5 written, 3 `not_resident` | `partial` | `partial` | 5 × `SUCCESS`, 3 × `UNAVAILABLE` |
| 8 requested, transfer creation failed | `failed` | `aborted` | 8 × `ABORTED` |
| 8 requested, target deadline passed | `timeout` (if it noticed) | `timeout` | 8 × `TIMEOUT` |

---

## 5. Reporting per channel

| Channel | Source | Target |
|---|---|---|
| `kvcr_duration_seconds{result}` | Labels from §3.2, classified by the source's own reasons | Same labels, classified by the delivery outcome (§4.1) |
| `kvcr_unserved_blocks{scope, reason}` | Not recorded | `scope` ∈ {`remote_deliver`, `remote_fetch`}: every requested block not delivered, counted once, by reason |
| WARNING logs | Own faults only: pin callback errors, layout mismatch (one line per operation), NIXL create/transfer errors, malformed requests | Own faults only: control send/receive failures, invalid notifications; quarantine at ERROR |
| DEBUG `*_transfer_completed` | `requested=`, `blocks=`, `result=`, `unserved=reason:count,…` | Same, plus `source=` |
| NVTX | Own-side status and reasons | Own-side status and reasons |
| Wire | `write_done.reason`, `write_done.unserved` | Accepts known reason names only |

**Why the counter lives on the target only.** The target is the side that
requested the blocks, sees every one of them exactly once, and receives the
source's per-block reasons in `write_done`. A source-side copy of the counter
counted the same blocks a second time under a different scope, could not see
blocks the source was never asked about (early rejections, remembered misses),
and added nothing an operator acts on that `source_write{result}` and the
source WARNINGs do not already give.
