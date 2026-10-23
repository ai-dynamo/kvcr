# KV Hint Dispatcher

Design for handling multi-hint envelopes in KVCR. This document covers the hint-agnostic infrastructure only: parsing every hint, routing each to a handler, and reporting a result per hint.

## Summary

KVCR treats a KV hint envelope as a batch of independent hints. Each hint type has its own parser and handler. A handler runs its work inline at submit time, on the caller's thread. Every hint produces a result (accepted or rejected, with a reason).

## Context

KVCR receives KV hints from a KV router through a framework integration layer that calls `KVCR.submit_hint(envelope, request_id)`. The router protocol is posted: no acknowledgement flows back to the router. The envelope format is shared across the framework RFCs (TensorRT-LLM #18153, vLLM #59708, SGLang #36224): several actions per envelope, per-hint failure isolation, and the rule that submitted does not mean ready.

## Design

### Hint model

Each parsed hint is a frozen `msgspec.Struct` with at least `action_id: str | None` and a class-level `ACTION_TYPE: str`. Each hint module defines a `HintType(action_type, version, parse, handler_factory)`, and a fixed table in the dispatcher, `_HINT_TYPES`, lists the supported types. A `HintDispatcher` builds one handler per entry as `handler_factory(core, dispatcher)` and passes the same table to the parser, so a type always has a parser and a handler. A handler uses `core` for backend access and the dispatcher to `report` results. Core holds one dispatcher and never names a handler.

### Results

```python
class HintStatus(Enum):
    ACCEPTED = "ACCEPTED"    # inline work done
    REJECTED = "REJECTED"    # invalid, unsupported, handler_error

class HintResult(msgspec.Struct, frozen=True):
    action_id: str | None
    action_type: str
    status: HintStatus
    reason: str | None = None
    request_id: str | None = None
    message_id: str | None = None
```

Every result is logged at debug level; rejections and handler errors also warn once per issue. Nothing is retained, because inline results are terminal at submit time and no integration consumes them. A bounded, drainable result queue arrives with the deferred lane, whose outcomes are asynchronous. A hint whose `action_type` cannot be read (the hint is not a mapping, or its `action_id` or type is not a string) is reported with the sentinel `action_type="unknown"`. A hint reaches a terminal status (`ACCEPTED` or `REJECTED`) at submit time; a handler that raises has its hints reported `REJECTED` with reason `handler_error`. A fatal KVCR failure (the progress thread failed) is not a hint result: `submit_hint` raises it before dispatching, as `poll_completed` and `get_stats` do, and report nothing. A handler that surfaces it while dispatching (for example through the remote store's submit) does not turn it into `handler_error`: it propagates the same way.

### Parsing

`parse_hints(envelope) -> (hints, rejections)`. A structurally invalid envelope (not a mapping or struct, `actions` not a list) raises `ValueError`, as today. Everything else is per hint and never raises: a non-mapping hint, a non-string `action_id`, an unknown `action_type`, a non-mapping payload, or a payload the type's parser refuses (`ValueError` reason, or `invalid_hint` for any other exception) produces a `HintResult(REJECTED, reason)` and the loop continues. Protocol and hint version mismatches warn once per issue, and rejections warn once per (supported type, reason), with all unsupported types sharing one entry. An envelope whose hints are all rejected returns an empty hint list and the rejections; it does not raise.

### Inline dispatch

`handler.submit(hint, request_id, message_id)` runs on the caller's thread, without the state lock, once per hint in envelope order. It must only touch state that is safe there and must report exactly one result for the hint. A handler exception is caught, logged once, and reported `REJECTED` with reason `handler_error` for that hint. No new threads are added.

### `kv.fetch`

Module `hints/fetch.py`, the only supported hint type today; it stores the per-request remote hint that the engine later turns into `fetch` and `deliver` calls.

The `kv.fetch` payload parser keeps its current rules (source endpoint required, `mode` must be `copy`, `no_retain` refused, 64-bit hashes).

Each `kv.fetch` hint is one call to the existing per-request hint store, reported `ACCEPTED`. The store's semantics are unchanged: a later hint from the same source replaces the stored hashes, and a hint naming a different source marks the request's stored hint failed, so the request is no longer served remotely. Both apply within one envelope exactly as across calls, and the store does not report either case, so one `kv.fetch` per envelope is the supported shape. The engine path takes no lock and starts no transfer beyond the pre-existing eager control connect. Reporting `REJECTED` when the store refuses a conflicting source would need the store to report acceptance; that is left for later.

### Configuration (`KVCRConfig`)

This change adds no configuration.

### Telemetry

This change adds no metric. Integrations forward KVCR stats under their own namespace and may reject names they do not declare (vLLM's KVCR adapter asserts on them), so a hint result counter waits until the integrations declare it.

## Adding a hint type

A new hint type is one module in `kvcr/hints/`: the parsed struct with its `ACTION_TYPE`, the payload parser, a handler class implementing `submit`, the `HintType` that ties them together, tests, and a config block of its own if it needs one. Its `HintType` is added to `_HINT_TYPES` in `hints/dispatcher.py`. Envelope handling, isolation, and results are shared and unchanged, and `core.py` is not touched.

## Potential future work

- Per-type grouping of an envelope's hints, so a handler sees all of its hints from one envelope at once.
- Deferred lane (stacked on this change): handlers whose work must run under KVCR's state lock queue their hints and the dispatcher admits them from `poll_completed` under a per-poll budget with stale expiry, reporting exactly one terminal result per queued hint, and retains results in a bounded drainable queue. Prefetch is its first consumer.
- Prefetch into KVCR local DRAM as a deferred hint (a `destination` payload field or a dedicated action type): a key-adapter method to resolve router hashes to keys, prefetch layout, eviction flag, per-pool reservation budget, explicit-source remote pulls with a prefetch timeout at both peers.
- `kv.invalidate`, `kv.evict`, priority or retention, `mode=move`, `no_retain`.
- Side-band ingress (for example a ZMQ listener) feeding the dispatcher parsed hints that did not arrive in an envelope.
- Priority between deferred hints and over engine pulls on the progress thread.
