# KV Cache Runner (KVCR): Design and API

Inference engines manage KV caches across GPU memory, host DRAM, local SSD, and remote object storage, both within a single node and across large clusters. Current approaches suffer from one of two failure modes: tight GPU coupling, where external entities manage GPU-side KV blocks and compete for kernel launch bandwidth, requiring continuous synchronization with the engine; or naive simplicity, where eviction happens reactively under peak memory pressure with no admission filtering and no cross-node coordination. Neither scales cleanly. In multi-node disaggregated serving, the problem compounds: prefix caches are distributed, routing decisions affect hit rates, and coordination without a clear authority creates either duplicated state or missed reuse.

**KV Cache Runner (KVCR)** uses capabilities already present in the KV router to simplify the cache-management boundary. The router tracks cache placement for routing, can incorporate harness-level context such as relationships among sub-agent requests, and passes relevant hints to KVCR. KVCR can therefore focus on local cache policy, residency, and data movement without duplicating the router's global view.

**Contents**

- [Goals and Design Principles](#goals-and-design-principles)
- [Architecture](#architecture)
- [Framework–KVCR API](#frameworkkvcr-api)
- [Router–KVCR API](#routerkvcr-api)
- [Policy API](#policy-api)
- [Telemetry API](#telemetry-api)

---

## Goals and Design Principles

### Non-Intrusiveness

**Goal:** The KVCR must not put blocking work on the engine's hot path or interfere with memory the framework uses for execution. Cache data that the framework depends on must retain a local, fast-access guarantee, whether the framework holds the bytes directly or delegates that guarantee to the cache layer. Framework-facing calls either read local metadata, enqueue work, or complete asynchronously, while internal caching remains transparent to the engine.

**Resulting principle — The framework owns and orchestrates GPU memory; it delegates transfer execution to purpose-built tools.**

Adding GPU management to the KVCR introduces coordination on the hot path, shared GPU state between two entities, and the need to continuously synchronize scheduling context. The framework already has all the context needed: scheduling state, active requests, decode step count. Any external entity would need to be told this continuously.

The framework's orchestration role extends across both caching and disaggregated serving. For prefill/decode (P/D) transfers, it delegates the physical DMA to NIXL. For KV caching, it delegates host-side movement and cross-node transfers to the KVCR — providing GPU pointers when needed so the KVCR can execute GPU↔host DMA via NIXL on its behalf. Framework-owned memory movement and GPU scheduling remain framework decisions; the KVCR decides KVCR-owned tier movement in response to framework calls, router hints, and policy. GPU-bound bandwidth is not exclusively local: in disaggregated serving, remote writes into GPU memory are already in the picture. The invariant that matters is GPU memory ownership and scheduling, not who physically issues the DMA.

### Scalability and Locality

**Goal:** Only the KV router's global inventory should scale with the cluster. It receives partial, eventually-consistent information from each KVCR — enough to make good decisions without requiring a complete, strongly consistent global view. A KVCR's role and behavior are the same on a single node and in a thousand-node cluster; adding a node adds its inventory to the router without changing existing KVCR instances.

**Resulting principle — Separating cross-node coordination as a global inventory where KVCR instances have no peer block inventory; same-node and cross-node reuse use the same source model.**

Each KVCR manages only its own local DRAM and SSD and has no knowledge of peer block inventories. The KV router holds the cross-node mapping — kept as minimal as possible, keyed by `BlockKey` and optional prefix-path metadata — and provides request-scoped source hints. KVCR instances execute transfers directly via NIXL.

This design uses the same router-directed source model for same-node and cross-node reuse. A shared node-level pool would add another ownership, coherence, isolation, and recovery boundary. Instead, the router identifies the source and NIXL selects the appropriate transport—for example, directly from one process's host memory into another process's GPU. Colocation changes the transport, not the ownership or coordination model.

### Extensibility and Adoptability

**Goals:** Frameworks can plug in caching policies through the standard control-plane interface without knowing or modifying the underlying data-movement mechanism. Framework hints bridge the gap between scheduling context known to the engine and storage policy owned by the KVCR.

Frameworks can adopt the KVCR incrementally, with each integration level delivering independent value. A framework can use the KVCR to offload cache data into KVCR-owned DRAM or storage, or pair it with a KV router to share framework-owned memory and KVCR-managed data across instances. These capabilities can be combined without replacing the framework’s routing or scheduling layers.

**Resulting principle — Separation of Control and Data Plane: Mechanism is stable, policy is pluggable.**

All host-side data movement — slot management, SSD writes, object-store operations, cross-node transfers — is the KVCR's data plane and does not change across deployments. The policy governing admission, eviction, placement, and hint-guided staging is the control plane and is fully replaceable.

### Resilience

**Goal:** The normal KVCR remains in the inference-engine process for efficiency. Deployments may optionally preserve KVCR-owned memory across an engine or GPU failure; framework-owned GPU and host memory remain outside this guarantee.

**Resulting principle — KVCR-owned memory can outlive the in-process KVCR without moving the normal path out of process.**

KVCR-Guard provides this resilience by owning the memory and hosting a backup KVCR without changing the normal cache-management interface. The active KVCR maps the pool directly. After fencing a failed owner, the backup can take ownership and serve committed KV data. A restarted in-process KVCR attaches, recovers the full committed state, and takes ownership back through a fenced handoff.

---

## Architecture

### Component Roles

Figure 1 summarizes the component boundaries.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/kv-architecture-detailed-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="figures/kv-architecture-detailed-light.svg">
  <img src="docs/figures/kv-architecture-detailed-light.svg" alt="KV Cache Runner Design Diagram">
</picture>

**Framework / Engine**
- Sole owner of GPU memory: all block allocation, eviction decisions, scheduling
- Decides which blocks enter or leave framework-owned memory and when; performs transfers directly or delegates to the KVCR for caching and cross-node serving
- During framework initialization, supplies runtime-provided configuration, including the peer control endpoint, and the model and KV-layout information required by KVCR.
- Exposes framework-owned GPU and host memory to the KVCR through descriptors and an optional pinning interface; the KVCR registers exposed memory with its NIXL agent
- Reports framework-owned memory to the router

**KV Router**
- Maintains eventually-consistent global `BlockKey` inventory from all engine and KVCR instances, with tier labels
- Makes cache-aware routing decisions based on KV overlap score and node load
- Sends hints to destination KVCR instances — for cross-node retrieval with a source node and peer control endpoint, or optionally for same-node SSD or remote object-store retrieval

**KVCR**
- Main runtime component linked directly into the engine as an in-process library
- Manages KVCR-owned DRAM and SSD residency, plus configured object storage
- Reports KVCR-tier inventory to the router
- Executes data movement through NIXL, which provides a common abstraction across tiers for both local and remote transfers
- Enforces a user-supplied or built-in policy: admission, eviction, tier placement, and hint-guided staging
- Responds to framework queries about block availability
- Avoids blocking the engine's hot path

**Optional KVCR-Guard**
- Owns the KVCR-owned DRAM allocation when engine-failure resilience is enabled
- Exposes one socket endpoint through which the active KVCR attaches and recovers the full committed state
- Makes the pool available for direct access after attachment, so normal cache operations do not require an IPC round trip
- Hosts a backup KVCR with its own NIXL agent; after fencing the failed active owner, the backup may serve already committed KVCR-owned KV data
- Preserves committed state for hot start and transfers ownership back to a restarted in-process KVCR through a fenced handoff
- Does not preserve framework-owned memory

### Memory Ownership and Recovery

A KVCR-owned DRAM pool may be allocated by the framework and passed to KVCR, or owned by KVCR-Guard. This choice changes the pool lifetime and recovery guarantee, but not the cache API or policy semantics. The pool should remain near the GPUs it serves when the NUMA topology permits. When KVCR-Guard owns the pool, the active in-process KVCR attaches through a socket endpoint and then accesses the pool directly through shared memory, so normal cache operations require no IPC round trip.

`KVLayoutManifest` identifies the framework, model, KV layout, and host representation needed to interpret cached data. A pool may contain multiple internal pools for different attention-head requirements. When `kvcr_guard_endpoint` is provided, KVCR attaches to the relevant preserved pool and verifies that the supplied manifest is compatible; initialization fails if the pool is unavailable or incompatible. Matching TP layouts are sufficient for most prefill-to-prefill, decode-to-decode, and aggregated deployments. A TP-independent host representation further removes that constraint as well.

If the engine or GPU fails, KVCR-Guard fences the failed owner before activating its backup KVCR. A replacement in-process KVCR can attach to the preserved pool, recover the committed state, resynchronize inventory, and assume ownership through a fenced handoff. Partial writes, in-flight operations, and framework-owned GPU or host memory are not recovered. Recovery and handoff must preserve committed-data integrity and prevent concurrent ownership.

### State Model

The memory, routing, and policy flows share a node-local state and operation model. Its central lookup structure is the `block_index`, which maps each `BlockKey` to a `BlockRecord`. The `BlockRecord` is the authority for that block's known locations and residency state. A block may be present in several locations at once, for example KVCR-owned DRAM and SSD, or KVCR-owned and framework-owned memory while a transfer is in flight. Because this map contains only local state, it does not grow with the cluster; the router owns the global inventory.

Framework-owned memory and KVCR-owned memory use different residency types. Framework residency carries the framework descriptor and pin. KVCR residency carries the KVCR allocation, readiness, claims, and eviction state. Storage tiers likewise use location types suited to their own behavior. Remote endpoints and request-scoped hints belong to operation state rather than the block record.

Residency state and operation state are deliberately disjoint. A `BlockRecord` does not embed or own operation lifecycle state; it only keeps references to related operation IDs when needed. The progress thread's polymorphic `in_flight_ops` collection owns the operations it advances and their progress-side lifecycle; each operation type defines how completion affects its resources. This lets block lookup remain stable while operations are created, coalesced, completed, cancelled, or removed independently.

Ready, unclaimed KVCR-owned residencies are tracked by a policy-scored eviction queue, while available pool slots are tracked separately for allocation. Auxiliary structures may support completion polling, pins that cover multiple keys, and reuse of established peer connections.

Claims are attached to concrete residencies. They prevent KVCR-owned memory from being evicted while in use and keep framework-owned memory pinned until all dependent operations finish. Concurrent operations may share existing residencies, pins, or in-flight work; cancellation of one caller must not invalidate resources still needed by another. When an operation needs a KVCR residency that is currently evictable, the KVCR removes it from the evictable queue and acquires a claim before using it, making it non-evictable. After the last claim is released, a ready residency that is still retained returns to the evictable queue; otherwise its KVCR-owned memory is freed.

### Asynchronous Execution

The event loop owns KVCR metadata mutations and never performs blocking external calls. A dedicated progress thread owns the NIXL agent, submits and polls transfers, and posts completions back to the event loop. Framework-agent and storage work follow the same asynchronous completion pattern.

The active in-process KVCR owns the NIXL agent used for KVCR-owned memory and framework memory exposed through the KVCR bindings. A future integration may instead coordinate with a framework-owned agent. When resilience is enabled, the backup KVCR has its own agent but uses it only after a fenced takeover.

Each operation is bounded by a deadline. When an operation times out or is cancelled, KVCR reports caller-visible completion and begins safe release immediately. Framework pins are released as soon as their dependent work finishes, minimizing interference with framework scheduling. If NIXL may still access an underlying descriptor, physical release waits until the transfer has quiesced. Expired pins are not reused; any still-needed keys are acquired again. If physical cleanup extends beyond the deadline, it does so only for safe release, not further KVCR scheduling.

---

## Framework–KVCR API

The framework and KVCR interact through the following API.

```python
# Python-style pseudocode
# Framework → KVCR
kvcr = KVCR(
    pools=[PoolSpec(...)],
    layout_manifest=layout_manifest,
    kvcr_guard_endpoint=None,
    peer_control_endpoint=peer_control_endpoint,
    config=config,
)

kvcr.deposit(blocks, no_evict=False, hints=None, callback=None)  # blocks: dict[BlockKey, MemDescriptor]; completion value is None or (ptr, release_handle)
kvcr.query(block_key_list, request_id=None) -> list[tuple[Status, CacheTier | None]] # HIT/MISS/FETCHING/FETCHABLE with known location
kvcr.fetch(block_key_list, request_id=None, hints=None, callback=None) -> OperationHandle # completion value is (ptr, release_handle)
kvcr.deliver(destinations, request_id=None, callback=None) -> OperationHandle # destinations: dict[BlockKey, MemDescriptor]
kvcr.release(release_handle_list) -> list[Result[None, Error]]      # release fetch/no-evict claims

kvcr.poll_completed() -> list[Completion]                          # drain individual completions
kvcr.abort(operation_handle, block_key_list=None)                  # cancel this fetch/deliver operation, optionally for selected keys
kvcr.close()                                                       # synchronous teardown after framework-submitted jobs are drained

# KVCR → Framework
framework.capacity_needed(num_slots)                               # last-resort capacity pressure signal

framework.request_pin(block_key_list) -> PinRequestId                             # enqueue a framework-owned source pin request
framework.poll_pin_results() -> list[tuple[PinRequestId, PinResult]]               # drain completed requests; one pin may cover the full list
framework.cancel_pin_request(pin_request_id)                                      # cancel an unresolved request
framework.release_pin(pin_handle)                                                 # release an acquired framework-owned source pin
```

### Operating flow

**Using KVCR-owned DRAM**

`deposit`/`deliver` — framework initiates both; `deposit` pushes a block into the KVCR-owned pool and may retain an evictable KVCR copy; its callback, when provided, fires when the source pointer is safe to reuse. `deliver` requests that the KVCR place a block into a framework-provided GPU or host-memory destination; its callback, when provided, fires when the destination is ready.

`fetch`/`release` — framework asks the KVCR to make a block resident in its KVCR-owned DRAM pool and pin it there. On success, `fetch` returns the resident pointer and a release handle; the framework calls `deliver` to place the block into a framework-provided destination and calls `release` when the pool claim is no longer needed.

`deposit` also accepts a `no_evict` flag (batch-level, applies to all entries): when set, the KVCR keeps every completed slot non-evictable and returns the pointer plus release handle per entry. A framework that wants guaranteed local DRAM residency behavior for selected KV blocks can get that behavior through `no_evict`, while the KVCR still handles sharing, routing visibility, transfer setup, and tiering policy. The framework calls `release` with the corresponding handle to clear the no-evict claim.

The tradeoff is backpressure: when policy cannot free enough capacity—for example, because `no_evict` claims occupy the pool or an attempted eviction does not free its source—KVCR may invoke `capacity_needed` as a last-resort pressure signal. The framework should release enough claims to free the requested slots. If sufficient capacity remains unavailable, affected committed entries complete with errors. Pool size and the free-slot threshold that triggers `capacity_needed` are deployment knobs.

**Block availability query**

When the framework asks the KVCR whether a block is available, the KVCR responds with one of:
- `HIT(tier)` — locally available at the reported tier at the time of the query
- `FETCHING(destination_tier)` — a fetch into the reported tier is in progress
- `FETCHABLE(source_tier)` — not started, but the KVCR knows a source tier
- `MISS(None)` — not found in any tier the KVCR currently knows about

These statuses describe current KVCR knowledge, not a reservation or guarantee. A local DRAM `HIT` may be evicted before a later operation claims it, and a `FETCHABLE` result for peer DRAM means a request-scoped router hint identifies a peer KVCR that may source the block. The peer may ultimately serve it from KVCR-owned or pinned framework-owned DRAM; actual readability is confirmed only when the transfer and any required `request_pin` succeed. Framework-owned HBM can follow the same path if a deployment exposes GPU KV through the pin API. The framework may use the query result to decide whether to issue either call. To obtain the data, the framework calls `fetch` or `deliver` and waits for completion; either operation may return an error.

`query` is fast and non-blocking because it reads local KVCR knowledge rather than calling the router. Remote source information arrives through `submit_hint`. If a two-step `query` is enabled, it may enqueue a configured object-store existence check, but it does not allocate memory, acquire pins, change residency, or start a transfer. After the first `MISS`, the framework can issue a follow-up `query` to get an updated evaluation.

**Data movement and framework pins**

`deposit` copies from framework-owned memory into the KVCR's pool, while `deliver` places data into a framework-provided destination. `deliver` does not name a source; source selection remains with the KVCR and router. The KVCR does not allocate or free framework memory.

To serve from framework-owned memory, the KVCR acquires a pin asynchronously through `request_pin` and `poll_pin_results`, reusing covered keys and requesting only the remainder; the framework keeps it valid until the KVCR calls `release_pin`.

A deployment may choose to use only framework-owned memory. In that case, it uses the pinning mechanism together with `deliver` and does not use `deposit`, `fetch`, or `release`.

**Completion and cancellation**

`fetch`, `deposit`, and `deliver` accept batches and may report per-entry outcomes. When provided, a batch callback fires after the batch completes, while `poll_completed` exposes individual completions as they arrive.

`abort` is best effort and may target selected entries. Shared acquisition or transfer work continues while other operations still depend on it.

---

## Router–KVCR API

The router and KVCR interact through inventory events and request-scoped source hints.

```python
# Python-style pseudocode
class InventoryEvent:
    keys: tuple[BlockKey, ...]
    tier: CacheTier
    removed: bool

# KVCR → integration
inventory_sink(event: InventoryEvent)

# Router → KVCR, directly or through the framework
kvcr.submit_hint(block_key_list, src=None, mode=copy|move, hints=None, request_id=None) # src includes node and KVCR control endpoint; default mode=copy
kvcr.discard_hint(request_id) # request is over or its hints should be discarded early
```

### Inventory and block identity

The router maintains an eventually-consistent global prefix inventory for cache-aware request routing. The router is the routing-key authority and supplies the request's content-derived `BlockKey` values. A `BlockKey` identifies one cacheable KV unit, while a `BlockKeyList` is an ordered exact request path rather than an implicit prefix expansion. The framework normally passes these keys unchanged to the KVCR; an integration with distinct local hashes translates them at its boundary. The key space may be namespaced by model version, quantization, KV layout, tenant, or hash algorithm, and the KVCR treats each full key as opaque.

Framework-owned memory is reported through the framework's existing router event path. If needed, `tier_enter`/`tier_exit` reporting or `inventory_snapshot()` can be added to the KVCR API later.

KVCR-owned inventory events use the same `BlockKey` values as fetch and `submit_hint` in the normal case. KVCR deliberately reports only the affected keys, tier, and whether the residency was removed. Events may be batched and are tier-specific because one block may be present in several tiers at once.

### Request-scoped hints

For data shared among engines on the same node, the router can use one KVCR as the source in hints sent to the other local KVCRs. A destination may retain a local copy when replication is worthwhile. Otherwise, the router includes `no_retain` in `submit_hint`. The destination still satisfies any committed framework request; `no_retain` only advises it not to keep an additional KVCR-owned copy after the framework no longer needs it.

KVCR-to-KVCR movement is destination-initiated. A router hint gives the destination the source node and KVCR control endpoint. `mode=copy` retains the source copy, while `mode=move` permits source eviction only after successful completion. Timeout or cancellation keeps the source copy. When the router sends `src=None`, the destination uses its local storage information or checks object storage when enabled. The router does not need object-store inventory for locality decisions, but may track object presence to issue source hints.

`submit_hint` may establish the peer connection, but does not start data movement or remote pinning; proactive fetching, staging, or pinning may be added later if useful. `discard_hint` normally reports that the relevant request is over, and also permits an integration to discard the request-scoped hint early. A route-time `submit_hint` may carry a whole `BlockKeyList` from one source. In the future, if necessary, multi-source assembly can be added by splitting the list into multiple hinted operations.

The router may choose not to send a hint based on its internal cost model. Router hints may later be extended to request proactive copy or move for cache rebalancing, which may require periodic utilization reports from KVCR.

### Peer control and liveness

The router is never on the data path: KVCR instances execute transfers peer-to-peer via NIXL. Source endpoints remain request-scoped rather than becoming part of global block state, so this coordination adds no KVCR-side peer inventory.

Peer transfers use a separate control channel for connection metadata, acknowledgements, and transfer control; payload bytes move directly through NIXL and never traverse the router or control channel. Peer-protocol versioning and compatibility checks may be added as needed.

The engine and router already maintain engine liveness, so loss of an in-process KVCR is covered by the engine's existing heartbeat path and does not require another KVCR heartbeat. KVCR-Guard sends its own heartbeat to the main process. If a future recovery design requires KVCR-Guard to advertise liveness or takeover directly to the router, that channel can be added then.

---

## Policy API

Policies implement the following interface.

```python
# Python-style pseudocode
class KVCachePolicy:
    required_tiers: frozenset[CacheTier] = frozenset()

    def decide_ingest(self, meta, source, required_local,
                      router_hints=None, framework_hints=None) -> PlacementDecision: ...
    def eviction_score(self, meta, source) -> float: ...
    def decide_eviction(self, meta, source) -> PlacementDecision: ...

    # Optional lifecycle hooks.
    def on_ingest(self, meta, source) -> None: ...
    def on_remove(self, meta) -> None: ...

    # Optional recovery override.
    def decide_recovery(self, meta, failure) -> PlacementDecision:
        return (PlacementAction.DROP, None)

class PlacementAction:
    KEEP = "keep"
    DROP = "drop"
    COPY_TO = "copy_to"
    MOVE_TO = "move_to"

PlacementDecision = tuple[PlacementAction, CacheTier | None]
```

At initialization, an integration selects one built-in policy or supplies an external `KVCachePolicy` instance; otherwise the default policy is LRU. Policy calls must complete quickly and never block. A policy declares its configuration dependencies through `required_tiers`; KVCR rejects initialization if any declared tier is not configured. `CacheTier` identifies the relevant framework memory, KVCR-managed storage, peer memory, or object-store tier.

`meta` is a read-only snapshot of the block's identity, size, access history, and current managed residency. `failure` describes the attempted placement, its source, the failure reason, and the number of previous failures.

Policy decides whether to stage, retain, move, or drop data within KVCR-owned DRAM and downstream storage. Fetch and `no_evict` deposit claims are hard mechanism constraints: while they exist, policy cannot evict the entry. `no_retain` is an advisory router retention hint. If it conflicts with a framework `no_evict` claim, `no_evict` wins until the framework calls `release` with the returned handle; policy may then honor or override `no_retain`.

The policy methods are invoked as follows:

- **Admission — `decide_ingest`:** Called before KVCR creates a managed residency. `required_local` means the admission cannot be dropped, as required for a `no_evict` deposit or a fetch. `KEEP` admits it locally, while `DROP` declines optional admission and completes it as `DROPPED`. `COPY_TO` retains the local residency and also writes to the destination; `MOVE_TO` removes the local residency only after committing the destination. Framework hints and any policy-relevant router hints may inform this decision.
- **Eviction — `eviction_score` and `decide_eviction`:** When a ready, unclaimed residency enters or re-enters the evictable queue, KVCR calls `eviction_score` and stores the result. When capacity is needed, candidates with lower finite scores are considered first and KVCR calls `decide_eviction`: `KEEP` declines that eviction, `DROP` removes the residency, and `MOVE_TO` removes it after committing the destination. Scored entries are not rescored while they remain in the queue; dynamic reprioritization may be added later if needed.
- **Lifecycle — `on_ingest` and `on_remove`:** These optional hooks notify stateful policies when the first managed residency becomes ready and when the final managed residency disappears. Internal movement between managed tiers does not create another lifecycle event.
- **Recovery — `decide_recovery`:** Called when a policy-requested copy or move fails. The default decision logs the failure and drops the source residency.

---

## Telemetry API

```python
# Python-style pseudocode
kvcr.get_stats() -> TelemetryStats | None
```

`get_stats` returns the current snapshot when telemetry is enabled. This snapshot covers:

- operation outcomes and end-to-end latency;
- control-channel setup latency, message counts, acknowledgements, and failures;
- framework pin acquisition latency and acquisition/release outcomes;
- NIXL setup, submission, transfer, and notification latency, plus failures;
- blocks and bytes transferred by bounded source and destination tier;
- failures such as unavailable sources, timeouts, capacity errors, and incompatible peer metadata; and
- current resource counts, including known blocks, in-flight operations, pins, claims, peer connections, and available capacity.

Telemetry is optional and cheap when disabled. Labels use bounded categories rather than block keys, request IDs, or raw endpoints.
