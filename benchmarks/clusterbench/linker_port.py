# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SGLang's KVCR direct linker, without the GPU.

A port of ``KVCRDirectLinker`` (with the parts of ``KVCRRuntime`` and
``KVCRLayout`` it uses) from SGLang PR #45650 at revision ``002d6d8d``:

- ``python/sglang/srt/mem_cache/storage/kvcr/kvcr_direct_linker.py``
- ``python/sglang/srt/mem_cache/storage/kvcr/runtime.py``
- ``python/sglang/srt/mem_cache/storage/kvcr/device_layout.py``
- ``LayerWiseLoadCounter`` in ``mem_cache/unified_cache/unified_cache_linker.py``

What KVCR sees, and must keep seeing if this file changes:

- One ``kvcr-owner`` thread makes every KVCR call. The scheduler thread posts
  a command and blocks on its result. While a load or an offload is pending
  the owner spins (a zero-timeout queue read, then ``sleep(0)``), and every
  loop iteration ends with ``poll_completed``.
- ``lookup`` is one ``query`` over all of a request's uncached pages, with the
  request's scope as ``request_id``. ``submit_hint`` precedes it whenever the
  request carries a hint, on every scheduling pass.
- A load is delivered layer by layer: one ``deliver`` per load per layer,
  naming that layer's K and V piece of every page, at most four deliver
  operations outstanding. One failed key fails the whole batch.
- An offload becomes ``deposit`` calls of at most 4096 pieces (or 512 MiB),
  each naming every piece of its pages, strictly one at a time, and only once
  the loads accepted before it are done and no command is waiting.
- ``discard_hint`` follows a released scope once no load uses it.
  ``align_sequence``, ``fetch`` and ``release`` are never called (``fetch``
  only confirms sliding-window state, which a dense model does not have).

Left out: TP>1 gathers, sharding, CUDA events (always ready here), Guard
service memory. Everything marked "instrumentation" is this benchmark's and
has no upstream counterpart.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import NamedTuple

import msgspec

from .engine import HintEnvelope
from .ids import LINKER_KEY_PREFIX, LINKER_POOL, hash_to_id
from .metrics import OwnerStats

# Upstream: bound admission work and DMA per native deposit. A key's complete
# set of pieces must stay in one deposit.
MAX_DEPOSIT_BYTES = 512 << 20
MAX_DEPOSIT_DESCRIPTORS = 4096
MAX_DELIVERIES = 4
STATS_INTERVAL_S = 5.0
_FLUSH_LOOPS = 1024
_HIT = {("HIT", "DRAM"), ("FETCHABLE", "REMOTE_G2")}
_REMOTE = ("FETCHABLE", "REMOTE_G2")


class Handle(NamedTuple):
    """SGLang ``CacheRequestHandle``."""

    rid: str
    attempt_id: int


def decode_hints(value) -> HintEnvelope:
    """SGLang ``KvHintsEnvelope``; KVCR parses it as its ``kv.fetch`` hint."""
    if isinstance(value, HintEnvelope):
        return value
    try:
        return msgspec.convert(value, HintEnvelope)
    except (msgspec.ValidationError, TypeError) as error:
        raise ValueError(f"Malformed kv_hints envelope: {error}") from error


class Transfer(NamedTuple):
    """Upstream ``PreparedTransfer`` of the one dense pool."""

    keys: tuple[bytes, ...]
    locations: tuple[int, ...]


@dataclass
class _Load:
    rid: str
    scope: str | None
    transfers: tuple
    done: bool = False
    error: BaseException | None = None
    # Instrumentation: how many of its per-layer delivers returned, and
    # KVCR's result for each page that failed (None if it named none).
    layers: int = 0
    failed: dict = field(default_factory=dict)


@dataclass
class _Batch:
    index: int
    loads: tuple[_Load, ...]
    work: object
    remaining: list[int]
    active: int = 0
    exhausted: bool = False
    error: BaseException | None = None
    started: float = field(default_factory=time.perf_counter)
    completion: Future = field(default_factory=Future)


@dataclass
class _Offload:
    transfers: tuple
    dependencies: tuple[_Load, ...]
    keys: set[bytes]
    work: object = None
    active: int = 0
    exhausted: bool = False
    success: bool = True


@dataclass
class _Operation:
    owner: _Batch | _Offload
    keys: set[bytes]
    layer: int | None = None
    load: _Load | None = None
    result: object = None


class BatchInfo(NamedTuple):
    """Instrumentation: a finished load batch, on the owner's clock."""

    loads: tuple[_Load, ...]
    started: float
    finished: float


class LayerCounter:
    """Upstream ``LayerWiseLoadCounter``: one future per layer of a batch."""

    def __init__(self, num_layers: int, timeout_s: float) -> None:
        self.num_layers = num_layers
        self._timeout = timeout_s
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}

    def update_producer(self) -> int:
        self.producer_index += 1
        self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        self.futures[index][layer].set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        for future in self.futures.get(index, ()):
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        futures = self.futures.get(self.consumer_index)
        if futures is None:
            return
        # Blocks until the layer resolves; the batch's completion says whether
        # it failed. A stuck owner raises here instead of hanging the worker.
        futures[threshold].exception(self._timeout)

    def finish(self, index: int) -> None:
        self.futures.pop(index, None)


class LinkerPort:
    """The linker and its owner thread.

    ``host`` provides ``start()`` (build KVCR on the calling thread; upstream
    builds it on the owner), ``kvcr``, ``agent_name``, ``labels`` (per
    component, one label per layer), ``piece_bytes`` and ``close()``.

    With ``inline=True`` no thread is started: commands run in the caller and
    the test drives ``progress()`` itself.
    """

    def __init__(
        self,
        host,
        *,
        inline: bool = False,
        call_timeout_s: float = 60.0,
    ) -> None:
        self._host = host
        self._inline = inline
        self._call_timeout = call_timeout_s
        self._components: tuple[tuple[str, ...], ...] = tuple(host.labels)
        self.num_layers = len(self._components[0])
        pieces = sum(len(component) for component in self._components)
        self._key_bytes = host.piece_bytes * pieces
        self.layer_done_counter = LayerCounter(self.num_layers, call_timeout_s)
        self._key_prefix = LINKER_KEY_PREFIX
        self._batch_results: dict[int, Future] = {}
        self._commands: Queue = Queue()
        self._command_lock = threading.Lock()
        self._completed_offloads: Queue = Queue()
        self._loads: dict[str, _Load] = {}
        self._queued: list[_Load] = []
        self._batches: deque[_Batch] = deque()
        self._offloads: deque[_Offload] = deque()
        self._unacknowledged_keys: Counter = Counter()
        self._operations: dict = {}
        self._hazards: set = set()
        self._scopes: dict[Handle, str] = {}
        self._request_scopes: dict[str, str] = {}
        self._released_scopes: set[str] = set()
        # Upstream ``KVCRRuntime``: resilience events and pending removals.
        self.events: deque = deque()
        self.removals: dict[bytes, str] = {}
        self._fatal: BaseException | None = None
        self._closed = False
        self._kvcr = None
        self._endpoint = ""
        self._last_stats = time.monotonic()
        # Instrumentation.
        self._stats = OwnerStats()
        self._swap_box: Future | None = None
        self._empty_polls = 0
        self._empty_poll_seconds = 0.0
        self.batch_info: dict[int, BatchInfo] = {}
        self.resilience_samples: list[str] = []
        self._started: Future = Future()
        if inline:
            self._start_runtime()
        else:
            self._thread = threading.Thread(
                target=self._run, name="kvcr-owner", daemon=True
            )
            self._thread.start()
            self._started.result()

    # ---- owner thread ------------------------------------------------------

    def _start_runtime(self) -> None:
        self._host.bind(self)
        self._host.start()
        self._kvcr = self._host.kvcr
        self._endpoint = self._host.agent_name

    def _call(self, function, *args):
        if self._inline:
            if self._fatal is not None and function != self._close:
                raise self._fatal
            return function(*args)
        future: Future = Future()
        with self._command_lock:
            if self._closed:
                raise RuntimeError("KVCR linker is closed")
            self._commands.put((function, args, future))
        return future.result(self._call_timeout)

    def _run(self) -> None:
        try:
            self._start_runtime()
            self._started.set_result(None)
        except BaseException as error:
            self._fatal = error
            self._closed = True
            self._started.set_exception(error)
            return
        perf = time.perf_counter
        commands = self._commands
        idle = spin = 0.0
        loops = active_loops = 0
        while not self._closed:
            # Spin only while transfers are in flight.
            active = self._fatal is None and (
                self._offloads
                or any(not b.exhausted or b.active for b in self._batches)
            )
            started = perf()
            try:
                function, args, future = commands.get(timeout=0 if active else 0.001)
            except Empty:
                function = None
                if active:
                    # Yield the GIL without pacing active transfers at 1 ms.
                    time.sleep(0)
            waited = perf() - started
            if active:
                spin += waited
                active_loops += 1
            else:
                idle += waited
            loops += 1
            if function is not None:
                try:
                    if self._fatal is not None and function != self._close:
                        raise self._fatal
                    future.set_result(function(*args))
                except BaseException as error:
                    if function == self._close:
                        self._fatal = error
                        for batch in self._batches:
                            self._fail_batch(batch, error, fatal=True)
                    future.set_exception(error)
            if self._fatal is None and not self._closed:
                try:
                    self._progress()
                    self._log_stats()
                except BaseException as error:
                    self._fatal = error
                    for batch in self._batches:
                        self._fail_batch(batch, error, fatal=True)
            if self._swap_box is not None or loops >= _FLUSH_LOOPS:
                counters = self._stats.counters
                counters["owner_idle_seconds"] += idle
                counters["owner_spin_seconds"] += spin
                counters["owner_loops"] += loops
                counters["owner_active_loops"] += active_loops
                idle = spin = 0.0
                loops = active_loops = 0
                self._flush_polls()
                if self._swap_box is not None:
                    self._swap()
        while True:
            try:
                _, _, future = commands.get_nowait()
            except Empty:
                break
            future.set_exception(RuntimeError("KVCR linker is closed"))

    def _flush_polls(self) -> None:
        if self._empty_polls:
            self._stats.call(
                "poll_empty", self._empty_poll_seconds, self._empty_polls, 0
            )
            self._empty_polls = 0
            self._empty_poll_seconds = 0.0

    def _swap(self) -> None:
        """Instrumentation: hand the accumulated stats to the engine thread."""
        box, self._swap_box = self._swap_box, None
        if self._fatal is None and not self._closed:
            # Close the interval here so telemetry does not straddle the edge.
            try:
                stats = self._kvcr.get_stats()
                if stats is not None:
                    self._stats.telemetry.append(stats)
            except Exception:  # noqa: BLE001 - diagnostics only
                pass
            self._last_stats = time.monotonic()
        old, self._stats = self._stats, OwnerStats()
        box.set_result(old)

    def take_stats(self) -> OwnerStats:
        """Instrumentation: detach what the owner has accumulated so far.

        Not a queued command: a waiting command holds back the next deposit.
        """
        if self._inline or self._closed:
            self._flush_polls()
            old, self._stats = self._stats, OwnerStats()
            return old
        box: Future = Future()
        self._swap_box = box
        return box.result(self._call_timeout)

    def _log_stats(self) -> None:
        """Upstream ``KVCRRuntime.log_stats``: drain telemetry every five seconds."""
        now = time.monotonic()
        if now - self._last_stats < STATS_INTERVAL_S:
            return
        self._last_stats = now
        started = time.perf_counter()
        stats = self._kvcr.get_stats()
        self._stats.call("get_stats", time.perf_counter() - started)
        if stats is not None:
            self._stats.telemetry.append(stats)

    # ---- keys (upstream KVCRLayout / KVCRRuntime key adapter) --------------

    def encode_key(self, storage_hash: str) -> bytes:
        return self._key_prefix + f"{LINKER_POOL}:{storage_hash}".encode()

    def decode_key(self, key: bytes) -> str | None:
        prefix = self._key_prefix
        if not key.startswith(prefix):
            return None
        try:
            name, storage_hash = key[len(prefix) :].decode().split(":", 1)
        except (ValueError, UnicodeDecodeError):
            return None
        if name != LINKER_POOL:
            return None
        return storage_hash

    def encode_to_framework(self, key: bytes) -> bytes:
        return key

    def decode_from_framework(self, framework_key: object) -> bytes:
        if not isinstance(framework_key, bytes):
            raise TypeError("KVCR keys must be encoded layout keys")
        return framework_key

    def encode_to_router(self, key: bytes) -> int:
        decoded = self.decode_key(key)
        if decoded is None:
            raise ValueError("Unknown KVCR layout key")
        return hash_to_id(decoded)

    def decode_from_router(self, router_key: int | bytes) -> bytes:
        # A router hash names a page across all pools, so it has no single key.
        raise NotImplementedError("no single KVCR key per router hash")

    # ---- KVCR callbacks (they run inside KVCR calls, on the owner) ---------

    def record_inventory(self, event) -> None:
        if event.tier.value == "DRAM":
            removals = self.removals
            for key in event.keys:
                decoded = self.decode_key(key)
                if decoded is not None:
                    if event.removed:
                        removals[key] = decoded
                    else:
                        removals.pop(key, None)

    def on_resilience_event(self, error) -> None:
        self.events.append(error)
        self._stats.counters["resilience_events"] += 1
        if len(self.resilience_samples) < 20:
            self.resilience_samples.append(str(error)[:200])

    # ---- request scopes and hints ------------------------------------------

    def set_request_context(self, handle: Handle, kv_hints) -> None:
        hints = msgspec.to_builtins(kv_hints) if kv_hints is not None else None
        self._call(self._set_context, handle, hints)

    def _set_context(self, handle: Handle, hints) -> None:
        scope = f"{len(handle.rid)}:{handle.rid}:{handle.attempt_id}"
        self._scopes[handle] = scope
        self._request_scopes[handle.rid] = scope
        if hints is not None:
            try:
                envelope = decode_hints(hints)
                started = time.perf_counter()
                self._kvcr.submit_hint(envelope, request_id=scope)
                self._stats.call("submit_hint", time.perf_counter() - started)
            except ValueError:
                # SGLang envelopes may carry actions that KVCR does not consume.
                self._stats.counters["hints_rejected"] += 1

    def release_request_context(self, handle: Handle) -> None:
        if not self._closed:
            self._call(self._release_context, handle)

    def _release_context(self, handle: Handle) -> None:
        scope = self._scopes.pop(handle, None)
        if scope is not None:
            if self._request_scopes.get(handle.rid) == scope:
                self._request_scopes.pop(handle.rid)
            self._released_scopes.add(scope)
            self._cleanup_scopes()

    def _cleanup_scopes(self) -> None:
        in_use = {load.scope for load in self._loads.values()}
        for scope in self._released_scopes - in_use:
            started = time.perf_counter()
            self._kvcr.discard_hint(scope)
            self._stats.call("discard_hint", time.perf_counter() - started)
            self._released_scopes.remove(scope)

    # ---- lookup -------------------------------------------------------------

    def lookup(self, rid: str, hashes: list[str]) -> tuple[list[int], list[bool]]:
        """Every restorable prefix length in pages, and which pages are remote.

        The second value is instrumentation; upstream returns the first only.
        """
        return self._call(self._lookup, rid, tuple(hashes))

    def _lookup(self, rid: str, keys: tuple[str, ...]):
        scope = self._request_scopes.get(rid)
        encoded = [self.encode_key(key) for key in keys]
        started = time.perf_counter()
        result = self._kvcr.query(encoded, request_id=scope)
        self._stats.call("query", time.perf_counter() - started, 1, len(encoded))
        if len(result) != len(encoded):
            raise RuntimeError("KVCR query returned an incomplete result")
        found = [
            (status.value, tier.value if tier is not None else None)
            for status, tier in result
        ]
        hits = [entry in _HIT for entry in found]
        missing = [0]
        for hit in hits:
            missing.append(missing[-1] + (not hit))
        # ALL_PAGES hit policy: every page before the end must exist.
        valid = [end for end in range(1, len(keys) + 1) if missing[end] == missing[0]]
        return valid, [entry == _REMOTE for entry in found]

    # ---- load ---------------------------------------------------------------

    def load(self, rid: str, hashes: list[str], locations: list[int]) -> bool:
        keys = tuple(self.encode_key(key) for key in hashes)
        if len(keys) != len(locations):
            raise ValueError("a load needs one location per page")
        prepared = (Transfer(keys, tuple(locations)),) if keys else ()
        return self._call(self._load, rid, prepared)

    def _load(self, rid: str, prepared: tuple) -> bool:
        scope = self._request_scopes.get(rid)
        if not prepared:
            return False
        if rid in self._loads:
            raise RuntimeError(f"KVCR load already accepted for {rid}")
        load = _Load(rid, scope, prepared)
        self._loads[rid] = load
        self._queued.append(load)
        return True

    def start_layer_wise_loading(self) -> int:
        if self._closed:
            raise RuntimeError("KVCR linker is closed")
        if self._fatal is not None:
            raise self._fatal
        # Synchronous load/start calls keep this queue visible to the scheduler.
        if not self._queued:
            return -1
        return self._call(self._start)

    def _start(self) -> int:
        if not self._queued:
            return -1
        loads, self._queued = tuple(self._queued), []
        index = self.layer_done_counter.update_producer()
        batch = _Batch(index, loads, self._layer_work(loads), [1] * self.num_layers)
        self._batch_results[index] = batch.completion
        self._batches.append(batch)
        return index

    def finish_layer_wise_loading(self, counter_index: int) -> bool:
        if counter_index < 0:
            return True
        # Layer readiness alone is not terminal native ownership: wait for the
        # whole batch to drain, including failed operations.
        completion = self._call(self._batch_results.pop, counter_index)
        landed = completion.result(self._call_timeout)
        self.layer_done_counter.finish(counter_index)
        return landed

    def batch_completion(self, counter_index: int) -> Future:
        """Instrumentation (mesh warm): the batch's future, without waiting."""
        return self._call(self._batch_results.pop, counter_index)

    def _layer_work(self, loads):
        for layer in range(self.num_layers):
            for load in loads:
                for blocks in self._mappings(load.transfers, layer):
                    yield layer, load, blocks
            # The sentinel accounts for work not yet submitted in this layer.
            yield layer, None, None

    def _mappings(self, prepared, layer: int | None = None) -> list[dict]:
        """Upstream ``KVCRLayout.mappings``: named pieces of every page, or of
        one layer."""
        from kvcr.types import MemoryRef

        batches: list[dict] = []
        occurrences: Counter = Counter()
        endpoint = self._endpoint
        for transfer in prepared:
            if layer is None:
                labels = [label for comp in self._components for label in comp]
            else:
                labels = [comp[layer] for comp in self._components]
            for key, location in zip(transfer.keys, transfer.locations):
                batch_index = occurrences[key]
                occurrences[key] += 1
                if batch_index == len(batches):
                    batches.append({})
                batches[batch_index][key] = [
                    MemoryRef(
                        end_point_name=endpoint, label=label, element_index=location
                    )
                    for label in labels
                ]
        return batches

    # ---- offload ------------------------------------------------------------

    def offload(self, hashes: list[str], locations: list[int]) -> bool:
        keys = {self.encode_key(key) for key in hashes}
        if not keys:
            return False
        return self._call(self._offload, (tuple(hashes), tuple(locations)), keys)

    def _offload(self, transfers, keys: set[bytes]) -> bool:
        self._unacknowledged_keys.update(keys)
        self._offloads.append(_Offload(transfers, tuple(self._loads.values()), keys))
        return True

    def _prepare_offload(self, transfers) -> list[dict]:
        hashes, locations = transfers
        prepared = (Transfer(tuple(self.encode_key(key) for key in hashes), locations),)
        key_bytes = self._key_bytes
        batches = []
        for blocks in self._mappings(prepared):
            batch: dict = {}
            size = descriptors = 0
            for key, refs in blocks.items():
                if key_bytes > MAX_DEPOSIT_BYTES or len(refs) > MAX_DEPOSIT_DESCRIPTORS:
                    raise ValueError("KVCR page exceeds native deposit limits")
                if batch and (
                    size + key_bytes > MAX_DEPOSIT_BYTES
                    or descriptors + len(refs) > MAX_DEPOSIT_DESCRIPTORS
                ):
                    batches.append(batch)
                    batch, size, descriptors = {}, 0, 0
                batch[key] = refs
                size += key_bytes
                descriptors += len(refs)
            if batch:
                batches.append(batch)
        # Prepare before submitting: invalid pages cannot leave partial work.
        return batches

    def num_completed_offloads(self) -> int:
        return self._completed_offloads.qsize()

    def pop_completed_offload(self) -> tuple[bool, float]:
        """Whether the oldest finished offload landed, and when it finished
        (the time is instrumentation; upstream returns the first only)."""
        return self._call(self._pop_completed_offload)

    def _pop_completed_offload(self) -> tuple[bool, float]:
        success, keys, finished = self._completed_offloads.get_nowait()
        unacknowledged = self._unacknowledged_keys
        for key in keys:
            unacknowledged[key] -= 1
            if not unacknowledged[key]:
                del unacknowledged[key]
        return success, finished

    def pop_storage_removals(self) -> list[str]:
        return self._call(self._pop_storage_removals)

    def _pop_storage_removals(self) -> list[str]:
        # A delayed acknowledgement must not resurrect an eviction.
        removals = self.removals
        waiting = self._unacknowledged_keys
        return [removals.pop(key) for key in list(removals) if key not in waiting]

    def drop_queued_offloads(self) -> int:
        """Instrumentation: forget offloads not yet started, to end a run."""
        return self._call(self._drop_queued_offloads)

    def _drop_queued_offloads(self) -> int:
        kept = [task for task in self._offloads if task.active or task.work is not None]
        dropped = len(self._offloads) - len(kept)
        for task in kept:
            task.exhausted = True
        self._offloads = deque(kept)
        return dropped

    # ---- progress -----------------------------------------------------------

    def _fail_batch(self, batch: _Batch, error, *, fatal: bool = False) -> None:
        batch.error = error
        batch.exhausted = True
        batch.work = None
        self.layer_done_counter.fail(batch.index, error)
        if fatal and not batch.completion.done():
            batch.completion.set_exception(error)

    def progress(self) -> None:
        """One owner iteration's ``_progress`` (for inline tests)."""
        self._progress()
        self._log_stats()

    def _progress(self) -> None:
        kvcr = self._kvcr
        started = time.perf_counter()
        results = list(kvcr.poll_completed())
        elapsed = time.perf_counter() - started
        if results:
            self._stats.call("poll", elapsed, 1, len(results))
        else:
            self._empty_polls += 1
            self._empty_poll_seconds += elapsed
        events = self.events
        while events:
            event = events.popleft()
            state = getattr(event, "state", None)
            if state == "uncertain":
                self._hazards.add(event.op_handle)
            elif state == "quiesced":
                self._hazards.discard(event.op_handle)
            else:
                raise event
        operations = self._operations
        for handle, result in results:
            operation = operations[handle]
            operation.result = result
            load = operation.load
            if load is not None:
                load.layers += 1
            if set(result) != operation.keys or not all(
                entry.success for entry in result.values()
            ):
                owner = operation.owner
                if isinstance(owner, _Batch):
                    for key in operation.keys:
                        entry = result.get(key)
                        if entry is None or not entry.success:
                            # Several layers name a page: its first failure counts.
                            load.failed.setdefault(key, entry)
                    self._fail_batch(
                        owner, RuntimeError(f"KVCR delivery {handle} failed")
                    )
                else:
                    owner.success = False
                    owner.exhausted = True
        # Conservatively retain every accepted operation while DMA is uncertain.
        if self._hazards:
            return
        for handle, operation in list(operations.items()):
            if operation.result is None:
                continue
            owner = operation.owner
            owner.active -= 1
            if isinstance(owner, _Batch):
                owner.remaining[operation.layer] -= 1
                self._complete_layer(owner, operation.layer)
            del operations[handle]
        deliveries = sum(isinstance(op.owner, _Batch) for op in operations.values())
        batches = self._batches
        for batch in batches:
            while not batch.exhausted and deliveries < MAX_DELIVERIES:
                try:
                    layer, load, blocks = next(batch.work)
                except StopIteration:
                    batch.exhausted = True
                    break
                if blocks is None:
                    batch.remaining[layer] -= 1
                    self._complete_layer(batch, layer)
                    continue
                started = time.perf_counter()
                handle = kvcr.deliver(blocks, request_id=load.scope)
                self._stats.call(
                    "deliver", time.perf_counter() - started, 1, len(blocks)
                )
                operations[handle] = _Operation(batch, set(blocks), layer, load)
                batch.active += 1
                batch.remaining[layer] += 1
                deliveries += 1
        # Completion queues preserve admission order even if transport does not.
        while batches and batches[0].exhausted and not batches[0].active:
            batch = batches.popleft()
            for load in batch.loads:
                load.done, load.error = True, batch.error
                del self._loads[load.rid]
            self._cleanup_scopes()
            self.batch_info[batch.index] = BatchInfo(
                batch.loads, batch.started, time.perf_counter()
            )
            if not batch.completion.done():
                batch.completion.set_result(batch.error is None)
        if self._offloads:
            task = self._offloads[0]
            if any(not load.done for load in task.dependencies):
                return
            if any(load.error for load in task.dependencies):
                task.success, task.exhausted = False, True
            if not task.exhausted and not task.active:
                if not self._commands.empty():
                    return
                if task.work is None:
                    try:
                        task.work = iter(self._prepare_offload(task.transfers))
                    except Exception:  # noqa: BLE001 - as upstream
                        task.success, task.exhausted = False, True
                if not task.exhausted:
                    blocks = next(task.work, None)
                    if blocks is None:
                        task.exhausted = True
                    else:
                        started = time.perf_counter()
                        handle = kvcr.deposit(blocks)
                        self._stats.call(
                            "deposit", time.perf_counter() - started, 1, len(blocks)
                        )
                        operations[handle] = _Operation(task, set(blocks))
                        task.active += 1
            if task.exhausted and not task.active:
                self._completed_offloads.put(
                    (task.success, task.keys, time.perf_counter())
                )
                self._offloads.popleft()

    def _complete_layer(self, batch: _Batch, layer: int) -> None:
        if not batch.error and batch.remaining[layer] == 0:
            self.layer_done_counter.complete(batch.index, layer)

    # ---- state the engine thread may read without a round trip -------------

    def busy(self) -> bool:
        return bool(self._offloads or self._batches or self._queued or self._operations)

    def fatal(self) -> BaseException | None:
        return self._fatal

    # ---- shutdown -----------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        if self._inline:
            self._close()
            return
        try:
            self._call(self._close)
        finally:
            self._thread.join(self._call_timeout)

    def _close(self) -> None:
        try:
            self._host.close()
        finally:
            for batch in self._batches:
                self._fail_batch(batch, RuntimeError("KVCR linker closed"))
                if not batch.completion.done():
                    batch.completion.set_result(False)
            with self._command_lock:
                self._closed = True
