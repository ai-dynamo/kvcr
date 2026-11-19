# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""A GPU-free engine that calls KVCR the way vLLM's KVCR tier does.

Modelled on vLLM ``origin/main``:

- ``vllm/v1/kv_offload/tiering/kvcr/manager.py`` (``KVCRSecondaryTierManager``)
- ``vllm/v1/kv_offload/tiering/manager.py`` (``TieringOffloadingManager``)
- ``vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py``
- ``vllm/v1/kv_offload/cpu/policies/lru.py``

One step is, in vLLM's order: schedule (lookups, then store jobs), schedule
end (poll, serve pins, flush promotions into ``deliver``), the forward pass
(a GIL-released sleep), then output handling (store completions into
``deposit``, finished requests, ``get_stats``).

What KVCR sees, and must keep seeing if this file changes:

- ``query`` is one key per call, only on a primary-tier miss, and the scan
  runs to the end of the prompt unless a key misses.
- A request's promotions go out as one ``deliver`` per step, into primary-tier
  slots. ``fetch`` and ``release`` are never used.
- Every GPU->CPU store is followed by a ``deposit`` of the same blocks.
- ``align_sequence`` covers the request's whole prefix up to the job's last
  block after every completed deliver or deposit, and the whole request at
  finish, followed by ``discard_hint``.
- ``poll_completed`` runs on the first lookup of a step, again for every
  request that stores, and at schedule end if it has not run yet.
- Pins are resolved once per step from the primary tier and released at the
  next poll. ``get_stats`` runs every step.

The device tier is bookkeeping: a request is admitted when all of its blocks
fit beside the running requests', first come first served, and every token a
step computes, prompt or answer, comes out of ``max_batched_tokens``.
"""

from __future__ import annotations

import time
from collections import deque

from kvcr.types import CacheTier, MemoryRef, QueryStatus

from .config import RunConfig
from .engine import count_failure, failure_cause, fetch_hint
from .ids import (
    KeyCache,
    block_id,
    id_to_key,
    key_to_id,
    keys_to_ids,
    segment_ids,
    warm_stream,
)
from .metrics import Window
from .primary_tier import KeyEvent, PrimaryTier, StreamCache
from .protocol import (
    TIER_DEVICE,
    TIER_G2,
    TIER_PRIMARY,
    Done,
    InventoryDelta,
    Request,
)
from .router import PrefixTracker

_LOAD, _STORE, _WARM = 0, 1, 2
_WARM_ATTEMPTS = 5
_FETCHING, _Q_HIT, _FETCHABLE = (
    QueryStatus.FETCHING,
    QueryStatus.HIT,
    QueryStatus.FETCHABLE,
)
_REMOTE = CacheTier.REMOTE_G2
_LOCAL_G2 = CacheTier.LOCAL_G2


class _Req:
    __slots__ = (
        "rid", "req_id", "keys", "all_keys", "prompt_tokens", "output_tokens",
        "arrived", "started", "first_token", "rank", "gpu_rank", "gpu_hit",
        "computed", "sched", "next_stored", "promote",
        "local_ok", "remote_ok", "failed", "timed_out", "primary_hit",
        "prefill_tokens", "lookups", "hinted", "looked", "segments", "all_segments",
        "pins", "need",
    )  # fmt: skip


class _Job:
    __slots__ = ("kind", "req", "keys", "remaining", "ok", "bad", "end",
                 "started", "remote", "hinted", "warm")  # fmt: skip

    def __init__(self, kind, req, keys, end, remote=None, hinted=None):
        self.kind = kind
        self.req = req
        self.keys = keys
        self.remaining = len(keys)
        self.ok: list[bytes] = []
        self.bad: dict[bytes, object] = {}  # failed key -> KVCR's result entry
        self.end = end
        self.started = time.perf_counter()
        self.remote = remote
        self.hinted = hinted
        self.warm = None


class VllmEngine:
    def __init__(self, config: RunConfig, window: Window, worker: int) -> None:
        tiers, engine = config.tiers, config.engine
        self._tiers = tiers
        self._cfg = engine
        self._window = window
        self._worker = worker
        self._block_tokens = tiers.tokens_per_block
        self._verify = engine.verify
        self._warm_slots = config.phases.warm_concurrency
        # Slot 0 holds this worker's warm block; the next ones are warm scratch.
        self._primary = PrimaryTier(tiers.primary_blocks, 1 + self._warm_slots)
        self._gpu = StreamCache(tiers.gpu_blocks)
        self._refs: list[MemoryRef] = []  # one per primary slot
        self._key_cache = KeyCache()
        self._waiting: list[_Req] = []
        self._running: list[_Req] = []
        self._jobs: dict[int, _Job] = {}
        self._stores: list[tuple[_Req, list[tuple[bytes, int]], int]] = []
        self._loads: list[list[bytes]] = []
        self._polled = False
        # Framework pin adapter state.
        self._next_pin = 0
        self._next_handle = 0
        self._pending_pins: dict[int, tuple[bytes, ...]] = {}
        self._completed_pins: list = []
        self._pins: dict[str, list[bytes]] = {}
        self._pin_released: list[list[bytes]] = []
        self._g2_events: list[KeyEvent] = []
        # What the router is told the host tiers hold.
        self._primary_seen = PrefixTracker(TIER_PRIMARY)
        self._g2_seen = PrefixTracker(TIER_G2)
        self._resilience: list[str] = []
        # Mesh warm.
        self._warm_queue: deque[tuple[int, str, int]] = deque()
        self._warm_free: list[int] = []
        self._warm_active = 0
        self._warm_ok = 0
        self._warm_failed = 0
        self._kvcr = None
        self._host = None
        self._agent = ""
        self._peers: list[tuple[str, str]] = []

    # ---- KVCRBindings callbacks (all on the engine thread) -----------------

    def request_pin(self, keys):
        request = self._next_pin
        self._next_pin += 1
        self._pending_pins[request] = tuple(keys)
        return request

    def poll_pin_results(self):
        done, self._completed_pins = self._completed_pins, []
        return done

    def release_pin(self, handle) -> bool:
        keys = self._pins.pop(handle, None)
        if keys is not None:
            self._pin_released.append(keys)
        return True

    def cancel_pin_request(self, request) -> None:
        self._pending_pins.pop(request, None)

    def record_inventory(self, event) -> None:
        if event.tier is _LOCAL_G2:
            self._g2_events.append((event.removed, event.keys))

    def on_resilience_event(self, error: Exception) -> None:
        self._window.counters["resilience_events"] += 1
        if len(self._resilience) < 20:
            self._resilience.append(str(error)[:200])

    # ---- setup -------------------------------------------------------------

    def attach(self, host) -> None:
        self._host = host
        self._kvcr = host.kvcr
        self._agent = host.agent_name
        self._refs = [
            MemoryRef(end_point_name=self._agent, element_index=slot)
            for slot in range(self._tiers.primary_blocks)
        ]
        warm = block_id(warm_stream(self._worker), 0)
        key = id_to_key(warm)
        self._primary.insert_reserved(key, 0)
        host.stamp(0, warm)
        if self._tiers.g2_blocks:
            # Also in the KVCR pool, so a pinless source can serve it.
            op = self._kvcr.deposit({key: [self._ref(0)]})
            self._primary.prepare_read((key,))
            self._jobs[op] = _Job(_STORE, None, [key], 0)

    def set_peers(self, peers: list[tuple[str, str]]) -> None:
        self._peers = peers

    def _ref(self, slot: int) -> MemoryRef:
        return self._refs[slot]

    # ---- requests ----------------------------------------------------------

    def submit(self, request: Request) -> None:
        r = _Req()
        r.rid = request.rid
        r.req_id = f"r{request.rid}"
        r.keys = self._key_cache.keys(request.segments)
        tail = self._key_cache.keys((request.tail,))
        r.all_keys = r.keys + tail
        r.segments = request.segments
        r.all_segments = (*request.segments, request.tail)
        r.prompt_tokens = request.prompt_tokens
        r.output_tokens = max(1, request.output_tokens)
        r.arrived = time.perf_counter()
        r.started = r.first_token = r.looked = 0.0
        r.rank = self._primary.new_rank()
        r.gpu_rank = self._gpu.new_rank()
        r.gpu_hit = r.computed = r.sched = r.next_stored = 0
        r.promote = []
        r.local_ok = r.remote_ok = r.failed = r.timed_out = r.primary_hit = 0
        r.prefill_tokens = r.lookups = r.hinted = 0
        r.pins = ()
        r.need = 0
        if len(r.all_keys) > self._gpu.capacity:
            raise ValueError(
                f"request {r.rid} needs {len(r.all_keys)} device blocks, more "
                f"than tiers.gpu_blocks ({self._gpu.capacity})"
            )
        if request.hint_source >= 0 and request.hint_blocks > 0:
            r.hinted = request.hint_blocks
            # vLLM on_new_request: submit_hint only when the router sent one.
            hashes = segment_ids(request.segments)[: request.hint_blocks]
            endpoint = self._peers[request.hint_source][1]
            hint = fetch_hint(r.req_id, endpoint, hashes)
            t = time.perf_counter()
            self._kvcr.submit_hint(hint, r.req_id)
            self._window.call("submit_hint", time.perf_counter() - t, 1, len(hashes))
        self._waiting.append(r)

    # ---- one engine step ---------------------------------------------------

    def step(self) -> list[Done]:
        perf = time.perf_counter
        window = self._window
        cfg = self._cfg
        t_step = perf()
        kvcr_before = sum(window.secs.values())
        budget = cfg.max_batched_tokens
        prefill = 0
        scheduled: list[_Req] = []

        # Schedule: running requests first, then the waiting queue.
        for r in self._running:
            if budget <= 0:
                break
            remaining = r.prompt_tokens - r.computed
            if remaining > 0:
                n = min(remaining, budget)
                prefill += n
            else:
                n = 1  # a decode step is one token of the budget
            budget -= n
            r.sched = n
            scheduled.append(r)
        if self._waiting:
            waiting = []
            admitting = True
            for r in self._waiting:
                if (
                    not admitting
                    or len(self._running) >= cfg.max_running
                    or budget <= 0
                ):
                    waiting.append(r)
                    continue
                hit = self._lookup(r)
                if hit is None:
                    waiting.append(r)  # deferred: retried next step
                    continue
                need = len(r.all_keys) - r.gpu_hit
                pins = self._gpu.admit(r.segments, need)
                if pins is None:
                    # No room on the device: it waits, and so does every
                    # request behind it. That wait is not cache resolution.
                    admitting = False
                    r.looked = 0.0
                    window.counters["device_alloc_failures"] += 1
                    waiting.append(r)
                    continue
                r.pins, r.need = pins, need
                self._start(r, hit)
                n = min(r.prompt_tokens - r.computed, budget)
                budget -= n
                prefill += n
                r.sched = n
                scheduled.append(r)
            self._waiting = waiting

        # build_connector_meta: GPU->CPU store jobs for what this step computes.
        for r in scheduled:
            self._build_store(r)

        # on_schedule_end.
        if not self._polled:
            self._poll()
        if self._pending_pins:
            self._serve_pins()
        self._polled = False
        self._flush_promotions()
        if self._warm_queue:
            self._warm_tick()
        t_scheduled = perf()

        # Forward pass: the GIL is released, as it is while the GPU runs.
        if scheduled:
            time.sleep(cfg.step_ms / 1e3 + prefill * cfg.prefill_us_per_token / 1e6)
        else:
            time.sleep(cfg.idle_sleep_ms / 1e3)
        t_output = perf()
        # Where the engine thread is neither in KVCR nor in this harness.
        window.counters["forward_seconds" if scheduled else "idle_seconds"] += (
            t_output - t_scheduled
        )

        # update_from_output.
        self._complete_copies()
        dones: list[Done] = []
        if scheduled:
            now = perf()
            for r in scheduled:
                before = r.computed
                r.computed = before + r.sched
                if before < r.prompt_tokens:
                    if r.computed < r.prompt_tokens:
                        continue
                    r.first_token = now  # prefill done: first token sampled
                if r.computed >= r.prompt_tokens + r.output_tokens - 1:
                    dones.append(self._finish(r, now))
        t = perf()
        stats = self._kvcr.get_stats()
        window.call("get_stats", perf() - t)
        window.merge_telemetry(stats)
        t_end = perf()
        window.hists["step_scheduler"].add(t_scheduled - t_step + t_end - t_output)
        window.hists["step_kvcr"].add(sum(window.secs.values()) - kvcr_before)
        window.counters["steps"] += 1
        if scheduled:
            window.counters["busy_steps"] += 1
        return dones

    def _lookup(self, r: _Req) -> int | None:
        """vLLM ``_maximal_prefix_lookup`` over ``TieringOffloadingManager.lookup``.

        Returns the number of offloaded blocks after the device-cached prefix,
        or None when the request must wait for a promotion.
        """
        perf = time.perf_counter
        keys = r.keys
        gpu = self._gpu.prefix(r.segments)
        r.gpu_hit = gpu
        if not r.looked:
            r.looked = perf()
        r.lookups += 1
        end = len(keys)
        if gpu >= end:
            return 0
        if not self._polled:
            self._poll()
        primary = self._primary
        scan, reserve = primary.scan, primary.reserve
        rank = r.rank
        query = self._kvcr.query
        req_id = r.req_id
        promote = r.promote
        defer = False
        spent = 0.0
        queries = 0
        position = gpu
        while True:
            # CPU-tier hits and writes in flight, up to the first key it lacks.
            position, pending = scan(keys, position, rank)
            if pending:
                defer = True
            if position >= end:
                break
            key = keys[position]
            t = perf()
            status, tier = query((key,), req_id)[0]
            spent += perf() - t
            queries += 1
            if status is _FETCHING:
                defer = True
            elif status is _Q_HIT or status is _FETCHABLE:
                slot = reserve(key, rank, position)
                if slot is None:
                    break  # primary tier full: treated as a miss
                promote.append((key, slot, tier is _REMOTE, position))
                defer = True
            else:
                break
            position += 1
        if queries:
            self._window.call("query", spent, queries, queries)
        return None if defer else position - gpu

    def _start(self, r: _Req, hit: int) -> None:
        gpu = r.gpu_hit
        tokens = self._block_tokens
        # At least one prompt token is always computed.
        r.computed = min((gpu + hit) * tokens, r.prompt_tokens - 1)
        r.prefill_tokens = r.prompt_tokens - r.computed
        # Blocks promoted twice (evicted while the request waited) count twice
        # on the right, so clamp: this is a lower bound.
        r.primary_hit = max(0, hit - r.local_ok - r.remote_ok)
        r.started = time.perf_counter()
        if hit:
            load = r.keys[gpu : gpu + hit]
            # CPU->GPU load job: the slots stay pinned until the step ends.
            self._primary.prepare_read(load)
            self._loads.append(load)
            r.next_stored = gpu + hit  # chunk-level policy skips loaded chunks
        self._running.append(r)

    def _build_store(self, r: _Req) -> None:
        # Prompt blocks only (vLLM's ``offload_prompt_only`` default).
        limit = min((r.computed + r.sched) // self._block_tokens, len(r.keys))
        start = r.next_stored
        if limit <= start:
            return
        keys = r.keys[start:limit]
        # TieringOffloadingManager.prepare_store polls unconditionally.
        self._poll()
        taken = self._primary.prepare_write(keys, r.rank, range(start, limit))
        if taken is None:
            return  # the CPU tier is full: the range is tried again next step
        r.next_stored = limit
        self._primary.access(keys, r.rank, start)
        if taken:
            self._stores.append((r, taken, limit - 1))

    def _complete_copies(self) -> None:
        primary = self._primary
        if self._stores:
            perf = time.perf_counter
            host = self._host
            refs = self._refs
            for r, taken, end in self._stores:
                keys = [key for key, _ in taken]
                host.stamp_many([slot for _, slot in taken], keys)
                # complete_store cascades to KVCR; the slots stay pinned (one
                # reader) until the deposit finishes.
                primary.complete_write(keys, True, 1)
                blocks = {key: [refs[slot]] for key, slot in taken}
                t = perf()
                op = self._kvcr.deposit(blocks)
                self._window.call("deposit", perf() - t, 1, len(blocks))
                self._jobs[op] = _Job(_STORE, r, keys, end)
            self._stores = []
        if self._loads:
            for load in self._loads:
                primary.complete_read(load)
            self._loads = []

    def _flush_promotions(self) -> None:
        """One ``deliver`` per request per step (``_flush_pending_promotions``)."""
        perf = time.perf_counter
        refs = self._refs
        for r in self._waiting:
            if not r.promote:
                continue
            promote, r.promote = r.promote, []
            blocks = {key: [refs[slot]] for key, slot, _, _ in promote}
            t = perf()
            op = self._kvcr.deliver(blocks, request_id=r.req_id)
            self._window.call("deliver", perf() - t, 1, len(blocks))
            keys = [key for key, _, _, _ in promote]
            remote = {key for key, _, is_remote, _ in promote if is_remote}
            # Blocks the router said the source holds; with opportunistic
            # query the rest of the prompt is requested too and cannot succeed.
            hinted = {key for key, _, _, position in promote if position < r.hinted}
            end = max(position for _, _, _, position in promote)
            self._jobs[op] = _Job(_LOAD, r, keys, end, remote, hinted)

    def _poll(self) -> None:
        """``KVCRSecondaryTierManager._poll_finished_jobs`` and its completions."""
        self._polled = True
        perf = time.perf_counter
        window = self._window
        t = perf()
        completed = list(self._kvcr.poll_completed())
        window.call("poll", perf() - t, 1, len(completed))
        if completed:
            primary = self._primary
            aligned: list[tuple[_Req, int]] = []
            for op, entries in completed:
                job = self._jobs.get(op)
                if job is None:
                    continue
                job.remaining -= len(entries)
                ok, bad = job.ok, job.bad
                for key, entry in entries.items():
                    if entry.success:
                        ok.append(key)
                    else:
                        bad[key] = entry
                if job.remaining > 0:
                    continue
                del self._jobs[op]
                if job.kind == _WARM:
                    self._warm_done(job)
                    continue
                r = job.req
                elapsed = perf() - job.started
                if job.kind == _LOAD:
                    primary.complete_write(job.ok, True)
                    primary.complete_write(job.bad, False)
                    self._loaded(r, job, elapsed)
                else:
                    primary.complete_read(job.keys)
                    window.hists["deposit"].add(elapsed)
                    window.counters["deposit_blocks_ok"] += len(job.ok)
                    window.counters["deposit_blocks_failed"] += len(job.bad)
                if r is None:
                    continue
                end = job.end
                # Preserve ordering across requests, which may share keys.
                if aligned and aligned[-1][0] is r:
                    end = max(end, aligned.pop()[1])
                aligned.append((r, end))
            for r, end in aligned:
                # Earlier resident keys are included even when only the tail moved.
                keys = r.all_keys[: end + 1]
                t = perf()
                self._kvcr.align_sequence(keys)
                window.call("align_sequence", perf() - t, 1, len(keys))
        if self._pin_released:
            for keys in self._pin_released:
                self._primary.complete_read(keys)
            self._pin_released = []

    def _loaded(self, r: _Req, job: _Job, elapsed: float) -> None:
        window = self._window
        remote = sum(1 for key in job.ok if key in job.remote)
        local = len(job.ok) - remote
        r.remote_ok += remote
        r.local_ok += local
        r.failed += len(job.bad)
        window.hists["deliver_remote" if job.remote else "deliver_local"].add(elapsed)
        window.counters["deliver_blocks_remote"] += remote
        window.counters["deliver_blocks_local"] += local
        window.counters["deliver_blocks_failed"] += len(job.bad)
        if job.bad:
            counters = window.counters
            counters["deliver_ops_partial" if job.ok else "deliver_ops_failed"] += 1
            # Blocks asked of a peer. Those the hint named are counted by
            # KVCR's status; the opportunistic query asks the peer for the
            # rest of the prompt too, which it cannot have.
            peers, hinted = job.remote, job.hinted
            for key, entry in job.bad.items():
                if key not in peers:
                    continue
                if key in hinted:
                    cause = count_failure(counters, entry)
                else:
                    counters["deliver_blocks_failed_beyond_hint"] += 1
                    cause = failure_cause(entry)
                if cause == "timeout":
                    r.timed_out += 1
        if self._verify != "none" and job.ok:
            check = job.ok if self._verify == "full" else job.ok[:1]
            host, slot = self._host, self._primary.slot
            for key in check:
                window.counters["bytes_checked"] += 1
                if host.stamp_at(slot(key)) != key_to_id(key):
                    window.counters["bytes_mismatch"] += 1

    def _serve_pins(self) -> None:
        """``_FrameworkPinAdapter.process_pending``: once per step, primary only."""
        pending, self._pending_pins = self._pending_pins, {}
        primary = self._primary
        slot, all_refs = primary.slot, self._refs
        window = self._window
        for request, keys in pending.items():
            hit = primary.readable(keys)
            window.counters["pin_requests"] += 1
            if not hit:
                self._completed_pins.append((request, None))
                window.counters["pin_misses"] += 1
                continue
            primary.prepare_read(hit)
            handle = f"pin-{self._next_handle}"
            self._next_handle += 1
            self._pins[handle] = hit
            refs: dict = dict.fromkeys(keys)
            for key in hit:
                refs[key] = [all_refs[slot(key)]]
            self._completed_pins.append((request, (handle, refs)))
            window.counters["pin_keys"] += len(hit)

    def _finish(self, r: _Req, now: float) -> Done:
        self._running.remove(r)
        window = self._window
        keys = r.keys
        # on_request_finished: final recency, then drop the request's hints.
        if keys:
            t = time.perf_counter()
            self._kvcr.align_sequence(keys, True)
            window.call("align_sequence", time.perf_counter() - t, 1, len(keys))
        t = time.perf_counter()
        self._kvcr.discard_hint(r.req_id)
        window.call("discard_hint", time.perf_counter() - t)
        # Everything the request computed is now in the device cache.
        self._gpu.release(r.pins, r.need)
        self._gpu.insert(r.all_segments, r.gpu_rank)
        ttft = r.first_token - r.arrived
        # Queueing for the engine's token budget is not KVCR's doing; the time
        # from the first lookup to the request starting is.
        load = r.started - r.looked
        window.hists["ttft"].add(ttft)
        window.hists["queue"].add(r.looked - r.arrived)
        window.hists["cache_resolve"].add(load)
        if r.remote_ok or r.failed:
            window.hists["cache_resolve_remote"].add(load)
        window.counters["requests"] += 1
        window.counters["prompt_blocks"] += len(r.keys)
        window.counters["gpu_hit_blocks"] += r.gpu_hit
        window.counters["primary_hit_blocks"] += r.primary_hit
        window.counters["prefill_tokens"] += r.prefill_tokens
        window.counters["lookups"] += r.lookups
        return Done(
            rid=r.rid,
            ttft_ms=ttft * 1e3,
            load_ms=load * 1e3,
            prompt_blocks=len(r.keys),
            gpu_blocks=r.gpu_hit,
            primary_blocks=r.primary_hit,
            local_blocks=r.local_ok,
            remote_blocks=r.remote_ok,
            failed_blocks=r.failed,
            recomputed_tokens=r.prefill_tokens,
            timed_out_blocks=r.timed_out,
        )

    # ---- mesh warm ---------------------------------------------------------

    def start_warm(self, peers: list[tuple[int, str]]) -> None:
        self._warm_queue.extend((peer, endpoint, 0) for peer, endpoint in peers)
        self._warm_free = list(range(1, 1 + self._warm_slots))

    def warm_status(self) -> tuple[int, int, int]:
        pending = len(self._warm_queue) + self._warm_active
        return pending, self._warm_ok, self._warm_failed

    def _warm_tick(self) -> None:
        kvcr = self._kvcr
        while self._warm_queue and self._warm_free:
            peer, endpoint, attempt = self._warm_queue.popleft()
            slot = self._warm_free.pop()
            bid = block_id(warm_stream(peer), 0)
            key = id_to_key(bid)
            req_id = f"warm-{peer}-{attempt}"
            kvcr.submit_hint(fetch_hint(req_id, endpoint, [bid]), req_id)
            kvcr.query((key,), req_id)
            op = kvcr.deliver({key: [self._ref(slot)]}, request_id=req_id)
            job = _Job(_WARM, None, [key], 0)
            job.warm = (peer, endpoint, attempt, slot, req_id, bid)
            self._jobs[op] = job
            self._warm_active += 1

    def _warm_done(self, job: _Job) -> None:
        peer, endpoint, attempt, slot, req_id, bid = job.warm
        self._kvcr.discard_hint(req_id)
        self._warm_active -= 1
        ok = bool(job.ok) and self._host.stamp_at(slot) == bid
        self._host.stamp(slot, 0)
        self._warm_free.append(slot)
        if ok:
            self._warm_ok += 1
        elif attempt + 1 < _WARM_ATTEMPTS:
            self._warm_queue.append((peer, endpoint, attempt + 1))
        else:
            self._warm_failed += 1

    # ---- reporting ---------------------------------------------------------

    def take_inventory(self) -> list[InventoryDelta]:
        gpu = self._gpu
        out: list[InventoryDelta] = [(TIER_DEVICE, *event) for event in gpu.events]
        gpu.events = []
        g2_events, self._g2_events = self._g2_events, []
        for tracker, events in (
            (self._primary_seen, self._primary.take_events()),
            (self._g2_seen, g2_events),
        ):
            for removed, keys in events:
                tracker.apply(removed, keys_to_ids(keys))
            out.extend(tracker.take())
        return out

    def abort_requests(self) -> None:
        """Drop every request; KVCR operations already issued still complete."""
        primary = self._primary
        for r in self._waiting + self._running:
            self._kvcr.discard_hint(r.req_id)
            # Slots reserved for promotions that were never delivered.
            primary.complete_write([key for key, _, _, _ in r.promote], False)
            r.promote = []
            self._gpu.release(r.pins, r.need)
            r.pins, r.need = (), 0
        self._waiting = []
        self._running = []
        # Stores whose device->CPU copy will now never finish.
        for _, taken, _ in self._stores:
            primary.complete_write([key for key, _ in taken], False)
        self._stores = []
        for load in self._loads:
            self._primary.complete_read(load)
        self._loads = []

    def busy(self) -> bool:
        return bool(
            self._waiting
            or self._running
            or self._jobs
            or self._pins
            or self._pending_pins
        )

    def gauges(self) -> dict[str, float]:
        primary = self._primary
        return {
            "waiting": len(self._waiting),
            "running": len(self._running),
            "jobs": len(self._jobs),
            "pins": len(self._pins),
            "primary_occupancy": primary.occupancy(),
            "primary_fill": primary.fill_level(),
            "primary_evictions": primary.evictions,
            "primary_alloc_failures": primary.allocation_failures,
            "gpu_blocks": len(self._gpu),
            "gpu_reserved": self._gpu.live,
        }

    def resilience_samples(self) -> list[str]:
        return self._resilience

    def fill_level(self) -> float:
        return self._primary.fill_level()

    def begin_window(self) -> None:
        pass

    def end_window(self) -> None:
        pass

    def close(self) -> None:
        pass
