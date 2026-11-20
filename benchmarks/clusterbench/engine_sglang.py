# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""A GPU-free engine that uses KVCR the way SGLang's direct linker does.

Modelled on SGLang PR #45650 at revision ``002d6d8d`` (the one KVCR's quick
start pins), dense model, TP1:

- ``python/sglang/srt/managers/scheduler.py`` (the scheduler loop)
- ``python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py``
  (``Wrapper``: match, load_back, finish_loads, offload_nodes, the per-iteration
  event check)
- the linker itself, ported in ``linker_port.py``

This file is the scheduler side. One iteration, in SGLang's order:

1. ``check_hicache_events``: commit completed offloads, then ask the linker
   for pages KVCR removed. This runs on every iteration, idle ones too, and
   the scheduler does not sleep when idle (``--sleep-on-idle`` is off).
2. A request stopped mid-prefill is checkpointed: its new pages enter the
   tree and are offloaded.
3. Admission, first come first served: a request with at least one full page
   the device lacks gets ``set_request_context`` and ``lookup``. It needs
   device room for every token it will hold, cacheable or not; one that
   finds none (``NO_TOKEN``) stays queued and closes
   admission until a running request leaves; with nothing running it is
   matched again on every iteration. An admitted request with a host hit
   gets one ``load``.
4. A new prefill batch starts its queued loads and runs its forward, in which
   each layer waits for its own K and V, so the layers before it compute
   while it loads. Then it waits for the batch to drain. If the batch failed,
   every request that loaded in it is aborted: nothing is recomputed.
5. Results: finished prefills and finished requests insert their pages, and
   every node on the request's path that KVCR does not hold is offloaded,
   including pages just restored from KVCR.

Without a new prefill batch the running requests decode one token.

All KVCR calls happen on the linker's owner thread; this thread blocks on
each linker call and on loads.
"""

from __future__ import annotations

import gc
import time
from collections import deque

from .config import RunConfig
from .device_tree import DeviceTree, Node
from .engine import count_failure, fetch_hint, live_blocks
from .ids import (
    INDEX_BITS,
    NameCache,
    block_id,
    hash_to_id,
    page_hash,
    segment_ids,
    split_id,
    warm_stream,
)
from .linker_port import Handle, LinkerPort
from .metrics import Window
from .protocol import TIER_DEVICE, TIER_G2, Done, InventoryDelta, Request
from .router import PrefixTracker

_WARM_ATTEMPTS = 5


def forward_end(start: float, landed: list[float], per_layer: float) -> float:
    """When a forward that started at ``start`` ends: every attention layer
    waits for its K and V, so a layer computes once it has landed and the one
    before it has computed."""
    end = start
    for at in landed:
        end = max(end, at) + per_layer
    return end


class _Req:
    __slots__ = (
        "rid", "sid", "handle", "hint", "segments", "all_segments", "parents",
        "hashes", "prompt_tokens", "output_tokens", "prompt_pages", "match_pages",
        "total_pages", "live_pages", "arrived", "looked", "started", "first_token",
        "device_hit", "host_hit", "remote_pages", "slots", "path", "computed",
        "sched", "inserted", "walked", "lookups", "prefill_tokens", "loaded",
        "contexts", "remote_flags", "failed", "timed_out",
    )  # fmt: skip


class SglangEngine:
    def __init__(self, config: RunConfig, window: Window, worker: int) -> None:
        tiers, engine = config.tiers, config.engine
        self._tiers = tiers
        self._cfg = engine
        self._window = window
        self._worker = worker
        self._page_tokens = tiers.tokens_per_block
        self._layers = tiers.layers
        self._warm_slots = config.phases.warm_concurrency
        # Slot 0 holds this worker's warm page; the next ones are warm scratch.
        self._tree = DeviceTree(tiers.gpu_blocks, 1 + self._warm_slots)
        self._usable = tiers.gpu_blocks - 1 - self._warm_slots
        self._hash_cache = NameCache(page_hash)
        # KVCR evicts in access order here (the linker never aligns a
        # sequence), so its pool can lose a page from the middle of a stream.
        self._g2 = PrefixTracker(TIER_G2)
        pieces = tiers.pieces
        if engine.verify == "full":
            self._stamped = range(pieces)
        elif engine.verify == "sample":
            self._stamped = range(0, pieces, pieces - 1)  # first and last piece
        else:
            self._stamped = range(0)
        self._waiting: deque[_Req] = deque()
        self._running: list[_Req] = []
        self._chunked: _Req | None = None
        # SGLang ``running_batch.batch_is_full``: admission is shut until a
        # request leaves the running batch.
        self._batch_full = False
        # Offloads in the order they were accepted: (node, hashes, accepted at).
        self._pending: deque[tuple[Node, list[Node], float]] = deque()
        self._blocked = 0.0
        self._linker: LinkerPort | None = None
        self._host = None
        self._peers: list[tuple[str, str]] = []
        # Mesh warm.
        self._warm_queue: deque[tuple[int, str, int]] = deque()
        self._warm_active: list[tuple] = []
        self._warm_free: list[int] = []
        self._warm_ok = 0
        self._warm_failed = 0

    # ---- setup -------------------------------------------------------------

    def attach(self, host, linker: LinkerPort | None = None) -> None:
        self._host = host
        self._linker = linker or LinkerPort(
            host, call_timeout_s=60.0 + self._cfg.abandon_timeout_ms / 1e3
        )
        host.pin_scheduler()
        # The linker freezes the heap and all but disables full collections
        # when it first moves data (its default gc_full_collection_threshold).
        gen0, gen1, _ = gc.get_threshold()
        gc.set_threshold(gen0, gen1, 1_000_000)
        # This worker's warm page, kept in KVCR so peers can pull it.
        warm = block_id(warm_stream(self._worker), 0)
        host.stamp([0], [warm], range(self._tiers.pieces))
        node = Node(warm_stream(self._worker), 0, [0], 0)
        self._offload(node)

    def set_peers(self, peers: list[tuple[str, str]]) -> None:
        self._peers = peers

    def _call(self, function, *args):
        """A linker call: this thread blocks until the owner has served it."""
        started = time.perf_counter()
        result = function(*args)
        self._blocked += time.perf_counter() - started
        return result

    # ---- requests ----------------------------------------------------------

    def submit(self, request: Request) -> None:
        r = _Req()
        r.rid = request.rid
        r.sid = f"{request.rid:032x}"
        r.handle = Handle(r.sid, 0)
        r.segments = request.segments
        r.all_segments = (*request.segments, request.tail)
        parents = {}
        previous = None
        for stream, _, _ in r.all_segments:
            if previous is not None and stream != previous:
                parents[stream] = previous
            previous = stream
        r.parents = parents
        r.prompt_tokens = request.prompt_tokens
        r.output_tokens = max(1, request.output_tokens)
        r.hashes = self._hash_cache.names(request.segments)
        r.prompt_pages = len(r.hashes)
        # The matched length is at most one less than the input length.
        r.match_pages = min(r.prompt_pages, (r.prompt_tokens - 1) // self._page_tokens)
        tail = request.tail
        # The pages that enter the cache, and the pages it holds while it runs.
        r.total_pages = r.prompt_pages + tail[2] - tail[1]
        r.live_pages = live_blocks(request, self._page_tokens, self._usable)
        r.arrived = time.perf_counter()
        r.looked = r.started = r.first_token = 0.0
        r.loaded = 0.0  # when its load finished, on the owner's clock
        r.device_hit = r.host_hit = r.remote_pages = 0
        r.slots = []
        r.path = []
        r.computed = r.sched = r.inserted = r.walked = 0
        r.lookups = r.prefill_tokens = 0
        r.contexts = 0
        r.remote_flags = ()
        r.failed = r.timed_out = 0
        r.hint = None
        if request.hint_source >= 0 and request.hint_blocks > 0:
            hashes = segment_ids(request.segments)[: request.hint_blocks]
            endpoint = self._peers[request.hint_source][1]
            r.hint = fetch_hint(r.sid, endpoint, hashes)
        self._waiting.append(r)

    # ---- one scheduler iteration -------------------------------------------

    def step(self) -> list[Done]:
        perf = time.perf_counter
        window = self._window
        cfg = self._cfg
        linker = self._linker
        fatal = linker.fatal()
        if fatal is not None:
            raise RuntimeError("the KVCR owner thread failed") from fatal
        t_step = perf()
        self._blocked = 0.0
        load_wait = forward = 0.0
        dones: list[Done] = []

        # check_hicache_events: every iteration.
        self._check_events()
        if self._warm_queue or self._warm_active:
            self._warm_tick()

        # A request stopped mid-prefill is checkpointed before the next batch.
        chunked = self._chunked
        if chunked is not None:
            self._checkpoint(chunked, chunked.computed // self._page_tokens)

        # get_new_batch_prefill.
        budget = cfg.max_batched_tokens
        batch: list[_Req] = []
        if chunked is not None:
            n = min(chunked.prompt_tokens - chunked.computed, budget)
            chunked.sched = n
            budget -= n
            batch.append(chunked)
            if n < chunked.prompt_tokens - chunked.computed:
                budget = 0  # still the chunked request: nothing else fits
            else:
                self._chunked = None
        waiting = self._waiting
        if not self._running:
            # An empty running batch has no request left to leave.
            self._batch_full = False
        while waiting and budget > 0 and not self._batch_full:
            if len(self._running) + len(batch) >= cfg.max_running:
                self._batch_full = True
                break
            r = waiting[0]
            self._match(r)
            if not self._admit(r):
                # NO_TOKEN: matched, but the device has no room for it yet.
                self._batch_full = bool(batch) or bool(self._running)
                break
            waiting.popleft()
            n = min(r.prefill_tokens, budget)
            r.sched = n
            budget -= n
            batch.append(r)
            if n < r.prefill_tokens:
                self._chunked = r
                break

        if batch:
            # ready_to_load_host_cache: once per new prefill batch.
            index = self._call(linker.start_layer_wise_loading)
            loaders = [r for r in batch if r.host_hit and not r.loaded]
            tokens = sum(r.sched for r in batch)
            compute = cfg.step_ms / 1e3 + tokens * cfg.prefill_us_per_token / 1e6
            t0 = perf()
            end = t0 + compute
            if index >= 0:
                # Each layer waits for its K and its V buffer. One timeline,
                # slept out at the end.
                counter = linker.layer_done_counter
                counter.set_consumer(index)
                wait_until = counter.wait_until
                landed = []
                for layer in range(self._layers):
                    wait_until(layer)
                    wait_until(layer)
                    landed.append(perf())
                end = forward_end(t0, landed, compute / self._layers)
                load_wait += max(0.0, end - t0 - compute)
            time.sleep(max(0.0, end - perf()))
            t2 = perf()
            forward = t2 - t0 - load_wait
            failed = False
            if loaders:
                blocked = self._blocked
                failed = not self._call(linker.finish_layer_wise_loading, index)
                # Waiting for the batch to drain is transfer time, not a call.
                load_wait += self._blocked - blocked
                self._blocked = blocked
                self._loads_finished(loaders, index, failed)
            now = perf()
            for r in batch:
                if failed and r in loaders:
                    dones.append(self._abort(r))
                    continue
                before = r.computed
                r.computed = before + r.sched
                if r.computed < r.prompt_tokens:
                    continue  # the chunked request: next chunk next iteration
                r.first_token = now
                self._checkpoint(r, r.prompt_pages)
                if r.output_tokens <= 1:
                    dones.append(self._finish(r))
                else:
                    self._running.append(r)
        elif self._running:
            # Decode: every running request produces one token.
            t1 = perf()
            time.sleep(cfg.step_ms / 1e3)
            now = perf()
            forward = now - t1
            still: list[_Req] = []
            for r in self._running:
                r.computed += 1
                if r.computed >= r.prompt_tokens + r.output_tokens - 1:
                    dones.append(self._finish(r))
                else:
                    still.append(r)
            self._running = still
        # With nothing to run the loop spins, as SGLang's does by default.

        total = perf() - t_step
        counters = window.counters
        counters["steps"] += 1
        blocked = self._blocked
        if batch or forward:
            counters["busy_steps"] += 1
            counters["forward_seconds"] += forward
            counters["load_wait_seconds"] += load_wait
            counters["linker_wait_seconds"] += blocked
            window.hists["step_kvcr"].add(blocked)
            if load_wait:
                window.hists["step_load_wait"].add(load_wait)
        else:
            counters["idle_steps"] += 1
            counters["idle_seconds"] += total
            counters["idle_linker_wait_seconds"] += blocked
        return dones

    # ---- match and admission -----------------------------------------------

    def _match(self, r: _Req) -> None:
        """``Wrapper.match``: the device prefix, then the host tail lookup."""
        device = min(self._tree.prefix(r.segments), r.match_pages)
        r.device_hit = device
        r.host_hit = r.remote_pages = 0
        r.remote_flags = ()
        if device >= r.match_pages:
            return  # the device covers it, or the tail is under one page
        if not r.lookups:
            r.looked = time.perf_counter()
        r.lookups += 1
        linker = self._linker
        self._call(linker.set_request_context, r.handle, r.hint)
        r.contexts += 1
        valid, remote = self._call(
            linker.lookup, r.sid, r.hashes[device : r.match_pages]
        )
        if valid:
            r.host_hit = valid[-1]
            r.remote_flags = remote[: r.host_hit]
            r.remote_pages = sum(r.remote_flags)

    def _admit(self, r: _Req) -> bool:
        """``add_one_req``: device slots for everything new, then the load."""
        tree = self._tree
        path = tree.path(r.segments, r.device_hit)
        tree.touch(path)
        tree.lock(path)
        slots = tree.allocate(r.live_pages - r.device_hit)
        if slots is None:
            tree.unlock(path)
            self._window.counters["device_alloc_failures"] += 1
            return False
        r.path = path
        r.slots = slots
        r.inserted = r.device_hit
        tokens = self._page_tokens
        r.computed = min((r.device_hit + r.host_hit) * tokens, r.prompt_tokens - 1)
        r.prefill_tokens = r.prompt_tokens - r.computed
        r.started = time.perf_counter()
        if r.host_hit:
            load_slots = slots[: r.host_hit]
            if self._stamped:
                self._host.clear(load_slots, self._stamped)
            hashes = r.hashes[r.device_hit : r.device_hit + r.host_hit]
            self._call(self._linker.load, r.sid, hashes, load_slots)
            self._window.counters["loads"] += 1
        return True

    def _loads_finished(self, loaders: list[_Req], index: int, failed: bool) -> None:
        window = self._window
        counters = window.counters
        info = self._linker.batch_info.pop(index)
        remote = sum(r.remote_pages for r in loaders)
        counters["load_batches"] += 1
        duration = info.finished - info.started
        window.hists["deliver_remote" if remote else "deliver_local"].add(duration)
        for r in loaders:
            r.loaded = info.finished
        if failed:
            # The batch is discarded whole, but KVCR's results say what became
            # of each request's pages: failed, delivered (every layer of the
            # request returned and the page never failed) or never issued. The
            # linker only loads what KVCR called restorable, so a page from a
            # peer was named by the hint.
            counters["load_batches_failed"] += 1
            encode = self._linker.encode_key
            loads = {load.rid: load for load in info.loads}
            for r in loaders:
                load = loads[r.sid]
                whole = load.layers == self._layers
                hashes = r.hashes[r.device_hit : r.device_hit + r.host_hit]
                for page, is_remote in zip(hashes, r.remote_flags):
                    key = encode(page)
                    if key in load.failed:
                        r.failed += 1
                        counters["deliver_blocks_failed"] += 1
                        if is_remote:
                            cause = count_failure(counters, load.failed[key])
                            r.timed_out += cause == "timeout"
                    elif not whole:
                        counters["deliver_blocks_unissued"] += 1
                    elif is_remote:
                        counters["deliver_blocks_remote"] += 1
                    else:
                        counters["deliver_blocks_local"] += 1
            return
        counters["deliver_blocks_remote"] += remote
        counters["deliver_blocks_local"] += sum(r.host_hit for r in loaders) - remote
        for r in loaders:
            resolve = r.loaded - r.looked
            window.hists["cache_resolve"].add(resolve)
            if r.remote_pages:
                window.hists["cache_resolve_remote"].add(resolve)
        if self._stamped:
            host = self._host
            everything = self._cfg.verify == "full"
            for r in loaders:
                count = r.host_hit if everything else 1
                ids = segment_ids(r.segments)[r.device_hit : r.device_hit + count]
                counters["bytes_checked"] += count
                counters["bytes_mismatch"] += min(
                    count, host.mismatches(r.slots[:count], ids, self._stamped)
                )

    # ---- inserts and offloads ----------------------------------------------

    def _checkpoint(self, r: _Req, upto: int) -> None:
        """Insert the request's pages up to ``upto`` and offload what is new.

        ``checkpoint_kv_cache`` -> insert -> ``BackupKV`` -> ``offload_nodes``:
        one offload per node on the request's path that KVCR does not hold and
        that has none in flight. Restored pages enter the tree as new nodes,
        so they are offloaded again.
        """
        tree = self._tree
        if upto > r.inserted:
            first = r.inserted
            offset = 0
            base = r.device_hit
            restored_end = base + r.host_hit
            stamp_slots: list[int] = []
            stamp_ids: list[int] = []
            for stream, start, end in r.all_segments:
                length = end - start
                low, high = max(first, offset), min(upto, offset + length)
                if low < high:
                    slots = r.slots[low - base : high - base]
                    page = start + low - offset
                    if self._stamped and high > restored_end:
                        # Computed pages get their bytes; restored ones have them.
                        skip = max(0, restored_end - low)
                        stamp_slots.extend(slots[skip:])
                        first_id = (stream << INDEX_BITS) + page + skip
                        stamp_ids.extend(range(first_id, first_id + len(slots) - skip))
                    if slots:
                        tree.insert(stream, page, slots, r.parents.get(stream))
                offset += length
                if offset >= upto:
                    break
            if stamp_slots:
                self._host.stamp(stamp_slots, stamp_ids, self._stamped)
            r.inserted = upto
            # The request now uses the tree's path up to there, whoever
            # inserted it: another request may have computed the same pages.
            path = tree.path(r.all_segments, upto)
            tree.lock(path[len(r.path) :])
            r.path = path
        # The insert walk: every node past what this request walked before.
        walked = r.walked
        position = 0
        for node in r.path:
            position += node.end - node.start
            if position <= walked:
                continue
            if not node.stored and not node.pending:
                self._offload(node)
        r.walked = max(walked, position)

    def _offload(self, node: Node) -> None:
        hashes = self._hash_cache.stream(node.stream, node.end)[node.start : node.end]
        # Stamped before the call: the owner can finish a deposit before this
        # thread runs again.
        accepted = time.perf_counter()
        if self._call(self._linker.offload, hashes, node.slots):
            node.pending = True
            # inc_lock_ref: the node and its ancestors stay until it completes
            # (none for the warm page, which is outside the tree).
            locked = self._tree.ancestry(node)
            self._tree.lock(locked)
            self._pending.append((node, locked, accepted))
            counters = self._window.counters
            counters["offloads"] += 1
            counters["offload_pages"] += node.end - node.start

    def _check_events(self) -> None:
        linker = self._linker
        pending = self._pending
        completed = min(linker.num_completed_offloads(), len(pending))
        if completed:
            window = self._window
            counters = window.counters
            for _ in range(completed):
                ok, finished = self._call(linker.pop_completed_offload)
                node, locked, accepted = pending.popleft()
                node.pending = False
                self._tree.unlock(locked)
                pages = node.end - node.start
                window.hists["deposit"].add(finished - accepted)
                if ok:
                    node.stored = True
                    self._g2.add(node.stream, node.start, node.end)
                    counters["deposit_blocks_ok"] += pages
                else:
                    counters["deposit_blocks_failed"] += pages
        removals = self._call(linker.pop_storage_removals)
        if removals:
            tree, g2 = self._tree, self._g2
            for storage_hash in removals:
                stream, index = split_id(hash_to_id(storage_hash))
                node = tree.node_at(stream, index)
                if node is not None:
                    node.stored = False
                g2.remove(stream, index)
            self._window.counters["kvcr_removals"] += len(removals)

    # ---- request ends -------------------------------------------------------

    def _release(self, r: _Req) -> None:
        self._tree.unlock(r.path)
        r.path = []
        if r.contexts:
            self._call(self._linker.release_request_context, r.handle)

    def _finish(self, r: _Req) -> Done:
        window = self._window
        self._batch_full = False  # a request leaves the running batch
        # release_kv_cache(checkpoint=True): the answer's pages join the tree;
        # what held tokens that make no whole page goes back.
        self._checkpoint(r, r.total_pages)
        self._tree.release(r.slots[r.total_pages - r.device_hit :])
        self._release(r)
        ttft = r.first_token - r.arrived
        window.hists["ttft"].add(ttft)
        window.hists["queue"].add((r.looked or r.started) - r.arrived)
        counters = window.counters
        counters["requests"] += 1
        counters["prompt_blocks"] += r.prompt_pages
        counters["gpu_hit_blocks"] += r.device_hit
        counters["prefill_tokens"] += r.prefill_tokens
        counters["lookups"] += r.lookups
        return Done(
            rid=r.rid,
            ttft_ms=ttft * 1e3,
            load_ms=self._load_ms(r),
            prompt_blocks=r.prompt_pages,
            gpu_blocks=r.device_hit,
            primary_blocks=0,
            local_blocks=r.host_hit - r.remote_pages,
            remote_blocks=r.remote_pages,
            failed_blocks=0,
            recomputed_tokens=r.prefill_tokens,
        )

    def _abort(self, r: _Req) -> Done:
        """A failed load: the request ends without output and inserts nothing."""
        if self._chunked is r:
            self._chunked = None
        self._batch_full = False
        self._tree.release(r.slots[r.inserted - r.device_hit :])
        r.slots = []
        self._release(r)
        counters = self._window.counters
        counters["aborted_requests"] += 1
        counters["lookups"] += r.lookups
        return Done(
            rid=r.rid,
            ttft_ms=0.0,
            load_ms=self._load_ms(r),
            prompt_blocks=r.prompt_pages,
            gpu_blocks=r.device_hit,
            primary_blocks=0,
            local_blocks=0,
            remote_blocks=0,
            failed_blocks=r.failed,
            recomputed_tokens=0,
            timed_out_blocks=r.timed_out,
            aborted=True,
        )

    @staticmethod
    def _load_ms(r: _Req) -> float:
        """From the first lookup to the end of the load, or to admission if
        nothing was loaded."""
        return ((r.loaded or r.started) - r.looked) * 1e3 if r.looked else 0.0

    # ---- mesh warm ----------------------------------------------------------

    def start_warm(self, peers: list[tuple[int, str]]) -> None:
        self._warm_queue.extend((peer, endpoint, 0) for peer, endpoint in peers)
        self._warm_free = list(range(1, 1 + self._warm_slots))

    def warm_status(self) -> tuple[int, int, int]:
        pending = len(self._warm_queue) + len(self._warm_active)
        return pending, self._warm_ok, self._warm_failed

    def _warm_tick(self) -> None:
        """Pull one page from each peer through the normal load path.

        One load per batch, so a peer that fails does not take the others'
        attempts with it.
        """
        linker, host = self._linker, self._host
        still = []
        for entry in self._warm_active:
            future, index, peer, endpoint, attempt, slot, handle, bid = entry
            if not future.done():
                still.append(entry)
                continue
            try:
                landed = future.result()
            except Exception:  # noqa: BLE001 - a fatal owner error surfaces in step()
                landed = False
            linker.layer_done_counter.finish(index)
            linker.batch_info.pop(index, None)
            ok = landed and not host.mismatches(
                [slot], [bid], range(self._tiers.pieces)
            )
            self._warm_done(peer, endpoint, attempt, slot, handle, ok)
        self._warm_active = still
        while self._warm_queue and self._warm_free:
            peer, endpoint, attempt = self._warm_queue.popleft()
            slot = self._warm_free.pop()
            bid = block_id(warm_stream(peer), 0)
            sid = f"warm-{peer}-{attempt}"
            handle = Handle(sid, 0)
            storage_hash = page_hash(bid)
            linker.set_request_context(handle, fetch_hint(sid, endpoint, [bid]))
            valid, _ = linker.lookup(sid, [storage_hash])
            if not valid:
                self._warm_done(peer, endpoint, attempt, slot, handle, False)
                continue
            host.clear([slot], range(self._tiers.pieces))
            linker.load(sid, [storage_hash], [slot])
            index = linker.start_layer_wise_loading()
            future = linker.batch_completion(index)
            self._warm_active.append(
                (future, index, peer, endpoint, attempt, slot, handle, bid)
            )

    def _warm_done(self, peer, endpoint, attempt, slot, handle, ok: bool) -> None:
        self._linker.release_request_context(handle)
        self._warm_free.append(slot)
        if ok:
            self._warm_ok += 1
        elif attempt + 1 < _WARM_ATTEMPTS:
            self._warm_queue.append((peer, endpoint, attempt + 1))
        else:
            self._warm_failed += 1

    # ---- reporting ---------------------------------------------------------

    def take_inventory(self) -> list[InventoryDelta]:
        out: list[InventoryDelta] = [
            (TIER_DEVICE, *event) for event in self._tree.take_events()
        ]
        out.extend(self._g2.take())
        return out

    def abort_requests(self) -> None:
        """Drop every request and every offload that has not started."""
        linker, tree = self._linker, self._tree
        requests = [*self._waiting, *self._running]
        if self._chunked is not None and self._chunked not in requests:
            requests.append(self._chunked)
        for r in requests:
            tree.unlock(r.path)
            r.path = []
            if r.contexts:
                linker.release_request_context(r.handle)
        self._waiting.clear()
        self._running = []
        self._chunked = None
        dropped = linker.drop_queued_offloads()
        for _ in range(min(dropped, len(self._pending))):
            self._pending.pop()

    def busy(self) -> bool:
        return bool(
            self._waiting
            or self._running
            or self._chunked is not None
            or self._pending
            or self._warm_active
            or self._linker.busy()
        )

    def fill_level(self) -> float:
        """How full KVCR's pool is: the tier that must be evicting to be steady."""
        return min(1.0, self._g2.count / self._tiers.g2_blocks)

    def gauges(self) -> dict[str, float]:
        tree = self._tree
        return {
            "waiting": len(self._waiting),
            "running": len(self._running),
            "offload_backlog": len(self._pending),
            "device_fill": tree.fill_level(),
            "device_evictions": tree.evictions,
            "device_alloc_failures": tree.allocation_failures,
            "device_duplicate_pages": tree.duplicate_pages,
            "kvcr_pool_fill": self.fill_level(),
        }

    def resilience_samples(self) -> list[str]:
        return self._linker.resilience_samples

    def begin_window(self) -> None:
        self._linker.take_stats()  # the fill phase's: dropped
        self._window.counters["offload_backlog_at_start"] = len(self._pending)

    def end_window(self) -> None:
        self._window.merge_owner(self._linker.take_stats())
        self._window.counters["offload_backlog_at_stop"] = len(self._pending)

    def close(self) -> None:
        if self._linker is not None:
            self._linker.close()
