# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Watch a KVCR from its own API.

The engine builds the KVCR and makes every call; the probe stands between the
two. It times each call on the engine thread, follows an operation from the
call that starts it to the poll that completes it, counts what moved and why
the rest did not, and checks the bytes: a chunk is stamped with its block id
when it is deposited, and the stamp is read back when a deliver into a chunk
completes. Nothing here knows which engine is calling.

``KvcrProbe`` is given to the engine in place of the ``KVCR`` class: calling
it builds the real KVCR and returns the probe.
"""

from __future__ import annotations

import ctypes
import struct
import time

import msgspec

from kvcr.types import CacheTier

from .ids import key_to_id
from .metrics import Window

_STAMP = struct.Struct(">Q")
_REMOTE = CacheTier.REMOTE_G2
_MAX_RESILIENCE_SAMPLES = 20


class _Op:
    __slots__ = ("load", "request", "chunks", "remaining", "ok", "bad", "started")


class _Request:
    __slots__ = ("hint", "cross", "remote", "remote_ok", "local_ok", "failed",
                 "timed_out", "first_query", "last_done")  # fmt: skip

    def __init__(self) -> None:
        self.hint = ()  # the ids the router said the source holds
        self.cross = False
        self.remote: set[bytes] = set()  # keys KVCR placed on the peer
        self.remote_ok = self.local_ok = self.failed = self.timed_out = 0
        self.first_query = self.last_done = 0.0


class KvcrProbe:
    def __init__(
        self,
        kvcr_cls,
        window: Window,
        *,
        verify: str = "sample",
        cross_node: frozenset[str] = frozenset(),
        on_stats=None,
    ) -> None:
        """``cross_node`` holds the control endpoints of peers on other nodes;
        ``on_stats`` is given every telemetry object KVCR hands back."""
        self._kvcr_cls = kvcr_cls
        self._window = window
        self._verify = verify
        self._cross_node = cross_node
        self._on_stats = on_stats
        self._kvcr = None
        self._ops: dict[object, _Op] = {}
        self._requests: dict[str, _Request] = {}
        self._pins: set[object] = set()
        self._query_secs = 0.0
        self._query_calls = 0
        self.resilience: list[str] = []

    def __call__(self, config, bindings, backends, *rest) -> KvcrProbe:
        region = backends.framework_regions[0]
        self._block = region.size
        self.region = (region.size, region.count)  # bytes per chunk, chunks
        memory = (ctypes.c_char * (region.size * region.count)).from_address(
            region.addr
        )
        self._view = memoryview(memory)
        self._kvcr = self._kvcr_cls(config, self._watch(bindings), backends, *rest)
        return self

    def __getattr__(self, name):  # config, close and whatever else is not watched
        return getattr(self._kvcr, name)

    # ---- callbacks KVCR makes ------------------------------------------------

    def _watch(self, bindings):
        counters = lambda: self._window.counters  # noqa: E731  (the window resets)
        changes = {"on_resilience_event": self._resilience_event}
        self._on_resilience = bindings.on_resilience_event
        if bindings.request_pin is not None:
            request_pin = bindings.request_pin
            poll_pin_results = bindings.poll_pin_results
            release_pin = bindings.release_pin

            def counted_request(keys):
                counters()["pin_requests"] += 1
                return request_pin(keys)

            def counted_results():
                results = list(poll_pin_results())
                for _, result in results:
                    if result is None:
                        counters()["pin_misses"] += 1
                        continue
                    handle, refs = result
                    self._pins.add(handle)
                    counters()["pin_keys"] += sum(
                        1 for ref in refs.values() if ref is not None
                    )
                return results

            def counted_release(handle):
                self._pins.discard(handle)
                return release_pin(handle)

            changes.update(
                request_pin=counted_request,
                poll_pin_results=counted_results,
                release_pin=counted_release,
            )
        return msgspec.structs.replace(bindings, **changes)

    def _resilience_event(self, error: Exception) -> None:
        self._window.counters["resilience_events"] += 1
        if len(self.resilience) < _MAX_RESILIENCE_SAMPLES:
            self.resilience.append(str(error)[:200])
        if self._on_resilience is not None:
            self._on_resilience(error)

    # ---- calls the engine makes ----------------------------------------------

    def submit_hint(self, hints, request_id=None):
        t = time.perf_counter()
        result = self._kvcr.submit_hint(hints=hints, request_id=request_id)
        spent = time.perf_counter() - t
        payload = hints.actions[0].payload
        state = self._request(request_id)
        state.hint = payload["block_hashes"]
        state.cross = payload["source_control_endpoint"] in self._cross_node
        self._window.call("submit_hint", spent, 1, len(state.hint))
        return result

    def query(self, keys, request_id=None):
        t = time.perf_counter()
        result = self._kvcr.query(keys, request_id)
        # One key a call, thousands of calls a request: totalled at the poll.
        self._query_secs += time.perf_counter() - t
        self._query_calls += len(keys)
        state = self._requests.get(request_id)
        if state is None:
            state = self._request(request_id)
        if not state.first_query:
            state.first_query = t
        for key, (_, tier) in zip(keys, result):
            if tier is _REMOTE:
                state.remote.add(key)
        return result

    def deliver(self, blocks, request_id=None):
        t = time.perf_counter()
        handle = self._kvcr.deliver(blocks, request_id=request_id)
        self._window.call("deliver", time.perf_counter() - t, 1, len(blocks))
        self._started(handle, True, request_id, blocks, t)
        return handle

    def deposit(self, blocks):
        # The bytes a peer will check, in place before KVCR can read them.
        pack, view, block = _STAMP.pack_into, self._view, self._block
        for key, refs in blocks.items():
            pack(view, refs[0].element_index * block, key_to_id(key))
        t = time.perf_counter()
        handle = self._kvcr.deposit(blocks)
        self._window.call("deposit", time.perf_counter() - t, 1, len(blocks))
        self._started(handle, False, None, blocks, t)
        return handle

    def poll_completed(self):
        window = self._window
        t = time.perf_counter()
        completed = list(self._kvcr.poll_completed())
        now = time.perf_counter()
        window.call("poll", now - t, 1, len(completed))
        if self._query_calls:
            window.call("query", self._query_secs, self._query_calls, self._query_calls)
            self._query_secs = 0.0
            self._query_calls = 0
        for handle, entries in completed:
            op = self._ops.get(handle)
            if op is None:
                continue
            op.remaining -= len(entries)
            for key, entry in entries.items():
                if entry.success:
                    op.ok.append(key)
                else:
                    op.bad[key] = entry
            if op.remaining <= 0:  # an operation can complete over several polls
                del self._ops[handle]
                if op.load:
                    self._loaded(op, now)
                else:
                    window.hists["deposit"].add(now - op.started)
                    window.counters["deposit_blocks_ok"] += len(op.ok)
                    window.counters["deposit_blocks_failed"] += len(op.bad)
        return completed

    def align_sequence(self, keys, use_current_time=False):
        t = time.perf_counter()
        self._kvcr.align_sequence(keys, use_current_time=use_current_time)
        self._window.call("align_sequence", time.perf_counter() - t, 1, len(keys))

    def discard_hint(self, request_id):
        t = time.perf_counter()
        self._kvcr.discard_hint(request_id)
        self._window.call("discard_hint", time.perf_counter() - t)
        self._requests.pop(request_id, None)

    def get_stats(self):
        t = time.perf_counter()
        stats = self._kvcr.get_stats()
        self._window.call("get_stats", time.perf_counter() - t)
        if stats is not None and self._on_stats is not None:
            self._on_stats(stats)
        return stats

    # ---- what the engine's driver reads ----------------------------------------

    def outcome(self, request_id) -> tuple[int, int, int, int, float]:
        """A finished request's delivered blocks from a peer and from this
        worker's own pool, its failed blocks, those of them that timed out, and
        the seconds from its first query to its last deliver (0 if nothing
        was delivered)."""
        state = self._requests.pop(request_id, None)
        if state is None:
            return 0, 0, 0, 0, 0.0
        waited = max(0.0, state.last_done - state.first_query)
        return state.remote_ok, state.local_ok, state.failed, state.timed_out, waited

    def pending(self) -> int:
        """Operations and source pins still open."""
        return len(self._ops) + len(self._pins)

    # ---- bookkeeping -----------------------------------------------------------

    def _request(self, request_id) -> _Request:
        state = self._requests[request_id] = _Request()
        return state

    def _started(self, handle, load, request_id, blocks, t) -> None:
        op = self._ops[handle] = _Op()
        op.load = load
        op.request = request_id
        op.chunks = {key: refs[0].element_index for key, refs in blocks.items()}
        op.remaining = len(blocks)
        op.ok = []
        op.bad = {}
        op.started = t

    def _loaded(self, op: _Op, now: float) -> None:
        counters = self._window.counters
        state = self._requests.get(op.request)
        peers = state.remote if state is not None else ()
        remote = sum(1 for key in op.ok if key in peers)
        asked_a_peer = any(key in peers for key in op.chunks)
        self._window.hists["deliver_remote" if asked_a_peer else "deliver_local"].add(
            now - op.started
        )
        counters["deliver_blocks_remote"] += remote
        if remote and state.cross:
            counters["deliver_blocks_remote_cross_node"] += remote
        counters["deliver_blocks_local"] += len(op.ok) - remote
        counters["deliver_blocks_failed"] += len(op.bad)
        if state is not None:
            state.remote_ok += remote
            state.local_ok += len(op.ok) - remote
            state.failed += len(op.bad)
            state.last_done = now
        if op.bad:
            counters["deliver_ops_partial" if op.ok else "deliver_ops_failed"] += 1
            # Blocks asked of a peer. Those the hint named are counted by
            # KVCR's status; an opportunistic query asks the peer for the rest
            # of the prompt too, which it cannot have.
            hinted = set(state.hint) if state is not None else ()
            for key, entry in op.bad.items():
                if key not in peers:
                    continue
                cause = entry.status.name.lower()
                if key_to_id(key) in hinted:
                    counters[f"hinted_failed|{cause}"] += 1
                    if entry.reason:
                        counters[f"hinted_failed_reason|{entry.reason}"] += 1
                else:
                    counters["deliver_blocks_failed_beyond_hint"] += 1
                if cause == "timeout":
                    state.timed_out += 1
        if self._verify != "none":
            unpack, view, block = _STAMP.unpack_from, self._view, self._block
            for key in op.ok if self._verify == "full" else op.ok[:1]:
                counters["bytes_checked"] += 1
                if unpack(view, op.chunks[key] * block)[0] != key_to_id(key):
                    counters["bytes_mismatch"] += 1
