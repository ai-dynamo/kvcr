# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The fake SGLang engine must put the real linker's call pattern on KVCR.

These drive the linker port and the scheduler against a recording stand-in
for KVCR. If one fails after a change, compare against SGLang's
``KVCRDirectLinker`` (PR #45650, revision 002d6d8d) before touching the
expectation.
"""

import threading
import time

import pytest
from clusterbench.config import load_config
from clusterbench.engine_sglang import SglangEngine, forward_end
from clusterbench.ids import (
    LINKER_KEY_PREFIX,
    block_id,
    page_hash,
    segment_ids,
    warm_stream,
)
from clusterbench.linker_port import Handle, LayerCounter, LinkerPort
from clusterbench.metrics import Window
from clusterbench.protocol import TIER_DEVICE, TIER_G2, Request
from clusterbench.router import Inventory

from kvcr.types import CacheTier, OpEntryResult, OpEntryStatus, QueryStatus

PAGE = 64
LAYERS = 4
SYSTEM, SESSION = 1, 70


def key(bid):
    return LINKER_KEY_PREFIX + f"kv:{page_hash(bid)}".encode()


OK = OpEntryResult(OpEntryStatus.SUCCESS)


class _Event:
    tier = CacheTier.LOCAL_G2

    def __init__(self, keys, removed):
        self.keys = keys
        self.removed = removed


class StubKvcr:
    """Records every call. Operations complete ``delay`` polls after issue."""

    def __init__(self, delay=1, auto=True):
        self.calls = []
        self.threads = set()
        self.resident = set()
        self.fetchable = {}  # request id -> keys a hinted peer holds
        self.fail = set()  # keys whose delivery fails
        self.fail_scope = None  # ... only for this request id
        self.fail_layer = None  # ... only in this layer
        self.failure = OpEntryResult(OpEntryStatus.UNAVAILABLE, reason="not_resident")
        self.delay = delay
        self.auto = auto
        self.polls = 0
        self.raise_on_poll = None
        self.outstanding = {}  # handle -> (kind, keys, issued at poll, scope, layer)
        self.max_delivers = 0
        self.max_deposits = 0
        self.closed_on = None
        self.ready = []  # completions the next poll returns
        self._next = 0

    def _record(self, *call):
        self.calls.append(call)
        self.threads.add(threading.current_thread().name)

    def of(self, name):
        return [call for call in self.calls if call[0] == name]

    def submit_hint(self, hints, request_id=None):
        self._record("submit_hint", request_id, hints.actions[0].payload)

    def discard_hint(self, request_id):
        self._record("discard_hint", request_id)

    def query(self, keys, request_id=None):
        self._record("query", list(keys), request_id)
        hinted = self.fetchable.get(request_id, ())
        out = []
        for k in keys:
            if k in self.resident:
                out.append((QueryStatus.HIT, CacheTier.LOCAL_G2))
            elif k in hinted:
                out.append((QueryStatus.FETCHABLE, CacheTier.REMOTE_G2))
            else:
                out.append((QueryStatus.MISS, None))
        return out

    def _issue(self, kind, blocks, scope=None, layer=None):
        self._next += 1
        self.outstanding[self._next] = (kind, list(blocks), self.polls, scope, layer)
        count = sum(1 for entry in self.outstanding.values() if entry[0] == kind)
        if kind == "deliver":
            self.max_delivers = max(self.max_delivers, count)
        else:
            self.max_deposits = max(self.max_deposits, count)
        return self._next

    def deliver(self, blocks, request_id=None):
        refs = {k: [(r.label, r.element_index) for r in v] for k, v in blocks.items()}
        self._record("deliver", refs, request_id)
        label = next(iter(refs.values()))[0][0]
        return self._issue("deliver", blocks, request_id, int(label.rsplit("b", 1)[1]))

    def deposit(self, blocks):
        refs = {k: [(r.label, r.element_index) for r in v] for k, v in blocks.items()}
        self._record("deposit", refs)
        return self._issue("deposit", blocks)

    def complete(self, handle):
        """Finish an operation: the next poll reports it."""
        kind, keys, _, scope, layer = self.outstanding.pop(handle)
        if kind == "deposit":
            self.resident.update(keys)
        failed = self.fail if kind == "deliver" else ()
        if self.fail_scope not in (None, scope) or self.fail_layer not in (None, layer):
            failed = ()
        results = {k: self.failure if k in failed else OK for k in keys}
        self.ready.append((handle, results))

    def poll_completed(self):
        self.threads.add(threading.current_thread().name)
        if self.raise_on_poll is not None:
            raise self.raise_on_poll
        self.polls += 1
        if self.auto:
            for handle, entry in list(self.outstanding.items()):
                if self.polls - entry[2] >= self.delay:
                    self.complete(handle)
        ready, self.ready = self.ready, []
        return ready

    def get_stats(self):
        return None

    def close(self):
        self.closed_on = threading.current_thread().name

    def align_sequence(self, *args, **kwargs):
        raise AssertionError("the linker never aligns a sequence")

    fetch = release = align_sequence


class StubHost:
    """What ``LinkerHost`` offers, without memory."""

    agent_name = "agent"
    piece_bytes = 128

    def __init__(self, kvcr, layers=LAYERS):
        self.kvcr = kvcr
        self.labels = tuple(
            tuple(f"kv_128:c{c}.b{layer}" for layer in range(layers)) for c in range(2)
        )
        self.port = None

    def bind(self, port):
        self.port = port

    def start(self):
        pass

    def pin_scheduler(self):
        pass

    def stamp(self, slots, ids, pieces):
        pass

    def clear(self, slots, pieces):
        pass

    def mismatches(self, slots, ids, pieces):
        return 0

    def close(self):
        self.kvcr.close()


def inline_port(layers=LAYERS, **stub):
    kvcr = StubKvcr(auto=False, **stub)
    return LinkerPort(StubHost(kvcr, layers), inline=True), kvcr


def drain(port, kvcr, rounds=500):
    """Complete every operation as it appears until the port has nothing left."""
    for _ in range(rounds):
        port.progress()
        if not kvcr.outstanding and not kvcr.ready and not port.busy():
            return
        for handle in list(kvcr.outstanding):
            kvcr.complete(handle)
    raise AssertionError("operations never drained")


# ---- the linker port, driven inline -----------------------------------------


def test_a_load_is_one_deliver_per_layer_at_most_four_outstanding():
    port, kvcr = inline_port(layers=6)
    handle = Handle("r", 0)
    port.set_request_context(handle, None)
    assert port.load("r", [page_hash(5), page_hash(6)], [40, 41])
    index = port.start_layer_wise_loading()
    assert index == 0 and not kvcr.of("deliver")  # nothing moves before progress

    port.progress()
    assert len(kvcr.of("deliver")) == 4 and len(kvcr.outstanding) == 4
    port.progress()
    assert len(kvcr.of("deliver")) == 4  # the window is full
    drain(port, kvcr)

    delivers = kvcr.of("deliver")
    assert len(delivers) == 6 and kvcr.max_delivers == 4
    for layer, (_, refs, scope) in enumerate(delivers):
        assert scope == "1:r:0"
        # That layer's K and V piece of every page, at the page's device slot.
        assert refs == {
            key(5): [(f"kv_128:c0.b{layer}", 40), (f"kv_128:c1.b{layer}", 40)],
            key(6): [(f"kv_128:c0.b{layer}", 41), (f"kv_128:c1.b{layer}", 41)],
        }
    assert port.finish_layer_wise_loading(index) is True
    (load,) = port.batch_info[index].loads
    assert load.layers == 6 and not load.failed


def test_an_offload_is_chunked_deposits_one_at_a_time():
    port, kvcr = inline_port(layers=64)
    pages = list(range(100, 170))
    assert port.offload([page_hash(b) for b in pages], list(range(70)))
    port.progress()
    port.progress()
    assert len(kvcr.of("deposit")) == 1  # the next waits for this one
    drain(port, kvcr)

    deposits = kvcr.of("deposit")
    # 4096 pieces per deposit and 128 pieces per page: 32 pages each.
    assert [len(refs) for _, refs in deposits] == [32, 32, 6]
    assert kvcr.max_deposits == 1
    labels = [f"kv_128:c{c}.b{layer}" for c in range(2) for layer in range(64)]
    first = deposits[0][1]
    assert list(first) == [key(b) for b in pages[:32]]
    assert first[key(103)] == [(label, 3) for label in labels]
    assert port.num_completed_offloads() == 1
    assert port.pop_completed_offload()[0] is True


def test_an_offload_waits_for_the_loads_accepted_before_it():
    port, kvcr = inline_port()
    port.set_request_context(Handle("r", 0), None)
    port.load("r", [page_hash(5)], [40])
    port.offload([page_hash(7)], [3])  # while the load is still queued
    port.progress()
    assert not kvcr.of("deposit")
    index = port.start_layer_wise_loading()
    for _ in range(3):
        port.progress()
    assert not kvcr.of("deposit") and kvcr.of("deliver")
    drain(port, kvcr)
    assert port.finish_layer_wise_loading(index)
    assert len(kvcr.of("deposit")) == 1


def test_a_scope_is_discarded_once_released_and_unused():
    port, kvcr = inline_port()
    handle = Handle("r", 0)
    port.set_request_context(handle, None)
    port.load("r", [page_hash(5)], [40])
    index = port.start_layer_wise_loading()
    port.release_request_context(handle)
    assert not kvcr.of("discard_hint")  # an accepted load keeps its scope
    drain(port, kvcr)
    assert port.finish_layer_wise_loading(index)
    assert kvcr.of("discard_hint") == [("discard_hint", "1:r:0")]


def test_a_removal_is_held_back_until_its_offload_is_acknowledged():
    port, kvcr = inline_port()
    port.offload([page_hash(7)], [3])
    drain(port, kvcr)
    port.record_inventory(_Event([key(7)], removed=True))
    assert port.pop_storage_removals() == []
    assert port.pop_completed_offload()[0] is True
    assert port.pop_storage_removals() == [page_hash(7)]
    assert port.pop_storage_removals() == []
    # A later add cancels a removal that was never reported.
    port.record_inventory(_Event([key(7)], removed=True))
    port.record_inventory(_Event([key(7)], removed=False))
    assert port.pop_storage_removals() == []


def test_the_router_hash_is_the_first_sixteen_digits():
    port, _ = inline_port()
    assert port.encode_to_router(key(0xABCDEF)) == 0xABCDEF
    assert port.decode_key(b"other:key") is None
    with pytest.raises(NotImplementedError):
        port.decode_from_router(1)


# ---- the scheduler, with a real owner thread --------------------------------

OVERRIDES = [
    "engine.profile=sglang",
    f"tiers.layers={LAYERS}",
    "tiers.block_bytes=4096",
    "tiers.gpu_blocks=256",
    "tiers.g2_blocks=1024",
    "engine.step_ms=0.05",
    "engine.prefill_us_per_token=0",
    "engine.verify=none",
    "phases.warm_concurrency=2",
]


class Rig:
    def __init__(self, *overrides, **stub):
        self.config = load_config(None, [*OVERRIDES, *overrides])
        self.kvcr = StubKvcr(**stub)
        self.window = Window()
        self.engine = SglangEngine(self.config, self.window, worker=0)
        self.engine.set_peers([("n0", "tcp://self:1"), ("n1", "tcp://peer:1")])
        port = LinkerPort(StubHost(self.kvcr), call_timeout_s=10)
        self.engine.attach(port._host, port)
        self.port = port
        self.dones = []
        self.run(lambda: not self.engine._pending)  # the warm page's backup
        self.kvcr.calls.clear()
        self.window.reset()
        self.router = Inventory(1)
        self.held()

    def held(self, tier=TIER_G2, stream=SESSION, pages=64):
        """What the router believes this worker holds of a stream."""
        self.router.apply(0, self.engine.take_inventory())
        return self.router.prefix(0, [(stream, 0, pages)], (tier,))

    def run(self, until, limit=5.0):
        deadline = time.monotonic() + limit
        while not until():
            self.dones.extend(self.engine.step())
            if time.monotonic() > deadline:
                raise AssertionError("the engine never got there")

    def finish_requests(self, count):
        self.run(lambda: len(self.dones) >= count)

    def finish(self, count):
        self.run(lambda: len(self.dones) >= count and not self.engine._pending)

    def close(self):
        self.engine.close()


@pytest.fixture
def rig_factory():
    rigs = []

    def make(*overrides, **stub):
        rigs.append(Rig(*overrides, **stub))
        return rigs[-1]

    yield make
    for rig in rigs:
        rig.close()


def request(rid, system=4, context=6, output=1, extra=10, hint_source=-1, hint=0):
    segments = [(SYSTEM, 0, system), (SESSION, 0, context)]
    return Request(
        rid=rid,
        session=1,
        segments=segments,
        tail=(SESSION, context, context + output),
        prompt_tokens=(system + context) * PAGE + extra,
        output_tokens=output * PAGE,
        hint_source=hint_source,
        hint_blocks=hint,
    )


def ids(system=4, context=6):
    return segment_ids([(SYSTEM, 0, system), (SESSION, 0, context)])


def deposited(kvcr):
    return [k for _, refs in kvcr.of("deposit") for k in refs]


def test_a_cold_request_is_looked_up_once_then_backed_up(rig_factory):
    rig = rig_factory()
    rig.engine.submit(request(1))
    rig.finish(1)
    kvcr = rig.kvcr

    scope = f"32:{1:032x}:0"
    assert kvcr.of("query") == [("query", [key(b) for b in ids()], scope)]
    assert not kvcr.of("submit_hint") and not kvcr.of("deliver")
    # One node per stream after the prefill, then the answer's page.
    assert [list(refs) for _, refs in kvcr.of("deposit")] == [
        [key(b) for b in ids()[:4]],
        [key(b) for b in ids()[4:]],
        [key((SESSION << 24) + 6)],
    ]
    refs = kvcr.of("deposit")[0][1][key(ids()[0])]
    assert len(refs) == 2 * LAYERS and len({slot for _, slot in refs}) == 1
    assert kvcr.of("discard_hint") == [("discard_hint", scope)]
    done = rig.dones[0]
    assert not done.aborted and done.prompt_blocks == 10 and done.gpu_blocks == 0
    assert done.recomputed_tokens == 10 * PAGE + 10
    assert rig.window.counters["deposit_blocks_ok"] == 11

    # The router hears about both tiers, as root-aligned runs.
    assert rig.held(TIER_DEVICE, SYSTEM) == 4 and rig.held(TIER_DEVICE) == 7
    assert rig.held(TIER_G2, SYSTEM) == 4 and rig.held(TIER_G2) == 7


def test_the_next_turn_hits_the_device_and_only_backs_up_new_pages(rig_factory):
    rig = rig_factory()
    rig.engine.submit(request(1))
    rig.finish(1)
    rig.kvcr.calls.clear()
    rig.engine.submit(request(2, context=9))
    rig.finish(2)
    kvcr = rig.kvcr

    # Pages 0..6 of the session are on the device: only 7 and 8 are looked up.
    looked = [key((SESSION << 24) + i) for i in (7, 8)]
    assert kvcr.of("query") == [("query", looked, f"32:{2:032x}:0")]
    assert deposited(kvcr) == [*looked, key((SESSION << 24) + 9)]
    assert rig.dones[1].gpu_blocks == 4 + 7


def test_a_fully_cached_prompt_makes_no_linker_lookup(rig_factory):
    rig = rig_factory()
    rig.engine.submit(request(1))
    rig.finish(1)
    rig.kvcr.calls.clear()
    rig.engine.submit(request(2, context=7, extra=1))
    rig.finish(2)
    assert not rig.kvcr.of("query") and not rig.kvcr.of("discard_hint")


def test_hinted_pages_are_loaded_per_layer_then_backed_up_again(rig_factory):
    rig = rig_factory()
    scope = f"32:{1:032x}:0"
    rig.kvcr.fetchable[scope] = {key(b) for b in ids()[:8]}
    rig.engine.submit(request(1, hint_source=1, hint=8))
    rig.finish(1)
    kvcr = rig.kvcr

    hint = kvcr.of("submit_hint")
    assert hint == [
        (
            "submit_hint",
            scope,
            {"source_control_endpoint": "tcp://peer:1", "block_hashes": ids()[:8]},
        )
    ]
    delivers = kvcr.of("deliver")
    assert len(delivers) == LAYERS and kvcr.max_delivers <= 4
    assert all(list(refs) == [key(b) for b in ids()[:8]] for _, refs, _ in delivers)
    assert all(request_id == scope for _, _, request_id in delivers)
    # Restored pages enter the tree as new nodes and are deposited again.
    assert set(deposited(kvcr)) >= {key(b) for b in ids()}
    done = rig.dones[0]
    assert done.remote_blocks == 8 and done.recomputed_tokens == 2 * PAGE + 10
    assert rig.window.counters["deliver_blocks_remote"] == 8


def test_a_failed_key_aborts_every_request_loading_in_the_batch(rig_factory):
    rig = rig_factory()
    for rid, session in ((1, 70), (2, 80)):
        scope = f"32:{rid:032x}:0"
        pages = segment_ids([(SYSTEM, 0, 4), (session, 0, 6)])
        rig.kvcr.fetchable[scope] = set(map(key, pages[:8]))
        rig.engine.submit(
            Request(
                rid=rid,
                session=rid,
                segments=[(SYSTEM, 0, 4), (session, 0, 6)],
                tail=(session, 6, 7),
                prompt_tokens=10 * PAGE + 10,
                output_tokens=PAGE,
                hint_source=1,
                hint_blocks=8,
            )
        )
    # A third request in the same batch that loads nothing.
    rig.engine.submit(
        Request(
            rid=3,
            session=3,
            segments=[(90, 0, 3)],
            tail=(90, 3, 4),
            prompt_tokens=3 * PAGE + 5,
            output_tokens=PAGE,
        )
    )
    rig.kvcr.fail = {key((80 << 24) + 2)}
    rig.finish(3)

    by_rid = {done.rid: done for done in rig.dones}
    assert by_rid[1].aborted and by_rid[2].aborted and not by_rid[3].aborted
    counters = rig.window.counters
    assert counters["aborted_requests"] == 2 and counters["requests"] == 1
    assert counters["load_batches_failed"] == 1
    # Nothing of the aborted requests entered the tree or KVCR.
    assert set(deposited(rig.kvcr)) == {key((90 << 24) + i) for i in range(4)}
    assert rig.engine._tree.held(70) == 0 and rig.engine._tree.held(80) == 0
    assert len(rig.engine._tree) == 4


def test_a_request_without_room_waits_for_a_running_one_to_leave(rig_factory):
    rig = rig_factory("tiers.gpu_blocks=24", "engine.max_running=8")
    scope = f"32:{2:032x}:0"
    # A long answer keeps the first request, and its locks, alive for a while.
    rig.engine.submit(request(1, output=6))
    rig.run(lambda: bool(rig.engine._running))
    rig.kvcr.calls.clear()
    second = Request(
        rid=2,
        session=2,
        segments=[(SYSTEM, 0, 4), (80, 0, 12)],
        tail=(80, 12, 13),
        prompt_tokens=16 * PAGE + 5,
        output_tokens=PAGE,
        hint_source=1,
        hint_blocks=4,
    )
    rig.engine.submit(second)
    rig.finish(2)

    queries = [call for call in rig.kvcr.of("query") if call[2] == scope]
    hints = [call for call in rig.kvcr.of("submit_hint") if call[1] == scope]
    # Matched once and refused (NO_TOKEN), which shuts admission; matched
    # again, with its hint, only when the running request left.
    assert len(queries) == 2 and len(hints) == 2
    assert rig.window.counters["device_alloc_failures"] == 1
    assert not rig.dones[1].aborted


def test_with_nothing_running_a_refused_request_is_matched_every_iteration(
    rig_factory,
):
    rig = rig_factory("tiers.gpu_blocks=24")
    rig.kvcr.delay = 10**9  # backups stay pending, so their nodes stay locked
    rig.engine.submit(request(1))
    rig.finish_requests(1)
    rig.kvcr.calls.clear()
    rig.engine.submit(
        Request(
            rid=2,
            session=2,
            segments=[(80, 0, 14)],
            tail=(80, 14, 15),
            prompt_tokens=14 * PAGE + 5,
            output_tokens=PAGE,
        )
    )
    for _ in range(20):
        rig.engine.step()
    assert len(rig.kvcr.of("query")) == 20
    assert rig.window.counters["device_alloc_failures"] == 20
    assert not rig.kvcr.of("deliver")
    # Once the backups complete the nodes can be evicted and it is admitted.
    rig.kvcr.delay = 1
    rig.finish(2)
    assert not rig.dones[1].aborted


def test_a_kvcr_removal_is_backed_up_again_on_the_next_turn(rig_factory):
    rig = rig_factory()
    rig.engine.submit(request(1))
    rig.finish(1)
    assert rig.held() == 7
    evicted = key((SESSION << 24) + 2)
    rig.port.record_inventory(_Event([evicted], removed=True))
    rig.run(lambda: rig.window.counters["kvcr_removals"] == 1)
    # The router's prefix for the stream is cut at the removed page.
    assert rig.held() == 2

    rig.kvcr.calls.clear()
    rig.kvcr.resident.discard(evicted)
    rig.engine.submit(request(2, context=9))
    rig.finish(2)
    # The whole node holding the removed page goes again, with the new pages.
    assert set(deposited(rig.kvcr)) == {
        key((SESSION << 24) + i) for i in (0, 1, 2, 3, 4, 5, 7, 8, 9)
    }
    assert rig.held() == 10


def test_a_long_prefill_is_chunked_and_each_chunk_backed_up(rig_factory):
    rig = rig_factory(f"engine.max_batched_tokens={4 * PAGE}")
    rig.engine.submit(request(1, system=0, context=10))
    rig.finish(1)
    sizes = [len(refs) for _, refs in rig.kvcr.of("deposit")]
    # Three chunks of four pages (the last one short), then the answer's page.
    assert sizes == [4, 4, 2, 1]
    assert len(rig.kvcr.of("query")) == 1  # a continuing chunk is not matched


def test_every_kvcr_call_is_made_by_the_owner_thread(rig_factory):
    rig = rig_factory()
    scope = f"32:{1:032x}:0"
    rig.kvcr.fetchable[scope] = {key(b) for b in ids()[:8]}
    rig.engine.submit(request(1, hint_source=1, hint=8))
    rig.finish(1)
    assert rig.kvcr.threads == {"kvcr-owner"}
    assert rig.port.take_stats().calls["deposit"] == 4
    # Idle iterations still reach KVCR: the owner polls on every command.
    polls = rig.kvcr.polls
    for _ in range(50):
        rig.engine.step()
    assert rig.kvcr.polls >= polls + 50
    stats = rig.port.take_stats()
    assert stats.calls["poll_empty"] >= 50 and stats.calls["deposit"] == 0


def test_the_window_gets_the_owner_threads_calls(rig_factory):
    rig = rig_factory()
    rig.engine.begin_window()
    rig.engine.submit(request(1))
    rig.finish(1)
    rig.engine.end_window()
    window = rig.window
    assert window.calls["query"] == 1 and window.items["query"] == 10
    assert window.calls["deposit"] == 3 and window.items["deposit"] == 11
    assert window.calls["poll_empty"] > 0 and window.secs["deposit"] > 0
    assert window.counters["owner_loops"] > 0
    assert window.counters["linker_wait_seconds"] > 0


def test_an_owner_failure_surfaces_in_the_next_step(rig_factory):
    rig = rig_factory()
    rig.kvcr.raise_on_poll = ValueError("boom")
    with pytest.raises(RuntimeError):
        for _ in range(50):
            rig.engine.step()
            time.sleep(0.002)


def test_close_shuts_kvcr_down_on_the_owner_thread(rig_factory):
    rig = rig_factory()
    rig.engine.close()
    assert rig.kvcr.closed_on == "kvcr-owner"
    assert not rig.port._thread.is_alive()
    rig.engine.close()  # idempotent


def test_stopping_drops_requests_and_unstarted_backups(rig_factory):
    rig = rig_factory()
    rig.kvcr.delay = 10**9  # nothing completes on its own from here
    rig.engine.submit(request(1))
    rig.engine.submit(
        Request(
            rid=2,
            session=2,
            segments=[(90, 0, 3)],
            tail=(90, 3, 4),
            prompt_tokens=3 * PAGE + 5,
            output_tokens=4 * PAGE,
        )
    )
    rig.run(lambda: len(rig.engine._pending) >= 3)
    rig.engine.abort_requests()
    assert not rig.engine._waiting and not rig.engine._running
    # Only the deposit already issued is still owed.
    assert len(rig.engine._pending) == 1
    rig.kvcr.delay = 1
    rig.run(lambda: not rig.engine.busy())


# Trace replay: a prompt is the branch-free runs on its path through the
# session's prefix tree, and the answer's page opens the run the next
# request continues on.
PATH = [(900, 0, 2), (901, 0, 1), (902, 0, 3), (903, 0, 1), (904, 0, 2)]


def test_an_answer_that_opens_a_stream_is_cached_for_the_turn_that_names_it(
    rig_factory,
):
    rig = rig_factory()
    rig.engine.submit(
        Request(
            rid=1,
            session=1,
            segments=PATH,
            tail=(905, 0, 1),
            prompt_tokens=9 * PAGE + 10,
            output_tokens=PAGE,
        )
    )
    rig.finish(1)
    # The answer's page is backed up last, and each stream hangs from the one
    # before it, the answer's from the end of the path.
    assert deposited(rig.kvcr)[-1] == key(905 << 24)
    tree = rig.engine._tree
    chain = tree.ancestry(tree.node_at(905, 0))
    assert [node.stream for node in chain] == [900, 901, 902, 903, 904, 905]
    rig.kvcr.calls.clear()
    rig.engine.submit(
        Request(
            rid=2,
            session=1,
            segments=[*PATH, (905, 0, 3)],
            tail=(905, 3, 3),
            prompt_tokens=12 * PAGE + 5,
            output_tokens=5,
        )
    )
    rig.finish(2)
    assert rig.dones[1].gpu_blocks == 10  # the whole old path and the answer
    looked = [key((905 << 24) + i) for i in (1, 2)]
    assert rig.kvcr.of("query") == [("query", looked, f"32:{2:032x}:0")]


def test_a_layer_computes_once_it_has_landed_and_the_one_before_has_computed():
    # Four layers of 10 ms each, landing 10 ms apart: only the first landing
    # holds the forward up, so it ends 10 ms late, not 40.
    assert forward_end(0.0, [0.01, 0.02, 0.03, 0.04], 0.01) == pytest.approx(0.05)
    # Loads that land before they are needed cost nothing.
    assert forward_end(1.0, [0.2, 0.4], 0.5) == 2.0
    # A late last layer delays only itself.
    assert forward_end(0.0, [0.0, 0.0, 0.0, 0.1], 0.01) == pytest.approx(0.11)


def test_a_layer_that_never_lands_raises_instead_of_hanging():
    counter = LayerCounter(2, timeout_s=0.01)
    counter.set_consumer(counter.update_producer())
    with pytest.raises(TimeoutError):
        counter.wait_until(0)


# ---- what a failed batch transferred, and whose failure it was ----------------


def test_pages_delivered_before_a_late_failure_count_as_transferred(rig_factory):
    rig = rig_factory()
    rig.kvcr.fetchable[f"32:{1:032x}:0"] = {key(b) for b in ids()[:8]}
    rig.kvcr.fail = {key(ids()[7])}
    rig.kvcr.fail_layer = LAYERS - 1
    rig.engine.submit(request(1, hint_source=1, hint=8))
    rig.finish_requests(1)
    (done,) = rig.dones
    counters = rig.window.counters
    # The request is aborted, but seven of its eight pages had every layer.
    assert done.aborted and done.failed_blocks == 1
    assert counters["deliver_blocks_remote"] == 7
    assert counters["hinted_failed|unavailable"] == 1
    assert counters["deliver_blocks_unissued"] == 0


def test_a_failure_belongs_to_the_request_whose_delivery_failed(rig_factory):
    rig = rig_factory()
    # Both requests load the system prefix; only the second one's delivery of
    # its first page times out.
    rig.kvcr.failure = OpEntryResult(OpEntryStatus.TIMEOUT, reason="deadline")
    rig.kvcr.fail_scope = f"32:{2:032x}:0"
    for rid, session in ((1, 70), (2, 80)):
        pages = segment_ids([(SYSTEM, 0, 4), (session, 0, 6)])
        rig.kvcr.fetchable[f"32:{rid:032x}:0"] = set(map(key, pages[:8]))
        rig.kvcr.fail.add(key(pages[0]))
        rig.engine.submit(
            Request(
                rid=rid,
                session=rid,
                segments=[(SYSTEM, 0, 4), (session, 0, 6)],
                tail=(session, 6, 7),
                prompt_tokens=10 * PAGE + 10,
                output_tokens=PAGE,
                hint_source=1,
                hint_blocks=8,
            )
        )
    rig.finish_requests(2)
    by_rid = {done.rid: done for done in rig.dones}
    assert (by_rid[1].failed_blocks, by_rid[2].failed_blocks) == (0, 1)
    assert (by_rid[1].timed_out_blocks, by_rid[2].timed_out_blocks) == (0, 1)
    counters = rig.window.counters
    assert counters["hinted_failed|timeout"] == 1
    assert counters["hinted_failed_reason|deadline"] == 1
    # It failed in the first layer: no other page had every layer delivered.
    assert counters["deliver_blocks_failed"] == 1
    assert counters["deliver_blocks_unissued"] == 15
    assert counters["deliver_blocks_remote"] == 0


def test_a_chunked_request_whose_load_fails_is_dropped(rig_factory):
    rig = rig_factory("engine.max_batched_tokens=128")
    rig.kvcr.fetchable[f"32:{1:032x}:0"] = {key(b) for b in ids()[:2]}
    rig.kvcr.fail = {key(ids()[0])}
    rig.engine.submit(request(1, hint_source=1, hint=2))
    rig.finish_requests(1)
    assert rig.dones[0].aborted
    assert rig.engine._chunked is None and not rig.engine.busy()
    assert len(rig.engine._tree) == 0


# ---- device room, locks and timing ------------------------------------------


def test_a_running_request_holds_device_pages_for_all_its_tokens(rig_factory):
    rig = rig_factory()
    # No cacheable block at all: 40 prompt tokens and a 200-token answer.
    rig.engine.submit(
        Request(
            rid=1,
            session=1,
            segments=[],
            tail=(SESSION, 0, 0),
            prompt_tokens=40,
            output_tokens=200,
        )
    )
    rig.run(lambda: rig.engine._running)
    assert len(rig.engine._tree) == 4  # 240 tokens of 64-token pages
    rig.finish(1)
    assert len(rig.engine._tree) == 0  # nothing of it is cacheable


def test_a_request_the_device_cannot_hold_is_refused(rig_factory):
    rig = rig_factory()
    with pytest.raises(ValueError, match="needs 300 device blocks"):
        rig.engine.submit(request(1, context=295, extra=0))


def test_a_request_keeps_pages_another_request_inserted_locked(rig_factory):
    rig = rig_factory()
    # Two requests compute the same cold prefix in one prefill batch. The
    # first finishes at once; the second keeps decoding.
    rig.engine.submit(request(1, extra=0, output=0))
    rig.engine.submit(
        Request(
            rid=2,
            session=1,
            segments=[(SYSTEM, 0, 4), (SESSION, 0, 6)],
            tail=(SESSION, 6, 6),
            prompt_tokens=10 * PAGE,
            output_tokens=10_000,
        )
    )
    rig.run(lambda: rig.dones and not rig.engine._pending)
    assert [done.rid for done in rig.dones] == [1]
    tree = rig.engine._tree
    # The first request and its backups are gone; the second still holds the
    # pages it uses, so nothing can evict them from under it.
    assert tree.node_at(SYSTEM, 0).locks == 1
    assert tree.node_at(SESSION, 0).locks == 1
    assert tree.allocate(250) is None


def test_load_time_runs_to_the_end_of_the_transfer(rig_factory):
    rig = rig_factory("engine.step_ms=30", delay=200)
    rig.kvcr.fetchable[f"32:{1:032x}:0"] = {key(b) for b in ids()[:8]}
    rig.engine.submit(request(1, output=0, hint_source=1, hint=8))
    rig.finish_requests(1)
    (done,) = rig.dones
    hists = rig.window.hists
    # From the first lookup to the end of the load on the owner's clock: it
    # covers the transfer, and the modelled forward is no part of it.
    assert done.load_ms >= hists["deliver_remote"].total * 1e3 > 0
    assert hists["cache_resolve_remote"].total * 1e3 == pytest.approx(done.load_ms)
    assert done.load_ms < 30


def test_backup_latency_ends_when_the_deposit_does(rig_factory):
    rig = rig_factory("engine.step_ms=40")
    rig.engine.submit(
        Request(
            rid=1,
            session=1,
            segments=[(SESSION, 0, 6)],
            tail=(SESSION, 6, 6),
            prompt_tokens=6 * PAGE,
            output_tokens=3,
        )
    )
    rig.finish(1)
    # Deposits finish during 40 ms forwards; the scheduler only looks after.
    assert 0 < rig.window.hists["deposit"].max < 0.03


def test_a_page_in_kvcrs_pool_is_restored_as_a_local_block(rig_factory):
    rig = rig_factory()
    rig.kvcr.resident.update(key(b) for b in ids()[:8])
    rig.engine.submit(request(1))
    rig.finish(1)
    (done,) = rig.dones
    assert (done.local_blocks, done.remote_blocks) == (8, 0)
    counters = rig.window.counters
    assert counters["deliver_blocks_local"] == 8
    assert counters["deliver_blocks_remote"] == 0


def test_mesh_warm_pulls_one_page_from_each_peer_and_retries(rig_factory):
    rig = rig_factory()
    good, bad = block_id(warm_stream(1), 0), block_id(warm_stream(2), 0)
    kvcr = rig.kvcr
    for attempt in range(5):
        kvcr.fetchable[f"8:warm-1-{attempt}:0"] = {key(good)}
    rig.engine.start_warm([(1, "tcp://peer:1"), (2, "tcp://gone:1")])
    rig.run(lambda: rig.engine.warm_status()[0] == 0)
    # The reachable peer's page lands on the first try; the other is asked
    # five times, then counted as failed.
    assert rig.engine.warm_status() == (0, 1, 1)
    assert [list(refs) for _, refs, _ in kvcr.of("deliver")] == [[key(good)]] * LAYERS
    asked = [keys for _, keys, _ in kvcr.of("query")]
    assert asked.count([key(bad)]) == 5 and asked.count([key(good)]) == 1
    assert not rig.engine.busy()
