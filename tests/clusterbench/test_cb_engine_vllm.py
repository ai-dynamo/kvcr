# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The fake vLLM engine must put the real tier's call pattern on KVCR.

These drive the engine against a recording stand-in for KVCR and assert the
exact calls per step. If one fails after an engine change, compare against
vLLM's KVCRSecondaryTierManager before touching the expectation.
"""

from clusterbench.config import EngineConfig, PhaseConfig, RunConfig, TierConfig
from clusterbench.engine_vllm import VllmEngine, _encode
from clusterbench.ids import block_id, id_to_key, key_to_id, segment_ids
from clusterbench.metrics import Window
from clusterbench.protocol import TIER_DEVICE, TIER_PRIMARY, Request
from clusterbench.router import Inventory

from kvcr.types import CacheTier, QueryStatus

BLOCK = 64
SEGMENTS = [(1, 0, 4), (70, 0, 6)]
TAIL = (70, 6, 7)
KEYS = [id_to_key(bid) for bid in segment_ids(SEGMENTS)]
PROMPT = 10 * BLOCK + 10


class _Entry:
    def __init__(self, success):
        self.success = success


class StubKvcr:
    """Records every call; operations complete at the next poll."""

    def __init__(self, host):
        self.host = host
        self.calls = []
        self.hinted = set()  # request ids with a usable hint
        self.remote_ok = set()  # keys a peer can serve
        self.deposit_ok = True
        self._failed = set()
        self._pending = []
        self._next = 100
        self.on_deliver = None

    def names(self):
        return [call[0] for call in self.calls]

    def of(self, name):
        return [call for call in self.calls if call[0] == name]

    def submit_hint(self, hints, request_id=None):
        payload = hints.actions[0].payload
        self.calls.append(("submit_hint", request_id, payload))
        self.hinted.add(request_id)

    def discard_hint(self, request_id):
        self.calls.append(("discard_hint", request_id))
        self.hinted.discard(request_id)

    def query(self, keys, request_id=None):
        self.calls.append(("query", tuple(keys), request_id))
        out = []
        for key in keys:
            if request_id in self.hinted and (request_id, key) not in self._failed:
                out.append((QueryStatus.FETCHABLE, CacheTier.REMOTE_G2))
            else:
                out.append((QueryStatus.MISS, None))
        return out

    def _op(self, entries):
        self._next += 1
        self._pending.append((self._next, entries))
        return self._next

    def deliver(self, blocks, request_id=None):
        self.calls.append(("deliver", dict(blocks), request_id))
        if self.on_deliver:
            self.on_deliver()
        entries = {}
        for key, refs in blocks.items():
            ok = key in self.remote_ok
            entries[key] = _Entry(ok)
            if ok:  # the peer's bytes land in the destination slot
                self.host.stamp(refs[0].element_index, key_to_id(key))
            else:
                self._failed.add((request_id, key))
        return self._op(entries)

    def deposit(self, blocks):
        self.calls.append(("deposit", dict(blocks)))
        return self._op({key: _Entry(self.deposit_ok) for key in blocks})

    def poll_completed(self):
        self.calls.append(("poll",))
        done, self._pending = self._pending, []
        return done

    def align_sequence(self, keys, use_current_time=False):
        self.calls.append(("align", list(keys), use_current_time))

    def get_stats(self):
        self.calls.append(("get_stats",))
        return None


class StubHost:
    agent_name = "stub-agent"

    def __init__(self):
        self.stamps = {}
        self.kvcr = StubKvcr(self)

    def stamp(self, slot, bid):
        self.stamps[slot] = bid

    def stamp_many(self, slots, keys):
        for slot, key in zip(slots, keys):
            self.stamps[slot] = key_to_id(key)

    def stamp_at(self, slot):
        return self.stamps.get(slot, 0)


def _engine(**engine_overrides):
    config = RunConfig(
        tiers=TierConfig(
            gpu_blocks=64, primary_blocks=64, g2_blocks=0, tokens_per_block=BLOCK
        ),
        engine=EngineConfig(
            step_ms=0,
            prefill_us_per_token=0,
            idle_sleep_ms=0,
            verify="full",
            **engine_overrides,
        ),
        phases=PhaseConfig(warm_concurrency=2),
    )
    window = Window()
    engine = VllmEngine(config, window, worker=0)
    engine.set_peers([("n0", "tcp://a:1"), ("n1", "tcp://b:2")])
    host = StubHost()
    engine.attach(host)
    return engine, host.kvcr, window


def _request(rid, **kwargs):
    return Request(
        rid=rid,
        session=0,
        segments=SEGMENTS,
        tail=TAIL,
        prompt_tokens=PROMPT,
        output_tokens=3,
        **kwargs,
    )


def _run(engine, kvcr, steps=20):
    """Step until a request finishes; return (done, calls per step)."""
    per_step = []
    for _ in range(steps):
        before = len(kvcr.calls)
        dones = engine.step()
        per_step.append(kvcr.names()[before:])
        if dones:
            return dones[0], per_step
    raise AssertionError("request did not finish")


def test_cold_request_stores_once_and_aligns_the_whole_prefix():
    engine, kvcr, window = _engine()
    engine.submit(_request(1))
    assert kvcr.names() == []  # no hint: submit_hint is not called
    done, steps = _run(engine, kvcr)

    # Lookup: one single-key query, which misses and stops the scan.
    assert kvcr.of("query") == [("query", (KEYS[0],), "r1")]
    assert not kvcr.of("deliver")
    # The whole prompt is prefilled in one step and deposited as one batch.
    (deposit,) = kvcr.of("deposit")
    assert list(deposit[1]) == KEYS
    slots = [refs[0].element_index for refs in deposit[1].values()]
    assert len(set(slots)) == 10 and min(slots) >= 3  # past the reserved slots
    assert all(refs[0].end_point_name == "stub-agent" for refs in deposit[1].values())
    # align_sequence: whole prefix after the deposit completes, and at finish.
    aligns = kvcr.of("align")
    assert [(a[1], a[2]) for a in aligns] == [(KEYS, False), (KEYS, True)]
    assert kvcr.names()[-3:-1] == ["align", "discard_hint"]

    # Step 1: poll at the first lookup, again for the request's store, and
    # the deposit after the forward pass. Later steps poll once.
    assert steps[0] == ["poll", "query", "poll", "deposit", "get_stats"]
    assert steps[1] == ["poll", "align", "get_stats"]
    assert steps[2] == ["poll", "align", "discard_hint", "get_stats"]
    assert (done.prompt_blocks, done.gpu_blocks, done.remote_blocks) == (10, 0, 0)
    assert done.recomputed_tokens == PROMPT

    deltas = engine.take_inventory()
    assert (TIER_PRIMARY, 1, 0, 4, False) in deltas
    assert (TIER_PRIMARY, 70, 0, 6, False) in deltas
    assert (TIER_DEVICE, 70, 0, 7, False) in deltas  # the answer's block too
    assert engine.take_inventory() == []
    assert not engine.busy()


def test_second_turn_on_the_same_worker_hits_the_device_cache():
    engine, kvcr, _ = _engine()
    engine.submit(_request(1))
    _run(engine, kvcr)
    kvcr.calls.clear()
    grown = Request(
        rid=2,
        session=0,
        segments=[(1, 0, 4), (70, 0, 9)],
        tail=(70, 9, 10),
        prompt_tokens=13 * BLOCK + 5,
        output_tokens=2,
    )
    engine.submit(grown)
    done, _ = _run(engine, kvcr)
    assert done.gpu_blocks == 11  # ten prompt blocks and the answer's block
    # The scan resumes after the device prefix: one query, for block 11.
    assert [key_to_id(q[1][0]) for q in kvcr.of("query")] == [block_id(70, 7)]
    # No load happened, so storing starts from block 0 again; only blocks the
    # primary tier lacks are written and deposited.
    (deposit,) = kvcr.of("deposit")
    assert [key_to_id(k) for k in deposit[1]] == [block_id(70, i) for i in (6, 7, 8)]


def test_hinted_request_promotes_through_one_deliver_and_keeps_the_prefix():
    engine, kvcr, window = _engine()
    kvcr.remote_ok = set(KEYS[:6])
    engine.submit(_request(2, hint_source=1, hint_blocks=6))
    (hint,) = kvcr.of("submit_hint")
    assert hint[1] == "r2"
    assert hint[2] == {
        "source_control_endpoint": "tcp://b:2",
        "block_hashes": segment_ids(SEGMENTS)[:6],
    }
    done, steps = _run(engine, kvcr)

    # Opportunistic query: the scan walks the whole prompt, one key per call,
    # then asks again only for the first key that failed.
    queries = kvcr.of("query")
    assert all(len(q[1]) == 1 and q[2] == "r2" for q in queries)
    assert [q[1][0] for q in queries] == KEYS + [KEYS[6]]
    # One deliver for the request, into distinct primary-tier slots.
    (deliver,) = kvcr.of("deliver")
    assert list(deliver[1]) == KEYS and deliver[2] == "r2"
    slots = [refs[0].element_index for refs in deliver[1].values()]
    assert len(set(slots)) == 10 and min(slots) >= 3
    assert steps[0] == ["poll"] + ["query"] * 10 + ["deliver", "get_stats"]
    # The loaded prefix is not stored again; only the recomputed blocks are.
    (deposit,) = kvcr.of("deposit")
    assert list(deposit[1]) == KEYS[6:]
    # Every completed job aligns the prefix up to its last block.
    assert [(len(a[1]), a[2]) for a in kvcr.of("align")] == [
        (10, False),  # deliver completed (its last block is block 9)
        (10, False),  # deposit completed
        (10, True),  # request finished
    ]
    assert (done.remote_blocks, done.failed_blocks, done.primary_blocks) == (6, 4, 0)
    assert done.recomputed_tokens == PROMPT - 6 * BLOCK
    counters = window.counters
    assert counters["deliver_blocks_failed_hinted"] == 0  # all past the hint
    assert counters["bytes_checked"] == 6 and not counters["bytes_mismatch"]


def test_failures_inside_the_hint_and_wrong_bytes_are_counted():
    engine, kvcr, window = _engine()
    kvcr.remote_ok = set(KEYS[:3])
    engine.submit(_request(3, hint_source=1, hint_blocks=6))
    original = kvcr.host.stamp
    kvcr.host.stamp = lambda slot, bid: original(slot, bid + 1)  # corrupt
    engine.step()
    kvcr.host.stamp = original
    done, _ = _run(engine, kvcr)
    assert (done.remote_blocks, done.failed_blocks) == (3, 7)
    assert window.counters["deliver_blocks_failed_hinted"] == 3
    assert window.counters["bytes_mismatch"] == 3


def test_full_primary_tier_turns_a_fetchable_block_into_a_miss():
    engine, kvcr, _ = _engine()
    kvcr.remote_ok = set(KEYS)
    engine._primary._free = engine._primary._free[:4]  # room for four blocks
    engine.submit(_request(4, hint_source=1, hint_blocks=10))
    engine.step()
    (deliver,) = kvcr.of("deliver")
    assert list(deliver[1]) == KEYS[:4]
    assert len(kvcr.of("query")) == 5  # the fifth found no slot and stopped the scan


def test_pins_are_served_once_per_step_from_the_primary_tier():
    engine, kvcr, _ = _engine()
    engine.submit(_request(1))
    _run(engine, kvcr)
    engine.step()  # let the deposit's read pin drop
    missing = id_to_key(block_id(999, 0))
    request = engine.request_pin([KEYS[0], KEYS[1], missing])
    nothing = engine.request_pin([missing])
    cancelled = engine.request_pin([KEYS[2]])
    engine.cancel_pin_request(cancelled)
    assert engine.poll_pin_results() == []  # not resolved until the step ends
    engine.step()
    results = dict(engine.poll_pin_results())
    assert set(results) == {request, nothing}
    assert results[nothing] is None
    handle, refs = results[request]
    assert refs[missing] is None
    primary = engine._primary
    assert refs[KEYS[0]][0].element_index == primary.slot(KEYS[0])
    assert primary._ref[KEYS[0]] == primary._ref[KEYS[1]] == 1
    assert engine.busy()
    assert engine.release_pin(handle) is True
    assert primary._ref[KEYS[0]] == 1  # released at the next poll, not now
    engine.step()
    assert primary._ref[KEYS[0]] == primary._ref[KEYS[1]] == 0
    assert not engine.busy()


def test_pins_are_resolved_before_promotions_are_delivered():
    engine, kvcr, _ = _engine()
    engine.submit(_request(1))
    _run(engine, kvcr)
    engine.step()
    kvcr.remote_ok = set()
    seen = []
    kvcr.on_deliver = lambda: seen.append(len(engine._completed_pins))
    engine.request_pin([KEYS[0]])
    other = Request(
        rid=5,
        session=1,
        segments=[(2, 0, 3)],
        tail=(71, 0, 0),
        prompt_tokens=200,
        output_tokens=1,
        hint_source=1,
        hint_blocks=3,
    )
    engine.submit(other)
    engine.step()
    assert seen == [1]


def test_decode_blocks_are_deposited_when_prompt_only_is_off():
    engine, kvcr, _ = _engine(offload_prompt_only=False)
    request = Request(
        rid=6,
        session=0,
        segments=[(70, 0, 2)],
        tail=(70, 2, 4),
        prompt_tokens=2 * BLOCK,
        output_tokens=2 * BLOCK + 1,
    )
    engine.submit(request)
    done, _ = _run(engine, kvcr, steps=400)
    deposits = [[key_to_id(k) for k in call[1]] for call in kvcr.of("deposit")]
    assert deposits == [
        [block_id(70, 0), block_id(70, 1)],  # the prompt
        [block_id(70, 2)],  # one deposit per block the decode completes
        [block_id(70, 3)],
    ]
    # The last block's deposit is issued in the step the request finishes
    # and completes afterwards; that completion still aligns the prefix.
    engine.step()
    assert [(len(a[1]), a[2]) for a in kvcr.of("align")] == [
        (2, False),
        (3, False),
        (4, True),
        (4, False),
    ]


def test_idle_steps_still_poll_and_report_stats():
    engine, kvcr, window = _engine()
    for _ in range(3):
        assert engine.step() == []
    assert kvcr.names() == ["poll", "get_stats"] * 3
    assert window.counters["steps"] == 3 and not window.counters["busy_steps"]


def test_warm_pulls_one_block_from_every_peer_and_retries():
    engine, kvcr, _ = _engine()
    peer_key = id_to_key(block_id((1 << 20) + 1, 0))
    engine.start_warm([(1, "tcp://b:2")])
    engine.step()
    assert engine.warm_status() == (1, 0, 0)
    (deliver,) = kvcr.of("deliver")
    assert list(deliver[1]) == [peer_key]
    assert deliver[1][peer_key][0].element_index in (1, 2)  # a warm scratch slot
    kvcr.remote_ok = {peer_key}  # the first attempt failed; the retry succeeds
    engine.step()
    engine.step()
    assert engine.warm_status() == (0, 1, 0)
    assert len(kvcr.of("deliver")) == 2
    assert len(kvcr.of("discard_hint")) == 2


def _key(stream, index):
    return id_to_key(block_id(stream, index))


def test_inventory_events_are_run_length_encoded_in_order():
    out = []
    events = [
        (False, [_key(5, 0), _key(5, 1), _key(5, 2)]),
        (True, [_key(5, 2)]),
        (True, [_key(5, 1)]),  # tail-first eviction, joined to the one before
        (False, [_key(6, 0)]),
        (False, [_key(5, 1)]),
    ]
    _encode(out, TIER_PRIMARY, events)
    assert out == [
        (TIER_PRIMARY, 5, 0, 3, False),
        (TIER_PRIMARY, 5, 1, 3, True),
        (TIER_PRIMARY, 6, 0, 1, False),
        (TIER_PRIMARY, 5, 1, 2, False),
    ]


def test_one_inventory_event_can_hold_several_runs():
    out = []
    crossing = [_key(5, 8), _key(5, 9), _key(6, 0), _key(6, 1), _key(6, 2)]
    downward = [_key(6, 2), _key(6, 1), _key(6, 0), _key(5, 9), _key(5, 3)]
    _encode(out, TIER_PRIMARY, [(False, crossing), (True, downward), (False, [])])
    assert out == [
        (TIER_PRIMARY, 5, 8, 10, False),
        (TIER_PRIMARY, 6, 0, 3, False),
        (TIER_PRIMARY, 6, 0, 3, True),
        (TIER_PRIMARY, 5, 3, 10, True),  # one cut at the lowest block removed
    ]
    # Blocks added tail first are not one root-aligned run for the router.
    out = []
    _encode(out, TIER_PRIMARY, [(False, [_key(5, 2), _key(5, 1), _key(5, 0)])])
    assert out == [(TIER_PRIMARY, 5, index, index + 1, False) for index in (2, 1, 0)]


def test_time_outside_kvcr_and_the_harness_is_split_into_forward_and_idle():
    engine, _, window = _engine()
    engine.step()  # nothing to run
    assert "idle_seconds" in window.counters
    assert "forward_seconds" not in window.counters
    engine.submit(_request(1))
    engine.step()
    assert window.counters["forward_seconds"] >= 0.0


def test_scattered_removals_of_a_stream_travel_as_one_delta():
    out = []
    events = [
        (True, [_key(5, 9)]),
        (True, [_key(6, 4)]),
        (True, [_key(5, 2)]),  # not next to block 9: still the same cut
        (True, [_key(6, 7)]),
        (False, [_key(5, 2)]),  # the stream grows again
        (True, [_key(5, 1)]),  # so this is a new cut, after the addition
    ]
    _encode(out, TIER_PRIMARY, events)
    assert out == [
        (TIER_PRIMARY, 5, 2, 10, True),
        (TIER_PRIMARY, 6, 4, 8, True),
        (TIER_PRIMARY, 5, 2, 3, False),
        (TIER_PRIMARY, 5, 1, 2, True),
    ]
    # The router ends up where applying every event one by one would.
    one_by_one = []
    for event in events:
        _encode(one_by_one, TIER_PRIMARY, [event])
    for deltas in (out, one_by_one):
        inventory = Inventory(1)
        inventory.apply(
            0, [(TIER_PRIMARY, 5, 0, 12, False), (TIER_PRIMARY, 6, 0, 12, False)]
        )
        inventory.apply(0, deltas)
        assert inventory.prefix(0, [(5, 0, 12)], (TIER_PRIMARY,)) == 1
        assert inventory.prefix(0, [(6, 0, 12)], (TIER_PRIMARY,)) == 4


def test_inventory_reports_the_device_tier_by_stream():
    engine, kvcr, _ = _engine()
    engine.submit(_request(1))
    for _ in range(8):
        engine.step()
    deltas = engine.take_inventory()
    device = [delta for delta in deltas if delta[0] == TIER_DEVICE]
    assert device == [(TIER_DEVICE, 1, 0, 4, False), (TIER_DEVICE, 70, 0, 7, False)]
    primary = [delta for delta in deltas if delta[0] == TIER_PRIMARY]
    assert primary == [(TIER_PRIMARY, 1, 0, 4, False), (TIER_PRIMARY, 70, 0, 6, False)]
    assert engine.take_inventory() == []


def test_aborting_requests_leaves_only_kvcr_operations_to_drain():
    engine, kvcr, _ = _engine()
    kvcr.remote_ok = set(KEYS)
    engine.submit(_request(7, hint_source=1, hint_blocks=10))
    engine.step()  # the deliver is in flight, the request is waiting on it
    assert engine.busy()
    engine.abort_requests()
    assert kvcr.names()[-1] == "discard_hint"
    assert engine.busy()  # the deliver itself still has to complete
    engine.step()
    assert not engine.busy()
    assert engine.step() == []


def test_aborting_requests_frees_slots_reserved_for_unfinished_writes():
    engine, kvcr, _ = _engine()
    engine.submit(_request(8))
    primary = engine._primary
    before = len(primary)
    # Stop between scheduling and the forward pass: the store is prepared
    # (slots reserved, write in flight) but its copy has not completed.
    engine._complete_copies = lambda: None
    engine.step()
    assert len(primary) == before + 10
    engine.abort_requests()
    assert len(primary) == before
    assert not engine.busy()


def test_what_if_aligning_only_at_finish_drops_the_mid_request_aligns():
    engine, kvcr, _ = _engine(align_on_finish_only=True)
    engine.submit(_request(1))
    _run(engine, kvcr)
    # The deposit still happens; only the align after it completes is gone.
    assert len(kvcr.of("deposit")) == 1
    assert [(a[1], a[2]) for a in kvcr.of("align")] == [(KEYS, True)]
