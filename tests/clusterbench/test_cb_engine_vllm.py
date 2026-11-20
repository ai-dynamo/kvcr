# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The fake vLLM engine must put the real tier's call pattern on KVCR.

These drive the engine against a recording stand-in for KVCR and assert the
exact calls per step. If one fails after an engine change, compare against
vLLM's KVCRSecondaryTierManager before touching the expectation.
"""

import time
from types import SimpleNamespace

import pytest
from clusterbench.config import EngineConfig, PhaseConfig, RunConfig, TierConfig
from clusterbench.engine_vllm import VllmEngine
from clusterbench.ids import block_id, id_to_key, key_to_id, segment_ids
from clusterbench.metrics import Window
from clusterbench.protocol import TIER_DEVICE, TIER_G2, TIER_PRIMARY, Request
from clusterbench.router import Inventory

from kvcr.types import CacheTier, OpEntryResult, OpEntryStatus, QueryStatus

BLOCK = 64
SEGMENTS = [(1, 0, 4), (70, 0, 6)]
TAIL = (70, 6, 7)
KEYS = [id_to_key(bid) for bid in segment_ids(SEGMENTS)]
PROMPT = 10 * BLOCK + 10


OK = OpEntryResult(OpEntryStatus.SUCCESS)


class StubKvcr:
    """Records every call; operations complete at the next poll."""

    def __init__(self, host):
        self.host = host
        self.calls = []
        self.hinted = set()  # request ids with a usable hint
        self.remote_ok = set()  # keys a peer can serve
        self.deposit_ok = True
        # What a key no peer can serve comes back as.
        self.failure = OpEntryResult(OpEntryStatus.UNAVAILABLE, reason="not_resident")
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
            entries[key] = OK if ok else self.failure
            if ok:  # the peer's bytes land in the destination slot
                self.host.stamp(refs[0].element_index, key_to_id(key))
            else:
                self._failed.add((request_id, key))
        return self._op(entries)

    def deposit(self, blocks):
        self.calls.append(("deposit", dict(blocks)))
        failed = OpEntryResult(OpEntryStatus.FAILED, reason="transfer_failed")
        return self._op({key: OK if self.deposit_ok else failed for key in blocks})

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
            **{
                "step_ms": 0,
                "prefill_us_per_token": 0,
                "idle_sleep_ms": 0,
                "verify": "full",
                **engine_overrides,
            }
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
    # align_sequence: the prefix after the deposit completes, and every block
    # the request processed, the answer's too, at finish.
    aligns = kvcr.of("align")
    assert [(a[1], a[2]) for a in aligns] == [
        (KEYS, False),
        (KEYS + [_key(*TAIL[:2])], True),
    ]
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
        (11, True),  # request finished: the answer's block too
    ]
    assert (done.remote_blocks, done.failed_blocks, done.primary_blocks) == (6, 4, 0)
    assert done.recomputed_tokens == PROMPT - 6 * BLOCK
    counters = window.counters
    assert counters["deliver_blocks_failed_beyond_hint"] == 4  # all past the hint
    assert not any(name.startswith("hinted_failed") for name in counters)
    assert counters["bytes_checked"] == 6 and not counters["bytes_mismatch"]


def test_wrong_bytes_are_counted():
    engine, kvcr, window = _engine()
    kvcr.remote_ok = set(KEYS[:3])
    engine.submit(_request(3, hint_source=1, hint_blocks=6))
    original = kvcr.host.stamp
    kvcr.host.stamp = lambda slot, bid: original(slot, bid + 1)  # corrupt
    engine.step()
    kvcr.host.stamp = original
    done, _ = _run(engine, kvcr)
    assert (done.remote_blocks, done.failed_blocks) == (3, 7)
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


def test_time_outside_kvcr_and_the_harness_is_split_into_forward_and_idle():
    engine, _, window = _engine()
    engine.step()  # nothing to run
    assert "idle_seconds" in window.counters
    assert "forward_seconds" not in window.counters
    # "KVCR per step" is over steps that ran something.
    assert window.hists["step_kvcr"].n == 0
    engine.submit(_request(1))
    engine.step()
    assert "forward_seconds" in window.counters
    assert window.hists["step_kvcr"].n == 1


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


# Trace replay: a prompt is the branch-free runs on its path through the
# session's prefix tree, and the answer's block opens the run the next
# request continues on.
PATH = [(900, 0, 2), (901, 0, 1), (902, 0, 3), (903, 0, 1), (904, 0, 2)]
PATH_KEYS = [id_to_key(bid) for bid in segment_ids(PATH)]


def test_a_replayed_request_spans_many_streams_and_its_answer_opens_another():
    engine, kvcr, _ = _engine()
    kvcr.remote_ok = set(PATH_KEYS[:4])
    engine.submit(
        Request(
            rid=1,
            session=0,
            segments=PATH,
            tail=(905, 0, 1),
            prompt_tokens=9 * BLOCK + 10,
            output_tokens=3,
            hint_source=1,
            hint_blocks=4,
        )
    )
    # The hint names the first four blocks of the path, across three streams.
    (hint,) = kvcr.of("submit_hint")
    assert hint[2]["block_hashes"] == [
        block_id(900, 0),
        block_id(900, 1),
        block_id(901, 0),
        block_id(902, 0),
    ]
    done, _ = _run(engine, kvcr)
    (deliver,) = kvcr.of("deliver")
    assert list(deliver[1]) == PATH_KEYS  # path order, root first
    (deposit,) = kvcr.of("deposit")
    assert list(deposit[1]) == PATH_KEYS[4:]
    assert (done.prompt_blocks, done.remote_blocks) == (9, 4)

    # One run per stream, in both tiers; the answer's stream on the device.
    deltas = engine.take_inventory()
    runs = [(stream, 0, end, False) for stream, _, end in PATH]
    assert sorted(d[1:] for d in deltas if d[0] == TIER_PRIMARY) == runs
    assert sorted(d[1:] for d in deltas if d[0] == TIER_DEVICE) == [
        *runs,
        (905, 0, 1, False),
    ]

    # The next turn names the answer's stream as prompt and adds no block.
    kvcr.calls.clear()
    engine.submit(
        Request(
            rid=2,
            session=0,
            segments=[*PATH, (905, 0, 3)],
            tail=(905, 3, 3),
            prompt_tokens=12 * BLOCK + 5,
            output_tokens=2,
        )
    )
    done, _ = _run(engine, kvcr)
    assert done.gpu_blocks == 10  # the whole old path and the answer's block
    assert [key_to_id(q[1][0]) for q in kvcr.of("query")] == [block_id(905, 1)]
    assert (TIER_DEVICE, 905, 1, 3, False) in engine.take_inventory()


# ---- why a block asked of a peer failed ---------------------------------------


def _hinted_failures(window):
    return {
        name.partition("|")[2]: count
        for name, count in window.counters.items()
        if name.startswith("hinted_failed|")
    }


def test_hinted_blocks_a_peer_did_not_deliver_are_counted_by_kvcr_status():
    engine, kvcr, window = _engine()
    kvcr.remote_ok = set(KEYS[:3])
    engine.submit(_request(1, hint_source=1, hint_blocks=6))
    done, _ = _run(engine, kvcr)
    # Seven blocks failed: three the hint named and four beyond it.
    counters = window.counters
    assert counters["deliver_blocks_failed"] == 7
    assert counters["deliver_blocks_failed_beyond_hint"] == 4
    assert _hinted_failures(window) == {"unavailable": 3}
    assert counters["hinted_failed_reason|not_resident"] == 3
    assert (done.failed_blocks, done.timed_out_blocks) == (7, 0)


def test_a_timeout_is_what_kvcr_calls_one():
    engine, kvcr, window = _engine()
    kvcr.remote_ok = set(KEYS[:3])
    kvcr.failure = OpEntryResult(OpEntryStatus.TIMEOUT, reason="deadline")
    engine.submit(_request(1, hint_source=1, hint_blocks=6))
    done, _ = _run(engine, kvcr)
    assert _hinted_failures(window) == {"timeout": 3}
    # The request's own count covers the blocks beyond the hint too.
    assert (done.failed_blocks, done.timed_out_blocks) == (7, 7)


def test_a_result_without_a_reason_is_left_unclassified():
    engine, kvcr, window = _engine()
    kvcr.remote_ok = set(KEYS[:3])
    # KVCR trees before per-block reasons: every failure is FAILED.
    kvcr.failure = SimpleNamespace(success=False, status=OpEntryStatus.FAILED)
    engine.submit(_request(1, hint_source=1, hint_blocks=6))
    done, _ = _run(engine, kvcr)
    assert _hinted_failures(window) == {"unclassified": 3}
    assert done.timed_out_blocks == 0


def test_a_failed_local_block_is_not_blamed_on_a_peer():
    engine, kvcr, window = _engine()
    asked = set()

    def query(keys, request_id=None):
        """Every block is in this worker's own pool, the first time it is asked."""
        kvcr.calls.append(("query", tuple(keys), request_id))
        fresh = [key not in asked for key in keys]
        asked.update(keys)
        hit = (QueryStatus.HIT, CacheTier.LOCAL_G2)
        return [hit if new else (QueryStatus.MISS, None) for new in fresh]

    kvcr.query = query
    kvcr.failure = OpEntryResult(OpEntryStatus.TIMEOUT, reason="deadline")
    engine.submit(_request(1))
    done, _ = _run(engine, kvcr)
    counters = window.counters
    assert counters["deliver_blocks_failed"] == 10
    assert not _hinted_failures(window)
    assert "deliver_blocks_failed_beyond_hint" not in counters
    assert (done.failed_blocks, done.timed_out_blocks) == (10, 0)


# ---- the device and the step budget -------------------------------------------


def _long(rid, stream, blocks=40):
    """A request of ``blocks`` device blocks on a stream of its own."""
    return Request(
        rid=rid,
        session=rid,
        segments=[(stream, 0, blocks - 1)],
        tail=(stream, blocks - 1, blocks),
        prompt_tokens=(blocks - 1) * BLOCK + 5,
        output_tokens=3,
    )


def test_a_request_waits_until_the_device_has_room_for_all_of_it():
    engine, kvcr, window = _engine()  # a 64-block device
    engine.submit(_long(1, 100))
    engine.submit(_long(2, 200))
    engine.submit(_request(3))  # would fit beside request 1, but came after 2
    finished = [done.rid for done in engine.step()]
    assert [r.rid for r in engine._running] == [1]
    assert window.counters["device_alloc_failures"] == 1
    # Nothing behind the request that did not fit was looked up.
    assert {query[2] for query in kvcr.of("query")} == {"r1", "r2"}
    for _ in range(20):
        finished += [done.rid for done in engine.step()]
    assert finished == [1, 2, 3]
    gpu = engine._gpu
    assert gpu.live == 0 and not gpu._pins and len(gpu) <= 64
    # Making room evicted request 1's blocks, tail first.
    assert (TIER_DEVICE, 100, 24, 40, True) in engine.take_inventory()


def test_a_running_requests_cached_prefix_is_not_evicted_to_admit_another():
    engine, kvcr, _ = _engine()
    engine.submit(_long(1, 100, blocks=30))
    _run(engine, kvcr)
    grown = Request(
        rid=2,
        session=1,
        segments=[(100, 0, 30), (101, 0, 9)],
        tail=(101, 9, 10),
        prompt_tokens=39 * BLOCK + 5,
        output_tokens=200,
    )
    engine.submit(grown)  # reuses the 30 cached blocks and needs 10 more
    engine.step()
    engine.submit(_long(3, 300, blocks=30))  # 30 + 10 + 30 > 64
    engine.step()
    assert [r.rid for r in engine._running] == [2]
    assert engine._gpu.prefix([(100, 0, 30)]) == 30
    engine.abort_requests()
    assert engine._gpu.live == 0 and not engine._gpu._pins


def test_a_request_the_device_could_never_hold_is_refused():
    engine, _, _ = _engine()
    with pytest.raises(ValueError, match="needs 65 device blocks.*tiers.gpu_blocks"):
        engine.submit(_long(1, 100, blocks=65))


def test_a_running_request_reserves_device_blocks_for_all_its_tokens():
    engine, kvcr, _ = _engine()
    # No cacheable block at all: 40 prompt tokens and a 200-token answer.
    engine.submit(
        Request(
            rid=1,
            session=0,
            segments=[],
            tail=(70, 0, 0),
            prompt_tokens=40,
            output_tokens=200,
        )
    )
    engine.step()
    assert engine._gpu.live == 4  # 240 tokens of 64-token blocks
    _run(engine, kvcr, steps=300)
    assert engine._gpu.live == 0 and len(engine._gpu) == 0


def test_a_request_is_the_most_recently_used_when_it_finishes():
    engine, kvcr, _ = _engine()
    # Submitted first, finishes last: 20 blocks and a 40-token answer.
    engine.submit(
        Request(
            rid=1,
            session=1,
            segments=[(100, 0, 19)],
            tail=(100, 19, 20),
            prompt_tokens=19 * BLOCK + 5,
            output_tokens=40,
        )
    )
    engine.submit(_long(2, 200, blocks=20))
    finished = []
    while len(finished) < 2:
        finished += [done.rid for done in engine.step()]
    assert finished == [2, 1]
    # A third request needs room in both tiers. What goes is the request that
    # finished first, not the one that arrived first.
    engine.submit(_long(3, 300, blocks=40))
    engine.step()
    assert (engine._gpu.held(100), engine._gpu.held(200)) == (20, 4)
    kept = engine._primary.readable
    assert len(kept([_key(100, i) for i in range(19)])) == 19
    assert len(kept([_key(200, i) for i in range(19)])) == 3


def test_load_time_is_the_promotion_and_not_the_wait_for_the_device():
    engine, kvcr, window = _engine(step_ms=1)
    kvcr.remote_ok = set(KEYS)
    # 55 of the 64 device blocks are busy for 30 steps.
    engine.submit(
        Request(
            rid=1,
            session=1,
            segments=[(100, 0, 54)],
            tail=(100, 54, 55),
            prompt_tokens=54 * BLOCK + 5,
            output_tokens=30,
        )
    )
    engine.step()
    started = time.perf_counter()
    engine.submit(_request(2, hint_source=1, hint_blocks=10))
    dones = []
    while len(dones) < 2:
        dones += engine.step()
    elapsed_ms = (time.perf_counter() - started) * 1e3
    assert window.counters["device_alloc_failures"] > 20
    hinted = dones[1]
    assert hinted.rid == 2 and hinted.remote_blocks == 10
    # From the first lookup to the one that found the delivered blocks: the
    # promotion is in it, the steps spent waiting for device room are not.
    assert window.hists["deliver_remote"].total * 1e3 <= hinted.load_ms
    assert hinted.load_ms < elapsed_ms / 2


def test_a_block_kvcr_is_still_fetching_defers_the_request():
    engine, kvcr, _ = _engine()
    fetching = [KEYS[0]]  # KVCR is bringing the first block in, once

    def query(keys, request_id=None):
        kvcr.calls.append(("query", tuple(keys), request_id))
        if keys[0] in fetching:
            fetching.remove(keys[0])
            return [(QueryStatus.FETCHING, CacheTier.LOCAL_G2)]
        return [(QueryStatus.MISS, None)]

    kvcr.query = query
    engine.submit(_request(1))
    engine.step()
    # Not a miss: the scan goes on, and nothing is computed until KVCR has an
    # answer for the block.
    assert [call[1] for call in kvcr.of("query")] == [(KEYS[0],), (KEYS[1],)]
    assert not engine._running and not kvcr.of("deposit")
    engine.step()
    assert [r.rid for r in engine._running] == [1]
    assert kvcr.of("query")[-1][1] == (KEYS[0],)


def test_decode_tokens_come_out_of_the_step_budget():
    engine, _, _ = _engine(max_batched_tokens=100)
    decoding = Request(
        rid=1,
        session=0,
        segments=[(70, 0, 1)],
        tail=(70, 1, 1),
        prompt_tokens=BLOCK + 5,
        output_tokens=50,
    )
    engine.submit(decoding)
    engine.step()  # its whole prompt: it decodes from here on
    prefilling = Request(
        rid=2,
        session=1,
        segments=[(71, 0, 4)],
        tail=(71, 4, 4),
        prompt_tokens=4 * BLOCK + 5,
        output_tokens=2,
    )
    engine.submit(prefilling)
    engine.step()
    first, second = engine._running
    assert (first.sched, second.sched) == (1, 99)


def test_a_store_the_cpu_tier_has_no_room_for_is_tried_again():
    engine, kvcr, _ = _engine()
    primary = engine._primary
    free, primary._free = primary._free, []
    engine.submit(_request(1))
    engine.step()
    assert primary.allocation_failures == 1 and not kvcr.of("deposit")
    primary._free = free
    engine.step()
    (deposit,) = kvcr.of("deposit")
    assert list(deposit[1]) == KEYS


def test_the_kvcr_pool_is_reported_with_its_gaps_closed():
    engine, _, _ = _engine()
    router = Inventory(1)

    def kvcr_says(removed, *indexes):
        keys = [_key(5, index) for index in indexes]
        engine.record_inventory(
            SimpleNamespace(tier=CacheTier.LOCAL_G2, removed=removed, keys=keys)
        )
        router.apply(0, engine.take_inventory())
        return router.prefix(0, [(5, 0, 10)], (TIER_G2,))

    assert kvcr_says(False, *range(10)) == 10
    assert kvcr_says(True, 3) == 3  # one block from the middle
    assert kvcr_says(False, 3) == 10  # it is back: the rest never left
