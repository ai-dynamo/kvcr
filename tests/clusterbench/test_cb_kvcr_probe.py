# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The probe against a scripted KVCR: what it counts at KVCR's API."""

import ctypes
import struct
from types import SimpleNamespace

from clusterbench.ids import block_hash
from clusterbench.kvcr_probe import KvcrProbe
from clusterbench.metrics import Window

from kvcr import KVCRBindings
from kvcr.types import CacheTier, MemoryRef, OpEntryResult, OpEntryStatus, QueryStatus

BLOCK = 64
OK = OpEntryResult(OpEntryStatus.SUCCESS)
TIMEOUT = OpEntryResult(OpEntryStatus.TIMEOUT, reason="deadline")
GONE = OpEntryResult(OpEntryStatus.UNAVAILABLE, reason="not_resident")
PEER = "tcp://other-node:1"


def key(bid):
    return block_hash(bid) + bytes(4)


def ref(chunk):
    return [MemoryRef(end_point_name="here", element_index=chunk)]


class ScriptedKvcr:
    """Hands out operation handles; the test says what each poll returns."""

    def __init__(self, config, bindings, backends):
        self.bindings = bindings
        self.remote = set()  # keys a query places on the peer
        self.polls = []
        self.stats = None
        self.handle = 0

    def submit_hint(self, hints, request_id=None):
        pass

    def query(self, keys, request_id=None):
        tier = lambda k: CacheTier.REMOTE_G2 if k in self.remote else CacheTier.LOCAL_G2  # noqa: E731
        return [(QueryStatus.FETCHABLE, tier(k)) for k in keys]

    def deliver(self, blocks, request_id=None):
        self.handle += 1
        return self.handle

    deposit = deliver

    def poll_completed(self):
        return self.polls.pop(0) if self.polls else []

    def align_sequence(self, keys, use_current_time=False):
        pass

    def discard_hint(self, request_id):
        pass

    def get_stats(self):
        return self.stats


def probe_with(bindings=KVCRBindings(), chunks=8, **kwargs):
    window = Window()
    memory = ctypes.create_string_buffer(BLOCK * chunks)
    region = SimpleNamespace(addr=ctypes.addressof(memory), size=BLOCK, count=chunks)
    backends = SimpleNamespace(framework_regions=[region])
    probe = KvcrProbe(ScriptedKvcr, window, cross_node=frozenset({PEER}), **kwargs)
    assert probe(None, bindings, backends) is probe
    return probe, probe._kvcr, window, memory


def hint(endpoint, ids):
    payload = {"source_control_endpoint": endpoint, "block_hashes": ids}
    return SimpleNamespace(actions=[SimpleNamespace(payload=payload)])


def test_an_operation_that_completes_over_two_polls_is_counted_once():
    probe, kvcr, window, _ = probe_with(verify="none")
    blocks = {key(i): ref(i) for i in range(3)}
    handle = probe.deliver(blocks, request_id="r1")
    kvcr.polls = [[(handle, {key(0): OK, key(1): OK})], [(handle, {key(2): OK})]]
    probe.poll_completed()
    assert probe.pending() == 1 and not window.counters
    probe.poll_completed()
    assert probe.pending() == 0
    assert window.counters["deliver_blocks_local"] == 3
    assert window.hists["deliver_local"].n == 1


def test_delivered_blocks_are_counted_by_tier_hint_and_kvcr_status():
    probe, kvcr, window, _ = probe_with(verify="none")
    # The router named blocks 0-3 on a peer of another node. KVCR places 0-5
    # there (an opportunistic query goes past the hint) and 6-7 in its own pool.
    kvcr.remote = {key(i) for i in range(6)}
    probe.submit_hint(hint(PEER, [0, 1, 2, 3]), request_id="r1")
    for i in range(8):
        probe.query((key(i),), "r1")
    handle = probe.deliver({key(i): ref(i) for i in range(8)}, request_id="r1")
    results = [OK, OK, TIMEOUT, GONE, GONE, GONE, OK, OK]
    kvcr.polls = [[(handle, {key(i): result for i, result in enumerate(results)})]]
    probe.poll_completed()
    counters = dict(window.counters)
    assert counters == {
        "deliver_blocks_remote": 2,
        "deliver_blocks_remote_cross_node": 2,
        "deliver_blocks_local": 2,
        "deliver_blocks_failed": 4,
        "deliver_ops_partial": 1,
        "hinted_failed|timeout": 1,
        "hinted_failed|unavailable": 1,
        "hinted_failed_reason|deadline": 1,
        "hinted_failed_reason|not_resident": 1,
        "deliver_blocks_failed_beyond_hint": 2,
    }
    assert window.hists["deliver_remote"].n == 1
    # From a peer, from the own pool, failed, of those timed out.
    assert probe.outcome("r1")[:4] == (2, 2, 4, 1)
    assert probe.outcome("r1") == (0, 0, 0, 0, 0.0)  # taken once
    # A hint at a peer on this worker's node is not cross-node.
    probe.submit_hint(hint("tcp://same-node:2", [0]), request_id="r2")
    probe.query((key(0),), "r2")
    handle = probe.deliver({key(0): ref(0)}, request_id="r2")
    kvcr.polls = [[(handle, {key(0): OK})]]
    probe.poll_completed()
    assert window.counters["deliver_blocks_remote"] == 3
    assert window.counters["deliver_blocks_remote_cross_node"] == 2


def test_bytes_are_stamped_at_deposit_and_checked_when_a_deliver_completes():
    probe, kvcr, window, memory = probe_with(verify="full")
    handle = probe.deposit({key(41): ref(2), key(42): ref(5)})
    stamps = [struct.unpack_from(">Q", memory, c * BLOCK)[0] for c in (2, 5)]
    assert stamps == [41, 42]  # before KVCR could read the chunks
    kvcr.polls = [[(handle, {key(41): OK, key(42): GONE})]]
    probe.poll_completed()
    assert window.counters["deposit_blocks_ok"] == 1
    assert window.counters["deposit_blocks_failed"] == 1
    # A deliver into chunk 2 that brought block 41's bytes, one into chunk 5
    # that should have brought block 7's.
    handle = probe.deliver({key(41): ref(2), key(7): ref(5)}, request_id="r1")
    kvcr.polls = [[(handle, {key(41): OK, key(7): OK})]]
    probe.poll_completed()
    assert window.counters["bytes_checked"] == 2
    assert window.counters["bytes_mismatch"] == 1


def test_sampling_checks_the_first_block_of_a_deliver():
    probe, kvcr, window, _ = probe_with(verify="sample")
    handle = probe.deliver({key(i): ref(i) for i in range(4)}, request_id="r1")
    kvcr.polls = [[(handle, {key(i): OK for i in range(4)})]]
    probe.poll_completed()
    assert window.counters["bytes_checked"] == 1


def test_source_pins_are_counted_and_open_until_released():
    released = []
    results = [(1, None), (2, ("pin-2", {key(0): ref(0), key(1): None}))]
    bindings = KVCRBindings(
        request_pin=lambda keys: 7,
        poll_pin_results=lambda: results,
        release_pin=lambda handle: released.append(handle) or True,
    )
    probe, kvcr, window, _ = probe_with(bindings)
    watched = kvcr.bindings
    assert watched.request_pin((key(0), key(1))) == 7
    assert watched.poll_pin_results() == results
    assert (
        window.counters["pin_requests"],
        window.counters["pin_misses"],
        window.counters["pin_keys"],
    ) == (1, 1, 1)
    assert probe.pending() == 1  # the pin a peer still holds
    assert watched.release_pin("pin-2") and released == ["pin-2"]
    assert probe.pending() == 0


def test_every_call_is_timed_and_kvcr_events_reach_the_window():
    seen = []
    forwarded = []
    bindings = KVCRBindings(on_resilience_event=forwarded.append)
    probe, kvcr, window, _ = probe_with(bindings, on_stats=seen.append)
    probe.submit_hint(hint(PEER, [1, 2, 3]), request_id="r1")
    for i in range(5):
        probe.query((key(i),), "r1")
    assert "query" not in window.calls  # totalled at the next poll
    probe.poll_completed()
    probe.align_sequence([key(0), key(1)], use_current_time=True)
    probe.discard_hint("r1")
    kvcr.stats = object()
    assert probe.get_stats() is kvcr.stats and seen == [kvcr.stats]
    assert dict(window.calls) == {
        "submit_hint": 1,
        "poll": 1,
        "query": 5,
        "align_sequence": 1,
        "discard_hint": 1,
        "get_stats": 1,
    }
    assert window.items["submit_hint"] == 3 and window.items["align_sequence"] == 2
    error = RuntimeError("peer went away")
    kvcr.bindings.on_resilience_event(error)
    assert window.counters["resilience_events"] == 1
    assert probe.resilience == ["peer went away"] and forwarded == [error]
