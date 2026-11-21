# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The driver around vLLM's real scheduler, with a recording stand-in for KVCR.

vLLM decides what to call and when; these pin what the driver adds: the block
ids it hands vLLM, the job reports it fakes, and what it reads back out.
"""

import ctypes
import itertools
import mmap
import os
import socket
import struct

import pytest

pytest.importorskip(
    "vllm", reason="needs the pinned vLLM: see benchmarks/clusterbench/README.md"
)

from clusterbench.config import EngineConfig, RunConfig, TierConfig  # noqa: E402
from clusterbench.engine_vllm import VllmEngine  # noqa: E402
from clusterbench.ids import (  # noqa: E402
    INDEX_BITS,
    PRIVATE_BASE,
    block_hash,
    key_to_id,
    segment_ids,
)
from clusterbench.metrics import Window  # noqa: E402
from clusterbench.protocol import (  # noqa: E402
    TIER_DEVICE,
    TIER_G2,
    TIER_PRIMARY,
    Request,
)

from kvcr.types import (  # noqa: E402
    CacheTier,
    InventoryEvent,
    OpEntryResult,
    OpEntryStatus,
    QueryStatus,
)

B = 16
SEGMENTS = [(1, 0, 4), (70, 0, 6)]
TAIL = (70, 6, 7)
IDS = segment_ids(SEGMENTS)
PROMPT = 10 * B + 5
OK = OpEntryResult(OpEntryStatus.SUCCESS)
GONE = OpEntryResult(OpEntryStatus.UNAVAILABLE, reason="not_resident")
_engine_ids = itertools.count()


def key(bid):
    return block_hash(bid) + bytes(4)


class RecordingKvcr:
    """KVCR's API as vLLM's tier calls it. Operations complete at the next poll."""

    def __init__(self, config, bindings, backends, guard=None):
        self.config = config
        self.bindings = bindings
        region = backends.framework_regions[0]
        size = region.size * region.count
        self.memory = (ctypes.c_char * size).from_address(region.addr)
        self.block = region.size
        self.peer = set()  # keys the hinted peer holds
        self.pool = set()  # keys deposited here
        self.calls = []
        self.completed = []
        self.handle = 0

    def of(self, name):
        return [call for call in self.calls if call[0] == name]

    def submit_hint(self, hints, request_id=None):
        self.calls.append(("submit_hint", request_id, hints.actions[0].payload))

    def query(self, keys, request_id=None):
        self.calls.append(("query", tuple(keys), request_id))
        return [
            (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
            if k in self.peer
            else (QueryStatus.HIT, CacheTier.LOCAL_G2)
            if k in self.pool
            else (QueryStatus.MISS, None)
            for k in keys
        ]

    def deliver(self, blocks, request_id=None):
        self.calls.append(("deliver", list(blocks), request_id))
        entries = {}
        for k, refs in blocks.items():
            if k in self.peer or k in self.pool:  # the block's bytes arrive
                offset = refs[0].element_index * self.block
                struct.pack_into(">Q", self.memory, offset, key_to_id(k))
                entries[k] = OK
            else:
                entries[k] = GONE
        return self._complete(entries)

    def deposit(self, blocks):
        self.calls.append(("deposit", list(blocks)))
        self.pool.update(blocks)
        event = InventoryEvent(
            keys=tuple(blocks), tier=CacheTier.LOCAL_G2, removed=False
        )
        self.bindings.inventory_sink(event)
        return self._complete(dict.fromkeys(blocks, OK))

    def _complete(self, entries):
        self.handle += 1
        self.completed.append((self.handle, entries))
        return self.handle

    def poll_completed(self):
        completed, self.completed = self.completed, []
        return completed

    def align_sequence(self, keys, use_current_time=False):
        self.calls.append(("align_sequence", len(keys), use_current_time))

    def discard_hint(self, request_id):
        self.calls.append(("discard_hint", request_id))

    def get_stats(self):
        return None

    def close(self):
        self.bindings.framework_control.close()


@pytest.fixture
def make():
    engines = []

    def build(**tiers):
        tiers = {"gpu_blocks": 64, "primary_blocks": 64, "g2_blocks": 64, **tiers}
        config = RunConfig(
            tiers=TierConfig(block_bytes=mmap.PAGESIZE, tokens_per_block=B, **tiers),
            engine=EngineConfig(
                step_ms=0, prefill_us_per_token=0, idle_sleep_ms=0, verify="full"
            ),
        )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        window = Window()
        engine = VllmEngine(
            config,
            window,
            0,
            [("n0", "tcp://a:1"), ("n1", "tcp://b:2")],
            bind_host="127.0.0.1",
            control_port=port,
            engine_id=f"cb-test-{os.getpid()}-{next(_engine_ids)}",
            kvcr_cls=RecordingKvcr,
        )
        engines.append(engine)
        return engine, engine.probe._kvcr, window

    yield build
    for engine in engines:
        engine.close()


def _request(rid, **kwargs):
    fields = dict(segments=SEGMENTS, tail=TAIL, prompt_tokens=PROMPT, output_tokens=B)
    return Request(rid=rid, session=0, **{**fields, **kwargs})


def _run(engine, steps=200):
    """Step until a request finishes."""
    for _ in range(steps):
        dones = engine.step()
        if dones:
            return dones[0]
    raise AssertionError("the request did not finish")


def _settle(engine, steps=5):
    for _ in range(steps):
        engine.step()


def test_a_cold_request_is_computed_stored_and_deposited_under_its_block_ids(make):
    engine, kvcr, window = make()
    engine.submit(_request(1))
    done = _run(engine)
    _settle(engine)  # the store is reported the step after, then deposited
    assert not kvcr.of("submit_hint") and not kvcr.of("deliver")
    (deposit,) = kvcr.of("deposit")
    assert [key_to_id(k) for k in deposit[1]] == IDS and len(deposit[1][0]) == 36
    assert (done.rid, done.remote_blocks, done.failed_blocks) == (1, 0, 0)
    counters = window.counters
    assert (counters["requests"], counters["prompt_blocks"]) == (1, 10)
    assert counters["prefill_tokens"] == PROMPT and counters["gpu_hit_blocks"] == 0
    # The router hears of every tier from vLLM's own events: the device holds
    # the answer's block too, the CPU tier and the KVCR pool the prompt's.
    assert sorted(engine.take_inventory()) == [
        (TIER_DEVICE, 1, 0, 4, False),
        (TIER_DEVICE, 70, 0, 7, False),
        (TIER_PRIMARY, 1, 0, 4, False),
        (TIER_PRIMARY, 70, 0, 6, False),
        (TIER_G2, 1, 0, 4, False),
        (TIER_G2, 70, 0, 6, False),
    ]
    assert engine.fill_level() == 10 / 64
    assert not engine.busy()


def test_a_store_is_reported_the_step_after_the_one_that_issued_it(make):
    engine, kvcr, _ = make()
    engine.submit(_request(1))
    engine.step()  # the prompt is computed and its store job issued
    assert not kvcr.of("deposit")
    engine.step()  # vLLM's worker defers a store to the next step
    assert kvcr.of("deposit")


def test_a_hinted_request_pulls_the_named_blocks_and_computes_the_rest(make):
    engine, kvcr, window = make()
    kvcr.peer = {key(bid) for bid in IDS[:6]}
    engine.submit(_request(2, hint_source=1, hint_blocks=6))
    done = _run(engine)
    (hint,) = kvcr.of("submit_hint")
    assert hint[1:] == (
        "r2",
        {"source_control_endpoint": "tcp://b:2", "block_hashes": IDS[:6]},
    )
    (deliver,) = kvcr.of("deliver")
    assert [key_to_id(k) for k in deliver[1]] == IDS[:6] and deliver[2] == "r2"
    assert (done.remote_blocks, done.failed_blocks, done.timed_out_blocks) == (6, 0, 0)
    counters = window.counters
    assert counters["prefill_tokens"] == PROMPT - 6 * B
    # The hinted worker is on another node; every delivered block had its bytes.
    assert counters["deliver_blocks_remote_cross_node"] == 6
    assert counters["bytes_checked"] == 6 and not counters["bytes_mismatch"]
    assert counters["primary_hit_blocks"] == 0


def test_the_next_turn_finds_the_answer_on_the_device(make):
    engine, _, window = make()
    engine.submit(_request(1))
    _run(engine)
    grown = _request(
        2, segments=[(1, 0, 4), (70, 0, 9)], tail=(70, 9, 10), prompt_tokens=13 * B + 5
    )
    engine.submit(grown)
    _run(engine)
    # The first turn's ten prompt blocks and the block its answer filled.
    assert window.counters["gpu_hit_blocks"] == 11


def test_blocks_no_segment_names_get_ids_of_their_own(make):
    engine, kvcr, _ = make()
    # Twelve full blocks of prompt, of which the segments name ten.
    engine.submit(_request(5, prompt_tokens=12 * B + 5))
    _run(engine)
    _settle(engine)
    (deposit,) = kvcr.of("deposit")
    stored = [key_to_id(k) for k in deposit[1]]
    assert stored[:10] == IDS and min(stored[10:]) >= PRIVATE_BASE << INDEX_BITS
    engine.take_inventory()
    # A prompt no segment names at all is stored too, and the router, which
    # could never route a request to it, is not told.
    unnamed = _request(6, segments=[], tail=(70, 0, 0), prompt_tokens=2 * B + 5)
    engine.submit(unnamed)
    _run(engine)
    _settle(engine)
    assert len(kvcr.of("deposit")[-1][1]) == 2
    assert [d for d in engine.take_inventory() if not d[4]] == []


def test_aborted_requests_leave_only_kvcr_operations_to_drain(make):
    engine, kvcr, _ = make()
    kvcr.peer = {key(bid) for bid in IDS}
    engine.submit(_request(7, hint_source=1, hint_blocks=10))
    engine.step()  # the deliver is issued; the request waits on it
    assert kvcr.of("deliver") and engine.busy()
    engine.abort_requests()
    _settle(engine)
    assert ("discard_hint", "r7") in kvcr.calls and not engine.busy()


def test_a_request_the_device_could_never_hold_is_refused(make):
    engine, _, _ = make()
    with pytest.raises(ValueError, match="needs 65 device blocks.*tiers.gpu_blocks"):
        engine.submit(_request(1, prompt_tokens=64 * B, output_tokens=1))


def test_the_default_device_size_is_within_what_the_model_config_allows(make):
    # vLLM checks the token limit, here the device's size, against the model's.
    engine, _, _ = make(gpu_blocks=RunConfig().tiers.gpu_blocks)
    engine.submit(_request(1))
    assert _run(engine).rid == 1


def test_block_bytes_must_fit_the_page_size_before_anything_is_built():
    tiers = TierConfig(block_bytes=mmap.PAGESIZE + 4 * B, tokens_per_block=B)
    with pytest.raises(ValueError, match="multiple of this host's page size"):
        VllmEngine(
            RunConfig(tiers=tiers),
            Window(),
            0,
            [("n0", "tcp://a:1")],
            bind_host="127.0.0.1",
            control_port=1,
            engine_id="cb-test-unused",
        )
