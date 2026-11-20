# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Trace replay: the compile step and the trace-backed workload."""

import itertools
import json
import random

import msgspec
import pytest
from clusterbench.config import (
    PhaseConfig,
    RunConfig,
    TierConfig,
    WorkloadConfig,
    load_config,
)
from clusterbench.coordinator import Coordinator
from clusterbench.ids import TRACE_BASE
from clusterbench.primary_tier import StreamCache
from clusterbench.trace import (
    _ROW,
    Compiled,
    Header,
    Stats,
    TraceError,
    TraceWorkload,
    _Trace,
    compile_corpus,
    compile_trace,
    load,
)
from clusterbench.workload import Workload, make_workload

BLOCK = 64


def req(t, ids, out=0, api=1.0, tokens=None):
    return {
        "t": t,
        "type": "s",
        "in": BLOCK * len(ids) if tokens is None else tokens,
        "out": out,
        "hash_ids": list(ids),
        "api_time": api,
    }


def group(t, requests):
    return {"type": "subagent", "t": t, "status": "completed", "requests": requests}


def trace(requests, tid="t0", block=BLOCK):
    return {
        "id": tid,
        "block_size": block,
        "hash_id_scope": "local",
        "requests": requests,
    }


def common(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class View:
    """A compiled trace, with each turn expanded to its blocks."""

    def __init__(self, compiled):
        self.compiled = compiled
        self.trace = _Trace(compiled)
        rows = compiled.rows
        self.turns = []  # (lane, prompt blocks, tail blocks, segments, tail)
        for offset in range(0, len(rows), _ROW):
            lane, leaf, used, tail_edge, start, end = rows[offset : offset + 6]
            segments = []
            if leaf >= 0:
                segments = [(edge, 0, length) for edge, length in self.trace.path(leaf)]
                segments.append((leaf, 0, used))
            prompt = [(edge, i) for edge, a, b in segments for i in range(a, b)]
            tail = [(tail_edge, i) for i in range(start, end)]
            self.turns.append((lane, prompt, tail, segments, (tail_edge, start, end)))

    def lane(self, lane):
        return [turn for turn in self.turns if turn[0] == lane]


def compiled_view(requests, stats=None):
    return View(compile_trace(trace(requests), stats))


# ---- mapping --------------------------------------------------------------------


def test_a_linear_conversation_is_one_stream_and_answers_open_the_next_prompt():
    view = compiled_view(
        [
            req(0, range(10), out=2 * BLOCK),
            req(5, range(15), out=BLOCK),
            req(9, range(20)),
        ]
    )
    assert view.compiled.edges == [-1, 20]
    assert [turn[3] for turn in view.turns] == [
        [(0, 0, 10)],
        [(0, 0, 15)],
        [(0, 0, 20)],
    ]
    assert [turn[4] for turn in view.turns] == [(0, 10, 12), (0, 15, 16), (0, 20, 20)]


def test_a_trailing_partial_block_id_is_not_a_block():
    # 212 tokens are three whole blocks; the fourth id names a partial block
    # whose content, and so its id, changes on the next turn.
    view = compiled_view(
        [req(0, [1, 2, 3, 4], tokens=212), req(5, [1, 2, 3, 9, 10], tokens=5 * BLOCK)]
    )
    assert view.compiled.edges == [-1, 5]
    assert view.turns[0][3] == [(0, 0, 3)]


def test_a_prompt_under_one_block_has_no_segments(tmp_path):
    view = compiled_view([req(0, [], tokens=40, out=200), req(5, [1, 2])])
    assert view.turns[0][3] == [] and view.turns[0][2] == []
    workload = _workload(tmp_path, [trace([req(0, [], tokens=40), req(5, [1, 2])])])
    turn = workload.next_turn(workload.new_session().id)
    assert turn.segments == () and turn.tail[1] == turn.tail[2]
    assert turn.prompt_tokens == 40


def _random_requests(rng, count, alphabet):
    """Requests that extend, truncate and branch off earlier ones."""
    prompts, requests = [], []
    for n in range(count):
        if prompts and rng.random() < 0.85:
            base = rng.choice(prompts)
            ids = base[: rng.randint(0, len(base))]
        else:
            ids = []
        ids = ids + [rng.randrange(alphabet) for _ in range(rng.randint(0, 6))]
        prompts.append(ids)
        requests.append(req(float(n), ids, out=rng.randint(0, 4) * BLOCK))
    return prompts, requests


@pytest.mark.parametrize("seed", range(12))
def test_requests_share_exactly_the_blocks_their_hash_lists_share(seed):
    rng = random.Random(seed)
    # A small alphabet repeats ids under different prefixes.
    prompts, requests = _random_requests(rng, 40, alphabet=rng.choice((3, 8, 1000)))
    stats = Stats()
    view = compiled_view(requests, stats)
    lengths = view.trace.lengths
    for (_, blocks, _, segments, _), ids in zip(view.turns, prompts):
        assert len(blocks) == len(ids)
        # The shape the device caches rely on.
        assert all(start == 0 for _, start, _ in segments)
        assert all(end == lengths[edge] for edge, _, end in segments[:-1])
        assert all(0 < end <= lengths[edge] for edge, _, end in segments)
    for (a, ids_a), (b, ids_b) in itertools.combinations(zip(view.turns, prompts), 2):
        assert common(a[1], b[1]) == common(ids_a, ids_b)
    kept = sum(len(turn[2]) for turn in view.turns)
    assert stats.tail_blocks == kept
    assert stats.answer_blocks == (
        kept + stats.uncarried_blocks + stats.cut_blocks + stats.dropped_blocks
    )


@pytest.mark.parametrize("seed", range(12))
def test_a_tail_names_only_blocks_a_later_request_of_its_lane_brings(seed):
    rng = random.Random(100 + seed)
    prompts, requests = _random_requests(rng, 40, alphabet=rng.choice((3, 8, 1000)))
    view = compiled_view(requests)
    first_named = {}
    for index, (_, blocks, _, _, _) in enumerate(view.turns):
        for block in blocks:
            first_named.setdefault(block, index)
    taken = set()
    for index, (_, blocks, tail, _, _) in enumerate(view.turns):
        assert not taken & set(tail)  # one answer per block
        taken |= set(tail)
        if not tail or tail[0] not in first_named:
            # Past the end of a stream nothing extends: no prompt names it.
            assert not any(block in first_named for block in tail)
            continue
        later = first_named[tail[0]]
        assert later > index  # nobody had these blocks before the answer
        extended = view.turns[later][1]
        assert extended[: len(blocks)] == blocks
        assert extended[len(blocks) : len(blocks) + len(tail)] == tail


# ---- answers --------------------------------------------------------------------


def test_an_answer_stops_where_the_next_prompt_stops():
    stats = Stats()
    view = compiled_view(
        [req(0, range(4), out=5 * BLOCK), req(5, range(6)), req(9, range(9))], stats
    )
    assert view.turns[0][4] == (0, 4, 6)  # two of five blocks reached a prompt
    assert stats.uncarried_blocks == 3 and stats.tail_blocks == 2


def test_a_request_nobody_extends_puts_its_answer_past_the_end_of_its_stream():
    view = compiled_view([req(0, [1, 2, 3], out=2 * BLOCK), req(5, [7, 8], out=BLOCK)])
    assert view.turns[0][4] == (0, 3, 5) and view.turns[1][4] == (1, 2, 3)


def _conversation(rng, turns):
    """A session whose true answers are known: (requests, [(prompt, answer)])."""
    fresh = itertools.count(1)

    def new(low, high):
        return [next(fresh) for _ in range(rng.randint(low, high))]

    # The context as chunks, so a rewind never lands inside an answer.
    chunks = [new(2, 5)]
    history, requests, truth, prompts = [], [], [], set()
    for n in range(turns):
        prompt = [block for chunk in chunks for block in chunk]
        answer = new(1, 4)
        requests.append(req(float(n), prompt, out=BLOCK * len(answer)))
        truth.append((prompt, answer))
        prompts.add(tuple(prompt))
        history.append(list(chunks))
        kind = rng.random()
        if kind < 0.6:  # the conversation goes on: the answer, then new input
            chunks = [*chunks, answer, *([new(1, 3)] if rng.random() < 0.8 else [])]
        elif kind < 0.7:  # a retry: the same prompt again, the answer discarded
            pass
        elif kind < 0.85:  # compaction or rewind: part of an old context, new input
            old = rng.choice(history)
            kept = old[: rng.randint(0, len(old) - 1)]
            if tuple(b for chunk in kept for b in chunk) in prompts:
                kept = []  # extending an old prompt with input is not modelled
            chunks = [*kept, new(1, 3)]
        else:  # back to an earlier context as it was
            chunks = list(rng.choice(history))
    return requests, truth


def _replay_stream_cache(view):
    cache = StreamCache(1 << 30)
    hits = []
    for _, _, _, segments, tail in view.turns:
        hits.append(cache.prefix(segments))
        cache.insert([*segments, tail], cache.new_rank())
    return hits


@pytest.mark.parametrize("seed", range(20))
def test_replayed_caches_hit_what_the_true_answers_would_give(seed):
    requests, truth = _conversation(random.Random(seed), 40)
    view = compiled_view(requests)
    expected = []
    for index, (prompt, _) in enumerate(truth):
        held = [earlier + answer for earlier, answer in truth[:index]]
        expected.append(max((common(prompt, path) for path in held), default=0))
    assert _replay_stream_cache(view) == expected
    # A request that is extended got its whole answer, in order.
    block_of = {}
    for (_, blocks, _, _, _), (prompt, _) in zip(view.turns, truth):
        for depth, block in enumerate(blocks):
            block_of[tuple(prompt[: depth + 1])] = block
    for (_, _, tail, _, _), (prompt, answer) in zip(view.turns, truth):
        path = prompt + answer
        if tuple(path) in block_of:
            assert tail == [
                block_of[tuple(path[: depth + 1])]
                for depth in range(len(prompt), len(path))
            ]


# ---- lanes ----------------------------------------------------------------------

AGENTS = [
    req(0, [1, 2], api=1.0),
    group(1.5, [req(1.5, [10, 11], api=1.0), req(3, [10, 11, 12], api=1.0)]),
    group(2, [req(2, [20], api=0.5)]),
    req(6, [1, 2, 3], api=1.0),
    req(8, [1, 2, 3, 4], api=1.0),
    group(9.5, [req(9.5, [30], api=1.0)]),
]

# A sub-agent that ends at 9, after the parent turn that follows its start.
OVERLAP = [
    req(0, [1, 2], api=1.0),
    group(2, [req(2, [10], api=7.0)]),
    req(6, [1, 2, 3], api=1.0),
    req(12, [1, 2, 3, 4], api=1.0),
]


def _waits(compiled):
    rows = compiled.rows
    return [(rows[o], rows[o + 8]) for o in range(0, len(rows), _ROW)]


def _config(path, **workload):
    return RunConfig(workload=WorkloadConfig(trace=str(path), **workload))


def _write(tmp_path, traces, name="replay.cbtrace"):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(item) + "\n" for item in traces))
    out = tmp_path / name
    compile_corpus(source, out)
    return out


def _workload(tmp_path, traces, **workload):
    workload.setdefault("think_cap_s", 100.0)
    return TraceWorkload(_config(_write(tmp_path, traces), **workload))


def _ids(path):
    """The source ids of the traces in a compiled file, in order."""
    lines = path.read_bytes().splitlines()[1:]
    return [msgspec.json.decode(line, type=Compiled).id for line in lines]


def test_a_request_waits_the_recorded_gap_after_the_one_before_it_in_its_lane():
    compiled = compile_trace(trace(AGENTS))
    assert compiled.spawn == [-1, 0, 0, 2]
    # Both early sub-agents end before the turn at 6; no turn follows the last.
    assert compiled.join == [-1, 1, 1, -1]
    # Main lane: nothing, then 6 - 1 (the sub-agents it joins end later, at 4,
    # and do not count), then 8 - 7. Sub-agents: from the start of their
    # branch, then after their own requests (3 - 2.5).
    assert _waits(compiled) == [
        (0, 0),
        (0, 5000),
        (0, 1000),
        (1, 0),
        (1, 500),
        (2, 500),
        (3, 0),
    ]


def test_sub_agents_with_one_spawn_and_join_start_from_the_earliest_of_their_groups():
    def first_waits(api, parent_api=1.0):
        requests = [
            req(0, [1], api=parent_api),
            group(2, [req(2.5, [10], api=1.0)]),
            group(3, [req(4, [20], api=api)]),
            req(6, [1, 2]),
            req(12, [1, 2, 3]),
        ]
        compiled = compile_trace(trace(requests))
        return compiled.join[1:], [wait for lane, wait in _waits(compiled) if lane]

    # Both join the turn at 6, so both count from 2: 2.5 - 2 and 4 - 2.
    assert first_waits(1.0) == ([1, 1], [500, 2000])
    # The second ends at 9 and joins the turn at 12: it counts from its own 3.
    assert first_waits(5.0) == ([1, 2], [500, 1000])
    # A branch that began before the parent turn ended (at 2.2) counts from
    # that end: nothing starts before the turn returns.
    assert first_waits(1.0, parent_api=2.2) == ([1, 1], [300, 1800])


def test_a_negative_or_non_finite_api_time_counts_as_no_time():
    for api in (-3.0, float("nan"), float("inf")):
        compiled = compile_trace(trace([req(0, [1], api=api), req(5, [1, 2])]))
        assert _waits(compiled) == [(0, 0), (0, 5000)], api


def test_a_sub_agent_joins_the_first_parent_turn_recorded_at_or_after_its_end():
    compiled = compile_trace(trace(OVERLAP))
    assert (compiled.spawn, compiled.join) == ([-1, 0], [-1, 2])

    def join(api):
        child = group(2, [req(2, [10], api=api)])
        return compile_trace(trace([OVERLAP[0], child, *OVERLAP[2:]])).join

    assert join(4.0) == [-1, 1]  # ends at 6, as the next parent turn starts
    assert join(10.5) == [-1, -1]  # ends after the last parent turn starts


def test_a_sub_agent_ends_after_its_recorded_duration_not_with_its_last_call():
    def join(**recorded):
        child = {**group(2, [req(2, [10], api=3.0)]), **recorded}
        turns = [req(3, [1, 2]), req(6, [1, 2, 3]), req(13, [1, 2, 3, 4])]
        return compile_trace(trace([req(0, [1]), child, *turns])).join[1]

    assert join() == 2  # no duration: the call's end, 5, so the turn at 6
    assert join(duration_ms=10_000) == 3  # ends at 12: the turn at 13
    # A negative or non-finite duration counts as 0: it ends at 2, as it starts.
    assert join(duration_ms=-1) == join(duration_ms=float("nan")) == 1


def test_inner_times_relative_to_the_group_are_shifted():
    absolute = compile_trace(trace(AGENTS))
    relative = [dict(item) for item in AGENTS]
    relative[1] = group(1.5, [req(0, [10, 11], api=1.0), req(1.5, [10, 11, 12])])
    assert compile_trace(trace(relative)).rows == absolute.rows


@pytest.mark.parametrize("first", ["a", "b"])
def test_sub_agents_run_together_and_the_parent_waits_for_all_of_them(tmp_path, first):
    workload = _workload(tmp_path, [trace(AGENTS)])
    main = workload.new_session().id
    turn = workload.next_turn(main)
    a, b = main + 1, main + 2
    step = workload.completed(turn)
    assert sorted(step.wake) == [(0.0, a), (0.5, b)] and not step.ended  # no parent

    def finish_a():
        again = workload.completed(workload.next_turn(a))
        assert again == ([(0.5, a)], [])
        return workload.completed(workload.next_turn(a))

    def finish_b():
        return workload.completed(workload.next_turn(b))

    order = (finish_a, finish_b) if first == "a" else (finish_b, finish_a)
    early, late = order[0](), order[1]()
    assert early.wake == [] and len(early.ended) == 1
    # The parent is parked: the last sub-agent to finish sends it out at once.
    assert late.wake == [(0.0, main)] and sorted(early.ended + late.ended) == [a, b]
    turn = workload.next_turn(main)
    assert turn.turn == 1 and turn.segments == ((TRACE_BASE, 0, 3),)


def test_a_parent_turn_before_the_join_runs_beside_the_sub_agent(tmp_path):
    workload = _workload(tmp_path, [trace(OVERLAP)])
    main = workload.new_session().id
    child = main + 1
    step = workload.completed(workload.next_turn(main))
    # The turn at 6 is due without the sub-agent having run.
    assert sorted(step.wake) == [(0.0, child), (5.0, main)] and not step.ended
    assert workload.next_turn(child).turn == 0 and workload.next_turn(main).turn == 1


# The gap recorded before OVERLAP's turn at 12 is 5 s, after the turn at 6.
@pytest.mark.parametrize(("first", "delay"), [("child", 5.0), ("parent", 0.0)])
def test_a_join_turn_waits_its_gap_unless_it_was_parked_for_its_sub_agents(
    tmp_path, first, delay
):
    workload = _workload(tmp_path, [trace(OVERLAP)])
    main = workload.new_session().id
    child = main + 1
    workload.completed(workload.next_turn(main))
    running, beside = workload.next_turn(child), workload.next_turn(main)
    order = (running, beside) if first == "child" else (beside, running)
    early, late = workload.completed(order[0]), workload.completed(order[1])
    # The turn at 12 waits for both. The later one releases it: the turn at 6
    # after the gap, the sub-agent (the parent was parked for it) at once.
    assert early.wake == [] and late.wake == [(delay, main)]
    assert early.ended + late.ended == [child]
    last = workload.next_turn(main)
    assert last.turn == 2
    step = workload.completed(last)
    assert step.ended == [main] and workload.started == 2


def test_a_play_is_replaced_only_when_every_lane_has_ended(tmp_path):
    workload = _workload(tmp_path, [trace(AGENTS)])
    main = workload.new_session().id
    workload.completed(workload.next_turn(main))
    for lane in (main + 1, main + 1, main + 2):
        workload.completed(workload.next_turn(lane))
    assert workload.completed(workload.next_turn(main)) == ([(1.0, main)], [])
    # The last main turn starts a sub-agent nobody waits for.
    step = workload.completed(workload.next_turn(main))
    assert step == ([(0.0, main + 3)], [main]) and workload.started == 1
    step = workload.completed(workload.next_turn(main + 3))
    assert step.ended == [main + 3]
    ((delay, fresh),) = step.wake
    assert delay == 0.0 and fresh == main + 4 and workload.started == 2
    assert workload.next_turn(fresh).turn == 0
    summary = workload.summary()
    assert summary["plays_finished"] == 1 and summary["child_turns"] == 4


def test_waits_are_scaled_then_capped(tmp_path):
    # Recorded gaps of 0.5 s and 2 s.
    turns = [req(0, [1]), req(1.5, [1, 2]), req(4.5, [1, 2, 3])]
    workload = _workload(
        tmp_path, [trace(turns)], trace_think_scale=2.0, think_cap_s=1.5
    )
    main = workload.new_session().id
    assert workload.completed(workload.next_turn(main)).wake == [(1.0, main)]
    assert workload.completed(workload.next_turn(main)).wake == [(1.5, main)]


def test_an_answer_to_a_prompt_under_one_block_is_shortened_too(tmp_path):
    short = [req(0, [], tokens=40, out=2000), req(5, [1, 2])]
    workload = _workload(tmp_path, [trace(short)], trace_output_scale=0.02)
    assert workload.next_turn(workload.new_session().id).output_tokens == 40


def test_shortened_answers_complete_fewer_blocks(tmp_path):
    session = [req(0, range(4), out=4 * BLOCK), req(5, range(10))]

    def first_turn(scale):
        workload = _workload(tmp_path, [trace(session)], trace_output_scale=scale)
        return workload.next_turn(workload.new_session().id)

    full, half, tiny = first_turn(1.0), first_turn(0.5), first_turn(0.1)
    assert (full.output_tokens, full.tail) == (4 * BLOCK, (TRACE_BASE, 4, 8))
    assert (half.output_tokens, half.tail) == (2 * BLOCK, (TRACE_BASE, 4, 6))
    # Under one block: the next prompt computes what the answer no longer does.
    assert (tiny.output_tokens, tiny.tail) == (26, (TRACE_BASE, 4, 4))
    assert full.segments == half.segments == tiny.segments
    with pytest.raises(ValueError, match="trace_output_scale"):
        load_config(None, ["workload.trace_output_scale=0"])


def test_plays_of_one_trace_share_no_stream(tmp_path):
    workload = _workload(tmp_path, [trace(AGENTS)])
    streams = []
    for _ in range(2):
        play = workload.new_session()
        seen = set()
        for lane in (0, 1, 1, 2):  # the first main turn, then both sub-agents
            turn = workload.next_turn(play.id + lane)
            workload.completed(turn)
            seen |= {stream for stream, _, _ in (*turn.segments, turn.tail)}
        assert len(seen) == 3
        streams.append(seen)
    assert not streams[0] & streams[1]
    assert min(streams[0] | streams[1]) >= TRACE_BASE


def _many(count):
    traces = []
    for n in range(count):
        # Trace n's first prompt is 2 + n blocks and grows by one a turn.
        requests = [
            req(100.0 * turn, range(n * 1000, n * 1000 + 2 + n + turn))
            for turn in range(30)
        ]
        requests.insert(3, group(250.0, [req(250.0, [n * 1000 + 900])]))
        traces.append(trace(requests, tid=f"t{n}"))
    return traces


def test_aged_plays_start_part_way_and_replacements_at_the_beginning(tmp_path):
    path = _write(tmp_path, _many(8))

    def starts(seed, aged):
        workload = TraceWorkload(_config(path, seed=seed))
        out = []
        for _ in range(16):
            play = workload.new_session(aged=aged)
            turn = workload.next_turn(play.id)
            out.append((turn.segments[0][2], turn.turn))
        return out

    aged = starts(3, True)
    assert aged == starts(3, True) and aged != starts(4, True)
    assert sum(turn > 0 for _, turn in aged) >= 12
    assert all(turn == 0 for _, turn in starts(3, False))
    # Every trace is played once before any is repeated.
    played = [blocks - turn - 2 for blocks, turn in aged]
    assert sorted(played[:8]) == sorted(played[8:]) == list(range(8))


def test_a_play_that_starts_past_a_sub_agent_does_not_wait_for_it(tmp_path):
    workload = _workload(tmp_path, _many(1))
    for _ in range(20):
        play = workload.new_session(aged=True)
        turn = workload.next_turn(play.id)
        if turn.turn >= 3:  # the sub-agent follows turn 2
            break
    else:
        pytest.fail("no aged play started past the sub-agent")
    for _ in range(turn.turn, 29):  # on to the last of its 30 turns
        workload.completed(turn)
        turn = workload.next_turn(play.id)
    # The sub-agent is over: the play ends with its main lane.
    started = workload.started
    assert workload.completed(turn).ended == [play.id]
    assert workload.started == started + 1


def test_a_play_that_starts_between_a_spawn_and_its_join_does_not_wait(tmp_path):
    workload = _workload(tmp_path, [trace(OVERLAP)])
    for _ in range(20):
        play = workload.new_session(aged=True)
        turn = workload.next_turn(play.id)
        if turn.turn == 1:  # the turn at 6: after the spawn, before the join
            break
    else:
        pytest.fail("no aged play started between the spawn and the join")
    # The sub-agent is over: the turn at 12 follows the turn at 6 alone.
    assert workload.completed(turn) == ([(5.0, play.id)], [])
    assert workload.next_turn(play.id).turn == 2


# ---- files and errors -----------------------------------------------------------


def test_the_compiled_file_round_trips(tmp_path):
    traces = [trace(AGENTS, tid="a"), trace([req(0, [5, 6, 7], out=BLOCK)], tid="b")]
    path = _write(tmp_path, traces)
    header, loaded = load(path)
    assert (header.traces, header.turns, header.block_tokens) == (2, 8, BLOCK)
    assert (header.max_prompt_blocks, header.max_request_blocks) == (4, 4)
    assert header.source == "source.jsonl" and len(header.sha256) == 64
    assert len(loaded) == 2 and _ids(path) == ["a", "b"]
    lines = path.read_bytes().splitlines()
    decoded = msgspec.json.decode(lines[1], type=Compiled)
    assert decoded == compile_trace(traces[0])


def test_the_longest_request_counts_every_token_not_only_whole_cached_blocks(tmp_path):
    # Two cached blocks and one answered, but 188 + 10 tokens reach a fourth.
    request = req(0, [1, 2], tokens=2 * BLOCK + 60, out=10)
    header, _ = load(_write(tmp_path, [trace([request])]))
    assert (header.max_prompt_blocks, header.max_request_blocks) == (2, 4)


def test_compile_errors_name_the_trace_and_request():
    with pytest.raises(TraceError, match="trace t0: request 1: nested"):
        compile_trace(trace([req(0, [1]), group(1, [group(1, [req(1, [2])])])]))
    bad = req(0, [1])
    bad["in"] = "many"
    with pytest.raises(TraceError, match="trace t0: request 0: bad in/out"):
        compile_trace(trace([bad]))
    with pytest.raises(TraceError, match="bad block_size"):
        compile_trace(trace([req(0, [1])], block=0))


def test_groups_with_nothing_to_run_or_nothing_to_follow_are_left_out():
    stats = Stats()
    compiled = compile_trace(
        trace([group(0, [req(0, [9])]), req(1, [1]), group(2, []), req(3, [1, 2])]),
        stats,
    )
    assert compiled.spawn == [-1]
    assert (stats.orphan_groups, stats.empty_groups, stats.groups) == (1, 1, 0)
    assert compile_trace(trace([group(0, [req(0, [9])])]), stats) is None


def test_a_corpus_must_use_one_block_size_and_hold_a_trace(tmp_path):
    source = tmp_path / "mixed.jsonl"
    rows = [trace([req(0, [1])]), trace([req(0, [1])], tid="t1", block=32)]
    source.write_text("".join(json.dumps(item) + "\n" for item in rows))
    with pytest.raises(TraceError, match="trace t1: block_size 32 differs from 64"):
        compile_corpus(source, tmp_path / "out")
    source.write_text(json.dumps(trace([])) + "\n")
    with pytest.raises(TraceError, match="no trace with a main-lane request"):
        compile_corpus(source, tmp_path / "out")


def test_the_limit_takes_the_first_traces(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(item) + "\n" for item in _many(5)))
    stats = compile_corpus(source, tmp_path / "out", limit=2)
    assert stats.traces == 2 and _ids(tmp_path / "out") == ["t0", "t1"]


def test_a_trace_that_does_not_fit_the_run_is_refused(tmp_path):
    long = trace([req(0, range(300), out=20 * BLOCK), req(5, range(330))])
    path = _write(tmp_path, [long])
    workload = WorkloadConfig(trace=str(path))

    def run(**tiers):
        return RunConfig(workload=workload, tiers=TierConfig(**tiers))

    TraceWorkload(run())
    with pytest.raises(TraceError, match="64-token blocks, but tiers.tokens_per_block"):
        TraceWorkload(run(tokens_per_block=16))
    with pytest.raises(TraceError, match=r"longest request is 330 .* tiers.gpu_blocks"):
        TraceWorkload(run(gpu_blocks=329))
    # The vLLM model promotes a whole prompt into the CPU tier before starting.
    reserved = 1 + PhaseConfig().warm_concurrency
    TraceWorkload(run(primary_blocks=330 + reserved))
    with pytest.raises(TraceError, match="longest prompt is 330 .*primary_blocks"):
        TraceWorkload(run(primary_blocks=329 + reserved))


def test_unreadable_trace_files_say_what_to_do(tmp_path):
    with pytest.raises(
        TraceError, match="cannot read workload.trace .*clusterbench trace"
    ):
        TraceWorkload(_config(tmp_path / "missing.cbtrace"))
    path = _write(tmp_path, [trace(AGENTS)])
    lines = path.read_bytes().splitlines()
    header = msgspec.json.decode(lines[0], type=Header)
    old = msgspec.structs.replace(header, clusterbench_trace=0)
    path.write_bytes(msgspec.json.encode(old) + b"\n" + lines[1] + b"\n")
    with pytest.raises(TraceError, match="format 0, this build reads 3; compile"):
        TraceWorkload(_config(path))
    path.write_bytes(b"not json\n")
    with pytest.raises(TraceError, match="not a compiled trace"):
        TraceWorkload(_config(path))
    deep = msgspec.structs.replace(header, max_segments=300)
    path.write_bytes(msgspec.json.encode(deep) + b"\n" + lines[1] + b"\n")
    with pytest.raises(TraceError, match="crosses 300 streams"):
        TraceWorkload(_config(path))


def test_a_coordinator_refuses_a_bad_trace_before_any_worker_starts(tmp_path):
    with pytest.raises(TraceError, match="cannot read workload.trace"):
        Coordinator(_config(tmp_path / "missing.cbtrace"), tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_the_workload_follows_the_config(tmp_path):
    assert isinstance(make_workload(RunConfig()), Workload)
    path = _write(tmp_path, [trace(AGENTS)])
    workload = make_workload(_config(path))
    assert isinstance(workload, TraceWorkload)
    assert workload.summary()["trace"] == "source.jsonl"
    with pytest.raises(ValueError, match="trace_think_scale"):
        load_config(None, ["workload.trace_think_scale=-1"])
