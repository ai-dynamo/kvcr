# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from clusterbench.config import WorkloadConfig
from clusterbench.ids import system_stream
from clusterbench.workload import Session, Workload

BLOCK = 64


def _turns(session, limit=10_000):
    turns = []
    while (turn := session.next_turn()) is not None and len(turns) < limit:
        turns.append(turn)
    return turns


def test_sessions_are_deterministic_and_independent():
    config = WorkloadConfig(seed=3)
    a = _turns(Session(11, config, BLOCK))
    # Another session drawing in between must not disturb session 11.
    other = Session(12, config, BLOCK)
    b_session = Session(11, config, BLOCK)
    b = []
    while (turn := b_session.next_turn()) is not None:
        other.next_turn()
        b.append(turn)
    assert a == b
    assert a != _turns(Session(11, WorkloadConfig(seed=4), BLOCK))


def test_context_grows_as_a_prefix_until_it_is_compacted():
    config = WorkloadConfig(seed=1, turns_median=400, turns_sigma=0.01)
    turns = _turns(Session(0, config, BLOCK))
    assert len(turns) > 50
    resets = 0
    for before, after in zip(turns, turns[1:]):
        assert after.prompt_tokens + after.output_tokens <= config.max_context_tokens
        assert after.segments[0] == before.segments[0]  # shared system prefix
        if after.segments[-1][0] == before.segments[-1][0]:
            # Same context chain: last turn's prompt and answer are a prefix.
            assert after.segments[-1][2] >= before.tail[2]
            assert before.tail[1] == before.segments[-1][2]
        else:
            resets += 1
    assert resets >= 1  # a 400-turn session must hit the cap
    assert turns[-1].last and not turns[0].last


def test_system_prefix_is_shared_by_agent_type():
    config = WorkloadConfig(seed=0, agent_types=2)
    sessions = [Session(i, config, BLOCK) for i in range(20)]
    first = [s.next_turn() for s in sessions]
    streams = {t.segments[0][0] for t in first}
    assert streams == {system_stream(0), system_stream(1)}
    blocks = config.system_prefix_tokens // BLOCK
    assert all(t.segments[0] == (t.segments[0][0], 0, blocks) for t in first)


def test_workload_replaces_ended_sessions_with_new_ids():
    workload = Workload(WorkloadConfig(turns_median=1, turns_sigma=0.01), BLOCK)
    session = workload.new_session()
    turn = workload.next_turn(session.id)
    assert turn.last
    assert workload.next_turn(session.id) is None
    assert workload.new_session().id == session.id + 1


def test_aged_sessions_start_mid_life_with_grown_contexts():
    config = WorkloadConfig(seed=5)
    fresh = [Session(i, config, BLOCK).next_turn() for i in range(200)]
    aged = [Session(i, config, BLOCK, aged=True).next_turn() for i in range(200)]
    assert all(t.turn == 0 for t in fresh)
    assert sum(t.turn > 0 for t in aged) > 150

    def mean(turns):
        return sum(t.prompt_tokens for t in turns) / len(turns)

    assert mean(aged) > 2 * mean(fresh)
    cap = config.max_context_tokens
    assert all(t.prompt_tokens + t.output_tokens <= cap for t in aged)


def test_a_completed_turn_wakes_its_session_or_the_replacement():
    workload = Workload(WorkloadConfig(turns_median=2, turns_sigma=0.01), BLOCK)
    session = workload.new_session()
    first = workload.next_turn(session.id)
    assert workload.completed(first) == (((first.think_s, session.id),), ())
    last = workload.next_turn(session.id)
    assert last.last
    # The session is over: a fresh one takes its place after the think time.
    step = workload.completed(last)
    assert step.ended == (session.id,)
    assert step.wake == ((last.think_s, session.id + 1),)
    assert workload.started == 2 and workload.next_turn(session.id + 1).turn == 0
