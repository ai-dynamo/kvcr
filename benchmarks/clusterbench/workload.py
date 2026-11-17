# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Closed-loop agentic session generator.

Each session is a coding-agent style conversation: a system/tool prefix shared
by every session of the same agent type, then a context that grows by one tool
result or user message plus the previous answer every turn, until it hits the
context cap and is compacted. Every session draws from its own RNG, so its
sizes and think times do not depend on how fast the system under test is.
"""

from __future__ import annotations

import math
import random
from typing import NamedTuple

from .config import WorkloadConfig
from .ids import Segment, session_stream, system_stream


class Turn(NamedTuple):
    session: int
    turn: int
    segments: tuple[Segment, ...]  # cacheable prompt blocks, root first
    tail: Segment  # blocks the answer adds to the context stream
    prompt_tokens: int
    output_tokens: int
    think_s: float  # wait after this turn completes before the next one
    last: bool


def _lognormal(rng: random.Random, median: float, sigma: float) -> float:
    return rng.lognormvariate(math.log(median), sigma)


class Session:
    def __init__(
        self, session: int, config: WorkloadConfig, block: int, aged: bool = False
    ) -> None:
        self.id = session
        self._config = config
        self._block = block
        self._rng = random.Random((config.seed << 32) ^ (session * 2654435761))
        rng = self._rng
        self.agent_type = rng.randrange(config.agent_types)
        self._turns = max(
            1, round(_lognormal(rng, config.turns_median, config.turns_sigma))
        )
        self._turn = 0
        self._generation = 0
        self._context = self._sample(
            config.initial_context_median, config.initial_context_sigma
        )

        if aged:
            # A session that was already running when the benchmark started:
            # somewhere along its life, with a context to match.
            room = config.max_context_tokens - config.system_prefix_tokens
            self._context = rng.randint(min(self._context, room), max(1, room))
            self._turn = int(rng.random() * self._turns)

    def _sample(self, median: float, sigma: float) -> int:
        return max(1, round(_lognormal(self._rng, median, sigma)))

    def next_turn(self) -> Turn | None:
        config = self._config
        if self._turn >= self._turns:
            return None
        if self._turn:
            self._context += self._sample(
                config.turn_tokens_median, config.turn_tokens_sigma
            )
        output = self._sample(config.output_tokens_median, config.output_tokens_sigma)
        system = config.system_prefix_tokens
        if system + self._context + output > config.max_context_tokens:
            # Compaction: a fresh context chain behind the same system prefix.
            self._generation += 1
            self._context = min(
                self._sample(config.reset_context_median, config.initial_context_sigma),
                max(1, config.max_context_tokens - system - output),
            )
        output = min(output, max(1, config.max_context_tokens - system - self._context))
        block = self._block
        segments: list[Segment] = []
        if system >= block:
            segments.append((system_stream(self.agent_type), 0, system // block))
        # Context blocks are aligned to the session's own stream, so the block
        # straddling the system/context boundary is simply not cacheable.
        stream = session_stream(self.id, self._generation)
        if self._context >= block:
            segments.append((stream, 0, self._context // block))
        tail = (stream, self._context // block, (self._context + output) // block)
        think = min(
            config.think_cap_s,
            _lognormal(self._rng, config.think_median_s, config.think_sigma),
        )
        turn = Turn(
            session=self.id,
            turn=self._turn,
            segments=tuple(segments),
            tail=tail,
            prompt_tokens=system + self._context,
            output_tokens=output,
            think_s=think,
            last=self._turn + 1 >= self._turns,
        )
        # The answer becomes part of the next turn's context.
        self._context += output
        self._turn += 1
        return turn


class Workload:
    """Hands out sessions; ended sessions are replaced by fresh ones."""

    def __init__(self, config: WorkloadConfig, tokens_per_block: int) -> None:
        self._config = config
        self._block = tokens_per_block
        self._next_session = 0
        self._sessions: dict[int, Session] = {}

    def new_session(self, aged: bool = False) -> Session:
        session = Session(self._next_session, self._config, self._block, aged)
        self._next_session += 1
        self._sessions[session.id] = session
        return session

    def next_turn(self, session: int) -> Turn | None:
        turn = self._sessions[session].next_turn()
        if turn is None:
            del self._sessions[session]
        return turn

    @property
    def started(self) -> int:
        return self._next_session
