# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Trace replay: compile a recorded agent corpus once, then replay it.

The source is the "Weka" trace format that the AgentX benchmark replays: one
JSON object per session, whose requests carry the block hashes of their whole
prompt. Two requests share a cached prefix exactly as far as their hash lists
agree from the start, so the requests of a session form a prefix tree.

``compile_corpus`` turns each tree into the streams the rest of clusterbench
works with. The tree is cut into its branch-free runs and each run is one
stream, so a request is the runs on its path, root first: every segment
starts at 0 and all but the last cover a whole stream. That is the shape the
synthetic workload produces, and the one the device caches rely on (a stream
grows from its root, a stream that others hang from is used in full). The
cuts have to be known up front: a block's ``(stream, index)`` is its KVCR
key, which cannot change once a peer holds it.

``TraceWorkload`` replays the compiled file closed loop, following aiperf's
AgentX rules: a request goes out the recorded gap after its latest dependency
ends, a sub-agent starts after the parent turn before it and runs beside its
siblings and the parent turns that follow, the first of those recorded at or
after the sub-agent's end waits for it, and every play of a trace gets block
ids of its own.
"""

from __future__ import annotations

import hashlib
import random
from bisect import bisect_left
from collections import Counter
from pathlib import Path

import msgspec

from .config import RunConfig
from .ids import MAX_STREAM, TRACE_BASE, Segment
from .workload import Step, Turn

FORMAT = 2
# Ints per turn: lane, last stream, blocks used of it, tail stream, tail start,
# tail end, prompt tokens, output tokens, wait (ms), recorded time (ms).
_ROW = 10
MAX_SEGMENTS = 255  # the device cache orders a request's streams in one byte
# A parent turn recorded this close before a sub-agent's end still joins it;
# mirrors aiperf's join epsilon.
_JOIN_EPSILON_S = 1e-6


class TraceError(ValueError):
    pass


class Header(msgspec.Struct, kw_only=True):
    clusterbench_trace: int
    source: str
    sha256: str
    block_tokens: int
    traces: int
    turns: int
    max_prompt_blocks: int
    max_request_blocks: int
    max_segments: int


class Compiled(msgspec.Struct, array_like=True):
    """One source trace."""

    id: str
    # (parent, length) per stream; a parent comes before its children.
    edges: list[int]
    # Per lane, the main-lane turn it starts after (-1 for the main lane).
    spawn: list[int]
    # Per lane, the main-lane turn that waits for it: the first one after
    # ``spawn`` recorded at or after the lane's end (-1 for the main lane and
    # for a lane no later turn waits for).
    join: list[int]
    # ``_ROW`` ints per turn, lane by lane.
    rows: list[int]


# ---- compile ------------------------------------------------------------------


class Stats:
    def __init__(self) -> None:
        self.traces = 0
        self.skipped_traces = 0
        self.main_turns = 0
        self.child_turns = 0
        self.groups = 0
        self.empty_groups = 0
        self.orphan_groups = 0
        # Where the answers' blocks went.
        self.answer_blocks = 0
        self.tail_blocks = 0
        self.cut_blocks = 0  # a branch falls inside the answer
        self.uncarried_blocks = 0  # the next prompt grew by less than the answer
        self.dropped_blocks = 0  # nothing later holds them, or they are not new
        self.prompt_blocks: list[int] = []
        self.new_blocks: list[int] = []
        self.segments: list[int] = []
        self.streams: list[int] = []
        self.fanout: list[int] = []
        self.waits_ms: list[int] = []
        self.max_prompt_blocks = 0
        self.max_request_blocks = 0
        self.max_segments = 0

    @property
    def turns(self) -> int:
        return self.main_turns + self.child_turns

    def format(self) -> str:
        def spread(values: list[int], scale: float = 1.0) -> str:
            if not values:
                return "-"
            ordered = sorted(values)
            last = len(ordered) - 1
            picks = [ordered[min(last, len(ordered) * p // 100)] for p in (50, 90, 99)]
            cells = [f"{value * scale:g}" for value in (*picks, ordered[-1])]
            return "p50 {} / p90 {} / p99 {} / max {}".format(*cells)

        answers = max(1, self.answer_blocks)
        return "\n".join(
            [
                f"traces                 {self.traces}"
                f" (skipped, no main turn: {self.skipped_traces})",
                f"turns                  {self.turns}"
                f" (main {self.main_turns}, sub-agent {self.child_turns})",
                f"sub-agent groups       {self.groups}"
                f" (empty {self.empty_groups}, before any main turn"
                f" {self.orphan_groups}: both left out)",
                f"prompt blocks          {spread(self.prompt_blocks)}",
                f"new blocks per turn    {spread(self.new_blocks)}",
                f"streams per trace      {spread(self.streams)}",
                f"segments per request   {spread(self.segments)}",
                f"groups after one turn  {spread(self.fanout)}",
                f"wait before a turn (s) {spread(self.waits_ms, 1e-3)}",
                f"longest request        {self.max_request_blocks} blocks"
                f" (prompt {self.max_prompt_blocks})",
                f"answer blocks          {self.answer_blocks}:"
                f" kept {self.tail_blocks / answers:.1%},"
                f" not in a later prompt {self.uncarried_blocks / answers:.1%},"
                f" cut at a branch {self.cut_blocks / answers:.1%},"
                f" dropped {self.dropped_blocks / answers:.1%}",
            ]
        )


class _Row:
    __slots__ = (
        "t", "end", "tokens", "out", "ids", "answer", "node", "new", "leaf", "used",
        "tail", "wait",
    )  # fmt: skip


def _row(entry: dict, block: int, tid: str, index: int) -> _Row:
    tokens, out, ids = entry.get("in"), entry.get("out"), entry.get("hash_ids")
    if (
        not isinstance(tokens, int)
        or not isinstance(out, int)
        or tokens < 0
        or out < 0
        or not isinstance(ids, list | type(None))
    ):
        raise TraceError(f"trace {tid}: request {index}: bad in/out/hash_ids")
    ids = ids or []
    row = _Row()
    # Only whole blocks can be cached; a trailing partial block has an id too.
    blocks = min(tokens // block, len(ids))
    row.ids = ids if blocks == len(ids) else ids[:blocks]
    # Where the answer's blocks sit is only known when the ids cover the prompt.
    whole = blocks == tokens // block
    row.answer = (tokens + out) // block - tokens // block if whole else 0
    row.tokens = tokens
    row.out = out
    row.t = float(entry.get("t") or 0.0)
    row.end = row.t + float(entry.get("api_time") or 0.0)
    row.node = -1
    row.new = 0
    row.leaf = -1
    row.used = 0
    row.tail = (-1, 0, 0)
    row.wait = 0.0
    return row


def _common(a: list[int], b: list[int]) -> int:
    """Length of the common prefix, by bisection over slice comparisons."""
    high = min(len(a), len(b))
    if a[:high] == b[:high]:
        return high
    low = 0
    while high - low > 1:
        mid = (low + high) // 2
        if a[:mid] == b[:mid]:
            low = mid
        else:
            high = mid
    return low


def _toward(parents: list[int], low: int, high: int) -> int:
    """The child of ``high`` on the way down to ``low``, or -1."""
    while low >= 0:
        up = parents[low]
        if up == high:
            return low
        low = up
    return -1


def compile_trace(trace: dict, stats: Stats | None = None) -> Compiled | None:
    """Compile one source trace; None if it has no main-lane request."""
    stats = stats if stats is not None else Stats()
    tid = str(trace.get("id", "?"))
    block = trace.get("block_size")
    if not isinstance(block, int) or block <= 0:
        raise TraceError(f"trace {tid}: bad block_size {block!r}")

    # Lanes: the main conversation, then one per sub-agent group.
    lanes: list[list[_Row]] = [[]]
    spawn = [-1]
    for index, entry in enumerate(trace.get("requests") or ()):
        if entry.get("type") != "subagent":
            lanes[0].append(_row(entry, block, tid, index))
            continue
        inner = entry.get("requests") or ()
        if any(item.get("type") == "subagent" for item in inner):
            raise TraceError(f"trace {tid}: request {index}: nested sub-agent group")
        if not inner:
            stats.empty_groups += 1
        elif not lanes[0]:
            stats.orphan_groups += 1  # nothing to start after; aiperf drops it too
        else:
            rows = [_row(item, block, tid, index) for item in inner]
            # Inner times are absolute in the corpora seen so far; a producer
            # that writes them relative to the group starts them near zero.
            start = float(entry.get("t") or 0.0)
            if min(row.t for row in rows) + 1e-6 < start:
                for row in rows:
                    row.t += start
                    row.end += start
            lanes.append(rows)
            spawn.append(len(lanes[0]) - 1)
    if not lanes[0]:
        stats.skipped_traces += 1
        return None

    # The prefix tree, built in recorded order so a node knows who named it
    # first. A block is keyed by its path, not by its id alone.
    order = sorted(
        (row.t, lane, index)
        for lane, rows in enumerate(lanes)
        for index, row in enumerate(rows)
    )
    child: dict[tuple[int, int], int] = {}
    parent: list[int] = []
    kids: list[int] = []
    born_lane: list[int] = []
    born_index: list[int] = []
    # First id -> (ids, nodes) of the latest request starting with it: most
    # requests extend an earlier one, so the walk resumes where they part.
    recent: dict[int, tuple[list[int], list[int]]] = {}
    for _, lane, index in order:
        row = lanes[lane][index]
        ids = row.ids
        if not ids:
            continue
        seen = recent.get(ids[0])
        if seen is None:
            nodes: list[int] = []
        else:
            nodes = seen[1][: _common(ids, seen[0])]
        node = nodes[-1] if nodes else -1
        made = 0
        for position in range(len(nodes), len(ids)):
            key = (node, ids[position])
            found = child.get(key)
            if found is None:
                found = child[key] = len(parent)
                parent.append(node)
                kids.append(0)
                born_lane.append(lane)
                born_index.append(index)
                if node >= 0:
                    kids[node] += 1
                made += 1
            nodes.append(found)
            node = found
        recent[ids[0]] = (ids, nodes)
        row.node = node
        row.new = made
    del child, recent

    # Radix edges: a node starts a new one if it is a root or a sibling.
    count = len(parent)
    edge_of = [0] * count
    place_of = [0] * count
    parents: list[int] = []
    lengths: list[int] = []
    depths: list[int] = []
    members: list[list[int]] = []
    for node in range(count):
        above = parent[node]
        if above < 0 or kids[above] > 1:
            edge = len(parents)
            up = edge_of[above] if above >= 0 else -1
            parents.append(up)
            lengths.append(0)
            depths.append(depths[up] + 1 if up >= 0 else 1)
            members.append([])
        else:
            edge = edge_of[above]
        edge_of[node] = edge
        place_of[node] = lengths[edge]
        lengths[edge] += 1
        members[edge].append(node)
    for rows in lanes:
        for row in rows:
            if row.node >= 0:
                row.leaf = edge_of[row.node]
                row.used = place_of[row.node] + 1

    # Tails. An answer's blocks are the first blocks of the next request that
    # extends this prompt, as far as that request is the first to name them.
    # Later requests claim first, so of two identical prompts the one right
    # before the continuation gets them.
    claimed: set[int] = set()
    leaf_taken: set[int] = set()
    for lane, rows in enumerate(lanes):
        for index in range(len(rows) - 1, -1, -1):
            row = rows[index]
            answer = row.answer
            stats.answer_blocks += answer
            if row.node < 0:
                stats.dropped_blocks += answer
                continue
            leaf, used = row.leaf, row.used
            edge, start, room, growth = leaf, used, 0, 0
            for later in rows[index + 1 :]:
                if later.leaf == leaf:
                    if later.used > used:
                        room = later.used - used
                        growth = room
                        break
                elif later.leaf >= 0 and depths[later.leaf] > depths[leaf]:
                    below = _toward(parents, later.leaf, leaf)
                    if below < 0:
                        continue
                    growth = len(later.ids) - len(row.ids)
                    if used < lengths[leaf]:
                        room = lengths[leaf] - used
                    else:
                        edge, start = below, 0
                        room = later.used if below == later.leaf else lengths[below]
                    break
            kept = 0
            if growth:
                wanted = min(answer, growth)
                stats.uncarried_blocks += answer - wanted
                fits = min(wanted, room)
                stats.cut_blocks += wanted - fits
                nodes = members[edge]
                while kept < fits:
                    node = nodes[start + kept]
                    if (
                        born_lane[node] != lane
                        or born_index[node] <= index
                        or node in claimed
                    ):
                        break
                    kept += 1
                stats.dropped_blocks += fits - kept
                claimed.update(nodes[start : start + kept])
                if not kept:
                    edge, start = leaf, used
            elif answer and not kids[row.node] and leaf not in leaf_taken:
                # Nobody extends this prompt: the answer lies past the end of
                # its stream, where no other request can reach.
                leaf_taken.add(leaf)
                kept = answer
            else:
                stats.dropped_blocks += answer
            stats.tail_blocks += kept
            row.tail = (edge, start, start + kept)

    # Joins. Recorded times need not rise, so each lane scans from its spawn.
    main = lanes[0]
    join = [-1] * len(lanes)
    joined: dict[int, float] = {}  # main turn -> latest end of the lanes it joins
    for lane in range(1, len(lanes)):
        end = max(row.end for row in lanes[lane])
        for index in range(spawn[lane] + 1, len(main)):
            if main[index].t + _JOIN_EPSILON_S >= end:
                join[lane] = index
                joined[index] = max(end, joined.get(index, end))
                break

    # Waits: the recorded gap after a turn's latest dependency ended.
    for index in range(1, len(main)):
        latest = main[index - 1].end
        latest = max(latest, joined.get(index, latest))
        main[index].wait = max(0.0, main[index].t - latest)
    for lane in range(1, len(lanes)):
        before = main[spawn[lane]].end
        for row in lanes[lane]:
            row.wait = max(0.0, row.t - before)
            before = row.end

    origin = main[0].t
    flat: list[int] = []
    for lane, rows in enumerate(lanes):
        for row in rows:
            tail = row.tail
            prompt = len(row.ids)
            total = prompt + tail[2] - tail[1]
            segments = (depths[row.leaf] if row.leaf >= 0 else 0) + 1
            wait = round(row.wait * 1000)
            flat += (
                lane, row.leaf, row.used, tail[0], tail[1], tail[2], row.tokens,
                row.out, wait, max(0, round((row.t - origin) * 1000)),
            )  # fmt: skip
            stats.prompt_blocks.append(prompt)
            stats.new_blocks.append(row.new)
            stats.segments.append(segments)
            stats.waits_ms.append(wait)
            stats.max_prompt_blocks = max(stats.max_prompt_blocks, prompt)
            stats.max_request_blocks = max(stats.max_request_blocks, total)
            stats.max_segments = max(stats.max_segments, segments)
    stats.main_turns += len(main)
    stats.child_turns += sum(len(rows) for rows in lanes[1:])
    stats.groups += len(lanes) - 1
    stats.streams.append(len(parents))
    stats.fanout.extend(Counter(spawn[1:]).values())
    edges = [value for pair in zip(parents, lengths) for value in pair]
    return Compiled(tid, edges, spawn, join, flat)


def compile_corpus(
    source: str | Path, out: str | Path, limit: int | None = None
) -> Stats:
    """Compile a Weka-format corpus (one trace per line) into a replay file."""
    stats = Stats()
    digest = hashlib.sha256()
    lines: list[bytes] = []
    block = None
    with open(source, "rb") as file:
        for line in file:
            digest.update(line)
            if not line.strip() or (limit is not None and stats.traces >= limit):
                continue
            trace = msgspec.json.decode(line)
            if block is None:
                block = trace.get("block_size")
            elif trace.get("block_size") != block:
                raise TraceError(
                    f"trace {trace.get('id', '?')}: block_size"
                    f" {trace.get('block_size')!r} differs from {block!r}"
                )
            compiled = compile_trace(trace, stats)
            if compiled is not None:
                lines.append(msgspec.json.encode(compiled))
                stats.traces += 1
    if not lines:
        raise TraceError(f"{source}: no trace with a main-lane request")
    header = Header(
        clusterbench_trace=FORMAT,
        source=Path(source).name,
        sha256=digest.hexdigest(),
        block_tokens=block,
        traces=stats.traces,
        turns=stats.turns,
        max_prompt_blocks=stats.max_prompt_blocks,
        max_request_blocks=stats.max_request_blocks,
        max_segments=stats.max_segments,
    )
    with open(out, "wb") as file:
        file.write(msgspec.json.encode(header) + b"\n")
        for line in lines:
            file.write(line + b"\n")
    return stats


# ---- replay -------------------------------------------------------------------


class _Trace:
    __slots__ = (
        "id", "parents", "lengths", "spawn", "join", "lanes", "rows", "children",
        "paths", "streams", "main", "at_ms",
    )  # fmt: skip

    def __init__(self, compiled: Compiled) -> None:
        self.id = compiled.id
        self.parents = compiled.edges[0::2]
        self.lengths = compiled.edges[1::2]
        self.spawn = compiled.spawn
        self.join = compiled.join
        rows = self.rows = compiled.rows
        # (first row, turns) per lane.
        counts = [0] * len(compiled.spawn)
        for offset in range(0, len(rows), _ROW):
            counts[rows[offset]] += 1
        self.lanes: list[tuple[int, int]] = []
        first = 0
        for turns in counts:
            self.lanes.append((first, turns))
            first += turns
        self.main = counts[0]
        self.children: dict[int, list[int]] = {}
        for lane in range(1, len(self.spawn)):
            self.children.setdefault(self.spawn[lane], []).append(lane)
        self.paths: dict[int, tuple[tuple[int, int], ...]] = {}
        # One spare stream for prompts under one block.
        self.streams = len(self.parents) + 1
        # Recorded time of each main-lane turn, made monotonic for bisection.
        self.at_ms: list[int] = []
        latest = 0
        for turn in range(self.main):
            latest = max(latest, rows[turn * _ROW + 9])
            self.at_ms.append(latest)

    def path(self, leaf: int) -> tuple[tuple[int, int], ...]:
        """The streams above ``leaf`` with their lengths, root first."""
        found = self.paths.get(leaf)
        if found is None:
            chain = []
            edge = self.parents[leaf]
            while edge >= 0:
                chain.append((edge, self.lengths[edge]))
                edge = self.parents[edge]
            chain.reverse()
            found = self.paths[leaf] = tuple(chain)
        return found


def load(path: str | Path) -> tuple[Header, list[_Trace]]:
    try:
        with open(path, "rb") as file:
            header = msgspec.json.decode(file.readline(), type=Header)
            if header.clusterbench_trace != FORMAT:
                raise TraceError(
                    f"{path}: format {header.clusterbench_trace}, this build reads"
                    f" {FORMAT}; compile the corpus again"
                )
            decoder = msgspec.json.Decoder(Compiled)
            traces = [_Trace(decoder.decode(line)) for line in file if line.strip()]
    except OSError as error:
        raise TraceError(
            f"cannot read workload.trace {str(path)!r} ({error.strerror}); make it"
            " with: python -m clusterbench trace SOURCE -o FILE"
        ) from None
    except msgspec.DecodeError as error:
        raise TraceError(f"{path}: not a compiled trace ({error})") from None
    if not traces:
        raise TraceError(f"{path}: no traces")
    return header, traces


class _Play:
    """One replay of one trace: a main lane and its sub-agent lanes."""

    __slots__ = ("trace", "base", "id", "cursor", "blocked", "live")

    def __init__(self, trace: _Trace, base: int, first_session: int) -> None:
        self.trace = trace
        self.base = base
        self.id = first_session  # lane n is session ``id + n``
        self.cursor = [0] * len(trace.lanes)
        # Main turn -> what it still waits for: the turn before it and the
        # sub-agents it joins. A turn that joins none has no entry.
        self.blocked: dict[int, int] = {}
        self.live = 0  # lanes that have not ended


class TraceWorkload:
    """Replays a compiled trace behind the ``Workload`` interface."""

    def __init__(self, config: RunConfig) -> None:
        workload, tiers = config.workload, config.tiers
        header, self._traces = load(workload.trace)
        self._header = header
        if header.block_tokens != tiers.tokens_per_block:
            raise TraceError(
                f"{workload.trace}: {header.block_tokens}-token blocks, but"
                f" tiers.tokens_per_block is {tiers.tokens_per_block}"
            )
        reserved = 1 + config.phases.warm_concurrency
        sglang = config.engine.profile == "sglang"
        device = tiers.gpu_blocks - (reserved if sglang else 0)
        if header.max_request_blocks > device:
            raise TraceError(
                f"{workload.trace}: the longest request is"
                f" {header.max_request_blocks} blocks, more than the device tier"
                f" can hold ({device}); raise tiers.gpu_blocks"
            )
        if not sglang and header.max_prompt_blocks > tiers.primary_blocks - reserved:
            raise TraceError(
                f"{workload.trace}: the longest prompt is"
                f" {header.max_prompt_blocks} blocks, more than the CPU tier can"
                f" promote at once ({tiers.primary_blocks - reserved}); raise"
                " tiers.primary_blocks"
            )
        if header.max_segments > MAX_SEGMENTS:
            raise TraceError(
                f"{workload.trace}: a request crosses {header.max_segments}"
                f" streams, more than the {MAX_SEGMENTS} the device cache orders"
            )
        self._scale = workload.trace_think_scale
        self._cap = workload.think_cap_s
        self._output_scale = workload.trace_output_scale
        self._block = tiers.tokens_per_block
        self._seed = workload.seed
        self._order = list(range(len(self._traces)))
        random.Random((workload.seed << 16) ^ 0x7ACE).shuffle(self._order)
        self._plays = 0
        self._finished = 0
        self._turns = 0
        self._child_turns = 0
        self._next_session = 0
        self._next_stream = TRACE_BASE
        self._lanes: dict[int, tuple[_Play, int]] = {}

    @property
    def started(self) -> int:
        return self._plays

    def summary(self) -> dict:
        header = self._header
        return {
            "kind": "trace",
            "trace": header.source,
            "sha256": header.sha256,
            "traces": header.traces,
            "plays_started": self._plays,
            "plays_finished": self._finished,
            "turns": self._turns,
            "child_turns": self._child_turns,
            "max_segments": header.max_segments,
        }

    def new_session(self, aged: bool = False) -> _Play:
        index = self._plays
        self._plays += 1
        trace = self._traces[self._order[index % len(self._order)]]
        if self._next_stream + trace.streams > MAX_STREAM:
            raise TraceError("stream ids exhausted")
        play = _Play(trace, self._next_stream, self._next_session)
        self._next_stream += trace.streams
        self._next_session += len(trace.lanes)
        start = 0
        if aged and trace.main > 1:
            # A play that was already running: a point of its recorded span.
            rng = random.Random((self._seed << 32) ^ (index * 2654435761) ^ 0xA6ED)
            target = rng.random() * trace.at_ms[-1]
            start = min(bisect_left(trace.at_ms, target), trace.main - 1)
        play.cursor[0] = start
        # Sub-agents that started before that point are over: no turn waits
        # for them.
        play.live = 1
        blocked = play.blocked
        for lane in range(1, len(trace.spawn)):
            if trace.spawn[lane] >= start:
                play.live += 1
                at = trace.join[lane]
                if at >= 0:
                    blocked[at] = blocked.get(at, 1) + 1
        self._lanes[play.id] = (play, 0)
        return play

    def next_turn(self, session: int) -> Turn | None:
        play, lane = self._lanes[session]
        trace = play.trace
        first, turns = trace.lanes[lane]
        at = play.cursor[lane]
        if at >= turns:
            return None
        play.cursor[lane] = at + 1
        rows = trace.rows
        offset = (first + at) * _ROW
        leaf, base = rows[offset + 1], play.base
        prompt_tokens, output_tokens = rows[offset + 6], rows[offset + 7]
        segments: list[Segment] = []
        if leaf >= 0:
            segments = [(base + edge, 0, length) for edge, length in trace.path(leaf)]
            segments.append((base + leaf, 0, rows[offset + 2]))
            start, end = rows[offset + 4], rows[offset + 5]
            if self._output_scale != 1.0:
                # A shorter answer completes fewer blocks.
                output_tokens = max(1, round(output_tokens * self._output_scale))
                block = self._block
                whole = (
                    prompt_tokens + output_tokens
                ) // block - prompt_tokens // block
                end = min(end, start + whole)
            tail = (base + rows[offset + 3], start, end)
        else:
            tail = (base + trace.streams - 1, 0, 0)
        self._turns += 1
        if lane:
            self._child_turns += 1
        return Turn(
            session=session,
            turn=at,
            segments=tuple(segments),
            tail=tail,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            think_s=0.0,
            last=False,
        )

    def _wait(self, trace: _Trace, lane: int, at: int) -> float:
        wait_ms = trace.rows[(trace.lanes[lane][0] + at) * _ROW + 8]
        return min(self._cap, wait_ms * 1e-3 * self._scale)

    def _release(self, play: _Play, at: int, wake: list[tuple[float, int]]) -> None:
        """Main turn ``at`` waits for one thing less; the last one wakes it."""
        left = play.blocked.get(at, 1) - 1
        if left:
            play.blocked[at] = left
        else:
            play.blocked.pop(at, None)
            wake.append((self._wait(play.trace, 0, at), play.id))

    def completed(self, turn: Turn) -> Step:
        session = turn.session
        play, lane = self._lanes[session]
        trace = play.trace
        wake: list[tuple[float, int]] = []
        ended: list[int] = []
        if lane == 0:
            for child in trace.children.get(turn.turn, ()):
                self._lanes[play.id + child] = (play, child)
                wake.append((self._wait(trace, child, 0), play.id + child))
            if turn.turn + 1 >= trace.main:
                ended.append(session)
            else:
                self._release(play, turn.turn + 1, wake)
        elif play.cursor[lane] < trace.lanes[lane][1]:
            wake.append((self._wait(trace, lane, play.cursor[lane]), session))
        else:
            ended.append(session)
            # The main lane may be parked at the join turn or still running
            # the turn before it: the cursor cannot tell, the count can.
            if trace.join[lane] >= 0:
                self._release(play, trace.join[lane], wake)
        for gone in ended:
            del self._lanes[gone]
            play.live -= 1
        if not play.live:
            # The whole tree has drained: the lane starts its next play.
            self._finished += 1
            wake.append((0.0, self.new_session().id))
        return Step(wake, ended)
