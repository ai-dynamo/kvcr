# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Coordinator <-> worker messages (msgpack over a ZMQ ROUTER/DEALER pair)."""

from __future__ import annotations

import msgspec

from .ids import Segment

# Inventory tiers reported to the router.
TIER_DEVICE = 0
TIER_PRIMARY = 1
TIER_G2 = 2

# (tier, stream, start, end, removed)
InventoryDelta = tuple[int, int, int, int, bool]


class _Msg(msgspec.Struct, tag=True, array_like=True, frozen=True):
    pass


# Worker -> coordinator.


class Hello(_Msg):
    node: str
    local_id: int
    host: str
    pid: int
    control_endpoint: str
    agent_name: str
    arm: str
    kvcr_file: str
    kvcr_sha: str
    cpus: list[int]
    limits: dict[str, int]


class Ready(_Msg):
    phase: str
    info: dict[str, float] = {}


class Done(_Msg):
    """One finished request, measured on the worker's clock."""

    rid: int
    ttft_ms: float
    total_ms: float
    load_ms: float
    prompt_blocks: int
    gpu_blocks: int
    primary_blocks: int
    local_blocks: int  # promoted from this worker's own KVCR tiers
    remote_blocks: int  # promoted from a peer
    failed_blocks: int  # promotions that failed
    recomputed_tokens: int
    # The engine dropped the request (SGLang: a cache load in its batch failed).
    aborted: bool = False
    # Of the failed blocks, those whose transfer ran into the operation timeout.
    timed_out_blocks: int = 0


class Inventory(_Msg):
    deltas: list[InventoryDelta]


class Stats(_Msg):
    t: float
    values: dict[str, float]


class Report(_Msg):
    payload: bytes  # JSON


class Failure(_Msg):
    text: str


# Coordinator -> worker.


class Configure(_Msg):
    worker: int
    config: bytes  # JSON RunConfig
    # worker id -> (node, control endpoint), indexed by worker id.
    peers: list[tuple[str, str]]


class Phase(_Msg):
    name: str  # "warm", "fill", "measure", "stop"


class Request(_Msg):
    rid: int
    session: int
    segments: list[Segment]
    tail: Segment
    prompt_tokens: int
    output_tokens: int
    hint_source: int = -1
    hint_blocks: int = 0


Message = (
    Hello
    | Ready
    | Done
    | Inventory
    | Stats
    | Report
    | Failure
    | Configure
    | Phase
    | Request
)

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(Message)


def encode(message: _Msg) -> bytes:
    return _encoder.encode(message)


def decode(data: bytes) -> Message:
    return _decoder.decode(data)
