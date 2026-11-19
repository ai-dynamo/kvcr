# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The fake-engine interface a worker drives.

A profile reproduces one engine integration's KVCR call pattern: ``vllm`` is
vLLM's KVCR offload tier, ``sglang`` is SGLang's KVCR direct linker.
"""

from __future__ import annotations

from typing import Protocol

import msgspec

from .config import RunConfig
from .metrics import Window
from .protocol import Done, InventoryDelta, Request


class HintAction(msgspec.Struct, frozen=True, kw_only=True):
    action_id: str
    action_type: str
    action_version: str
    payload: dict = {}


class HintEnvelope(msgspec.Struct, frozen=True, kw_only=True):
    """The ``kv_hints`` envelope a router attaches to a request."""

    protocol_version: str
    message_id: str
    actions: list[HintAction] = []


def fetch_hint(message_id: str, endpoint: str, hashes: list[int]) -> HintEnvelope:
    """The envelope Dynamo's router attaches (one ``kv.fetch`` action)."""
    payload = {"source_control_endpoint": endpoint, "block_hashes": hashes}
    action = HintAction(
        action_id="a1", action_type="kv.fetch", action_version="1.0", payload=payload
    )
    return HintEnvelope(protocol_version="0.1", message_id=message_id, actions=[action])


def failure_cause(entry) -> str:
    """Why KVCR did not deliver a block: its result's status in lower case
    (``timeout``, ``unavailable``, ``aborted``, ``failed``), or
    ``unclassified`` on an older tree, whose results have no reason and call
    every failure ``FAILED``."""
    if not hasattr(entry, "reason"):
        return "unclassified"
    return entry.status.name.lower()


def count_failure(counters, entry) -> str:
    """Count one hinted block that a peer did not deliver; returns its cause."""
    cause = failure_cause(entry)
    counters[f"hinted_failed|{cause}"] += 1
    reason = getattr(entry, "reason", None)
    if reason:
        counters[f"hinted_failed_reason|{reason}"] += 1
    return cause


class Engine(Protocol):
    """Driven from the worker's main thread.

    In the vLLM profile that thread is also the only one calling KVCR; the
    SGLang profile hands every KVCR call to its own owner thread.
    """

    # KVCRBindings callbacks.
    def request_pin(self, keys): ...
    def poll_pin_results(self): ...
    def release_pin(self, handle) -> bool: ...
    def cancel_pin_request(self, request) -> None: ...
    def record_inventory(self, event) -> None: ...
    def on_resilience_event(self, error: Exception) -> None: ...

    def attach(self, host) -> None:
        """Bind the started KVCR and its memory."""

    def set_peers(self, peers: list[tuple[str, str]]) -> None:
        """(node, control endpoint) of every worker, indexed by worker id."""

    def submit(self, request: Request) -> None:
        """A new request arrived."""

    def step(self) -> list[Done]:
        """Run one engine step, including the modelled forward pass."""

    def take_inventory(self) -> list[InventoryDelta]:
        """Residency changes since the last call, in order."""

    def start_warm(self, peers: list[tuple[int, str]]) -> None:
        """Pull one block from each peer to set up every connection."""

    def warm_status(self) -> tuple[int, int, int]:
        """(pending, succeeded, failed) warm pulls."""

    def abort_requests(self) -> None:
        """Drop every request; KVCR operations already issued still complete."""

    def busy(self) -> bool:
        """Whether requests or KVCR operations are still in flight."""

    def resilience_samples(self) -> list[str]:
        """A few KVCR resilience-event messages, for the report."""

    def gauges(self) -> dict[str, float]: ...

    def fill_level(self) -> float:
        """How full the host tier that must reach steady state is, 0 to 1."""

    def begin_window(self) -> None:
        """The measurement window was just reset."""

    def end_window(self) -> None:
        """The window is closing: fold in anything held off the engine thread."""

    def close(self) -> None:
        """Stop the engine's own threads. Called before the host is closed."""


def make_engine(config: RunConfig, window: Window, worker: int) -> Engine:
    if config.engine.profile == "vllm":
        from .engine_vllm import VllmEngine

        return VllmEngine(config, window, worker)
    if config.engine.profile == "sglang":
        from .engine_sglang import SglangEngine

        return SglangEngine(config, window, worker)
    raise ValueError(f"unknown engine profile {config.engine.profile!r}")
