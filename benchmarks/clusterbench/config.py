# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run configuration. One ``RunConfig`` fully describes a cell."""

from __future__ import annotations

from pathlib import Path

import msgspec


class TierConfig(msgspec.Struct, frozen=True, kw_only=True):
    # Bytes KVCR moves per block. Host-path cost scales with block count, so
    # the default is far below a real KV block (several MiB) to fit many
    # workers per node; raise it for a data-plane calibration cell.
    block_bytes: int = 16384
    tokens_per_block: int = 64
    # Device (GPU) cache, bookkeeping only.
    gpu_blocks: int = 32768
    # vLLM primary CPU offload tier: real memory, KVCR's framework region.
    primary_blocks: int = 8192
    # KVCR-owned DRAM pool (vLLM ``secondary_g2_slots``); 0 is peer-only.
    g2_blocks: int = 65536
    framework_pinning: bool = True
    # SGLang profile only: transformer layers. A page is one KVCR key with a K
    # and a V piece per layer (``block_bytes / (2 * layers)`` bytes each), and
    # ``gpu_blocks`` is real memory there, since KVCR reads and writes it.
    layers: int = 64

    @property
    def pieces(self) -> int:
        return 2 * self.layers

    @property
    def piece_bytes(self) -> int:
        return self.block_bytes // self.pieces


class EngineConfig(msgspec.Struct, frozen=True, kw_only=True):
    # "vllm": vLLM's KVCR offload tier. "sglang": SGLang's KVCR direct linker,
    # which fixes several of the options below itself (no source pins, no
    # opportunistic query, prompt and output both backed up).
    profile: str = "vllm"
    # Modelled forward pass, GIL released: step_ms + prefill_us_per_token * tokens.
    step_ms: float = 20.0
    prefill_us_per_token: float = 20.0
    idle_sleep_ms: float = 1.0
    max_batched_tokens: int = 8192
    max_running: int = 64
    offload_prompt_only: bool = True
    # vLLM adapter defaults.
    operation_timeout_ms: int = 1000
    abandon_timeout_ms: int = 5000
    eager_ctrl_connect: bool = True
    opportunistic_query: bool = True
    metadata_retry_interval_ms: int = 100
    # Check delivered bytes: "none", "sample" (first block per load) or "full".
    verify: str = "sample"
    # SGLang profile: its scheduler loop spins when it has nothing to run
    # (``--sleep-on-idle`` off, the default); True sleeps ``idle_sleep_ms``.
    sleep_on_idle: bool = False
    # What-if switches. The defaults are what the engines do today; anything
    # else measures a change to the engine's integration, not to KVCR.
    # vLLM: align a request's sequence only when it finishes, not after every
    # completed deliver or deposit as well.
    align_on_finish_only: bool = False
    # SGLang linker: layers named by one deliver (upstream 1), deliver
    # operations outstanding at once (upstream 4), and whether pages just
    # restored from KVCR are backed up again (upstream: yes).
    linker_layers_per_deliver: int = 1
    linker_max_deliveries: int = 4
    linker_backup_restored: bool = True
    # SGLang linker: pieces per deposit (upstream 4096, which is 32 pages of a
    # 64-layer model; upstream's 512 MiB bound gives the same 32 real pages).
    linker_deposit_pieces: int = 4096


class WorkloadConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Closed-loop agent sessions. Sizes are lognormal(median, sigma) in tokens.

    Defaults follow the Claude Code traces the team replays (64-token blocks):
    new uncached tokens per turn p50 ~600 / mean ~2.4k, output p50 ~212 / mean
    ~620, think time p50 ~1 s / p90 ~12 s, ~110 turns per session.
    """

    seed: int = 0
    sessions_per_worker: float = 4.0
    agent_types: int = 4
    system_prefix_tokens: int = 32000
    initial_context_median: int = 5000
    initial_context_sigma: float = 1.0
    turn_tokens_median: int = 600
    turn_tokens_sigma: float = 1.6
    output_tokens_median: int = 212
    output_tokens_sigma: float = 1.45
    think_median_s: float = 1.0
    think_sigma: float = 1.95
    think_cap_s: float = 30.0
    turns_median: int = 110
    turns_sigma: float = 1.0
    max_context_tokens: int = 200000
    # Context kept after a compaction/reset at the cap (besides the system prefix).
    reset_context_median: int = 8000
    # Spread session starts over this long so workers do not all prefill at once.
    ramp_s: float = 5.0
    # Start the initial sessions mid-life (contexts already grown), so the
    # tiers reach their steady state within the fill phase. Sessions that
    # replace ended ones always start fresh.
    aged_start: bool = True
    # Replay a compiled trace (``python -m clusterbench trace``) instead of
    # generating sessions. Of the fields above only ``seed``,
    # ``sessions_per_worker`` (live plays), ``think_cap_s``, ``ramp_s`` and
    # ``aged_start`` (start part-way through the trace) then apply; recorded
    # waits are multiplied by ``trace_think_scale`` before the cap, and answer
    # lengths by ``trace_output_scale`` (below 1 a rank spends less of its
    # time decoding; the blocks a shortened answer no longer covers are
    # computed as part of the next prompt instead).
    trace: str = ""
    trace_think_scale: float = 1.0
    trace_output_scale: float = 1.0


class RouterConfig(msgspec.Struct, frozen=True, kw_only=True):
    # "sticky": a session stays on its home worker and moves with
    # migration_prob per turn, independent of load (same offered work in both
    # A/B arms). "cost": Dynamo's default KV-router formula.
    mode: str = "sticky"
    migration_prob: float = 0.05
    # A migrating session goes to the emptiest worker; when several are equally
    # empty, the probability that it picks one on its own node.
    same_node_prob: float = 0.0
    hints: bool = True
    device_credit: float = 1.0
    host_credit: float = 0.75
    seed: int = 0


class PhaseConfig(msgspec.Struct, frozen=True, kw_only=True):
    # "full": every ordered pair exchanges one block before the workload
    # starts, so the window does not measure first-contact peer setup.
    mesh_warm: str = "full"
    warm_concurrency: int = 8
    warm_timeout_s: float = 300.0
    fill_s: float = 20.0
    measure_s: float = 60.0
    drain_s: float = 10.0
    stats_interval_s: float = 1.0
    inventory_flush_ms: float = 50.0
    register_timeout_s: float = 300.0
    # Flag the run if any worker's CPU tier is emptier than this when the
    # window opens: the fill phase was too short to reach steady state.
    min_primary_occupancy: float = 0.0


class RunConfig(msgspec.Struct, frozen=True, kw_only=True):
    name: str = "run"
    tiers: TierConfig = TierConfig()
    engine: EngineConfig = EngineConfig()
    workload: WorkloadConfig = WorkloadConfig()
    router: RouterConfig = RouterConfig()
    phases: PhaseConfig = PhaseConfig()


def _coerce(text: str) -> object:
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def load_config(path: str | Path | None, overrides: list[str] = ()) -> RunConfig:
    """Load a TOML scenario and apply ``section.field=value`` overrides."""
    data: dict = {}
    if path is not None:
        data = msgspec.toml.decode(Path(path).read_bytes())
    for item in overrides:
        dotted, _, value = item.partition("=")
        if not _:
            raise ValueError(f"override must be section.field=value: {item!r}")
        *sections, field = dotted.split(".")
        node = data
        for section in sections:
            node = node.setdefault(section, {})
        node[field] = _coerce(value)
    config = msgspec.convert(data, RunConfig)
    validate(config)
    return config


def validate(config: RunConfig) -> None:
    tiers, engine = config.tiers, config.engine
    if tiers.block_bytes < 4096 or tiers.block_bytes % 4096:
        raise ValueError("tiers.block_bytes must be a multiple of 4096")
    if tiers.primary_blocks < 64:
        raise ValueError("tiers.primary_blocks is too small")
    if engine.profile not in ("vllm", "sglang"):
        raise ValueError(f"unknown engine profile {engine.profile!r}")
    if engine.profile == "sglang":
        if tiers.layers < 1 or tiers.block_bytes % (8 * tiers.pieces):
            raise ValueError(
                "tiers.block_bytes must split into 2 x tiers.layers pieces of "
                "a multiple of 8 bytes"
            )
        if not tiers.g2_blocks:
            raise ValueError("the SGLang linker needs a KVCR pool (tiers.g2_blocks)")
    elif not tiers.g2_blocks and not tiers.framework_pinning:
        raise ValueError("a worker with no G2 pool and no pinning cannot be a source")
    if engine.abandon_timeout_ms < 2 * engine.operation_timeout_ms:
        raise ValueError("engine.abandon_timeout_ms must be >= 2x operation timeout")
    if (
        engine.linker_layers_per_deliver < 1
        or engine.linker_max_deliveries < 1
        or engine.linker_deposit_pieces < 1
    ):
        raise ValueError("engine.linker_* counts must be at least 1")
    if engine.verify not in ("none", "sample", "full"):
        raise ValueError("engine.verify must be none, sample or full")
    # The trace file itself is opened by the coordinator only: workers decode
    # the same config on nodes that need not have it.
    if config.workload.trace_think_scale < 0:
        raise ValueError("workload.trace_think_scale must not be negative")
    if config.workload.trace_output_scale <= 0:
        raise ValueError("workload.trace_output_scale must be positive")
    if config.router.mode not in ("sticky", "cost"):
        raise ValueError("router.mode must be sticky or cost")
    if config.phases.mesh_warm not in ("full", "none"):
        raise ValueError("phases.mesh_warm must be full or none")


def encode_config(config: RunConfig) -> bytes:
    return msgspec.json.encode(config)


def decode_config(data: bytes) -> RunConfig:
    return msgspec.json.decode(data, type=RunConfig)
