# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""vLLM's own scheduler, with no model behind it, driving a real KVCR.

The worker hosts vLLM's ``Scheduler`` with the offloading connector and the
KVCR secondary tier, built the way vLLM's unit tests build them
(``_make_tiering_engine_scheduler`` in
``tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py``). vLLM
constructs the KVCR, owns the device and CPU tiers, and makes every KVCR
call; ``kvcr_probe.py`` watches those calls. The vLLM commit is pinned in
``requirements.txt``.

Only what needs a GPU is faked:

- The forward pass: a sleep that releases the GIL, then one sampled token for
  every request past its prefill.
- The worker side of the connector. No bytes move between device and CPU;
  each load and store job is reported finished in the step after the one
  that issued it, which is when vLLM's worker reports them (it defers stores
  a step and starts loads after the forward pass). A flushed job, or a load
  the scheduler asked for synchronously, is reported in its own step.

Block hashes are given to vLLM, not computed from tokens: the block id
(``ids.py``) for every block a segment names, and ids private to the request
for the rest.
"""

# ruff: noqa: E402  (the environment is prepared before vLLM is imported)
from __future__ import annotations

import functools
import glob
import mmap
import os
import time
from pathlib import Path

from .kvcr_host import prepare_environment

prepare_environment()  # vLLM picks its platform the moment it is imported

import numpy as np
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVEventsConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_events import MEDIUM_CPU, MEDIUM_GPU, BlockRemoved, BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroup,
    KVCacheLayerConfig,
)
from vllm.v1.kv_hints.protocol import KvHintAction, KvHintsEnvelope
from vllm.v1.kv_offload.tiering.kvcr import manager as kvcr_tier
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request as VllmRequest
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

import kvcr

from .config import RunConfig
from .ids import INDEX_BITS, PRIVATE_BASE, block_hash, block_id, segment_ids
from .kvcr_probe import KvcrProbe
from .metrics import Window
from .protocol import (
    TIER_DEVICE,
    TIER_G2,
    TIER_PRIMARY,
    Done,
    InventoryDelta,
    Request,
)
from .router import PrefixTracker

_MODEL = str(Path(__file__).resolve().parent / "model")
_KVCR = kvcr_tier.KVCR
_PRIVATE = PRIVATE_BASE << INDEX_BITS
# (medium, ownership) of a vLLM KV event -> the tier the router is told about.
_TIERS = {
    (MEDIUM_GPU, None): TIER_DEVICE,
    (MEDIUM_CPU, None): TIER_PRIMARY,
    (MEDIUM_CPU, "kvcr"): TIER_G2,
}


def _later_hashes(hashes: list[bytes], block: int, request) -> list[bytes]:
    """vLLM's ``block_hasher``: the hashes of the blocks filled since it last asked."""
    return hashes[len(request.block_hashes) : request.num_tokens // block]


class _Live:
    __slots__ = ("rid", "arrived", "first_token", "prompt_blocks", "device_hit",
                 "external_hit", "computed")  # fmt: skip


class _Events:
    """In place of vLLM's KV event publisher: keeps the batches for the router."""

    def __init__(self) -> None:
        self.batches: list = []

    def publish(self, batch) -> None:
        self.batches.append(batch)

    def shutdown(self) -> None:
        pass


class VllmEngine:
    def __init__(
        self,
        config: RunConfig,
        window: Window,
        worker: int,
        peers: list[tuple[str, str]],
        *,
        bind_host: str,
        control_port: int,
        engine_id: str,
        kvcr_cls=_KVCR,
    ) -> None:
        tiers, engine = config.tiers, config.engine
        block = self._block = tiers.tokens_per_block
        if tiers.block_bytes % mmap.PAGESIZE or tiers.block_bytes % (4 * block):
            raise ValueError(
                f"tiers.block_bytes ({tiers.block_bytes}) must be a multiple of "
                f"this host's page size ({mmap.PAGESIZE}) and of 4 x "
                "tiers.tokens_per_block: vLLM's CPU tier is page aligned"
            )
        self._window = window
        self._cfg = engine
        self._peers = peers
        self._device_blocks = tiers.gpu_blocks
        self._primary_blocks = tiers.primary_blocks
        node = peers[worker][0]
        self.probe = KvcrProbe(
            kvcr_cls,
            window,
            verify=engine.verify,
            cross_node=frozenset(endpoint for name, endpoint in peers if name != node),
            on_stats=lambda stats: window.merge_stats(*stats.metric_sections()),
        )
        torch.set_num_threads(1)
        max_tokens = tiers.gpu_blocks * block
        vllm_config = VllmConfig(
            model_config=ModelConfig(
                model=_MODEL,
                dtype="float16",
                seed=0,
                max_model_len=max_tokens,
                skip_tokenizer_init=True,
            ),
            device_config=DeviceConfig("cpu"),
            scheduler_config=SchedulerConfig(
                max_num_seqs=engine.max_running,
                max_num_batched_tokens=engine.max_batched_tokens,
                max_model_len=max_tokens,
                enable_chunked_prefill=True,
                is_encoder_decoder=False,
            ),
            cache_config=CacheConfig(
                block_size=block, gpu_memory_utilization=0.9, enable_prefix_caching=True
            ),
        )
        vllm_config.scheduler_config.async_scheduling = False
        vllm_config.cache_config.kv_cache_layout = "LBNHC"
        # Block 0 of vLLM's pool is its null block.
        vllm_config.cache_config.num_gpu_blocks = tiers.gpu_blocks + 1
        vllm_config.kv_events_config = KVEventsConfig(
            enable_kv_cache_events=True, publisher="null"
        )
        vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            engine_id=engine_id,
            kv_connector_extra_config={
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": tiers.primary_blocks * tiers.block_bytes,
                "self_describing_kv_events": True,
                "secondary_tiers": [
                    {
                        "type": "kvcr",
                        "router_capabilities": sorted(kvcr.ROUTER_HINT_CAPABILITIES),
                        "control_host": bind_host,
                        "control_ports": [control_port],
                        "control_advertise_host": bind_host,
                        "secondary_g2_slots": tiers.g2_blocks,
                        "operation_timeout_ms": engine.operation_timeout_ms,
                        "abandon_timeout_ms": engine.abandon_timeout_ms,
                        # vLLM's default is off; the report reads it.
                        "enable_telemetry": True,
                    }
                ],
            },
        )
        # One layer whose KV for a block is tiers.block_bytes: that is the size
        # of a CPU-tier chunk, and so of a KVCR block.
        spec = AttentionSpec(
            block_size=block,
            num_kv_heads=1,
            head_size=tiers.block_bytes // (4 * block),
            dtype=torch.float16,
        )
        layer = KVCacheLayerConfig(FullAttentionManager.config(), spec)
        kv_cache_config = KVCacheConfig(
            num_blocks=tiers.gpu_blocks + 1,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroup.from_layers({"layer": layer})],
        )
        kvcr_tier.KVCR = self.probe  # what vLLM's tier builds its KVCR with
        try:
            self._scheduler = Scheduler(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                structured_output_manager=StructuredOutputManager(vllm_config),
                block_size=block,
                log_stats=True,
            )
        finally:
            kvcr_tier.KVCR = _KVCR
        # The CPU tier is a file in /dev/shm that only this process maps:
        # unlinked now, a killed worker leaves nothing behind.
        for path in glob.glob(f"/dev/shm/vllm_offload_{engine_id}.mmap*"):
            os.unlink(path)
        if self.probe.region != (tiers.block_bytes, tiers.primary_blocks):
            raise RuntimeError(
                f"vLLM's CPU tier is {self.probe.region[1]} chunks of "
                f"{self.probe.region[0]} bytes, not {tiers.primary_blocks} of "
                f"{tiers.block_bytes}"
            )
        self._events = self._scheduler.kv_event_publisher = _Events()
        self._trackers = [PrefixTracker(tier) for tier in sorted(_TIERS.values())]
        self._tokens = np.zeros(max_tokens, np.int32)  # token values are never read
        self._live: dict[str, _Live] = {}
        # Jobs to report next step, and the requests whose loads are among them.
        self._due: dict[int, int] = {}
        self._received: set[str] = set()
        self._primary = 0
        self._evicted = False

    # ---- requests ----------------------------------------------------------

    def submit(self, turn: Request) -> None:
        block = self._block
        prompt, output = turn.prompt_tokens, max(1, turn.output_tokens)
        need = -(-(prompt + output) // block)
        if need > self._device_blocks:
            raise ValueError(
                f"request {turn.rid} needs {need} device blocks, more than "
                f"tiers.gpu_blocks ({self._device_blocks})"
            )
        request_id = f"r{turn.rid}"
        named = segment_ids(turn.segments)
        hint = None
        if turn.hint_source >= 0 and turn.hint_blocks > 0:
            # vLLM hands a hint to KVCR only when the router sent one.
            payload = {
                "source_control_endpoint": self._peers[turn.hint_source][1],
                "block_hashes": named[: turn.hint_blocks],
            }
            action = KvHintAction(
                action_id="a1",
                action_type="kv.fetch",
                action_version="1.0",
                payload=payload,
            )
            hint = KvHintsEnvelope(
                protocol_version="0.1", message_id=request_id, actions=[action]
            )
        # One id per full block of the prompt, then per block the answer fills.
        private = block_id(PRIVATE_BASE + turn.rid, 0)
        full = prompt // block
        ids = named[:full]
        ids.extend(range(private + len(ids), private + full))
        answer = (prompt + output) // block - full
        tail = segment_ids((turn.tail,))[:answer]
        tail.extend(range(private + full + len(tail), private + full + answer))
        hashes = [block_hash(i) for i in ids]
        later = hashes + [block_hash(i) for i in tail]
        live = self._live[request_id] = _Live()
        live.rid = turn.rid
        live.first_token = 0.0
        live.prompt_blocks = full
        live.device_hit = live.external_hit = 0
        live.computed = prompt
        live.arrived = t = time.perf_counter()
        request = VllmRequest(
            request_id=request_id,
            prompt_token_ids=self._tokens[:prompt],
            sampling_params=SamplingParams(max_tokens=output),
            pooling_params=None,
            block_hasher=functools.partial(_later_hashes, later, block),
            kv_hints=hint,
            block_hashes=hashes,
        )
        self._scheduler.add_request(request)
        self._window.counters["vllm_seconds"] += time.perf_counter() - t

    # ---- one engine step ---------------------------------------------------

    def step(self) -> list[Done]:
        perf = time.perf_counter
        window, cfg, scheduler = self._window, self._cfg, self._scheduler
        counters = window.counters
        kvcr_before = sum(window.secs.values())
        t_schedule = perf()
        out = scheduler.schedule()
        t_scheduled = perf()

        # The connector's worker side: what finished since the last step.
        finished, received = self._due, self._received
        self._due, self._received = {}, set()
        meta = out.kv_connector_metadata
        if meta is not None:
            flush = meta.jobs_to_flush or ()
            for job_id, job in meta.load_jobs.items():
                if job.sync:
                    finished[job_id] = 1
                else:
                    self._due[job_id] = 1
                    self._received.add(job.req_id)
            for job_id in meta.store_jobs:
                if job_id in flush:
                    finished[job_id] = 1
                else:
                    self._due[job_id] = 1
        connector_output = None
        if finished or received:
            connector_output = KVConnectorOutput(
                finished_recving=received or None,
                kv_connector_worker_meta=OffloadingWorkerMetadata(
                    completed_jobs=finished
                ),
            )
        request_ids = list(out.num_scheduled_tokens)
        requests = scheduler.requests
        runner_output = ModelRunnerOutput(
            req_ids=request_ids,
            sampled_token_ids=[
                [] if requests[request_id].is_prefill_chunk else [0]
                for request_id in request_ids
            ],
            kv_connector_output=connector_output,
        )

        # Forward pass: the GIL is released, as it is while the GPU runs.
        scheduled = out.total_num_scheduled_tokens
        t_forward = perf()
        if scheduled:
            prefill = scheduled - len(request_ids)
            time.sleep(cfg.step_ms / 1e3 + prefill * cfg.prefill_us_per_token / 1e6)
        else:
            time.sleep(cfg.idle_sleep_ms / 1e3)  # vLLM's engine sleeps too
        t_output = perf()
        counters["forward_seconds" if scheduled else "idle_seconds"] += (
            t_output - t_forward
        )

        outputs = scheduler.update_from_output(out, runner_output)
        counters["vllm_seconds"] += t_scheduled - t_schedule + perf() - t_output
        dones: list[Done] = []
        for client in outputs.values():
            stats = client.scheduler_stats
            if stats is not None and stats.num_preempted_reqs:
                counters["preemptions"] += stats.num_preempted_reqs
            for output in client.outputs:
                live = self._live.get(output.request_id)
                if live is None:
                    continue
                if not live.first_token and output.new_token_ids:
                    live.first_token = t_output
                prefill_stats = output.prefill_stats
                if prefill_stats is not None:
                    live.device_hit = prefill_stats.num_local_cached_tokens
                    live.external_hit = prefill_stats.num_external_cached_tokens
                    live.computed = prefill_stats.num_computed_tokens
                if output.finish_reason is not None:
                    dones.append(self._finish(output.request_id, live))
        counters["steps"] += 1
        if scheduled:
            counters["busy_steps"] += 1
            window.hists["step_kvcr"].add(sum(window.secs.values()) - kvcr_before)
        return dones

    def _finish(self, request_id: str, live: _Live) -> Done:
        del self._live[request_id]
        window, block = self._window, self._block
        remote, local, failed, timed_out, waited = self.probe.outcome(request_id)
        ttft = live.first_token - live.arrived
        window.hists["ttft"].add(ttft)
        window.hists["cache_resolve"].add(waited)
        if remote or failed:
            window.hists["cache_resolve_remote"].add(waited)
        counters = window.counters
        counters["requests"] += 1
        counters["prompt_blocks"] += live.prompt_blocks
        counters["gpu_hit_blocks"] += live.device_hit // block
        # Loaded from the CPU tier without KVCR having to put them there.
        counters["primary_hit_blocks"] += max(
            0, live.external_hit // block - remote - local
        )
        counters["prefill_tokens"] += live.computed
        return Done(
            rid=live.rid,
            ttft_ms=ttft * 1e3,
            load_ms=waited * 1e3,
            remote_blocks=remote,
            failed_blocks=failed,
            timed_out_blocks=timed_out,
        )

    # ---- reporting ---------------------------------------------------------

    def take_inventory(self) -> list[InventoryDelta]:
        """What entered and left each tier, from vLLM's own KV events."""
        batches, self._events.batches = self._events.batches, []
        trackers = self._trackers
        for batch in batches:
            for event in batch.events:
                kind = type(event)
                if kind is not BlockStored and kind is not BlockRemoved:
                    continue
                tier = _TIERS.get((event.medium, event.ownership))
                if tier is None:
                    continue
                removed = kind is BlockRemoved
                ids = event.block_hashes
                if tier == TIER_PRIMARY:
                    if removed:  # the CPU tier only removes what it evicts
                        self._primary -= len(ids)
                        self._evicted = True
                    else:
                        self._primary += len(ids)
                named = [bid for bid in ids if bid < _PRIVATE]
                if named:
                    trackers[tier].apply(removed, named)
        deltas: list[InventoryDelta] = []
        for tracker in trackers:
            deltas.extend(tracker.take())
        return deltas

    def abort_requests(self) -> None:
        """Drop every request; KVCR operations already issued still complete."""
        self._scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
        self._live.clear()

    def busy(self) -> bool:
        return bool(
            self._live
            or self._due
            or self.probe.pending()
            or self._scheduler.has_finished_requests()
        )

    def gauges(self) -> dict[str, float]:
        running, waiting = self._scheduler.get_request_counts()
        return {
            "waiting": waiting,
            "running": running,
            "jobs": self.probe.pending(),
            "primary_occupancy": self._primary / self._primary_blocks,
            "primary_fill": self.fill_level(),
            "gpu_usage": self._scheduler.get_kv_cache_usage(),
        }

    def fill_level(self) -> float:
        """How full the CPU tier has got: 1.0 once it has had to evict."""
        return 1.0 if self._evicted else self._primary / self._primary_blocks

    def close(self) -> None:
        self._scheduler.shutdown()  # closes KVCR and unmaps the tiers
